"""Database storage backend for execution history using dataknobs_data.

This module provides a unified storage backend that works with ANY dataknobs_data
database backend (SQLite, PostgreSQL, MongoDB, Elasticsearch, S3, etc.) through
the common AsyncDatabase interface.
"""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from dataknobs_common.lifecycle import close_if_owned
from dataknobs_data import AsyncKeyedRecordStore, SortSpec
from dataknobs_data.query import Query
from dataknobs_data.records import Record

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStatus, ExecutionStep
from dataknobs_fsm.storage.base import (
    BaseHistoryStorage,
    StorageBackend,
    StorageConfig,
    StorageFactory,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataknobs_data.database import AsyncDatabase

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


@dataclass
class _StepRecord:
    """Internal wrapper bundling a step with its persisted context.

    Used as the typed value ``T`` for
    :class:`AsyncKeyedRecordStore[_StepRecord]` so that the serializer
    signature ``(T) -> (data, metadata)`` makes the metadata channel part
    of the function's type — a future change to the persisted shape
    cannot silently drop the metadata channel without a type-visible
    diff at the serializer site.  The history sibling ``_HistoryRecord``
    follows the same pattern.
    """

    step: ExecutionStep
    execution_id: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _step_to_columns(
    record: _StepRecord,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a ``_StepRecord`` into ``(data, metadata)`` for storage.

    The two-channel return type is load-bearing: the metadata column is
    part of this function's signature, so it cannot be silently dropped
    by future edits without a type-visible diff.
    """
    step = record.step
    data: dict[str, Any] = {
        "id": record.record_id,
        "execution_id": record.execution_id,
        "step_id": step.step_id,
        "parent_id": record.parent_id,
        "state_name": step.state_name,
        "network_name": step.network_name,
        "status": step.status.value,
        "timestamp": step.timestamp,
        "record_type": "step",
        "step_data": step.to_dict(),
    }
    return data, dict(record.metadata)


def _step_from_record(record: Record) -> _StepRecord:
    """Reconstruct a ``_StepRecord`` from a stored ``Record``.

    Reads the persisted step state from the ``step_data`` nested dict in
    the data column and pulls metadata from the metadata column.
    """
    step_data = record.get_value("step_data") or {}

    step = ExecutionStep(
        step_id=step_data["step_id"],
        state_name=step_data["state_name"],
        network_name=step_data["network_name"],
        timestamp=step_data["timestamp"],
        data_mode=DataHandlingMode(step_data["data_mode"]),
        status=ExecutionStatus(step_data["status"]),
    )

    for attr in (
        "start_time",
        "end_time",
        "arc_taken",
        "metrics",
        "resource_usage",
        "stream_progress",
        "chunks_processed",
        "records_processed",
    ):
        if attr in step_data:
            setattr(step, attr, step_data[attr])

    if step_data.get("error"):
        step.error = Exception(step_data["error"])

    return _StepRecord(
        step=step,
        execution_id=record.get_value("execution_id") or "",
        parent_id=record.get_value("parent_id"),
        metadata=dict(record.metadata or {}),
        record_id=record.storage_id or record.get_value("id") or str(uuid.uuid4()),
    )


@dataclass
class _HistoryRecord:
    """Internal wrapper bundling an execution history with its persisted context.

    Sibling of :class:`_StepRecord` — same rationale: used as the typed
    value ``T`` for :class:`AsyncKeyedRecordStore[_HistoryRecord]` so the
    ``(T) -> (data, metadata)`` serializer signature keeps the metadata
    channel part of the function's type, preventing silent drops on
    future shape changes.

    The store keys on ``history.execution_id``, so re-saving the same
    history is an idempotent upsert (the prior ``Record({"id": uuid(),
    ...})`` shape appended a new row on every save).

    ``history_data`` carries the pre-serialized payload (optionally
    compressed; see :meth:`BaseHistoryStorage._serialize_history`).
    Keeping serialization at the caller, not the serializer, lets the
    serializer remain a pure module-level function while the
    config-dependent compression decision lives on the storage instance.

    ``created_at`` carries the original creation timestamp across
    upserts: ``save_history`` reads the prior row (if any) and threads
    its ``created_at`` through here so the serializer can preserve it.
    ``None`` means "first save — stamp now".
    """

    history: ExecutionHistory
    history_data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float | None = None


def _history_to_columns(
    record: _HistoryRecord,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a ``_HistoryRecord`` into ``(data, metadata)`` for storage.

    The two-channel return type is load-bearing: the metadata column is
    part of this function's signature, so it cannot be silently dropped
    by future edits without a type-visible diff.

    ``created_at`` is preserved across upserts when the caller supplies
    a prior value on the ``_HistoryRecord`` (the read-before-write in
    :meth:`UnifiedDatabaseStorage.save_history`); otherwise it is
    stamped to ``now`` on first save.  ``updated_at`` always advances.
    """
    history = record.history
    now = time.time()
    data: dict[str, Any] = {
        "id": history.execution_id,
        "execution_id": history.execution_id,
        "fsm_name": history.fsm_name,
        "data_mode": history.data_mode.value,
        "status": "completed" if history.end_time else "in_progress",
        "start_time": history.start_time,
        "end_time": history.end_time,
        "total_steps": history.total_steps,
        "failed_steps": history.failed_steps,
        "skipped_steps": history.skipped_steps,
        "record_type": "history",
        "history_data": record.history_data,
        "created_at": record.created_at if record.created_at is not None else now,
        "updated_at": now,
    }
    return data, dict(record.metadata)


def _build_store(
    db: AsyncDatabase,
    serializer: Callable[[_T], tuple[dict[str, Any], dict[str, Any]]],
    deserializer: Callable[[Record], _T],
) -> AsyncKeyedRecordStore[_T]:
    """Construct a typed keyed store over an ``AsyncDatabase``.

    Module-level helper used at four lazy-init sites in
    :class:`UnifiedDatabaseStorage` (``__init__`` x2 + ``_setup_backend``
    x2) so the constructor call isn't duplicated.  Lives at module level
    rather than as a ``@staticmethod`` on the class because the
    ``TypeVar`` ``_T`` is naturally module-scoped — the owning class
    holds two stores with different element types
    (``_StepRecord`` / ``_HistoryRecord``) so it cannot itself be
    ``Generic[_T]``.
    """
    return AsyncKeyedRecordStore[_T](
        db,
        serializer=serializer,
        deserializer=deserializer,
    )


def _require_store(
    store: AsyncKeyedRecordStore[_T] | None, name: str
) -> AsyncKeyedRecordStore[_T]:
    """Runtime guard that survives ``python -O`` (unlike ``assert``).

    Module-level helper used by every store-using method.  The
    ``TypeVar`` narrows the return type per call so callers keep typed
    access to ``_StepRecord`` / ``_HistoryRecord`` operations.  Stores
    are set either in ``__init__`` (when a database is injected) or in
    ``_setup_backend`` (factory path); the check guards against direct
    misuse.
    """
    if store is None:
        raise RuntimeError(
            f"{name} store not initialized; call initialize() first"
        )
    return store


def _history_from_record(record: Record) -> _HistoryRecord:
    """Reconstruct a ``_HistoryRecord`` from a stored ``Record``.

    Decompresses ``history_data`` when the embedded ``"compressed"`` flag
    is set (mirror of :meth:`BaseHistoryStorage._deserialize_history`).
    Pulls metadata from the metadata column.
    """
    history_data = record.get_value("history_data") or {}

    if history_data.get("compressed"):
        # Inline decompression so this stays a pure module-level
        # function (no ``self`` dependency).  Mirrors the decode in
        # :meth:`BaseHistoryStorage._deserialize_history`.
        import base64
        import json as _json
        import zlib

        compressed = base64.b64decode(history_data["data"])
        history_data = _json.loads(
            zlib.decompress(compressed).decode("utf-8")
        )

    history = ExecutionHistory.from_dict(history_data)
    return _HistoryRecord(
        history=history,
        history_data=history_data,
        metadata=dict(record.metadata or {}),
        created_at=record.get_value("created_at"),
    )


class UnifiedDatabaseStorage(BaseHistoryStorage):
    """Unified database storage that works with any dataknobs_data backend.

    This single implementation works with:
    - Memory (AsyncMemoryDatabase)
    - SQLite (AsyncSQLiteDatabase)
    - PostgreSQL (AsyncPostgresDatabase)
    - MongoDB (AsyncMongoDatabase)
    - Elasticsearch (AsyncElasticsearchDatabase)
    - S3 (AsyncS3Database)
    - File (AsyncFileDatabase)

    All through the same AsyncDatabase interface from dataknobs_data.
    """

    def __init__(
        self,
        config: StorageConfig,
        *,
        database: AsyncDatabase | None = None,
        steps_database: AsyncDatabase | None = None,
        owns_databases: bool | None = None,
    ):
        """Initialize database storage.

        Args:
            config: Storage configuration. Backend selection is driven
                by ``config.backend`` (the canonical ``StorageBackend``
                enum); ``config.connection_params`` carries
                backend-specific options (host/port/path/etc.) that are
                forwarded to the underlying ``AsyncDatabase``.
            database: Optional pre-built AsyncDatabase instance. When provided,
                ``_setup_backend()`` skips factory creation and uses this instance
                directly. Enables connection pool sharing across components.
            steps_database: Optional separate AsyncDatabase for step records.
                Defaults to ``database`` when only ``database`` is provided.
            owns_databases: Explicit ownership override. When ``True``, this
                instance will close both databases on ``close()``. When
                ``False``, neither is closed. When ``None`` (default),
                ownership is inferred: databases created by the factory are
                owned; injected databases are not.
        """
        super().__init__(config)
        if owns_databases is not None:
            self._owns_db = owns_databases
            self._owns_steps_db = owns_databases
        else:
            self._owns_db = database is None
            self._owns_steps_db = steps_database is None and database is None
        self._db: AsyncDatabase | None = database
        self._steps_db: AsyncDatabase | None = steps_database or database
        # Typed keyed stores centralize Record construction at one site
        # per record type (closes the historical inline ``Record({...})``
        # defect class).  Built lazily once the backing database is
        # established — see :meth:`_setup_backend`.
        self._step_store: AsyncKeyedRecordStore[_StepRecord] | None = None
        self._history_store: AsyncKeyedRecordStore[_HistoryRecord] | None = (
            None
        )
        if self._steps_db is not None:
            self._step_store = _build_store(
                self._steps_db, _step_to_columns, _step_from_record
            )
        if self._db is not None:
            self._history_store = _build_store(
                self._db, _history_to_columns, _history_from_record
            )

    async def _setup_backend(self) -> None:
        """Set up the database backend using the dataknobs_data factory.

        Backend selection is driven by ``self.config.backend`` (the
        typed ``StorageBackend`` enum on ``StorageConfig``).  Connection
        parameters in ``self.config.connection_params`` are forwarded
        as-is to the dataknobs_data backend constructor.

        If a database was injected via ``__init__()``, this method
        reuses it instead of creating a new instance through the
        factory.
        """
        if self._db is not None:
            # Database was injected — skip factory creation
            return

        backend_type = self.config.backend.value

        # Honor the deprecated ``type`` alias for one release: emit a
        # warning, ignore the value, and continue with the canonical
        # enum.  Existing consumers that redundantly populate ``type``
        # keep working across the transition.  Alias removal is
        # scheduled for the next minor release — see CHANGELOG.
        if self.config.connection_params.get("type") is not None:
            warnings.warn(
                "Passing 'type' in connection_params to "
                "UnifiedDatabaseStorage is deprecated; "
                "StorageConfig.backend is the source of truth. "
                "Remove 'type' from connection_params.",
                DeprecationWarning,
                # stacklevel=3 attributes the warning to the caller of
                # the public ``initialize()`` (frame layout at the call
                # site is: warnings.warn → _setup_backend →
                # initialize → user code).
                stacklevel=3,
            )

        db_config: dict[str, Any] = {
            **self.config.connection_params,
            "backend": backend_type,
        }
        db_config.pop("type", None)  # strip the deprecated alias

        from dataknobs_data.factory import AsyncDatabaseFactory

        self._db = AsyncDatabaseFactory().create(**db_config)

        # Connect to the database if it has a connect method
        if hasattr(self._db, "connect"):
            await self._db.connect()

        # Use the same database for steps unless one was injected.
        # When shared, read methods filter by field existence
        # (history_data / step_data) to isolate record types —
        # see _history_query() and _steps_query().
        if self._steps_db is None:
            self._steps_db = self._db

        # Build the typed stores now that the databases are established.
        if self._step_store is None and self._steps_db is not None:
            self._step_store = _build_store(
                self._steps_db, _step_to_columns, _step_from_record
            )
        if self._history_store is None and self._db is not None:
            self._history_store = _build_store(
                self._db, _history_to_columns, _history_from_record
            )

    @property
    def _uses_shared_db(self) -> bool:
        """True when history and step records share one database namespace.

        This covers both the injected-shared-DB path (caller passes one
        ``database``) and the factory path (``_setup_backend()`` assigns
        ``_steps_db = _db``).  The ``_db is not None`` guard avoids a
        false positive when neither database has been set yet.
        """
        return self._db is not None and self._db is self._steps_db

    def _history_query(self, base_query: Query | None = None) -> Query:
        """Build a query scoped to history records.

        When history and step records share a database, adds an
        ``EXISTS`` filter on ``history_data`` to exclude step records.
        This is backward compatible with legacy records that lack the
        ``record_type`` discriminator — any record with ``history_data``
        is a history record regardless of when it was written.
        """
        query = base_query or Query()
        if self._uses_shared_db:
            query = query.filter("history_data", "exists")
        return query

    def _steps_query(self, base_query: Query | None = None) -> Query:
        """Build a query scoped to step records.

        Mirror of :meth:`_history_query` — filters on ``step_data``
        existence to exclude history records from shared databases.
        """
        query = base_query or Query()
        if self._uses_shared_db:
            query = query.filter("step_data", "exists")
        return query

    @staticmethod
    def _apply_filter_metadata(
        query: Query, filter_metadata: Mapping[str, Any] | None
    ) -> Query:
        """Route a ``filter_metadata`` map through the ``metadata.X`` convention.

        Centralizes the per-key ``query.filter("metadata.K", "=", V)``
        emission used by both ``load_steps`` and ``query_histories`` so
        the routing convention lives at one site.
        """
        if not filter_metadata:
            return query
        for k, v in filter_metadata.items():
            query = query.filter(f"metadata.{k}", "=", v)
        return query

    async def save_history(
        self, history: ExecutionHistory, metadata: dict[str, Any] | None = None
    ) -> str:
        """Save execution history to database.

        Idempotent on ``history.execution_id`` — re-saving the same
        history overwrites the row in place (prior inline-``Record``
        shape appended a fresh row keyed by a random UUID on every save,
        leaving ``load_history`` to return whichever row the backend's
        result ordering happened to surface first).

        Caller-supplied metadata is stored in ``Record.metadata`` (the
        SQL ``metadata`` column) so that dot-notation filters like
        ``metadata.work_order_id`` route to the correct column on all
        backends — the keyed-store serializer signature
        ``_history_to_columns: (_HistoryRecord) -> (data, metadata)``
        makes the channel split type-visible.
        """
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        # Preserve ``created_at`` across upserts: read the existing row
        # (if any) directly via the underlying database so we only pay
        # for the ``created_at`` column lookup, not full
        # ``history_data`` decompression + ``ExecutionHistory.from_dict``
        # reconstruction.  This is the documented legitimate use of the
        # store's ``db`` escape hatch — narrow column read, not a CRUD
        # bypass.  Without this, every resave would reset the original
        # creation timestamp to the most recent save's wall-clock.
        existing_record = await history_store.db.read(history.execution_id)
        existing_created_at: float | None = (
            existing_record.get_value("created_at")
            if existing_record is not None
            else None
        )

        wrapped = _HistoryRecord(
            history=history,
            history_data=self._serialize_history(history),
            metadata=dict(metadata or {}),
            created_at=existing_created_at,
        )
        await history_store.put(history.execution_id, wrapped)
        return history.execution_id

    async def load_history(self, history_id: str) -> ExecutionHistory | None:
        """Load execution history from database."""
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        # Route through the typed store's ``search`` escape hatch (not
        # ``get``) so the ``_history_query`` discriminator filter applies
        # in the shared-db case (history and step records in one
        # namespace).  Matches ``load_steps`` which uses the same pattern.
        records = await history_store.search(
            self._history_query(Query().filter("execution_id", "=", history_id))
        )
        if not records:
            return None
        return _history_from_record(records[0]).history

    async def save_step(
        self,
        execution_id: str,
        step: ExecutionStep,
        parent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Save a single execution step.

        Args:
            execution_id: Execution ID this step belongs to.
            step: Execution step to save.
            parent_id: Parent step ID if branching.
            metadata: Cross-cutting context (``tenant_id``, ``correlation_id``,
                audit info) routed to the underlying record's ``metadata``
                column so it is independently filterable via
                ``metadata.X`` dot-notation on SQL/JSONB backends without
                scanning every row.

        Returns:
            The step's logical id (``step.step_id``).
        """
        if not self._steps_db:
            await self.initialize()
        step_store = _require_store(self._step_store, "Step")

        wrapped = _StepRecord(
            step=step,
            execution_id=execution_id,
            parent_id=parent_id,
            metadata=dict(metadata or {}),
        )
        await step_store.put(wrapped.record_id, wrapped)
        return step.step_id

    async def load_steps(
        self,
        execution_id: str,
        filters: dict[str, Any] | None = None,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[ExecutionStep]:
        """Load execution steps from database.

        Surface mirrors the registry layer (``filter_metadata`` /
        ``sort`` / ``limit`` / ``offset`` kw-only) so consumers
        composing FSM history with bot registries see one consistent
        pagination shape.  The legacy positional ``filters`` dict (data
        columns, equality) stays in place for back-compat.
        """
        if not self._steps_db:
            await self.initialize()
        step_store = _require_store(self._step_store, "Step")

        # Build query — filter to step records when sharing a database.
        # The ``step_data`` EXISTS filter scopes to step records (works
        # with legacy records lacking the ``record_type`` discriminator);
        # the typed store's ``list()`` only supports equality filters, so
        # we route through ``store.search()`` (the escape hatch) to keep
        # the EXISTS predicate, then deserialize each ``Record`` via the
        # store's serializer module.
        query = self._steps_query(Query().filter("execution_id", "=", execution_id))

        if filters:
            for key, value in filters.items():
                query = query.filter(key, "=", value)

        query = self._apply_filter_metadata(query, filter_metadata)

        if sort is not None:
            query.sort_specs = list(sort)
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)

        records = await step_store.search(query)
        return [_step_from_record(r).step for r in records]

    async def query_histories(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
    ) -> list[dict[str, Any]]:
        """Query execution histories.

        Surface mirrors the registry layer (``filter_metadata`` /
        ``sort`` kw-only) so consumers composing FSM history with bot
        registries see one consistent filter/pagination shape.  The
        legacy positional ``filters`` dict — including the
        ``metadata.<key>`` prefix convention — stays in place for
        back-compat; both routes AND-combine when supplied together.
        """
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        # Build query using dataknobs_data Query — scope to history records
        query = self._history_query()

        # Map filter keys to database fields
        for key, value in (filters or {}).items():
            if key in ["fsm_name", "data_mode", "status"]:
                query = query.filter(key, "=", value)
            elif key == "start_time_after":
                query = query.filter("start_time", ">=", value)
            elif key == "start_time_before":
                query = query.filter("start_time", "<=", value)
            elif key == "failed":
                if value:
                    query = query.filter("failed_steps", ">", 0)
                else:
                    query = query.filter("failed_steps", "=", 0)
            elif key.startswith("metadata."):
                query = query.filter(key, "=", value)
            else:
                logger.warning(
                    "Unknown filter key %r in query_histories(), ignoring",
                    key,
                )

        # Symmetry kwarg: ``filter_metadata`` routes through the shared
        # helper so ``query_histories(filter_metadata={"tenant": "A"})``
        # is equivalent to ``query_histories({"metadata.tenant": "A"})``.
        query = self._apply_filter_metadata(query, filter_metadata)

        # Apply pagination.  Default sort is ``start_time DESC`` to
        # preserve back-compat (most-recent-first); a caller-supplied
        # ``sort`` overrides this default in full.
        if sort is not None:
            query.sort_specs = list(sort)
        else:
            query = query.sort_by("start_time", "desc")
        query = query.limit(limit).offset(offset)

        # Use the store's ``search`` escape hatch (returns raw Records,
        # skipping deserialization) so the projection-to-dict below
        # doesn't pay for ``history_data`` decompression and
        # ``ExecutionHistory.from_dict`` reconstruction per row.
        results = []
        search_results = await history_store.search(query)
        for record in search_results:
            # Read metadata from the correct location, with fallback for
            # records stored before the metadata-column migration.
            stored_metadata = record.metadata or {}
            if not stored_metadata:
                legacy = record.get_value("metadata")
                if legacy and isinstance(legacy, dict):
                    stored_metadata = legacy
                    logger.warning(
                        "Record %s has metadata in legacy data-column location; "
                        "re-save to migrate to metadata column",
                        record.get_value("execution_id"),
                    )
            results.append(
                {
                    "id": record["execution_id"],
                    "fsm_name": record["fsm_name"],
                    "data_mode": record["data_mode"],
                    "status": record["status"],
                    "start_time": record["start_time"],
                    "end_time": record.get_value("end_time"),
                    "total_steps": record["total_steps"],
                    "failed_steps": record["failed_steps"],
                    "metadata": stored_metadata,
                }
            )

        return results

    async def delete_history(self, history_id: str) -> bool:
        """Delete execution history."""
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        # Route deletes through the typed history store so a future
        # change to the persisted ``_HistoryRecord`` shape cannot silently
        # bypass the serializer at this site.  The ``_history_query``
        # filter scopes the search in the shared-db case.
        query = self._history_query(
            Query().filter("execution_id", "=", history_id)
        )
        records = await history_store.search(query)

        deleted_count = 0
        for record in records:
            record_id = record.storage_id or record.get_value("id")
            if record_id and await history_store.delete(record_id):
                deleted_count += 1

        # Delete associated steps via the typed step store for the same
        # reason — matches ``save_step`` / ``load_steps`` which both
        # route through ``_step_store``.
        if self._steps_db:
            step_store = _require_store(self._step_store, "Step")
            step_query = self._steps_query(
                Query().filter("execution_id", "=", history_id)
            )
            step_records = await step_store.search(step_query)
            for step_record in step_records:
                step_id = step_record.storage_id or step_record.get_value("id")
                if step_id:
                    await step_store.delete(step_id)

        return deleted_count > 0

    async def get_statistics(self, execution_id: str | None = None) -> dict[str, Any]:
        """Get storage statistics."""
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        if execution_id:
            # Specific execution stats
            query = self._history_query(Query().filter("execution_id", "=", execution_id))

            search_results = await history_store.search(query)
            for record in search_results:
                return {
                    "execution_id": execution_id,
                    "fsm_name": record["fsm_name"],
                    "data_mode": record["data_mode"],
                    "status": record["status"],
                    "total_steps": record["total_steps"],
                    "failed_steps": record["failed_steps"],
                    "start_time": record["start_time"],
                    "end_time": record.get_value("end_time"),
                }
            return {}
        else:
            # Overall stats
            stats = {
                "total_histories": 0,
                "mode_distribution": {},
                "status_distribution": {},
                "backend_type": self.config.backend.value,
            }

            all_records = await history_store.search(self._history_query())
            for record in all_records:
                stats["total_histories"] += 1

                mode = record["data_mode"]
                stats["mode_distribution"][mode] = stats["mode_distribution"].get(mode, 0) + 1

                status = record["status"]
                stats["status_distribution"][status] = (
                    stats["status_distribution"].get(status, 0) + 1
                )

            return stats

    async def cleanup(self, before_timestamp: float | None = None, keep_failed: bool = True) -> int:
        """Clean up old histories."""
        if not self._db:
            await self.initialize()
        history_store = _require_store(self._history_store, "History")

        if before_timestamp is None:
            before_timestamp = time.time() - (7 * 86400)  # 7 days

        # Build query — scope to history records only
        query = self._history_query(Query().filter("start_time", "<", before_timestamp))

        if keep_failed:
            query = query.filter("failed_steps", "=", 0)

        # Get histories to delete
        to_delete = []
        search_results = await history_store.search(query)
        for record in search_results:
            to_delete.append(record["execution_id"])

        # Delete each
        deleted = 0
        for history_id in to_delete:
            if await self.delete_history(history_id):
                deleted += 1

        return deleted

    async def close(self) -> None:
        """Close owned database connections.

        Only closes connections that this instance created via the
        factory.  Injected databases are externally owned — closing
        them would break sibling components sharing the same pool.
        """
        await close_if_owned(
            self._steps_db,
            self._owns_steps_db and self._steps_db is not self._db,
        )
        await close_if_owned(self._db, self._owns_db)


# Register all backends with the same implementation
for backend in [
    StorageBackend.MEMORY,
    StorageBackend.SQLITE,
    StorageBackend.POSTGRES,
    StorageBackend.MONGODB,
    StorageBackend.ELASTICSEARCH,
    StorageBackend.S3,
]:
    StorageFactory.register(backend, UnifiedDatabaseStorage)
