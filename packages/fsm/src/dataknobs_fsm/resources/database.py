"""Database resource adapter for dataknobs_data backends."""

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Dict, List

from dataknobs_common import CapabilityNotSupportedError
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.database import SyncDatabase, AsyncDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.functions.library.identity import (
    KeyColumnsIdentity,
    RecordIdentity,
)
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceStatus,
)

logger = logging.getLogger(__name__)


class DatabaseResourceAdapter(BaseResourceProvider):
    """Adapter to use dataknobs_data databases as FSM resources.
    
    This adapter wraps dataknobs_data database backends to provide
    resource management capabilities for FSM states.
    """
    
    def __init__(
        self,
        name: str,
        backend: str = "memory",
        **backend_config
    ):
        """Initialize database resource adapter.
        
        Args:
            name: Resource name.
            backend: Database backend type (memory, file, postgres, sqlite, etc).
            **backend_config: Backend-specific configuration passed to DatabaseFactory.
        """
        config = {"backend": backend, **backend_config}
        super().__init__(name, config)
        
        self.backend = backend
        self.factory = DatabaseFactory()
        self._database: SyncDatabase | None = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the database backend."""
        try:
            # Create database using factory
            self._database = self.factory.create(**self.config)
            self.status = ResourceStatus.IDLE
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to initialize database backend '{self.backend}': {e}",
                resource_name=self.name,
                operation="initialize"
            ) from e
    
    def acquire(self, **kwargs) -> SyncDatabase:
        """Acquire database connection/instance.
        
        The returned database object can be used for all database operations.
        For backends that support connection pooling (postgres, etc), this
        manages the underlying connections transparently.
        
        Args:
            **kwargs: Additional parameters (unused, for interface compatibility).
            
        Returns:
            Database instance for operations.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        if self._database is None:
            raise ResourceError(
                "Database not initialized",
                resource_name=self.name,
                operation="acquire"
            )
        
        try:
            # For most backends, we return the same instance
            # Connection pooling is handled internally by the backend
            self.status = ResourceStatus.ACTIVE
            self._resources.append(self._database)
            return self._database
        except Exception as e:
            self.status = ResourceStatus.ERROR
            raise ResourceError(
                f"Failed to acquire database resource: {e}",
                resource_name=self.name,
                operation="acquire"
            ) from e
    
    def release(self, resource: Any) -> None:
        """Release database resource.
        
        For pooled backends, this returns connections to the pool.
        For non-pooled backends, this is a no-op.
        
        Args:
            resource: The database resource to release.
        """
        if resource in self._resources:
            self._resources.remove(resource)
        
        if not self._resources:
            self.status = ResourceStatus.IDLE
        
        # Most backends handle connection cleanup internally
        # We don't need to do anything special here
    
    def validate(self, resource: Any) -> bool:
        """Validate database resource is still usable.
        
        Args:
            resource: The database resource to validate.
            
        Returns:
            True if the resource is valid and usable.
        """
        if resource is None or not isinstance(resource, (SyncDatabase, AsyncDatabase)):
            return False
        
        try:
            # Try a simple operation to validate the connection
            # This will vary by backend but should be lightweight
            if hasattr(resource, 'count'):
                # Try to count records (should return quickly even if 0)
                _ = resource.count()
            return True
        except Exception:
            return False
    
    def health_check(self) -> ResourceHealth:
        """Check database health.
        
        Returns:
            Health status of the database backend.
        """
        if self._database is None:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNKNOWN
        
        try:
            # Perform a simple health check operation
            valid = self.validate(self._database)
            
            if valid:
                self.metrics.record_health_check(True)
                return ResourceHealth.HEALTHY
            else:
                self.metrics.record_health_check(False)
                return ResourceHealth.UNHEALTHY
        except Exception:
            self.metrics.record_health_check(False)
            return ResourceHealth.UNHEALTHY
    
    @contextmanager
    def transaction_context(self, database: SyncDatabase | None = None):
        """Context manager for database transactions.
        
        Note: Transaction support depends on the backend.
        Some backends (memory, file) may not support true transactions.
        
        Args:
            database: Optional database instance to use.
            
        Yields:
            Database instance for operations within transaction.
        """
        if database is None:
            database = self.acquire()
            should_release = True
        else:
            should_release = False
        
        try:
            # For backends that support transactions, we could add
            # transaction begin/commit/rollback logic here
            # For now, we just ensure proper resource cleanup
            yield database
        finally:
            if should_release:
                self.release(database)
    
    def close(self) -> None:
        """Close the database resource and clean up."""
        # Release all tracked resources first
        super().close()
        
        # Close the database backend if it has a close method
        if self._database and hasattr(self._database, 'close'):
            try:
                # Attempt to flush any pending operations
                if hasattr(self._database, 'flush'):
                    self._database.flush()
                
                # Close the connection
                self._database.close()
                logger.debug(f"Successfully closed database connection for {self.name}")
            except AttributeError as e:
                logger.warning(f"Database {self.name} missing expected close method: {e}")
            except Exception as e:
                logger.error(f"Error closing database {self.name}: {e}")
                # Store error for debugging but don't re-raise
                if not hasattr(self, '_cleanup_errors'):
                    self._cleanup_errors = []
                self._cleanup_errors.append(f"Database close error: {e}")
        
        self._database = None
    
    # Convenience methods that delegate to the database
    
    def create(self, record: Record, database: SyncDatabase | None = None) -> str:
        """Create a record in the database.
        
        Args:
            record: Record to create.
            database: Optional database instance.
            
        Returns:
            ID of the created record.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="create")
        
        return database.create(record)
    
    def read(self, record_id: str, database: SyncDatabase | None = None) -> Record | None:
        """Read a record from the database.
        
        Args:
            record_id: ID of the record to read.
            database: Optional database instance.
            
        Returns:
            The record if found, None otherwise.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="read")
        
        return database.read(record_id)
    
    def update(self, record_id: str, record: Record, database: SyncDatabase | None = None) -> bool:
        """Update a record in the database.
        
        Args:
            record_id: ID of the record to update.
            record: Record with updates.
            database: Optional database instance.
            
        Returns:
            True if update was successful.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="update")
        
        return database.update(record_id, record)
    
    def delete(self, record_id: str, database: SyncDatabase | None = None) -> bool:
        """Delete a record from the database.
        
        Args:
            record_id: ID of the record to delete.
            database: Optional database instance.
            
        Returns:
            True if deletion was successful.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="delete")
        
        return database.delete(record_id)
    
    def search(self, query: Query, database: SyncDatabase | None = None) -> list[Record]:
        """Search for records in the database.
        
        Args:
            query: Search query.
            database: Optional database instance.
            
        Returns:
            List of matching records.
        """
        if database is None:
            database = self._database
        
        if database is None:
            raise ResourceError("No database available", resource_name=self.name, operation="search")
        
        return database.search(query)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the database backend.

        Returns:
            Backend information including capabilities.
        """
        if self.factory:
            return self.factory.get_backend_info(self.backend)
        return {"backend": self.backend, "status": self.status.value}


class AsyncDatabaseResourceAdapter(BaseResourceProvider):
    """Async-transport database resource for FSM functions.

    Unlike :class:`DatabaseResourceAdapter` (sync, wraps a ``SyncDatabase``),
    this adapter is built on :class:`dataknobs_data.AsyncDatabase` so its
    ``upsert`` / ``execute_query`` / ``bulk_insert`` operations are real
    coroutines that never block the event loop. It is the resource the
    ``functions/library/database.py`` transforms acquire from
    ``FunctionContext.resources`` (the async engine injects the *adapter*, so
    ``await resource.upsert(...)`` reaches these methods).

    The underlying ``AsyncDatabase`` is created lazily on first use (so
    construction stays sync and cheap, matching the resource-provider
    contract) and is flushed/closed via :meth:`aclose`
    (``ResourceManager.cleanup`` awaits it).
    """

    def __init__(
        self,
        name: str,
        backend: str | None = None,
        **backend_config: Any,
    ):
        """Initialize the async database resource adapter.

        Args:
            name: Resource name.
            backend: Database backend key (``memory``, ``file``, ``postgres``,
                ``sqlite``, …). When omitted, it is resolved from a ``type`` or
                ``backend`` key inside ``backend_config`` — the patterns pass
                the backend dict (``{'type': 'file', 'path': …}``) verbatim, so
                the discriminator is keyed on ``type``.
            **backend_config: Remaining backend parameters (e.g. ``path``,
                ``connection_string``) forwarded to
                ``AsyncDatabase.from_backend``.
        """
        # Resolve the backend from the explicit arg or the dict discriminator.
        # The FSM resource builder hands us the backend dict as **kwargs, where
        # the backend is keyed on ``type`` (mirroring ``run()``'s direct
        # ``AsyncDatabase.from_backend(cfg['type'], cfg)`` usage). Accept both
        # ``type`` and ``backend`` so either spelling works.
        resolved = backend or backend_config.pop("type", None) or backend_config.pop(
            "backend", None
        ) or "memory"
        config = {"backend": resolved, **backend_config}
        super().__init__(name, config)

        self.backend = resolved
        self._backend_config = dict(backend_config)
        self._database: AsyncDatabase | None = None
        # Serialize the cold lazy-open so concurrent first-use (e.g. several
        # records of a batch hitting the shared adapter at once) opens exactly
        # one backend. Without it, each racer would observe ``None``, each
        # ``await from_backend(...)``, and all but the last instance would be
        # orphaned and never closed — a connection/pool leak for pooled
        # backends (postgres). 3.10+ ``asyncio.Lock`` binds to the running loop
        # lazily, so constructing it here (outside any loop) is safe.
        self._db_lock = asyncio.Lock()

    async def _ensure_db(self) -> AsyncDatabase:
        """Lazily open the underlying ``AsyncDatabase`` (once, race-safe)."""
        if self._database is not None:
            return self._database
        async with self._db_lock:
            # Double-checked: a racer may have opened it while we waited.
            if self._database is None:
                self._database = await AsyncDatabase.from_backend(
                    self.backend,
                    {"type": self.backend, **self._backend_config},
                )
                self.status = ResourceStatus.ACTIVE
        return self._database

    def acquire(self, **kwargs: Any) -> "AsyncDatabaseResourceAdapter":
        """Acquire the adapter itself.

        The database functions call ``await resource.upsert(...)`` /
        ``await resource.execute_query(...)`` — methods defined on the adapter,
        not on the raw ``AsyncDatabase`` — so the adapter is the resource handed
        to ``FunctionContext.resources``.

        Args:
            **kwargs: Unused (interface compatibility).

        Returns:
            This adapter instance.
        """
        self.status = ResourceStatus.ACTIVE
        return self

    def release(self, resource: Any) -> None:
        """Release the adapter (no-op; the lazy DB lives until ``aclose``)."""
        # The shared async database stays open across acquisitions; it is torn
        # down once in ``aclose`` so pooled connections are not churned per
        # per-record acquire/release.
        return None

    def validate(self, resource: Any) -> bool:
        """Validate the acquired resource is this adapter."""
        return resource is self

    #: Backend keys whose ``*_batch`` operations are wrapped in a backend-level
    #: transaction, so ``create_batch`` is all-or-nothing. PR-A interim source
    #: of truth for :meth:`_supports_atomic_batch`; superseded by the data-layer
    #: ``AsyncDatabase`` transaction capability when it lands.
    _ATOMIC_BATCH_BACKENDS = frozenset(
        {"sqlite", "sqlite3", "postgres", "postgresql", "pg", "duckdb"}
    )

    def _supports_atomic_batch(self) -> bool:
        """Whether the backing backend gives all-or-nothing ``create_batch``."""
        return self.backend in self._ATOMIC_BATCH_BACKENDS

    @staticmethod
    def _resolve_identity(
        identity: RecordIdentity | None, key_columns: List[str] | None
    ) -> RecordIdentity | None:
        """Pick the identity strategy: explicit ``identity`` wins, else key columns."""
        if identity is not None:
            return identity
        if key_columns:
            return KeyColumnsIdentity(key_columns)
        return None

    async def upsert(
        self,
        table: str,
        records: List[Dict[str, Any]],
        key_columns: List[str] | None = None,
        value_columns: List[str] | None = None,
        on_conflict: str = "update",
        identity: RecordIdentity | None = None,
    ) -> Dict[str, Any]:
        """Upsert rows into the backing async database.

        Args:
            table: Logical table name (informational for backends without
                tables; the records are keyed by their derived id).
            records: Row dicts to upsert.
            key_columns: Columns forming the unique key. Used both to derive the
                storage id (when ``identity`` is not supplied) and to scope the
                ``value_columns`` projection.
            value_columns: Columns to persist (if ``None``, persist the whole
                row).
            on_conflict: ``"update"`` (default, write-through), ``"ignore"``
                (skip existing ids), or ``"error"`` (raise on an existing id).
            identity: Explicit :class:`RecordIdentity`; overrides ``key_columns``
                for id derivation. When neither is supplied, ids are
                backend-assigned.

        Returns:
            ``{"affected_rows": <count>}``.
        """
        db = await self._ensure_db()
        ident = self._resolve_identity(identity, key_columns)
        affected = 0
        for row in records:
            record_id = ident.derive(row) if ident is not None else None
            if value_columns is not None:
                cols = (key_columns or []) + value_columns
                payload = {col: row.get(col) for col in cols}
            else:
                payload = dict(row)

            if record_id is not None and on_conflict in ("ignore", "error"):
                if await db.exists(record_id):
                    if on_conflict == "error":
                        raise ResourceError(
                            f"Record '{record_id}' already exists in '{table}'",
                            resource_name=self.name,
                            operation="upsert",
                        )
                    continue  # on_conflict == "ignore"

            record = Record(payload)
            if record_id is not None:
                await db.upsert(record_id, record)
            else:
                await db.upsert(record)
            affected += 1
        return {"affected_rows": affected}

    async def bulk_insert(
        self,
        table: str,
        records: List[Dict[str, Any]],
        columns: List[str] | None = None,
        on_duplicate: str = "error",
        identity: RecordIdentity | None = None,
    ) -> Dict[str, Any]:
        """Insert rows into the backing async database, honoring ``on_duplicate``.

        Args:
            table: Logical table name (informational).
            records: Row dicts to insert.
            columns: Columns to persist (if ``None``, persist the whole row).
            on_duplicate: Conflict policy when ``identity`` resolves a row to an
                id that already exists: ``"error"`` (raise), ``"ignore"`` (skip
                the row), or ``"update"`` (overwrite). Evaluated only when an
                ``identity`` is supplied — without one, rows are created with
                backend-assigned ids and no duplicate detection occurs.
            identity: :class:`RecordIdentity` deriving each row's id, enabling
                duplicate detection. ``None`` → pure create.

        Returns:
            ``{"affected_rows": <count>}`` (skipped ``ignore`` rows are not
            counted).
        """
        db = await self._ensure_db()
        affected = 0
        for row in records:
            payload = (
                {col: row.get(col) for col in columns} if columns else dict(row)
            )
            record_id = identity.derive(row) if identity is not None else None

            if record_id is not None:
                if await db.exists(record_id):
                    if on_duplicate == "error":
                        raise ResourceError(
                            f"Record '{record_id}' already exists in '{table}'",
                            resource_name=self.name,
                            operation="bulk_insert",
                        )
                    if on_duplicate == "ignore":
                        continue
                    # on_duplicate == "update": fall through and overwrite.
                await db.upsert(record_id, Record(payload))
            else:
                await db.create(Record(payload))
            affected += 1
        return {"affected_rows": affected}

    async def commit_batch(
        self,
        records: List[Dict[str, Any]],
        *,
        identity: RecordIdentity | None = None,
        atomicity: str = "best_effort",
    ) -> Dict[str, Any]:
        """Persist a batch of records, atomically where the backend allows.

        Without ``identity`` the batch is written via ``create_batch`` — which
        is all-or-nothing on transactional backends (postgres/sqlite/duckdb).
        With ``identity`` each row is upserted under its derived id, so a
        re-commit of the same batch is idempotent.

        Args:
            records: Row dicts to persist.
            identity: Optional :class:`RecordIdentity`. ``None`` → create_batch
                (backend-assigned ids); set → idempotent per-row upsert.
            atomicity: ``"best_effort"`` (default) proceeds on any backend,
                logging at DEBUG when the backend cannot guarantee
                all-or-nothing; ``"require"`` raises
                :class:`CapabilityNotSupportedError` on a non-transactional
                backend (and on the idempotent-upsert path, which is not
                batch-atomic without the transaction capability).

        Returns:
            ``{"affected_rows": <count>}``.

        Raises:
            ConfigurationError: Unknown ``atomicity`` policy.
            CapabilityNotSupportedError: ``atomicity="require"`` on a backend
                that cannot provide an atomic batch.
        """
        if atomicity not in ("best_effort", "require"):
            raise ConfigurationError(
                f"Unknown atomicity policy '{atomicity}' "
                "(expected 'best_effort' or 'require')"
            )
        db = await self._ensure_db()
        if not records:
            return {"affected_rows": 0}

        atomic = self._supports_atomic_batch()
        if atomicity == "require" and not atomic:
            raise CapabilityNotSupportedError("atomic batch commit", self)
        if not atomic:
            logger.debug(
                "commit_batch on non-transactional backend '%s' is not "
                "all-or-nothing (atomicity=best_effort)",
                self.backend,
            )

        if identity is None:
            ids = await db.create_batch([Record(dict(row)) for row in records])
            return {"affected_rows": len(ids)}

        # Idempotent per-row upsert. The loop is not batch-atomic without the
        # transaction capability, so "require" cannot be honored here.
        if atomicity == "require":
            raise CapabilityNotSupportedError(
                "atomic idempotent batch commit", self
            )
        affected = 0
        for row in records:
            record_id = identity.derive(row)
            record = Record(dict(row))
            if record_id is not None:
                await db.upsert(record_id, record)
            else:
                await db.upsert(record)
            affected += 1
        return {"affected_rows": affected}

    async def execute_query(
        self,
        query: Query | None = None,
        params: Dict[str, Any] | None = None,
        fetch_one: bool = False,
        as_dict: bool = True,
    ) -> Any:
        """Read records from the backing async database.

        Args:
            query: A dataknobs :class:`~dataknobs_data.Query` (or ``None`` to
                read all). Raw SQL strings are not supported by the
                ``AsyncDatabase`` abstraction.
            params: Unused (interface compatibility).
            fetch_one: If ``True``, return the first record (or ``None``).
            as_dict: If ``True``, return records as dicts.

        Returns:
            A list of records (or a single record / ``None`` when
            ``fetch_one``).
        """
        if isinstance(query, str):
            raise ResourceError(
                "AsyncDatabaseResourceAdapter.execute_query does not support "
                "raw SQL strings; pass a dataknobs Query or None",
                resource_name=self.name,
                operation="execute_query",
            )

        db = await self._ensure_db()
        search = query if isinstance(query, Query) else Query()
        results: list[Any] = []
        async for record in db.stream_read(search):
            results.append(record.to_dict() if as_dict else record)
            if fetch_one:
                break

        if fetch_one:
            return results[0] if results else None
        return results

    async def aclose(self) -> None:
        """Flush and close the underlying async database."""
        if self._database is not None:
            try:
                flush = getattr(self._database, "flush", None)
                if flush is not None:
                    result = flush()
                    if hasattr(result, "__await__"):
                        await result
                await self._database.close()
            except Exception as exc:  # pragma: no cover - defensive teardown
                logger.error(
                    "Error closing async database resource %s: %s", self.name, exc
                )
            finally:
                self._database = None
        self.status = ResourceStatus.CLOSED
