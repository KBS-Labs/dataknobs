"""Rubric registry for storing and retrieving rubrics.

This module provides persistent rubric storage backed by AsyncDatabase,
supporting versioned rubric management and target-type lookups.

Internally composes :class:`AsyncKeyedRecordStore[Rubric]` so every
``Record`` is built in one place (the store's serializer) and the
``metadata`` channel is preserved by construction.  The serializer
signature ``(Rubric) -> (data, metadata)`` makes the metadata column
part of the function's type — a future change to the model cannot
silently drop the metadata channel without a type-visible diff.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = RubricRegistry(db)
    >>> rubric_id = await registry.register(rubric)
    >>> retrieved = await registry.get(rubric_id)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dataknobs_data import (
    AsyncDatabase,
    AsyncKeyedRecordStore,
    Filter,
    Operator,
    Query,
    SortSpec,
)

from ..utils.versioned_records import iter_latest_records
from .models import Rubric

if TYPE_CHECKING:
    from dataknobs_data import Record

logger = logging.getLogger(__name__)


def _rubric_to_columns(
    rubric: Rubric,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a :class:`Rubric` into the ``(data, metadata)`` channels.

    The two-channel return type is load-bearing: the metadata column is
    part of the function's *signature*, so a future change to the
    Rubric model can't accidentally drop the metadata channel without
    a type-visible diff at this site.

    ``_version_key`` is a storage-layer field used by :meth:`list_all`
    to distinguish "latest pointer" records (stored at ``rubric.id``)
    from versioned snapshots (stored at ``f"{rubric.id}:{version}"``).
    """
    data = rubric.to_dict()
    metadata = dict(data.pop("metadata", None) or {})
    data["_version_key"] = f"{rubric.id}:{rubric.version}"
    return data, metadata


def _rubric_from_record(record: Record) -> Rubric:
    """Reconstruct a :class:`Rubric` from a stored ``Record``.

    Prefers the new ``metadata`` column; falls back to a legacy
    ``"metadata"`` key in ``record.data`` for records written before
    the migration to ``AsyncKeyedRecordStore`` (pre-migration writes
    routed the entire payload — including the model's ``metadata``
    field — into the data column).  This dual-shape read is the
    one-time backwards-compat surface and lives at exactly one site
    so consumers cannot bypass it.
    """
    data = dict(record.data or {})
    data.pop("_version_key", None)
    if record.metadata:
        data["metadata"] = dict(record.metadata)
    elif "metadata" not in data:
        data["metadata"] = {}
    return Rubric.from_dict(data)


class RubricRegistry:
    """Persistent rubric storage backed by AsyncDatabase.

    Rubrics are stored as versioned records with composite keys
    ``"{rubric_id}:{version}"``. A pointer record at ``"{rubric_id}"``
    tracks the latest version.

    Internally composes :class:`AsyncKeyedRecordStore[Rubric]` so
    every ``Record`` is constructed in one place (the store's
    serializer) and the metadata column is preserved by construction.

    Args:
        db: The async database backend for storage.
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self._store: AsyncKeyedRecordStore[Rubric] = AsyncKeyedRecordStore[
            Rubric
        ](
            db,
            serializer=_rubric_to_columns,
            deserializer=_rubric_from_record,
        )

    async def register(self, rubric: Rubric) -> str:
        """Store a rubric and return its ID.

        Creates both a versioned record and a latest-pointer record.

        Write order: **snapshot first, then pointer**.  Matches
        :meth:`ArtifactRegistry._store_artifact` and is the
        transactionally safer ordering — at any partial-write
        observation point the pointer either does not yet exist or
        references a snapshot that already does, never a missing
        snapshot.

        The :attr:`Rubric.metadata` field is routed to the underlying
        record's ``metadata`` column so it is independently filterable
        via ``filter_metadata`` on :meth:`get_for_target` and
        :meth:`list_all` without scanning every row.

        Args:
            rubric: The rubric to store.

        Returns:
            The rubric ID.
        """
        version_key = f"{rubric.id}:{rubric.version}"
        await self._store.put(version_key, rubric)
        await self._store.put(rubric.id, rubric)

        logger.info(
            "Registered rubric '%s' version '%s'",
            rubric.id,
            rubric.version,
        )
        return rubric.id

    async def get(
        self, rubric_id: str, version: str | None = None
    ) -> Rubric | None:
        """Retrieve a rubric by ID, optionally at a specific version.

        Args:
            rubric_id: The rubric identifier.
            version: If provided, retrieves that specific version.
                If None, retrieves the latest version.

        Returns:
            The rubric, or None if not found.
        """
        key = f"{rubric_id}:{version}" if version else rubric_id
        return await self._store.get(key)

    async def get_for_target(
        self,
        target_type: str,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Rubric]:
        """Get all rubrics applicable to a target type.

        This searches the latest version of each rubric. Returns only
        rubrics whose target_type matches the given type.

        Args:
            target_type: The type to filter by (e.g., "content", "rubric").
            filter_metadata: Equality filter over the ``metadata`` column.
                Entries are AND-combined with the ``target_type`` filter
                and routed via the ``metadata.X`` field-path convention
                so SQL/JSONB backends push the filter into the indexable
                column.
            sort: Optional multi-key sort specification.  Pushed down to
                the database query so SQL backends can use indexes when
                available.  The first-occurrence-per-id deduplication
                pass preserves the database's ordering — pointer and
                snapshot rows for the same rubric carry identical
                sort-relevant fields, so whichever pointer comes first
                wins and the overall order is the database's order.
            limit: Optional row limit applied **after** dedup.  Not
                pushed to the database because the dual-write storage
                shape (latest-pointer + versioned snapshots) makes the
                pre-dedup row count diverge from the post-dedup rubric
                count: a database-level ``LIMIT 5`` could return 5
                snapshot rows that all dedup away, leaving an empty
                result when rubrics exist further down.
            offset: Optional row offset applied **after** dedup; same
                reason as ``limit``.

        Returns:
            List of matching rubrics.
        """
        q = Query(
            filters=self._build_filters(
                target_type=target_type,
                filter_metadata=filter_metadata,
            )
        )
        if sort is not None:
            q.sort_specs = list(sort)

        records = await self._store.search(q)
        rubrics = [
            _rubric_from_record(r) for r in iter_latest_records(records)
        ]

        # Apply pagination after dedup.  See docstring for why these
        # cannot be pushed to the database alongside the filters.
        if offset is not None:
            rubrics = rubrics[offset:]
        if limit is not None:
            rubrics = rubrics[:limit]

        return rubrics

    async def update(self, rubric: Rubric) -> str:
        """Store a new version of an existing rubric.

        The rubric should have its version already bumped. This creates
        a new versioned record and updates the latest pointer.

        Args:
            rubric: The rubric with updated version.

        Returns:
            The rubric ID.
        """
        return await self.register(rubric)

    async def delete(self, rubric_id: str) -> bool:
        """Remove a rubric's latest pointer record.

        Note: Versioned records are retained for audit history.

        Args:
            rubric_id: The rubric to remove.

        Returns:
            True if the rubric was found and deleted, False otherwise.
        """
        if not await self._store.exists(rubric_id):
            return False
        return await self._store.delete(rubric_id)

    async def list_all(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Rubric]:
        """List the latest version of all registered rubrics.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` column.  Routed via the ``metadata.X``
                field-path convention so SQL/JSONB backends push the
                filter into the indexable column.
            sort: Optional multi-key sort specification, pushed down to
                the database query.  See :meth:`get_for_target` for the
                interaction between sort and the dedup pass.
            limit: Optional row limit applied **after** dedup.  See
                :meth:`get_for_target` for why pagination is post-dedup
                under the dual-write storage shape.
            offset: Optional row offset applied **after** dedup.

        Returns:
            List of all rubrics (latest versions only).
        """
        q = Query(
            filters=self._build_filters(
                target_type=None,
                filter_metadata=filter_metadata,
            )
        )
        if sort is not None:
            q.sort_specs = list(sort)

        records = await self._store.search(q)
        rubrics = [
            _rubric_from_record(r) for r in iter_latest_records(records)
        ]

        if offset is not None:
            rubrics = rubrics[offset:]
        if limit is not None:
            rubrics = rubrics[:limit]

        return rubrics

    async def count_for_target(
        self,
        target_type: str,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count rubrics matching :meth:`get_for_target`'s filter shape.

        Mirrors :meth:`get_for_target` parameter-for-parameter (minus
        ``sort``/``limit``/``offset``, which don't affect the count)
        and is equivalent to ``len(await self.get_for_target(...))``
        after dedup.

        Cost note:
            The storage shape (dual-write: latest pointer + versioned
            snapshot per version) means the database's row count is
            not the rubric count — pre-dedup it includes snapshots,
            post-dedup it does not.  A pushdown-only count is therefore
            not safe.  This implementation runs the same search as
            :meth:`get_for_target`, applies the dedup pass, and returns
            the count.  Performance scales with the matching row count.

        Args:
            target_type: The type to filter by.
            filter_metadata: Equality filter over the ``metadata`` column.

        Returns:
            Number of matching rubrics (deduplicated to one per
            rubric ID).
        """
        records = await self._store.search(
            Query(
                filters=self._build_filters(
                    target_type=target_type,
                    filter_metadata=filter_metadata,
                )
            )
        )
        return sum(1 for _ in iter_latest_records(records))

    async def count_all(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count rubrics matching :meth:`list_all`'s filter shape.

        See :meth:`count_for_target` for the cost note.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` column.

        Returns:
            Number of registered rubrics (deduplicated to one per
            rubric ID).
        """
        records = await self._store.search(
            Query(
                filters=self._build_filters(
                    target_type=None,
                    filter_metadata=filter_metadata,
                )
            )
        )
        return sum(1 for _ in iter_latest_records(records))

    def _build_filters(
        self,
        *,
        target_type: str | None,
        filter_metadata: Mapping[str, Any] | None,
    ) -> list[Filter]:
        """Compose the structural filter list for rubric listing methods.

        Shared by :meth:`get_for_target` and :meth:`list_all`.  Routes
        ``filter_metadata`` entries through the ``metadata.X``
        field-path convention so SQL/JSONB backends push them into
        the indexable column.
        """
        filters: list[Filter] = []
        if target_type:
            filters.append(Filter("target_type", Operator.EQ, target_type))
        for k, v in (filter_metadata or {}).items():
            filters.append(Filter(f"metadata.{k}", Operator.EQ, v))
        return filters

    @classmethod
    async def from_config(
        cls, config: dict[str, Any], db: AsyncDatabase
    ) -> RubricRegistry:
        """Create a registry and load rubrics from a config dict.

        The config should have a ``"rubrics"`` key containing a list
        of rubric definitions (each as a dict matching Rubric.from_dict).

        Args:
            config: Configuration dict with rubric definitions.
            db: The async database backend.

        Returns:
            A populated RubricRegistry.
        """
        registry = cls(db)
        rubric_defs = config.get("rubrics", [])
        for rubric_data in rubric_defs:
            rubric = Rubric.from_dict(rubric_data)
            await registry.register(rubric)
            logger.info(
                "Loaded rubric '%s' from config",
                rubric.id,
            )
        return registry
