"""Adapter to use dataknobs_data backends as RegistryBackend.

The adapter composes :class:`AsyncKeyedRecordStore[Registration]` rather
than constructing ``Record`` instances inline.  Routing through the
store is the structural prevention contract: the
``(Registration) -> (data, metadata)`` serializer is the only Record-
construction site, so the metadata channel cannot be silently dropped.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dataknobs_data import (
    AsyncDatabase,
    AsyncKeyedRecordStore,
    Record,
    async_database_factory,
)

from .models import Registration

if TYPE_CHECKING:
    from dataknobs_data import SortSpec, StreamConfig

logger = logging.getLogger(__name__)


def _registration_to_columns(
    reg: Registration,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a :class:`Registration` into the ``(data, metadata)`` channels.

    The two-channel return type is load-bearing: the metadata column is
    part of the function's *signature*, so a future change to the
    Registration model can't accidentally drop the metadata channel
    without a type-visible diff at this site.
    """
    data: dict[str, Any] = {
        "bot_id": reg.bot_id,
        "config": reg.config,
        "status": reg.status,
        "created_at": reg.created_at.isoformat() if reg.created_at else None,
        "updated_at": reg.updated_at.isoformat() if reg.updated_at else None,
        "last_accessed_at": (
            reg.last_accessed_at.isoformat() if reg.last_accessed_at else None
        ),
    }
    metadata = dict(reg.metadata or {})
    return data, metadata


def _parse_datetime(val: Any) -> datetime:
    """Parse a datetime value from datetime/ISO string/None."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now(timezone.utc)


def _registration_from_record(record: Record) -> Registration:
    """Reconstruct a :class:`Registration` from a stored ``Record``."""
    return Registration(
        bot_id=record.get_value("bot_id"),
        config=record.get_value("config") or {},
        status=record.get_value("status") or "active",
        metadata=dict(record.metadata or {}),
        created_at=_parse_datetime(record.get_value("created_at")),
        updated_at=_parse_datetime(record.get_value("updated_at")),
        last_accessed_at=_parse_datetime(record.get_value("last_accessed_at")),
    )


class DataKnobsRegistryAdapter:
    """Adapts any dataknobs_data database to the RegistryBackend protocol.

    Internally composes :class:`AsyncKeyedRecordStore[Registration]` so
    every ``Record`` is constructed in one place (the store's
    serializer) and the metadata column is preserved by construction.
    Filter routing for the metadata column happens inside the store via
    the ``filter_metadata`` channel; the adapter never builds Queries
    inline.

    Args:
        database: A dataknobs_data AsyncDatabase instance (optional)
        backend_type: Backend type string ("memory", "postgres", "s3", etc.)
        backend_config: Configuration for the backend

    Example - with explicit database:
        ```python
        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        db = AsyncMemoryDatabase()
        await db.connect()
        adapter = DataKnobsRegistryAdapter(database=db)
        await adapter.initialize()
        ```

    Example - with factory:
        ```python
        adapter = DataKnobsRegistryAdapter(
            backend_type="postgres",
            backend_config={
                "host": "localhost",
                "database": "myapp",
                "table": "bot_registrations"
            }
        )
        await adapter.initialize()
        ```

    Example - configuration-driven:
        ```python
        # config.yaml:
        # registration:
        #   backend: postgres
        #   host: localhost
        #   database: myapp

        adapter = DataKnobsRegistryAdapter.from_config(config["registration"])
        await adapter.initialize()
        ```
    """

    def __init__(
        self,
        database: AsyncDatabase | None = None,
        backend_type: str | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            database: Pre-configured AsyncDatabase instance
            backend_type: Backend type for factory creation
            backend_config: Backend configuration for factory creation
        """
        self._db: AsyncDatabase | None = database
        self._backend_type = backend_type or "memory"
        self._backend_config = backend_config or {}
        self._initialized = False
        self._owns_database = database is None
        self._store: AsyncKeyedRecordStore[Registration] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DataKnobsRegistryAdapter:
        """Create adapter from configuration dictionary.

        Args:
            config: Configuration dict with 'backend' key and backend-specific options

        Returns:
            New DataKnobsRegistryAdapter instance

        Example:
            ```python
            adapter = DataKnobsRegistryAdapter.from_config({
                "backend": "postgres",
                "host": "localhost",
                "database": "myapp",
                "table": "registrations"
            })
            ```
        """
        config = dict(config)
        backend_type = config.pop("backend", "memory")
        return cls(backend_type=backend_type, backend_config=config)

    async def initialize(self) -> None:
        """Initialize the underlying database and wrap it in a keyed store.

        Creates the database connection if not already connected.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        if self._db is None:
            logger.debug(
                "Creating %s database for registry",
                self._backend_type,
            )
            self._db = async_database_factory.create(
                backend=self._backend_type,
                **self._backend_config,
            )

        await self._db.connect()
        self._store = AsyncKeyedRecordStore[Registration](
            self._db,
            serializer=_registration_to_columns,
            deserializer=_registration_from_record,
        )
        self._initialized = True
        logger.info(
            "Registry adapter initialized with %s backend",
            self._backend_type,
        )

    async def close(self) -> None:
        """Close the underlying database.

        Only closes the database if we created it (not if it was passed in).
        """
        if self._db and self._owns_database:
            await self._db.close()
            logger.debug("Registry adapter closed")
        self._initialized = False
        self._store = None

    # --- RegistryBackend Protocol Implementation ---

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Registration:
        """Register a bot or update existing registration.

        If a registration with the same bot_id exists, ``created_at`` is
        preserved and ``updated_at`` / ``last_accessed_at`` are bumped.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary
            status: Registration status (default: "active")
            metadata: Cross-cutting context (tenant_id, audit, feature
                flags). Routed to the underlying ``Record``'s
                ``metadata`` column so it is independently filterable
                via ``filter_metadata`` on list/count.

        Returns:
            Registration object with metadata
        """
        assert self._store is not None, "Adapter not initialized"
        now = datetime.now(timezone.utc)
        existing = await self._store.get(bot_id)
        created_at = existing.created_at if existing else now

        reg = Registration(
            bot_id=bot_id,
            config=config,
            status=status,
            metadata=dict(metadata or {}),
            created_at=created_at,
            updated_at=now,
            last_accessed_at=now,
        )
        await self._store.put(bot_id, reg)
        logger.debug(
            "%s registration for bot %s",
            "Updated" if existing else "Created",
            bot_id,
        )
        return reg

    async def get(self, bot_id: str) -> Registration | None:
        """Get registration by ID, updating ``last_accessed_at``.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration if found, None otherwise
        """
        assert self._store is not None, "Adapter not initialized"
        reg = await self._store.get(bot_id)
        if reg is None:
            return None

        reg.last_accessed_at = datetime.now(timezone.utc)
        await self._store.put(bot_id, reg)
        return reg

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config dict for a bot.

        Also updates ``last_accessed_at`` (touching read).

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        reg = await self.get(bot_id)
        return reg.config if reg else None

    async def peek_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config dict WITHOUT updating ``last_accessed_at``.

        Suitable for infrastructure reads (e.g. preserving a derived
        field across re-registration) that should not register as user
        activity.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        assert self._store is not None, "Adapter not initialized"
        reg = await self._store.get(bot_id)
        return reg.config if reg else None

    async def exists(self, bot_id: str) -> bool:
        """Check if an active registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if registration exists and is active
        """
        assert self._store is not None, "Adapter not initialized"
        reg = await self._store.get(bot_id)
        return reg is not None and reg.status == "active"

    async def unregister(self, bot_id: str) -> bool:
        """Hard delete a registration.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted, False if not found
        """
        assert self._store is not None, "Adapter not initialized"
        result = await self._store.delete(bot_id)
        if result:
            logger.debug("Unregistered bot %s", bot_id)
        return result

    async def deactivate(self, bot_id: str) -> bool:
        """Soft delete (set status to inactive).

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found
        """
        assert self._store is not None, "Adapter not initialized"
        reg = await self._store.get(bot_id)
        if reg is None:
            return False

        reg.status = "inactive"
        reg.updated_at = datetime.now(timezone.utc)
        await self._store.put(bot_id, reg)
        logger.debug("Deactivated bot %s", bot_id)
        return True

    async def list_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List registrations, optionally filtered by status and metadata.

        Args:
            status: Optional equality filter on the ``status`` data
                column (e.g. ``"active"``, ``"inactive"``).  ``None``
                means no status filter (all statuses returned).
            filter_metadata: Optional equality filter over the
                ``metadata`` column.  Routed via the ``metadata.X``
                field-path convention so SQL/JSONB backends push the
                filter into the indexable column.
            sort: Optional sort specification (forwarded to the
                underlying ``AsyncKeyedRecordStore.list``).
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of matching Registration objects.
        """
        assert self._store is not None, "Adapter not initialized"
        filter_data = {"status": status} if status is not None else None
        return await self._store.list(
            filter_data=filter_data,
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_active(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all active registrations.

        Convenience wrapper over :meth:`list_all` that pins
        ``status="active"``.  See :meth:`list_all` for kwarg semantics.
        """
        return await self.list_all(
            status="active",
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all inactive (soft-deleted) registrations.

        Convenience wrapper over :meth:`list_all` that pins
        ``status="inactive"`` — symmetric counterpart to
        :meth:`list_active`.  See :meth:`list_all` for kwarg semantics.
        """
        return await self.list_all(
            status="inactive",
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_ids(self) -> list[str]:
        """List active bot IDs only.

        Returns:
            List of active bot IDs
        """
        regs = await self.list_active()
        return [r.bot_id for r in regs]

    async def count_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count registrations, optionally filtered by status and metadata.

        Routes through :meth:`AsyncKeyedRecordStore.count` (and
        ultimately ``AsyncDatabase.count(query)``) so backends shipping
        ``SELECT COUNT(*) WHERE`` pushdown benefit transparently.

        Args:
            status: Optional equality filter on the ``status`` data
                column.  ``None`` means count all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` column.

        Returns:
            Number of matching registrations.
        """
        assert self._store is not None, "Adapter not initialized"
        filter_data = {"status": status} if status is not None else None
        return await self._store.count(
            filter_data=filter_data,
            filter_metadata=filter_metadata,
        )

    async def count(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count active registrations.

        Convenience wrapper over :meth:`count_all` that pins
        ``status="active"`` — preserves the historical convenience
        shape of "count active bots" without forcing callers to spell
        out the status filter.
        """
        return await self.count_all(
            status="active", filter_metadata=filter_metadata
        )

    async def count_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count inactive (soft-deleted) registrations.

        Symmetric counterpart to :meth:`count`.
        """
        return await self.count_all(
            status="inactive", filter_metadata=filter_metadata
        )

    async def stream(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[Registration]:
        """Stream registrations matching the supplied filters.

        Yields :class:`Registration` values one at a time as the backend
        delivers batches; suitable for large tenant populations that
        won't fit in memory.

        Args:
            status: Optional equality filter on the ``status`` data
                column.  ``None`` means no status filter.
            filter_metadata: Optional equality filter over the
                ``metadata`` column.
            config: Optional :class:`StreamConfig` (batch size, etc.).
        """
        assert self._store is not None, "Adapter not initialized"
        filter_data = {"status": status} if status is not None else None
        async for reg in self._store.stream(
            filter_data=filter_data,
            filter_metadata=filter_metadata,
            config=config,
        ):
            yield reg

    async def clear(self) -> None:
        """Clear all registrations.

        Primarily useful for testing.  Delegates to
        :meth:`AsyncKeyedRecordStore.clear` so the adapter never reaches
        into the underlying database directly for CRUD/bulk operations.
        """
        assert self._store is not None, "Adapter not initialized"
        await self._store.clear()
        logger.debug("Cleared all registrations")
