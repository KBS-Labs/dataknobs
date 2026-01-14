"""Adapter to use dataknobs_data backends as RegistryBackend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from dataknobs_data import (
    AsyncDatabase,
    Filter,
    Operator,
    Query,
    Record,
    async_database_factory,
)

from .models import Registration

logger = logging.getLogger(__name__)


class DataKnobsRegistryAdapter:
    """Adapts any dataknobs_data database to the RegistryBackend protocol.

    This allows using Memory, PostgreSQL, S3, SQLite, or any other
    dataknobs_data backend for bot registration storage.

    The adapter stores registrations as Records with the following fields:
    - bot_id: Unique bot identifier (also used as storage_id)
    - config: Bot configuration dictionary
    - status: Registration status (active, inactive, error)
    - created_at: ISO timestamp of creation
    - updated_at: ISO timestamp of last update
    - last_accessed_at: ISO timestamp of last access

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
        self._db = database
        self._backend_type = backend_type or "memory"
        self._backend_config = backend_config or {}
        self._initialized = False
        self._owns_database = database is None  # Track if we created the database

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
        # Make a copy to avoid modifying the original
        config = dict(config)
        backend_type = config.pop("backend", "memory")
        return cls(backend_type=backend_type, backend_config=config)

    async def initialize(self) -> None:
        """Initialize the underlying database.

        Creates the database connection if not already connected.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        if self._db is None:
            # Create database from factory
            logger.debug(
                "Creating %s database for registry",
                self._backend_type,
            )
            self._db = async_database_factory.create(
                backend=self._backend_type,
                **self._backend_config,
            )

        await self._db.connect()
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

    # --- RegistryBackend Protocol Implementation ---

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
    ) -> Registration:
        """Register a bot or update existing registration.

        If a registration with the same bot_id exists, it will be updated
        (config replaced, status updated, updated_at set to now).

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary
            status: Registration status (default: "active")

        Returns:
            Registration object with metadata
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Check for existing registration
        existing = await self._db.read(bot_id)

        if existing:
            # Update existing - preserve created_at
            created_at = existing.get_value("created_at")
            record = Record(
                {
                    "bot_id": bot_id,
                    "config": config,
                    "status": status,
                    "created_at": created_at,
                    "updated_at": now_iso,
                    "last_accessed_at": now_iso,
                },
                storage_id=bot_id,
            )
            await self._db.update(bot_id, record)
            logger.debug("Updated registration for bot %s", bot_id)
        else:
            # Create new
            record = Record(
                {
                    "bot_id": bot_id,
                    "config": config,
                    "status": status,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "last_accessed_at": now_iso,
                },
                storage_id=bot_id,
            )
            await self._db.create(record)
            logger.debug("Created registration for bot %s", bot_id)

        return self._record_to_registration(record)

    async def get(self, bot_id: str) -> Registration | None:
        """Get registration by ID, updating last_accessed_at.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration if found, None otherwise
        """
        record = await self._db.read(bot_id)
        if not record:
            return None

        # Update last_accessed_at
        now_iso = datetime.now(timezone.utc).isoformat()
        record.set_value("last_accessed_at", now_iso)
        await self._db.update(bot_id, record)

        return self._record_to_registration(record)

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config dict for a bot.

        Also updates last_accessed_at timestamp.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        record = await self._db.read(bot_id)
        if not record:
            return None

        # Update last_accessed_at
        now_iso = datetime.now(timezone.utc).isoformat()
        record.set_value("last_accessed_at", now_iso)
        await self._db.update(bot_id, record)

        return record.get_value("config")

    async def exists(self, bot_id: str) -> bool:
        """Check if an active registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if registration exists and is active
        """
        record = await self._db.read(bot_id)
        if not record:
            return False
        return record.get_value("status") == "active"

    async def unregister(self, bot_id: str) -> bool:
        """Hard delete a registration.

        Permanently removes the registration from storage.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted, False if not found
        """
        result = await self._db.delete(bot_id)
        if result:
            logger.debug("Unregistered bot %s", bot_id)
        return result

    async def deactivate(self, bot_id: str) -> bool:
        """Soft delete (set status to inactive).

        Marks the registration as inactive without deleting.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found
        """
        record = await self._db.read(bot_id)
        if not record:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        record.set_value("status", "inactive")
        record.set_value("updated_at", now_iso)
        result = await self._db.update(bot_id, record)
        if result:
            logger.debug("Deactivated bot %s", bot_id)
        return result

    async def list_active(self) -> list[Registration]:
        """List all active registrations.

        Returns:
            List of active Registration objects
        """
        query = Query(filters=[Filter("status", Operator.EQ, "active")])
        records = await self._db.search(query)
        return [self._record_to_registration(r) for r in records]

    async def list_all(self) -> list[Registration]:
        """List all registrations including inactive.

        Returns:
            List of all Registration objects
        """
        records = await self._db.all()
        return [self._record_to_registration(r) for r in records]

    async def list_ids(self) -> list[str]:
        """List active bot IDs only.

        More efficient than list_active() when only IDs are needed.

        Returns:
            List of active bot IDs
        """
        regs = await self.list_active()
        return [r.bot_id for r in regs]

    async def count(self) -> int:
        """Count active registrations.

        Returns:
            Number of active registrations
        """
        return len(await self.list_active())

    async def clear(self) -> None:
        """Clear all registrations.

        Primarily useful for testing.
        """
        # Get all records and delete them
        records = await self._db.all()
        for record in records:
            storage_id = record.storage_id or record.get_value("bot_id")
            if storage_id:
                await self._db.delete(storage_id)
        logger.debug("Cleared all registrations")

    # --- Helpers ---

    def _record_to_registration(self, record: Record) -> Registration:
        """Convert a dataknobs Record to a Registration.

        Args:
            record: Record from dataknobs_data backend

        Returns:
            Registration instance
        """
        return Registration(
            bot_id=record.get_value("bot_id"),
            config=record.get_value("config") or {},
            status=record.get_value("status") or "active",
            created_at=self._parse_datetime(record.get_value("created_at")),
            updated_at=self._parse_datetime(record.get_value("updated_at")),
            last_accessed_at=self._parse_datetime(record.get_value("last_accessed_at")),
        )

    @staticmethod
    def _parse_datetime(val: Any) -> datetime:
        """Parse a datetime value from various formats.

        Args:
            val: datetime, string (ISO format), or None

        Returns:
            datetime instance (defaults to now if val is None)
        """
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            return datetime.fromisoformat(val)
        return datetime.now(timezone.utc)
