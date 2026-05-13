"""Registry backend protocol for pluggable storage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dataknobs_data import SortSpec, StreamConfig

    from .models import Registration


@runtime_checkable
class RegistryBackend(Protocol):
    """Protocol for bot registry storage backends.

    Implementations store bot configurations with metadata.
    The backend is responsible for persistence; the BotRegistry
    handles caching and bot instantiation.

    This protocol defines the interface for storage backends:
    - InMemoryBackend: Simple dict storage (default, good for tests)
    - PostgreSQLBackend: Database persistence (future/external)
    - RedisBackend: Distributed caching (future/external)

    All methods are async to support both sync and async backends.

    Example:
        ```python
        class MyCustomBackend:
            async def initialize(self) -> None:
                # Setup database connection
                ...

            async def register(self, bot_id: str, config: dict, status: str = "active"):
                # Store in database
                ...

            # ... implement other methods
        ```
    """

    async def initialize(self) -> None:
        """Initialize the backend (create tables, connections, etc.).

        Called before the backend is used. Should be idempotent.
        """
        ...

    async def close(self) -> None:
        """Close the backend (release connections, etc.).

        Called when the registry is shutting down.
        """
        ...

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Registration:
        """Register a bot or update existing registration.

        If a registration with the same bot_id exists, it should be updated
        (config replaced, status updated, updated_at set to now).

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary (should be portable)
            status: Registration status (default: active)
            metadata: Cross-cutting context (tenant_id, audit, feature flags).
                Backends should route these fields to their ``metadata``
                column so they are independently filterable via
                ``filter_metadata`` on list/count.

        Returns:
            Registration object with metadata
        """
        ...

    async def get(self, bot_id: str) -> Registration | None:
        """Get registration by ID.

        Should update last_accessed_at timestamp on access.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration if found, None otherwise
        """
        ...

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config for a bot.

        Convenience method that returns only the config dict.
        Should also update last_accessed_at.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        ...

    async def peek_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config for a bot WITHOUT updating ``last_accessed_at``.

        Companion to :meth:`get_config` for callers that need to read a
        stored config as part of internal bookkeeping (e.g. preserving a
        derived field across re-registration, audit-counting bound bots)
        and don't want the read to register as user activity. Backends
        MUST guarantee this method leaves any activity-tracking state
        the backend itself maintains unchanged.

        The contract scopes only to backend-local activity-tracking
        state. Backends without local activity tracking — for example,
        clients that delegate to a server which owns its own
        access-tracking semantics — MAY satisfy this contract by
        delegating to :meth:`get_config`; the non-mutation guarantee is
        then trivially satisfied at the client surface. Such backends
        deliberately do not impose a wire-protocol distinction (e.g. a
        ``?peek=true`` query parameter) on the server; servers that
        want to honor the distinction define their own contract
        independently.

        Use ``get_config`` for user-facing reads where "this bot was
        accessed" is the right signal. Use ``peek_config`` for
        infrastructure reads where it is not.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        ...

    async def exists(self, bot_id: str) -> bool:
        """Check if an active registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if registration exists and is active
        """
        ...

    async def unregister(self, bot_id: str) -> bool:
        """Hard delete a registration.

        Permanently removes the registration from storage.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def deactivate(self, bot_id: str) -> bool:
        """Soft delete (set status to inactive).

        Marks the registration as inactive without deleting.
        Inactive registrations should not be returned by exists()
        or list_active().

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found
        """
        ...

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
                column.  ``None`` returns all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.  ``None`` / empty mapping is
                no-filter.
            sort: Optional sort specification; backends should honor
                this when possible.
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of matching Registration objects.
        """
        ...

    async def list_active(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all active registrations.

        Convenience for :meth:`list_all` with ``status="active"``.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` channel. Each key/value pair must match for a
                registration to be included. ``None`` / empty mapping is
                no-filter.
            sort: Optional sort specification.
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of active Registration objects
        """
        ...

    async def list_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List all inactive (soft-deleted) registrations.

        Symmetric counterpart to :meth:`list_active`.  Convenience for
        :meth:`list_all` with ``status="inactive"``.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.
            sort: Optional sort specification.
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of inactive Registration objects
        """
        ...

    async def list_ids(self) -> list[str]:
        """List active bot IDs only.

        More efficient than list_active() when only IDs are needed.

        Returns:
            List of active bot IDs
        """
        ...

    async def count_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count registrations, optionally filtered by status and metadata.

        Args:
            status: Optional equality filter on the ``status`` data
                column.  ``None`` counts all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.

        Returns:
            Number of matching registrations.
        """
        ...

    async def count(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count active registrations.

        Convenience for :meth:`count_all` with ``status="active"``.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` channel. ``None`` / empty mapping is
                no-filter.

        Returns:
            Number of active registrations
        """
        ...

    async def count_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count inactive (soft-deleted) registrations.

        Symmetric counterpart to :meth:`count`.

        Args:
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.

        Returns:
            Number of inactive registrations.
        """
        ...

    def stream(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[Registration]:
        """Stream registrations matching the supplied filters.

        Yields :class:`Registration` instances one at a time so backends
        with large tenant populations can be iterated without loading
        everything into memory.

        Note this method is an async generator (declared as ``def``
        returning ``AsyncIterator``) — call it with ``async for``, not
        ``await``.

        Args:
            status: Optional equality filter on the ``status`` data
                column.  ``None`` streams all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.
            config: Optional :class:`StreamConfig` (batch size, etc.).
        """
        ...

    async def clear(self) -> None:
        """Clear all registrations.

        Primarily useful for testing.
        """
        ...
