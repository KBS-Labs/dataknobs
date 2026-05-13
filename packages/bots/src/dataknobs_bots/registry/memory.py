"""In-memory implementation of RegistryBackend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dataknobs_data import SortOrder

from .models import Registration

if TYPE_CHECKING:
    from dataknobs_data import SortSpec, StreamConfig


def _matches_metadata(
    reg: Registration, filter_metadata: Mapping[str, Any] | None
) -> bool:
    """Return True if every key/value in ``filter_metadata`` matches ``reg.metadata``.

    Empty / ``None`` filter mapping is treated as no-filter (always matches).
    """
    if not filter_metadata:
        return True
    return all(reg.metadata.get(k) == v for k, v in filter_metadata.items())


def _sort_key(reg: Registration, field: str) -> Any:
    """Resolve a sort field on a Registration to a comparable value.

    Whitelist known structural fields; falls back to ``metadata.X``
    lookup for metadata-prefixed fields (matches the SQL/JSONB
    backends' field-path convention).
    """
    if field.startswith("metadata."):
        return reg.metadata.get(field[len("metadata.") :])
    return getattr(reg, field, None)


def _apply_sort_limit_offset(
    regs: list[Registration],
    sort: list[SortSpec] | None,
    limit: int | None,
    offset: int | None,
) -> list[Registration]:
    """Apply Python-side sort, offset, and limit to ``regs`` in that order."""
    if sort:
        for spec in reversed(sort):
            regs.sort(
                key=lambda r, f=spec.field: (_sort_key(r, f) is None, _sort_key(r, f)),
                reverse=(spec.order == SortOrder.DESC),
            )
    if offset is not None and offset > 0:
        regs = regs[offset:]
    if limit is not None:
        regs = regs[:limit]
    return regs


class InMemoryBackend:
    """In-memory implementation of RegistryBackend.

    Simple dict-based storage suitable for:
    - Testing without database dependencies
    - Single-instance deployments
    - Development environments

    Not suitable for:
    - Multi-instance deployments (no persistence)
    - Production with persistence requirements

    Thread-safety is provided via asyncio.Lock.

    Example:
        ```python
        backend = InMemoryBackend()
        await backend.initialize()

        reg = await backend.register("my-bot", {"llm": {...}})
        print(f"Created at: {reg.created_at}")

        config = await backend.get_config("my-bot")
        print(f"Config: {config}")

        # List all bots
        for reg in await backend.list_active():
            print(f"Bot: {reg.bot_id}")

        # Cleanup
        await backend.close()
        ```
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._registrations: dict[str, Registration] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the backend (no-op for in-memory)."""
        self._initialized = True

    async def close(self) -> None:
        """Close the backend (clears all data)."""
        async with self._lock:
            self._registrations.clear()
            self._initialized = False

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Registration:
        """Register or update a bot.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary
            status: Registration status (default: active)
            metadata: Cross-cutting context (tenant_id, audit, feature
                flags). Stored on the registration and filterable via
                ``filter_metadata`` on list/count.

        Returns:
            Registration object with metadata
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            meta = dict(metadata or {})

            if bot_id in self._registrations:
                # Update existing - preserve created_at
                old = self._registrations[bot_id]
                reg = Registration(
                    bot_id=bot_id,
                    config=config,
                    status=status,
                    metadata=meta,
                    created_at=old.created_at,
                    updated_at=now,
                    last_accessed_at=now,
                )
            else:
                # Create new
                reg = Registration(
                    bot_id=bot_id,
                    config=config,
                    status=status,
                    metadata=meta,
                    created_at=now,
                    updated_at=now,
                    last_accessed_at=now,
                )

            self._registrations[bot_id] = reg
            return reg

    async def get(self, bot_id: str) -> Registration | None:
        """Get registration and update access time.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration if found, None otherwise
        """
        async with self._lock:
            reg = self._registrations.get(bot_id)
            if reg:
                # Update access time
                self._registrations[bot_id] = Registration(
                    bot_id=reg.bot_id,
                    config=reg.config,
                    status=reg.status,
                    metadata=reg.metadata,
                    created_at=reg.created_at,
                    updated_at=reg.updated_at,
                    last_accessed_at=datetime.now(timezone.utc),
                )
                return self._registrations[bot_id]
            return None

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config.

        Updates ``last_accessed_at`` via :meth:`get` (registers an
        access). For non-touching inspection reads, use
        :meth:`peek_config`.

        Note:
            The returned dict is the same object stored internally.
            Callers that mutate it will mutate stored state. Copy
            before mutation if isolation is required.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        reg = await self.get(bot_id)
        return reg.config if reg else None

    async def peek_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config WITHOUT updating last_accessed_at.

        Reads directly from internal storage; does not invoke ``get``.

        Note:
            As with :meth:`get_config`, the returned dict is aliased
            to internal storage. Callers that mutate it will mutate
            stored state. Copy before mutation if isolation is
            required.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        async with self._lock:
            reg = self._registrations.get(bot_id)
            return reg.config if reg else None

    async def exists(self, bot_id: str) -> bool:
        """Check if active registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if registration exists and is active
        """
        async with self._lock:
            reg = self._registrations.get(bot_id)
            return reg is not None and reg.status == "active"

    async def unregister(self, bot_id: str) -> bool:
        """Hard delete registration.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if bot_id in self._registrations:
                del self._registrations[bot_id]
                return True
            return False

    async def deactivate(self, bot_id: str) -> bool:
        """Soft delete (set inactive).

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found
        """
        async with self._lock:
            if bot_id in self._registrations:
                reg = self._registrations[bot_id]
                self._registrations[bot_id] = Registration(
                    bot_id=reg.bot_id,
                    config=reg.config,
                    status="inactive",
                    metadata=reg.metadata,
                    created_at=reg.created_at,
                    updated_at=datetime.now(timezone.utc),
                    last_accessed_at=reg.last_accessed_at,
                )
                return True
            return False

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
            status: Optional equality filter on the ``status`` field.
                ``None`` returns all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.
            sort: Optional sort spec (applied Python-side).
            limit: Optional row limit.
            offset: Optional row offset for pagination.

        Returns:
            List of matching Registration objects.
        """
        async with self._lock:
            regs = [
                reg
                for reg in self._registrations.values()
                if (status is None or reg.status == status)
                and _matches_metadata(reg, filter_metadata)
            ]
        return _apply_sort_limit_offset(regs, sort, limit, offset)

    async def list_active(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Registration]:
        """List active registrations.

        Convenience for :meth:`list_all` with ``status="active"``.
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
        """List inactive registrations.

        Convenience for :meth:`list_all` with ``status="inactive"``.
        """
        return await self.list_all(
            status="inactive",
            filter_metadata=filter_metadata,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def list_ids(self) -> list[str]:
        """List active bot IDs.

        Returns:
            List of active bot IDs
        """
        async with self._lock:
            return [
                reg.bot_id
                for reg in self._registrations.values()
                if reg.status == "active"
            ]

    async def count_all(
        self,
        *,
        status: str | None = None,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count registrations matching the supplied filters.

        Args:
            status: Optional equality filter on the ``status`` field.
                ``None`` counts all statuses.
            filter_metadata: Optional equality filter over the
                ``metadata`` channel.

        Returns:
            Number of matching registrations.
        """
        async with self._lock:
            return sum(
                1
                for reg in self._registrations.values()
                if (status is None or reg.status == status)
                and _matches_metadata(reg, filter_metadata)
            )

    async def count(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count active registrations.

        Convenience for :meth:`count_all` with ``status="active"``.
        """
        return await self.count_all(
            status="active", filter_metadata=filter_metadata
        )

    async def count_inactive(
        self,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count inactive registrations.

        Convenience for :meth:`count_all` with ``status="inactive"``.
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

        The ``config`` parameter is accepted for protocol compatibility
        but ignored — the in-memory backend has no I/O batching.
        """
        del config  # No batching needed for the in-memory backend.
        regs = await self.list_all(status=status, filter_metadata=filter_metadata)
        for reg in regs:
            yield reg

    async def clear(self) -> None:
        """Clear all registrations."""
        async with self._lock:
            self._registrations.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"InMemoryBackend(count={len(self._registrations)})"
