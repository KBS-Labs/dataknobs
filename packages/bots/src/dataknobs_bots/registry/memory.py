"""In-memory implementation of RegistryBackend."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from .models import Registration


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
    ) -> Registration:
        """Register or update a bot.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary
            status: Registration status (default: active)

        Returns:
            Registration object with metadata
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            if bot_id in self._registrations:
                # Update existing - preserve created_at
                old = self._registrations[bot_id]
                reg = Registration(
                    bot_id=bot_id,
                    config=config,
                    status=status,
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
                    created_at=reg.created_at,
                    updated_at=reg.updated_at,
                    last_accessed_at=datetime.now(timezone.utc),
                )
                return self._registrations[bot_id]
            return None

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get just the config.

        Args:
            bot_id: Bot identifier

        Returns:
            Config dict if found, None otherwise
        """
        reg = await self.get(bot_id)
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
                    created_at=reg.created_at,
                    updated_at=datetime.now(timezone.utc),
                    last_accessed_at=reg.last_accessed_at,
                )
                return True
            return False

    async def list_active(self) -> list[Registration]:
        """List active registrations.

        Returns:
            List of active Registration objects
        """
        async with self._lock:
            return [
                reg for reg in self._registrations.values() if reg.status == "active"
            ]

    async def list_all(self) -> list[Registration]:
        """List all registrations.

        Returns:
            List of all Registration objects
        """
        async with self._lock:
            return list(self._registrations.values())

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

    async def count(self) -> int:
        """Count active registrations.

        Returns:
            Number of active registrations
        """
        async with self._lock:
            return sum(
                1 for reg in self._registrations.values() if reg.status == "active"
            )

    async def clear(self) -> None:
        """Clear all registrations."""
        async with self._lock:
            self._registrations.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"InMemoryBackend(count={len(self._registrations)})"
