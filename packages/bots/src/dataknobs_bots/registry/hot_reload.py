"""Hot-reload manager for dynamic configuration updates.

This module provides a coordinator that manages hot-reloading of registered
configurations when they change, using either event-driven or polling-based
change detection.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from dataknobs_common.events import EventBus, EventType

if TYPE_CHECKING:
    from .backend import RegistryBackend
    from .caching import CachingRegistryManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ReloadMode(Enum):
    """Mode for detecting configuration changes."""

    EVENT_DRIVEN = "event_driven"
    """Use EventBus for change notifications (recommended for production)."""

    POLLING = "polling"
    """Poll backend periodically for changes (for backends without events)."""

    HYBRID = "hybrid"
    """Use events when available, fall back to polling."""


class HotReloadManager(Generic[T]):
    """Coordinates hot-reloading of cached instances.

    This manager ties together:
    - A CachingRegistryManager for instance caching
    - An EventBus for change notifications
    - A RegistryPoller for backends without native events

    It ensures cached instances are automatically refreshed when their
    configurations change, with support for:
    - Event-driven invalidation (Postgres LISTEN/NOTIFY, Redis pub/sub)
    - Polling-based detection (for memory, file, S3 backends)
    - Hybrid mode (events preferred, polling as fallback)
    - Manual reload triggers

    Args:
        caching_manager: The caching manager to coordinate
        event_bus: Optional event bus for event-driven mode
        backend: Optional backend for polling mode
        mode: Reload detection mode (default: event_driven if event_bus provided)
        poll_interval: Seconds between polls in polling mode
        auto_start: Start automatically on initialize (default: True)

    Example:
        ```python
        from dataknobs_bots.registry import (
            ConfigCachingManager,
            HotReloadManager,
            InMemoryBackend,
        )
        from dataknobs_common.events import create_event_bus

        # Create components
        backend = InMemoryBackend()
        event_bus = create_event_bus({"backend": "memory"})
        caching_manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
        )

        # Create hot-reload manager
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            event_bus=event_bus,
            backend=backend,
            mode=ReloadMode.HYBRID,
        )

        # Initialize everything
        await backend.initialize()
        await event_bus.connect()
        await caching_manager.initialize()
        await hot_reload.initialize()

        # Get config (will be auto-refreshed on changes)
        config = await caching_manager.get_or_create("my-config")

        # Manual reload trigger
        await hot_reload.reload("my-config")

        # Cleanup
        await hot_reload.close()
        ```
    """

    def __init__(
        self,
        caching_manager: CachingRegistryManager[T],
        event_bus: EventBus | None = None,
        backend: RegistryBackend | None = None,
        mode: ReloadMode | None = None,
        poll_interval: float = 5.0,
        auto_start: bool = True,
    ) -> None:
        """Initialize the hot-reload manager.

        Args:
            caching_manager: Caching manager to coordinate
            event_bus: Event bus for event-driven mode
            backend: Backend for polling mode
            mode: Reload detection mode
            poll_interval: Poll interval in seconds
            auto_start: Auto-start on initialize
        """
        self._caching_manager = caching_manager
        self._event_bus = event_bus
        self._backend = backend or caching_manager.backend
        self._poll_interval = poll_interval
        self._auto_start = auto_start

        # Determine mode
        if mode is not None:
            self._mode = mode
        elif event_bus is not None:
            self._mode = ReloadMode.EVENT_DRIVEN
        else:
            self._mode = ReloadMode.POLLING

        # Components
        self._poller: Any | None = None  # RegistryPoller when in polling/hybrid mode
        self._initialized = False
        self._started = False

        # Reload callbacks
        self._reload_callbacks: list[Callable[[str], Any]] = []

        # Statistics
        self._reload_count = 0
        self._last_reload_time: float | None = None

    @property
    def mode(self) -> ReloadMode:
        """Get the current reload mode."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """Check if hot-reload is active."""
        return self._started

    @property
    def reload_count(self) -> int:
        """Get total number of reloads triggered."""
        return self._reload_count

    def add_reload_callback(self, callback: Callable[[str], Any]) -> None:
        """Add a callback to be invoked after reloads.

        The callback receives the instance_id that was reloaded.

        Args:
            callback: Function to call after reload
        """
        self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: Callable[[str], Any]) -> None:
        """Remove a previously added callback.

        Args:
            callback: The callback to remove
        """
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)

    async def initialize(self) -> None:
        """Initialize the hot-reload manager.

        Sets up polling if needed and subscribes to events.
        """
        if self._initialized:
            return

        # Set up poller for polling or hybrid mode
        if self._mode in (ReloadMode.POLLING, ReloadMode.HYBRID):
            from .polling import RegistryPoller

            self._poller = RegistryPoller(
                backend=self._backend,
                event_bus=self._event_bus or self._create_null_event_bus(),
                poll_interval=self._poll_interval,
            )
            # Add callback to handle poll-detected changes
            self._poller.add_change_callback(self._on_change_detected)

        self._initialized = True
        logger.info("HotReloadManager initialized (mode=%s)", self._mode.value)

        # Auto-start if configured
        if self._auto_start:
            await self.start()

    async def close(self) -> None:
        """Close the hot-reload manager.

        Stops polling and cleans up resources.
        """
        await self.stop()
        self._initialized = False
        logger.info("HotReloadManager closed")

    async def start(self) -> None:
        """Start hot-reload monitoring.

        Begins polling (if in polling/hybrid mode) and activates
        event handling.
        """
        if self._started:
            return

        if self._poller and self._mode in (ReloadMode.POLLING, ReloadMode.HYBRID):
            await self._poller.start()

        self._started = True
        logger.info("Hot-reload monitoring started")

    async def stop(self) -> None:
        """Stop hot-reload monitoring.

        Stops polling and deactivates event handling.
        """
        if not self._started:
            return

        if self._poller:
            await self._poller.stop()

        self._started = False
        logger.info("Hot-reload monitoring stopped")

    async def reload(self, instance_id: str) -> T:
        """Manually trigger a reload for an instance.

        This invalidates the cached instance and forces a fresh load
        from the backend.

        Args:
            instance_id: The instance to reload

        Returns:
            The reloaded instance

        Raises:
            KeyError: If the instance doesn't exist in the backend
        """
        logger.debug("Manual reload triggered for %s", instance_id)

        # Invalidate cache
        await self._caching_manager.invalidate(instance_id)

        # Reload (force refresh)
        instance = await self._caching_manager.get_or_create(
            instance_id, force_refresh=True
        )

        # Update stats and invoke callbacks
        await self._on_reload_complete(instance_id)

        return instance

    async def reload_all(self) -> int:
        """Reload all cached instances.

        Returns:
            Number of instances reloaded
        """
        logger.info("Reloading all cached instances")

        # Get list of cached IDs
        stats = self._caching_manager.get_cache_stats()
        cache_size = stats["size"]

        # Invalidate all
        await self._caching_manager.invalidate_all()

        logger.info("Invalidated %d cached instances", cache_size)
        return cache_size

    async def _on_change_detected(
        self, instance_id: str, event_type: EventType
    ) -> None:
        """Handle detected changes from poller or events.

        Args:
            instance_id: The changed instance
            event_type: Type of change
        """
        if event_type == EventType.CREATED:
            # New instance - no action needed, will be loaded on demand
            logger.debug("New registration detected: %s", instance_id)
            return

        if event_type in (EventType.UPDATED, EventType.DELETED):
            # Invalidate cache for updated/deleted instances
            if self._caching_manager.is_cached(instance_id):
                await self._caching_manager.invalidate(instance_id)
                logger.debug(
                    "Cache invalidated for %s (event=%s)",
                    instance_id,
                    event_type.value,
                )

        if event_type == EventType.UPDATED:
            # Optionally pre-warm cache for updated instances
            # For now, let it load on demand
            pass

    async def _on_reload_complete(self, instance_id: str) -> None:
        """Handle post-reload actions.

        Args:
            instance_id: The reloaded instance
        """
        import time

        self._reload_count += 1
        self._last_reload_time = time.time()

        # Invoke callbacks
        for callback in self._reload_callbacks:
            try:
                result = callback(instance_id)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in reload callback for %s",
                    instance_id,
                )

    def _create_null_event_bus(self) -> EventBus:
        """Create a null event bus that does nothing.

        Used when in polling-only mode without an event bus.
        """
        from dataknobs_common.events import InMemoryEventBus

        return InMemoryEventBus()

    def get_stats(self) -> dict[str, Any]:
        """Get hot-reload statistics.

        Returns:
            Dict with stats: mode, running, reload_count, last_reload, etc.
        """
        stats: dict[str, Any] = {
            "mode": self._mode.value,
            "running": self._started,
            "reload_count": self._reload_count,
            "last_reload_time": self._last_reload_time,
        }

        if self._poller:
            stats["poll_interval"] = self._poller.poll_interval
            stats["poller_running"] = self._poller.is_running

        return stats
