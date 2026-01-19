"""Registry polling for change detection.

This module provides polling-based change detection for registry backends
that don't support push notifications (like memory or file backends).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from dataknobs_common.events import Event, EventBus, EventType

if TYPE_CHECKING:
    from .backend import RegistryBackend

logger = logging.getLogger(__name__)


class RegistryPoller:
    """Polls a registry backend for changes and emits events.

    This component is useful for backends that don't support native
    change notifications (like InMemoryBackend, FileBackend, S3, etc.).
    For PostgreSQL, use LISTEN/NOTIFY via PostgresEventBus instead.

    The poller tracks:
    - New registrations (CREATED events)
    - Updated registrations (UPDATED events)
    - Deleted registrations (DELETED events)

    Changes are detected by comparing the current state against a snapshot.
    For updates, the poller compares updated_at timestamps.

    Args:
        backend: The registry backend to poll
        event_bus: Event bus for publishing change events
        event_topic: Topic for registry events
        poll_interval: Seconds between polls (default: 5)
        track_content_changes: If True, detect config changes even without
            timestamp updates (more expensive, default: False)

    Example:
        ```python
        from dataknobs_bots.registry import InMemoryBackend, RegistryPoller
        from dataknobs_common.events import create_event_bus

        backend = InMemoryBackend()
        event_bus = create_event_bus({"backend": "memory"})

        poller = RegistryPoller(
            backend=backend,
            event_bus=event_bus,
            poll_interval=5,
        )

        # Start polling in background
        await poller.start()

        # ... registrations change ...

        # Stop when done
        await poller.stop()
        ```
    """

    def __init__(
        self,
        backend: RegistryBackend,
        event_bus: EventBus,
        event_topic: str = "registry:changes",
        poll_interval: float = 5.0,
        track_content_changes: bool = False,
    ) -> None:
        """Initialize the registry poller.

        Args:
            backend: Registry backend to poll
            event_bus: Event bus for change notifications
            event_topic: Topic for registry change events
            poll_interval: Seconds between polls
            track_content_changes: Track config content changes (expensive)
        """
        self._backend = backend
        self._event_bus = event_bus
        self._event_topic = event_topic
        self._poll_interval = poll_interval
        self._track_content_changes = track_content_changes

        # Snapshot of known registrations: id -> (updated_at, config_hash)
        self._snapshot: dict[str, tuple[str | None, int | None]] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False

        # Callbacks for change events (in addition to event bus)
        self._change_callbacks: list[Callable[[str, EventType], Any]] = []

    @property
    def is_running(self) -> bool:
        """Check if the poller is currently running."""
        return self._running

    @property
    def poll_interval(self) -> float:
        """Get the poll interval in seconds."""
        return self._poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        """Set the poll interval in seconds."""
        if value <= 0:
            raise ValueError("Poll interval must be positive")
        self._poll_interval = value

    def add_change_callback(
        self, callback: Callable[[str, EventType], Any]
    ) -> None:
        """Add a callback to be invoked on changes.

        The callback receives (instance_id, event_type) arguments.
        This is in addition to events published to the event bus.

        Args:
            callback: Function to call on changes
        """
        self._change_callbacks.append(callback)

    def remove_change_callback(
        self, callback: Callable[[str, EventType], Any]
    ) -> None:
        """Remove a previously added callback.

        Args:
            callback: The callback to remove
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    async def start(self) -> None:
        """Start polling for changes.

        This starts a background task that periodically checks for changes.
        Call stop() to terminate the polling.
        """
        if self._running:
            logger.warning("Poller already running")
            return

        self._running = True
        # Take initial snapshot
        await self._update_snapshot()
        # Start polling task
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Started registry poller with %ss interval",
            self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop polling for changes.

        Cancels the background polling task.
        """
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped registry poller")

    async def poll_once(self) -> dict[str, EventType]:
        """Perform a single poll and return detected changes.

        Returns:
            Dict mapping instance_id to EventType for each detected change
        """
        changes = await self._detect_changes()

        # Publish events and invoke callbacks
        for instance_id, event_type in changes.items():
            await self._emit_change(instance_id, event_type)

        # Update snapshot after processing
        await self._update_snapshot()

        return changes

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                changes = await self._detect_changes()
                for instance_id, event_type in changes.items():
                    await self._emit_change(instance_id, event_type)

                # Update snapshot after processing
                await self._update_snapshot()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error during registry poll")
                # Continue polling despite errors

    async def _update_snapshot(self) -> None:
        """Update the internal snapshot of registry state."""
        registrations = await self._backend.list_all()
        self._snapshot = {}

        for reg in registrations:
            updated_at = (
                reg.updated_at.isoformat() if reg.updated_at else None
            )
            config_hash = (
                hash(str(sorted(reg.config.items())))
                if self._track_content_changes and reg.config
                else None
            )
            self._snapshot[reg.bot_id] = (updated_at, config_hash)

    async def _detect_changes(self) -> dict[str, EventType]:
        """Detect changes since last snapshot.

        Returns:
            Dict mapping instance_id to EventType for detected changes
        """
        changes: dict[str, EventType] = {}

        # Get current state
        registrations = await self._backend.list_all()
        current_ids = set()

        for reg in registrations:
            current_ids.add(reg.bot_id)
            updated_at = (
                reg.updated_at.isoformat() if reg.updated_at else None
            )
            config_hash = (
                hash(str(sorted(reg.config.items())))
                if self._track_content_changes and reg.config
                else None
            )

            if reg.bot_id not in self._snapshot:
                # New registration
                changes[reg.bot_id] = EventType.CREATED
                logger.debug("Detected new registration: %s", reg.bot_id)
            else:
                old_updated_at, old_config_hash = self._snapshot[reg.bot_id]

                # Check for updates
                timestamp_changed = (
                    updated_at is not None
                    and updated_at != old_updated_at
                )
                content_changed = (
                    self._track_content_changes
                    and config_hash != old_config_hash
                )

                if timestamp_changed or content_changed:
                    changes[reg.bot_id] = EventType.UPDATED
                    logger.debug("Detected update: %s", reg.bot_id)

        # Check for deletions
        for old_id in self._snapshot:
            if old_id not in current_ids:
                changes[old_id] = EventType.DELETED
                logger.debug("Detected deletion: %s", old_id)

        return changes

    async def _emit_change(self, instance_id: str, event_type: EventType) -> None:
        """Emit a change event.

        Args:
            instance_id: The changed instance ID
            event_type: Type of change
        """
        # Publish to event bus
        event = Event(
            type=event_type,
            topic=f"{self._event_topic}:{instance_id}",
            payload={"instance_id": instance_id},
            source="registry_poller",
        )
        await self._event_bus.publish(self._event_topic, event)

        # Invoke callbacks
        for callback in self._change_callbacks:
            try:
                result = callback(instance_id, event_type)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in change callback for %s",
                    instance_id,
                )
