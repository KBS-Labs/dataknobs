"""Caching registry manager base class with event-driven invalidation."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from dataknobs_common.events import Event, EventBus, EventType, Subscription

from .backend import RegistryBackend

logger = logging.getLogger(__name__)

# Type variable for the cached instance type
T = TypeVar("T")


class CachingRegistryManager(ABC, Generic[T]):
    """Base class for caching registry managers with event-driven invalidation.

    This class provides the common caching infrastructure for managing
    instances backed by a RegistryBackend. It handles:

    - TTL-based cache expiration
    - Maximum cache size with LRU-style eviction
    - Event-driven cache invalidation via EventBus
    - Thread-safe cache access

    Subclasses implement instance creation and destruction:
    - `_create_instance()`: Creates the cached object from config
    - `_destroy_instance()`: Cleans up when evicting from cache

    The EventBus integration enables distributed cache invalidation:
    when a registration is updated or deleted on one instance,
    all instances receive the event and invalidate their caches.

    Args:
        backend: Storage backend for configurations
        event_bus: Event bus for cache invalidation events (optional)
        event_topic: Topic for registry events (default: "registry:instances")
        cache_ttl: Cache time-to-live in seconds (default: 300)
        max_cache_size: Maximum cached instances (default: 1000)

    Example:
        ```python
        from dataknobs_bots.registry import CachingRegistryManager, InMemoryBackend
        from dataknobs_common.events import create_event_bus

        class MyBotManager(CachingRegistryManager[MyBot]):
            async def _create_instance(self, bot_id: str, config: dict) -> MyBot:
                return await MyBot.from_config(config)

            async def _destroy_instance(self, instance: MyBot) -> None:
                await instance.close()

        # Create with event bus for distributed invalidation
        bus = create_event_bus({"backend": "postgres", "connection_string": "..."})
        await bus.connect()

        manager = MyBotManager(
            backend=InMemoryBackend(),
            event_bus=bus,
        )
        await manager.initialize()

        # Get or create instance
        bot = await manager.get_or_create("my-bot")
        ```
    """

    def __init__(
        self,
        backend: RegistryBackend,
        event_bus: EventBus | None = None,
        event_topic: str = "registry:instances",
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
    ) -> None:
        """Initialize the caching registry manager.

        Args:
            backend: Storage backend for configurations
            event_bus: Event bus for cache invalidation (optional)
            event_topic: Topic for registry events
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum cached instances
        """
        self._backend = backend
        self._event_bus = event_bus
        self._event_topic = event_topic
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size

        # Instance cache: id -> (instance, cached_timestamp)
        self._cache: dict[str, tuple[T, float]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._subscription: Subscription | None = None

    @property
    def backend(self) -> RegistryBackend:
        """Get the storage backend."""
        return self._backend

    @property
    def event_bus(self) -> EventBus | None:
        """Get the event bus, if configured."""
        return self._event_bus

    @property
    def cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return self._cache_ttl

    @property
    def max_cache_size(self) -> int:
        """Get maximum cache size."""
        return self._max_cache_size

    @property
    def cache_size(self) -> int:
        """Get current number of cached instances."""
        return len(self._cache)

    async def initialize(self) -> None:
        """Initialize the manager, backend, and event subscription.

        Must be called before using the manager.
        """
        if self._initialized:
            return

        await self._backend.initialize()

        # Subscribe to cache invalidation events
        if self._event_bus:
            # Subscribe to exact topic (events published to the base topic)
            self._subscription = await self._event_bus.subscribe(
                self._event_topic,
                self._handle_event,
            )
            logger.debug(
                "Subscribed to event topic %s for cache invalidation",
                self._event_topic,
            )

        self._initialized = True
        logger.info("CachingRegistryManager initialized")

    async def close(self) -> None:
        """Close the manager, clearing cache and releasing resources."""
        # Cancel event subscription
        if self._subscription:
            await self._subscription.cancel()
            self._subscription = None

        # Destroy all cached instances
        async with self._lock:
            for instance_id, (instance, _) in list(self._cache.items()):
                try:
                    await self._destroy_instance(instance)
                except Exception:
                    logger.exception(
                        "Error destroying instance %s during close",
                        instance_id,
                    )
            self._cache.clear()

        await self._backend.close()
        self._initialized = False
        logger.info("CachingRegistryManager closed")

    async def get_or_create(
        self,
        instance_id: str,
        force_refresh: bool = False,
    ) -> T:
        """Get an instance from cache or create a new one.

        If a cached instance exists and hasn't expired, it's returned.
        Otherwise, loads config from backend and creates a new instance.

        Args:
            instance_id: Unique identifier for the instance
            force_refresh: If True, bypass cache and create fresh instance

        Returns:
            The cached or newly created instance

        Raises:
            KeyError: If no configuration exists for the instance_id
        """
        async with self._lock:
            # Check cache
            if not force_refresh and instance_id in self._cache:
                instance, cached_at = self._cache[instance_id]
                if time.time() - cached_at < self._cache_ttl:
                    logger.debug("Returning cached instance: %s", instance_id)
                    return instance
                else:
                    # Expired - destroy and remove
                    try:
                        await self._destroy_instance(instance)
                    except Exception:
                        logger.exception(
                            "Error destroying expired instance %s",
                            instance_id,
                        )
                    del self._cache[instance_id]

            # Load configuration from backend
            config = await self._backend.get_config(instance_id)
            if config is None:
                raise KeyError(f"No configuration found for: {instance_id}")

            # Create instance
            instance = await self._create_instance(instance_id, config)

            # Cache the instance
            self._cache[instance_id] = (instance, time.time())
            logger.debug("Created and cached instance: %s", instance_id)

            # Evict oldest if cache is full
            await self._evict_if_full()

            return instance

    async def invalidate(self, instance_id: str) -> bool:
        """Invalidate a cached instance.

        Removes the instance from cache and calls `_destroy_instance()`.
        Does not affect the backend registration.

        Args:
            instance_id: Instance identifier to invalidate

        Returns:
            True if an instance was invalidated, False if not cached
        """
        async with self._lock:
            if instance_id not in self._cache:
                return False

            instance, _ = self._cache.pop(instance_id)
            try:
                await self._destroy_instance(instance)
            except Exception:
                logger.exception(
                    "Error destroying instance %s during invalidation",
                    instance_id,
                )

            logger.debug("Invalidated cached instance: %s", instance_id)
            return True

    async def invalidate_all(self) -> int:
        """Invalidate all cached instances.

        Returns:
            Number of instances invalidated
        """
        async with self._lock:
            count = len(self._cache)
            for instance_id, (instance, _) in list(self._cache.items()):
                try:
                    await self._destroy_instance(instance)
                except Exception:
                    logger.exception(
                        "Error destroying instance %s during invalidate_all",
                        instance_id,
                    )
            self._cache.clear()

            logger.debug("Invalidated %d cached instances", count)
            return count

    async def publish_invalidation(
        self,
        instance_id: str,
        event_type: EventType = EventType.UPDATED,
    ) -> None:
        """Publish a cache invalidation event.

        Used to notify other instances to invalidate their caches
        when a registration is updated or deleted.

        Args:
            instance_id: Instance identifier to invalidate
            event_type: Type of change (UPDATED, DELETED, etc.)
        """
        if not self._event_bus:
            return

        event = Event(
            type=event_type,
            topic=f"{self._event_topic}:{instance_id}",
            payload={"instance_id": instance_id},
            source="caching_manager",
        )

        await self._event_bus.publish(self._event_topic, event)
        logger.debug(
            "Published invalidation event for %s (type=%s)",
            instance_id,
            event_type.value,
        )

    async def _handle_event(self, event: Event) -> None:
        """Handle incoming cache invalidation events.

        Args:
            event: The event to handle
        """
        if event.type not in (EventType.UPDATED, EventType.DELETED, EventType.DEACTIVATED):
            return

        instance_id = event.payload.get("instance_id")
        if not instance_id:
            return

        logger.debug(
            "Received invalidation event for %s (type=%s)",
            instance_id,
            event.type.value,
        )
        await self.invalidate(instance_id)

    async def _evict_if_full(self) -> None:
        """Evict oldest entries if cache exceeds max size.

        Must be called while holding the lock.
        """
        while len(self._cache) > self._max_cache_size:
            # Find oldest entry
            oldest_id = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1],
            )

            instance, _ = self._cache.pop(oldest_id)
            try:
                await self._destroy_instance(instance)
            except Exception:
                logger.exception(
                    "Error destroying evicted instance %s",
                    oldest_id,
                )

            logger.debug("Evicted oldest instance: %s", oldest_id)

    def is_cached(self, instance_id: str) -> bool:
        """Check if an instance is currently cached.

        Args:
            instance_id: Instance identifier

        Returns:
            True if cached (regardless of TTL)
        """
        return instance_id in self._cache

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats: size, max_size, ttl, hit_rate, etc.
        """
        now = time.time()
        expired_count = sum(
            1
            for _, (_, cached_at) in self._cache.items()
            if now - cached_at >= self._cache_ttl
        )

        return {
            "size": len(self._cache),
            "max_size": self._max_cache_size,
            "ttl_seconds": self._cache_ttl,
            "expired_count": expired_count,
            "valid_count": len(self._cache) - expired_count,
        }

    # --- Abstract methods for subclasses ---

    @abstractmethod
    async def _create_instance(self, instance_id: str, config: dict[str, Any]) -> T:
        """Create a new instance from configuration.

        Called when an instance is not in cache or cache has expired.
        Subclasses implement this to create the appropriate instance type.

        Args:
            instance_id: Unique identifier for the instance
            config: Configuration dict from the backend

        Returns:
            The created instance
        """
        raise NotImplementedError

    @abstractmethod
    async def _destroy_instance(self, instance: T) -> None:
        """Destroy an instance when evicting from cache.

        Called when an instance is removed from cache (eviction, invalidation,
        or close). Subclasses implement this to clean up resources.

        Args:
            instance: The instance to destroy
        """
        raise NotImplementedError
