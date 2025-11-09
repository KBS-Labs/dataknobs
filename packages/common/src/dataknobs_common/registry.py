"""Generic registry pattern for managing named items.

This module provides reusable registry implementations that packages can extend
to manage collections of named items (tools, bots, resources, etc.).

The registry patterns support:
- Thread-safe item management
- Optional caching with TTL
- Optional metrics collection
- Generic typing for type safety
- Both sync and async variants

Example:
    ```python
    from dataknobs_common.registry import Registry

    # Create a simple registry
    class ToolRegistry(Registry[Tool]):
        def __init__(self):
            super().__init__("tools")

        def register_tool(self, tool: Tool) -> None:
            self.register(tool.name, tool, metadata={"type": "tool"})

    registry = ToolRegistry()
    registry.register_tool(my_tool)
    tool = registry.get("my_tool")
    ```

With Caching:
    ```python
    from dataknobs_common.registry import CachedRegistry

    class BotRegistry(CachedRegistry[Bot]):
        def __init__(self):
            super().__init__("bots", cache_ttl=300)

        def get_or_create_bot(self, client_id: str) -> Bot:
            return self.get_cached(
                client_id,
                factory=lambda: self._create_bot(client_id)
            )
    ```
"""

import asyncio
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    TypeVar,
)

from dataknobs_common.exceptions import NotFoundError, OperationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Base registry for managing named items with optional metrics.

    This is a thread-safe registry that manages a collection of items by
    unique keys. It provides core operations for registration, lookup,
    and enumeration.

    The registry is generic, so you can specify the type of items it
    manages for better type safety.

    Attributes:
        name: Name of the registry (for logging/debugging)

    Args:
        name: Name for this registry instance
        enable_metrics: Whether to track registration metrics

    Example:
        ```python
        registry = Registry[str]("my_registry")
        registry.register("key1", "value1")
        registry.get("key1")
        # 'value1'
        registry.count()
        # 1
        ```
    """

    def __init__(self, name: str, enable_metrics: bool = False):
        """Initialize the registry.

        Args:
            name: Registry name for identification
            enable_metrics: Enable metrics tracking
        """
        self._name = name
        self._items: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._metrics: Dict[str, Dict[str, Any]] | None = {} if enable_metrics else None

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    def register(
        self,
        key: str,
        item: T,
        metadata: Dict[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        """Register an item by key.

        Args:
            key: Unique identifier for the item
            item: Item to register
            metadata: Optional metadata about the item
            allow_overwrite: Whether to allow overwriting existing items

        Raises:
            OperationError: If item already exists and allow_overwrite is False

        Example:
            ```python
            registry.register("tool1", my_tool, metadata={"version": "1.0"})
            ```
        """
        with self._lock:
            if not allow_overwrite and key in self._items:
                raise OperationError(
                    f"Item '{key}' already registered in {self._name}",
                    context={"key": key, "registry": self._name},
                )

            self._items[key] = item

            if self._metrics is not None:
                self._metrics[key] = {
                    "registered_at": time.time(),
                    "metadata": metadata or {},
                }

    def unregister(self, key: str) -> T:
        """Unregister and return an item by key.

        Args:
            key: Key of item to unregister

        Returns:
            The unregistered item

        Raises:
            NotFoundError: If item not found

        Example:
            ```python
            item = registry.unregister("tool1")
            ```
        """
        with self._lock:
            if key not in self._items:
                raise NotFoundError(
                    f"Item not found: {key}",
                    context={"key": key, "registry": self._name},
                )

            item = self._items.pop(key)

            if self._metrics is not None and key in self._metrics:
                del self._metrics[key]

            return item

    def get(self, key: str) -> T:
        """Get an item by key.

        Args:
            key: Key of item to retrieve

        Returns:
            The registered item

        Raises:
            NotFoundError: If item not found

        Example:
            ```python
            item = registry.get("tool1")
            ```
        """
        with self._lock:
            if key not in self._items:
                raise NotFoundError(
                    f"Item not found: {key}",
                    context={"key": key, "registry": self._name, "available_keys": list(self._items.keys())},
                )
            return self._items[key]

    def get_optional(self, key: str) -> T | None:
        """Get an item by key, returning None if not found.

        Args:
            key: Key of item to retrieve

        Returns:
            The registered item or None

        Example:
            ```python
            item = registry.get_optional("tool1")
            if item is None:
                print("Not found")
            ```
        """
        with self._lock:
            return self._items.get(key)

    def has(self, key: str) -> bool:
        """Check if item exists.

        Args:
            key: Key to check

        Returns:
            True if item exists

        Example:
            ```python
            if registry.has("tool1"):
                print("Found")
            ```
        """
        with self._lock:
            return key in self._items

    def list_keys(self) -> List[str]:
        """List all registered keys.

        Returns:
            List of registered keys

        Example:
            ```python
            keys = registry.list_keys()
            print(keys)
            # ['tool1', 'tool2']
            ```
        """
        with self._lock:
            return list(self._items.keys())

    def list_items(self) -> List[T]:
        """List all registered items.

        Returns:
            List of registered items

        Example:
            ```python
            items = registry.list_items()
            for item in items:
                print(item)
            ```
        """
        with self._lock:
            return list(self._items.values())

    def items(self) -> List[tuple[str, T]]:
        """Get all key-item pairs.

        Returns:
            List of (key, item) tuples

        Example:
            ```python
            for key, item in registry.items():
                print(f"{key}: {item}")
            ```
        """
        with self._lock:
            return list(self._items.items())

    def count(self) -> int:
        """Get count of registered items.

        Returns:
            Number of items in registry

        Example:
            ```python
            count = registry.count()
            print(f"Registry has {count} items")
            ```
        """
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        """Clear all items from registry.

        Example:
            ```python
            registry.clear()
            registry.count()
            # 0
            ```
        """
        with self._lock:
            self._items.clear()
            if self._metrics is not None:
                self._metrics.clear()

    def get_metrics(self, key: str | None = None) -> Dict[str, Any]:
        """Get registration metrics.

        Args:
            key: Optional specific key to get metrics for

        Returns:
            Metrics dictionary

        Example:
            ```python
            metrics = registry.get_metrics()
            print(metrics)
            # {'tool1': {'registered_at': 1699456789.0, 'metadata': {}}}
            ```
        """
        with self._lock:
            if self._metrics is None:
                return {}

            if key:
                return self._metrics.get(key, {})

            return dict(self._metrics)

    def __len__(self) -> int:
        """Get number of registered items using len()."""
        return self.count()

    def __contains__(self, key: str) -> bool:
        """Check if item exists using 'in' operator."""
        return self.has(key)

    def __iter__(self):
        """Iterate over registered items."""
        return iter(self.list_items())


class CachedRegistry(Registry[T]):
    """Registry with time-based caching support.

    Extends the base registry with caching capabilities. Items can be
    retrieved from cache with automatic expiration and refresh based on TTL.
    Implements LRU eviction when cache size exceeds limits.

    Args:
        name: Registry name
        cache_ttl: Cache time-to-live in seconds (default: 300)
        max_cache_size: Maximum number of cached items (default: 1000)

    Example:
        ```python
        registry = CachedRegistry[Bot]("bots", cache_ttl=300)
        bot = registry.get_cached(
            "client1",
            factory=lambda: create_bot("client1")
        )
        ```
    """

    def __init__(
        self,
        name: str,
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
    ):
        """Initialize cached registry.

        Args:
            name: Registry name
            cache_ttl: Time-to-live for cached items in seconds
            max_cache_size: Maximum cache size before eviction
        """
        super().__init__(name, enable_metrics=True)
        self._cache: Dict[str, tuple[T, float]] = {}
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cached(
        self,
        key: str,
        factory: Callable[[], T],
        force_refresh: bool = False,
    ) -> T:
        """Get item from cache with automatic refresh.

        If item exists in cache and is not expired, returns cached version.
        Otherwise, calls factory to create new item and caches it.

        Args:
            key: Cache key
            factory: Callable that creates the item if not cached
            force_refresh: Force refresh even if cached

        Returns:
            Cached or newly created item

        Example:
            ```python
            def create_bot():
                return Bot("my-bot")
            bot = registry.get_cached("bot1", create_bot)
            ```
        """
        with self._lock:
            # Check cache
            if not force_refresh and key in self._cache:
                item, cached_at = self._cache[key]
                if time.time() - cached_at < self._cache_ttl:
                    self._cache_hits += 1
                    return item

            # Cache miss - create new item
            self._cache_misses += 1
            item = factory()
            self._cache[key] = (item, time.time())

            # Evict if cache too large
            if len(self._cache) > self._max_cache_size:
                self._evict_oldest()

            return item

    def invalidate_cache(self, key: str | None = None) -> None:
        """Invalidate cache for a key or all keys.

        Args:
            key: Specific key to invalidate, or None to invalidate all

        Example:
            ```python
            registry.invalidate_cache("bot1")  # Invalidate one
            registry.invalidate_cache()  # Invalidate all
            ```
        """
        with self._lock:
            if key:
                if key in self._cache:
                    del self._cache[key]
            else:
                self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics

        Example:
            ```python
            stats = registry.get_cache_stats()
            print(f"Hit rate: {stats['hit_rate']:.2%}")
            ```
        """
        with self._lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self._max_cache_size,
                "ttl_seconds": self._cache_ttl,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "total_requests": total,
                "hit_rate": hit_rate,
            }

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries (LRU).

        Removes oldest 10% of cache entries when max size is exceeded.
        """
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
        num_to_remove = max(1, len(sorted_items) // 10)

        for key, _ in sorted_items[:num_to_remove]:
            del self._cache[key]


class AsyncRegistry(Generic[T]):
    """Async-safe registry for managing named items.

    Similar to Registry but uses asyncio locks for async-safe operations.
    Use this when working in async contexts.

    Args:
        name: Registry name
        enable_metrics: Enable metrics tracking

    Example:
        >>> registry = AsyncRegistry[Tool]("tools")
        >>> await registry.register("tool1", my_tool)
        >>> tool = await registry.get("tool1")
    """

    def __init__(self, name: str, enable_metrics: bool = False):
        """Initialize async registry.

        Args:
            name: Registry name
            enable_metrics: Enable metrics tracking
        """
        self._name = name
        self._items: Dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._metrics: Dict[str, Dict[str, Any]] | None = {} if enable_metrics else None

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    async def register(
        self,
        key: str,
        item: T,
        metadata: Dict[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        """Register an item by key.

        Args:
            key: Unique identifier
            item: Item to register
            metadata: Optional metadata
            allow_overwrite: Allow overwriting existing items

        Raises:
            OperationError: If item exists and allow_overwrite is False
        """
        async with self._lock:
            if not allow_overwrite and key in self._items:
                raise OperationError(
                    f"Item '{key}' already registered in {self._name}",
                    context={"key": key, "registry": self._name},
                )

            self._items[key] = item

            if self._metrics is not None:
                self._metrics[key] = {
                    "registered_at": time.time(),
                    "metadata": metadata or {},
                }

    async def unregister(self, key: str) -> T:
        """Unregister and return an item.

        Args:
            key: Key to unregister

        Returns:
            The unregistered item

        Raises:
            NotFoundError: If item not found
        """
        async with self._lock:
            if key not in self._items:
                raise NotFoundError(
                    f"Item not found: {key}",
                    context={"key": key, "registry": self._name},
                )

            item = self._items.pop(key)

            if self._metrics is not None and key in self._metrics:
                del self._metrics[key]

            return item

    async def get(self, key: str) -> T:
        """Get an item by key.

        Args:
            key: Key to retrieve

        Returns:
            The registered item

        Raises:
            NotFoundError: If item not found
        """
        async with self._lock:
            if key not in self._items:
                raise NotFoundError(
                    f"Item not found: {key}",
                    context={"key": key, "registry": self._name, "available_keys": list(self._items.keys())},
                )
            return self._items[key]

    async def get_optional(self, key: str) -> T | None:
        """Get an item, returning None if not found.

        Args:
            key: Key to retrieve

        Returns:
            The item or None
        """
        async with self._lock:
            return self._items.get(key)

    async def has(self, key: str) -> bool:
        """Check if item exists.

        Args:
            key: Key to check

        Returns:
            True if exists
        """
        async with self._lock:
            return key in self._items

    async def list_keys(self) -> List[str]:
        """List all registered keys.

        Returns:
            List of keys
        """
        async with self._lock:
            return list(self._items.keys())

    async def list_items(self) -> List[T]:
        """List all registered items.

        Returns:
            List of items
        """
        async with self._lock:
            return list(self._items.values())

    async def items(self) -> List[tuple[str, T]]:
        """Get all key-item pairs.

        Returns:
            List of (key, item) tuples
        """
        async with self._lock:
            return list(self._items.items())

    async def count(self) -> int:
        """Get count of registered items.

        Returns:
            Number of items
        """
        async with self._lock:
            return len(self._items)

    async def clear(self) -> None:
        """Clear all items."""
        async with self._lock:
            self._items.clear()
            if self._metrics is not None:
                self._metrics.clear()

    async def get_metrics(self, key: str | None = None) -> Dict[str, Any]:
        """Get registration metrics.

        Args:
            key: Optional specific key

        Returns:
            Metrics dictionary
        """
        async with self._lock:
            if self._metrics is None:
                return {}

            if key:
                return self._metrics.get(key, {})

            return dict(self._metrics)

    def __len__(self) -> int:
        """Get number of registered items using len()."""
        # Note: This is synchronous but safe since it just reads the dict
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        """Check if item exists using 'in' operator."""
        # Note: This is synchronous but safe since it just reads the dict
        return key in self._items

    def __iter__(self):
        """Iterate over registered items."""
        # Note: Returns iterator over current snapshot
        return iter(list(self._items.values()))


__all__ = [
    "Registry",
    "CachedRegistry",
    "AsyncRegistry",
]
