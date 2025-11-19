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


class PluginRegistry(Generic[T]):
    """Registry for plugins with factory support and defaults.

    A specialized registry pattern for managing plugins (adapters, handlers,
    providers, etc.) that supports:
    - Class or factory function registration
    - Lazy instantiation with configuration
    - Default fallback when plugin not found
    - Instance caching
    - Type validation

    This pattern is useful when you need to:
    - Register different implementations of an interface
    - Create instances on-demand with configuration
    - Provide graceful fallbacks for unregistered keys

    Args:
        name: Registry name
        default_factory: Default factory to use when key not found

    Example:
        ```python
        from dataknobs_common.registry import PluginRegistry

        # Define base class
        class Handler:
            def __init__(self, name: str, config: dict):
                self.name = name
                self.config = config

        class DefaultHandler(Handler):
            pass

        class CustomHandler(Handler):
            pass

        # Create registry with default
        registry = PluginRegistry[Handler]("handlers", default_factory=DefaultHandler)

        # Register plugins
        registry.register("custom", CustomHandler)

        # Get instances
        handler = registry.get("custom", config={"timeout": 30})
        default = registry.get("unknown", config={})  # Uses default
        ```

    With async factories:
        ```python
        async def create_async_handler(name, config):
            handler = AsyncHandler(name, config)
            await handler.initialize()
            return handler

        registry.register("async", create_async_handler)
        handler = await registry.get_async("async", config={"url": "..."})
        ```
    """

    def __init__(
        self,
        name: str,
        default_factory: type[T] | Callable[..., T] | None = None,
        validate_type: type | None = None,
    ):
        """Initialize plugin registry.

        Args:
            name: Registry name for identification
            default_factory: Default class or factory to use when key not found
            validate_type: Optional base type to validate registrations against
        """
        self._name = name
        self._factories: Dict[str, type[T] | Callable[..., T]] = {}
        self._instances: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._default_factory = default_factory
        self._validate_type = validate_type

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    def register(
        self,
        key: str,
        factory: type[T] | Callable[..., T],
        override: bool = False,
    ) -> None:
        """Register a plugin class or factory.

        Args:
            key: Unique identifier for the plugin
            factory: Plugin class or factory function that creates instances
            override: If True, allow overriding existing registration

        Raises:
            OperationError: If key already registered and override=False
            TypeError: If factory doesn't match validate_type

        Example:
            ```python
            # Register a class
            registry.register("handler1", MyHandler)

            # Register a factory function
            registry.register("handler2", lambda name, config: create_handler(name, config))
            ```
        """
        with self._lock:
            # Check for existing registration
            if not override and key in self._factories:
                raise OperationError(
                    f"Plugin '{key}' already registered in {self._name}. "
                    f"Use override=True to replace.",
                    context={"key": key, "registry": self._name},
                )

            # Validate type if specified
            if self._validate_type and isinstance(factory, type):
                if not issubclass(factory, self._validate_type):
                    raise TypeError(
                        f"Factory class must be a subclass of {self._validate_type.__name__}, "
                        f"got {factory.__name__}"
                    )
            elif not callable(factory):
                raise TypeError(
                    f"Factory must be a class or callable, got {type(factory).__name__}"
                )

            # Register
            self._factories[key] = factory

            # Clear cached instance if overriding
            if key in self._instances:
                del self._instances[key]

    def unregister(self, key: str) -> None:
        """Unregister a plugin.

        Args:
            key: Key to unregister

        Raises:
            NotFoundError: If key not registered
        """
        with self._lock:
            if key not in self._factories:
                raise NotFoundError(
                    f"Plugin not found: {key}",
                    context={"key": key, "registry": self._name},
                )

            del self._factories[key]

            # Clear cached instance
            if key in self._instances:
                del self._instances[key]

    def is_registered(self, key: str) -> bool:
        """Check if a plugin is registered.

        Args:
            key: Key to check

        Returns:
            True if registered
        """
        with self._lock:
            return key in self._factories

    def get(
        self,
        key: str,
        config: Dict[str, Any] | None = None,
        use_cache: bool = True,
        use_default: bool = True,
    ) -> T:
        """Get a plugin instance.

        Creates instance if not cached, using the registered factory.

        Args:
            key: Plugin identifier
            config: Configuration dictionary passed to factory
            use_cache: Return cached instance if available
            use_default: Use default factory if key not registered

        Returns:
            Plugin instance

        Raises:
            NotFoundError: If key not registered and use_default=False

        Example:
            ```python
            handler = registry.get("custom", config={"timeout": 30})
            ```
        """
        with self._lock:
            # Check cache
            if use_cache and key in self._instances:
                return self._instances[key]

            # Get factory
            if key in self._factories:
                factory = self._factories[key]
            elif use_default and self._default_factory:
                factory = self._default_factory
            else:
                raise NotFoundError(
                    f"Plugin '{key}' not registered and no default available",
                    context={
                        "key": key,
                        "registry": self._name,
                        "available": list(self._factories.keys()),
                    },
                )

            # Create instance
            try:
                if isinstance(factory, type):
                    instance = factory(key, config or {})
                else:
                    instance = factory(key, config or {})

                # Validate instance type if specified
                if self._validate_type and not isinstance(instance, self._validate_type):
                    raise TypeError(
                        f"Factory must return a {self._validate_type.__name__} instance, "
                        f"got {type(instance).__name__}"
                    )

            except Exception as e:
                raise OperationError(
                    f"Failed to create plugin '{key}': {e}",
                    context={"key": key, "registry": self._name},
                ) from e

            # Cache instance
            if use_cache:
                self._instances[key] = instance

            return instance

    async def get_async(
        self,
        key: str,
        config: Dict[str, Any] | None = None,
        use_cache: bool = True,
        use_default: bool = True,
    ) -> T:
        """Get a plugin instance, supporting async factories.

        Like get() but awaits the factory if it's a coroutine function.

        Args:
            key: Plugin identifier
            config: Configuration dictionary
            use_cache: Return cached instance if available
            use_default: Use default factory if key not registered

        Returns:
            Plugin instance

        Example:
            ```python
            handler = await registry.get_async("async-handler", config={"url": "..."})
            ```
        """
        with self._lock:
            # Check cache
            if use_cache and key in self._instances:
                return self._instances[key]

            # Get factory
            if key in self._factories:
                factory = self._factories[key]
            elif use_default and self._default_factory:
                factory = self._default_factory
            else:
                raise NotFoundError(
                    f"Plugin '{key}' not registered and no default available",
                    context={
                        "key": key,
                        "registry": self._name,
                        "available": list(self._factories.keys()),
                    },
                )

        # Create instance (outside lock for async)
        try:
            if isinstance(factory, type):
                instance = factory(key, config or {})
            else:
                result = factory(key, config or {})
                # Await if coroutine
                if asyncio.iscoroutine(result):
                    instance = await result
                else:
                    instance = result

            # Validate instance type
            if self._validate_type and not isinstance(instance, self._validate_type):
                raise TypeError(
                    f"Factory must return a {self._validate_type.__name__} instance, "
                    f"got {type(instance).__name__}"
                )

        except Exception as e:
            raise OperationError(
                f"Failed to create plugin '{key}': {e}",
                context={"key": key, "registry": self._name},
            ) from e

        # Cache instance
        with self._lock:
            if use_cache:
                self._instances[key] = instance

        return instance

    def list_keys(self) -> List[str]:
        """List all registered plugin keys.

        Returns:
            List of registered keys
        """
        with self._lock:
            return list(self._factories.keys())

    def clear_cache(self, key: str | None = None) -> None:
        """Clear cached instances.

        Args:
            key: Specific key to clear, or None for all
        """
        with self._lock:
            if key:
                if key in self._instances:
                    del self._instances[key]
            else:
                self._instances.clear()

    def get_factory(self, key: str) -> type[T] | Callable[..., T] | None:
        """Get the registered factory for a key.

        Args:
            key: Plugin identifier

        Returns:
            Factory class or function, or None if not registered
        """
        with self._lock:
            return self._factories.get(key)

    @property
    def cached_instances(self) -> Dict[str, T]:
        """Get the dictionary of cached instances.

        Returns:
            Dictionary mapping keys to cached instances

        Note:
            This returns the internal cache dictionary. Modifications
            will affect the cache directly.
        """
        return self._instances

    def set_default_factory(self, factory: type[T] | Callable[..., T]) -> None:
        """Set the default factory.

        Args:
            factory: New default factory

        Raises:
            TypeError: If factory doesn't match validate_type
        """
        if self._validate_type and isinstance(factory, type):
            if not issubclass(factory, self._validate_type):
                raise TypeError(
                    f"Default factory must be a subclass of {self._validate_type.__name__}"
                )

        self._default_factory = factory

    def bulk_register(
        self,
        factories: Dict[str, type[T] | Callable[..., T]],
        override: bool = False,
    ) -> None:
        """Register multiple plugins at once.

        Args:
            factories: Dictionary mapping keys to factories
            override: Allow overriding existing registrations

        Example:
            ```python
            registry.bulk_register({
                "handler1": Handler1,
                "handler2": Handler2,
            })
            ```
        """
        for key, factory in factories.items():
            self.register(key, factory, override=override)

    def copy(self) -> Dict[str, type[T] | Callable[..., T]]:
        """Get a copy of all registered factories.

        Returns:
            Dictionary of key to factory mappings
        """
        with self._lock:
            return dict(self._factories)

    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self._factories)

    def __contains__(self, key: str) -> bool:
        """Check if plugin is registered using 'in' operator."""
        return self.is_registered(key)

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"PluginRegistry("
            f"name='{self._name}', "
            f"plugins={len(self._factories)}, "
            f"cached={len(self._instances)}"
            f")"
        )


__all__ = [
    "Registry",
    "CachedRegistry",
    "AsyncRegistry",
    "PluginRegistry",
]
