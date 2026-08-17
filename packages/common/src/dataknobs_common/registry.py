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
import copy
import inspect
import logging
import threading
import time
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from dataknobs_common.exceptions import (
    DataknobsError,
    NotFoundError,
    OperationError,
)

logger = logging.getLogger(__name__)

#: Default text for :meth:`PluginRegistry.create` when the routing key was
#: absent and ``config_key_default`` supplied it. Generic on purpose: a
#: registry that can say what its own fallback costs passes
#: ``default_warning`` instead, and every registry with a consequential
#: default should.
DEFAULT_KEY_WARNING = (
    "No '%(config_key)s' key in this %(registry)s config; falling back to "
    "'%(key)s'. A config that names no %(config_key)s is indistinguishable "
    "from one that asks for the default, so this is reported rather than "
    "assumed."
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class BackendRegistry(Protocol, Generic[T_co]):
    """Common surface across :class:`Registry`-shape and
    :class:`PluginRegistry`-shape adopters.

    A registry-like object — addressable by string key, list-able,
    membership-testable, unregister-able. Both :class:`Registry` and
    :class:`PluginRegistry` structurally conform without inheritance.

    Consumers writing tooling that introspects "is this thing a
    registry?" should ``isinstance`` against this Protocol, not against
    the concrete classes (which cover different specialization axes —
    :class:`Registry` holds items, :class:`PluginRegistry` constructs
    them from factories — and have diverged method sets accordingly).

    The Protocol is deliberately minimal: only the four methods every
    registry-like object must offer. ``PluginRegistry``-specific methods
    (``create``, ``create_async``, ``get_factory``) and
    ``Registry``-specific methods (``get_metrics``, ``list_items``,
    ``items``, ``count``, ``clear``) are NOT in the Protocol —
    consumers needing those features should ``isinstance`` against the
    concrete class.
    """

    @property
    def name(self) -> str:
        """Registry name for identification / logging."""
        ...

    def has(self, key: str) -> bool:
        """Test membership."""
        ...

    def list_keys(self) -> List[str]:
        """Enumerate registered keys."""
        ...

    def unregister(self, key: str) -> Any:
        """Remove a registration. Return value varies by registry kind."""
        ...


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
                    context={
                        "key": key,
                        "registry": self._name,
                        "available_keys": list(self._items.keys()),
                    },
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

    def __iter__(self) -> Iterator[T]:
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
                    context={
                        "key": key,
                        "registry": self._name,
                        "available_keys": list(self._items.keys()),
                    },
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

    def __iter__(self) -> Iterator[T]:
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
    - Per-domain not-found error shape (``not_found_kind`` /
      ``not_found_exception``) so consolidating shims preserve their
      historical error text and exception class

    This pattern is useful when you need to:
    - Register different implementations of an interface
    - Create instances on-demand with configuration
    - Provide graceful fallbacks for unregistered keys

    Registry split convention:
        When using ``PluginRegistry`` for a Protocol parameterized by
        input shape (e.g. ``ResourceResolver[KeyT, ValueT]``,
        ``Discriminator[InputT, KindT]``), prefer N typed registries
        (one per concrete input shape) over one flat registry with
        ``validate_type=Any``. The typed ``validate_type=`` is
        load-bearing under consumer-extensibility: an out-of-tree
        backend that structurally conforms to the wrong Protocol shape
        would silently register and only fail at use-time without the
        constraint.

        Example: ``resolver_backends`` (for ``KeyT -> ValueT`` lookups)
        is separate from ``partition_resolver_backends`` (for
        ``record -> str | None`` lookups) — distinct input shapes get
        distinct typed registries.

        If a consumer later surfaces "actually we wanted one flat
        registry," the cost of being wrong is one line per entry (move
        entries between registries; deprecate the smaller one). The
        choice is reversible; the typed pin is not.

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
        *,
        canonicalize_keys: bool = False,
        config_key: str | None = None,
        config_key_default: str | None = None,
        strip_config_key: bool = False,
        on_first_access: Callable[["PluginRegistry[T]"], None] | None = None,
        not_found_kind: str | None = None,
        not_found_exception: type[Exception] = NotFoundError,
        default_warning: str | None = None,
    ):
        """Initialize plugin registry.

        Args:
            name: Registry name for identification
            default_factory: Default class or factory to use when key not found
            validate_type: Optional base type to validate registrations against
            canonicalize_keys: When True, all keys are lowercased
            config_key: Field name to extract lookup key from config dict in
                ``create()`` when ``key`` is ``None``
            config_key_default: Fallback value when ``config_key`` field is
                absent from config dict
            strip_config_key: When True and key is extracted from config,
                remove the key field from config before passing to the factory
            on_first_access: Callback invoked once before first public method
                access. Receives the registry instance. Supports re-entrant
                calls (e.g. callback calling ``register()``).
            not_found_kind: Opt-in kind label rendered into the
                ``create()`` / ``create_async()`` not-found error message
                when a domain shim wants per-kind text. When ``None``
                (default), the historical
                ``"Plugin '<key>' not registered"`` text is used.
                Setting this to e.g. ``"event bus backend"`` produces
                ``"Unknown event bus backend: <key>. Available backends:
                <sorted-keys>"``.
            not_found_exception: Exception class raised on
                ``create()`` / ``create_async()`` not-found.  Defaults to
                :class:`NotFoundError` (the more principled
                ``DataknobsError``-rooted shape for consumers catching
                programmatically). Domain shims preserving a historical
                ``ValueError`` contract can opt in by passing
                ``not_found_exception=ValueError``. Non-
                ``DataknobsError`` classes are called with the message
                only (no ``context=`` kwarg) since stdlib exceptions
                would crash on the unknown keyword.
            default_warning: Text logged at WARNING by ``create()`` /
                ``create_async()`` when the routing key was absent from
                config and ``config_key_default`` supplied it. Interpolated
                with ``%(config_key)s``, ``%(key)s`` and ``%(registry)s``.
                Defaults to :data:`DEFAULT_KEY_WARNING`. A registry whose
                default has consequences -- an in-process lock that
                coordinates nothing, an unpersisted store that loses
                everything on restart -- should say so here, because the
                generic sentence cannot.
        """
        self._name = name
        #: Every factory is invoked as ``factory(key, config)``, so the stored
        #: shape is the callable one. A class satisfies it: ``type[T]`` is
        #: assignable to ``Callable[..., T]``, and ``register`` still spells
        #: the union to document that both are accepted. Storing the union
        #: instead makes mypy resolve a call against ``type[T]``'s ``__init__``
        #: -- ``object.__init__`` for an unbound TypeVar -- so every call site
        #: reported "Too many arguments" and returned ``Any``.
        self._factories: Dict[str, Callable[..., T]] = {}
        self._instances: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._default_factory: Callable[..., T] | None = default_factory
        self._validate_type = validate_type
        self._canonicalize_keys = canonicalize_keys
        self._config_key = config_key
        self._config_key_default = config_key_default
        self._strip_config_key = strip_config_key
        self._initializer = on_first_access
        self._initialized = on_first_access is None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        #: Keys this registry knows about but cannot create, mapped to the
        #: reason. Kept apart from ``_factories`` so ``is_registered`` keeps
        #: meaning "creatable", while ``get_metadata`` can still answer the
        #: one question that is only ever asked while the answer is
        #: unavailable: what would I have to install?
        self._unavailable: Dict[str, str] = {}
        self._not_found_kind = not_found_kind
        self._not_found_exception = not_found_exception
        self._default_warning = default_warning or DEFAULT_KEY_WARNING

    def _check_validate_type(self, key: str, instance: Any) -> None:
        """Reject a factory result that is not the registered type.

        Raised as an ``OperationError`` rather than a ``TypeError`` because
        every caller sits inside a ``except Exception`` that wraps a factory
        failure into a bounded message — a factory builds a backend from
        deployment config, so its text can carry a connection URL. This
        message is authored here and names two class names, so it is bounded
        by construction and deserves to survive that wrap intact; the callers
        let an ``OperationError`` through untouched. Flattening it would leave
        a caller unable to tell a factory that *failed* from one that returned
        the wrong thing.

        Four call sites built this check inline, which is why correcting the
        wrap once would otherwise have had to be done four times.
        """
        if self._validate_type and not isinstance(instance, self._validate_type):
            raise OperationError(
                f"Factory for plugin '{key}' must return a "
                f"{self._validate_type.__name__} instance, "
                f"got {type(instance).__name__}",
                context={"key": key, "registry": self._name},
            )

    def _canon(self, key: str) -> str:
        """Canonicalize a key if configured."""
        return key.lower() if self._canonicalize_keys else key

    def _ensure_initialized(self) -> None:
        """Run on_first_access callback if configured and not yet run.

        Thread safety: uses double-checked locking with ``self._lock``.
        ``_initialized`` is set to ``True`` *before* calling the
        initializer so that re-entrant calls from within the callback
        (e.g. ``register()`` → ``_ensure_initialized()``) see the flag
        and return immediately instead of deadlocking.  Concurrent
        threads are safe because every public method that reads or
        writes ``_factories`` / ``_instances`` also acquires
        ``self._lock``, serialising access with the initializer.
        """
        if self._initialized:
            return
        with self._lock:
            # Double-checked locking. mypy narrows _initialized to False from
            # the unlocked check above and calls this one unreachable, which
            # holds only for a single thread -- another thread completing
            # initialisation between the two reads is the case this exists for.
            if self._initialized:
                return  # type: ignore[unreachable]
            # Snapshot so a partial-failure init can be rolled back atomically.
            # Without this, a populator that registers some keys and then
            # raises leaves those keys behind; because we reset _initialized
            # below, the next access re-runs the populator from the top and
            # hits "already registered", masking the real error.
            factories_snapshot = dict(self._factories)
            instances_snapshot = dict(self._instances)
            metadata_snapshot = dict(self._metadata)
            self._initialized = True
            try:
                self._initializer(self)  # type: ignore[misc]
            except Exception:
                # Restore pre-init state (preserving dict identity) so a retry
                # starts clean and surfaces the populator's real error.
                self._factories.clear()
                self._factories.update(factories_snapshot)
                self._instances.clear()
                self._instances.update(instances_snapshot)
                self._metadata.clear()
                self._metadata.update(metadata_snapshot)
                self._initialized = False
                raise

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    def register(
        self,
        key: str,
        factory: type[T] | Callable[..., T],
        override: bool = False,
        metadata: Dict[str, Any] | None = None,
        *,
        allow_overwrite: bool | None = None,
    ) -> None:
        """Register a plugin class or factory.

        Args:
            key: Unique identifier for the plugin
            factory: Plugin class or factory function that creates instances
            override: If True, allow overriding existing registration
            metadata: Optional metadata to attach to the registration
            allow_overwrite: Alias for ``override`` matching the
                :class:`Registry`-style spelling. When ``None`` (default),
                the ``override`` flag is used unmodified. When explicitly
                ``True`` or ``False``, this value wins (the more explicit
                opt-in / opt-out spelling). Use whichever name fits the
                surrounding code; positional registrations and
                ``override=`` keyword registrations behave identically to
                before this alias was added.

        Raises:
            OperationError: If key already registered and override=False
            TypeError: If factory doesn't match validate_type

        Example:
            ```python
            # Register a class
            registry.register("handler1", MyHandler)

            # Register a factory function (for use with get())
            # get() calls factories with (key, config) signature
            registry.register("handler2", lambda name, config: create_handler(name, config))

            # For use with create(), factories should accept (config, **kwargs)
            # or define a from_config(config, **kwargs) classmethod.
            # See create() docstring for details.
            ```
        """
        if allow_overwrite is not None:
            override = allow_overwrite

        self._ensure_initialized()
        key = self._canon(key)

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
            if metadata is not None:
                self._metadata[key] = metadata

            # A key that was declared unavailable is creatable again. The two
            # states are exclusive, so registering clears the mark rather than
            # leaving the order of the two calls to decide the outcome.
            self._unavailable.pop(key, None)

            # Clear cached instance if overriding
            if key in self._instances:
                del self._instances[key]

    def declare_unavailable(
        self,
        key: str,
        *,
        metadata: Dict[str, Any] | None = None,
        reason: str,
    ) -> None:
        """Record a plugin this registry knows of but cannot create.

        A plugin behind an optional dependency has three states, not two:
        creatable, absent because the dependency is missing, and absent
        because the name is a typo. A registry holding only ``_factories``
        conflates the last two, so the one question worth asking about an
        uninstalled plugin -- what would I install? -- had no answer,
        because the metadata carrying it went unregistered along with the
        factory.

        Declaring the key keeps its metadata reachable while leaving
        :meth:`is_registered` and :meth:`list_keys` meaning "creatable", so
        nothing that asks what it can build sees a plugin it cannot.

        Args:
            key: Plugin identifier.
            metadata: Metadata to attach, as :meth:`register` takes. This is
                what makes ``requires_install`` readable while the plugin is
                uninstalled.
            reason: Why it cannot be created here, in a sentence a caller
                can act on -- ``create()`` raises this instead of reporting
                an unknown key.

        Example:
            ```python
            try:
                from .postgres import SyncPostgresDatabase
            except ImportError:
                registry.declare_unavailable(
                    "postgres",
                    metadata={"requires_install": "pip install ...[postgres]"},
                    reason="psycopg2 is not installed",
                )
            ```
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            self._factories.pop(key, None)
            self._instances.pop(key, None)
            if metadata is not None:
                self._metadata[key] = metadata
            self._unavailable[key] = reason

    def unregister(self, key: str) -> None:
        """Unregister a plugin.

        Args:
            key: Key to unregister

        Raises:
            NotFoundError: If key not registered
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            if key not in self._factories:
                raise NotFoundError(
                    f"Plugin not found: {key}",
                    context={"key": key, "registry": self._name},
                )

            del self._factories[key]
            self._metadata.pop(key, None)

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
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            return key in self._factories

    def has(self, key: str) -> bool:
        """Alias for :meth:`is_registered`.

        Provided so :class:`PluginRegistry` structurally conforms to the
        :class:`BackendRegistry` Protocol, which mirrors the
        :meth:`Registry.has` spelling.
        """
        return self.is_registered(key)

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
        self._ensure_initialized()
        key = self._canon(key)

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
                instance = factory(key, config or {})

                self._check_validate_type(key, instance)

            except OperationError:
                raise
            except Exception as e:
                # Bounded message: a plugin factory builds a backend — a
                # database, an event bus, an LLM client — from deployment
                # config, so `e` here is a driver's or an SDK's text and can
                # carry the connection URL it was handed. The key and the
                # registry name are ours; __cause__ carries the rest.
                raise OperationError(
                    f"Failed to create plugin '{key}' ({type(e).__name__})",
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
        self._ensure_initialized()
        key = self._canon(key)

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
        # Annotated: the dispatch below reaches the factory through
        # isinstance/hasattr probes and an optional await, none of which
        # mypy can follow back to T. This is where the produced type is
        # promised, and _check_validate_type is what enforces it.
        instance: T
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

            self._check_validate_type(key, instance)

        except Exception as e:
            raise OperationError(
                f"Failed to create plugin '{key}' ({type(e).__name__})",
                context={"key": key, "registry": self._name},
            ) from e

        # Cache instance
        with self._lock:
            if use_cache:
                self._instances[key] = instance

        return instance

    def create(
        self,
        key: str | None = None,
        config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """Create a fresh instance without caching.

        Unlike ``get()``, this method:

        - Never returns or stores cached instances
        - Uses ``(config, **kwargs)`` factory signature instead of
          ``(key, config)``
        - Detects ``from_config`` classmethods on class factories

        Factory invocation:

        - If factory is a class with a ``from_config`` classmethod:
          ``factory.from_config(config, **kwargs)``
        - Otherwise: ``factory(config, **kwargs)``

        Key resolution:

        - If ``key`` is provided, use it directly.
        - If ``key`` is ``None`` and ``config_key`` was set at init,
          extract from ``config[config_key]`` (falling back to
          ``config_key_default``).
        - If neither, raise ``ValueError``.

        When ``strip_config_key`` is ``True`` and the key was extracted
        from config, a shallow copy of config is made with the key field
        removed before passing to the factory.

        Args:
            key: Plugin identifier. Optional when ``config_key`` is
                configured on the registry.
            config: Configuration dictionary passed to factory.  ``None``
                is treated as ``{}`` for both key resolution and factory
                invocation.
            **kwargs: Additional keyword arguments forwarded to factory.

        Returns:
            Fresh plugin instance.

        Raises:
            ValueError: If ``key`` is ``None`` and cannot be resolved.
            NotFoundError: If resolved key is not registered.
            OperationError: If factory raises an exception (including
                type validation failures from ``validate_type``).

        Note:
            Type validation errors from ``register()`` raise ``TypeError``
            directly (caught at registration time), while type validation
            errors from ``create()`` are wrapped in ``OperationError``
            (caught at creation time).  This asymmetry is intentional:
            registration errors are programming mistakes caught immediately,
            while creation errors may arise from dynamic factory behavior.
        """
        factory, key, config = self._resolve_factory(key, config)

        # Annotated: the dispatch below reaches the factory through
        # isinstance/hasattr probes and an optional await, none of which
        # mypy can follow back to T. This is where the produced type is
        # promised, and _check_validate_type is what enforces it.
        instance: T
        try:
            if isinstance(factory, type) and hasattr(factory, "from_config"):
                instance = factory.from_config(config or {}, **kwargs)
            else:
                instance = factory(config or {}, **kwargs)

            self._check_validate_type(key, instance)

        except (NotFoundError, OperationError):
            raise
        except Exception as e:
            raise OperationError(
                f"Failed to create plugin '{key}' ({type(e).__name__})",
                context={"key": key, "registry": self._name},
            ) from e

        return instance

    async def create_async(
        self,
        key: str | None = None,
        config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """Create a fresh instance, awaiting an asynchronous factory.

        The async counterpart to :meth:`create`. Identical key
        resolution and factory lookup (both delegate to the shared
        ``_resolve_factory`` prologue), but the factory result is
        awaited before the ``validate_type`` guard runs — so the guard
        checks the resolved instance, not a coroutine. Use this for
        plugins whose construction is asynchronous (an eager-connecting
        backend, an LLM-warmed component, a knowledge base that ingests
        on build).

        Factory invocation:

        - If the factory is a class with a ``from_config_async``
          classmethod: ``await factory.from_config_async(config, **kwargs)``
          (the canonical async entry point).
        - Else if it exposes ``from_config``:
          ``factory.from_config(config, **kwargs)``, awaited if the
          result is awaitable.
        - Otherwise: ``factory(config, **kwargs)``, awaited if the result
          is awaitable.

        A purely synchronous factory works unchanged — its non-awaitable
        result passes through. Like :meth:`create`, this never caches.

        Args:
            key: Plugin identifier. Optional when ``config_key`` is
                configured on the registry.
            config: Configuration dictionary passed to the factory.
                ``None`` is treated as ``{}``.
            **kwargs: Additional keyword arguments forwarded to the
                factory (e.g. injected collaborators threaded into a
                ``StructuredConfigConsumer`` consumer's ``_ainit``).

        Returns:
            Fresh plugin instance.

        Raises:
            ValueError: If ``key`` is ``None`` and cannot be resolved.
            NotFoundError: If the resolved key is not registered.
            OperationError: If the factory raises (including
                ``validate_type`` failures).
        """
        factory, key, config = self._resolve_factory(key, config)

        # Annotated: the dispatch below reaches the factory through
        # isinstance/hasattr probes and an optional await, none of which mypy
        # can follow back to T. This is where the produced type is promised,
        # and _check_validate_type is what enforces it.
        instance: T
        try:
            if isinstance(factory, type) and hasattr(factory, "from_config_async"):
                instance = await factory.from_config_async(config or {}, **kwargs)
            else:
                if isinstance(factory, type) and hasattr(factory, "from_config"):
                    result = factory.from_config(config or {}, **kwargs)
                else:
                    result = factory(config or {}, **kwargs)
                instance = await result if inspect.isawaitable(result) else result

            self._check_validate_type(key, instance)

        except (NotFoundError, OperationError):
            raise
        except Exception as e:
            raise OperationError(
                f"Failed to create plugin '{key}' ({type(e).__name__})",
                context={"key": key, "registry": self._name},
            ) from e

        return instance

    def _resolve_factory(
        self,
        key: str | None,
        config: Dict[str, Any] | None,
    ) -> tuple[Callable[..., T], str, Dict[str, Any] | None]:
        """Resolve ``(factory, canonical_key, config)`` for create paths.

        Shared prologue for :meth:`create` and :meth:`create_async` — the
        single source of truth for routing-key resolution and factory
        lookup, so the sync and async paths cannot drift. Handles:

        - Explicit ``key``, or extraction from ``config[config_key]``
          (falling back to ``config_key_default``).
        - Reporting a key that came from the default rather than from the
          config, at WARNING — see :meth:`_warn_key_defaulted`.
        - Optional stripping of the routing key from ``config`` when
          ``strip_config_key`` is set.
        - Key canonicalization and factory lookup.

        Returns the resolved factory, the canonical key (for error
        context), and the possibly key-stripped config. Raises
        ``ValueError`` (unresolvable key) or ``self._not_found_exception``
        (unknown key, or a key declared unavailable —
        :class:`NotFoundError` by default; opt-in via the
        ``not_found_exception`` ctor kwarg) directly — both callers
        invoke this *before* their factory-invocation ``try`` so these
        propagate unwrapped, matching the historical contract.
        """
        self._ensure_initialized()

        # Resolve key
        if key is None:
            if self._config_key is None:
                raise ValueError("key is required when config_key is not configured")
            if self._config_key in (config or {}):
                key = (config or {})[self._config_key]
            else:
                key = self._config_key_default
                # Reported before the lookup, so a config that names nothing
                # *and* resolves to nothing still says which of the two
                # happened. Guarded on the default having been used at all:
                # a registry with no default raises below instead.
                if key is not None:
                    self._warn_key_defaulted(key)
            if key is None:
                raise ValueError(
                    f"config must contain '{self._config_key}' (no default configured)"
                )

            # Strip the routing key from config before passing to factory
            if self._strip_config_key and config is not None:
                config = {k: v for k, v in config.items() if k != self._config_key}

        key = self._canon(key)

        with self._lock:
            if key not in self._factories:
                available = sorted(self._factories.keys())
                if key in self._unavailable:
                    # Known, but not creatable here. Reporting it as unknown
                    # sends the reader looking for a typo in a name that is
                    # spelled correctly.
                    message = (
                        f"{self._not_found_kind or 'Plugin'} '{key}' is not "
                        f"available here: {self._unavailable[key]}"
                    )
                elif self._not_found_kind is not None:
                    message = (
                        f"Unknown {self._not_found_kind}: {key}. "
                        f"Available backends: {', '.join(available)}"
                    )
                else:
                    message = f"Plugin '{key}' not registered"

                context = {
                    "key": key,
                    "registry": self._name,
                    "available": available,
                }
                exc_cls = self._not_found_exception
                if issubclass(exc_cls, DataknobsError):
                    raise exc_cls(message, context=context)
                raise exc_cls(message)
            return self._factories[key], key, config

    def _warn_key_defaulted(self, key: str) -> None:
        """Report that nothing in the config chose ``key``.

        An absent routing key and an explicit one naming the same value
        produce the same object, which is exactly what made the difference
        invisible: the only place the distinction still exists is here,
        between reading the config and resolving the name.
        """
        logger.warning(
            self._default_warning,
            {
                "config_key": self._config_key,
                "key": key,
                "registry": self._name,
            },
        )

    def list_keys(self) -> List[str]:
        """List all registered plugin keys.

        Every accepted spelling is reported, aliases included — this is the
        lookup surface. For one name per plugin, see
        :meth:`list_canonical_keys`.

        Returns:
            List of registered keys
        """
        self._ensure_initialized()

        with self._lock:
            return list(self._factories.keys())

    def list_known_keys(self) -> List[str]:
        """Every key this registry knows of, creatable or not.

        :meth:`list_keys` answers "what can I build?"; this answers "what
        does this registry know about?", which for a registry using
        :meth:`declare_unavailable` is the larger set.

        Returns:
            Sorted keys, including those declared unavailable.
        """
        self._ensure_initialized()

        with self._lock:
            return sorted(set(self._factories) | set(self._unavailable))

    def list_canonical_keys(self) -> List[str]:
        """Registered keys with aliases collapsed, one name per plugin.

        A registry lists every spelling it accepts, so three aliases for one
        plugin read as three plugins. That is right for the lookup
        :meth:`list_keys` serves and wrong for a list shown to someone
        choosing between plugins.

        Keys sharing a factory form one group, and the name reported is the
        one carrying metadata — the convention this package registers by,
        the canonical key taking the metadata and its aliases taking none. A
        group where no key has metadata reports its first-registered key, so
        a registration following a different convention still yields one
        name per plugin rather than an error.

        Two genuinely distinct plugins deliberately registered against the
        same factory are one group by this rule, and report one name while
        both stay creatable. Register distinct plugins against distinct
        factories to keep them distinct here.

        Returns:
            Sorted canonical keys, one per registered plugin.
        """
        self._ensure_initialized()

        with self._lock:
            # Grouped under one lock rather than by re-reading through
            # get_factory: a concurrent unregister between the listing and
            # the lookups would drop keys into a group keyed on None.
            groups: Dict[int, List[str]] = {}
            for key, factory in self._factories.items():
                # Keyed by identity rather than by the factory itself: a
                # registered callable need not be hashable, and the registry
                # holds a reference to every one of them for as long as this
                # dict lives, so an id cannot be reused underneath us.
                groups.setdefault(id(factory), []).append(key)

            names = []
            for keys in groups.values():
                described = [key for key in keys if self._metadata.get(key)]
                names.append(described[0] if described else keys[0])
            return sorted(names)

    def clear_cache(self, key: str | None = None) -> None:
        """Clear cached instances.

        Args:
            key: Specific key to clear, or None for all
        """
        self._ensure_initialized()

        with self._lock:
            if key:
                key = self._canon(key)
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
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            return self._factories.get(key)

    def get_metadata(self, key: str, *, follow_alias: bool = False) -> Dict[str, Any]:
        """Get metadata for a plugin, registered or declared unavailable.

        Args:
            key: Plugin identifier.
            follow_alias: When the key itself carries no metadata, answer
                with the metadata of a key sharing its factory. An alias is
                registered without metadata of its own, so asking about
                ``pg`` otherwise returns an empty dict while every other
                question about it answers for postgres. Opt-in, because the
                historical shape is "metadata stored against this key".

        Returns:
            A deep copy of the metadata, or an empty dict if none stored.
            Copied all the way down: a shallow copy hands back the live
            nested dicts, so a caller reading ``config_options`` and
            editing it changed what every later caller saw.
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            metadata = self._metadata.get(key)
            if not metadata and follow_alias:
                metadata = self._aliased_metadata(key)
            return copy.deepcopy(metadata) if metadata else {}

    def _aliased_metadata(self, key: str) -> Dict[str, Any] | None:
        """Metadata of a key sharing ``key``'s factory, if any carries some.

        Caller holds the lock.
        """
        factory = self._factories.get(key)
        if factory is None:
            return None
        for other, candidate in self._factories.items():
            if candidate is factory and self._metadata.get(other):
                return self._metadata[other]
        return None

    @property
    def cached_instances(self) -> Dict[str, T]:
        """Get the dictionary of cached instances.

        Returns:
            Dictionary mapping keys to cached instances

        Note:
            This returns the internal cache dictionary. Modifications
            will affect the cache directly.
        """
        self._ensure_initialized()
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
        self._ensure_initialized()

        with self._lock:
            return dict(self._factories)

    def __len__(self) -> int:
        """Get number of registered plugins."""
        self._ensure_initialized()
        return len(self._factories)

    def __contains__(self, key: str) -> bool:
        """Check if plugin is registered using 'in' operator."""
        return self.is_registered(key)  # delegates _ensure_initialized + _canon

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
    "AsyncRegistry",
    "BackendRegistry",
    "CachedRegistry",
    "PluginRegistry",
    "Registry",
]
