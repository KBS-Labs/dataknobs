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
import typing
from collections.abc import Awaitable, Iterator, Mapping
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from dataknobs_common.exceptions import (
    DataknobsError,
    NotFoundError,
    OperationError,
)

logger = logging.getLogger(__name__)

#: Text for :meth:`PluginRegistry.create` when the routing key was absent
#: and ``config_key_default`` supplied it, logged at **DEBUG**. Generic on
#: purpose, and quiet on purpose: it can only report that a guess happened,
#: which is worth recording and is not worth interrupting anyone for.
#:
#: A registry that can say what its own fallback *costs* passes
#: ``default_warning`` instead and is reported at WARNING. Every registry
#: with a consequential default should; a registry whose default is the
#: recommended answer should not, because a documented config omits that key
#: deliberately and a warning on it is a warning on correct usage.
DEFAULT_KEY_WARNING = (
    "No '%(config_key)s' key in this %(registry)s config; falling back to "
    "'%(key)s'. A config that names no %(config_key)s is indistinguishable "
    "from one that asks for the default, so this is reported rather than "
    "assumed."
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

PluginFactory: TypeAlias = type[T] | Callable[..., T] | Callable[..., Awaitable[T]]
"""What a :class:`PluginRegistry` stores under a key.

Three shapes, and the third is the one the declaration used to omit: a
class, a callable returning an instance, and a callable returning an
*awaitable* instance. The class has always accepted all three at runtime
and has documented registering the third since it grew ``get_async``; only
the annotation excluded it, so the exclusion reported as a type error on
the documented pattern's own tests.

Exported because :meth:`PluginRegistry.get_factory` returns one and
:meth:`PluginRegistry.copy` returns a mapping of them — it is public
surface whether or not it has a name, and a consumer annotating a factory
they are about to register needs the name.

**An awaitable factory is not usable from every entry point.** Only
``get_async`` and ``create_async`` can await one; the synchronous
``get`` and ``create`` refuse it, naming the async method to call
instead. That refusal is the reason this alias is safe to widen: without
it, admitting the third shape would have deleted the type error that was
the only thing stopping a caller from silently receiving a coroutine.
"""


class Unavailable(NamedTuple):
    """Why a known key cannot be created, and which key describes it.

    ``describes_key`` is the spelling whose metadata answers for this one.
    It is the key itself for a canonical declaration and the canonical key
    for an alias, which is what lets ``get_metadata(follow_alias=True)``
    work for a plugin that has no factory to group by.

    ``type_loader`` imports the plugin's class on demand, for the callers
    that want to *read* something off it rather than build it. It is not a
    factory and is deliberately not stored as one: ``is_registered`` has to
    keep meaning "creatable", and this plugin is not. Kept lazy because the
    usual reason a plugin is unavailable is that importing its module is
    expensive or unwise, and the introspecting caller is rarer than the
    ones who only wanted to know the plugin exists.
    """

    reason: str
    describes_key: str
    type_loader: Callable[[], Any] | None = None


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


def _protocol_members(base: type) -> frozenset[str] | None:
    """The members a Protocol declares, or ``None`` if *base* is not one.

    ``typing.get_protocol_members`` is the public spelling and exists from
    3.13; ``__protocol_attrs__`` is what CPython sets below that. One of the
    two is asserted by a test rather than assumed, because
    :meth:`PluginRegistry._check_factory_class` has no footing without it and
    would otherwise degrade silently into re-raising.
    """
    getter = getattr(typing, "get_protocol_members", None)
    if getter is not None:
        try:
            return frozenset(getter(base))
        except TypeError:
            # 3.13+ raises for anything that is not a Protocol, which is the
            # same answer `__protocol_attrs__` gives by being absent.
            return None
    members = getattr(base, "__protocol_attrs__", None)
    return None if members is None else frozenset(members)


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
        default_factory: PluginFactory[T] | None = None,
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
            validate_type: Optional base type to validate registrations
                against. A class, an ABC, or a ``@runtime_checkable``
                Protocol --- including one carrying properties, which
                ``issubclass`` cannot check and which is checked against
                the protocol's declared members instead (see
                :meth:`_check_factory_class`). A Protocol without the
                decorator supports neither check and is refused at
                registration, naming the decorator.
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
            default_warning: What this registry's fallback costs, logged at
                WARNING by ``create()`` / ``create_async()`` when the
                routing key was absent from config and
                ``config_key_default`` supplied it. Interpolated with
                ``%(config_key)s``, ``%(key)s`` and ``%(registry)s``.

                Pass this when the default has a consequence someone needs
                to act on -- an in-process lock that coordinates nothing, a
                bus whose events reach nobody, an unpersisted store that
                loses everything on restart -- because the generic sentence
                cannot say what went wrong.

                Leave it ``None`` when the default is simply the
                recommended answer. The fallback is then recorded at DEBUG
                using :data:`DEFAULT_KEY_WARNING`, so the provenance is
                still there without warning anyone about a config written
                the way the documentation writes it.
        """
        self._name = name
        #: Two arities, chosen by the entry point rather than by the
        #: factory: the ``get`` lane calls ``factory(key, config)`` and
        #: the ``create`` lane calls ``factory(config, **kwargs)``.
        #: :meth:`_invoke_factory` is where that choice is made, and is
        #: the only place either spelling appears.
        #:
        #: The stored shape is the full :data:`PluginFactory` union, which
        #: it previously could not be. Storing the union used to make mypy
        #: resolve each call against ``type[T]``'s ``__init__`` --
        #: ``object.__init__`` for an unbound TypeVar -- so *every call
        #: site* reported "Too many arguments" and returned ``Any``, and
        #: the storage was narrowed to ``Callable[..., T]`` to avoid it.
        #: There is now exactly one call site, inside ``_invoke_factory``,
        #: whose parameter is ``Callable[..., Any]``; the union is never
        #: resolved against a call, so the cascade has nothing to cascade
        #: through. Narrowing the storage again would restore the lie
        #: without restoring the reason for it.
        self._factories: Dict[str, PluginFactory[T]] = {}
        self._instances: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._default_factory: PluginFactory[T] | None = default_factory
        self._validate_type = validate_type
        self._canonicalize_keys = canonicalize_keys
        self._config_key = config_key
        self._config_key_default = config_key_default
        self._strip_config_key = strip_config_key
        self._initializer = on_first_access
        self._initialized = on_first_access is None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        #: Keys this registry knows about but cannot create, each mapped
        #: to its reason and to the key whose metadata describes it. Kept
        #: apart from ``_factories`` so ``is_registered`` keeps meaning
        #: "creatable", while ``get_metadata`` can still answer the one
        #: question that is only ever asked while the answer is unavailable:
        #: what would I have to install?
        self._unavailable: Dict[str, Unavailable] = {}
        self._not_found_kind = not_found_kind
        self._not_found_exception = not_found_exception
        #: ``None`` means this registry claims no consequence for its own
        #: default, so the fallback is recorded at DEBUG rather than WARNING.
        #: Not defaulted to :data:`DEFAULT_KEY_WARNING` here: doing so made
        #: every registry with a ``config_key_default`` warn, including six
        #: whose default is the recommended answer and whose documented
        #: configs therefore omit the key on purpose.
        self._default_warning = default_warning
        if default_warning is not None:
            # Interpolated lazily by `logging` against a dict, so a literal
            # `%` in consumer-supplied text is a format spec to the handler
            # and raises there -- inside logging, at the first fallback,
            # long after the registry was built and only on the branch
            # nobody exercises. Proved here instead, where the text is
            # authored: an authoring fault is fatal at authoring time.
            try:
                default_warning % {"config_key": "k", "key": "v", "registry": "r"}
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"default_warning for registry {name!r} is not a valid "
                    f"%-format template: {exc}. Placeholders are "
                    f"%(config_key)s, %(key)s and %(registry)s; a literal "
                    f"percent sign must be written %%."
                ) from exc

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

    def _check_factory_class(self, factory: type, *, what: str) -> None:
        """Reject a factory class that cannot produce the registered type.

        ``issubclass`` is the right check and is what runs wherever it can.
        It cannot run against a ``@runtime_checkable`` Protocol carrying a
        non-method member --- a property is not a method, and CPython
        refuses the whole call rather than the offending member --- so such
        a protocol is checked against its declared members instead.

        That fallback is the *same* check rather than a quieter one:
        ``issubclass`` against a method-only Protocol is itself a
        member-presence scan of the class, so a property-carrying protocol
        reaches exactly the verdict its method-only twin would have
        reached. Nor is it a relaxation, because the alternative is not a
        stricter check --- it is a ``TypeError`` raised on the conforming
        class as readily as on a wrong one, naming neither the registry,
        the protocol, nor the class.

        A Protocol that was never decorated ``@runtime_checkable`` is the
        other way the check cannot run, and there the fault is in the
        registry rather than in the factory: ``isinstance`` refuses such a
        base at ``create()`` too, so the registry is broken in both
        directions. It is reported here, where the missing decorator can be
        named, rather than absorbed into a member scan that would let
        registration pass and leave ``create()`` to fail for a cause
        nobody stated.

        Anything else ``issubclass`` refuses --- a ``validate_type`` that
        is not a class at all --- keeps landing on ``issubclass``'s own
        ``TypeError``. That is the disposition
        :func:`~dataknobs_common.imports.resolve_class` documents for the
        identical condition: a constraint the calling code got wrong.

        Two sites built this check inline, and the second named neither the
        class it rejected nor the registry that rejected it, which is why
        the fix is one method rather than two edits. ``bulk_register``
        delegates to ``register`` and so needs nothing.

        Args:
            factory: The class being registered.
            what: The subject of the message --- ``"Factory class"`` or
                ``"Default factory"``.

        Raises:
            TypeError: If *factory* cannot produce ``validate_type``, or if
                ``validate_type`` cannot be checked against at all.
        """
        base = self._validate_type
        if base is None:
            return

        try:
            conforms = issubclass(factory, base)
        except TypeError:
            members = _protocol_members(base)
            if members is None:
                raise
            if not getattr(base, "_is_runtime_protocol", False):
                raise TypeError(
                    f"Registry {self._name!r} cannot check {what.lower()} "
                    f"{factory.__name__}: validate_type={base.__name__} is a "
                    f"Protocol that was never decorated @runtime_checkable, "
                    f"so no runtime check can run against it and create() "
                    f"fails the same way. Decorate {base.__name__} with "
                    f"typing.runtime_checkable."
                ) from None
            # CPython's own subclass hook reads a class-dict entry of None
            # as "declared, not implemented", so the fallback reads it the
            # same way rather than inventing a second rule.
            missing = sorted(m for m in members if getattr(factory, m, None) is None)
            if missing:
                raise TypeError(
                    f"{what} {factory.__name__} does not implement "
                    f"{base.__name__}: missing {', '.join(missing)} "
                    f"(registry {self._name!r})"
                ) from None
            return

        if not conforms:
            raise TypeError(
                f"{what} must be a subclass of {base.__name__}, "
                f"got {factory.__name__} (registry {self._name!r})"
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
            #
            # Every piece of state a populator can reach belongs here, not
            # every dict: `_unavailable` was added to the class without being
            # added to this list, so an abandoned run's marks outlived the
            # metadata that explained them, and `_default_factory` -- which
            # `set_default_factory` lets a populator replace -- was missed for
            # the same reason one scope wider. Naming the invariant as *state*
            # is what stops the next attribute from being missed too.
            factories_snapshot = dict(self._factories)
            instances_snapshot = dict(self._instances)
            metadata_snapshot = dict(self._metadata)
            unavailable_snapshot = dict(self._unavailable)
            default_factory_snapshot = self._default_factory
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
                self._unavailable.clear()
                self._unavailable.update(unavailable_snapshot)
                self._default_factory = default_factory_snapshot
                self._initialized = False
                raise

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    def register(
        self,
        key: str,
        factory: PluginFactory[T],
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
            TypeError: If *factory* is not a class or callable, if it is a
                class that cannot produce ``validate_type``, or if
                ``validate_type`` cannot be checked against at all. A
                callable factory's result is checked instead by
                ``get()`` / ``create()``, which raise ``OperationError``.

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
                self._check_factory_class(factory, what="Factory class")
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
        aliases: Sequence[str] = (),
        type_loader: Callable[[], Any] | None = None,
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
            aliases: Other accepted spellings of the same plugin, withdrawn
                with it and answering with its metadata. Declared here
                rather than by the caller repeating the call, because
                :meth:`get_metadata`'s ``follow_alias`` groups by shared
                factory and an unavailable plugin has none -- so a caller
                that could not say "these are the same plugin" had to copy
                the metadata under every spelling to keep
                ``requires_install`` reachable.
            type_loader: Imports and returns the plugin's class, for
                :meth:`load_declared_type`. Supply it when the class is
                importable even though the plugin is not creatable -- a
                backend whose module guards its optional driver behind a
                flag, for instance -- so a caller reading a typed schema
                off the class is not forced to keep its own second copy of
                the key-to-class mapping. Omit it when importing the module
                is exactly what fails.

        Example:
            ```python
            registry.declare_unavailable(
                "postgres",
                metadata={"requires_install": "pip install ...[postgres]"},
                reason="psycopg2 is not installed",
                aliases=("pg", "postgresql"),
            )
            ```
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            if metadata is not None:
                self._metadata[key] = metadata
            for spelling in (key, *(self._canon(alias) for alias in aliases)):
                self._factories.pop(spelling, None)
                self._instances.pop(spelling, None)
                self._unavailable[spelling] = Unavailable(reason, key, type_loader)

    def unregister(self, key: str) -> None:
        """Unregister a plugin, or drop a key declared unavailable.

        Both states are things this registry knows about, so both are
        things it can be told to forget. Removing only the creatable half
        left a key that :meth:`list_known_keys` reported and nothing could
        withdraw: :meth:`register` was the only way to clear a mark, which
        made "forget this" reachable only by first supplying a factory.

        Args:
            key: Key to unregister. Unregistering a canonical key also
                drops the unavailable marks that named it as their
                describing key, because those marks answer with metadata
                that is being removed here -- leaving them would strand a
                key that :meth:`list_known_keys` still reports, whose
                ``requires_install`` has just become ``{}``. An alias
                unregistered on its own drops only itself.

        Raises:
            NotFoundError: If the key is neither registered nor declared
                unavailable -- that is, if the registry has never heard of
                it at all.
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            if key not in self._factories and key not in self._unavailable:
                raise NotFoundError(
                    f"Plugin not found: {key}",
                    context={"key": key, "registry": self._name},
                )

            self._factories.pop(key, None)
            self._unavailable.pop(key, None)
            self._metadata.pop(key, None)

            # An unavailable alias carries no metadata of its own -- it
            # answers through `describes_key`, which is the key being
            # dropped here. Withdrawing them together is what keeps
            # "a mark cannot outlive the metadata that explains it" true
            # in the one direction `declare_unavailable(aliases=...)` made
            # reachable.
            for spelling, declared in list(self._unavailable.items()):
                if declared.describes_key == key:
                    del self._unavailable[spelling]
                    self._instances.pop(spelling, None)

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
            # Annotated for the same reason the other three entry points
            # are: the result reaches T through `_invoke_factory`, whose
            # parameter is deliberately `Callable[..., Any]`, and
            # `_check_validate_type` is what enforces the promise.
            instance: T
            try:
                instance = self._refuse_awaitable(
                    key,
                    self._invoke_factory(
                        factory,
                        key,
                        config,
                        {},
                        positional_key=True,
                        allow_async_classmethod=False,
                    ),
                    async_method="get_async",
                )

                self._check_validate_type(key, instance)

            except (NotFoundError, OperationError):
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
            result = self._invoke_factory(
                factory,
                key,
                config,
                {},
                positional_key=True,
                allow_async_classmethod=False,
            )
            instance = await result if inspect.isawaitable(result) else result

            self._check_validate_type(key, instance)

        except (NotFoundError, OperationError):
            raise
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
            instance = self._refuse_awaitable(
                key,
                self._invoke_factory(
                    factory,
                    key,
                    config,
                    kwargs,
                    positional_key=False,
                    allow_async_classmethod=False,
                ),
                async_method="create_async",
            )

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
            result = self._invoke_factory(
                factory,
                key,
                config,
                kwargs,
                positional_key=False,
                allow_async_classmethod=True,
            )
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
    ) -> tuple[PluginFactory[T], str, Dict[str, Any] | None]:
        """Resolve ``(factory, canonical_key, config)`` for create paths.

        Shared prologue for :meth:`create` and :meth:`create_async` — the
        single source of truth for routing-key resolution and factory
        lookup, so the sync and async paths cannot drift. Handles:

        - Explicit ``key``, or extraction from ``config[config_key]``
          (falling back to ``config_key_default``).
        - Reporting a key that came from the default rather than from the
          config — at WARNING for a registry that declared what its
          fallback costs, at DEBUG otherwise. See
          :meth:`_report_key_defaulted`.
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
                    self._report_key_defaulted(key)
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
                        f"available here: {self._unavailable[key].reason}"
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

    # ------------------------------------------------------------------
    # The invocation epilogue
    #
    # `_resolve_factory` was extracted "so sync create and create_async
    # cannot drift" -- and it covers the prologue only, so the epilogue
    # went on being written four times and drifted four ways: the call
    # arity, whether the result is awaited, which predicate decides
    # "awaitable", and whether the type guard's own message survives.
    # These three methods are that epilogue, shared. What genuinely
    # differs between the four entry points is two booleans, and they are
    # parameters here rather than four copies of a body.
    # ------------------------------------------------------------------

    def _invoke_factory(
        self,
        factory: Callable[..., Any],
        key: str,
        config: Dict[str, Any] | None,
        kwargs: Dict[str, Any],
        *,
        positional_key: bool,
        allow_async_classmethod: bool,
    ) -> Any:
        """Call ``factory`` and return its result, awaited or not.

        The one place this class decides how a factory is invoked, and
        therefore the only place the two arities are stated:

        * ``positional_key`` -- the ``get`` lane's ``factory(key, config)``.
          The key identifies the plugin to the factory, and no extra
          keyword arguments are threaded.
        * otherwise -- the ``create`` lane's ``factory(config, **kwargs)``,
          which prefers a class's ``from_config`` / ``from_config_async``
          constructor when it has one.

        ``allow_async_classmethod`` is separate from the lane because only
        a caller that can await may reach ``from_config_async``; a
        synchronous caller must fall through to ``from_config`` and get a
        real instance rather than a coroutine it cannot resolve.

        The result may be an awaitable. Deciding what to do about that is
        the caller's, through :meth:`_refuse_awaitable` or an ``await``.
        """
        if positional_key:
            return factory(key, config or {})
        if isinstance(factory, type):
            if allow_async_classmethod and hasattr(factory, "from_config_async"):
                return factory.from_config_async(config or {}, **kwargs)
            if hasattr(factory, "from_config"):
                return factory.from_config(config or {}, **kwargs)
        return factory(config or {}, **kwargs)

    def _refuse_awaitable(self, key: str, result: Any, *, async_method: str) -> Any:
        """Pass a plain result through; refuse an awaitable one.

        A synchronous caller cannot await, so the only two honest answers
        are the instance or an error. Returning the awaitable was the
        third, and it is the defect this method exists to remove: an
        un-awaited coroutine reaching a caller produces no exception, no
        log line, and a ``RuntimeWarning`` at interpreter shutdown
        attributed to the factory rather than to the registry.

        It is refused *before* the caller caches anything. ``get`` cached
        whatever it received, so the coroutine became the stored instance
        and every later caller got the same already-awaited object --
        ``RuntimeError: cannot reuse already awaited coroutine``, naming
        neither the registry nor the key.

        The awaitable is closed rather than dropped, so refusing it does
        not itself emit the warning the refusal exists to make unnecessary.
        """
        if not inspect.isawaitable(result):
            return result

        close = getattr(result, "close", None)
        if callable(close):
            close()

        name = getattr(result, "__qualname__", None) or type(result).__name__
        raise OperationError(
            f"Plugin '{key}' in registry '{self._name}' has an asynchronous "
            f"factory ({name}); this method cannot await it. "
            f"Call {async_method}() instead.",
            context={
                "key": key,
                "registry": self._name,
                "async_method": async_method,
            },
        )

    def _report_key_defaulted(self, key: str) -> None:
        """Report that nothing in the config chose ``key``.

        An absent routing key and an explicit one naming the same value
        produce the same object, which is exactly what made the difference
        invisible: the only place the distinction still exists is here,
        between reading the config and resolving the name.

        The level says whose problem it is. A registry that passed
        ``default_warning`` is claiming its fallback has a consequence
        someone needs to act on -- a lock coordinating nothing, a bus
        whose events reach nobody -- and gets WARNING. One that passed
        nothing has a default that is the recommended answer, so the
        omission is ordinary and the record belongs at DEBUG, where it is
        still there for whoever goes looking.
        """
        context = {
            "config_key": self._config_key,
            "key": key,
            "registry": self._name,
        }
        if self._default_warning is None:
            logger.debug(DEFAULT_KEY_WARNING, context)
        else:
            logger.warning(self._default_warning, context)

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

    def is_known(self, key: str) -> bool:
        """Whether this registry has heard of ``key`` at all.

        :meth:`is_registered` answers "can I build this?"; this answers
        "do I recognise the name?", which is the larger set once
        :meth:`declare_unavailable` is in use. The two differ exactly over
        the plugins whose driver is missing, which is the case a caller
        distinguishing a typo from an uninstalled backend needs -- asking
        it via truthy metadata instead gets the answer wrong for a plugin
        declared without any.

        Args:
            key: Key to check. Aliases are accepted.

        Returns:
            True if the key is registered or declared unavailable.
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            return key in self._factories or key in self._unavailable

    def load_declared_type(self, key: str) -> type | None:
        """The class of a plugin that cannot be created here, if reachable.

        For a caller that wants to *read* something off the class -- a
        typed config schema, a docstring, a class attribute -- without
        building the plugin. Deliberately separate from
        :meth:`get_factory`, which returns ``None`` here and must keep
        doing so: a declared-unavailable plugin is not creatable, and a
        caller reaching for a factory means to create.

        Whether the class is reachable is discovered rather than assumed.
        A plugin whose module guards its optional driver behind a flag
        imports fine without it; one whose module imports the driver at
        top level does not, and that is the case this returns ``None`` for
        rather than propagating. Either way the caller gets a definite
        answer instead of having to model the two idioms itself.

        Args:
            key: Key to look up. Aliases are accepted.

        Returns:
            The class, or ``None`` if the key is not declared unavailable,
            was declared without a ``type_loader``, or its module cannot be
            imported here after all.
        """
        self._ensure_initialized()
        key = self._canon(key)

        with self._lock:
            declared = self._unavailable.get(key)
        if declared is None or declared.type_loader is None:
            return None

        # Outside the lock: the loader imports a module, which runs
        # arbitrary top-level code and can re-enter this registry.
        try:
            loaded = declared.type_loader()
        except ImportError as exc:
            logger.debug(
                "Plugin '%s' in %s is declared unavailable and its class "
                "cannot be imported either: %s",
                key,
                self._name,
                exc,
            )
            return None
        return loaded if isinstance(loaded, type) else None

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

    def get_factory(self, key: str) -> PluginFactory[T] | None:
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
        """Metadata of another spelling of ``key``'s plugin, if any has some.

        Two ways to be another spelling, because the two states record the
        relationship differently. A creatable alias shares its canonical
        key's factory, which is what identifies the group. An unavailable
        one has no factory to share, so :meth:`declare_unavailable` names
        the describing key outright.

        Caller holds the lock.
        """
        declared = self._unavailable.get(key)
        if declared is not None:
            return self._metadata.get(declared.describes_key)

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

    def set_default_factory(self, factory: PluginFactory[T]) -> None:
        """Set the default factory.

        Args:
            factory: New default factory

        Raises:
            TypeError: If *factory* is a class that cannot produce
                ``validate_type``, or if ``validate_type`` cannot be
                checked against at all.
        """
        if self._validate_type and isinstance(factory, type):
            self._check_factory_class(factory, what="Default factory")

        self._default_factory = factory

    def bulk_register(
        self,
        factories: Mapping[str, PluginFactory[T]],
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

    def copy(self) -> Dict[str, PluginFactory[T]]:
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
    # Both are part of the `declare_unavailable` / `default_warning`
    # surface: `Unavailable` is what `PluginRegistry` records for a
    # withdrawn plugin and is named in its docstrings, and
    # `DEFAULT_KEY_WARNING` is the text a registry falls back to at DEBUG.
    # Referenced from public documentation, so exported rather than
    # reachable only by knowing the module layout.
    "DEFAULT_KEY_WARNING",
    "PluginRegistry",
    "Registry",
    "Unavailable",
]
