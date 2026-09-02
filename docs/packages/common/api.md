# Common API Reference

Curated reference for the most-used parts of the `dataknobs-common` package.

> **📖 For everything:** [Auto-generated API Reference](../../api/reference/common.md) — every public symbol, generated from source docstrings

This page covers a selection — the exception hierarchy, the four registries,
serialization, metadata, retry, transitions and lifecycle — with worked
examples and the surrounding rationale. It is deliberately not exhaustive:
`dataknobs_common` exports roughly two hundred names, and the auto-generated
reference above is the complete one. When the two disagree, the generated page
is built from the source and wins.

---

## Module Overview

The modules this page covers:

- **`dataknobs_common.exceptions`** - Exception hierarchy with context support
- **`dataknobs_common.registry`** - Generic registry implementations
- **`dataknobs_common.serialization`** - Serialization protocols and utilities
- **`dataknobs_common.metadata`** - Layered-merge primitive with immutable-key enforcement
- **`dataknobs_common.retry`** - Configurable retry execution with backoff strategies
- **`dataknobs_common.lifecycle`** - Owned-vs-injected collaborator teardown guard
- **`dataknobs_common.transitions`** - Stateless transition validation for status graphs

The package ships more than these. Several of the modules below have a guide
of their own linked from the [package overview](index.md); all of them appear
in full in the [auto-generated reference](../../api/reference/common.md):

| Module | What it provides |
|---|---|
| `events` | Event bus for pub/sub messaging (in-memory, PostgreSQL, Redis, SQS) |
| `locks` | Distributed and in-process locks, and the lock backend registry |
| `ratelimit` | Rate limiters, limits, and the rate-limiter backend registry |
| `capabilities` | Capability declaration and the `require_capability` / `supports_capability` guards |
| `resolver`, `discriminator`, `scope` | Resource resolution, routing, and scope projection protocols with reference implementations |
| `structured_config` | `StructuredConfig` and the consumer mixin |
| `config_loading`, `paths`, `postgres_config` | YAML/JSON loading, safe path joining, PostgreSQL DSN normalization |
| `tenancy`, `packs` | Tenant contexts and composable configuration packs |
| `callbacks`, `async_iter`, `sync_bridge` | Callback registries, sync-iterator offloading, and the sync/async bridge |
| `expressions` | Safe expression evaluation over a restricted builtin set |
| `imports`, `copying`, `bounded_cache`, `aws` | Dotted-path resolution, structure copying, an LRU cache, and the shared aioboto3 session |
| `testing` | Test utilities, skip markers, and configuration factories |

## Exceptions Module

### Base Exception

#### `DataknobsError`

Base exception for all dataknobs packages.

```python
class DataknobsError(Exception):
    """Base exception for all dataknobs packages."""
```

**Constructor:**
```python
DataknobsError(
    message: str,
    context: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
)
```

**Parameters:**
- `message` (str): Error message
- `context` (dict[str, Any] | None): Optional context dictionary with additional error details
- `details` (dict[str, Any] | None): The same thing under the name FSM-derived code uses. The two are never merged — the attribute becomes the first non-empty of `details`, `context`, `{}` — so `details` wins over `context` except when `details` is itself empty

**Attributes:**
- `context` (dict[str, Any]): The context dictionary — always a dict, `{}` when neither argument was given, so it is safe to subscript without a `None` check
- `details` (dict[str, Any]): The same object as `context`, not a copy

There is no `message` attribute. The message is the exception's `args[0]`,
reached with `str(e)` as for any built-in exception.

**Example:**
```python
from dataknobs_common import DataknobsError

# Simple error
raise DataknobsError("Something went wrong")

# Error with context
raise DataknobsError(
    "Operation failed",
    context={"operation": "save", "item_id": "123"}
)

# Access message and context
try:
    operation()
except DataknobsError as e:
    print(str(e))     # "Operation failed"
    print(e.context)  # {"operation": "save", "item_id": "123"}
    print(e.details)  # the same dict object as e.context
```

### Standard Exceptions

All standard exceptions extend `DataknobsError` and follow the same constructor pattern.

Beyond the eight below, the hierarchy also carries `ConsentRequiredError`,
`RateLimitError` (which extends `OperationError`), and the dotted-path pair
`DottedPathError` / `DottedPathTypeError` (both extending
`ConfigurationError`). See the
[auto-generated reference](../../api/reference/common.md) for those.

#### `ValidationError`

Raised for data validation failures.

```python
class ValidationError(DataknobsError):
    """Data validation failed."""
```

**Example:**
```python
from dataknobs_common import ValidationError

raise ValidationError(
    "Invalid email format",
    context={"email": "invalid-email", "field": "user.email"}
)
```

#### `ConfigurationError`

Raised for configuration issues.

```python
class ConfigurationError(DataknobsError):
    """Configuration error."""
```

**Example:**
```python
from dataknobs_common import ConfigurationError

raise ConfigurationError(
    "Missing required configuration",
    context={"missing_keys": ["api_key", "endpoint"]}
)
```

#### `ResourceError`

Raised for resource acquisition or management failures.

```python
class ResourceError(DataknobsError):
    """Resource error."""
```

**Example:**
```python
from dataknobs_common import ResourceError

raise ResourceError(
    "Database connection failed",
    context={"host": "db.example.com", "port": 5432}
)
```

#### `NotFoundError`

Raised when an item cannot be found.

```python
class NotFoundError(DataknobsError):
    """Item not found."""
```

**Example:**
```python
from dataknobs_common import NotFoundError

raise NotFoundError(
    "User not found",
    context={"user_id": "123", "searched_in": "users_table"}
)
```

#### `OperationError`

Raised for general operation failures.

```python
class OperationError(DataknobsError):
    """Operation failed."""
```

**Example:**
```python
from dataknobs_common import OperationError

raise OperationError(
    "Payment processing failed",
    context={"transaction_id": "txn_123", "error_code": "INSUFFICIENT_FUNDS"}
)
```

#### `ConcurrencyError`

Raised for concurrent operation conflicts.

```python
class ConcurrencyError(DataknobsError):
    """Concurrency error."""
```

**Example:**
```python
from dataknobs_common import ConcurrencyError

raise ConcurrencyError(
    "Resource locked by another process",
    context={"resource_id": "res_123", "locked_by": "process_456"}
)
```

#### `SerializationError`

Raised for serialization/deserialization failures.

```python
class SerializationError(DataknobsError):
    """Serialization error."""
```

**Example:**
```python
from dataknobs_common import SerializationError

raise SerializationError(
    "Failed to deserialize object",
    context={"class": "User", "error": "missing required field 'email'"}
)
```

#### `TimeoutError`

Raised for operation timeout errors.

```python
class TimeoutError(DataknobsError):
    """Operation timed out."""
```

**Example:**
```python
from dataknobs_common import TimeoutError

raise TimeoutError(
    "API request timed out",
    context={"url": "https://api.example.com", "timeout_seconds": 30}
)
```

## Registry Module

### Base Registry

#### `Registry[T]`

Generic, thread-safe registry for managing named items.

```python
class Registry(Generic[T]):
    """Thread-safe registry for managing named items."""
```

**Type Parameter:**
- `T`: Type of items stored in the registry

**Constructor:**
```python
Registry(
    name: str,
    enable_metrics: bool = False
)
```

**Parameters:**
- `name` (str): Registry name (for logging and metrics)
- `enable_metrics` (bool): Record a registration timestamp and the `metadata` argument for each key, readable through `get_metrics()`. When `False` (the default), `metadata` passed to `register()` is accepted and discarded

**Properties:**
- `name` (str): The registry name given at construction

**Methods:**

##### `register(key: str, item: T, metadata: dict[str, Any] | None = None, allow_overwrite: bool = False) -> None`

Register an item with a key.

**Parameters:**
- `key` (str): Unique identifier for the item
- `item` (T): The item to register
- `metadata` (dict[str, Any] | None): Optional metadata. Retained only when the registry was built with `enable_metrics=True`
- `allow_overwrite` (bool): Replace an existing registration instead of raising. Per call, not per registry

**Raises:**
- `OperationError`: If key already exists and `allow_overwrite` is False

**Example:**
```python
from dataknobs_common import Registry

registry = Registry[str]("messages")
registry.register("greeting", "Hello, world!")
registry.register("farewell", "Goodbye!", allow_overwrite=True)
```

##### `get(key: str) -> T`

Get an item by key.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `T`: The registered item

**Raises:**
- `NotFoundError`: If key not found. Its `context` carries `key`, `registry` and `available_keys`

**Example:**
```python
message = registry.get("greeting")  # "Hello, world!"
```

##### `get_optional(key: str) -> T | None`

Get an item by key, returning None if not found.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `T | None`: The registered item or None

**Example:**
```python
message = registry.get_optional("greeting")  # "Hello, world!"
missing = registry.get_optional("unknown")   # None
```

##### `has(key: str) -> bool`

Check if a key exists.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `bool`: True if key exists

**Example:**
```python
if registry.has("greeting"):
    print("Greeting exists")
```

##### `unregister(key: str) -> T`

Remove and return an item.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `T`: The removed item

**Raises:**
- `NotFoundError`: If key not found

**Example:**
```python
removed = registry.unregister("greeting")
```

##### `list_items() -> list[T]`

Get list of all items.

**Returns:**
- `list[T]`: All registered items

**Example:**
```python
all_messages = registry.list_items()
```

##### `list_keys() -> list[str]`

Get list of all keys.

**Returns:**
- `list[str]`: All registered keys

**Example:**
```python
keys = registry.list_keys()
```

##### `items() -> list[tuple[str, T]]`

Get list of (key, item) tuples.

**Returns:**
- `list[tuple[str, T]]`: All (key, item) pairs

**Example:**
```python
for key, item in registry.items():
    print(f"{key}: {item}")
```

##### `count() -> int`

Get number of registered items.

**Returns:**
- `int`: Number of items

**Example:**
```python
total = registry.count()
```

##### `clear() -> None`

Remove all items.

**Example:**
```python
registry.clear()
```

##### `get_metrics(key: str | None = None) -> dict[str, Any]`

Get registration metrics. Empty unless the registry was built with
`enable_metrics=True`; this is also where the `metadata` passed to
`register()` is read back from.

**Parameters:**
- `key` (str | None): A specific key, or `None` for every key

**Returns:**
- `dict[str, Any]`: For a single key, `{"registered_at": float, "metadata": dict}`. For `None`, a dict of those keyed by registration key. `{}` when metrics are off, or when the key is unknown

**Example:**
```python
registry = Registry[str]("messages", enable_metrics=True)
registry.register("farewell", "Goodbye!", metadata={"lang": "en"})

registry.get_metrics("farewell")
# {"registered_at": 1699456789.0, "metadata": {"lang": "en"}}
```

**Magic Methods:**

```python
len(registry)           # Same as count()
key in registry         # Same as has(key)
for item in registry    # Iterate over items — the same sequence as list_items()
```

### Cached Registry

#### `CachedRegistry[T]`

Registry with automatic TTL-based caching.

```python
class CachedRegistry(Registry[T]):
    """Registry with TTL-based caching."""
```

**Constructor:**
```python
CachedRegistry(
    name: str,
    cache_ttl: int = 300,
    max_cache_size: int = 1000
)
```

**Parameters:**
- `name` (str): Registry name
- `cache_ttl` (int): Cache TTL in seconds (default: 300)
- `max_cache_size` (int): Entries to hold before evicting the oldest tenth (default: 1000)

Metrics are always on for a cached registry — it passes `enable_metrics=True`
to the base constructor and takes no argument for it.

**Additional Methods:**

##### `get_cached(key: str, factory: Callable[[], T], force_refresh: bool = False) -> T`

Get cached item or create with factory.

**Parameters:**
- `key` (str): Cache key
- `factory` (Callable[[], T]): Factory function to create item if not cached
- `force_refresh` (bool): Call the factory and re-cache even on a live entry

**Returns:**
- `T`: Cached or newly created item

The cache is separate from the registry's own items: `get_cached` never
consults `register()`ed entries, and `clear()` does not empty it. Use
`invalidate_cache()` for that.

**Example:**
```python
from dataknobs_common import CachedRegistry

cache = CachedRegistry[Bot]("bots", cache_ttl=300)

def create_bot():
    return Bot(client_id="client1")

bot = cache.get_cached("client1", factory=create_bot)
# First call: creates bot
# Second call: returns cached bot (if within TTL)
```

##### `invalidate_cache(key: str | None = None) -> None`

Invalidate cache entry or entire cache.

**Parameters:**
- `key` (str | None): Specific key to invalidate, or None to invalidate all

**Example:**
```python
cache.invalidate_cache("client1")  # Invalidate specific item
cache.invalidate_cache()           # Invalidate all items
```

##### `get_cache_stats() -> dict[str, Any]`

Get cache statistics.

**Returns:**
- `dict[str, Any]`: `size`, `max_size`, `ttl_seconds`, `hits`, `misses`, `total_requests`, `hit_rate` (a float in `0.0..1.0`, and `0.0` before the first request)

**Example:**
```python
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

### Async Registry

#### `AsyncRegistry[T]`

Async version of Registry.

```python
class AsyncRegistry(Generic[T]):
    """Async registry for managing named items."""
```

**Constructor:**
```python
AsyncRegistry(
    name: str,
    enable_metrics: bool = False
)
```

**Methods:**

The same surface as `Registry`, with every method a coroutine and the same
`NotFoundError` / `OperationError` behaviour:

```python
await registry.register(key, item, metadata=None, allow_overwrite=False)
item = await registry.get(key)
item = await registry.get_optional(key)
exists = await registry.has(key)
item = await registry.unregister(key)
items = await registry.list_items()
keys = await registry.list_keys()
pairs = await registry.items()
count = await registry.count()
await registry.clear()
metrics = await registry.get_metrics(key)
```

It is not a subclass of `Registry` — it reimplements the surface over an
`asyncio.Lock` — so `isinstance(reg, Registry)` is `False` for one.

Four members are deliberately synchronous, reading the dict without taking the
lock: the `name` property, and `len()` / `in` / iteration, which give the count,
key membership, and a snapshot of the items. They are the same shapes the sync
registry offers, usable without an `await`.

**Example:**
```python
from dataknobs_common import AsyncRegistry

registry = AsyncRegistry[Resource]("resources")

await registry.register("db", db_resource)
resource = await registry.get("db")
count = await registry.count()
```

### Plugin Registry

#### `PluginRegistry[T]`

Registry with factory support for creating fresh instances on demand, lazy initialization, and configuration-driven key resolution.

```python
class PluginRegistry(Generic[T]):
    """Registry with factory support and lazy initialization."""
```

**Type Parameter:**
- `T`: Type of items produced by factories

**Constructor:**
```python
PluginRegistry(
    name: str,
    default_factory: type[T] | Callable[..., T] | None = None,
    validate_type: type | None = None,
    *,
    canonicalize_keys: bool = False,
    config_key: str | None = None,
    config_key_default: str | None = None,
    strip_config_key: bool = False,
    on_first_access: Callable[[PluginRegistry[T]], None] | None = None,
    not_found_kind: str | None = None,
    not_found_exception: type[Exception] = NotFoundError,
    default_warning: str | None = None,
)
```

**Parameters:**
- `name` (str): Registry name for identification
- `default_factory` (type[T] | Callable | None): Default factory when key not found
- `validate_type` (type[T] | None): Base type to validate registrations against. A class, an ABC, or a `@runtime_checkable` Protocol — including one carrying properties. See [What `validate_type` checks](plugin-registry.md#what-validate_type-checks)
- `canonicalize_keys` (bool): Lowercase all keys for case-insensitive lookup
- `config_key` (str | None): Field name to extract lookup key from config dicts in `create()`
- `config_key_default` (str | None): Fallback value when `config_key` field is absent
- `strip_config_key` (bool): Remove the config key field from config before passing to factory
- `on_first_access` (Callable | None): Callback invoked once before first public method access. Supports re-entrant calls (e.g., callback can call `register()`)
- `not_found_kind` (str | None): Kind label for the not-found message from `create()` / `create_async()`. Setting it to e.g. `"event bus backend"` produces `"Unknown event bus backend: <key>. Available backends: <sorted-keys>"`; leaving it `None` keeps `"Plugin '<key>' not registered"`
- `not_found_exception` (type[Exception]): Class raised on not-found by `create()` / `create_async()`. `NotFoundError` by default; a shim preserving a historical `ValueError` contract passes that instead. A class not rooted in `DataknobsError` is constructed with the message only, since a stdlib exception would reject the `context=` keyword
- `default_warning` (str | None): What this registry's fallback costs, logged at WARNING when the routing key was absent from config and `config_key_default` supplied it. Interpolated with `%(config_key)s`, `%(key)s` and `%(registry)s`; a literal percent must be written `%%`. Leave it `None` when the default is simply the recommended answer — the fallback is then recorded at DEBUG

**Methods:**

##### `register(key, factory, override=False, metadata=None, *, allow_overwrite=None) -> None`

Register a plugin class or factory function.

**Parameters:**
- `key` (str): Unique identifier
- `factory` (type[T] | Callable[..., T]): Plugin class or factory
- `override` (bool): Allow replacing existing registration
- `metadata` (dict[str, Any] | None): Optional metadata for the registration
- `allow_overwrite` (bool | None): Keyword alias for `override`, matching `Registry.register`. When not `None` it wins; use whichever name fits the surrounding code

**Raises:**
- `OperationError`: If key already registered and `override=False`
- `TypeError`: If `factory` is not a class or callable, if it is a class that cannot produce `validate_type`, or if `validate_type` cannot be checked against at all. A callable factory's result is checked instead by `get()` / `create()`, which raise `OperationError`

##### `get(key, config=None, use_cache=True, use_default=True) -> T`

Get or create a cached plugin instance. Factories are called with `(key, config)` signature.

**Parameters:**
- `key` (str): Plugin identifier
- `config` (dict | None): Configuration passed to factory
- `use_cache` (bool): Return cached instance if available
- `use_default` (bool): Use default factory if key not registered

**Returns:** Plugin instance

**Raises:** `NotFoundError` if key not registered and no default

##### `create(key=None, config=None, **kwargs) -> T`

Create a fresh instance without caching. Uses `(config, **kwargs)` factory signature. Detects `from_config` classmethods on class factories.

**Parameters:**
- `key` (str | None): Plugin identifier. Optional when `config_key` is configured.
- `config` (dict | None): Configuration passed to factory
- `**kwargs`: Additional keyword arguments forwarded to factory

**Returns:** Fresh plugin instance

**Raises:**
- `ValueError`: If `key` is None and cannot be resolved
- `NotFoundError`: If resolved key is not registered — or whatever class `not_found_exception` names
- `OperationError`: If the factory raises, or if it returns something that is not a `validate_type`. The factory's own message is not copied into the wrapper; it travels on `__cause__`

##### `get_factory(key) -> type[T] | Callable[..., T] | None`

Get the raw factory for a key without creating an instance.

##### `is_registered(key) -> bool`

Check whether a key is registered.

##### `list_keys() -> list[str]`

List all registered plugin keys (insertion order).

##### `get_metadata(key, *, follow_alias=False) -> dict[str, Any]`

Get metadata for a registration (returns empty dict if no metadata stored).
The returned dict is a deep copy, so editing a nested value in it does not
change what the next caller reads. With `follow_alias=True`, a key carrying no
metadata of its own answers with the metadata of a key sharing its factory —
which is what makes an alias like `pg` answer for `postgres` rather than
returning `{}`.

**Example:**
```python
from dataknobs_common.registry import PluginRegistry

# Define a registry with lazy initialization
def _register_builtins(registry):
    registry.register("default", DefaultHandler)
    registry.register("custom", CustomHandler)

handlers = PluginRegistry[Handler](
    "handlers",
    validate_type=Handler,
    canonicalize_keys=True,
    config_key="handler_type",
    config_key_default="default",
    on_first_access=_register_builtins,
)

# create() resolves key from config and calls from_config()
handler = handlers.create(config={"handler_type": "custom", "timeout": 30})

# get() returns cached instances with (key, config) signature
handler = handlers.get("custom", config={"timeout": 30})

# get_factory() returns the raw class/callable
handler_cls = handlers.get_factory("custom")
```

**Additional methods** (see auto-generated API reference for full details):

- `get_async(key, config, use_cache, use_default)` — Async version of `get()`, awaits coroutine factories
- `unregister(key)` — Forget a key, whether registered or declared unavailable (raises `NotFoundError` if the registry has never heard of it)
- `clear_cache(key=None)` — Clear cached instances (specific key or all)
- `set_default_factory(factory)` — Set/change the default factory
- `bulk_register(factories, override)` — Register multiple plugins at once, by delegating to `register()`
- `create_async(key, config, **kwargs)` — Async version of `create()`, awaiting coroutine factories before the `validate_type` check
- `declare_unavailable(key, *, reason, ...)` — Mark a key as known-but-unbuildable, so a lookup explains why instead of reporting it missing
- `has(key)` / `is_known(key)` — Registered, versus registered *or* declared unavailable
- `list_canonical_keys()` / `list_known_keys()` — Keys without aliases, and keys including the unavailable ones
- `load_declared_type(key)` — Load the type a declaration named, without instantiating it
- `copy()` — Get a copy of the factories dict
- `name` (property) — Registry name
- `cached_instances` (property) — Direct access to instance cache dict
- Supports `len()`, `in` operator, and `repr()`

---

## Serialization Module

### Protocol

#### `Serializable`

Protocol for objects that can be serialized to/from dictionaries.

```python
@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T: ...
```

**Example:**
```python
from dataknobs_common import Serializable
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(name=data["name"], email=data["email"])

# Type checking works
user = User("Alice", "alice@example.com")
assert isinstance(user, Serializable)  # True
```

### Utility Functions

#### `serialize(obj: Serializable) -> dict`

Serialize an object to a dictionary.

**Parameters:**
- `obj` (Serializable): Object to serialize

**Returns:**
- `dict`: Serialized dictionary

**Raises:**
- `SerializationError`: If serialization fails

**Example:**
```python
from dataknobs_common import serialize

user = User("Alice", "alice@example.com")
data = serialize(user)
# {"name": "Alice", "email": "alice@example.com"}
```

#### `deserialize(cls: type[T], data: dict) -> T`

Deserialize a dictionary to an object.

**Parameters:**
- `cls` (type[T]): Class to deserialize to
- `data` (dict): Dictionary to deserialize

**Returns:**
- `T`: Deserialized object

**Raises:**
- `SerializationError`: If deserialization fails

**Example:**
```python
from dataknobs_common import deserialize

data = {"name": "Alice", "email": "alice@example.com"}
user = deserialize(User, data)
```

#### `serialize_list(objects: list[Serializable]) -> list[dict]`

Serialize a list of objects.

**Parameters:**
- `objects` (list[Serializable]): List of objects to serialize

**Returns:**
- `list[dict]`: List of serialized dictionaries

**Raises:**
- `SerializationError`: If serialization fails

**Example:**
```python
from dataknobs_common import serialize_list

users = [
    User("Alice", "alice@example.com"),
    User("Bob", "bob@example.com")
]
data = serialize_list(users)
# [{"name": "Alice", ...}, {"name": "Bob", ...}]
```

#### `deserialize_list(cls: type[T], data_list: list[dict]) -> list[T]`

Deserialize a list of dictionaries.

**Parameters:**
- `cls` (type[T]): Class to deserialize to
- `data_list` (list[dict]): List of dictionaries to deserialize

**Returns:**
- `list[T]`: List of deserialized objects

**Raises:**
- `SerializationError`: If deserialization fails

**Example:**
```python
from dataknobs_common import deserialize_list

data = [
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"}
]
users = deserialize_list(User, data)
```

#### `is_serializable(obj: Any) -> bool`

Check if an object is serializable.

**Parameters:**
- `obj` (Any): Object to check

**Returns:**
- `bool`: True if object has `to_dict` method

**Example:**
```python
from dataknobs_common import is_serializable

user = User("Alice", "alice@example.com")
if is_serializable(user):
    data = serialize(user)
```

#### `is_deserializable(cls: type) -> bool`

Check if a class is deserializable.

**Parameters:**
- `cls` (type): Class to check

**Returns:**
- `bool`: True if class has `from_dict` classmethod

**Example:**
```python
from dataknobs_common import is_deserializable

if is_deserializable(User):
    user = deserialize(User, data)
```

### JSON Safety Functions

#### `sanitize_for_json(value: Any, on_drop: str = "silent") -> Any`

Recursively traverse a value and drop anything not JSON-serializable. Handles
dicts, lists, dataclasses, sets, tuples, bytes, datetime, Enum, and objects
with `to_dict()`.

**Parameters:**
- `value` (Any): The value to sanitize
- `on_drop` (str): Drop behavior — `"silent"` (DEBUG log, default), `"warn"` (WARNING log with key path), `"error"` (raises `SerializationError` listing all dropped paths)

**Returns:**
- `Any`: JSON-safe copy with non-serializable values removed

**Raises:**
- `SerializationError`: When `on_drop="error"` and non-serializable values are found

**Example:**
```python
from dataknobs_common.serialization import sanitize_for_json

data = {"name": "Alice", "callback": some_function, "count": 42}

# Silent mode (default) — drops with DEBUG log
safe = sanitize_for_json(data)
# {"name": "Alice", "count": 42}

# Warn mode — WARNING log with key path
safe = sanitize_for_json(data, on_drop="warn")

# Error mode — raises SerializationError
safe = sanitize_for_json(data, on_drop="error")
# SerializationError: Non-serializable values at: callback (type=function)
```

#### `validate_json_safe(value: Any) -> list[str]`

Read-only traversal returning paths to non-serializable values. Does not modify the input.

**Parameters:**
- `value` (Any): The value to check

**Returns:**
- `list[str]`: Paths to non-serializable values. Empty list means fully JSON-safe.

**Example:**
```python
from dataknobs_common.serialization import validate_json_safe

problems = validate_json_safe({"name": "ok", "fn": some_function})
# ["fn (type=function)"]

if not problems:
    print("Fully JSON-safe")
```

## Metadata Module

A primitive for "layered merge with a designated immutable source for some keys." Used wherever caller-supplied metadata needs to coexist with authoritative system metadata that the caller must not be able to overwrite (e.g. `domain_id` for tenant scoping, `chunk_index` for RAG chunk ordering, `node_type` for markdown chunker classification).

### `enforce_immutable_keys`

```python
from dataknobs_common.metadata import enforce_immutable_keys

# Caller-supplied metadata (may attempt overrides):
caller = {"category": "support", "domain_id": "tenant-a"}

# System-controlled fields the caller must not be able to overwrite:
system = {"domain_id": "tenant-b", "chunk_index": 0}

# Layered merge: caller wins on shared keys, EXCEPT for keys
# enumerated as immutable, where system always wins.
merged = enforce_immutable_keys(
    target=dict(caller),                  # mutated and returned
    caller=caller,                        # source for warning attribution
    source=system,                        # authoritative for immutable keys
    keys={"domain_id", "chunk_index"},    # the keys the caller cannot override
    context="MyComponent",                # prefixes the WARNING log
)

# merged == {
#     "category": "support",      # caller key, unaffected
#     "domain_id": "tenant-b",    # system wins (immutable)
#     "chunk_index": 0,           # system wins (immutable)
# }
# A WARNING is logged naming "domain_id" since the caller-supplied
# value differed from the system value for an immutable key.
```

**Signature:**

```python
def enforce_immutable_keys(
    *,
    target: dict[str, Any],
    caller: dict[str, Any] | None,
    source: dict[str, Any],
    keys: Iterable[str],
    logger: logging.Logger | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    ...
```

**Parameters:**

- `target` — The dict to mutate and return. Typically a copy of `caller` (or a fresh layered merge).
- `caller` — Caller-supplied metadata, used only for warning emission (compares each immutable key's value against `source`). Pass `None` to enforce immutability silently (e.g. on subsequent iterations of a loop after the warning has already been emitted once).
- `source` — Authoritative source for immutable-key values. For each `key in keys`, `target[key]` is set to `source[key]` (when the key is present in `source`).
- `keys` — Iterable of key names that the caller cannot override.
- `logger` — Logger to warn through. Defaults to the `dataknobs_common.metadata` logger, so a caller wanting the warning attributed to its own module passes its own.
- `context` — Prefix for the WARNING message, naming the component (e.g. `"VectorMemory"`, `"RAGKnowledgeBase"`). Omitted from the message when `None`.

**Behavior notes:**

- **Mutates and returns** the same `target` dict — use `dict(...)` to avoid aliasing if needed.
- **Warning is emitted at WARNING level** when any immutable key in `caller` has a different value than `source`, naming the key.
- **Array-safe equality:** `numpy` arrays, lists, and other non-scalar values are compared without raising `ValueError` from element-wise comparison's ambiguous truth value.
- **`caller=None` is silent:** the helper still enforces immutability but skips the warning. Useful for hoisting the warning out of a per-element loop.

**Used by:**

- `dataknobs_bots.memory.VectorMemory.add_message` — tenant-scope enforcement on caller metadata.
- `dataknobs_bots.knowledge.RAGKnowledgeBase._embed_and_store_chunks` — chunk-text and document-attribution protection.
- `dataknobs_xization.markdown.md_chunker.MarkdownChunker._create_chunk` — node-classification protection.

See the auto-generated reference for full type signatures, and the `dataknobs-bots` / `dataknobs-xization` CHANGELOGs for the concrete consumer-side fixes built on this helper.

## Retry Module

### `BackoffStrategy`

Enum defining backoff algorithms for retry delays.

```python
class BackoffStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    JITTER = "jitter"
    DECORRELATED = "decorrelated"
```

**Members:**

- `FIXED` — Constant delay between retries
- `LINEAR` — Delay increases linearly (`initial_delay * attempt`)
- `EXPONENTIAL` — Delay multiplied by `backoff_multiplier` each attempt
- `JITTER` — Exponential backoff with random jitter (controlled by `jitter_range`)
- `DECORRELATED` — Random delay between `initial_delay` and 3x previous delay

**Example:**
```python
from dataknobs_common.retry import BackoffStrategy

strategy = BackoffStrategy.EXPONENTIAL
strategy = BackoffStrategy("jitter")  # From string value
```

### `RetryConfig`

Dataclass configuring retry behavior.

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    retry_on_exceptions: list[type] | None = None
    retry_on_exception: Callable[[Exception], bool] | None = None
    retry_on_result: Callable[[Any], bool] | None = None
    on_retry: Callable[[int, Exception], None] | None = None
    on_failure: Callable[[Exception], None] | None = None
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `max_attempts` | `int` | `3` | Maximum execution attempts (including the first). Must be `>= 1`; a lower value raises `ValueError` at construction |
| `initial_delay` | `float` | `1.0` | Base delay in seconds before the first retry |
| `max_delay` | `float` | `60.0` | Upper bound on delay in seconds |
| `backoff_strategy` | `BackoffStrategy` | `EXPONENTIAL` | Algorithm for computing delay |
| `backoff_multiplier` | `float` | `2.0` | Multiplier for exponential/jitter strategies |
| `jitter_range` | `float` | `0.1` | Fractional jitter range for JITTER strategy (0.1 = +/-10%) |
| `retry_on_exceptions` | `list[type] \| None` | `None` | Only retry these exception types; others propagate immediately |
| `retry_on_exception` | `Callable \| None` | `None` | Called with the raised exception; return `True` to retry, `False` to re-raise. The value-based form of `retry_on_exceptions` for retryability that depends on an error attribute (HTTP status, SQLSTATE). Mutually exclusive with `retry_on_exceptions` |
| `retry_on_result` | `Callable \| None` | `None` | Return `True` to trigger retry based on result value |
| `on_retry` | `Callable \| None` | `None` | Hook called before retry sleep: `(attempt, exception)` |
| `on_failure` | `Callable \| None` | `None` | Hook called when all attempts exhausted: `(exception)` |

**Example:**
```python
from dataknobs_common.retry import RetryConfig, BackoffStrategy

config = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    backoff_strategy=BackoffStrategy.JITTER,
    retry_on_exceptions=[ConnectionError, TimeoutError],
    on_retry=lambda attempt, exc: logger.warning("Retry %d: %s", attempt, exc),
)
```

### `RetryExecutor`

Executes a callable with retry logic and configurable backoff.

```python
class RetryExecutor:
    def __init__(self, config: RetryConfig) -> None: ...
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def execute_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
```

**Constructor:**

- `config` (`RetryConfig`): The retry configuration

**Methods:**

##### `async execute(func, *args, **kwargs) -> Any`

Execute a callable with retry logic. Supports both sync and async callables —
any awaitable the callable returns is awaited, so plain functions, coroutine
functions, and async callable objects (`async def __call__`) are all handled.

**Parameters:**

- `func` (`Callable`): The callable to execute (sync or async; any awaitable result is awaited)
- `*args`: Positional arguments forwarded to func
- `**kwargs`: Keyword arguments forwarded to func

**Returns:**

- The return value of `func` on a successful attempt

**Raises:**

- The exception from the final failed attempt, or any non-retryable exception immediately

**Example:**
```python
from dataknobs_common.retry import RetryExecutor, RetryConfig, BackoffStrategy

config = RetryConfig(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.FIXED,
    initial_delay=1.0,
)
executor = RetryExecutor(config)

# Async callable
result = await executor.execute(fetch_data, url)

# Sync callable (also works from async context)
result = await executor.execute(parse_json, raw_text)
```

##### `execute_sync(func, *args, **kwargs) -> Any`

Synchronous entry point for the same bounded-retry engine. Applies the same
backoff, `retry_on_exceptions`, `retry_on_exception`, `retry_on_result`, and hook
policy as `execute`, but **blocks the calling thread** between attempts instead of
awaiting — use it from code that has no event loop.

**Parameters:**

- `func` (`Callable`): A synchronous callable to execute
- `*args`: Positional arguments forwarded to func
- `**kwargs`: Keyword arguments forwarded to func

**Returns:**

- The return value of `func` on a successful attempt

**Raises:**

- `TypeError`: If `func` is a coroutine function, or any callable whose return
  value is awaitable (an async callable object, or a sync callable returning a
  coroutine) — it cannot be awaited without an event loop, so it would otherwise
  return an un-awaited coroutine that never runs. Use `execute` instead.
- The exception from the final failed attempt, or any non-retryable exception immediately

**Example:**
```python
executor = RetryExecutor(config)

# Synchronous entry point (no event loop)
result = executor.execute_sync(parse_json, raw_text)
```

### `compute_backoff_delay`

Pure function that computes a single back-off delay for a given strategy
and attempt. Shared by `RetryExecutor` (bounded "give up after N"
retries) and the internal event-bus supervised-loop helper (unbounded
"never give up" listeners), so the delay math lives in exactly one
place. Stateless and side-effect-free (other than `random` for the
jittered/decorrelated strategies) — safe to call directly.

```python
def compute_backoff_delay(
    strategy: BackoffStrategy,
    *,
    attempt: int,
    initial_delay: float,
    max_delay: float,
    backoff_multiplier: float = 2.0,
    jitter_range: float = 0.1,
    previous_delay: float | None = None,
) -> float: ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | `BackoffStrategy` | — | The back-off algorithm to apply |
| `attempt` | `int` | — | The 1-based attempt number that just failed |
| `initial_delay` | `float` | — | Base delay in seconds |
| `max_delay` | `float` | — | Upper bound on the returned delay |
| `backoff_multiplier` | `float` | `2.0` | Multiplier for `EXPONENTIAL` and `JITTER` |
| `jitter_range` | `float` | `0.1` | Fractional jitter for `JITTER` (0.1 = +/-10%) |
| `previous_delay` | `float \| None` | `None` | Prior delay; only consulted by `DECORRELATED` |

**Returns:**

- Delay in seconds, capped at `max_delay`.

**Example:**
```python
from dataknobs_common import compute_backoff_delay
from dataknobs_common.retry import BackoffStrategy

# Exponential-with-jitter back-off for the 3rd consecutive failure.
delay = compute_backoff_delay(
    BackoffStrategy.JITTER,
    attempt=3,
    initial_delay=1.0,
    max_delay=30.0,
)
```

---

## Transitions Module

### `InvalidTransitionError`

Exception raised when a status transition is not allowed. Extends `OperationError`.

```python
class InvalidTransitionError(OperationError):
    def __init__(
        self,
        entity: str,
        current_status: str,
        target_status: str,
        allowed: set[str] | None = None,
    ) -> None: ...
```

**Parameters:**

- `entity` (`str`): Name of the entity or transition graph (e.g. `"run_status"`)
- `current_status` (`str`): The current status being transitioned from
- `target_status` (`str`): The target status that was rejected
- `allowed` (`set[str] | None`): Valid targets from `current_status`, or `None` if current status is unknown

**Attributes:**

- `entity` (`str`): The entity name
- `current_status` (`str`): The current status
- `target_status` (`str`): The rejected target status
- `allowed` (`set[str] | None`): Allowed targets, or `None` for unknown status
- `context` (`dict`): Structured context dict with keys `entity`, `current_status`, `target_status`, `allowed` (sorted list)

**Example:**
```python
from dataknobs_common.transitions import InvalidTransitionError

try:
    validator.validate("completed", "running")
except InvalidTransitionError as e:
    print(e.entity)          # "order"
    print(e.current_status)  # "completed"
    print(e.target_status)   # "running"
    print(e.allowed)         # set()
```

### `TransitionValidator`

Stateless validator for declarative transition graphs. Does not manage or store state — the caller owns the current status.

```python
class TransitionValidator:
    def __init__(self, name: str, transitions: dict[str, set[str]]) -> None: ...
```

**Constructor:**

- `name` (`str`): Human-readable name for the graph, used in error messages
- `transitions` (`dict[str, set[str]]`): Mapping from each status to its allowed target statuses. Statuses with empty sets are terminal.

**Properties:**

##### `name -> str`

The name of this transition graph.

##### `allowed_transitions -> dict[str, set[str]]`

Returns a copy of the full transition graph.

##### `statuses -> set[str]`

All known statuses (sources and targets).

**Methods:**

##### `validate(current_status: str | None, target_status: str) -> None`

Validate a proposed transition.

**Parameters:**

- `current_status` (`str | None`): The current status. If `None`, validation is skipped.
- `target_status` (`str`): The desired target status.

**Raises:**

- `InvalidTransitionError`: If the transition is not allowed.

**Example:**
```python
from dataknobs_common.transitions import TransitionValidator

ORDER = TransitionValidator("order", {
    "draft":     {"submitted"},
    "submitted": {"approved", "rejected"},
    "approved":  {"shipped"},
    "shipped":   {"delivered"},
    "rejected":  set(),
    "delivered":  set(),
})

ORDER.validate("draft", "submitted")  # ok
ORDER.validate(None, "submitted")     # ok (skip)
ORDER.validate("shipped", "draft")    # raises InvalidTransitionError
```

##### `is_valid(current_status: str | None, target_status: str) -> bool`

Check whether a transition is allowed without raising.

**Parameters:**

- `current_status` (`str | None`): The current status. If `None`, returns `True`.
- `target_status` (`str`): The desired target status.

**Returns:**

- `True` if the transition is allowed, `False` otherwise.

**Example:**
```python
if ORDER.is_valid(current, target):
    update_status(target)
else:
    logger.warning("Invalid transition: %s -> %s", current, target)
```

##### `get_reachable(from_status: str) -> set[str]`

Compute all statuses reachable from a given status (transitive closure).

**Parameters:**

- `from_status` (`str`): The starting status.

**Returns:**

- Set of all reachable statuses via one or more transitions. Does not include `from_status` itself unless there is a cycle.

**Raises:**

- `InvalidTransitionError`: If `from_status` is not a known status.

**Example:**
```python
reachable = ORDER.get_reachable("draft")
# {"submitted", "approved", "rejected", "shipped", "delivered"}

reachable = ORDER.get_reachable("delivered")
# set() — terminal status
```

---

## Lifecycle Module

Helpers for owned-vs-injected collaborator teardown. A class that holds a
collaborator (a database connection, an LLM provider, a connection pool)
must close it *only if it owns it*: a collaborator the holder built is
owned and torn down; a collaborator injected by a caller is left open for
its owner, so a resource shared across several holders survives one
holder's close.

### `close_if_owned`

```python
async def close_if_owned(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None
```

Closes `resource` only when `owns` is True, `resource` is not None, and it
exposes a `close()` method. Pass `on_error` to error-isolate the close —
the exception is caught and handed to the callback (e.g. a logger) instead
of propagating, so one failing subsystem in a teardown cascade does not
abort the rest. `asyncio.CancelledError` always propagates.

```python
from dataknobs_common import close_if_owned

class KnowledgeBase:
    async def close(self) -> None:
        # Built-from-config store is owned and closed; an injected,
        # shared store (owns=False) is left open for its owner.
        await close_if_owned(self.vector_store, self._owns_vector_store)

        # Error-isolated cascade: a failing close is logged, not raised.
        await close_if_owned(
            self.embedding_provider,
            self._owns_embedding_provider,
            on_error=lambda exc: logger.exception("Error closing provider"),
        )
```

### `close_if_owned_sync`

```python
def close_if_owned_sync(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None
```

Synchronous counterpart for collaborators whose `close()` is synchronous
(e.g. a sync database connection). Same ownership guard and optional error
isolation.

```python
from dataknobs_common import close_if_owned_sync

class MemoryBank:
    def close(self) -> None:
        # A db this bank built (owns_db=True) is closed; a caller-supplied
        # db shared across banks is left open.
        close_if_owned_sync(self._db, self._owns_db)
```

### `aclose_if_owned`

```python
async def aclose_if_owned(
    resource: Any,
    owns: bool,
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> None
```

For a collaborator exposing an `aclose()`, closed from async code. The
shape it exists for is the sync/async lifecycle pair — a *synchronous*
`close()` alongside an `aclose()` that awaits coroutine cleanup the sync
form skips — but any collaborator with an `aclose()` is served correctly,
including one whose `close()` is itself `async` and whose `aclose()` is an
alias for it.

Same ownership guard and same optional error isolation as its siblings;
only the probed method differs. Each helper probes exactly one method name
and skips when it is absent, so calling this one on a plain
`close()`-only collaborator closes **nothing** — that skip is logged at
DEBUG rather than passing in silence.

```python
from dataknobs_common import aclose_if_owned

class WizardReasoning:
    async def close(self) -> None:
        # WizardFSM mirrors AdvancedFSM's sync close() / async aclose();
        # only the latter awaits the resource manager's cleanup.
        await aclose_if_owned(
            self._fsm,
            self._owns_fsm,
            on_error=lambda exc: logger.exception("Error closing FSM: %s", exc),
        )
```

### Choosing between the three

The collaborator's interface decides, not the caller's:

| The collaborator exposes | Use |
|---|---|
| a synchronous `close()` | `close_if_owned_sync` |
| an `async def close()` | `close_if_owned` |
| an `aclose()`, closed from async | `aclose_if_owned` |

The third row is the one worth stating explicitly, because it is where
choosing by the *caller's* context instead of the collaborator's interface
goes wrong. For a synchronous `close()` alongside an `aclose()`, neither
sibling is correct rather than merely suboptimal: `close_if_owned` would
`await` that `close()`'s `None` return and raise `TypeError`, while
`close_if_owned_sync` would succeed silently through the lossy half,
skipping the cleanup `aclose()` exists to perform.

The rows are not disjoint, and need not be. A collaborator whose `close()`
is `async` and whose `aclose()` merely aliases it satisfies two of them,
and either helper is correct for it.

---

## Package Information

### Version

```python
from dataknobs_common import __version__
```

The version string for the dataknobs-common package.

**Type:** `str`

**Example:**
```python
from dataknobs_common import __version__

print(__version__)  # e.g. "3.1.0"
```

## Import Patterns

### Recommended Imports

```python
# Exceptions
from dataknobs_common import (
    DataknobsError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    NotFoundError,
    OperationError,
    ConcurrencyError,
    SerializationError,
    TimeoutError,
)

# Registry
from dataknobs_common import (
    Registry,
    CachedRegistry,
    AsyncRegistry,
)
from dataknobs_common.registry import PluginRegistry

# Serialization
from dataknobs_common import (
    Serializable,
    serialize,
    deserialize,
    serialize_list,
    deserialize_list,
    is_serializable,
    is_deserializable,
)

# JSON Safety
from dataknobs_common.serialization import sanitize_for_json, validate_json_safe

# Retry
from dataknobs_common import (
    BackoffStrategy,
    RetryConfig,
    RetryExecutor,
    compute_backoff_delay,
)

# Transitions
from dataknobs_common import (
    InvalidTransitionError,
    TransitionValidator,
)
```

### Module Imports

```python
# Import entire modules
from dataknobs_common import exceptions
from dataknobs_common import registry
from dataknobs_common import serialization
from dataknobs_common import retry
from dataknobs_common import transitions
```

## Type Annotations

### Registry Type Annotations

```python
from dataknobs_common import Registry
from typing import Protocol

class Tool(Protocol):
    name: str
    description: str

# Typed registry
tool_registry: Registry[Tool] = Registry("tools")

# Function accepting registry
def process_registry(registry: Registry[Tool]) -> None:
    for tool in registry.list_items():
        print(tool.name)
```

### Serializable Type Annotations

```python
from dataknobs_common import Serializable
from typing import TypeVar

T = TypeVar("T", bound=Serializable)

def save_to_file(obj: T, filepath: str) -> None:
    """Save any serializable object to file."""
    data = obj.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_from_file(cls: type[T], filepath: str) -> T:
    """Load any serializable object from file."""
    with open(filepath) as f:
        data = json.load(f)
    return cls.from_dict(data)
```

## Error Handling Patterns

### Catching All Dataknobs Errors

```python
from dataknobs_common import DataknobsError

try:
    # Any dataknobs operation
    result = some_dataknobs_operation()
except DataknobsError as e:
    logger.error("Dataknobs error: %s", e)
    if e.context:
        logger.error("Context: %s", e.context)
```

### Catching Specific Errors

```python
from dataknobs_common import ValidationError, NotFoundError, ResourceError

try:
    result = process_data(input_data)
except ValidationError as e:
    # Handle validation errors
    return {"error": "validation_failed", "details": e.context}
except NotFoundError as e:
    # Handle not found errors
    return {"error": "not_found", "id": e.context.get("id")}
except ResourceError as e:
    # Handle resource errors
    return {"error": "resource_unavailable", "resource": e.context.get("resource_id")}
```

### Registry Error Handling

```python
from dataknobs_common import NotFoundError, Registry

registry = Registry[Tool]("tools")

try:
    tool = registry.get("calculator")
except NotFoundError:
    # Handle missing key
    tool = default_tool

# Or use get_optional
tool = registry.get_optional("calculator")
if tool is None:
    tool = default_tool
```

### Serialization Error Handling

```python
from dataknobs_common import deserialize, SerializationError

try:
    user = deserialize(User, data)
except SerializationError as e:
    logger.error("Failed to deserialize: %s", e)
    logger.error("Error context: %s", e.context)
    # Handle error appropriately
```

## Advanced Usage Patterns

### Custom Registry with Validation

```python
from dataknobs_common import Registry

class ValidatedRegistry(Registry[T]):
    """Registry with validation on registration."""

    def register(
        self,
        key: str,
        item: T,
        metadata: dict | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        # Validate before registering
        if not self._validate(item):
            raise ValueError(f"Item validation failed: {key}")
        super().register(key, item, metadata, allow_overwrite)

    def _validate(self, item: T) -> bool:
        # Custom validation logic
        return True
```

### Serializable with Validation

```python
from dataknobs_common import Serializable, ValidationError
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        # Validate during deserialization
        if "@" not in data.get("email", ""):
            raise ValidationError(
                "Invalid email format",
                context={"email": data.get("email")}
            )
        return cls(name=data["name"], email=data["email"])
```

### Exception with Rich Context

```python
from dataknobs_common import OperationError

class ProcessingError(OperationError):
    """Custom processing error with rich context."""

    def __init__(
        self,
        stage: str,
        item_id: str,
        error: Exception,
        retry_count: int = 0
    ):
        super().__init__(
            f"Processing failed at stage '{stage}' for item '{item_id}'",
            context={
                "stage": stage,
                "item_id": item_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "retry_count": retry_count,
            }
        )
        self.stage = stage
        self.item_id = item_id
        self.original_error = error
```

## Best Practices

### 1. Use Type Parameters

```python
# Good: Typed registry
tool_registry = Registry[Tool]("tools")

# Less ideal: Untyped registry
tool_registry = Registry("tools")  # Type checking not enforced
```

### 2. Provide Context in Exceptions

```python
# Good: Rich context
raise NotFoundError(
    "User not found",
    context={"user_id": user_id, "search_criteria": criteria}
)

# Acceptable: Simple message
raise NotFoundError("User not found")
```

### 3. Use Serialization Utilities

```python
# Good: Use utilities for consistent error handling
from dataknobs_common import serialize, deserialize

data = serialize(user)
restored = deserialize(User, data)

# Less ideal: Direct calls (no error wrapping)
data = user.to_dict()
restored = User.from_dict(data)
```

### 4. Extend, Don't Replace

```python
# Good: Extend common base
class MyRegistry(Registry[Item]):
    def register_item(self, item: Item) -> None:
        self.register(item.id, item)

# Avoid: Reimplementing from scratch
class MyRegistry:
    def __init__(self):
        self._items = {}
```

## Dependencies

- **Python**: >= 3.12
- **Base install**: no third-party dependencies — `[project.dependencies]` is empty, so `pip install dataknobs-common` pulls in nothing else

Backends that need a driver ship it as an extra, lazy-imported at its use site
so importing the module never requires it:

| Extra | Pulls in | Used by |
|---|---|---|
| `dotenv` | `python-dotenv` | the optional `.env` layer in `normalize_postgres_connection_config` |
| `postgres` | `asyncpg` | the PostgreSQL LISTEN/NOTIFY event bus |
| `redis` | `redis` | the Redis pub/sub event bus and the Redis-bucket rate limiter |
| `aws` | `aioboto3` | the shared async AWS session factory |
| `sqs` | `dataknobs-common[aws]` | the SQS-backed event bus |

## Changelog

See [`packages/common/CHANGELOG.md`](https://github.com/kbs-labs/dataknobs/blob/main/packages/common/CHANGELOG.md).

This page no longer carries its own copy: a second changelog is a second thing
to remember to update, and it stopped being updated at 1.4.0 while the package
went on to 3.x.
