# Common API Reference

Complete API reference for the `dataknobs-common` package.

> **📖 Also see:** [Auto-generated API Reference](../../api/reference/common.md) - Complete documentation from source code docstrings

This page provides curated examples and usage patterns. The auto-generated reference provides exhaustive technical documentation with all methods, parameters, and type annotations.

---

## Module Overview

The `dataknobs-common` package provides three main modules:

- **`dataknobs_common.exceptions`** - Exception hierarchy with context support
- **`dataknobs_common.registry`** - Generic registry implementations
- **`dataknobs_common.serialization`** - Serialization protocols and utilities

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
DataknobsError(message: str, context: dict[str, Any] | None = None)
```

**Parameters:**
- `message` (str): Error message
- `context` (dict[str, Any] | None): Optional context dictionary with additional error details

**Attributes:**
- `message` (str): The error message
- `context` (dict[str, Any] | None): Context dictionary if provided
- `details` (property): Alias for `context` (for backward compatibility)

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

# Access context
try:
    operation()
except DataknobsError as e:
    print(e.message)  # "Operation failed"
    print(e.context)  # {"operation": "save", "item_id": "123"}
    print(e.details)  # Same as context (alias)
```

### Standard Exceptions

All standard exceptions extend `DataknobsError` and follow the same constructor pattern.

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
    enable_metrics: bool = False,
    allow_override: bool = False
)
```

**Parameters:**
- `name` (str): Registry name (for logging and metrics)
- `enable_metrics` (bool): Enable metrics tracking (default: False)
- `allow_override` (bool): Allow overriding existing keys (default: False)

**Methods:**

##### `register(key: str, item: T, metadata: dict[str, Any] | None = None) -> None`

Register an item with a key.

**Parameters:**
- `key` (str): Unique identifier for the item
- `item` (T): The item to register
- `metadata` (dict[str, Any] | None): Optional metadata

**Raises:**
- `ValueError`: If key already exists and `allow_override` is False

**Example:**
```python
from dataknobs_common import Registry

registry = Registry[str]("messages")
registry.register("greeting", "Hello, world!")
registry.register("farewell", "Goodbye!", metadata={"lang": "en"})
```

##### `get(key: str) -> T`

Get an item by key.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `T`: The registered item

**Raises:**
- `KeyError`: If key not found

**Example:**
```python
message = registry.get("greeting")  # "Hello, world!"
```

##### `get_or_none(key: str) -> T | None`

Get an item by key, returning None if not found.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `T | None`: The registered item or None

**Example:**
```python
message = registry.get_or_none("greeting")  # "Hello, world!"
missing = registry.get_or_none("unknown")   # None
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
- `KeyError`: If key not found

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

##### `get_metadata(key: str) -> dict[str, Any] | None`

Get metadata for a key.

**Parameters:**
- `key` (str): Item key

**Returns:**
- `dict[str, Any] | None`: Metadata or None

**Example:**
```python
meta = registry.get_metadata("farewell")  # {"lang": "en"}
```

**Magic Methods:**

```python
len(registry)           # Same as count()
key in registry         # Same as has(key)
for key in registry     # Iterate over keys
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
    enable_metrics: bool = True,
    allow_override: bool = False
)
```

**Parameters:**
- `name` (str): Registry name
- `cache_ttl` (int): Cache TTL in seconds (default: 300)
- `enable_metrics` (bool): Enable metrics tracking (default: True)
- `allow_override` (bool): Allow overriding existing keys (default: False)

**Additional Methods:**

##### `get_cached(key: str, factory: Callable[[], T]) -> T`

Get cached item or create with factory.

**Parameters:**
- `key` (str): Cache key
- `factory` (Callable[[], T]): Factory function to create item if not cached

**Returns:**
- `T`: Cached or newly created item

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
- `dict[str, Any]`: Statistics including hits, misses, hit_rate

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
    enable_metrics: bool = False,
    allow_override: bool = False
)
```

**Methods:**

All methods are async versions of the base Registry methods:

```python
await registry.register(key, item, metadata=None)
item = await registry.get(key)
item = await registry.get_or_none(key)
exists = await registry.has(key)
item = await registry.unregister(key)
items = await registry.list_items()
keys = await registry.list_keys()
pairs = await registry.items()
count = await registry.count()
await registry.clear()
meta = await registry.get_metadata(key)
```

**Example:**
```python
from dataknobs_common import AsyncRegistry

registry = AsyncRegistry[Resource]("resources")

await registry.register("db", db_resource)
resource = await registry.get("db")
count = await registry.count()
```

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
    def from_dict(cls, data: dict) -> Self: ...
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

print(__version__)  # "1.0.1"
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
```

### Module Imports

```python
# Import entire modules
from dataknobs_common import exceptions
from dataknobs_common import registry
from dataknobs_common import serialization
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
    logger.error(f"Dataknobs error: {e.message}")
    if e.context:
        logger.error(f"Context: {e.context}")
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
from dataknobs_common import Registry

registry = Registry[Tool]("tools")

try:
    tool = registry.get("calculator")
except KeyError:
    # Handle missing key
    tool = default_tool

# Or use get_or_none
tool = registry.get_or_none("calculator")
if tool is None:
    tool = default_tool
```

### Serialization Error Handling

```python
from dataknobs_common import deserialize, SerializationError

try:
    user = deserialize(User, data)
except SerializationError as e:
    logger.error(f"Failed to deserialize: {e.message}")
    logger.error(f"Error context: {e.context}")
    # Handle error appropriately
```

## Advanced Usage Patterns

### Custom Registry with Validation

```python
from dataknobs_common import Registry

class ValidatedRegistry(Registry[T]):
    """Registry with validation on registration."""

    def register(self, key: str, item: T, metadata: dict | None = None) -> None:
        # Validate before registering
        if not self._validate(item):
            raise ValueError(f"Item validation failed: {key}")
        super().register(key, item, metadata)

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

The Common package has minimal dependencies:

- **Python**: >= 3.10
- **Standard library only**: No external dependencies

## Changelog

### Version 1.0.1
- Added Registry, CachedRegistry, AsyncRegistry implementations
- Added comprehensive Exception hierarchy
- Added Serialization protocol and utilities
- Initial production release

### Version 1.0.0
- Initial release with basic version management
