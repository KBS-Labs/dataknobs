# dataknobs-common

Common utilities and base classes for all dataknobs packages.

## Installation

```bash
pip install dataknobs-common
```

## Overview

This package provides shared cross-cutting functionality used across all dataknobs packages:

- **Exception Framework**: Unified exception hierarchy with context support
- **Registry Pattern**: Generic registries for managing named items
- **Serialization Protocol**: Standard interfaces for to_dict/from_dict patterns

These patterns were extracted from common implementations across multiple packages to reduce duplication and provide consistency.

## Features

### 1. Exception Framework

A unified exception hierarchy that all dataknobs packages extend. Supports both simple exceptions and context-rich exceptions with detailed error information.

#### Basic Usage

```python
from dataknobs_common import DataknobsError, ValidationError, NotFoundError

# Simple exception
raise ValidationError("Invalid email format")

# Context-rich exception
raise NotFoundError(
    "User not found",
    context={"user_id": "123", "searched_in": "users_table"}
)

# Catch any dataknobs error
try:
    operation()
except DataknobsError as e:
    print(f"Error: {e}")
    if e.context:
        print(f"Context: {e.context}")
```

#### Available Exception Types

- `DataknobsError` - Base exception for all packages
- `ValidationError` - Data validation failures
- `ConfigurationError` - Configuration issues
- `ResourceError` - Resource acquisition/management failures
- `NotFoundError` - Item lookup failures
- `OperationError` - General operation failures
- `ConcurrencyError` - Concurrent operation conflicts
- `SerializationError` - Serialization/deserialization failures
- `TimeoutError` - Operation timeout errors

#### Package-Specific Extensions

```python
from dataknobs_common import DataknobsError

class MyPackageError(DataknobsError):
    """Base exception for mypackage."""
    pass

class SpecificError(MyPackageError):
    """Specific error with custom context."""
    def __init__(self, item_id: str, message: str):
        super().__init__(
            f"Item '{item_id}': {message}",
            context={"item_id": item_id}
        )
```

### 2. Registry Pattern

Generic, thread-safe registries for managing collections of named items. Includes variants for caching and async support.

#### Basic Registry

```python
from dataknobs_common import Registry

# Create a registry for tools
registry = Registry[Tool]("tools")

# Register items
registry.register("calculator", calculator_tool)
registry.register("search", search_tool, metadata={"version": "1.0"})

# Retrieve items
tool = registry.get("calculator")

# Check existence
if registry.has("search"):
    print("Search tool available")

# List all items
for key, tool in registry.items():
    print(f"{key}: {tool}")

# Get count
print(f"Registry has {registry.count()} tools")
```

#### Cached Registry

For items that should be cached with automatic TTL-based expiration:

```python
from dataknobs_common import CachedRegistry

# Create registry with 5-minute cache
registry = CachedRegistry[Bot]("bots", cache_ttl=300)

# Get or create with factory
bot = registry.get_cached(
    "client1",
    factory=lambda: create_bot("client1")
)

# Get cache statistics
stats = registry.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Invalidate cache
registry.invalidate_cache("client1")  # Single item
registry.invalidate_cache()  # All items
```

#### Async Registry

For async contexts:

```python
from dataknobs_common import AsyncRegistry

registry = AsyncRegistry[Resource]("resources")

# All operations are async
await registry.register("db", db_resource)
resource = await registry.get("db")
count = await registry.count()
```

#### Plugin Registry

For managing plugins with factory support, defaults, and lazy instantiation:

```python
from dataknobs_common import PluginRegistry

# Define a base class
class Handler:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

class DefaultHandler(Handler):
    pass

class CustomHandler(Handler):
    pass

# Create registry with default factory
registry = PluginRegistry[Handler]("handlers", default_factory=DefaultHandler)

# Register plugins
registry.register("custom", CustomHandler)

# Get instances (lazy creation with caching)
handler = registry.get("custom", config={"timeout": 30})
default = registry.get("unknown", config={})  # Uses default

# Async factory support
async def create_async_handler(name, config):
    handler = AsyncHandler(name, config)
    await handler.initialize()
    return handler

registry.register("async", create_async_handler)
handler = await registry.get_async("async", config={"url": "..."})
```

##### PluginRegistry Features

```python
# Bulk registration
registry.bulk_register({
    "handler1": Handler1,
    "handler2": Handler2,
})

# Check registration
if registry.is_registered("custom"):
    print("Custom handler available")

# List all registered plugins
keys = registry.list_keys()

# Clear cached instances
registry.clear_cache("custom")  # Single
registry.clear_cache()  # All

# Get factory without creating instance
factory = registry.get_factory("custom")

# Set default after init
registry.set_default_factory(NewDefault)
```

#### Custom Registry Extensions

```python
from dataknobs_common import Registry

class ToolRegistry(Registry[Tool]):
    """Registry for LLM tools."""

    def __init__(self):
        super().__init__("tools", enable_metrics=True)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with metadata."""
        self.register(
            tool.name,
            tool,
            metadata={"description": tool.description}
        )

    def get_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        return [
            tool for tool in self.list_items()
            if tool.category == category
        ]
```

### 3. Serialization Protocol

Standard protocol for objects that can be serialized to/from dictionaries.

#### Define Serializable Classes

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

#### Serialization Utilities

```python
from dataknobs_common import serialize, deserialize, serialize_list, deserialize_list

# Serialize single object
user = User("Alice", "alice@example.com")
data = serialize(user)
# {'name': 'Alice', 'email': 'alice@example.com'}

# Deserialize
restored_user = deserialize(User, data)

# Serialize list
users = [User("Alice", "a@ex.com"), User("Bob", "b@ex.com")]
data_list = serialize_list(users)

# Deserialize list
restored_users = deserialize_list(User, data_list)
```

#### Type Checking

```python
from dataknobs_common import is_serializable, is_deserializable

# Check if object can be serialized
if is_serializable(my_object):
    data = serialize(my_object)

# Check if class can deserialize
if is_deserializable(MyClass):
    obj = deserialize(MyClass, data)
```

## Integration with Dataknobs Packages

This package is designed to be imported and extended by all dataknobs packages:

```python
# In dataknobs_data package
from dataknobs_common import DataknobsError, NotFoundError

class DataknobsDataError(DataknobsError):
    """Base exception for data package."""
    pass

class RecordNotFoundError(NotFoundError):
    """Record not found in database."""
    pass
```

```python
# In dataknobs_llm package
from dataknobs_common import Registry

class ToolRegistry(Registry[Tool]):
    """Registry for LLM tools."""

    def to_function_definitions(self) -> list[dict]:
        """Convert tools to function definitions."""
        return [tool.to_function_definition() for tool in self.list_items()]
```

## Migration Guide

### For Package Developers

If your package has custom exceptions, consider extending from `dataknobs_common`:

**Before:**
```python
# packages/mypackage/exceptions.py
class MyPackageError(Exception):
    pass
```

**After:**
```python
# packages/mypackage/exceptions.py
from dataknobs_common import DataknobsError

class MyPackageError(DataknobsError):
    pass
```

### For Application Developers

You can now catch all dataknobs exceptions uniformly:

```python
from dataknobs_common import DataknobsError

try:
    # Use any dataknobs package
    result = database.query()
    llm.complete()
    bot.chat()
except DataknobsError as e:
    logger.error(f"Dataknobs error: {e}")
    if e.context:
        logger.error(f"Context: {e.context}")
```

## Why Common Package?

After building out core packages (data, llm, fsm, bots), we identified several patterns that were:

1. **Implemented multiple times** - Exception hierarchies in 4+ packages
2. **Highly similar** - Registry pattern ~100-150 lines duplicated 3 times
3. **Genuinely cross-cutting** - Used across domain boundaries

Extracting these patterns to `dataknobs-common` provides:

- **Reduced duplication** - Single source of truth
- **Consistency** - Same patterns everywhere
- **Type safety** - Shared protocols and interfaces
- **Better error handling** - Unified exception catching
- **Ecosystem coherence** - Packages feel integrated

## API Reference

See module docstrings for detailed API documentation:

- `dataknobs_common.exceptions` - Exception hierarchy
- `dataknobs_common.registry` - Registry implementations
- `dataknobs_common.serialization` - Serialization protocols

## Dependencies

- Python 3.10+
- No external dependencies (uses only standard library)

## License

See LICENSE file in the root repository.
