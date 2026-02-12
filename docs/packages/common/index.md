# Dataknobs Common

The `dataknobs-common` package provides shared cross-cutting functionality used across all Dataknobs packages.

## Installation

```bash
pip install dataknobs-common
```

Note: This package is automatically installed as a dependency of other Dataknobs packages.

## Overview

This package provides shared cross-cutting functionality used across all dataknobs packages:

- **Exception Framework**: Unified exception hierarchy with context support
- **Registry Pattern**: Generic registries for managing named items
- **Serialization Protocol**: Standard interfaces for to_dict/from_dict patterns
- **Retry**: Configurable retry execution with multiple backoff strategies
- **Transitions**: Stateless transition validation for declarative status graphs
- **Event Bus**: Pub/sub event system for distributed applications

These patterns were extracted from common implementations across multiple packages to reduce duplication and provide consistency.

## Design Philosophy

The common package follows a key design principle:

**"Common provides primitives, packages provide ergonomics"**

This means:
- Common provides simple, explicit base implementations
- Individual packages add convenience wrappers and domain-specific features
- Applications benefit from both consistency and usability

### Example: Registry Pattern

```python
# Common provides the primitive
from dataknobs_common import Registry

registry = Registry[Tool]("tools")
registry.register("calculator", calculator_tool)  # Explicit key required

# LLM package adds ergonomic wrapper
from dataknobs_llm import ToolRegistry

tool_registry = ToolRegistry()
tool_registry.register_tool(calculator_tool)  # Extracts key from tool.name
```

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

Packages extend the common exceptions to create their own hierarchies while maintaining cross-package compatibility:

```python
from dataknobs_common import DataknobsError, OperationError

# Create package-specific base (used in FSM)
FSMError = DataknobsError  # Alias for backward compatibility

# Extend with custom context
class StateExecutionError(OperationError):
    """Raised when state execution fails."""

    def __init__(self, state_name: str, message: str, details=None):
        super().__init__(
            f"State '{state_name}' execution failed: {message}",
            context=details  # Support both 'details' and 'context' parameter names
        )
        self.state_name = state_name  # Custom attribute
```

**Real-world example from FSM package:**

```python
# FSM package uses common exceptions as base classes
from dataknobs_common import (
    DataknobsError,
    ConfigurationError,
    OperationError,
    ResourceError,
    TimeoutError,
    ValidationError,
    ConcurrencyError,
)

# Backward compatibility alias
FSMError = DataknobsError

# Simple extensions
class InvalidConfigurationError(ConfigurationError):
    """Invalid FSM configuration."""
    pass

# Complex extensions with custom attributes
class TransitionError(OperationError):
    """Raised when state transition fails."""

    def __init__(self, from_state: str, to_state: str, message: str, details=None):
        super().__init__(
            f"Cannot transition from '{from_state}' to '{to_state}': {message}",
            context=details
        )
        self.from_state = from_state
        self.to_state = to_state

class CircuitBreakerError(ResourceError):
    """Raised when circuit breaker is open."""

    def __init__(self, resource_id: str, wait_time: float | None = None, details=None):
        msg = f"Circuit breaker open for resource '{resource_id}'"
        if wait_time:
            msg += f" (retry after {wait_time}s)"
        super().__init__(msg, context=details)
        self.resource_id = resource_id
        self.wait_time = wait_time
```

#### Cross-Package Error Handling

Applications can catch all dataknobs exceptions uniformly:

```python
from dataknobs_common import DataknobsError

try:
    # Use any dataknobs package
    result = database.query()
    llm.complete()
    bot.chat()
    fsm.execute()
except DataknobsError as e:
    logger.error(f"Dataknobs error: {e}")
    if e.context:
        logger.error(f"Context: {e.context}")
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

#### Custom Registry Extensions

Packages extend the base registry to add domain-specific features:

```python
from dataknobs_common import Registry

class ToolRegistry(Registry[Tool]):
    """Registry for LLM tools."""

    def __init__(self):
        super().__init__("tools", enable_metrics=True)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool using its name attribute."""
        self.register(
            tool.name,  # Extract key from tool
            tool,
            metadata={"description": tool.description}
        )

    def get_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        return [
            tool for tool in self.list_items()
            if tool.category == category
        ]

    def to_function_definitions(self) -> list[dict]:
        """Convert tools to OpenAI function definitions."""
        return [tool.to_function_definition() for tool in self.list_items()]
```

**Real-world example from LLM package:**

The LLM package's `ToolRegistry` extends `Registry[Tool]` to provide:
- `register_tool(tool)` - Automatically extracts `tool.name` as the key
- `to_function_definitions()` - Converts all tools to OpenAI-compatible format
- `get_by_tags(tags)` - Filters tools by tags
- Built-in metadata tracking for tool versions and descriptions

This demonstrates the design principle: Common provides `register(key, item)`, LLM adds `register_tool(tool)`.

### 3. Serialization Protocol

Standard protocol for objects that can be serialized to/from dictionaries. Unlike Registry and Exceptions, serialization provides a **protocol and utilities** rather than base implementations.

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

# Type checking works automatically
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

#### Complex Serialization Patterns

For enums, datetimes, and nested objects:

```python
from dataknobs_common import Serializable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

@dataclass
class Task:
    title: str
    status: Status
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "status": self.status.value,  # Convert enum to string
            "created_at": self.created_at.isoformat()  # Convert datetime to ISO string
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            title=data["title"],
            status=Status(data["status"]),  # Convert string back to enum
            created_at=datetime.fromisoformat(data["created_at"])  # Parse ISO string
        )
```

**Real-world example from LLM package:**

The LLM package's `LLMConfig` class implements complex serialization:
- Enum conversion (`CompletionMode` enum)
- Dataclass field introspection
- Optional parameter handling
- Filtering of internal attributes
- Special handling for factory defaults

While LLM implements its own serialization logic, it follows the `Serializable` protocol and can use the common utilities for consistency.

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

### 4. Retry

Configurable retry execution with multiple backoff strategies. Supports both sync and async callables, exception filtering, result-based retry, and lifecycle hooks.

#### Basic Usage

```python
from dataknobs_common.retry import RetryExecutor, RetryConfig, BackoffStrategy

# Configure retry behavior
config = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
)

executor = RetryExecutor(config)
result = await executor.execute(fetch_data, url)
```

#### Backoff Strategies

Five strategies are available via the `BackoffStrategy` enum:

- `FIXED` — constant delay between retries
- `LINEAR` — delay increases linearly with each attempt
- `EXPONENTIAL` — delay multiplied by `backoff_multiplier` each attempt
- `JITTER` — exponential with random jitter applied
- `DECORRELATED` — random delay between `initial_delay` and 3x previous delay

#### Exception Filtering

Only retry specific exception types:

```python
config = RetryConfig(
    max_attempts=3,
    retry_on_exceptions=[ConnectionError, TimeoutError],
)
executor = RetryExecutor(config)

# ConnectionError and TimeoutError are retried; others propagate immediately
result = await executor.execute(call_api, endpoint)
```

#### Result-Based Retry

Retry when a result value is unsatisfactory:

```python
config = RetryConfig(
    max_attempts=3,
    retry_on_result=lambda r: r is None,  # Retry if result is None
)
executor = RetryExecutor(config)
result = await executor.execute(poll_status, job_id)
```

#### Lifecycle Hooks

```python
config = RetryConfig(
    on_retry=lambda attempt, exc: logger.warning("Retry %d: %s", attempt, exc),
    on_failure=lambda exc: logger.error("All retries exhausted: %s", exc),
)
```

**Origin:** Extracted from `dataknobs_fsm.patterns.error_recovery` (zero FSM dependency). The FSM module re-exports from common for backward compatibility.

### 5. Transitions

Stateless transition validation for systems that need to enforce valid status transitions without a full state machine framework. Suitable for guarding database writes, API updates, or any operation where an entity moves between named statuses.

This is **not** a state machine — it does not manage state, execute actions, or track lifecycle. For full FSM capabilities, see `dataknobs_fsm`.

#### Basic Usage

```python
from dataknobs_common.transitions import TransitionValidator, InvalidTransitionError

RUN_STATUS = TransitionValidator(
    "run_status",
    {
        "pending":   {"running", "cancelled"},
        "running":   {"completed", "failed", "cancelled"},
        "failed":    {"pending"},       # allow retry
        "completed": set(),             # terminal
        "cancelled": set(),             # terminal
    },
)

RUN_STATUS.validate("pending", "running")       # ok
RUN_STATUS.validate("completed", "pending")     # raises InvalidTransitionError
RUN_STATUS.validate(None, "running")            # ok (skip when current unknown)
```

#### Boolean Check

Use `is_valid()` when you want a boolean instead of an exception:

```python
if RUN_STATUS.is_valid(current_status, target_status):
    update_status(target_status)
```

#### Graph Introspection

```python
# All statuses reachable from a starting point (transitive closure)
reachable = RUN_STATUS.get_reachable("pending")
# {"running", "completed", "failed", "cancelled", "pending"}

# All known statuses
RUN_STATUS.statuses
# {"pending", "running", "completed", "failed", "cancelled"}

# Full transition graph (returns a copy)
RUN_STATUS.allowed_transitions
```

#### InvalidTransitionError

Extends `OperationError` with structured context:

```python
try:
    RUN_STATUS.validate("completed", "running")
except InvalidTransitionError as e:
    e.entity          # "run_status"
    e.current_status  # "completed"
    e.target_status   # "running"
    e.allowed         # set() (terminal — no targets allowed)
    e.context         # {"entity": ..., "current_status": ..., "target_status": ..., "allowed": []}
```

## Package Integration Examples

### FSM Package (Exceptions)

The FSM package migrated from custom exceptions to common exceptions:

```python
from dataknobs_common import (
    DataknobsError,
    ConfigurationError,
    OperationError,
    ResourceError,
    TimeoutError,
    ValidationError,
)

# Backward compatibility
FSMError = DataknobsError

# All FSM exceptions now extend common base classes
class StateExecutionError(OperationError):
    pass

class TransitionError(OperationError):
    pass

class InvalidConfigurationError(ConfigurationError):
    pass
```

**Benefits achieved:**
- ~40 lines of duplicate exception code eliminated
- 100% backward compatible (via alias)
- All 21 FSM exception tests passed with ZERO code changes
- Can now catch FSM errors using `DataknobsError`

### LLM Package (Registry)

The LLM package migrated `ToolRegistry` to extend common `Registry[Tool]`:

```python
from dataknobs_common import Registry

class ToolRegistry(Registry[Tool]):
    """Registry for managing LLM tools."""

    def __init__(self):
        super().__init__("tools", enable_metrics=True)

    def register_tool(self, tool: Tool) -> None:
        """Convenience method that extracts key from tool."""
        self.register(tool.name, tool, metadata={
            "description": tool.description,
            "version": getattr(tool, "version", "1.0")
        })

    def to_function_definitions(self) -> list[dict]:
        """Convert all tools to OpenAI function definitions."""
        return [tool.to_function_definition() for tool in self.list_items()]
```

**Benefits achieved:**
- ~150 lines of registry boilerplate eliminated
- 100% backward compatible
- All 795 LLM tests passed
- Gained built-in metrics and thread-safety

### Bots Package (Registry)

The Bots package can use `CachedRegistry` for bot instance management:

```python
from dataknobs_common import CachedRegistry

class BotRegistry(CachedRegistry[Bot]):
    """Registry with caching for bot instances."""

    def __init__(self, cache_ttl: int = 300):
        super().__init__("bots", cache_ttl=cache_ttl)

    def get_or_create_bot(self, client_id: str, config: BotConfig) -> Bot:
        """Get cached bot or create new one."""
        return self.get_cached(
            client_id,
            factory=lambda: create_bot(config)
        )
```

## Migration Impact

### Code Reduction

Across the ecosystem, common components have eliminated significant duplicate code:

- **Registry**: ~150 lines per registry implementation
- **Exceptions**: ~40-50 lines per package
- **Total eliminated**: ~190 lines across 2 packages (FSM, LLM)
- **Potential**: 400-500 lines when all packages migrate

### Consistency Gained

**Before:**
- Each package had its own base exception class
- Each registry implemented its own boilerplate
- No cross-package exception handling

**After:**
- All packages use `DataknobsError` base
- All registries extend `Registry[T]`
- Can catch `DataknobsError` for any dataknobs exception
- Unified pattern across ecosystem

### Testing Results

All migrations achieved 100% backward compatibility:

| Package | Component | Tests | Result | Code Changes |
|---------|-----------|-------|--------|--------------|
| LLM | Registry | 21/21 | ✅ Pass | Minimal (API updates) |
| LLM | All tests | 795/795 | ✅ Pass | - |
| FSM | Exceptions | 21/21 | ✅ Pass | Zero changes |
| FSM | All tests | All | ✅ Pass | - |

## When to Use Common Components

### FSM Package (Retry Re-export)

The retry primitives were extracted from the FSM package's `error_recovery` module. The FSM module now re-exports from common for backward compatibility:

```python
# Both import paths work identically
from dataknobs_common.retry import RetryExecutor, RetryConfig, BackoffStrategy
from dataknobs_fsm.patterns.error_recovery import RetryExecutor, RetryConfig, BackoffStrategy
```

### Use Registry When:
- Managing collections of named objects
- Need thread-safe access
- Want built-in metrics
- Caching with TTL is beneficial
- Async operations are required

### Use Common Exceptions When:
- Creating package-specific errors
- Need context-rich error information
- Want cross-package error handling
- Maintaining backward compatibility with aliases

### Use Serialization Protocol When:
- Creating new serializable classes
- Want type safety for serialization
- Need consistent error handling
- Working with lists of serializable objects

### Use Retry When:
- Calling external APIs or services that may fail transiently
- Need configurable backoff strategies (exponential, jitter, etc.)
- Want exception filtering (only retry specific exception types)
- Need result-based retry (retry when result is unsatisfactory)
- Want lifecycle hooks for observability (on_retry, on_failure)

### Use Transitions When:
- Guarding database status updates (e.g. order lifecycle, job states)
- Need to validate that a status change is allowed before writing
- Want graph introspection (reachable statuses, allowed targets)
- Need a lightweight alternative to a full FSM
- Do NOT need state management, actions, or lifecycle tracking

### Use Event Bus When:
- Need decoupled communication between components
- Want event-driven cache invalidation
- Building distributed applications
- Need to switch between in-memory, PostgreSQL, or Redis backends

## Best Practices

### 1. Extend, Don't Replace

Create package-specific extensions that add features:

```python
# Good: Extends common base
class ToolRegistry(Registry[Tool]):
    def register_tool(self, tool: Tool) -> None:
        self.register(tool.name, tool)

# Avoid: Completely custom implementation
class ToolRegistry:
    def __init__(self):
        self.tools = {}  # Loses common benefits
```

### 2. Maintain Backward Compatibility

Use aliases when migrating:

```python
# Preserve old API
from dataknobs_common import DataknobsError

# Backward compatibility alias
FSMError = DataknobsError

# Existing code continues to work:
# raise FSMError("message")
```

### 3. Add Ergonomics in Packages

Follow the design principle - common provides primitives, packages add convenience:

```python
# Common primitive
registry.register("key", item)

# Package ergonomics
registry.register_tool(tool)  # Extracts key automatically
```

### 4. Use Context in Exceptions

Provide structured context for better debugging:

```python
# Good: Rich context
raise StateExecutionError(
    state_name="processing",
    message="Database connection failed",
    details={
        "database": "prod_db",
        "retry_count": 3,
        "last_error": str(db_error)
    }
)

# Basic: Still useful
raise ValidationError("Invalid input")
```

## API Reference

For complete API documentation, see the [Common API Reference](api.md).

## Package Dependencies

The Common package requires:
- Python 3.10+
- No external dependencies (uses only standard library)

The Common package is used by:
- **dataknobs-fsm**: Finite State Machine framework
- **dataknobs-llm**: LLM prompt and conversation management
- **dataknobs-bots**: AI bot framework
- **dataknobs-data**: Data storage and querying
- **dataknobs-structures**: Core data structures
- **dataknobs-utils**: Utility functions
- **dataknobs-config**: Configuration management
- **dataknobs-xization**: Text processing

## Next Steps

- See [Complete API Reference](api.md) for detailed documentation
- Learn about [Event Bus](events.md) for pub/sub event handling
- Explore [FSM Package](../fsm/index.md) for exception usage examples
- Learn about [LLM Package](../llm/index.md) for registry usage examples
- Check [Development Guide](../../development/index.md) for contributing guidelines
