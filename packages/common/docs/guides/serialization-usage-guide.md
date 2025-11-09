# Serialization Guide: Common Serialization Framework

## Table of Contents
1. [Overview](#overview)
2. [When to Use Common Serialization](#when-to-use-common-serialization)
3. [The Serializable Protocol](#the-serializable-protocol)
4. [Basic Usage](#basic-usage)
5. [Common Patterns](#common-patterns)
6. [Advanced Patterns](#advanced-patterns)
7. [Integration with Existing Code](#integration-with-existing-code)
8. [Testing Serialization](#testing-serialization)
9. [Best Practices](#best-practices)

---

## Overview

The `dataknobs_common.serialization` module provides a standard protocol and utilities for serializing objects to/from dictionaries. Unlike the Registry and Exception frameworks which provide base implementations, the serialization module provides:

1. **Protocol Definition**: `Serializable` protocol for type checking
2. **Utility Functions**: Convenience wrappers with consistent error handling
3. **Type Safety**: Runtime `isinstance()` checks
4. **Standard Pattern**: Consistent approach across all dataknobs packages

### What This Is NOT

- ❌ **NOT a serialization library** like pickle/json
- ❌ **NOT automatic serialization** like dataclasses.asdict()
- ❌ **NOT a base class** to inherit from
- ❌ **NOT an ORM** or database mapper

### What This IS

- ✅ **A protocol** to follow for serializable classes
- ✅ **Utility functions** for consistent error handling
- ✅ **Type checking** to verify serialization support
- ✅ **A standard pattern** for the dataknobs ecosystem

---

## When to Use Common Serialization

### Use Common Serialization When:

1. **Creating new data classes** that need dict conversion
2. **Building APIs** that return/accept dictionaries
3. **Implementing storage** (databases, files, caches)
4. **Working with JSON/YAML** configuration or data
5. **Building SDKs** with serializable models

### Examples Where It's Useful:

```python
# Configuration objects
class APIConfig:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig": ...

# Data models
class User:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "User": ...

# Storage objects
class CacheEntry:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry": ...
```

### When NOT to Use:

- Simple dictionaries (no need for protocol)
- Objects that won't be serialized
- External library objects (can't modify them)
- Performance-critical hot paths (direct calls may be faster)

---

## The Serializable Protocol

### Protocol Definition

```python
from typing import Protocol, runtime_checkable, Dict, Any, Type, TypeVar

T = TypeVar('T')

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create object from dictionary representation."""
        ...
```

### What the Protocol Means

Any class that implements both methods automatically satisfies the protocol:

```python
class MyClass:
    def to_dict(self) -> dict:
        return {"data": "value"}

    @classmethod
    def from_dict(cls, data: dict) -> "MyClass":
        return cls()

# Automatically satisfies Serializable protocol!
from dataknobs_common.serialization import Serializable
assert isinstance(MyClass(), Serializable)  # True
```

**No inheritance needed!** Just implement the two methods.

---

## Basic Usage

### Step 1: Implement to_dict() and from_dict()

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Point:
    x: int
    y: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point":
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"]
        )
```

### Step 2: Use Utility Functions

```python
from dataknobs_common.serialization import serialize, deserialize

# Serialize
point = Point(10, 20)
data = serialize(point)  # {"x": 10, "y": 20}

# Deserialize
restored = deserialize(Point, data)
assert restored.x == 10 and restored.y == 20
```

### Step 3: Type Checking

```python
from dataknobs_common.serialization import is_serializable, is_deserializable

# Runtime type checking
assert is_serializable(point)        # Has to_dict
assert is_deserializable(Point)      # Has from_dict

# Use in conditional logic
def save_object(obj):
    if is_serializable(obj):
        data = serialize(obj)
        storage.save(data)
    else:
        raise TypeError(f"{type(obj).__name__} is not serializable")
```

---

## Common Patterns

### Pattern 1: Simple Dataclass

For simple classes with no nested objects:

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            name=data["name"],
            email=data["email"],
            age=data["age"]
        )
```

**Pro tip**: For very simple dataclasses, you could use `dataclasses.asdict()`:

```python
from dataclasses import asdict

def to_dict(self) -> dict:
    return asdict(self)
```

But be careful - this won't handle nested objects, enums, or datetime properly!

### Pattern 2: Optional Fields

Handle optional fields with `.get()` and defaults:

```python
@dataclass
class UserProfile:
    name: str
    bio: str | None = None
    avatar_url: str | None = None

    def to_dict(self) -> dict:
        result = {"name": self.name}
        if self.bio:
            result["bio"] = self.bio
        if self.avatar_url:
            result["avatar_url"] = self.avatar_url
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            name=data["name"],
            bio=data.get("bio"),  # Returns None if missing
            avatar_url=data.get("avatar_url")
        )
```

### Pattern 3: Lists of Objects

Use `serialize_list` and `deserialize_list`:

```python
from dataknobs_common.serialization import serialize_list, deserialize_list

@dataclass
class Team:
    name: str
    members: list[User]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "members": serialize_list(self.members)  # Serialize each user
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Team":
        return cls(
            name=data["name"],
            members=deserialize_list(User, data["members"])  # Deserialize each user
        )
```

### Pattern 4: Nested Objects

For nested serializable objects:

```python
@dataclass
class Address:
    street: str
    city: str

    def to_dict(self) -> dict:
        return {"street": self.street, "city": self.city}

    @classmethod
    def from_dict(cls, data: dict) -> "Address":
        return cls(street=data["street"], city=data["city"])


@dataclass
class Person:
    name: str
    address: Address

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "address": serialize(self.address)  # Serialize nested object
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Person":
        return cls(
            name=data["name"],
            address=deserialize(Address, data["address"])  # Deserialize nested
        )
```

---

## Advanced Patterns

### Pattern 1: Enum Handling

Convert enums to/from strings:

```python
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class Task:
    title: str
    status: Status

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "status": self.status.value  # Enum to string
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            title=data["title"],
            status=Status(data["status"])  # String to Enum
        )
```

### Pattern 2: Datetime Handling

Use ISO format for datetime:

```python
from datetime import datetime

@dataclass
class Event:
    name: str
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat()  # Datetime to ISO string
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"])  # ISO string to datetime
        )
```

### Pattern 3: Backward Compatibility

Handle schema evolution with version checks and defaults:

```python
@dataclass
class Document:
    title: str
    content: str
    version: int = 2  # Current schema version
    tags: list[str] = field(default_factory=list)  # Added in v2

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "title": self.title,
            "content": self.content,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        version = data.get("version", 1)  # Default to v1 if missing

        if version == 1:
            # V1 didn't have tags
            return cls(
                title=data["title"],
                content=data["content"],
                version=2,  # Upgrade to v2
                tags=[]  # Default empty tags
            )
        else:
            # V2 format
            return cls(
                title=data["title"],
                content=data["content"],
                version=data["version"],
                tags=data.get("tags", [])
            )
```

### Pattern 4: Custom Validation

Add validation in from_dict:

```python
from dataknobs_common.serialization import SerializationError

@dataclass
class Email:
    address: str

    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError(f"Invalid email: {self.address}")

    def to_dict(self) -> dict:
        return {"address": self.address}

    @classmethod
    def from_dict(cls, data: dict) -> "Email":
        try:
            return cls(address=data["address"])
        except (KeyError, ValueError) as e:
            raise SerializationError(
                f"Failed to deserialize Email: {e}",
                context={"data": data, "error": str(e)}
            ) from e
```

### Pattern 5: Filtering Sensitive Data

Exclude sensitive fields from serialization:

```python
@dataclass
class User:
    username: str
    email: str
    password_hash: str  # Sensitive!

    def to_dict(self, include_sensitive: bool = False) -> dict:
        result = {
            "username": self.username,
            "email": self.email
        }
        if include_sensitive:
            result["password_hash"] = self.password_hash
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            username=data["username"],
            email=data["email"],
            password_hash=data.get("password_hash", "")
        )
```

---

## Integration with Existing Code

### Using with Existing LLM Classes

LLM classes like `LLMConfig` and `ConversationNode` already implement the Serializable protocol. You can use the utilities with them:

```python
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_common.serialization import serialize, deserialize

# Old way (still works)
config = LLMConfig(provider="openai", model="gpt-4")
data = config.to_dict()
restored = LLMConfig.from_dict(data)

# New way (using utilities)
config = LLMConfig(provider="openai", model="gpt-4")
data = serialize(config)  # Better error handling
restored = deserialize(LLMConfig, data)  # Better error handling
```

**Benefits of using utilities:**
- Consistent `SerializationError` exceptions with context
- Type checking: `assert is_serializable(config)`
- Standardized error messages

### Light Touch Adoption

You don't need to migrate existing code. Just use utilities in **new code**:

```python
# Existing code - leave unchanged
def old_save_config(config: LLMConfig):
    data = config.to_dict()
    storage.save(data)

# New code - use utilities
from dataknobs_common.serialization import serialize

def new_save_config(config: LLMConfig):
    data = serialize(config)  # Better error handling!
    storage.save(data)
```

---

## Testing Serialization

### Basic Round-Trip Test

```python
def test_serialization_round_trip():
    """Test that object survives serialization/deserialization."""
    original = User(name="Alice", email="alice@example.com", age=30)

    # Serialize
    data = serialize(original)

    # Deserialize
    restored = deserialize(User, data)

    # Verify
    assert restored.name == original.name
    assert restored.email == original.email
    assert restored.age == original.age
```

### Test Protocol Compliance

```python
from dataknobs_common.serialization import Serializable

def test_implements_serializable_protocol():
    """Test that class implements Serializable protocol."""
    user = User(name="Bob", email="bob@example.com", age=25)

    # Runtime type check
    assert isinstance(user, Serializable)

    # Has required methods
    assert hasattr(user, "to_dict")
    assert hasattr(User, "from_dict")
```

### Test Error Handling

```python
from dataknobs_common.serialization import SerializationError

def test_serialization_error_handling():
    """Test that invalid data raises SerializationError."""

    # Missing required field
    with pytest.raises(SerializationError):
        deserialize(User, {"name": "Alice"})  # Missing email and age

    # Invalid type
    class NotSerializable:
        pass

    with pytest.raises(SerializationError):
        serialize(NotSerializable())  # No to_dict method
```

### Test List Serialization

```python
from dataknobs_common.serialization import serialize_list, deserialize_list

def test_list_serialization():
    """Test serializing list of objects."""
    users = [
        User("Alice", "alice@example.com", 30),
        User("Bob", "bob@example.com", 25),
    ]

    # Serialize list
    data_list = serialize_list(users)
    assert len(data_list) == 2

    # Deserialize list
    restored = deserialize_list(User, data_list)
    assert len(restored) == 2
    assert restored[0].name == "Alice"
    assert restored[1].name == "Bob"
```

---

## Best Practices

### 1. Always Implement Both Methods

```python
# ✅ Good - implements both methods
class Good:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "Good": ...

# ❌ Bad - only implements one
class Bad:
    def to_dict(self) -> dict: ...
    # Missing from_dict!
```

### 2. Use Type Hints

```python
# ✅ Good - clear types
def to_dict(self) -> Dict[str, Any]:
    return {"field": self.value}

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "MyClass":
    return cls(value=data["field"])

# ❌ Bad - no type hints
def to_dict(self):
    return {"field": self.value}

@classmethod
def from_dict(cls, data):
    return cls(value=data["field"])
```

### 3. Handle Missing Fields Gracefully

```python
# ✅ Good - uses .get() with defaults
@classmethod
def from_dict(cls, data: dict) -> "MyClass":
    return cls(
        required=data["required"],  # Will raise KeyError if missing (good!)
        optional=data.get("optional", "default")  # Returns default if missing
    )

# ❌ Bad - no error handling
@classmethod
def from_dict(cls, data: dict) -> "MyClass":
    return cls(
        required=data["required"],
        optional=data["optional"]  # KeyError if missing!
    )
```

### 4. Document Serialization Format

```python
@dataclass
class User:
    """User model.

    Serialization format:
        {
            "name": str,
            "email": str,
            "age": int,
            "metadata": dict (optional)
        }

    Example:
        >>> user = User(name="Alice", email="alice@example.com", age=30)
        >>> data = user.to_dict()
        >>> data
        {'name': 'Alice', 'email': 'alice@example.com', 'age': 30}
    """
    name: str
    email: str
    age: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # ...
```

### 5. Raise SerializationError for Failures

```python
from dataknobs_common.serialization import SerializationError

@classmethod
def from_dict(cls, data: dict) -> "User":
    try:
        return cls(
            name=data["name"],
            email=data["email"],
            age=data["age"]
        )
    except (KeyError, ValueError, TypeError) as e:
        raise SerializationError(
            f"Failed to deserialize User: {e}",
            context={"data": data, "error": str(e)}
        ) from e
```

### 6. Keep Serialization Simple

```python
# ✅ Good - straightforward mapping
def to_dict(self) -> dict:
    return {
        "id": self.id,
        "name": self.name
    }

# ❌ Bad - complex logic in serialization
def to_dict(self) -> dict:
    # Don't do complex transformations here!
    if self.status == "active":
        computed_value = self._expensive_calculation()
    else:
        computed_value = None

    return {
        "id": self.id,
        "computed": computed_value  # This belongs elsewhere
    }
```

---

## Summary

The `dataknobs_common.serialization` module provides:

1. **Serializable Protocol**: Define what it means to be serializable
2. **Utility Functions**: `serialize()`, `deserialize()`, `serialize_list()`, `deserialize_list()`
3. **Type Checking**: `isinstance(obj, Serializable)`, `is_serializable()`, `is_deserializable()`
4. **Error Handling**: Consistent `SerializationError` with context

### Key Takeaways

- ✅ Use for NEW classes that need dict serialization
- ✅ Provides standard pattern across dataknobs ecosystem
- ✅ Gives runtime type checking and better error messages
- ❌ Doesn't reduce code (you still implement to_dict/from_dict)
- ❌ Doesn't provide automatic serialization
- ❌ Not a replacement for existing working code

### Next Steps

1. Use `Serializable` protocol for all new data classes
2. Import and use utilities in new code
3. Consider light-touch adoption in existing code (optional)
4. Document serialization format in docstrings
5. Write round-trip tests for all serializable classes
