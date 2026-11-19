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
9. [JSON Safety Utilities](#json-safety-utilities)
10. [Best Practices](#best-practices)

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

#### `jsonify` — recursive enum-to-value normalisation

When you already hold a JSON-shaped structure (e.g. the output of
`dataclasses.asdict`) whose only non-JSON-native values are enums, the
`jsonify` utility replaces every `Enum` member with its `.value`,
recursing through `dict` / `list` / `tuple` containers — so you don't
hand-write the conversion per field:

```python
from dataknobs_common import jsonify

jsonify({"status": Status.ACTIVE, "tags": [Status.PENDING]})
# {"status": "active", "tags": ["pending"]}
```

`jsonify` is *lossless and narrow*: it only normalises enums and passes
everything else (callables, `type` objects, `set`s) through untouched. It
is the engine behind
[`StructuredConfig.to_json_dict()`](structured-config.md#to_json_dict).
For lossy best-effort JSON safety that also converts dataclasses /
`Serializable` objects to dicts and **drops** anything non-representable,
use `sanitize_for_json` instead — it likewise normalises enums to their
`.value`, but additionally walks dataclasses and drops live objects
(callables, sockets, tasks) that `jsonify` would leave in place.

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

## JSON Safety Utilities

The serialization module also provides utilities for preparing arbitrary data structures
for JSON serialization. There are two policies over one walk, for two purposes:

| | `sanitize_for_json` | `as_data` |
|---|---|---|
| for | state that may hold live objects (functions, asyncio tasks), saved as far as it can be | a value that must arrive whole, such as a reply made of nested frozen dataclasses |
| order | `Enum`, non-finite float, primitives, `dict`, list/tuple, dataclass, `to_dict()` | `using`, `Enum`, non-finite float, primitives, `to_dict()`, dataclass, any `Mapping`, list/tuple, set |
| a mapping | `dict` only; keys kept as they are | any `Mapping`; keys written as text the way `json.dumps` writes them, two keys written the same refused |
| NaN, infinity | dropped, reported by `on_drop`; as a key, its entry dropped | refused; as a key, written as text (`"NaN"`, `"Infinity"`) |
| a set | dropped | a sorted list |
| a `to_dict()` that raises | dropped | refused, the original as `__cause__` |
| no rule | dropped, reported by `on_drop` | refused at once, naming the path |
| a container that holds itself | dropped where it repeats | refused |
| paths | `notes.tare`, `items[1]`, `<root>` | `value.notes['tare']`, `value.items[1]` |

`validate_json_safe` reports what `sanitize_for_json` would drop, without dropping it.

### sanitize_for_json()

Recursively traverses a value and drops anything that isn't JSON-serializable.
It keeps `None`, booleans, numbers, strings, dicts and lists; converts tuples
to lists, dataclasses to dicts (field by field), an `Enum` to its value, and an
object with a `to_dict()` returning a dict to that dict. A dataclass is
converted field by field even when it defines its own `to_dict()`. Anything
else is dropped, including sets, bytes and `datetime` values.

A NaN or infinite float is dropped too, and reported as a `non-finite float`
(`reading.value (non-finite float nan)`). `json.dumps` would write it as `NaN`
or `Infinity`, which is not JSON: `JSON.parse` and a Postgres `jsonb` column
refuse it, though Python's own `json.loads` reads it back.

A dict keeps the keys `json.dumps` can write (`str`, `int`, a finite `float`,
`bool`, `None`), and an `Enum` key becomes its value. An entry whose key is
anything else, a tuple for instance, is dropped and reported like any other
drop. So is an entry whose key is a NaN or infinite float
(`a.inf (non-finite float key inf)`): `json.dumps` refuses that key under
`allow_nan=False`, and two NaN keys are two entries it would write as one.

A dropped value is left out of the dict, list or dataclass that holds it. A
value dropped at the top level has no container to be left out of, so
`sanitize_for_json` returns `None` for it (or raises, with `on_drop="error"`).

A dict, list or dataclass that holds itself is dropped where it repeats, and
reported as a `cycle`. Only the containers on the current path count, so a
value reached twice by different routes is kept both times.

```python
from dataknobs_common.serialization import sanitize_for_json

# Basic usage — silently drops non-serializable values (DEBUG log)
data = {"name": "Alice", "callback": some_function, "count": 42}
safe = sanitize_for_json(data)
# {"name": "Alice", "count": 42}

# Warn mode — WARNING log with full key path for each drop
safe = sanitize_for_json(data, on_drop="warn")
# WARNING: Dropping non-serializable 'callback' (type=function)

# Error mode — raises SerializationError listing all dropped paths
from dataknobs_common.exceptions import SerializationError

try:
    safe = sanitize_for_json(data, on_drop="error")
except SerializationError as e:
    print(e.context["dropped_paths"])
    # ["callback (type=function)"]
```

**Parameters:**
- `value` (Any): The value to sanitize
- `on_drop` (`OnDrop`, i.e. `Literal["silent", "warn", "error"]`): `"silent"`
  (default), `"warn"`, or `"error"`. Any other value raises `ValueError`, so a
  misspelt `"error"` is not taken as silent.

**Returns:** A JSON-safe copy of the input with non-serializable values removed;
`None` when the input itself is dropped.

### validate_json_safe()

Read-only traversal that reports non-serializable paths without modifying the input.

```python
from dataknobs_common.serialization import validate_json_safe

data = {"name": "Alice", "callback": some_function, "nested": {"task": asyncio_task}}
problems = validate_json_safe(data)
# ["callback (type=function)", "nested.task (type=Task)"]

if not problems:
    print("Fully JSON-safe")
```

**Parameters:**
- `value` (Any): The value to check

**Returns:** `list[str]` of paths to non-serializable values. Empty list means fully safe.

It is the same walk as `sanitize_for_json(value, on_drop="error")`, and each
entry is what that call's `SerializationError` would list in
`context["dropped_paths"]`.

### as_data()

Walks a value into JSON-shaped data and **refuses** what it has no rule for,
rather than dropping it. Data that lost a field without saying so reads as data
where that field was empty: a dashboard shows no tags, with nothing to say why.

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from dataknobs_common.exceptions import SerializationError
from dataknobs_common.serialization import as_data


@dataclass(frozen=True)
class Price:
    cents: int
    currency: str

    def to_dict(self) -> dict[str, Any]:
        # A value that chose its own shape: an amount as text, never a float
        whole, part = divmod(self.cents, 100)
        return {"amount": f"{whole}.{part:02d}", "currency": self.currency}


@dataclass(frozen=True)
class Product:
    sku: str
    tags: frozenset[str]
    price: Price
    notes: dict[str, Any]


kettle = Product("SKU-1042", frozenset({"stainless", "kitchen"}), Price(3450, "EUR"), {})
as_data(kettle)
# {"sku": "SKU-1042", "tags": ["kitchen", "stainless"],
#  "price": {"amount": "34.50", "currency": "EUR"}, "notes": {}}

try:
    as_data(Product("SKU-1042", frozenset(), Price(3450, "EUR"), {"tare": Decimal("0.15")}))
except SerializationError as e:
    print(e)                   # value.notes['tare'] is of type Decimal, which has no rule as data
    print(e.context["path"])   # value.notes['tare']
```

Its rules, in order:

1. a type in `using` → that type's serialiser, its result walked. The
   serialiser is not applied to the value it returned, so one returning its own
   type cannot loop, but a value of that type anywhere inside the result is
   shown by it too. The exact type is looked up first, then the first listed base;
2. an `Enum` → its value, walked;
3. `None`, `bool`, `int`, `float`, `str` → itself, except a NaN or infinite
   float: JSON cannot write it, so it is refused;
4. an object with a callable `to_dict()` (not a class) → its result, walked.
   This comes before the dataclass rule, so a dataclass's own `to_dict()` is used;
5. a dataclass instance → its fields, in declaration order;
6. any `Mapping` (a `MappingProxyType`, say) → its items. An `Enum` key is
   read as its value first; a `None`, `bool`, `int` or `float` key is written
   as `json.dumps` writes it (`"null"`, `"true"`, `"1.5"`), so the two agree on
   the same dict; any other key is written with `str()`. Two keys written the
   same, `1` and `"1"`, are refused: one would replace the other;
7. a `list` or `tuple` → a list;
8. a `set` or `frozenset` → a list sorted by each item's JSON, so the same set
   always gives the same data;
9. anything else → refused.

Every refusal is a `SerializationError` whose message and `context["path"]`
name where the value was. A `to_dict()` or `using` serialiser that raises is
refused too, with the original exception as `__cause__`; when that exception
is itself a refusal from a nested `as_data`, the outer path is the one named.
A container that holds itself is refused, and so is a value nested past the
interpreter's recursion limit. A set's member has no position, so its path is
written `value.tags{}`.

`using` shows a type differently in one place without giving it a `to_dict()`
that every other caller would get:

```python
as_data(order, using={Supplier: lambda s: s.supplier_id})
```

**Parameters:**
- `value` (Any): The value to walk
- `using` (`Mapping[type, Callable[[Any], Any]]`, keyword-only): per-call
  serialisers by type. The type is exported as `Using`.

**Returns:** dicts with `str` keys, lists and JSON scalars.

### fields_data()

A dataclass's fields as data, with some given in place of the field's value —
for a value with one field too large to carry whole:

```python
from dataknobs_common.serialization import fields_data

fields_data(order, lines={"count": len(order.lines)})
# {"order_id": "ORD-88", "lines": {"count": 200}, "placed_by": "warehouse-2"}
```

A replacement is walked by the rules of `as_data`. A name that is not a field is
refused (`Order has no field items`), so a renamed field cannot leave a
replacement behind unnoticed. It takes only a dataclass instance, and a refusal
names its path from the class: `Order.lines[0]`.

**Parameters:**
- `value` (Any, positional-only): a dataclass instance
- `using` (keyword-only): as for `as_data`
- `**replaced`: field names, each with the value to give in its place

**Returns:** `dict[str, Any]` of the fields, in declaration order.

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
