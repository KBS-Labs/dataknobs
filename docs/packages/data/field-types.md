# Field Types

## Overview

The DataKnobs data package provides a comprehensive type system for structured data fields. Each field in a Record has an associated type that determines validation rules and serialization behavior.

`Field` and `FieldType` are *defined* in `dataknobs-common` and re-exported here, so the imports on this page are unchanged and resolve to the same objects. `VectorField` is `dataknobs-data`'s own, because it needs `numpy` at runtime.

## Available Field Types

| FieldType | Python Type | Description | Example |
|-----------|-------------|-------------|---------|
| `STRING` | `str` | Text data | `"Hello World"` |
| `INTEGER` | `int` | Whole numbers | `42` |
| `FLOAT` | `float` | Decimal numbers | `3.14159` |
| `BOOLEAN` | `bool` | True/False values | `True` |
| `DATETIME` | `datetime` | Date and time | `datetime.now()` |
| `JSON` | `dict/list` | Structured data | `{"key": "value"}` |
| `BINARY` | `bytes` | Binary data | `b"\x00\x01\x02"` |
| `TEXT` | `str` | Long text (auto-detected above 1000 characters) | `"..."` |
| `VECTOR` | `VectorField` | Dense embedding | `VectorField(value=[0.1, 0.2])` |
| `SPARSE_VECTOR` | `VectorField` | Sparse embedding | `VectorField(value=[0.0, 1.0])` |

## Using Field Types

### Basic Usage

```python
from dataknobs_data import Field, FieldType

# Create typed fields
name_field = Field(
    name="name",
    type=FieldType.STRING,
    value="Alice"
)

age_field = Field(
    name="age", 
    type=FieldType.INTEGER,
    value=30
)

score_field = Field(
    name="score",
    type=FieldType.FLOAT,
    value=98.5
)
```

### Type Validation

Fields automatically validate values against their declared type:

```python
from dataknobs_data import Field, FieldType

# This works
valid_field = Field(
    name="count",
    type=FieldType.INTEGER,
    value=10
)

# This raises TypeError
try:
    invalid_field = Field(
        name="count",
        type=FieldType.INTEGER,
        value="not a number"
    )
except TypeError as e:
    print(f"Validation error: {e}")
```

### Working with Complex Types

#### JSON Fields

JSON fields can store dictionaries or lists:

```python
config_field = Field(
    name="config",
    type=FieldType.JSON,
    value={
        "host": "localhost",
        "port": 8080,
        "features": ["auth", "logging", "metrics"]
    }
)

# Access nested data
config = config_field.value
print(config["host"])  # "localhost"
print(config["features"][0])  # "auth"
```

#### DateTime Fields

DateTime fields handle temporal data:

```python
from datetime import UTC, datetime

created_field = Field(
    name="created_at",
    type=FieldType.DATETIME,
    value=datetime.now(UTC)
)

# Serialization handles ISO format
iso_string = created_field.value.isoformat()
```

#### Binary Fields

Binary fields store raw bytes:

```python
# Store file content
with open("image.png", "rb") as f:
    image_field = Field(
        name="thumbnail",
        type=FieldType.BINARY,
        value=f.read()
    )

# Store encoded data
import base64
encoded_data = base64.b64encode(b"secret data")
encoded_field = Field(
    name="encoded",
    type=FieldType.BINARY,
    value=encoded_data
)
```

## Type Inference

The system can infer types from Python values:

```python
from dataknobs_data import Field

# Type is inferred from value
auto_string = Field(name="text", value="Hello")  # STRING
auto_int = Field(name="count", value=42)  # INTEGER
auto_float = Field(name="ratio", value=0.5)  # FLOAT
auto_bool = Field(name="active", value=True)  # BOOLEAN
auto_json = Field(name="data", value={"key": "val"})  # JSON
```

## Custom Validation

Add custom validation on top of type checking:

```python
def validate_email(value: str) -> str:
    """Validate email format."""
    import re
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
        raise ValueError(f"Invalid email: {value}")
    return value

email_field = Field(
    name="email",
    type=FieldType.STRING,
    value=validate_email("user@example.com")
)
```

## Type Conversion

Convert between compatible types:

```python
# String to Integer
str_value = "123"
int_field = Field(
    name="parsed",
    type=FieldType.INTEGER,
    value=int(str_value)
)

# Float to Integer (with rounding)
float_value = 3.7
rounded_field = Field(
    name="rounded",
    type=FieldType.INTEGER,
    value=round(float_value)
)

# Dict to JSON string
data = {"status": "active"}
json_str_field = Field(
    name="json_string",
    type=FieldType.STRING,
    value=json.dumps(data)
)
```

## Field Metadata

Attach metadata to fields for additional context:

```python
temperature_field = Field(
    name="temperature",
    type=FieldType.FLOAT,
    value=25.5,
    metadata={
        "unit": "celsius",
        "sensor_id": "TH100",
        "precision": 0.1
    }
)

# Access metadata
unit = temperature_field.metadata.get("unit")
print(f"Temperature: {temperature_field.value}°{unit[0].upper()}")
```

## SQL Types for Tables With Their Own Columns

A table read through the native column layout
([Query System](query.md#tables-with-their-own-columns-native-layout)) has
column types `FieldType` does not name. A schema field declares one with
`sql_type:` (or `metadata: {sql_type: ...}`, which wins), naming an entry of
the `sql_types` registry:

```python
from dataknobs_data.schema import DatabaseSchema

schema = DatabaseSchema.from_dict({"fields": {
    "node_id": {"type": "string", "sql_type": "uuid"},
    "seen_at": {"type": "datetime", "sql_type": "timestamptz"},
}})
```

Two ship:

- **`uuid`**: holds strings, as the uuid's canonical lower-case text. A `UUID`
  or a uuid string in any case compares equal to it, and an equality with
  one is compared in the column's own type, so its index serves it. Anything
  else, and every ordering or text operator, compares its text.
- **`timestamptz`**: holds zoned instants. A record holds the value as an
  aware `datetime` in UTC, and a plain `date` compares as its midnight in
  UTC. A naive `datetime` never compares with it, as `Filter.matches` never
  orders a naive value against an aware one.

The schema reader refuses a `sql_type` that is not a non-empty string. The
native layout refuses one that is not registered, listing those that are.
A registration does not change the built-in types: a `sql_type` naming
`integer` is refused as unregistered.

### Registering Your Own

```python
from dataknobs_data import SqlType, sql_types

# A case-insensitive text column, compared as text.
sql_types.register("citext", SqlType(kinds=frozenset({"string"}), text="stored"))
```

An `SqlType` says:

| Field | Meaning |
|---|---|
| `kinds` | which values the column holds: `"string"`, `"number"`, `"boolean"`, `NAIVE_TIME`, `ZONED_INSTANT`. A bound of any other kind matches nothing |
| `bind` | a bound as a record would hold it, or `None` when no value can equal it |
| `read` | a value the driver returned, as the record holds it |
| `text` | `"stored"`: the column's SQL value is text, so it compares with strings, a time bound reads it as a time, and the text operators apply. `"cast"`: the column is not text, but the record holds its text, so strings and the text operators compare `CAST(column AS TEXT)`. `None` (the default): neither |
| `holds` | for a `text="cast"` column, whether an equality bound can be compared in the column's own type |
| `placeholder` | given the dialect and the bounds, the SQL type a bound is sent as |

## Best Practices

1. **Always specify types explicitly** for clarity and validation
2. **Use appropriate types** - don't store numbers as strings
3. **Validate early** - catch type errors at field creation
4. **Document metadata** - explain what metadata fields mean
5. **Handle None values** - use Optional types where appropriate

## Building Your Own Field Class

`Field.from_dict` decides which class to build from a registry rather than from
a list of types written into the base class, so a subclass is reached by having
registered itself:

```python
from dataknobs_common import Field, FieldType, register_field_class


class MoneyField(Field):
    @classmethod
    def from_dict(cls, data):
        built = cls(
            name=data["name"],
            value=data["value"],
            type=FieldType.FLOAT,
            metadata=data.get("metadata", {}),
        )
        built.currency = data.get("metadata", {}).get("currency", "USD")
        return built


register_field_class(FieldType.FLOAT, MoneyField)

field = Field.from_dict({"name": "price", "value": 9.99, "type": "float"})
print(type(field).__name__)  # MoneyField
```

Registration happens at import of the module defining the subclass, and
re-registering a type overwrites the prior class — so a module substituting one
of the built-ins must be imported after it. `dataknobs_data.fields` registers
`VectorField` for `VECTOR` and `SPARSE_VECTOR` when it is imported.

A type with no registered class builds the class `from_dict` was called on.
That is why `dataknobs_common.Field.from_dict` returns a plain `Field` for a
vector payload in a process that never imported `dataknobs_data`: that package
is where the class able to hold a vector lives.

Two copies of a field are one class, too — `Field.copy()` copies through the
instance, so `MoneyField(...).copy()` is a `MoneyField` with `currency` intact
and needs no override.

## See Also

- [Record Model](record-model.md) - Working with records and fields
- [Validation](../../api/reference/data.md) - Advanced validation strategies
- [API Reference](api-reference.md) - Complete API documentation