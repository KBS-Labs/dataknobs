# Record Model

The Record class is the fundamental data structure in the DataKnobs data package, representing a single data entity with fields, metadata, and a unique identifier.

## Overview

The Record model provides:

- **First-class ID support**: Built-in unique identifier with UUID generation
- **Flexible field storage**: Dynamic field management with type safety
- **Rich metadata**: Extensible metadata for tracking and custom attributes
- **Deep operations**: Deep copy, merge, and projection capabilities
- **Type validation**: Field type enforcement and validation
- **Serialization**: JSON and dictionary conversion support

## Core Components

### Record Class

The `Record` class represents a data entity:

```python
from dataknobs_data import Record, Field, FieldType
from datetime import datetime

# Create a record with fields
record = Record(
    fields={
        "name": Field(name="name", type=FieldType.STRING, value="John Doe"),
        "age": Field(name="age", type=FieldType.INTEGER, value=30),
        "email": Field(name="email", type=FieldType.STRING, value="john@example.com")
    },
    metadata={
        "source": "api",
        "version": "1.0",
        "created_at": datetime.now()
    }
)

print(record.id)  # Auto-generated UUID
print(record.fields["name"].value)  # "John Doe"
print(record.metadata["source"])  # "api"
```

### ID Management (Enhanced in Phase 7)

Records now have first-class ID support with automatic UUID generation:

```python
# Automatic ID generation
record1 = Record()
print(record1.id)  # e.g., "550e8400-e29b-41d4-a716-446655440000"

# Explicit ID assignment
record2 = Record(id="custom_id_123")
print(record2.id)  # "custom_id_123"

# ID preservation in operations
record3 = record1.copy(deep=True)
print(record3.id == record1.id)  # True - ID is preserved

# ID in metadata (backward compatibility)
record4 = Record(metadata={"id": "legacy_id"})
print(record4.id)  # "legacy_id" - reads from metadata if not set
```

### Field Class

Fields represent individual data attributes:

```python
from dataknobs_data import Field, FieldType

# Create fields with different types
name_field = Field(
    name="name",
    type=FieldType.STRING,
    value="Alice Smith"
)

age_field = Field(
    name="age",
    type=FieldType.INTEGER,
    value=25
)

score_field = Field(
    name="score",
    type=FieldType.FLOAT,
    value=98.5
)

active_field = Field(
    name="active",
    type=FieldType.BOOLEAN,
    value=True
)

data_field = Field(
    name="data",
    type=FieldType.JSON,
    value={"key": "value", "nested": {"data": 123}}
)

created_field = Field(
    name="created_at",
    type=FieldType.DATETIME,
    value=datetime.now()
)
```

### Field Types

Available field types and their Python equivalents:

| FieldType | Python Type | Description | Example |
|-----------|-------------|-------------|---------|
| STRING | str | Text data | "Hello World" |
| INTEGER | int | Whole numbers | 42 |
| FLOAT | float | Decimal numbers | 3.14159 |
| BOOLEAN | bool | True/False values | True |
| DATETIME | datetime | Date and time | datetime.now() |
| JSON | dict/list | Structured data | {"key": "value"} |
| BINARY | bytes | Binary data | b"\x00\x01\x02" |

## Field Operations

### Ergonomic Field Access (New in Priority 5)

The Record class now provides multiple convenient ways to access and manipulate field values:

```python
# Create a record with data
record = Record({
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com",
    "temperature": 25.5
})

# 1. Dict-like access (NEW) - returns values directly
print(record["name"])         # "Alice"
record["age"] = 31            # Update value
print(record["age"])          # 31

# 2. Attribute access (NEW) - most convenient
print(record.name)            # "Alice"
record.email = "alice@new.com"  # Update via attribute
print(record.email)           # "alice@new.com"

# 3. Add new fields dynamically
record.country = "USA"        # Creates new field via attribute
record["city"] = "NYC"        # Creates new field via dict access

# 4. Simplified to_dict() (NEW default behavior)
data = record.to_dict()       # Returns flat dict by default
print(data)
# {'name': 'Alice', 'age': 31, 'email': 'alice@new.com', 
#  'temperature': 25.5, 'country': 'USA', 'city': 'NYC'}

# 5. Access Field objects when needed (for metadata)
temp_field = record.get_field_object("temperature")
print(temp_field.type)        # FieldType.FLOAT
print(temp_field.metadata)    # Field metadata if any

# 6. Check field existence
if "temperature" in record:
    print(f"Temperature: {record.temperature}°C")
```

### Traditional Field Access (Still Supported)

For backward compatibility and when you need access to Field objects:

```python
# Create a record
record = Record()

# Add fields with Field objects
record.fields["username"] = Field(
    name="username",
    type=FieldType.STRING,
    value="user123"
)

# Access field values through Field objects
username = record.fields["username"].value
print(f"Username: {username}")

# Update field values
record.fields["username"].value = "updated_user"

# Delete fields
del record.fields["username"]

# Check field existence
if "email" in record.fields:
    print(f"Email: {record.fields['email'].value}")
```

### Field Validation

```python
# Type validation
try:
    field = Field(
        name="age",
        type=FieldType.INTEGER,
        value="not a number"  # Will raise TypeError
    )
except TypeError as e:
    print(f"Validation error: {e}")

# Custom validation
def validate_email(value):
    """Validate email format"""
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(pattern, value):
        raise ValueError(f"Invalid email: {value}")
    return value

email_field = Field(
    name="email",
    type=FieldType.STRING,
    value=validate_email("user@example.com")
)
```

## Record Operations

### Deep Copy

Create independent copies of records:

```python
# Shallow copy (fields reference same objects)
shallow_copy = record.copy(deep=False)
shallow_copy.fields["data"].value["key"] = "modified"
print(record.fields["data"].value["key"])  # Also "modified"

# Deep copy (completely independent)
deep_copy = record.copy(deep=True)
deep_copy.fields["data"].value["key"] = "changed"
print(record.fields["data"].value["key"])  # Still original value

# ID preservation
copy_with_id = record.copy(deep=True)
print(copy_with_id.id == record.id)  # True - ID is preserved
```

### Merge Operations

Combine fields from multiple records:

```python
# Base record
record1 = Record(fields={
    "name": Field("name", FieldType.STRING, "John"),
    "age": Field("age", FieldType.INTEGER, 30)
})

# Updates
record2 = Record(fields={
    "age": Field("age", FieldType.INTEGER, 31),  # Update existing
    "email": Field("email", FieldType.STRING, "john@example.com")  # Add new
})

# Merge records
merged = record1.merge(record2)
print(merged.fields["name"].value)  # "John" (unchanged)
print(merged.fields["age"].value)   # 31 (updated)
print(merged.fields["email"].value) # "john@example.com" (added)

# Merge with custom strategy
def merge_strategy(field1, field2):
    """Custom merge logic"""
    if field1.name == "age":
        # Keep the higher age
        return field1 if field1.value > field2.value else field2
    return field2  # Default: use second value

merged_custom = record1.merge(record2, strategy=merge_strategy)
```

### Projection

Extract subset of fields:

```python
# Original record with many fields
full_record = Record(fields={
    "id": Field("id", FieldType.STRING, "123"),
    "name": Field("name", FieldType.STRING, "Alice"),
    "email": Field("email", FieldType.STRING, "alice@example.com"),
    "age": Field("age", FieldType.INTEGER, 28),
    "address": Field("address", FieldType.JSON, {"city": "NYC"}),
    "phone": Field("phone", FieldType.STRING, "555-1234")
})

# Project specific fields
public_fields = full_record.project(["id", "name", "email"])
print(list(public_fields.fields.keys()))  # ["id", "name", "email"]

# Project with transformation
def transform_field(field):
    """Transform field during projection"""
    if field.name == "email":
        # Mask email
        value = field.value
        parts = value.split("@")
        masked = f"{parts[0][:2]}***@{parts[1]}"
        return Field(field.name, field.type, masked)
    return field

masked_record = full_record.project(
    ["name", "email"],
    transform=transform_field
)
print(masked_record.fields["email"].value)  # "al***@example.com"
```

## Metadata Management

### Working with Metadata

```python
# Create record with metadata
record = Record(
    fields={"name": Field("name", FieldType.STRING, "Bob")},
    metadata={
        "source": "import",
        "version": "2.0",
        "tags": ["customer", "vip"],
        "imported_at": datetime.now().isoformat()
    }
)

# Access metadata
print(record.metadata["source"])  # "import"
print(record.metadata.get("tags", []))  # ["customer", "vip"]

# Update metadata
record.metadata["processed"] = True
record.metadata["processed_at"] = datetime.now().isoformat()

# Metadata in operations
copy = record.copy(deep=True)
print(copy.metadata["source"])  # Metadata is also copied
```

### System Metadata

Common metadata patterns:

```python
def add_system_metadata(record):
    """Add system metadata to record"""
    now = datetime.now()
    
    record.metadata.update({
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "version": 1,
        "schema_version": "1.0.0",
        "checksum": calculate_checksum(record)
    })
    
    return record

def track_changes(old_record, new_record):
    """Track changes between records"""
    changes = []
    
    for field_name, new_field in new_record.fields.items():
        if field_name not in old_record.fields:
            changes.append({"field": field_name, "action": "added"})
        elif old_record.fields[field_name].value != new_field.value:
            changes.append({
                "field": field_name,
                "action": "modified",
                "old": old_record.fields[field_name].value,
                "new": new_field.value
            })
    
    for field_name in old_record.fields:
        if field_name not in new_record.fields:
            changes.append({"field": field_name, "action": "removed"})
    
    new_record.metadata["changes"] = changes
    new_record.metadata["updated_at"] = datetime.now().isoformat()
    
    return new_record
```

## Serialization

### Dictionary Conversion (Enhanced in Priority 5)

The `to_dict()` method now has improved ergonomics with sensible defaults:

```python
# Simple flat dictionary (NEW DEFAULT)
simple_dict = record.to_dict()  # flatten=True by default
print(simple_dict)
# {
#     "name": "John",
#     "age": 30,
#     "email": "john@example.com",
#     "_id": "550e8400-e29b-41d4-a716-446655440000"
# }

# Include metadata in flat format
with_meta = record.to_dict(include_metadata=True)
print(with_meta)
# {
#     "name": "John",
#     "age": 30,
#     "email": "john@example.com",
#     "_id": "550e8400-e29b-41d4-a716-446655440000",
#     "_metadata": {"source": "api", "version": "1.0"}
# }

# Structured format for serialization (backward compatible)
structured = record.to_dict(flatten=False, include_metadata=True)
print(structured)
# {
#     "id": "550e8400-e29b-41d4-a716-446655440000",
#     "fields": {
#         "name": {"name": "name", "type": "STRING", "value": "John"},
#         "age": {"name": "age", "type": "INTEGER", "value": 30}
#     },
#     "metadata": {"source": "api", "version": "1.0"}
# }

# Create from dictionary (works with both formats)
new_record = Record.from_dict(simple_dict)
print(new_record["name"])  # "John" - using new dict-like access

structured_record = Record.from_dict(structured)
print(structured_record.id)  # Same ID preserved
print(structured_record.name)  # "John" - using new attribute access
```

### JSON Serialization

```python
import json

# Convert to JSON
record_json = record.to_json()
print(record_json)
# JSON string representation

# Create from JSON
json_str = '''
{
    "id": "custom_123",
    "fields": {
        "title": {"name": "title", "type": "STRING", "value": "Example"}
    },
    "metadata": {"created": "2024-01-01"}
}
'''
record = Record.from_json(json_str)

# Custom JSON encoder for complex types
class RecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Record):
            return obj.to_dict()
        elif isinstance(obj, Field):
            return {"name": obj.name, "type": str(obj.type), "value": obj.value}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Use custom encoder
json_output = json.dumps(record, cls=RecordEncoder, indent=2)
```

## Advanced Patterns

### Record Factory

Create records with consistent structure:

```python
class RecordFactory:
    """Factory for creating records with consistent schema"""
    
    def __init__(self, schema):
        self.schema = schema
    
    def create(self, **kwargs):
        """Create record with schema defaults"""
        fields = {}
        
        for field_name, field_def in self.schema.items():
            value = kwargs.get(field_name, field_def.get("default"))
            if value is not None:
                fields[field_name] = Field(
                    name=field_name,
                    type=field_def["type"],
                    value=value
                )
        
        return Record(fields=fields)

# Define schema
user_schema = {
    "name": {"type": FieldType.STRING, "required": True},
    "email": {"type": FieldType.STRING, "required": True},
    "age": {"type": FieldType.INTEGER, "default": 0},
    "active": {"type": FieldType.BOOLEAN, "default": True}
}

# Create factory
factory = RecordFactory(user_schema)

# Create records
user1 = factory.create(name="Alice", email="alice@example.com")
user2 = factory.create(name="Bob", email="bob@example.com", age=25)
```

### Record Comparison

Compare and diff records:

```python
def compare_records(record1, record2):
    """Compare two records and return differences"""
    diff = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": []
    }
    
    # Check all fields in record1
    for field_name, field1 in record1.fields.items():
        if field_name not in record2.fields:
            diff["removed"].append(field_name)
        elif field1.value != record2.fields[field_name].value:
            diff["modified"].append({
                "field": field_name,
                "old": field1.value,
                "new": record2.fields[field_name].value
            })
        else:
            diff["unchanged"].append(field_name)
    
    # Check for added fields in record2
    for field_name in record2.fields:
        if field_name not in record1.fields:
            diff["added"].append(field_name)
    
    return diff

# Compare records
diff = compare_records(old_record, new_record)
print(f"Added fields: {diff['added']}")
print(f"Modified fields: {len(diff['modified'])}")
```

### Record Pipelines

Process records through transformation pipelines:

```python
class RecordPipeline:
    """Pipeline for record transformations"""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, func):
        """Add transformation step"""
        self.steps.append(func)
        return self
    
    def process(self, record):
        """Process record through pipeline"""
        result = record
        for step in self.steps:
            result = step(result)
        return result

# Define transformation steps
def normalize_email(record):
    """Normalize email to lowercase"""
    if "email" in record.fields:
        record.fields["email"].value = record.fields["email"].value.lower()
    return record

def add_timestamp(record):
    """Add processing timestamp"""
    record.metadata["processed_at"] = datetime.now().isoformat()
    return record

def validate_required(record):
    """Validate required fields"""
    required = ["name", "email"]
    for field in required:
        if field not in record.fields:
            raise ValueError(f"Missing required field: {field}")
    return record

# Create and use pipeline
pipeline = RecordPipeline()
pipeline.add_step(normalize_email) \
        .add_step(validate_required) \
        .add_step(add_timestamp)

processed_record = pipeline.process(raw_record)
```

## Performance Considerations

### Memory Optimization

```python
# Use __slots__ for memory efficiency in custom fields
class OptimizedField:
    __slots__ = ['name', 'type', 'value']
    
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

# Lazy loading for large fields
class LazyField(Field):
    """Field that loads value on demand"""
    
    def __init__(self, name, type, loader):
        super().__init__(name, type, None)
        self._loader = loader
        self._loaded = False
    
    @property
    def value(self):
        if not self._loaded:
            self._value = self._loader()
            self._loaded = True
        return self._value
```

### Batch Processing

```python
def process_records_batch(records, batch_size=1000):
    """Process records in batches for memory efficiency"""
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        # Process batch
        for record in batch:
            # Transform record
            record.fields["processed"] = Field(
                "processed",
                FieldType.BOOLEAN,
                True
            )
        
        # Yield processed batch
        yield batch
        
        # Clear references for garbage collection
        del batch
```

## Best Practices

1. **Always use appropriate field types**: Choose the correct FieldType for your data
2. **Preserve IDs**: Maintain record IDs through transformations and copies
3. **Use deep copy when needed**: Use deep=True when modifying nested structures
4. **Validate field values**: Implement validation for critical fields
5. **Document metadata schema**: Clearly define metadata structure
6. **Handle missing fields gracefully**: Use `.get()` with defaults
7. **Consider memory usage**: Use projections to reduce memory footprint
8. **Implement equality properly**: Define clear equality semantics for records
9. **Use factories for consistency**: Create record factories for common schemas
10. **Track record lineage**: Use metadata to track record transformations

## See Also

- [Field Types](field-types.md) - Detailed field type documentation
- [Schema Validation](validation.md) - Record validation with schemas
- [Pandas Integration](pandas-integration.md) - Converting records to DataFrames
- [Migration Utilities](migration.md) - Record transformation and migration