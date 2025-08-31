# DataKnobs Data Package - API Reference

## Overview

The DataKnobs Data Package provides a unified data abstraction layer with support for multiple backends, data validation, migration, and streaming capabilities.

## Table of Contents

- [Core Components](#core-components)
  - [Database](#database)
  - [Records](#records)
  - [Query](#query)
  - [Streaming](#streaming)
- [Validation Module](#validation-module)
  - [Schemas](#schemas)
  - [Constraints](#constraints)
  - [Coercion](#coercion)
- [Migration Module](#migration-module)
  - [Operations](#operations)
  - [Migrations](#migrations)
  - [Migrator](#migrator)
- [Backends](#backends)
- [Exceptions](#exceptions)

## Core Components

### Database

The `Database` class provides async database operations, while `SyncDatabase` provides synchronous operations.

```python
from dataknobs_data import Database, SyncDatabase

# Async usage
async def main():
    db = await AsyncDatabase.from_backend("memory")  # Auto-connects
    record = Record({"name": "Alice", "age": 30})
    id = await db.create(record)
    retrieved = await db.read(id)
    await db.close()

# Sync usage
db = SyncDatabase.from_backend("memory")  # Auto-connects
record = Record({"name": "Bob", "age": 25})
id = db.create(record)
retrieved = db.read(id)
db.close()
```

#### Database Methods

- `create(record: Record) -> str`: Create a new record and return its ID
- `read(id: str) -> Record | None`: Read a record by ID
- `update(id: str, record: Record) -> bool`: Update an existing record
- `delete(id: str) -> bool`: Delete a record
- `exists(id: str) -> bool`: Check if a record exists
- `upsert(id: str, record: Record) -> str`: Update or insert a record
- `search(query: Query) -> List[Record]`: Search for records
- `count(query: Query | None) -> int`: Count matching records
- `clear() -> int`: Delete all records
- `stream_read(query, config) -> Iterator[Record]`: Stream records
- `stream_write(records, config) -> StreamResult`: Stream write records

### Records

`Record` represents a data record with fields and metadata.

```python
from dataknobs_data import Record

# Create from dict
record = Record({"name": "Alice", "age": 30})

# With metadata
record = Record(
    data={"name": "Bob", "age": 25},
    metadata={"created_by": "user1", "version": 1}
)

# Access values
name = record.get_value("name")
age = record.get_value("age", default=0)

# Set values
record.set_value("email", "alice@example.com")

# Check fields
if record.has_field("email"):
    email = record.get_value("email")

# Convert to dict
data_dict = record.to_dict()
```

### Query

Build queries to search and filter records.

```python
from dataknobs_data import Query, Operator, SortOrder

# Simple query
query = Query().filter("age", Operator.GT, 25)

# Complex query with multiple filters
query = (Query()
    .filter("status", Operator.EQ, "active")
    .filter("age", Operator.GTE, 18)
    .filter("age", Operator.LTE, 65)
    .sort("age", SortOrder.DESC)
    .limit(10)
    .offset(20))

# Available operators
# Operator.EQ - equals
# Operator.NEQ - not equals
# Operator.GT - greater than
# Operator.GTE - greater than or equal
# Operator.LT - less than
# Operator.LTE - less than or equal
# Operator.IN - in list
# Operator.NOT_IN - not in list
# Operator.LIKE - pattern match (SQL LIKE)
# Operator.EXISTS - field exists
# Operator.NOT_EXISTS - field doesn't exist
# Operator.REGEX - regular expression match
```

### Streaming

Stream large datasets efficiently.

```python
from dataknobs_data import StreamConfig, StreamResult

# Configure streaming
config = StreamConfig(
    batch_size=100,
    buffer_size=1000,
    on_error=lambda e, r: print(f"Error: {e}")
)

# Stream read
async for record in db.stream_read(query, config):
    process(record)

# Stream write
result = await db.stream_write(record_iterator, config)
print(f"Processed: {result.total_processed}")
print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")
```

## Validation Module

The validation module (`dataknobs_data.validation`) provides schema-based validation with constraints and type coercion.

### Schemas

Define validation schemas for your data.

```python
from dataknobs_data.validation import Schema, FieldType
from dataknobs_data.validation.constraints import Required, Range, Length, Pattern

# Create schema
schema = Schema("UserSchema")
schema.field("id", FieldType.INTEGER, required=True)
schema.field("name", FieldType.STRING, 
    constraints=[Length(min=1, max=100)])
schema.field("email", FieldType.STRING,
    constraints=[Pattern(r"^.+@.+\..+$")])
schema.field("age", FieldType.INTEGER,
    constraints=[Range(min=0, max=150)])

# Validate record
record = Record({"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30})
result = schema.validate(record)
if result.valid:
    print("Valid record!")
else:
    print("Errors:", result.errors)

# With type coercion
record = Record({"id": "1", "age": "30"})  # String values
result = schema.validate(record, coerce=True)  # Will convert to int
```

### Constraints

Built-in constraints for field validation:

```python
from dataknobs_data.validation.constraints import *

# Required fields
Required()  # Field must be present and non-None
Required(allow_empty=True)  # Allow empty strings/collections

# Numeric ranges
Range(min=0, max=100)  # Inclusive range
Range(min=0)  # Only minimum
Range(max=100)  # Only maximum

# String/collection length
Length(min=1, max=50)  # Length constraints
Length(min=5)  # Minimum length only

# Pattern matching
Pattern(r"^\d{3}-\d{4}$")  # Regex pattern
Pattern(r"^[A-Z]+$", re.IGNORECASE)  # With flags

# Enumeration
Enum(["active", "inactive", "pending"])  # Must be one of values

# Uniqueness (with context)
Unique()  # Value must be unique across all records

# Custom validation
def validate_email(value):
    return "@" in value and "." in value
Custom(validate_email, "Invalid email format")

# Composite constraints
All([Required(), Range(min=0)])  # All must pass
AnyOf([Pattern(r"^\d+$"), Pattern(r"^[A-Z]+$")])  # At least one must pass

# Constraint composition with operators
constraint = Required() & Range(min=0, max=100)  # AND
constraint = Length(min=10) | Pattern(r"^\d{5}$")  # OR
```

### Coercion

Type coercion for automatic type conversion:

```python
from dataknobs_data.validation import Coercer, FieldType

coercer = Coercer()

# Coerce single value
result = coercer.coerce("123", FieldType.INTEGER)
if result.valid:
    value = result.value  # 123 (int)

# Coerce multiple values
data = {"age": "30", "active": "true", "score": "95.5"}
results = coercer.coerce_many(data, {
    "age": FieldType.INTEGER,
    "active": FieldType.BOOLEAN,
    "score": FieldType.FLOAT
})
```

## Migration Module

The migration module (`dataknobs_data.migration`) provides data transformation and migration capabilities.

### Operations

Define operations to transform records:

```python
from dataknobs_data.migration.operations import *

# Add field
add_op = AddField("status", default="active")

# Remove field
remove_op = RemoveField("deprecated_field")

# Rename field
rename_op = RenameField("old_name", "new_name")

# Transform field
def uppercase(value):
    return value.upper() if value else value
transform_op = TransformField("name", uppercase)

# Composite operations
composite = CompositeOperation([
    AddField("created_at", default=datetime.now()),
    RemoveField("temp_field"),
    RenameField("user_name", "username")
])
```

### Migrations

Create migrations to evolve your data:

```python
from dataknobs_data.migration import Migration

# Create migration
migration = Migration(
    name="add_user_status",
    version="1.0.0",
    description="Add status field to user records"
)

# Add operations
migration.add_operation(AddField("status", default="active"))
migration.add_operation(RemoveField("legacy_field"))

# Apply to record
record = Record({"name": "Alice", "legacy_field": "old"})
result = migration.apply(record)
if result.valid:
    migrated = result.value  # Record with changes applied

# Reverse migration
result = migration.reverse(migrated)
```

### Migrator

Batch migration for databases:

```python
from dataknobs_data.migration import Migrator

# Create migrator
migrator = Migrator(source_db, target_db)

# Configure migration
migration = Migration("update_schema", "2.0.0")
migration.add_operation(AddField("version", default=2))

# Run migration
async def migrate():
    progress = await migrator.migrate(
        migration=migration,
        query=Query().filter("type", Operator.EQ, "user"),
        batch_size=100,
        on_progress=lambda p: print(f"Progress: {p.percentage}%")
    )
    
    print(f"Migrated: {progress.successful}")
    print(f"Failed: {progress.failed}")
    print(f"Duration: {progress.duration}s")
```

## Backends

### Memory Backend

In-memory storage for testing and development:

```python
db = await Database.create("memory")
```

### File Backend

JSON file-based storage:

```python
db = await Database.create("file", {
    "path": "/data/records.json",
    "pretty": True,
    "backup": True
})
```

### PostgreSQL Backend

PostgreSQL database storage:

```python
db = await Database.create("postgres", {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "user",
    "password": "pass",
    "table": "records",
    "schema": "public"
})
```

### S3 Backend

AWS S3 storage:

```python
db = await Database.create("s3", {
    "bucket": "my-bucket",
    "prefix": "records/",
    "region": "us-west-2",
    "aws_access_key_id": "key",
    "aws_secret_access_key": "secret"
})
```

### Elasticsearch Backend

Elasticsearch storage:

```python
db = await Database.create("elasticsearch", {
    "host": "localhost",
    "port": 9200,
    "index": "records",
    "refresh": True
})
```

## Exceptions

The package defines specific exceptions for different error scenarios:

```python
from dataknobs_data import (
    DataknobsDataError,  # Base exception
    RecordNotFoundError,  # Record doesn't exist
    RecordValidationError,  # Validation failed
    FieldTypeError,  # Invalid field type
    DatabaseConnectionError,  # Connection issues
    DatabaseOperationError,  # Operation failed
    QueryError,  # Invalid query
    SerializationError,  # Serialization failed
    BackendNotFoundError,  # Unknown backend
    ConfigurationError,  # Invalid configuration
    ConcurrencyError,  # Concurrent access issue
    TransactionError,  # Transaction failed
    MigrationError  # Migration failed
)

try:
    record = await db.read("unknown-id")
except RecordNotFoundError as e:
    print(f"Record not found: {e}")
```

## Factory Functions

Convenient factory functions for creating databases:

```python
from dataknobs_data import database_factory, async_database_factory

# Synchronous factory
db = database_factory("memory")
db = database_factory("postgres", config)

# Asynchronous factory
db = await async_database_factory("memory")
db = await async_database_factory("s3", config)
```

## Configuration

Many components support configuration through dictionaries or environment variables:

```python
# From environment variables (using python-dotenv)
db = await Database.create("postgres")  # Uses .env file

# From explicit config
config = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME")
}
db = await Database.create("postgres", config)
```

## Best Practices

1. **Always close connections**: Use context managers or explicitly call `close()`:
   ```python
   async with await Database.create("memory") as db:
       # Use db
       pass  # Auto-closes
   ```

2. **Use type hints**: The package is fully typed for better IDE support:
   ```python
   from dataknobs_data import Database, Record
   
   async def process_record(db: Database, id: str) -> Record | None:
       return await db.read(id)
   ```

3. **Handle exceptions**: Catch specific exceptions for better error handling:
   ```python
   try:
       await db.create(record)
   except RecordValidationError as e:
       # Handle validation error
       pass
   except DatabaseOperationError as e:
       # Handle database error
       pass
   ```

4. **Use streaming for large datasets**: Avoid loading all data into memory:
   ```python
   # Good - streams data
   async for record in db.stream_read(query):
       process(record)
   
   # Bad for large datasets - loads all into memory
   records = await db.search(query)
   for record in records:
       process(record)
   ```

5. **Validate data before storage**: Use schemas to ensure data quality:
   ```python
   schema = create_user_schema()
   result = schema.validate(record)
   if result.valid:
       await db.create(record)
   else:
       handle_validation_errors(result.errors)
   ```