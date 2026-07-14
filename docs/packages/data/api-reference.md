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
    db = await Database.create("memory")  # Auto-connects
    record = Record({"name": "Alice", "age": 30})
    id = await db.create(record)
    retrieved = await db.read(id)
    await db.close()

# Sync usage
db = SyncDatabase.create("memory")  # Auto-connects
record = Record({"name": "Bob", "age": 25})
id = db.create(record)
retrieved = db.read(id)
db.close()
```

#### Database Methods

- `create(record: Record) -> str`: Atomically insert a new record and return its ID; raises `DuplicateRecordError` if the id already exists
- `read(id: str) -> Record | None`: Read a record by ID
- `get_version(id: str) -> str | None`: Return an opaque optimistic-concurrency token for a record (or `None` if absent)
- `update(id: str, record: Record, *, expected_version: str | None = None) -> bool`: Update an existing record; with `expected_version`, a compare-and-set that raises `ConcurrencyError` on a stale token
- `delete(id: str, *, expected_version: str | None = None) -> bool`: Delete a record; with `expected_version`, a compare-and-set that raises `ConcurrencyError` on a stale token (a missing record returns `False`)
- `exists(id: str) -> bool`: Check if a record exists
- `upsert(id_or_record: str | Record, record: Record | None = None, *, expected_version: str | None = None) -> str`: Update or insert a record (enhanced to accept just a Record); with `expected_version`, a compare-and-set that never inserts
- `search(query: Query) -> List[Record]`: Search for records
- `count(query: Query | None) -> int`: Count matching records
- `clear() -> int`: Delete all records
- `stream_read(query, config) -> Iterator[Record]`: Stream records
- `stream_write(records, config) -> StreamResult`: Stream write records

#### Create semantics (atomic create-if-absent)

`create()` is a defined atomic insert across all backends. If a record with the
same id already exists, it fails closed with `DuplicateRecordError` rather than
overwriting the existing record — so a collision-safe insert needs no racy
`exists()`-then-`create()` guard:

```python
from dataknobs_data import DuplicateRecordError, Record

db.create(Record({"v": 1}, id="k"))
try:
    db.create(Record({"v": 2}, id="k"))
except DuplicateRecordError as e:
    print(e.id)  # "k" — the original record is untouched
```

`DuplicateRecordError` subclasses both the data-layer `ConcurrencyError` and
`ValueError`, so code that previously caught `ValueError` on a duplicate id
keeps working. It carries the colliding id on `.id` and in `context={"id": ...}`.
To overwrite when the id may already exist, use `upsert()` instead.

Backend notes:

- **memory, file, SQLite, DuckDB, Postgres, Elasticsearch** enforce the insert
  through their native uniqueness/constraint mechanism (in-lock check, primary
  key, or `op_type=create`).
- **S3** enforces it with a conditional PUT (`If-None-Match`); the atomic
  guarantee holds against any S3 implementation that honors conditional writes
  (real AWS S3, recent LocalStack) — both a pre-existing key (412) and a
  concurrent conditional-write race (409) fail closed — and degrades to
  last-writer-wins on stores that ignore the header.
- **`create_batch()` collision semantics are not uniform across backends.** On
  **memory, file, and S3** a colliding id — against an existing record or a
  duplicate within the same batch — fails closed with `DuplicateRecordError` and
  `record.id` is honored (S3 and the abstract-base default loop single
  `create()`). On **SQLite, DuckDB, PostgreSQL, and Elasticsearch** the bulk
  `create_batch()` mints a fresh id per record and ignores any `record.id` you
  set, so a colliding id does not fail. For collision-safe, id-preserving
  inserts on those backends, use single `create()` in a loop. (The *streaming*
  INSERT path is a separate axis — it fails closed on **memory and file**, but
  not on SQLite, DuckDB, PostgreSQL, S3, or Elasticsearch.)

#### Optimistic concurrency (conditional writes)

`update()`, `upsert()`, and `delete()` accept an opt-in, keyword-only
`expected_version` token so a read-modify-write (or a read-then-delete) can fail
closed on a concurrent change instead of silently clobbering it. Read the current
token with `get_version()`, pass it back, and the write becomes a
compare-and-set:

```python
from dataknobs_data import ConcurrencyError, Record

token = db.get_version("k")               # opaque, backend-local token
try:
    db.update("k", Record({"v": 2}, id="k"), expected_version=token)
except ConcurrencyError as e:
    # Someone else wrote "k" since we read the token; e.context has
    # {"id", "expected_version", "actual_version"}. Re-read and retry.
    ...
```

Semantics:

- **Opt-in and backward-compatible.** Omitting `expected_version` (the default)
  is an unconditional, last-writer-wins write — byte-identical to prior behavior.
- **`get_version(id)`** returns an opaque token, or `None` when the id does not
  exist. Treat it as backend-local; it is not comparable across backends.
- **`update()` with a token never inserts.** A missing record returns `False`;
  an existing record with a mismatched token raises `ConcurrencyError`.
- **`delete()` with a token** removes the record only if the token still
  matches; a mismatch raises `ConcurrencyError`, and a missing record returns
  `False` (an absent id never conflicts).
- **`upsert()` with a token never inserts.** A missing record is itself a
  conflict and raises `ConcurrencyError`; a mismatched token also raises.

Token source and atomicity by backend:

- **memory** — a per-instance monotonic sequence under the instance lock;
  ABA-safe on every path, including delete→recreate at the same id.
- **PostgreSQL** — the row's `xmin`, enforced server-side with
  `WHERE id = … AND xmin = …` (ABA-safe, atomic across connections).
- **Elasticsearch** — the document's `_seq_no`/`_primary_term`, enforced
  server-side with `if_seq_no`/`if_primary_term` (ABA-safe).
- **S3** — the object's `ETag`, enforced with a conditional PUT/DELETE
  (`If-Match`) against any store that honors it (real AWS S3, recent LocalStack).
- **file, SQLite, DuckDB** — a deterministic content hash of the stored record;
  the check is serialized within a single connection/instance. A content hash is
  subject to the classic **ABA** limitation (an A→B→A cycle yields the original
  token) and is not hardened across separate processes/connections — use a
  native-token backend when either matters.

#### Capability advertisement

The backends advertise their optional consistency features through the
`CapabilityContract` surface, so a consumer can query support before relying on
a behavior instead of knowing the backend matrix out-of-band. Every backend
enforces the conditional-write contract, so `AsyncDatabase` / `SyncDatabase` and
all 14 backends report `Capability.CONDITIONAL_WRITE`:

```python
from dataknobs_common import Capability, require_capability

if db.supports(Capability.CONDITIONAL_WRITE):
    token = db.get_version("k")
    db.update("k", record, expected_version=token)

# or fail-closed at a boundary:
require_capability(db, Capability.CONDITIONAL_WRITE)
```

The advertisement is uniform because every backend enforces the contract; the
ABA nuance of the content-hash backends (above) is documented rather than
encoded as a separate capability.

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

### Enhanced Upsert (New)

The `upsert` method now supports a more intuitive API that can accept just a Record object, leveraging the Record's built-in ID management:

```python
from dataknobs_data import Database, Record

async def example():
    db = await Database.create("memory")
    
    # Traditional usage (still supported)
    record = Record({"name": "Alice", "age": 30})
    await db.upsert("user-123", record)
    
    # New: Upsert with record that has an ID field
    record_with_id = Record({"id": "user-456", "name": "Bob", "age": 25})
    await db.upsert(record_with_id)  # Uses record's ID
    
    # New: Auto-generate ID if record has no ID
    record_no_id = Record({"name": "Charlie", "age": 35})
    generated_id = await db.upsert(record_no_id)  # Returns generated UUID
    print(f"Generated ID: {generated_id}")
    
    # ID Priority (when using new signature):
    # 1. record.storage_id (if set)
    # 2. record.id (from 'id' field in data)
    # 3. Generated UUID
```

This enhancement is available across all database backends:
- Memory (AsyncMemoryDatabase, SyncMemoryDatabase)
- File (AsyncFileDatabase, SyncFileDatabase)
- SQLite (AsyncSQLiteDatabase, SyncSQLiteDatabase)
- PostgreSQL (AsyncPostgresDatabase, SyncPostgresDatabase)
- Elasticsearch (AsyncElasticsearchDatabase, SyncElasticsearchDatabase)
- S3 (AsyncS3Database)

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
Any([Pattern(r"^\d+$"), Pattern(r"^[A-Z]+$")])  # At least one must pass

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

Define reversible operations to transform records:

```python
from datetime import datetime
from dataknobs_data.migration import (
    AddField, RemoveField, RenameField, TransformField, CompositeOperation,
)

# Add a field (uses default_value when the field is absent)
add_op = AddField("status", default_value="active")

# Remove a field
remove_op = RemoveField("deprecated_field")

# Rename a field
rename_op = RenameField("old_name", "new_name")

# Transform a field's value (pass reverse_fn to keep it reversible)
def uppercase(value):
    return value.upper() if value else value
transform_op = TransformField("name", uppercase)

# Composite operation applies its members in order
composite = CompositeOperation([
    AddField("created_at", default_value=datetime.now()),
    RemoveField("temp_field"),
    RenameField("user_name", "username"),
])
```

### Migrations

A `Migration` is an ordered, reversible set of operations between two versions:

```python
from dataknobs_data.migration import Migration, AddField, RemoveField

# Create a migration (from_version, to_version, optional description)
migration = Migration(
    from_version="1.0.0",
    to_version="2.0.0",
    description="Add status field to user records",
)

# Add operations (fluent — add() returns the migration)
migration.add(AddField("status", default_value="active"))
migration.add(RemoveField("legacy_field"))

# Apply to a record (returns the migrated Record)
record = Record({"name": "Alice", "legacy_field": "old"})
migrated = migration.apply(record)

# Reverse it by applying the operations backwards
original = migration.apply(migrated, reverse=True)
```

### Migrator

`Migrator` is a stateless orchestrator; the source and target are passed per
call. See the [Migration guide](migration.md) for the streaming methods
(`migrate_stream` / `migrate_parallel` / `migrate_async`) and the full
conflict-policy API.

```python
from dataknobs_data.migration import Migrator

migrator = Migrator()

# A transform may be a Transformer or a Migration; here we evolve schema.
migration = Migration(from_version="1.0.0", to_version="2.0.0")
migration.add(AddField("version", default_value=2))

# Batched migration is synchronous.
progress = migrator.migrate(
    source_db,
    target_db,
    transform=migration,
    query=Query().filter("type", Operator.EQ, "user"),
    batch_size=100,
    on_progress=lambda p: print(f"Progress: {p.percent:.0f}%"),
)

print(f"Migrated: {progress.succeeded}")
print(f"Failed: {progress.failed}")
print(f"Duration: {progress.duration:.2f}s")

# Conflict policy for idempotent re-runs into a populated target:
progress = migrator.migrate(source_db, target_db, on_conflict="upsert")
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

### SQLite Backend

SQLite database storage with full SQL capabilities:

```python
# In-memory database
db = await Database.create("sqlite", {
    "path": ":memory:"
})

# File-based database
db = await Database.create("sqlite", {
    "path": "/data/app.db",
    "journal_mode": "WAL",  # Better concurrency
    "synchronous": "NORMAL"  # Balance safety/speed
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