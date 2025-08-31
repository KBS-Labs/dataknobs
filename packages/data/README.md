# DataKnobs Data Package

A unified data abstraction layer that provides consistent database operations across multiple storage technologies.

**Version**: 0.1.0  
**Status**: Released ([PyPI](https://pypi.org/project/dataknobs-data/))  
**Python**: 3.10+  
**License**: MIT  

## Overview

The `dataknobs-data` package enables seamless data management regardless of the underlying storage mechanism, from in-memory structures to cloud storage and databases. It provides a simple, consistent API for CRUD operations, searching, and data manipulation across diverse backends.

## Features

### Core Capabilities
- **Unified Interface**: Same API regardless of storage backend
- **Multiple Backends**: Memory, File (JSON/CSV/Parquet), SQLite, PostgreSQL, Elasticsearch, S3
- **Record-Based**: Data represented as structured records with metadata and first-class ID support
- **Type Safety**: Strong typing with field validation and automatic type conversion
- **Async Support**: Both synchronous and asynchronous APIs

### Advanced Query Features (v0.1.0)
- **Boolean Logic**: Complex queries with AND, OR, NOT operators
- **Range Operators**: BETWEEN, IN, NOT_IN for efficient range queries
- **Ergonomic Field Access**: Both dict-like (`record["field"]`) and attribute (`record.field`) access
- **Query Builder**: Fluent API for complex query construction
- **Null Handling**: IS_NULL, NOT_NULL operators

### Data Processing
- **Pandas Integration**: Seamless bidirectional conversion to/from DataFrames with type preservation
- **Migration Utilities**: Backend-to-backend migration, schema evolution, and data transformation
- **Schema Validation**: Comprehensive validation system with constraints and type coercion
- **Streaming Support**: Efficient streaming APIs for large datasets
- **Batch Operations**: Efficient bulk insert, update, and upsert operations

### Infrastructure
- **Configuration Support**: Full integration with DataKnobs configuration system
- **Connection Management**: Automatic connection lifecycle management with pooling
- **Quality Assurance**: Comprehensive test suite with quality artifacts pattern
- **Extensible**: Easy to add custom storage backends, validators, and transformers

## Installation

```bash
# Basic installation
pip install dataknobs-data

# With specific backend support
pip install dataknobs-data[postgres]     # PostgreSQL support
pip install dataknobs-data[s3]          # AWS S3 support
pip install dataknobs-data[elasticsearch] # Elasticsearch support
pip install dataknobs-data[all]         # All backends
```

## Quick Start

```python
from dataknobs_data import AsyncDatabase, Record, Query, Operator

# Async usage
async def main():
    # Create and auto-connect to database
    db = await AsyncDatabase.from_backend("memory")
    
    # Create a record
    record = Record({
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "active": True
    })
    
    # CRUD operations
    id = await db.create(record)
    retrieved = await db.read(id)
    record.set_value("age", 31)
    await db.update(id, record)
    await db.delete(id)
    
    # Search with queries
    query = (Query()
        .filter("age", Operator.GTE, 25)
        .filter("active", Operator.EQ, True)
        .sort("name")
        .limit(10))
    
    results = await db.search(query)
    for record in results:
        print(f"{record.get_value('name')}: {record.get_value('age')}")
    
    await db.close()

# Synchronous usage
from dataknobs_data import SyncDatabase

db = SyncDatabase.from_backend("memory")
record = Record({"name": "Jane Doe", "age": 28})
id = db.create(record)
retrieved = db.read(id)
db.close()
```

## Backend Configuration

### File Backend
```python
db = await Database.create("file", {
    "path": "/data/records.json",
    "pretty": True,
    "backup": True
})
```

### SQLite Backend
```python
db = await Database.create("sqlite", {
    "path": "app.db",  # or ":memory:" for in-memory
    "journal_mode": "WAL",
    "synchronous": "NORMAL"
})
```

### PostgreSQL Backend
```python
db = await Database.create("postgres", {
    "host": "localhost",
    "database": "mydb",
    "user": "user",
    "password": "pass",
    "table": "records",
    "schema": "public"
})
```

### S3 Backend
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
```python
db = await Database.create("elasticsearch", {
    "host": "localhost",
    "port": 9200,
    "index": "records",
    "refresh": True
})
```

## Configuration Support

The data package fully integrates with the DataKnobs configuration system. All backends inherit from `ConfigurableBase` and can be instantiated from configuration files.

### Using Configuration Files

```yaml
# config.yaml
databases:
  - name: primary
    class: dataknobs_data.backends.postgres.PostgresDatabase
    host: ${DB_HOST:localhost}  # Environment variable with default
    port: ${DB_PORT:5432}
    database: myapp
    user: ${DB_USER:postgres}
    password: ${DB_PASSWORD}
    table: records
    
  - name: cache
    class: dataknobs_data.backends.memory.MemoryDatabase
    
  - name: archive
    class: dataknobs_data.backends.file.SyncFileDatabase
    path: /data/archive.json
    format: json
    compression: gzip
    
  - name: cloud_storage
    class: dataknobs_data.backends.s3.S3Database
    bucket: ${S3_BUCKET:my-data-bucket}
    prefix: ${S3_PREFIX:records/}
    region: ${AWS_REGION:us-east-1}
    endpoint_url: ${S3_ENDPOINT}  # Optional, for LocalStack/MinIO
```

### Loading from Configuration

```python
from dataknobs_config import Config
from dataknobs_data import Record, Query

# Load configuration
config = Config("config.yaml")

# Create database instances from config
primary_db = config.get_instance("databases", "primary")
cache_db = config.get_instance("databases", "cache")
archive_db = config.get_instance("databases", "archive")

# Use the databases normally
record = Record({"name": "test", "value": 42})
record_id = primary_db.create(record)

# Cache frequently accessed data
cache_db.create(record)

# Archive old records
archive_db.create(record)
```

### Direct Configuration

```python
from dataknobs_data.backends.postgres import PostgresDatabase

# All backends support from_config classmethod
db = PostgresDatabase.from_config({
    "host": "localhost",
    "database": "myapp",
    "user": "postgres",
    "password": "secret"
})
```

## Backend Factory

The data package provides a factory pattern for dynamic backend selection:

### Using the Factory Directly

```python
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

# Create different backends
memory_db = factory.create(backend="memory")
file_db = factory.create(backend="file", path="data.json", format="json")
s3_db = factory.create(backend="s3", bucket="my-bucket", prefix="data/")
```

### Factory with Configuration

```python
from dataknobs_config import Config
from dataknobs_data import database_factory

# Register factory for cleaner configs
config = Config()
config.register_factory("database", database_factory)

# Use registered factory in configuration
config.load({
    "databases": [{
        "name": "main",
        "factory": "database",  # Uses registered factory
        "backend": "postgres",
        "host": "localhost",
        "database": "myapp"
    }]
})

db = config.get_instance("databases", "main")
```

### Factory Configuration Examples

```yaml
# Using registered factory (cleaner)
databases:
  - name: main
    factory: database
    backend: ${DB_BACKEND:postgres}
    host: ${DB_HOST:localhost}
    
# Using module path (no registration needed)
databases:
  - name: main
    factory: dataknobs_data.factory.database_factory
    backend: postgres
    host: localhost
```

## Pandas Integration

The data package provides comprehensive pandas integration for data analysis workflows:

```python
import pandas as pd
from dataknobs_data.pandas import DataFrameConverter, BatchOperations

# Convert records to DataFrame with type preservation
converter = DataFrameConverter()
df = converter.records_to_dataframe(records, preserve_types=True)

# Perform pandas operations
df_filtered = df[df['age'] > 25]
df_aggregated = df.groupby('category').agg({'price': 'mean'})

# Convert back to records
new_records = converter.dataframe_to_records(df_filtered)

# Bulk operations with DataFrames
batch_ops = BatchOperations(database)
result = batch_ops.bulk_insert_dataframe(df, batch_size=1000)
print(f"Inserted {result.successful} records")

# Upsert from DataFrame
result = batch_ops.bulk_upsert_dataframe(
    df, 
    id_column="user_id",
    merge_strategy="update"
)
```

## Schema Validation

Define and enforce data schemas with comprehensive validation:

```python
from dataknobs_data.validation import Schema, FieldType
from dataknobs_data.validation.constraints import *

# Define schema with constraints
user_schema = Schema("UserSchema")
user_schema.field("email", FieldType.STRING, 
    required=True,
    constraints=[Pattern(r"^.+@.+\..+$"), Unique()])
user_schema.field("age", FieldType.INTEGER,
    constraints=[Range(min=0, max=150)])
user_schema.field("status", FieldType.STRING,
    default="active",
    constraints=[Enum(["active", "inactive", "suspended"])])

# Validate records
result = user_schema.validate(record)
if not result.valid:
    for error in result.errors:
        print(error)

# Automatic type coercion
record = Record({"age": "30"})  # String value
result = user_schema.validate(record, coerce=True)  # Converts to int
if result.valid:
    print(record.get_value("age"))  # 30 (as integer)
```

## Data Migration

Migrate data between backends with transformation support:

```python
from dataknobs_data.migration import Migration, Migrator
from dataknobs_data.migration.operations import *

# Define migration
migration = Migration("upgrade_schema", "2.0.0")
migration.add_operation(AddField("created_at", default=datetime.now()))
migration.add_operation(RenameField("user_name", "username"))
migration.add_operation(TransformField("email", lambda x: x.lower()))

# Migrate between backends
async def migrate_data():
    source_db = await Database.create("postgres", postgres_config)
    target_db = await Database.create("s3", s3_config)
    
    migrator = Migrator(source_db, target_db)
    
    # Run migration with progress tracking
    progress = await migrator.migrate(
        migration=migration,
        batch_size=1000,
        on_progress=lambda p: print(f"Progress: {p.percentage:.1f}%")
    )
    
    print(f"Migrated: {progress.successful} records")
    print(f"Failed: {progress.failed} records")
    print(f"Duration: {progress.duration}s")
    
    await source_db.close()
    await target_db.close()
```

## Advanced Queries

```python
# Complex query with multiple filters
query = (Query()
    .filter("status", Operator.IN, ["active", "pending"])
    .filter("created_at", Operator.GTE, "2024-01-01")
    .filter("name", Operator.LIKE, "John%")
    .sort("priority", SortOrder.DESC)
    .sort("created_at", SortOrder.ASC)
    .offset(20)
    .limit(10)
    .select(["name", "email", "status"]))  # Select specific fields

results = await db.search(query)
```

## Streaming Support

```python
from dataknobs_data import StreamConfig

# Stream large datasets efficiently
config = StreamConfig(
    batch_size=100,
    buffer_size=1000
)

# Stream read
async for record in db.stream_read(query, config):
    # Process each record without loading all into memory
    process_record(record)

# Stream write
result = await db.stream_write(record_generator(), config)
print(f"Streamed {result.total_processed} records")
```


## Documentation

For complete API documentation, see [API Reference](docs/API_REFERENCE.md).

## Custom Backend

```python
from dataknobs_data import AsyncDatabase, DatabaseBackend

class CustomBackend(DatabaseBackend):
    def create(self, record):
        # Implementation
        pass
    
    def read(self, record_id):
        # Implementation
        pass
    
    # ... other methods

# Register custom backend
AsyncDatabase.register_backend("custom", CustomBackend)

# Use custom backend
db = AsyncDatabase.from_backend("custom", config)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=dataknobs_data

# Type checking
mypy src/dataknobs_data

# Linting
ruff check src/dataknobs_data

# Format code
black src/dataknobs_data
```

## Architecture

The package follows a modular architecture:

- **Records**: Data representation with fields and metadata
- **Database Interface**: Abstract base classes (AsyncDatabase/SyncDatabase) for all backends
- **Query System**: Backend-agnostic query building
- **Backends**: Implementations for different storage technologies
- **Serializers**: Type conversion and format handling
- **Utils**: Pandas integration and migration tools

## Performance

The package is designed for optimal performance:

- Connection pooling for database backends
- Batch operations for efficiency
- Lazy loading and pagination
- Caching for frequently accessed data
- Async support for concurrent operations

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.