# DataKnobs Data Package

A unified data abstraction layer that provides consistent database operations across multiple storage technologies.

## Overview

The `dataknobs-data` package enables seamless data management regardless of the underlying storage mechanism, from in-memory structures to cloud storage and databases. It provides a simple, consistent API for CRUD operations, searching, and data manipulation across diverse backends.

## Features

- **Unified Interface**: Same API regardless of storage backend
- **Multiple Backends**: Memory, File (JSON/CSV/Parquet), PostgreSQL, Elasticsearch, S3
- **Record-Based**: Data represented as structured records with metadata and first-class ID support
- **Pandas Integration**: Seamless bidirectional conversion to/from DataFrames with type preservation
- **Migration Utilities**: Backend-to-backend migration, schema evolution, and data transformation
- **Schema Validation**: Comprehensive validation system with constraints and type coercion
- **Type Safety**: Strong typing with field validation and automatic type conversion
- **Async Support**: Both synchronous and asynchronous APIs
- **Query System**: Powerful, backend-agnostic query capabilities
- **Configuration Support**: Full integration with DataKnobs configuration system
- **Batch Operations**: Efficient bulk insert, update, and upsert operations
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
from dataknobs_data import Record, Field, FieldType
from dataknobs_data.backends.memory import MemoryDatabase

# Create a database instance
db = MemoryDatabase()

# Create a record with automatic ID generation
record = Record(
    fields={
        "name": Field("name", FieldType.STRING, "John Doe"),
        "age": Field("age", FieldType.INTEGER, 30),
        "email": Field("email", FieldType.STRING, "john@example.com"),
        "active": Field("active", FieldType.BOOLEAN, True)
    }
)
print(record.id)  # Auto-generated UUID

# CRUD operations
db.create(record)
retrieved = db.read(record.id)
record.fields["age"].value = 31
db.update(record.id, record)
db.delete(record.id)

# Search with queries
from dataknobs_data import Query, Filter, Sort

query = Query(
    filters=[
        Filter("age", ">=", 25),
        Filter("active", "=", True)
    ],
    sort=[Sort("name", "asc")],
    limit=10
)

results = db.search(query)
for record in results:
    print(f"{record.id}: {record.fields['name'].value}")
```

## Backend Configuration

### File Backend
```python
db = Database.create("file", {
    "path": "/data/records.json",
    "format": "json",  # or "csv", "parquet"
    "compression": "gzip"  # optional
})
```

### PostgreSQL Backend
```python
db = Database.create("postgres", {
    "host": "localhost",
    "database": "mydb",
    "user": "user",
    "password": "pass",
    "table": "records"
})
```

### S3 Backend
```python
db = Database.create("s3", {
    "bucket": "my-bucket",
    "prefix": "records/",
    "region": "us-west-2"
})
```

### Elasticsearch Backend
```python
db = Database.create("elasticsearch", {
    "hosts": ["localhost:9200"],
    "index": "records",
    "doc_type": "_doc"
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
from dataknobs_data.validation import Schema, FieldDefinition
from dataknobs_data.validation.constraints import *

# Define schema with constraints
user_schema = Schema(
    name="UserSchema",
    fields={
        "email": FieldDefinition(
            name="email",
            type=str,
            required=True,
            constraints=[EmailConstraint(), UniqueConstraint()]
        ),
        "age": FieldDefinition(
            name="age",
            type=int,
            constraints=[MinValueConstraint(0), MaxValueConstraint(150)]
        ),
        "status": FieldDefinition(
            name="status",
            type=str,
            default="active",
            constraints=[EnumConstraint(["active", "inactive", "suspended"])]
        )
    }
)

# Validate records
result = user_schema.validate(record)
if not result.is_valid:
    for error in result.errors:
        print(f"{error.field}: {error.message}")

# Automatic type coercion
schema_with_coercion = Schema(
    name="ProductSchema",
    fields=fields,
    coerce_types=True  # Automatically convert compatible types
)
```

## Data Migration

Migrate data between backends with transformation support:

```python
from dataknobs_data.migration import DataMigrator, DataTransformer

# Migrate between backends
source_db = Database.create("postgres", postgres_config)
target_db = Database.create("s3", s3_config)

migrator = DataMigrator(source_db, target_db)

# Simple migration
result = migrator.migrate_sync(batch_size=1000)
print(f"Migrated {result.successful_records} records")

# Migration with transformation
transformer = DataTransformer(
    field_mapping={"old_name": "new_name"},
    value_transformers={
        "email": lambda v: v.lower(),
        "age": lambda v: int(v) if v else 0
    }
)

result = migrator.migrate_sync(
    transform=transformer.transform,
    progress_callback=lambda p: print(f"Progress: {p.percentage:.1f}%")
)

# Schema evolution
from dataknobs_data.migration import SchemaEvolution

evolution = SchemaEvolution("user_schema")
evolution.add_version("1.0.0", schema_v1)
evolution.add_version("2.0.0", schema_v2)

# Auto-generate migration
migration = evolution.generate_migration("1.0.0", "2.0.0")
migrated_records = migration.apply(records)
```

## Advanced Queries

```python
# Complex query with multiple filters
query = (Query()
    .filter("status", "IN", ["active", "pending"])
    .filter("created_at", ">=", "2024-01-01")
    .filter("name", "LIKE", "John%")
    .sort([("priority", "DESC"), ("created_at", "ASC")])
    .offset(20)
    .limit(10)
    .project(["name", "email", "status"]))  # Select specific fields

results = db.search(query)
```

## Async Support

```python
import asyncio
from dataknobs_data import AsyncDatabase

async def main():
    db = AsyncDatabase.create("postgres", config)
    
    # Async CRUD operations
    record_id = await db.create(record)
    retrieved = await db.read(record_id)
    
    # Async search
    results = await db.search(query)
    
asyncio.run(main())
```


## Custom Backend

```python
from dataknobs_data import Database, DatabaseBackend

class CustomBackend(DatabaseBackend):
    def create(self, record):
        # Implementation
        pass
    
    def read(self, record_id):
        # Implementation
        pass
    
    # ... other methods

# Register custom backend
Database.register_backend("custom", CustomBackend)

# Use custom backend
db = Database.create("custom", config)
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
- **Database Interface**: Abstract base class for all backends
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