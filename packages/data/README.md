# DataKnobs Data Package

A unified data abstraction layer that provides consistent database operations across multiple storage technologies.

## Overview

The `dataknobs-data` package enables seamless data management regardless of the underlying storage mechanism, from in-memory structures to cloud storage and databases. It provides a simple, consistent API for CRUD operations, searching, and data manipulation across diverse backends.

## Features

- **Unified Interface**: Same API regardless of storage backend
- **Multiple Backends**: Memory, File (JSON/CSV/Parquet), PostgreSQL, Elasticsearch, S3
- **Record-Based**: Data represented as structured records with metadata
- **Pandas Integration**: Seamless conversion to/from DataFrames
- **Type Safety**: Strong typing with Pydantic validation
- **Async Support**: Both synchronous and asynchronous APIs
- **Query System**: Powerful, backend-agnostic query capabilities
- **Configuration Support**: Full integration with DataKnobs configuration system
- **Extensible**: Easy to add custom storage backends

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
from dataknobs_data import Database, Record, Query

# Create a database instance (memory backend for testing)
db = Database.create("memory")

# Create and store a record
record = Record({
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com",
    "active": True
})

# CRUD operations
record_id = db.create(record)
retrieved = db.read(record_id)
db.update(record_id, updated_record)
db.delete(record_id)

# Search with queries
query = (Query()
    .filter("age", ">=", 25)
    .filter("active", "=", True)
    .sort("name", "ASC")
    .limit(10))

results = db.search(query)
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

```python
import pandas as pd
from dataknobs_data import Database

db = Database.create("memory")

# Create records from DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "Chicago"]
})

records = db.from_dataframe(df)
db.create_batch(records)

# Query and get results as DataFrame
query = Query().filter("age", ">", 25)
results_df = db.search_as_dataframe(query)
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

## Migration Between Backends

```python
from dataknobs_data import Database, migrate_data

# Source and destination databases
source_db = Database.create("file", {"path": "data.json"})
dest_db = Database.create("postgres", postgres_config)

# Migrate all data
migrate_data(source_db, dest_db)

# Migrate with transformation
def transform(record):
    record["processed"] = True
    return record

migrate_data(source_db, dest_db, transform=transform)
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