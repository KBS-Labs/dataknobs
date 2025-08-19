# DataKnobs Data Package

The `dataknobs-data` package provides a unified data abstraction layer that works seamlessly across multiple storage backends. Whether you're working with in-memory caches, local files, SQL databases, search engines, or cloud storage, the data package offers a consistent API for all your data management needs.

## Key Features

- **Unified Interface**: Same API works with all backends
- **Multiple Backends**: Memory, File, SQLite, PostgreSQL, Elasticsearch, S3
- **Async Connection Pooling**: Event loop-aware pooling for 5-10x performance
- **Native Async Support**: Uses asyncpg, AsyncElasticsearch, and aioboto3
- **Configuration Support**: Full integration with dataknobs-config
- **Factory Pattern**: Dynamic backend selection at runtime
- **Environment Variables**: Automatic substitution in configurations
- **Batch Operations**: Efficient bulk operations for all backends
- **Query System**: Consistent querying across different storage types
- **Stream Processing**: Memory-efficient handling of large datasets

## Installation

```bash
# Core package
pip install dataknobs-data

# With specific backend support
pip install dataknobs-data[postgres]    # PostgreSQL support
pip install dataknobs-data[elasticsearch]  # Elasticsearch support
pip install dataknobs-data[s3]          # S3 support
pip install dataknobs-data[all]         # All backends
```

## Quick Start

```python
from dataknobs_data import Record, Query, DatabaseFactory

# Create a database using factory
factory = DatabaseFactory()
db = factory.create(backend="memory")

# Create and store a record
record = Record({
    "name": "Alice",
    "age": 30,
    "department": "Engineering"
})
record_id = db.create(record)

# Query data
query = Query().filter("department", "=", "Engineering")
results = db.search(query)

# Update record
record.fields["age"] = 31
db.update(record_id, record)

# Delete record
db.delete(record_id)
```

## Configuration Integration

The data package fully integrates with dataknobs-config for configuration-based instantiation:

```yaml
databases:
  - name: primary
    factory: database
    backend: ${DB_BACKEND:postgres}
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    database: ${DB_NAME:myapp}
    
  - name: cache
    factory: database
    backend: memory
    
  - name: archive
    factory: database
    backend: s3
    bucket: ${S3_BUCKET}
    prefix: records/
```

```python
from dataknobs_config import Config
from dataknobs_data import database_factory

config = Config("config.yaml")
config.register_factory("database", database_factory)

# Get configured instances
primary_db = config.get_instance("databases", "primary")
cache_db = config.get_instance("databases", "cache")
archive_db = config.get_instance("databases", "archive")
```

## Available Backends

| Backend | Description | Persistent | Use Case |
|---------|-------------|------------|----------|
| Memory | In-memory storage | No | Caching, testing |
| File | JSON/CSV/Parquet files | Yes | Small datasets, prototyping |
| SQLite | Embedded SQL database | Yes | Desktop apps, testing, prototyping |
| PostgreSQL | SQL database server | Yes | Production applications |
| Elasticsearch | Search engine | Yes | Full-text search, analytics |
| S3 | Object storage | Yes | Large files, archival |

## Performance Highlights

With the new async connection pooling system:

- **Elasticsearch**: 70% faster bulk operations with native AsyncElasticsearch
- **S3**: 5.3x faster batch uploads with aioboto3
- **PostgreSQL**: 3.2x faster bulk inserts with asyncpg
- **Zero "Event loop is closed" errors** with event loop-aware pooling

!!! tip "Quick Start with Async Pooling"
    Check out the [Async Pooling Quick Start](async-pooling-quickstart.md) guide for immediate performance improvements!

## Next Steps

- [Async Connection Pooling](async-pooling.md) - High-performance async operations
- [Performance Tuning](performance-tuning.md) - Optimization strategies
- [Backends Overview](backends.md) - Learn about all available backends
- [SQLite Backend](sqlite-backend.md) - Embedded SQL database
- [PostgreSQL Backend](postgres-backend.md) - Full-featured SQL server
- [S3 Backend](s3-backend.md) - Cloud storage with AWS S3
- [Factory Pattern](factory-pattern.md) - Dynamic backend selection
- [Configuration](configuration.md) - Advanced configuration options
- [Examples](examples.md) - Complete working examples
- [API Reference](api.md) - Detailed API documentation