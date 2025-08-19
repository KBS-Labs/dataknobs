# Database Backends

The dataknobs-data package supports multiple storage backends, each optimized for different use cases. All backends implement the same `Database` interface, making it easy to switch between them or use multiple backends in the same application.

## Backend Comparison

| Feature | Memory | File | SQLite | PostgreSQL | Elasticsearch | S3 |
|---------|--------|------|--------|------------|---------------|-----|
| **Persistence** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Query Performance** | ⚡ Instant | 🚶 Slow | ⚡ Fast | ⚡ Fast | ⚡ Fast | 🐌 Very Slow |
| **Scalability** | 📦 Limited | 📦 Limited | 📦 Limited | 🌐 High | 🌐 High | ♾️ Unlimited |
| **Full-text Search** | ❌ | ❌ | 🔍 Basic | 🔍 Basic | 🔍 Advanced | ❌ |
| **ACID Compliance** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Cost** | Free | Free | Free | 💰 Low | 💰 Medium | 💰 Per GB |
| **Setup Complexity** | None | None | None | 🔧 Medium | 🔧 High | 🔧 Medium |
| **Concurrent Access** | ✅ | ❌ | 🔄 Limited | ✅ | ✅ | ✅ |
| **Embedded** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

## Memory Backend

In-memory storage for testing and caching.

```python
from dataknobs_data.backends.memory import MemoryDatabase

db = MemoryDatabase()
# Or with configuration
db = MemoryDatabase.from_config({})
```

**Best for:**
- Unit testing
- Temporary caching
- Development environments
- Small datasets (<1GB)

**Limitations:**
- Data lost on restart
- Limited by available RAM
- No concurrent access from multiple processes

## File Backend

Store data in JSON, CSV, or Parquet files.

```python
from dataknobs_data.backends.file import FileDatabase

# JSON storage
db = FileDatabase.from_config({
    "path": "/data/records.json",
    "format": "json"
})

# CSV storage
db = FileDatabase.from_config({
    "path": "/data/records.csv",
    "format": "csv"
})

# Parquet storage (requires pyarrow)
db = FileDatabase.from_config({
    "path": "/data/records.parquet",
    "format": "parquet"
})
```

**Best for:**
- Small to medium datasets
- Simple persistence needs
- Data exchange/export
- Prototyping

**Limitations:**
- Poor query performance (full scan)
- No concurrent writes
- Limited to single machine

## SQLite Backend

Lightweight, embedded SQL database with full ACID compliance and zero configuration.

```python
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

# Synchronous in-memory database
db = SyncSQLiteDatabase({"path": ":memory:"})
db.connect()

# Synchronous file-based database
db = SyncSQLiteDatabase({
    "path": "/data/app.db",
    "journal_mode": "WAL",  # Better concurrency
    "synchronous": "NORMAL"  # Balance safety/speed
})
db.connect()

# Asynchronous database
async_db = AsyncSQLiteDatabase({
    "path": "/data/app.db",
    "pool_size": 5  # Connection pool for async
})
await async_db.connect()
```

**Best for:**
- Embedded applications
- Desktop applications
- Mobile apps
- Development and testing
- Small to medium datasets
- Applications requiring SQL without server setup

**Features:**
- Zero configuration - no server required
- ACID transactions
- SQL query support with JSON functions
- Small footprint (~1MB)
- Cross-platform
- WAL mode for better concurrency
- In-memory option for testing

**Limitations:**
- Single-writer, multiple-reader concurrency
- Limited to single machine
- Database size limited by disk space
- No built-in replication

**Configuration Options:**
```python
db = SyncSQLiteDatabase({
    "path": "/path/to/database.db",  # or ":memory:"
    "table": "records",  # Table name (default: "records")
    "timeout": 5.0,  # Connection timeout in seconds
    "journal_mode": "WAL",  # WAL, DELETE, TRUNCATE, PERSIST, MEMORY, OFF
    "synchronous": "NORMAL",  # FULL, NORMAL, OFF
})
```

## PostgreSQL Backend

Full-featured SQL database with ACID compliance.

```python
from dataknobs_data.backends.postgres import PostgresDatabase

db = PostgresDatabase.from_config({
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "user",
    "password": "pass",
    "table": "records"  # Optional, defaults to "records"
})
```

**Best for:**
- Production applications
- Transactional data
- Complex queries
- Multi-user applications

**Features:**
- ACID transactions
- Indexes for fast queries
- SQL query support
- Concurrent access
- Backup and recovery

**Installation:**
```bash
pip install dataknobs-data[postgres]
```

## Elasticsearch Backend

Distributed search and analytics engine.

```python
from dataknobs_data.backends.elasticsearch import ElasticsearchDatabase

db = ElasticsearchDatabase.from_config({
    "hosts": ["localhost:9200"],
    "index": "records",
    "username": "elastic",  # Optional
    "password": "pass"      # Optional
})
```

**Best for:**
- Full-text search
- Log analytics
- Real-time data
- Large-scale applications

**Features:**
- Full-text search with relevance scoring
- Aggregations and analytics
- Distributed and scalable
- Near real-time indexing
- RESTful API

**Installation:**
```bash
pip install dataknobs-data[elasticsearch]
```

## S3 Backend

AWS S3 object storage for large-scale data archival.

```python
from dataknobs_data.backends.s3 import S3Database

db = S3Database.from_config({
    "bucket": "my-data-bucket",
    "prefix": "records/",
    "region": "us-east-1",
    "access_key_id": "AKIAIOSFODNN7EXAMPLE",  # Optional
    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # Optional
    "endpoint_url": "http://localhost:4566"  # Optional, for LocalStack
})
```

**Best for:**
- Large file storage
- Long-term archival
- Backup and recovery
- Cost-effective storage

**Features:**
- Unlimited storage capacity
- 99.999999999% durability
- Lifecycle policies
- Versioning support
- Cross-region replication

**Installation:**
```bash
pip install dataknobs-data[s3]
```

## Choosing the Right Backend

### Development Workflow
```python
# Development: Use memory for speed
dev_db = factory.create(backend="memory")

# Testing: Use SQLite for persistence with SQL support
test_db = factory.create(backend="sqlite", path=":memory:")

# Staging: Use SQLite with file persistence
staging_db = factory.create(backend="sqlite", path="/tmp/staging.db")

# Production: Use PostgreSQL or Elasticsearch
prod_db = factory.create(backend="postgres", **db_config)
```

### Hybrid Architecture
```python
# Use multiple backends together
cache = factory.create(backend="memory")
primary = factory.create(backend="postgres", **pg_config)
archive = factory.create(backend="s3", **s3_config)

# Cache frequently accessed data
def get_user(user_id):
    # Check cache first
    user = cache.read(user_id)
    if user:
        return user
    
    # Fetch from primary database
    user = primary.read(user_id)
    if user:
        # Store in cache for next time
        cache.create(user)
    return user

# Archive old data
def archive_old_records():
    # Find old records
    query = Query().filter("created_at", "<", "2023-01-01")
    old_records = primary.search(query)
    
    # Move to S3
    for record in old_records:
        archive.create(record)
        primary.delete(record.metadata["id"])
```

## Migration Between Backends

Easily migrate data between different backends:

```python
def migrate_data(source_db, dest_db):
    """Migrate all data from source to destination."""
    # Get all records from source
    all_records = source_db.search(Query())
    
    # Batch create in destination
    dest_db.batch_create(all_records)
    
    print(f"Migrated {len(all_records)} records")

# Example: Migrate from SQLite to PostgreSQL
sqlite_db = factory.create(backend="sqlite", path="local.db")
pg_db = factory.create(backend="postgres", **pg_config)
migrate_data(sqlite_db, pg_db)

# Example: Migrate from file to SQLite
file_db = factory.create(backend="file", path="data.json")
sqlite_db = factory.create(backend="sqlite", path="app.db")
migrate_data(file_db, sqlite_db)
```

## Performance Tips

### Memory Backend
- Pre-allocate capacity if known
- Use weak references for large objects
- Clear cache periodically

### File Backend
- Use Parquet for better performance
- Compress large JSON files
- Consider splitting into multiple files

### SQLite Backend
- Use WAL mode for better concurrency
- Enable memory-mapped I/O for read-heavy workloads
- Use transactions for batch operations
- Consider PRAGMA optimizations for your use case
- Use in-memory databases for temporary data

### PostgreSQL Backend
- Create indexes on frequently queried fields
- Use connection pooling
- Batch operations when possible

### Elasticsearch Backend
- Tune index settings for your use case
- Use bulk API for batch operations
- Implement proper mapping for fields

### S3 Backend
- Use batch operations to reduce API calls
- Enable multipart upload for large files
- Cache frequently accessed objects locally
- Use prefixes for logical grouping