# DataKnobs Data Package Architecture

## Overview

The DataKnobs Data Package provides a unified abstraction layer for database operations across multiple storage technologies. It enables consistent data access patterns whether you're working with in-memory storage, files, SQL databases, NoSQL stores, or cloud object storage.

## Core Design Principles

1. **Unified API**: Single interface for all storage backends
2. **Type Safety**: Strong typing with runtime validation
3. **Async-First**: Native async/await support with sync fallbacks
4. **Configuration-Driven**: Seamless integration with dataknobs-config
5. **Extensible**: Easy to add new backends and features
6. **Performance-Focused**: Optimized for both single operations and batch processing

## Architecture Components

### 1. Core Abstractions

#### Record Model
```python
Record(
    data: Dict[str, Any],          # Core data fields
    storage_id: Optional[str],     # System-assigned storage ID
    metadata: Dict = {},           # Additional metadata
    created_at: datetime,          # Auto-set timestamp
    updated_at: datetime           # Auto-updated timestamp
)
```

**Features:**
- Dict-like access: `record["field"]`
- Attribute access: `record.field`
- Automatic timestamp management
- Metadata support for annotations
- Type validation and coercion
- **Dual ID system**: Separates user IDs from storage IDs (see [Record ID Architecture](RECORD_ID_ARCHITECTURE.md))
- **Advanced serialization**: Handles complex types including vectors (see [Record Serialization](RECORD_SERIALIZATION.md))

#### Database Interface
```python
class Database(ABC):
    # Lifecycle
    async def connect() -> None
    async def disconnect() -> None
    
    # CRUD Operations
    async def create(record: Record) -> str
    async def read(id: str) -> Optional[Record]
    async def update(id: str, record: Record) -> bool
    async def delete(id: str) -> bool
    
    # Batch Operations
    async def create_batch(records: List[Record]) -> List[str]
    async def read_batch(ids: List[str]) -> List[Optional[Record]]
    
    # Query Operations
    async def search(query: Query) -> List[Record]
    async def count(query: Optional[Query]) -> int
    
    # Streaming
    async def stream(query: Query, config: StreamConfig) -> AsyncIterator[StreamResult]
```

### 2. Query System

#### Simple Queries
```python
Query()
    .filter("status", Operator.EQ, "active")
    .filter("age", Operator.GT, 18)
    .sort("created_at", SortOrder.DESC)
    .limit(100)
    .offset(20)
```

#### Complex Boolean Logic
```python
Query().or_(
    Filter("city", Operator.EQ, "New York"),
    Filter("city", Operator.EQ, "Los Angeles")
).filter("active", Operator.EQ, True)
```

#### Supported Operators
- **Comparison**: EQ, NE, GT, GTE, LT, LTE
- **Range**: BETWEEN, IN, NOT_IN
- **Pattern**: LIKE, NOT_LIKE
- **Null**: IS_NULL, NOT_NULL
- **Logic**: AND (implicit), OR, NOT

### 3. Backend Implementations

#### Memory Backend
- **Use Case**: Testing, caching, temporary data
- **Features**: Thread-safe, fast, no persistence
- **Performance**: ~500K ops/sec for simple operations

#### File Backend
- **Use Case**: Local persistence, data export/import
- **Formats**: JSON, CSV, Parquet
- **Features**: Atomic writes, compression, file locking
- **Performance**: ~10K ops/sec (SSD)

#### PostgreSQL Backend
- **Use Case**: Relational data, ACID compliance
- **Features**: 
  - Connection pooling (asyncpg)
  - Transaction support
  - JSON field support
  - Index optimization
- **Performance**: ~5K ops/sec (local)

#### Elasticsearch Backend
- **Use Case**: Full-text search, analytics
- **Features**:
  - Bulk operations
  - Native async client
  - Index management
  - Query DSL translation
- **Performance**: ~10K ops/sec (bulk)

#### S3 Backend
- **Use Case**: Cloud storage, data archiving
- **Features**:
  - Parallel uploads/downloads
  - Metadata as tags
  - Cost optimization
  - Index caching
- **Performance**: ~1K ops/sec (depends on network)

### 4. Advanced Features

#### Streaming API
```python
async for result in db.stream(query, StreamConfig(chunk_size=1000)):
    for record in result.records:
        process(record)
    print(f"Progress: {result.progress.percentage:.1f}%")
```

#### Schema Validation
```python
schema = Schema(
    fields=[
        FieldDefinition("email", str, constraints=[
            EmailConstraint(),
            RequiredConstraint()
        ]),
        FieldDefinition("age", int, constraints=[
            RangeConstraint(min=0, max=150)
        ])
    ]
)

validator = Validator(schema)
result = validator.validate(record)
```

#### Data Migration
```python
migrator = Migrator(
    source=postgres_db,
    target=elasticsearch_db,
    transformer=Transformer(mappings={
        "old_field": "new_field"
    })
)

await migrator.migrate(
    query=Query(),
    on_progress=lambda p: print(f"{p.percentage:.1f}%")
)
```

#### Pandas Integration
```python
# DataFrame to Records
records = converter.from_dataframe(df)
await db.create_batch(records)

# Records to DataFrame
records = await db.search(query)
df = converter.to_dataframe(records)
```

## Connection Management

### Configuration-Based
```python
# Using dataknobs-config
db = Database.from_config(config)
await db.connect()  # Auto-connect

# Direct instantiation
db = PostgresDatabase(
    host="localhost",
    database="mydb"
)
await db.connect()
```

### Connection Pooling
- **PostgreSQL**: asyncpg native pooling
- **Elasticsearch**: AsyncElasticsearch client pooling
- **S3**: aioboto3 session management
- All backends support connection reuse and automatic reconnection

## Performance Characteristics

| Backend | Create | Read | Update | Delete | Search | Batch Create |
|---------|--------|------|--------|---------|---------|--------------|
| Memory | 500K/s | 1M/s | 400K/s | 600K/s | 100K/s | 2M/s |
| File | 10K/s | 50K/s | 8K/s | 20K/s | 5K/s | 30K/s |
| PostgreSQL | 5K/s | 20K/s | 4K/s | 10K/s | 10K/s | 50K/s |
| Elasticsearch | 3K/s | 15K/s | 2K/s | 5K/s | 50K/s | 100K/s |
| S3 | 1K/s | 2K/s | 800/s | 1.5K/s | 500/s | 10K/s |

*Note: Performance varies based on hardware, network, and data size*

## Error Handling

```python
from dataknobs_data.exceptions import (
    DatabaseError,          # Base exception
    ConnectionError,        # Connection issues
    NotFoundError,         # Record not found
    ValidationError,       # Data validation failed
    QueryError,           # Invalid query
    ConfigurationError    # Invalid configuration
)
```

## Best Practices

### 1. Connection Lifecycle
```python
async with Database.from_config(config) as db:
    # Connection automatically managed
    await db.create(record)
```

### 2. Batch Operations
```python
# Prefer batch operations for multiple records
ids = await db.create_batch(records)  # Faster

# Over individual operations
for record in records:
    await db.create(record)  # Slower
```

### 3. Query Optimization
```python
# Use specific filters
query = Query().filter("status", Operator.EQ, "active")

# Add indexes for frequently queried fields
# Use projections to limit returned fields
query = query.project(["id", "name", "status"])
```

### 4. Error Recovery
```python
try:
    record = await db.read(id)
except NotFoundError:
    # Handle missing record
    record = create_default_record()
except DatabaseError as e:
    # Handle database errors
    logger.error(f"Database error: {e}")
    # Implement retry logic if appropriate
```

## Future Enhancements

### Planned Features
- **Caching Layer**: Redis integration for query caching
- **GraphQL Support**: Query translation from GraphQL
- **Vector Search**: Support for embedding-based search
- **Time-Series**: Specialized time-series backend
- **Replication**: Multi-backend replication support

### Extension Points
- Custom backends via `Database` base class
- Custom operators via `Operator` enum extension
- Custom validators via `Constraint` base class
- Custom transformers via `Transformer` interface

## Integration with DataKnobs Ecosystem

### Config Package
```python
from dataknobs_config import Config

config = Config.from_file("config.yaml")
db = Database.from_config(config.get_section("database"))
```

### Utils Package
```python
from dataknobs_utils import RequestHelper

# Reuses utility components
helper = RequestHelper(host, port)
```

## Summary

The DataKnobs Data Package provides a robust, performant, and extensible foundation for data operations across diverse storage backends. Its unified API, comprehensive feature set, and seamless integration with the DataKnobs ecosystem make it an ideal choice for applications requiring flexible data management capabilities.
