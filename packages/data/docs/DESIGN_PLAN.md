# DataKnobs Data Package Design Plan

## Overview
A unified data abstraction layer that provides consistent database operations across multiple storage technologies. This package enables seamless data management regardless of the underlying storage mechanism, from in-memory structures to cloud storage and databases.

## Core Principles
1. **Technology Agnostic**: Uniform interface regardless of backend storage
2. **Record-Based**: Data represented as collections of structured records
3. **Pandas Integration**: Leverage pandas for powerful data manipulation
4. **Extensible**: Easy to add new storage backends
5. **Simple Interface**: Intuitive API that hides complexity

## Architecture

### 1. Record Abstraction

#### Record Structure
```python
Record = {
    "fields": OrderedDict[str, Field],
    "metadata": Dict[str, Any]
}

Field = {
    "name": str,
    "value": Any,
    "type": FieldType,
    "metadata": Dict[str, Any]
}

FieldType = Enum(
    "STRING", "INTEGER", "FLOAT", "BOOLEAN", 
    "DATETIME", "JSON", "BINARY", "TEXT"
)
```

#### Record Operations
- Field access by name or index
- Type conversion and validation
- Serialization/deserialization
- Metadata management

### 2. Database Interface

#### Abstract Base Class
```python
class Database(ABC):
    @abstractmethod
    async def create(self, record: Record) -> str
    
    @abstractmethod
    async def read(self, id: str) -> Optional[Record]
    
    @abstractmethod
    async def update(self, id: str, record: Record) -> bool
    
    @abstractmethod
    async def delete(self, id: str) -> bool
    
    @abstractmethod
    async def search(self, query: Query) -> List[Record]
    
    @abstractmethod
    async def exists(self, id: str) -> bool
    
    @abstractmethod
    async def upsert(self, id: str, record: Record) -> str
```

#### Synchronous Variant
```python
class SyncDatabase(ABC):
    # Same methods but without async/await
```

### 3. Query System

#### Query Structure
```python
Query = {
    "filters": List[Filter],
    "sort": List[SortSpec],
    "limit": Optional[int],
    "offset": Optional[int],
    "fields": Optional[List[str]]  # Field projection
}

Filter = {
    "field": str,
    "operator": Operator,
    "value": Any
}

Operator = Enum(
    "EQ", "NEQ", "GT", "GTE", "LT", "LTE",
    "IN", "NOT_IN", "LIKE", "REGEX",
    "EXISTS", "NOT_EXISTS"
)

SortSpec = {
    "field": str,
    "order": "ASC" | "DESC"
}
```

### 4. Storage Backends

All backends inherit from `ConfigurableBase` to ensure compatibility with the DataKnobs config system, allowing for:
- Configuration-based instantiation via `Config.build()`
- Factory pattern support via `FactoryBase`
- Consistent construction patterns across all backends

#### Memory Backend
- In-memory dictionary storage
- Fast for testing and small datasets
- Optional persistence to JSON/pickle
- Inherits from `ConfigurableBase`

#### File Backend
- JSON, CSV, Parquet file storage
- Atomic writes with temporary files
- Optional compression support
- File locking for concurrent access
- Inherits from `ConfigurableBase`

#### S3 Backend
- AWS S3 object storage
- Prefix-based organization
- Metadata as object tags
- Batch operations support
- Inherits from `ConfigurableBase`

#### PostgreSQL Backend
- Leverages existing `sql_utils`
- JSONB for flexible schema
- Index support for performance
- Transaction support
- Inherits from `ConfigurableBase`

#### Elasticsearch Backend
- Leverages existing `elasticsearch_utils`
- Full-text search capabilities
- Aggregation support
- Real-time indexing
- Inherits from `ConfigurableBase`

### 5. Serialization System

#### Type Converters
- Automatic type detection
- Custom serializers/deserializers
- Format-specific handlers (JSON, Parquet, etc.)
- Metadata preservation

### 6. Connection Management

#### Connection Pool
```python
class ConnectionPool:
    def acquire(self) -> Connection
    def release(self, conn: Connection)
    def close_all(self)
```

#### Resource Management
- Context managers for automatic cleanup
- Retry logic with exponential backoff
- Connection health checks

## API Design

### Basic Usage
```python
from dataknobs_data import Database, Record, Query

# Create database instance
db = Database.create("memory")  # or "file", "s3", "postgres", "elasticsearch"

# Create a record
record = Record({
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com"
})

# CRUD operations
id = await db.create(record)
retrieved = await db.read(id)
await db.update(id, updated_record)
await db.delete(id)

# Search
query = Query()
    .filter("age", ">=", 25)
    .filter("name", "LIKE", "John%")
    .sort("age", "DESC")
    .limit(10)

results = await db.search(query)
```

### Pandas Integration
```python
# Convert to/from pandas
df = records_to_dataframe(records)
records = dataframe_to_records(df)

# Batch operations
db.create_batch(df)
results_df = db.search_as_dataframe(query)
```

### Backend Configuration

#### Direct Instantiation
```python
# File backend
db = Database.create("file", {
    "path": "/data/records.json",
    "format": "json",
    "compression": "gzip"
})

# PostgreSQL backend
db = Database.create("postgres", {
    "host": "localhost",
    "database": "mydb",
    "table": "records",
    "schema": {"id": "UUID", "data": "JSONB"}
})

# S3 backend
db = Database.create("s3", {
    "bucket": "my-bucket",
    "prefix": "records/",
    "region": "us-west-2"
})
```

#### Config Package Integration
All database backends inherit from `ConfigurableBase` to support the DataKnobs config system:

```python
from dataknobs_config import Config

# Define configuration
config = Config()
config.load({
    "databases": {
        "primary": {
            "class": "dataknobs_data.backends.postgres.PostgresDatabase",
            "host": "localhost",
            "database": "mydb",
            "table": "records"
        },
        "cache": {
            "class": "dataknobs_data.backends.memory.MemoryDatabase"
        },
        "archive": {
            "class": "dataknobs_data.backends.s3.S3Database",
            "bucket": "my-archive",
            "prefix": "records/"
        }
    }
})

# Build databases from config
primary_db = config.build("databases.primary")
cache_db = config.build("databases.cache")
archive_db = config.build("databases.archive")
```

#### Factory Pattern
The backend factory inherits from `FactoryBase` to support dynamic backend creation:

```python
from dataknobs_data import DatabaseFactory

# Using factory with config
config.load({
    "database_factory": {
        "factory": "dataknobs_data.DatabaseFactory"
    },
    "database_config": {
        "type": "postgres",
        "host": "localhost",
        "database": "mydb"
    }
})

factory = config.build("database_factory")
db = factory.create(**config.get("database_config"))
```

## Implementation Phases

### Phase 1: Core Abstractions (Week 1)
1. Define Record and Field classes
2. Create Database abstract base class
3. Implement Query system
4. Set up package structure and tests

### Phase 2: Memory Backend (Week 1)
1. Implement in-memory storage
2. Add all CRUD operations
3. Implement search functionality
4. Create comprehensive tests

### Phase 3: File Backend (Week 2)
1. JSON serialization support
2. CSV and Parquet formats
3. File locking mechanism
4. Compression support

### Phase 4: Database Backends (Week 2-3)
1. PostgreSQL integration using sql_utils
2. Elasticsearch integration using elasticsearch_utils
3. Connection pooling
4. Transaction support

### Phase 5: Cloud Storage (Week 3)
1. S3 backend implementation
2. Batch operations
3. Metadata as tags
4. Cost optimization features

### Phase 6: Advanced Features (Week 4)
1. Async/await support
2. Migration utilities
3. Schema validation
4. Performance optimizations

## Testing Strategy

### Unit Tests
- Record operations
- Serialization/deserialization
- Query building
- Each backend in isolation

### Integration Tests
- Cross-backend operations
- Data migration
- Connection management
- Error handling

### Performance Tests
- Benchmark CRUD operations
- Search performance
- Batch operation efficiency
- Memory usage profiling

## Dependencies
```toml
[project]
dependencies = [
    "pandas>=2.0.0",
    "pydantic>=2.0.0",  # For data validation
    "aiofiles>=23.0.0",  # For async file operations
]

[project.optional-dependencies]
postgres = ["psycopg2>=2.9.0", "sqlalchemy>=2.0.0"]
s3 = ["boto3>=1.26.0", "aioboto3>=11.0.0"]
elasticsearch = ["elasticsearch>=8.0.0"]
```

## File Structure
```
packages/data/
├── README.md
├── DESIGN_PLAN.md (this file)
├── PROGRESS_CHECKLIST.md
├── pyproject.toml
├── src/
│   └── dataknobs_data/
│       ├── __init__.py
│       ├── abstractions.py      # Base classes and interfaces
│       ├── records.py           # Record implementation
│       ├── fields.py            # Field types and validation
│       ├── database.py          # Database interface
│       ├── query.py             # Query system
│       ├── serializers.py       # Type conversion
│       ├── connections.py       # Connection management
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── memory.py
│       │   ├── file.py
│       │   ├── s3.py
│       │   ├── postgres.py
│       │   └── elasticsearch.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── pandas.py       # Pandas integration
│       │   └── migrations.py   # Data migration utilities
│       └── exceptions.py
└── tests/
    ├── conftest.py
    ├── test_records.py
    ├── test_fields.py
    ├── test_query.py
    ├── test_serializers.py
    ├── test_backends/
    │   ├── test_memory.py
    │   ├── test_file.py
    │   ├── test_s3.py
    │   ├── test_postgres.py
    │   └── test_elasticsearch.py
    └── fixtures/
        ├── sample_records.json
        └── test_data.csv
```

## Migration Path

### From RecordStore
```python
# Old code
from dataknobs_structures import RecordStore
store = RecordStore("data.tsv")

# New code
from dataknobs_data import Database
db = Database.create("file", {"path": "data.tsv", "format": "tsv"})
```

### From Direct Database Access
```python
# Old code
conn = psycopg2.connect(...)
cursor = conn.cursor()
cursor.execute("SELECT * FROM table WHERE age > %s", (25,))

# New code
db = Database.create("postgres", config)
results = await db.search(Query().filter("age", ">", 25))
```

## Success Metrics
1. **Performance**: Operations within 10% of native backend performance
2. **Compatibility**: Support for 5+ storage backends
3. **Adoption**: Used by 3+ other dataknobs packages
4. **Reliability**: 99.9% test coverage
5. **Documentation**: Complete API docs with examples