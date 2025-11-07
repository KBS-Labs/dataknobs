# DataKnobs Data API Reference

Complete API documentation for the `dataknobs-data` package.

> **💡 Quick Links:**
> - [Package Guide](../packages/data/index.md) - Tutorials and examples
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/data/src/dataknobs_data) - View on GitHub
> - [API Index](complete-reference.md) - All packages

## Core Classes

### `dataknobs_data.Record`

Represents a data record with fields and metadata.

```python
class Record:
    def __init__(self, fields: Dict[str, Any] = None)
    def get_value(self, field_name: str, default: Any = None) -> Any
    def set_value(self, field_name: str, value: Any) -> None
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Record'
```

**Example:**
```python
from dataknobs_data import Record

record = Record({
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
})

# Access fields
name = record.get_value("name")  # "Alice"
record.set_value("age", 31)

# Metadata
record.metadata["created_at"] = "2024-01-01"
```

### `dataknobs_data.Query`

Query builder for database operations.

```python
class Query:
    def filter(self, field: str, operator: str, value: Any) -> 'Query'
    def sort(self, field: str, order: str = "ASC") -> 'Query'
    def limit(self, limit: int) -> 'Query'
    def offset(self, offset: int) -> 'Query'
    def project(self, fields: List[str]) -> 'Query'
```

**Operators:**
- `=`, `!=`: Equality
- `>`, `>=`, `<`, `<=`: Comparison
- `IN`, `NOT IN`: Membership
- `LIKE`: Pattern matching (% wildcard)

**Example:**
```python
from dataknobs_data import Query

query = (Query()
    .filter("age", ">", 25)
    .filter("active", "=", True)
    .sort("name", "ASC")
    .limit(10))
```

## Database Interface

### `dataknobs_data.Database`

Abstract base class for all database implementations.

```python
class Database(ABC):
    @abstractmethod
    def create(self, record: Record) -> str
    @abstractmethod
    def read(self, record_id: str) -> Optional[Record]
    @abstractmethod
    def update(self, record_id: str, record: Record) -> bool
    @abstractmethod
    def delete(self, record_id: str) -> bool
    @abstractmethod
    def search(self, query: Query) -> List[Record]
    @abstractmethod
    def count(self, query: Optional[Query] = None) -> int
    @abstractmethod
    def clear(self) -> None
    
    # Batch operations (with default implementations)
    def batch_create(self, records: List[Record]) -> List[str]
    def batch_read(self, record_ids: List[str]) -> List[Optional[Record]]
    def batch_update(self, updates: List[Tuple[str, Record]]) -> List[bool]
    def batch_delete(self, record_ids: List[str]) -> List[bool]
```

## Factory Pattern

### `dataknobs_data.DatabaseFactory`

Factory for creating database instances.

```python
class DatabaseFactory(FactoryBase):
    def create(self, **config) -> Database
    def get_available_backends(self) -> List[str]
    def get_backend_info(self, backend: str) -> Dict[str, Any]
    def is_backend_available(self, backend: str) -> bool
    def register_backend(self, name: str, backend_class: Type[Database]) -> None
```

**Example:**
```python
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

# Create different backends
memory_db = factory.create(backend="memory")
file_db = factory.create(backend="file", path="data.json")
pg_db = factory.create(backend="postgres", host="localhost", database="myapp")

# Get backend info
info = factory.get_backend_info("s3")
print(info["description"])
print(info["requires_install"])
```

### `dataknobs_data.database_factory`

Pre-instantiated factory instance for convenience.

```python
from dataknobs_data import database_factory

db = database_factory.create(backend="memory")
```

## Backend Implementations

### `dataknobs_data.backends.memory.MemoryDatabase`

In-memory database for testing and caching.

```python
class MemoryDatabase(Database, ConfigurableBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Configuration:**
```python
db = MemoryDatabase()
# or
db = MemoryDatabase.from_config({})
```

### `dataknobs_data.backends.file.FileDatabase`

File-based storage supporting JSON, CSV, and Parquet formats.

```python
class FileDatabase(Database, ConfigurableBase):
    def __init__(self, path: str = None, format: str = "json", 
                 config: Optional[Dict[str, Any]] = None)
```

**Configuration:**
```python
db = FileDatabase(path="data.json", format="json")
# or
db = FileDatabase.from_config({
    "path": "data.csv",
    "format": "csv"
})
```

### `dataknobs_data.backends.postgres.PostgresDatabase`

PostgreSQL database with full SQL support.

```python
class PostgresDatabase(Database, ConfigurableBase):
    def __init__(self, host: str = None, port: int = None,
                 database: str = None, user: str = None,
                 password: str = None, table: str = "records",
                 config: Optional[Dict[str, Any]] = None)
```

**Configuration:**
```python
db = PostgresDatabase.from_config({
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "dbuser",
    "password": "dbpass",
    "table": "records"
})
```

**Installation:**
```bash
pip install dataknobs-data[postgres]
```

### `dataknobs_data.backends.elasticsearch.ElasticsearchDatabase`

Elasticsearch for full-text search and analytics.

```python
class ElasticsearchDatabase(Database, ConfigurableBase):
    def __init__(self, hosts: List[str] = None, index: str = "records",
                 username: str = None, password: str = None,
                 config: Optional[Dict[str, Any]] = None)
```

**Configuration:**
```python
db = ElasticsearchDatabase.from_config({
    "hosts": ["localhost:9200"],
    "index": "myindex",
    "username": "elastic",
    "password": "password"
})
```

**Installation:**
```bash
pip install dataknobs-data[elasticsearch]
```

### `dataknobs_data.backends.s3.S3Database`

AWS S3 object storage backend.

```python
class S3Database(Database, ConfigurableBase):
    def __init__(self, bucket: str = None, prefix: str = "",
                 region: str = "us-east-1", endpoint_url: str = None,
                 access_key_id: str = None, secret_access_key: str = None,
                 max_workers: int = 10,
                 config: Optional[Dict[str, Any]] = None)
```

**Configuration:**
```python
db = S3Database.from_config({
    "bucket": "my-bucket",
    "prefix": "data/",
    "region": "us-east-1",
    "endpoint_url": "http://localhost:4566",  # For LocalStack
    "max_workers": 10
})
```

**Installation:**
```bash
pip install dataknobs-data[s3]
```

## Configuration Support

All backends inherit from `ConfigurableBase`:

```python
from dataknobs_config import ConfigurableBase

class MyDatabase(Database, ConfigurableBase):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MyDatabase':
        return cls(**config)
```

## Exceptions

```python
class DatabaseError(Exception):
    """Base exception for database errors."""

class ConnectionError(DatabaseError):
    """Database connection error."""

class QueryError(DatabaseError):
    """Query execution error."""

class RecordNotFoundError(DatabaseError):
    """Record not found error."""

class BackendNotAvailableError(DatabaseError):
    """Backend not available or not installed."""
```

## Utility Functions

### Type Conversion
```python
def convert_type(value: Any, target_type: type) -> Any:
    """Convert value to target type."""
```

### ID Generation
```python
def generate_id() -> str:
    """Generate a unique record ID (UUID)."""
```

## Constants

```python
# Default values
DEFAULT_BATCH_SIZE = 100
DEFAULT_POOL_SIZE = 10
DEFAULT_TIMEOUT = 30
DEFAULT_CACHE_TTL = 60

# S3 specific
S3_MULTIPART_THRESHOLD = 5 * 1024 * 1024  # 5MB
S3_MAX_WORKERS = 10

# Elasticsearch specific
ES_DEFAULT_INDEX = "records"
ES_BATCH_SIZE = 500

# PostgreSQL specific
PG_DEFAULT_TABLE = "records"
PG_DEFAULT_PORT = 5432
```

## Type Hints

```python
from typing import Dict, Any, List, Optional, Tuple, Type
from typing import Literal

RecordID = str
FieldName = str
FieldValue = Any
QueryOperator = Literal["=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "LIKE"]
SortOrder = Literal["ASC", "DESC"]
```

## Complete Example

```python
from dataknobs_data import Record, Query, DatabaseFactory
from dataknobs_config import Config

# Setup
factory = DatabaseFactory()
config = Config()
config.register_factory("database", factory)

# Configure databases
config.load({
    "databases": [
        {"name": "primary", "factory": "database", "backend": "postgres", 
         "host": "localhost", "database": "myapp"},
        {"name": "cache", "factory": "database", "backend": "memory"},
        {"name": "archive", "factory": "database", "backend": "s3", 
         "bucket": "archive"}
    ]
})

# Get instances
primary_db = config.get_instance("databases", "primary")
cache_db = config.get_instance("databases", "cache")

# Use databases
record = Record({"name": "Alice", "age": 30})
record_id = primary_db.create(record)

# Cache frequently accessed data
cache_db.create(record)

# Query
results = primary_db.search(
    Query()
    .filter("age", ">", 25)
    .sort("name", "ASC")
    .limit(10)
)
```