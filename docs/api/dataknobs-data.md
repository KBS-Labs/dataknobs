# DataKnobs Data API Reference

Complete API documentation for the `dataknobs-data` package.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/data.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/data/src/dataknobs_data) - Browse on GitHub
> - [Package Guide](../packages/data/index.md) - Detailed documentation

Every database class here comes in a pair, and the name says which half you
have: `SyncMemoryDatabase` and `AsyncMemoryDatabase`, `SyncPostgresDatabase`
and `AsyncPostgresDatabase`, and so on for all seven backends. There is no
unprefixed spelling — the base classes are `SyncDatabase` and `AsyncDatabase`,
and a backend implements one of them.

## Core Classes

### `dataknobs_data.Record`

Represents a data record with fields and metadata.

```python
class Record:
    def __init__(self, data: dict[str, Any] | OrderedDict[str, Field] | None = None,
                 metadata: dict[str, Any] | None = None,
                 id: str | None = None,
                 storage_id: str | None = None)
    def get_value(self, name: str, default: Any = None) -> Any
    def set_value(self, name: str, value: Any) -> None
    def to_dict(self, include_metadata: bool = False, flatten: bool = True,
                include_field_objects: bool = True) -> dict[str, Any]
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record
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

# to_dict() returns the fields alone unless you ask for the metadata,
# which arrives under a "_metadata" key rather than merged in
record.to_dict()                       # {"name": "Alice", "age": 31, ...}
record.to_dict(include_metadata=True)  # ... plus {"_metadata": {"created_at": ...}}
```

`record.id` is `None` until a database assigns one. `create()` returns the id
it assigned rather than mutating the record you passed it.

### `dataknobs_data.Query`

A query with filters, sorting, pagination, boolean combination and vector
search.

```python
class Query:
    def filter(self, field: str, operator: str | Operator, value: Any = None) -> Query
    def sort(self, field: str, order: str | SortOrder = "asc") -> Query
    def limit(self, value: int) -> Query
    def offset(self, value: int) -> Query
    def select(self, *fields: str) -> Query
    def and_(self, *filters: Filter | Query) -> Query
    def or_(self, *filters: Filter | Query) -> ComplexQuery
    def not_(self, filter: Filter) -> ComplexQuery
    def similar_to(self, vector: np.ndarray | list[float], field: str = "embedding",
                   k: int = 10, metric: DistanceMetric | str = "cosine",
                   include_source: bool = True,
                   score_threshold: float | None = None) -> Query
    def to_dict(self) -> dict[str, Any]
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Query
```

`or_` and `not_` return a `ComplexQuery` rather than a `Query`, because a
query whose conditions nest is no longer a flat list of filters. `and_` stays
a `Query`, since that is what a flat list already means.

**Example:**
```python
from dataknobs_data import Query

query = (Query()
    .filter("age", ">", 25)
    .filter("active", "=", True)
    .sort("name", "asc")
    .limit(10))
```

Operators and sort orders may be given as strings or as the enum members
below; the strings are the enum *values*, so `">"` and `Operator.GT` are the
same argument.

### `dataknobs_data.QueryBuilder`

A fluent builder for the same `ComplexQuery` that `Query.or_` and `Query.not_`
produce. Use it when the boolean structure is what you are expressing;
`Query` is the shorter road when it is not.

```python
class QueryBuilder:
    def where(self, field: str, operator: str | Operator, value: Any = None) -> QueryBuilder
    def and_(self, *conditions) -> QueryBuilder
    def or_(self, *conditions) -> QueryBuilder
    def not_(self, condition) -> QueryBuilder
    def select(self, *fields: str) -> QueryBuilder
    def sort_by(self, field: str, order: str = "asc") -> QueryBuilder
    def limit(self, value: int) -> QueryBuilder
    def offset(self, value: int) -> QueryBuilder
    def similar_to(self, vector: np.ndarray | list[float], field: str = "embedding",
                   k: int = 10, metric: DistanceMetric | str = "cosine",
                   include_source: bool = True,
                   score_threshold: float | None = None) -> QueryBuilder
    def build(self) -> ComplexQuery
```

**Example:**
```python
from dataknobs_data import QueryBuilder

complex_query = (QueryBuilder()
    .where("age", ">", 25)
    .sort_by("name")
    .limit(10)
    .build())
```

## Database Interface

### `dataknobs_data.SyncDatabase`

Abstract base class for the synchronous backends.

```python
class SyncDatabase(ABC):
    # Abstract -- a backend must implement these
    @abstractmethod
    def create(self, record: Record) -> str
    @abstractmethod
    def read(self, id: str) -> Record | None
    @abstractmethod
    def update(self, id: str, record: Record, *,
               expected_version: str | None = None) -> bool
    @abstractmethod
    def delete(self, id: str, *, expected_version: str | None = None) -> bool
    @abstractmethod
    def exists(self, id: str) -> bool
    @abstractmethod
    def search(self, query: Query | ComplexQuery) -> list[Record]
    @abstractmethod
    def stream_read(self, query: Query | None = None,
                    config: StreamConfig | None = None) -> Iterator[Record]
    @abstractmethod
    def stream_write(self, records: Iterator[Record],
                     config: StreamConfig | None = None) -> StreamResult

    # Provided -- a backend inherits these and may override for efficiency
    def count(self, query: Query | None = None) -> int
    def all(self) -> list[Record]
    def clear(self) -> int
    def upsert(self, id_or_record: str | Record, record: Record | None = None, *,
               expected_version: str | None = None) -> str
    def create_batch(self, records: list[Record]) -> list[str]
    def read_batch(self, ids: list[str]) -> list[Record | None]
    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]
    def delete_batch(self, ids: list[str]) -> list[bool]
    def upsert_batch(self, records: list[Record]) -> list[str]
    def connect(self) -> None
    def close(self) -> None
```

The batch methods are named `create_batch` and so on, not `batch_create`.

### `dataknobs_data.AsyncDatabase`

The same interface with `async def` throughout: `await db.create(record)`,
`await db.read(id)`, and `async for record in db.stream_read()`.

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase

db = AsyncMemoryDatabase()
record_id = await db.create(Record({"name": "Alice"}))
found = await db.read(record_id)

async for record in db.stream_read():
    ...
```

## Factory Pattern

### `dataknobs_data.DatabaseFactory`

Factory for creating database instances.

```python
class DatabaseFactory(FactoryBase):
    def create(self, **config) -> SyncDatabase
    def get_available_backends(self) -> list[str]
    def get_backend_info(self, backend_type: str) -> dict[str, Any]
    def is_backend_available(self, backend_type: str) -> bool
```

Registration is not a factory method — backends live in the registry the
factory reads, so a custom one is added there and every factory method
picks it up:

```python
from dataknobs_data import register_backend
from dataknobs_data.backends import sync_backends

register_backend(
    sync_backends,
    "my_backend",
    lambda: MyDatabase,          # imported on first use, not at registration
    metadata={
        "description": "Custom backend",
        "persistent": True,
        "requires_module": "my_driver",          # probed at registration
        "requires_install": "pip install my-driver",
    },
    aliases=("mine",),
)
```

`register_backend` probes the driver named by `requires_module` and, when
it is absent, records the backend as *known but not creatable* instead of
registering it. That is what keeps `is_backend_available()` an answer
about this machine rather than about the name, and what lets `create()`
say which driver to install rather than reporting the name as unknown.

`sync_backends.register(...)` is the lower-level call underneath it. It
skips the probe, so a backend registered that way reports as available
whether or not its driver is installed — use it only for a backend with no
optional dependency.

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

### `dataknobs_data.async_database_factory`

The same, reading the async registry — `async_database_factory.create(backend="memory")`
returns an `AsyncMemoryDatabase`. Both registries carry all seven backends.

## Backend Implementations

| Backend key | Sync class | Async class | Config class | Needs |
|---|---|---|---|---|
| `memory` | `SyncMemoryDatabase` | `AsyncMemoryDatabase` | `MemoryDatabaseConfig` | — |
| `file` | `SyncFileDatabase` | `AsyncFileDatabase` | `FileDatabaseConfig` | — |
| `sqlite` | `SyncSQLiteDatabase` | `AsyncSQLiteDatabase` | `SyncSQLiteDatabaseConfig` / `AsyncSQLiteDatabaseConfig` | — |
| `duckdb` | `SyncDuckDBDatabase` | `AsyncDuckDBDatabase` | `SyncDuckDBDatabaseConfig` / `AsyncDuckDBDatabaseConfig` | `pip install duckdb` |
| `postgres` | `SyncPostgresDatabase` | `AsyncPostgresDatabase` | `PostgresDatabaseConfig` | `pip install dataknobs-data[postgres]` |
| `elasticsearch` | `SyncElasticsearchDatabase` | `AsyncElasticsearchDatabase` | `SyncElasticsearchDatabaseConfig` / `AsyncElasticsearchDatabaseConfig` | `pip install dataknobs-data[elasticsearch]` |
| `s3` | `SyncS3Database` | `AsyncS3Database` | `SyncS3DatabaseConfig` / `AsyncS3DatabaseConfig` | `pip install dataknobs-data[s3]` |

The config classes live in `dataknobs_data.backends.config`, and each backend
names its own as `CONFIG_CLS`. The async classes for sqlite, elasticsearch and
s3 sit in `_async` modules beside their sync siblings; the other four share a
module with theirs.

Every backend takes the same constructor — a mapping checked against its
config class, or the keyword arguments that mapping would hold:

```python
def __init__(self, config: ConfigT | Mapping[str, Any] | None = None, **kwargs)
```

### `dataknobs_data.backends.memory.SyncMemoryDatabase`

In-memory database for testing and caching.

```python
from dataknobs_data.backends.memory import SyncMemoryDatabase

db = SyncMemoryDatabase()
# or
db = SyncMemoryDatabase.from_config({"vector_enabled": True})
```

Config keys: `schema`, `vector_enabled`, `vector_metric`.

### `dataknobs_data.backends.file.SyncFileDatabase`

File-based storage supporting JSON, CSV, and Parquet formats.

```python
from dataknobs_data.backends.file import SyncFileDatabase

db = SyncFileDatabase(path="data.json", format="json")
# or
db = SyncFileDatabase.from_config({
    "path": "data.csv",
    "format": "csv"
})
```

Config keys: the three shared ones, plus `path`, `format`, `compression`.

### `dataknobs_data.backends.sqlite.SyncSQLiteDatabase`

SQLite, with vector support and no optional driver to install.

```python
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

db = SyncSQLiteDatabase.from_config({
    "path": "app.db",
    "table": "records"
})
```

Config keys: the three shared ones, plus `path`, `table`, `timeout`,
`journal_mode`, `synchronous`, `auto_create_table`, `check_same_thread`.

### `dataknobs_data.backends.duckdb.SyncDuckDBDatabase`

DuckDB, for analytical queries over the same record interface.

```python
from dataknobs_data.backends.duckdb import SyncDuckDBDatabase

db = SyncDuckDBDatabase.from_config({
    "path": "analytics.duckdb",
    "table": "records"
})
```

Config keys: `schema`, `path`, `table`, `timeout`, `read_only`,
`auto_create_table`.

**Installation:**
```bash
pip install duckdb
```

### `dataknobs_data.backends.postgres.SyncPostgresDatabase`

PostgreSQL database with full SQL support.

```python
from dataknobs_data.backends.postgres import SyncPostgresDatabase

db = SyncPostgresDatabase.from_config({
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "dbuser",
    "password": "dbpass",
    "table": "records"
})
```

Config keys: the three shared ones, plus `host`, `port`, `database`, `user`,
`password`, `ssl`, `command_timeout`, `min_pool_size`, `max_pool_size`,
`table`, `schema_name`, `ensure_database`, `auto_create_table`.

**Installation:**
```bash
pip install dataknobs-data[postgres]
```

### `dataknobs_data.backends.elasticsearch.SyncElasticsearchDatabase`

Elasticsearch for full-text search and analytics.

```python
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase

db = SyncElasticsearchDatabase.from_config({
    "host": "localhost",
    "port": 9200,
    "index": "myindex"
})
```

Config keys: the three shared ones, plus `index`, `refresh`, `host`, `port`,
`vector_dimensions`, `default_vector_field`, `mappings`, `settings`. The async
class takes a different set — `hosts` alongside `host`/`port`, plus `api_key`,
`basic_auth` and the TLS keys — and each class checks the keys it declares, so
a key that works for one is rejected by the other.

**Installation:**
```bash
pip install dataknobs-data[elasticsearch]
```

### `dataknobs_data.backends.s3.SyncS3Database`

AWS S3 object storage backend.

```python
from dataknobs_data.backends.s3 import SyncS3Database

db = SyncS3Database.from_config({
    "bucket": "my-bucket",
    "prefix": "data/",
    "region_name": "us-east-1",
    "endpoint_url": "http://localhost:4566",  # For LocalStack
})
```

Config keys: the three shared ones, plus `bucket`, `region_name`,
`aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`,
`endpoint_url`, `prefix`, `multipart_threshold`, `multipart_chunksize`,
`max_pool_connections`, `max_attempts`, `retry_mode`, `extra_client_kwargs`.

The region key is `region_name`, matching boto3, not `region`.

**Installation:**
```bash
pip install dataknobs-data[s3]
```

## Configuration Support

Backends take their configuration through `StructuredConfigConsumer`, which
validates the mapping against the backend's `CONFIG_CLS`. An unknown key is
rejected rather than silently ignored:

```python
from dataknobs_data.backends.postgres import SyncPostgresDatabase

SyncPostgresDatabase.CONFIG_CLS      # -> PostgresDatabaseConfig

db = SyncPostgresDatabase.from_config(
    {"host": "localhost", "database": "myapp", "table": "records"}
)
```

`from_dict` is a method on the config class, not on the database class. For a
class of your own, see the
[configuration system guide](../development/configuration-system.md).

## Exceptions

These are the shared hierarchy from `dataknobs_common.exceptions`, specialised
here — so a consumer catching `NotFoundError` or `OperationError` catches the
ones below that derive from them, and one catching `DataknobsError` catches
every one. `DataknobsDataError` is a backward-compatible alias for that root
and not a separate base: it is the same object as
`dataknobs_common.exceptions.DataknobsError`. All of the names below are
exported from `dataknobs_data` directly.

```python
from dataknobs_data import (
    BackendNotFoundError,
    DatabaseConnectionError,
    DatabaseOperationError,
    DuplicateRecordError,
    QueryError,
    RecordNotFoundError,
    RecordValidationError,
)
```

| Exception | Base | Raised when |
|---|---|---|
| `DataknobsDataError` | `Exception` | alias for `DataknobsError`, the root of every dataknobs error |
| `DatabaseConnectionError` | `ResourceError` | a backend cannot reach its store |
| `DatabaseOperationError` | `OperationError` | an operation fails at the backend |
| `QueryError` | `OperationError` | a query cannot be executed |
| `TransactionError` | `OperationError` | a transaction fails or is misused |
| `MigrationError` | `OperationError` | a section migration fails |
| `RecordNotFoundError` | `NotFoundError` | no record has the given id |
| `BackendNotFoundError` | `NotFoundError` | the backend key is unknown or its driver is absent |
| `RecordValidationError` | `ValidationError` | a record does not satisfy its schema |
| `FieldTypeError` | `ValidationError` | a field-type operation fails |
| `DuplicateRecordError` | `ConcurrencyError`, `ValueError` | `create()` targets an id that already exists |
| `ConcurrencyError` | `ConcurrencyError` | a concurrency conflict occurs — an `expected_version` no longer matching is one |
| `ConfigurationError` | `ConfigurationError` | a backend's configuration is invalid |
| `SerializationError` | `SerializationError` | a record cannot be encoded or decoded |

The last three carry the same names as their `dataknobs_common` bases, so
which of the two an `except` clause names decides whether it catches this
package's error alone or every package's.

## Constants

```python
from dataknobs_data import (
    DEFAULT_BACKEND,             # "memory"
    DEFAULT_MAX_ATTEMPTS,        # 16
    RESERVED_KEY_FIELD,          # "id"
    VALID_TRANSACTION_POLICIES,  # ("strict", "emulate")
)
```

## Enums

### `dataknobs_data.Operator`

`EQ`, `NEQ`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NOT_IN`, `LIKE`, `NOT_LIKE`,
`REGEX`, `STARTS_WITH`, `EXISTS`, `NOT_EXISTS`, `BETWEEN`, `NOT_BETWEEN`.

The values are the strings a query accepts in place of the member: `"="`,
`"!="`, `">"`, `">="`, `"<"`, `"<="`, `"in"`, `"not_in"`, `"like"`,
`"not_like"`, `"regex"`, `"starts_with"`, `"exists"`, `"not_exists"`,
`"between"`, `"not_between"`.

### `dataknobs_data.SortOrder`

`ASC` and `DESC`, whose values are `"asc"` and `"desc"`.

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
    .sort("name", "asc")
    .limit(10)
)
```
