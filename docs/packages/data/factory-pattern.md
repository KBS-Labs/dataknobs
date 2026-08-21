# Factory Pattern

The DatabaseFactory provides dynamic backend selection and instantiation, making it easy to switch between different storage backends without changing your application code.

## Overview

The factory pattern in dataknobs-data allows you to:
- Create backends dynamically based on configuration
- Switch backends without code changes
- Query backend capabilities and requirements
- Register custom backend implementations
- Handle missing dependencies gracefully

## Basic Usage

```python
from dataknobs_data import DatabaseFactory

# Create factory instance
factory = DatabaseFactory()

# Create different backends
memory_db = factory.create(backend="memory")
file_db = factory.create(backend="file", path="/data/records.json")
pg_db = factory.create(backend="postgres", host="localhost", database="myapp")
s3_db = factory.create(backend="s3", bucket="my-bucket")
```

### When no backend is named

Every example above names a backend. A config that does not is still valid —
it falls back to `memory` — but the factory says so at **WARNING** rather than
letting the fallback pass unremarked:

```
No 'backend' key in this database config; falling back to 'memory'. That
default is in-process and unpersisted -- it answers every query with zero
results until something writes to it, and loses everything when the process
restarts. If this config came from resolving a resource reference, check that
the named resource is defined in this environment.
```

An explicit `backend="memory"` is logged at INFO instead. The two build the
same object, and the level is the only thing that distinguishes a store
somebody chose from one left over from a config that arrived empty — most
often a `$resource` reference to a resource the environment does not define,
which resolves to `{}` unless it is
[declared required](../config/environment-aware.md#4-missing-resources).

The same applies to `AsyncDatabaseFactory` and `VectorStoreFactory`, which
share one selector.

## Configuration-Based Creation

### Using Config Files
```yaml
# config.yaml
databases:
  primary:
    backend: ${DB_BACKEND:postgres}
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    database: ${DB_NAME:myapp}
    
  cache:
    backend: memory
    
  archive:
    backend: s3
    bucket: ${S3_BUCKET}
    prefix: archives/
```

```python
import yaml
from dataknobs_data import DatabaseFactory

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

factory = DatabaseFactory()

# Create databases from config
databases = {}
for name, db_config in config["databases"].items():
    databases[name] = factory.create(**db_config)
```

### With Factory Registration
```python
from dataknobs_config import Config
from dataknobs_data import database_factory

# Register factory with config system
config = Config("config.yaml")
config.register_factory("database", database_factory)

# Now configs can reference the factory
config.load({
    "databases": [{
        "name": "main",
        "factory": "database",  # Uses registered factory
        "backend": "postgres",
        "host": "localhost"
    }]
})

# Get instance
db = config.get_instance("databases", "main")
```

## Backend Information API

Query available backends and their requirements. All three factories —
`DatabaseFactory`, `AsyncDatabaseFactory` and `VectorStoreFactory` — answer
the same three questions about their own registry:

```python
factory = DatabaseFactory()

# Get all available backends
backends = factory.get_available_backends()
print(f"Available backends: {backends}")
# Output: ['duckdb', 'elasticsearch', 'file', 'memory', 'postgres', 's3', 'sqlite']

# Get information about a specific backend
info = factory.get_backend_info("s3")
print(info)
# Output: {
#     'description': 'AWS S3 object storage backend',
#     'persistent': True,
#     'requires_install': 'pip install dataknobs-data[s3]',
#     'requires_module': 'boto3',
#     'vector_support': False,
#     'config_options': {
#         'bucket': 'S3 bucket name (required)',
#         'prefix': 'Object key prefix (default: records/)',
#         ...
#     },
# }

# Check if backend is available
if factory.is_backend_available("postgres"):
    db = factory.create(backend="postgres", **config)
else:
    print("PostgreSQL backend not available")
    print(factory.get_backend_info("postgres")["requires_install"])
    # Install with: pip install dataknobs-data[postgres]
```

**Available means installed.** Registration probes the driver a backend
declares in `requires_module`, so a registered name is one whose optional
dependency is actually present. `is_backend_available("postgres")` is
therefore the check to make before offering it.

That probe is what makes the answer trustworthy, because the backends do
not agree among themselves about when to fail. Some import their driver at
module top level, so a missing driver fails the import; others catch their
own `ImportError` and raise only when you construct one. Asking whether the
module loaded would answer honestly for the first group and optimistically
for the second — `is_backend_available("faiss")` would return `True` on a
machine without `faiss-cpu`, and `create()` would then raise.

**A backend that is missing still describes itself.** `requires_install` is
only ever read by someone who does not have the backend installed, so a
backend whose driver is absent stays *known* rather than disappearing:
`get_backend_info(...)` answers for it, and `create()` reports the missing
driver rather than an unrecognised name.

```python
# On a machine without psycopg2:
factory.is_backend_available("postgres")           # False
"postgres" in factory.get_available_backends()     # False
factory.get_backend_info("postgres")["requires_install"]
# 'pip install dataknobs-data[postgres]'

factory.create(backend="postgres", host="localhost")
# ValueError: Backend 'postgres' is known but not available here.
#             Install with: pip install dataknobs-data[postgres]
```

**The reported list names each backend once.** `create()` accepts
registration aliases — `pg` and `postgresql` for postgres, `es` for
elasticsearch, `mem` for memory, `chromadb` for chroma — but
`get_available_backends()` reports the canonical name alone, so it is a list
of backends rather than a list of spellings. `is_backend_available()` and
`get_backend_info()` still answer for an alias, so every question about `pg`
agrees with the same question about `postgres`.

An unknown name is reported rather than raised:

```python
factory.get_backend_info("no-such-backend")
# {'description': 'Unknown backend', 'error': "Backend 'no-such-backend' not recognized"}
```

## Dynamic Backend Selection

### Environment-Based Selection
```python
import os
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

# Select backend based on environment
env = os.environ.get("APP_ENV", "development")

if env == "production":
    db = factory.create(
        backend="postgres",
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )
elif env == "staging":
    db = factory.create(
        backend="elasticsearch",
        host=os.environ["ES_HOST"],
        port=int(os.environ.get("ES_PORT", "9200")),
        index="staging"
    )
else:  # development
    db = factory.create(
        backend="file",
        path="./dev_data.json"
    )
```

### Feature-Based Selection
```python
def get_database_for_use_case(use_case: str):
    """Select backend based on use case requirements."""
    factory = DatabaseFactory()
    
    if use_case == "caching":
        # Need fast, temporary storage
        return factory.create(backend="memory")
    
    elif use_case == "full_text_search":
        # Need advanced search capabilities
        return factory.create(
            backend="elasticsearch",
            host="localhost",
            port=9200,
            index="search"
        )
    
    elif use_case == "archival":
        # Need cheap, long-term storage
        return factory.create(
            backend="s3",
            bucket="archive-bucket",
            prefix="long-term/"
        )
    
    elif use_case == "transactional":
        # Need ACID compliance
        return factory.create(
            backend="postgres",
            host="localhost",
            database="transactions"
        )
    
    else:
        # Default fallback
        return factory.create(backend="file", path="data.json")
```

## Custom Backend Registration

You can register custom backend implementations:

```python
from dataknobs_data import DatabaseFactory
from dataknobs_data import SyncDatabase
from dataknobs_data.records import Record
from typing import List, Optional

class CustomDatabase(SyncDatabase):
    """Custom database implementation."""
    
    def __init__(self, **config):
        self.config = config
        # Initialize your custom backend
    
    def create(self, record: Record) -> str:
        # Implement create
        pass
    
    def read(self, record_id: str) -> Optional[Record]:
        # Implement read
        pass
    
    def update(self, record_id: str, record: Record) -> bool:
        # Implement update
        pass
    
    def delete(self, record_id: str) -> bool:
        # Implement delete
        pass
    
    def search(self, query) -> List[Record]:
        # Implement search
        pass
    
    def count(self) -> int:
        # Implement count
        pass
    
    def clear(self) -> None:
        # Implement clear
        pass

# Register with the registry the factory reads.
#
# Registration is not a factory method: the factory reads a registry, it
# does not own one. `register_backend` populates that registry and probes
# the driver named by `requires_module` before registering, which is what
# keeps `is_backend_available()` an answer about this machine rather than
# about the name.
from dataknobs_data import register_backend
from dataknobs_data.backends import sync_backends

register_backend(
    sync_backends,
    "custom",
    lambda: CustomDatabase,
    metadata={
        "description": "A custom backend",
        "persistent": True,
        # Omit both when the backend needs no optional dependency; a
        # backend without `requires_module` always registers.
        "requires_install": "pip install my-driver",
        "requires_module": "my_driver",
    },
)

# Now you can create instances
factory = DatabaseFactory()
custom_db = factory.create(backend="custom", **config)
```

## Error Handling

A missing driver raises `ValueError`, not `ImportError` — the backend is
refused before construction, by the registry lookup rather than by the
backend's own import:

```python
try:
    # A backend whose optional driver is not installed here
    db = factory.create(backend="postgres", host="localhost")
except ValueError as e:
    print(e)
    # Backend 'postgres' is known but not available here.
    # Install with: pip install dataknobs-data[postgres]
```

The message already carries the install command, so there is nothing to
reconstruct from the exception type. A name the registry does not know at
all reads differently, and deliberately so:

```python
factory.create(backend="postgrez")
# ValueError: Unknown backend type: postgrez.
# Available backends: duckdb, elasticsearch, file, memory, postgres, s3, sqlite
```

Catching `ImportError` around `create()` catches nothing.

### A key the backend does not accept

A config key that matches no field on the chosen backend is a `ValueError`
too, rather than being discarded:

```python
factory.create(backend="postgres", hosst="db.internal", database="app")
# ValueError: PostgresDatabaseConfig does not accept 'hosst' (did you mean
# 'host'?). Accepted keys: auto_create_table, command_timeout,
# connection_string, database, ensure_database, host, max_pool_size,
# min_pool_size, password, port, schema, schema_name, ssl, table,
# table_name, user, vector_enabled, vector_metric.
```

This is the same event as an unrecognised backend name, one layer in, and
it reports the same way. It matters more than a misspelt backend name does:
every connection field has a working default, so a Postgres config built
entirely from misspelled keys used to succeed against `localhost` and log
nothing. The "synthesized default values" warning could not cover it — that
warning fires when *recognized* explicit keys mix with defaults, and an
unrecognized key enters neither bucket, so the config read as "nothing was
configured".

The accepted list includes input spellings the backend resolves away, so
`connection` is answered with `connection_string` rather than with a list
that appears not to contain it. The routing keys `backend`, `factory`,
`name` and `type` pass through untouched, so a config dict may still carry
the discriminator that selected it.

To supply a key only some backends have, ask first:

```python
from dataknobs_data.backends import sync_backends

backend_class = sync_backends.get_factory(backend_name)
config_cls = getattr(backend_class, "CONFIG_CLS", None)
if config_cls is not None and config_cls.accepts("table"):
    backend_config.setdefault("table", collection_name)
```

`get_factory` returns `None` — it does not raise — for a name this
installation cannot build, so reading `CONFIG_CLS` off it directly raises
`AttributeError` in exactly the case a reader is most likely to hit: a real
backend whose optional driver is not installed.

## Testing with Factory

```python
import pytest
from dataknobs_data import DatabaseFactory

@pytest.fixture
def database_factory():
    """Provide database factory for tests."""
    return DatabaseFactory()

@pytest.fixture
def test_database(database_factory, request):
    """Create test database based on marker."""
    if request.node.get_closest_marker("integration"):
        # Use real backend for integration tests
        return database_factory.create(
            backend="postgres",
            host="localhost",
            database="test_db"
        )
    else:
        # Use memory backend for unit tests
        return database_factory.create(backend="memory")

def test_create_record(test_database):
    """Test record creation."""
    record = Record({"name": "test"})
    record_id = test_database.create(record)
    assert record_id is not None
    
@pytest.mark.integration
def test_postgres_specific(test_database):
    """Test PostgreSQL-specific features."""
    # This will use real PostgreSQL
    pass
```

## Multi-Backend Applications

```python
class DataService:
    """Service that uses multiple backends."""
    
    def __init__(self):
        factory = DatabaseFactory()
        
        # Different backends for different purposes
        self.cache = factory.create(backend="memory")
        self.primary = factory.create(
            backend="postgres",
            host="db.example.com",
            database="production"
        )
        self.search = factory.create(
            backend="elasticsearch",
            host="search.example.com",
            port=9200,
            index="products"
        )
        self.archive = factory.create(
            backend="s3",
            bucket="archive-bucket"
        )
    
    def get_product(self, product_id: str):
        """Get product with caching."""
        # Check cache first
        cached = self.cache.read(product_id)
        if cached:
            return cached
        
        # Get from primary database
        product = self.primary.read(product_id)
        if product:
            # Store in cache
            self.cache.create(product)
        return product
    
    def search_products(self, query: str):
        """Search products using Elasticsearch."""
        return self.search.search(
            Query().filter("description", "LIKE", f"%{query}%")
        )
    
    def archive_old_products(self, days: int = 365):
        """Archive old products to S3."""
        cutoff = datetime.now() - timedelta(days=days)
        old_products = self.primary.search(
            Query().filter("updated_at", "<", cutoff.isoformat())
        )
        
        # Move to archive
        self.archive.batch_create(old_products)
        
        # Remove from primary
        for product in old_products:
            self.primary.delete(product.metadata["id"])
```

## Factory with Dependency Injection

```python
from dataclasses import dataclass
from typing import Protocol

class DatabaseProtocol(Protocol):
    """SyncDatabase interface for dependency injection."""
    def create(self, record: Record) -> str: ...
    def read(self, record_id: str) -> Optional[Record]: ...
    def update(self, record_id: str, record: Record) -> bool: ...
    def delete(self, record_id: str) -> bool: ...

@dataclass
class AppConfig:
    """Application configuration."""
    db_backend: str = "memory"
    db_config: dict = None

class Application:
    """Application with injected database."""
    
    def __init__(self, config: AppConfig):
        factory = DatabaseFactory()
        self.db: DatabaseProtocol = factory.create(
            backend=config.db_backend,
            **(config.db_config or {})
        )
    
    def process_data(self, data: dict):
        """Process data using injected database."""
        record = Record(data)
        return self.db.create(record)

# Different configurations for different environments
dev_config = AppConfig(db_backend="memory")
prod_config = AppConfig(
    db_backend="postgres",
    db_config={"host": "db.prod.example.com", "database": "app"}
)

# Create applications with different backends
dev_app = Application(dev_config)
prod_app = Application(prod_config)
```

## Best Practices

1. **Use configuration files** for backend settings
2. **Leverage environment variables** for sensitive data
3. **Create backend based on environment** (dev/staging/prod)
4. **Handle missing dependencies gracefully**
5. **Use dependency injection** for testability
6. **Document backend requirements** in your README
7. **Provide fallback options** when backends are unavailable
8. **Use factory registration** with config system
9. **Query backend capabilities** before using features
10. **Test with multiple backends** for compatibility