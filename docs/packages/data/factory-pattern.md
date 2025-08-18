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

Query available backends and their requirements:

```python
factory = DatabaseFactory()

# Get all available backends
backends = factory.get_available_backends()
print(f"Available backends: {backends}")
# Output: ['memory', 'file', 'postgres', 'elasticsearch', 's3']

# Get information about a specific backend
info = factory.get_backend_info("s3")
print(info)
# Output: {
#     'description': 'AWS S3 object storage backend',
#     'persistent': True,
#     'requires_install': 'pip install dataknobs-data[s3]',
#     'required_params': ['bucket'],
#     'optional_params': ['prefix', 'region', 'endpoint_url', ...]
# }

# Check if backend is available
if factory.is_backend_available("postgres"):
    db = factory.create(backend="postgres", **config)
else:
    print("PostgreSQL backend not available")
    print("Install with: pip install dataknobs-data[postgres]")
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
        hosts=[os.environ["ES_HOST"]],
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
            hosts=["localhost:9200"],
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
from dataknobs_data import Database, DatabaseFactory
from dataknobs_data.records import Record
from typing import List, Optional

class CustomDatabase(Database):
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

# Register with factory
factory = DatabaseFactory()
factory.register_backend("custom", CustomDatabase)

# Now you can create instances
custom_db = factory.create(backend="custom", **config)
```

## Error Handling

The factory provides helpful error messages:

```python
try:
    # Try to create backend with missing dependency
    db = factory.create(backend="postgres", host="localhost")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install dataknobs-data[postgres]")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Failed to create backend: {e}")
```

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
            hosts=["search.example.com:9200"],
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
    """Database interface for dependency injection."""
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