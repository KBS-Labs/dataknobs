# Factory Registration Pattern

The DataKnobs Config package provides a powerful factory pattern system for dynamic object construction. This enables flexible, configurable object creation with support for dependency injection, lazy initialization, and custom construction logic.

## Overview

The factory pattern in DataKnobs Config allows you to:

- Register reusable factories for object construction
- Define factories in configuration files
- Use different factory types (classes, functions, callables)
- Cache constructed objects for performance
- Handle complex initialization logic
- Support dependency injection

## Factory Types

### 1. Class-Based Factory

Inherit from `FactoryBase` for structured factories:

```python
from dataknobs_config import FactoryBase

class DatabaseFactory(FactoryBase):
    """Factory for creating database connections."""
    
    def __init__(self, default_pool_size=20):
        self.default_pool_size = default_pool_size
    
    def create(self, **config):
        """Create a database connection."""
        # Apply defaults
        config.setdefault("pool_size", self.default_pool_size)
        
        # Validate configuration
        if "host" not in config:
            raise ValueError("Database host is required")
        
        # Create and return instance
        return Database(**config)
```

### 2. Callable Factory

Use any callable object with `__call__` method:

```python
class CacheFactory:
    """Callable factory for cache instances."""
    
    def __init__(self, default_ttl=3600):
        self.default_ttl = default_ttl
    
    def __call__(self, **config):
        config.setdefault("ttl", self.default_ttl)
        
        backend = config.get("backend", "memory")
        if backend == "redis":
            return RedisCache(**config)
        elif backend == "memory":
            return MemoryCache(**config)
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
```

### 3. Function Factory

Simple function-based factories:

```python
def create_service(**config):
    """Factory function for services."""
    service_type = config.pop("type", "http")
    
    if service_type == "http":
        return HttpService(**config)
    elif service_type == "grpc":
        return GrpcService(**config)
    else:
        raise ValueError(f"Unknown service type: {service_type}")
```

### 4. Lambda Factory

Quick inline factories:

```python
# Simple lambda factory
redis_factory = lambda **config: RedisCache(
    host=config.get("host", "localhost"),
    port=config.get("port", 6379)
)

# Register lambda factory
config.register_factory("redis", redis_factory)
```

## Registration

### Register Factories

```python
from dataknobs_config import Config

config = Config()

# Register class-based factory
config.register_factory("database", DatabaseFactory())

# Register callable factory
config.register_factory("cache", CacheFactory())

# Register function factory
config.register_factory("service", create_service)

# Register with module path
config.register_factory("queue", "myapp.factories.QueueFactory")
```

### Unregister Factories

```python
# Remove a registered factory
config.unregister_factory("database")

# Check if factory is registered
if config.has_factory("cache"):
    config.unregister_factory("cache")
```

### List Registered Factories

```python
# Get all registered factory names
factories = config.get_registered_factories()
print(f"Registered factories: {factories}")
# Output: ['database', 'cache', 'service']

# Get factory instance
factory = config.get_factory("database")
```

## Configuration Usage

### Using Registered Factories

Reference registered factories by name in configuration:

```yaml
# config.yaml
databases:
  - name: primary
    factory: "database"  # Use registered factory
    host: localhost
    port: 5432
    
  - name: analytics
    factory: "database"  # Reuse same factory
    host: analytics.example.com
    port: 5432

caches:
  - name: main
    factory: "cache"
    backend: redis
    host: localhost
```

### Using Module Path Factories

Reference factories by module path:

```yaml
databases:
  - name: primary
    factory: "myapp.factories.DatabaseFactory"
    host: localhost
    
services:
  - name: api
    factory: "myapp.services.ApiServiceFactory"
    port: 8000
```

### Using Class References

Direct class instantiation without factory:

```yaml
caches:
  - name: simple
    class: "myapp.cache.SimpleCache"
    size: 1000
```

## Object Construction

### Manual Construction

```python
# Construct object using configuration
db = config.construct("databases", "primary")

# Construct all objects of a type
all_dbs = config.construct_all("databases")

# Construct with overrides
db = config.construct("databases", "primary", 
                      overrides={"pool_size": 50})
```

### Automatic Construction

```python
# Get configuration and construct if factory/class specified
db_config = config.get("databases", "primary", construct=True)
```

### Lazy Construction

```python
# Register lazy factory
class LazyDatabaseFactory(FactoryBase):
    def create(self, **config):
        # Return a lazy wrapper
        return LazyDatabase(config)

class LazyDatabase:
    def __init__(self, config):
        self.config = config
        self._connection = None
    
    @property
    def connection(self):
        if self._connection is None:
            self._connection = create_connection(**self.config)
        return self._connection
```

## Advanced Patterns

### Dependency Injection

```python
class ServiceFactory(FactoryBase):
    """Factory with dependency injection."""
    
    def __init__(self, config):
        self.config = config
    
    def create(self, **service_config):
        # Resolve dependencies
        db = self.config.construct("databases", 
                                   service_config.pop("database"))
        cache = self.config.construct("caches", 
                                       service_config.pop("cache"))
        
        # Inject dependencies
        return Service(database=db, cache=cache, **service_config)

# Register factory with config injection
config.register_factory("service", ServiceFactory(config))
```

Configuration:

```yaml
services:
  - name: api
    factory: "service"
    database: "xref:databases[primary]"  # Reference to database
    cache: "xref:caches[main]"           # Reference to cache
    port: 8000
```

### Factory Registry Pattern

```python
class FactoryRegistry:
    """Central registry for all factories."""
    
    def __init__(self):
        self._factories = {}
    
    def register(self, type_name, factory):
        """Register a factory for a type."""
        self._factories[type_name] = factory
    
    def create(self, type_name, **config):
        """Create object using registered factory."""
        if type_name not in self._factories:
            raise ValueError(f"No factory for type: {type_name}")
        return self._factories[type_name](**config)

# Use with config
registry = FactoryRegistry()
registry.register("postgres", PostgresFactory())
registry.register("mysql", MySqlFactory())

config.register_factory("database", registry.create)
```

### Abstract Factory Pattern

```python
from abc import ABC, abstractmethod

class AbstractDatabaseFactory(ABC):
    """Abstract factory for databases."""
    
    @abstractmethod
    def create_connection(self, **config):
        pass
    
    @abstractmethod
    def create_pool(self, **config):
        pass

class PostgresFactory(AbstractDatabaseFactory):
    def create_connection(self, **config):
        return PostgresConnection(**config)
    
    def create_pool(self, **config):
        return PostgresPool(**config)

class MySQLFactory(AbstractDatabaseFactory):
    def create_connection(self, **config):
        return MySQLConnection(**config)
    
    def create_pool(self, **config):
        return MySQLPool(**config)
```

### Builder Pattern Integration

```python
class DatabaseBuilder:
    """Builder for complex database configurations."""
    
    def __init__(self):
        self.config = {}
    
    def with_host(self, host):
        self.config["host"] = host
        return self
    
    def with_credentials(self, username, password):
        self.config["username"] = username
        self.config["password"] = password
        return self
    
    def with_pool(self, min_size=5, max_size=20):
        self.config["pool"] = {
            "min_size": min_size,
            "max_size": max_size
        }
        return self
    
    def build(self):
        return Database(**self.config)

class DatabaseFactory(FactoryBase):
    def create(self, **config):
        builder = DatabaseBuilder()
        
        # Use builder pattern
        if "host" in config:
            builder.with_host(config["host"])
        
        if "username" in config:
            builder.with_credentials(
                config["username"], 
                config.get("password")
            )
        
        if "pool" in config:
            builder.with_pool(**config["pool"])
        
        return builder.build()
```

## Caching

### Object Caching

```python
# Enable caching for factories
class CachedDatabaseFactory(FactoryBase):
    def __init__(self):
        self._cache = {}
    
    def create(self, **config):
        # Create cache key
        cache_key = (
            config.get("host"),
            config.get("port"),
            config.get("database")
        )
        
        # Return cached instance if exists
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create and cache new instance
        instance = Database(**config)
        self._cache[cache_key] = instance
        return instance
```

### Config-Level Caching

```python
# Objects are automatically cached by reference
db1 = config.construct("databases", "primary")
db2 = config.construct("databases", "primary")
assert db1 is db2  # Same instance

# Clear cache
config.clear_cache()

# Clear specific type cache
config.clear_cache("databases")

# Disable caching
db = config.construct("databases", "primary", use_cache=False)
```

## Validation

### Factory Validation

```python
class ValidatingFactory(FactoryBase):
    """Factory with built-in validation."""
    
    def validate_config(self, config):
        """Validate configuration before construction."""
        required = ["host", "port", "username"]
        missing = [k for k in required if k not in config]
        
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        if config.get("port", 0) < 1024:
            raise ValueError("Port must be >= 1024")
    
    def create(self, **config):
        self.validate_config(config)
        return Database(**config)
```

### Schema Validation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration schema."""
    host: str
    port: int = 5432
    username: str = "postgres"
    password: Optional[str] = None
    pool_size: int = 20

class TypedDatabaseFactory(FactoryBase):
    def create(self, **config):
        # Validate against schema
        db_config = DatabaseConfig(**config)
        return Database(
            host=db_config.host,
            port=db_config.port,
            username=db_config.username,
            password=db_config.password,
            pool_size=db_config.pool_size
        )
```

## Testing

### Mock Factories

```python
class MockDatabaseFactory(FactoryBase):
    """Mock factory for testing."""
    
    def create(self, **config):
        return MockDatabase(**config)

# Use in tests
def test_service():
    config = Config()
    config.register_factory("database", MockDatabaseFactory())
    
    service = config.construct("services", "api")
    assert isinstance(service.database, MockDatabase)
```

### Factory Testing

```python
import pytest

def test_database_factory():
    factory = DatabaseFactory()
    
    # Test valid configuration
    db = factory.create(
        host="localhost",
        port=5432,
        username="test"
    )
    assert db.host == "localhost"
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        factory.create()  # Missing required fields
```

## Best Practices

### 1. Single Responsibility

Each factory should handle one type of object:

```python
# Good: Specific factory
class PostgresConnectionFactory(FactoryBase):
    def create(self, **config):
        return PostgresConnection(**config)

# Bad: Generic factory doing too much
class DatabaseFactory(FactoryBase):
    def create(self, **config):
        if config["type"] == "postgres":
            return PostgresConnection(**config)
        elif config["type"] == "mysql":
            return MySQLConnection(**config)
        # ... many more conditions
```

### 2. Configuration Validation

Always validate configuration in factories:

```python
class SafeDatabaseFactory(FactoryBase):
    def create(self, **config):
        # Validate required fields
        self._validate_required(config)
        
        # Validate types
        self._validate_types(config)
        
        # Apply defaults
        config = self._apply_defaults(config)
        
        return Database(**config)
```

### 3. Immutable Factories

Keep factories stateless and immutable:

```python
# Good: Stateless factory
class StatelessFactory(FactoryBase):
    def create(self, **config):
        return Service(**config)

# Bad: Stateful factory
class StatefulFactory(FactoryBase):
    def __init__(self):
        self.counter = 0  # Mutable state
    
    def create(self, **config):
        self.counter += 1  # Modifying state
        config["id"] = self.counter
        return Service(**config)
```

### 4. Documentation

Document factory behavior and configuration:

```python
class DocumentedFactory(FactoryBase):
    """Factory for creating database connections.
    
    Configuration:
        host (str): Database host (required)
        port (int): Database port (default: 5432)
        username (str): Username (required)
        password (str): Password (optional)
        pool_size (int): Connection pool size (default: 20)
    
    Example:
        factory = DocumentedFactory()
        db = factory.create(
            host="localhost",
            port=5432,
            username="user"
        )
    """
    
    def create(self, **config):
        # Implementation
        pass
```

## Examples

### Complete Example

```python
# factories.py
from dataknobs_config import FactoryBase, Config
import asyncpg
import redis

class AsyncPostgresFactory(FactoryBase):
    """Factory for async PostgreSQL connections."""
    
    async def create(self, **config):
        return await asyncpg.create_pool(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            user=config.get("username", "postgres"),
            password=config.get("password"),
            database=config.get("database", "postgres"),
            min_size=config.get("min_pool_size", 5),
            max_size=config.get("max_pool_size", 20)
        )

class RedisCacheFactory(FactoryBase):
    """Factory for Redis cache connections."""
    
    def create(self, **config):
        return redis.Redis(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            decode_responses=config.get("decode_responses", True)
        )

# main.py
async def setup_application():
    # Load configuration
    config = Config.from_file("config.yaml")
    
    # Register factories
    config.register_factory("postgres", AsyncPostgresFactory())
    config.register_factory("redis", RedisCacheFactory())
    
    # Construct objects
    db_pool = await config.construct("databases", "primary")
    cache = config.construct("caches", "main")
    
    return db_pool, cache
```

Configuration:

```yaml
# config.yaml
databases:
  - name: primary
    factory: "postgres"
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    username: ${DB_USER:postgres}
    password: ${DB_PASSWORD}
    database: myapp
    min_pool_size: 10
    max_pool_size: 50

caches:
  - name: main
    factory: "redis"
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}
    db: 0
    decode_responses: true
```