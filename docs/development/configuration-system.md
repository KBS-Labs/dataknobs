# DataKnobs Configuration System

## Overview

The DataKnobs configuration system provides a standardized way to configure and instantiate objects across all packages in the DataKnobs ecosystem. Built on the `dataknobs-config` package, it enables:

- **Consistent Configuration**: All DataKnobs packages follow the same configuration patterns
- **Dynamic Instantiation**: Create objects from configuration files without hardcoding dependencies
- **Environment Overrides**: Override configuration values with environment variables
- **Cross-references**: Reference other configuration values within your config
- **Factory Pattern Support**: Use factories to create complex objects

## Core Concepts

### 1. Structured Configuration

A configurable class is a pair: a frozen `StructuredConfig` dataclass holding
the knobs, and a `StructuredConfigConsumer` reading them. The dataclass is the
schema, so the field set and the construction surface cannot drift apart.

```python
from dataclasses import dataclass
from typing import ClassVar, Literal

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)


@dataclass(frozen=True)
class MyClassConfig(StructuredConfig):
    host: str = "localhost"
    port: int = 5432

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


class MyClass(StructuredConfigConsumer[MyClassConfig]):
    CONFIG_CLS: ClassVar[type[MyClassConfig]] = MyClassConfig

    def _setup(self) -> None:
        self.dsn = f"postgresql://{self.config.host}:{self.config.port}"
```

`from_config` is provided by the mixin, so nothing above hand-writes it. The
backends shipped in `dataknobs_data` all use this pattern.

> `dataknobs_config.ConfigurableBase` is the deprecated predecessor. It still
> works and raises no runtime warning, but new code should use the pair above.
> It appears below wherever a comparison is useful; see
> [ConfigurableBase (deprecated)](../packages/config/configurable-base.md) and
> [Adding Configuration Support](./adding-config-support.md) for the migration.

### 2. Environment Variable Substitution

The configuration system supports environment variable substitution using `${VAR}` syntax:

```yaml
database:
  host: ${DB_HOST:localhost}      # Use DB_HOST or default to localhost
  port: ${DB_PORT:5432}           # Use DB_PORT or default to 5432
  password: ${DB_PASSWORD}        # Required - no default
  ssl: ${USE_SSL:true}            # Converts to boolean
  max_pool_size: ${MAX_POOL:100}  # Converts to integer
```

Features:
- `${VAR}` - Use environment variable, error if not found
- `${VAR:default}` - Use environment variable or default value
- Automatic type conversion for single variables (int, float, bool)
- Works recursively in nested structures

### 3. Configuration Structure

Configurations are organized by type, with each type containing a list of named configurations:

```yaml
databases:
  - name: primary
    class: dataknobs_data.backends.postgres.SyncPostgresDatabase
    host: localhost
    database: myapp
    
  - name: cache
    class: dataknobs_data.backends.memory.SyncMemoryDatabase
    
services:
  - name: processor
    class: myapp.services.DataProcessor
    database: ${databases.primary}  # Cross-reference
```

### 4. Object Building

The Config class provides methods to build objects from configurations:

```python
from dataknobs_config import Config

config = Config("config.yaml")

# Get configuration as dictionary
db_config = config.get("databases", "primary")

# Build object instance
db = config.get_instance("databases", "primary")
```

## Implementation Patterns

### Pattern 1: Simple Configuration

For a class whose configuration is a flat set of typed knobs:

```python
@dataclass(frozen=True)
class SimpleDatabaseConfig(StructuredConfig):
    host: str = "localhost"
    port: int = 5432

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


class SimpleDatabase(StructuredConfigConsumer[SimpleDatabaseConfig]):
    CONFIG_CLS: ClassVar[type[SimpleDatabaseConfig]] = SimpleDatabaseConfig
```

No `_setup` is needed when the class only reads `self.config.*`.

### Pattern 2: Complex Initialization

For a class requiring setup beyond holding its configuration. Derived state
goes in `_setup()`; anything awaitable goes in `_ainit()`, which runs only on
the `from_config_async` path:

```python
class ComplexService(StructuredConfigConsumer[SimpleDatabaseConfig]):
    CONFIG_CLS: ClassVar[type[SimpleDatabaseConfig]] = SimpleDatabaseConfig

    def _setup(self) -> None:
        self._resources = self._load_resources()
        self._connection = None

    async def _ainit(self) -> None:
        self._connection = await open_connection(self.config.host)
```

Reshaping the *input* — a legacy key, an alias, a value assembled from several
others — belongs on the config class in `_normalize_dict`, not here.

### Pattern 3: Factory Pattern

For creating different implementations based on configuration:

```python
from dataknobs_config import FactoryBase

class DatabaseFactory(FactoryBase):
    def create(self, **config):
        backend = config.pop("backend", "memory")
        
        if backend == "postgres":
            from .postgres import PostgresDatabase
            return PostgresDatabase(config)
        elif backend == "elasticsearch":
            from .elasticsearch import ElasticsearchDatabase
            return ElasticsearchDatabase(config)
        else:
            from .memory import MemoryDatabase
            return MemoryDatabase(config)
```

### Pattern 4: Factory Registration

Register factories with the Config class for cleaner configuration files:

```python
from dataknobs_config import Config
from myapp.factories import database_factory, cache_factory

# Create config and register factories
config = Config()
config.register_factory("database", database_factory)
config.register_factory("cache", cache_factory)

# Now use registered names in configuration
config.load({
    "services": [{
        "name": "main_db",
        "factory": "database",  # Uses registered factory
        "backend": "postgres",
        "host": "localhost"
    }]
})

# Get instance
db = config.get_instance("services", "main_db")

# Check registered factories
factories = config.get_registered_factories()
print(f"Registered: {list(factories.keys())}")  # ['database', 'cache']

# Unregister if needed
config.unregister_factory("cache")
```

Benefits of factory registration:
- Cleaner configuration files (no module paths)
- Runtime factory substitution (useful for testing)
- Pre-configured factory instances
- Better separation of concerns

## Package Integration Examples

### DataKnobs Data Package

The data package demonstrates comprehensive config integration:

```python
# The backends shipped here take their config through
# StructuredConfigConsumer, which validates the mapping against the
# backend's CONFIG_CLS -- the same pattern Core Concepts describes for
# the classes you write.
from dataknobs_data.backends.postgres import SyncPostgresDatabase

# CONFIG_CLS names the dataclass the mapping is checked against --
# an unknown key is rejected rather than silently ignored.
SyncPostgresDatabase.CONFIG_CLS  # -> PostgresDatabaseConfig

db = SyncPostgresDatabase.from_config(
    {"host": "localhost", "database": "myapp", "table": "records"}
)
```

Usage:

```yaml
# config.yaml
databases:
  - name: main
    class: dataknobs_data.backends.postgres.SyncPostgresDatabase
    host: ${DB_HOST:localhost}  # Environment variable with default
    database: myapp
    user: ${DB_USER:postgres}
    password: ${DB_PASSWORD}
```

```python
from dataknobs_config import Config
from dataknobs_data import Query, Record

# Load configuration
config = Config("config.yaml")

# Create database instance
db = config.get_instance("databases", "main")

# Use the database
record = Record({"name": "test", "value": 42})
record_id = db.create(record)
```

### DataKnobs Utils Package

Utility classes can also be configured:

```yaml
elasticsearch:
  - name: search_cluster
    class: dataknobs_utils.elasticsearch_utils.SimplifiedElasticsearchIndex
    host: ${ES_HOST:localhost}
    port: ${ES_PORT:9200}
    timeout: 30
```

## Best Practices

### 1. Give the Configuration a Schema

When creating new classes that might be configured, write the config dataclass
first. It is what makes every other practice below mechanical rather than
conventional:

```python
@dataclass(frozen=True)
class MyNewClassConfig(StructuredConfig):
    host: str = "localhost"
    port: int = 8080

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


class MyNewClass(StructuredConfigConsumer[MyNewClassConfig]):
    CONFIG_CLS: ClassVar[type[MyNewClassConfig]] = MyNewClassConfig
```

### 2. Support Both Direct and Config-based Construction

This one is free. The mixin's single `__init__` accepts a typed config, a
mapping, or loose keyword arguments, and all of them reach the same state:

```python
MyNewClass()                                    # all defaults
MyNewClass(MyNewClassConfig(host="db"))         # typed
MyNewClass({"host": "db", "port": 5432})        # a loaded mapping
MyNewClass(host="db", port=5432)                # loose kwargs
MyNewClass.from_config({"host": "db"})          # the registry path
```

### 3. Document Configuration Options

Document the options on the dataclass, which is the thing that defines them —
a docstring listing keys the constructor does not accept is the drift this
pattern exists to remove:

```python
@dataclass(frozen=True)
class WellDocumentedConfig(StructuredConfig):
    """Configuration for WellDocumentedClass.

    Attributes:
        host: Server hostname.
        port: Server port.
        timeout: Connection timeout in seconds.
        retry_count: Number of retries before giving up.
    """

    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    retry_count: int = 3
```

### 4. Validate Configuration

Per-class invariants go in `__post_init__`, so they run for every construction
shape rather than only the one that arrives as a dict. Presence and type are
already enforced by the field set:

```python
@dataclass(frozen=True)
class ValidatedConfig(StructuredConfig):
    host: str
    port: int = 8080

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("'port' must be between 1 and 65535")
```

`host` has no default, so omitting it is a `TypeError` from the dataclass
itself — the "required field" check no longer needs writing.

### 5. Use Environment Variables for Secrets

Never hardcode secrets in configuration files:

```yaml
database:
  - name: production
    class: dataknobs_data.backends.postgres.SyncPostgresDatabase
    host: ${DB_HOST}
    user: ${DB_USER}
    password: ${DB_PASSWORD}  # From environment variable
```

## Testing Configuration

When testing configurable classes:

```python
import pytest
from dataknobs_config import Config

def test_config_based_creation():
    """Test that class can be created from config."""
    config = Config()
    config.load({
        "test_objects": [{
            "name": "test",
            "class": "mypackage.MyClass",
            "param1": "value1",
            "param2": 42
        }]
    })
    
    obj = config.get_instance("test_objects", "test")
    assert obj is not None
    assert obj.param1 == "value1"
    assert obj.param2 == 42

def test_from_config_method():
    """Test from_config classmethod."""
    from mypackage import MyClass
    
    obj = MyClass.from_config({
        "param1": "value1",
        "param2": 42
    })
    assert obj is not None
```

## Migration Guide

### Migrating Existing Classes

To add configuration support to existing classes:

1. **Lift the constructor parameters into a config dataclass**:
```python
# Before
class MyClass:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

# After
@dataclass(frozen=True)
class MyClassConfig(StructuredConfig):
    param1: str = ""
    param2: int = 0

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


class MyClass(StructuredConfigConsumer[MyClassConfig]):
    CONFIG_CLS: ClassVar[type[MyClassConfig]] = MyClassConfig
```

Callers passing keyword arguments keep working — `MyClass(param1="x")` is one
of the shapes the mixin's `__init__` accepts — and `from_config` arrives
without being written.

2. **Move the constructor body** into `_setup()`, reading `self.config.*`
3. **Pin the adoption** with `assert_structured_config_consumer(MyClass)`, which
   fails if the dataclass and the constructor disagree
4. **Document** the configuration options on the dataclass
5. **Add examples** showing configuration-based usage

A class already inheriting `ConfigurableBase` migrates the same way; see
[Adding Configuration Support](./adding-config-support.md#migrating-from-configurablebase)
for the step-by-step.

## Advanced Features

### Cross-references

Reference other configuration values:

```yaml
defaults:
  - name: timeouts
    connection: 30
    request: 60

services:
  - name: api_client
    class: myapp.APIClient
    connection_timeout: ${defaults.timeouts.connection}
    request_timeout: ${defaults.timeouts.request}
```

### Environment Variables with Defaults

Use environment variables with fallback values:

```yaml
database:
  host: ${DB_HOST:localhost}  # Use DB_HOST or default to localhost
  port: ${DB_PORT:5432}       # Use DB_PORT or default to 5432
```

### Factory Registration

Register factories for dynamic object creation:

```python
from dataknobs_config import Config

config = Config()
config.register_factory("database", DatabaseFactory())

# Now can use factory in config
config.load({
    "databases": [{
        "name": "main",
        "factory": "database",
        "backend": "postgres",
        "host": "localhost"
    }]
})
```

## Troubleshooting

### Common Issues

1. **ImportError when building objects**
   - Ensure the module path in `class` attribute is correct
   - Check that required packages are installed

2. **TypeError: __init__() got an unexpected keyword argument**
   - Implement `from_config()` classmethod to handle config dict
   - Or accept `config` parameter in `__init__()`

3. **Configuration not found**
   - Check configuration structure (type -> list of configs)
   - Verify the name matches exactly

4. **Environment variables not resolved**
   - Ensure environment variables are set before loading config
   - Check syntax: `${VAR_NAME}` or `${VAR_NAME:default}`

## Further Reading

- [dataknobs-config Package Documentation](../packages/config/index.md)
- [Data Package Configuration Examples](../packages/data/configuration.md)
- [Environment Variables Guide](./environment-variables.md)
- [Testing Configured Objects](./testing.md#testing-configuration)