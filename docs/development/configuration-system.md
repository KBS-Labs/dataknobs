# DataKnobs Configuration System

## Overview

The DataKnobs configuration system provides a standardized way to configure and instantiate objects across all packages in the DataKnobs ecosystem. Built on the `dataknobs-config` package, it enables:

- **Consistent Configuration**: All DataKnobs packages follow the same configuration patterns
- **Dynamic Instantiation**: Create objects from configuration files without hardcoding dependencies
- **Environment Overrides**: Override configuration values with environment variables
- **Cross-references**: Reference other configuration values within your config
- **Factory Pattern Support**: Use factories to create complex objects

## Core Concepts

### 1. ConfigurableBase

All configurable classes in DataKnobs inherit from `ConfigurableBase`, which provides:

```python
from dataknobs_config import ConfigurableBase

class MyClass(ConfigurableBase):
    def __init__(self, config: dict):
        # Your initialization code
        pass
    
    @classmethod
    def from_config(cls, config: dict):
        """Create instance from configuration dictionary."""
        return cls(config)
```

### 2. Environment Variable Substitution

The configuration system supports environment variable substitution using `${VAR}` syntax:

```yaml
database:
  host: ${DB_HOST:localhost}      # Use DB_HOST or default to localhost
  port: ${DB_PORT:5432}           # Use DB_PORT or default to 5432
  password: ${DB_PASSWORD}        # Required - no default
  ssl: ${USE_SSL:true}            # Converts to boolean
  max_connections: ${MAX_CONN:100} # Converts to integer
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
    class: dataknobs_data.backends.postgres.PostgresDatabase
    host: localhost
    database: myapp
    
  - name: cache
    class: dataknobs_data.backends.memory.MemoryDatabase
    
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

For classes that accept configuration as a dictionary:

```python
from dataknobs_config import ConfigurableBase

class SimpleDatabase(ConfigurableBase):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 5432)
```

### Pattern 2: Complex Initialization

For classes requiring setup beyond simple attribute assignment:

```python
from dataknobs_config import ConfigurableBase

class ComplexService(ConfigurableBase):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._initialize()
    
    def _initialize(self):
        # Complex initialization logic
        self._setup_connections()
        self._load_resources()
    
    @classmethod
    def from_config(cls, config: dict):
        """Custom configuration handling."""
        # Pre-process config if needed
        processed_config = cls._validate_config(config)
        return cls(processed_config)
```

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
# All backends inherit from ConfigurableBase
from dataknobs_config import ConfigurableBase
from dataknobs_data.database import Database

class PostgresDatabase(Database, ConfigurableBase):
    def __init__(self, config: dict = None):
        super().__init__(config)
        # PostgreSQL-specific initialization
    
    @classmethod
    def from_config(cls, config: dict):
        return cls(config)
```

Usage:

```yaml
# config.yaml
databases:
  - name: main
    class: dataknobs_data.backends.postgres.PostgresDatabase
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
    class: dataknobs_utils.elasticsearch_utils.SimplifiedElasticsearchClient
    host: ${ES_HOST:localhost}
    port: ${ES_PORT:9200}
    timeout: 30
```

## Best Practices

### 1. Always Inherit from ConfigurableBase

When creating new classes that might be configured:

```python
from dataknobs_config import ConfigurableBase

class MyNewClass(ConfigurableBase):
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Your initialization
```

### 2. Support Both Direct and Config-based Construction

Allow flexibility in how objects are created:

```python
class FlexibleClass(ConfigurableBase):
    def __init__(self, host=None, port=None, config=None):
        if config:
            self.host = config.get("host", host or "localhost")
            self.port = config.get("port", port or 8080)
        else:
            self.host = host or "localhost"
            self.port = port or 8080
```

### 3. Document Configuration Options

Always document what configuration options your class accepts:

```python
class WellDocumentedClass(ConfigurableBase):
    """A well-documented configurable class.
    
    Configuration Options:
        host (str): Server hostname (default: localhost)
        port (int): Server port (default: 8080)
        timeout (int): Connection timeout in seconds (default: 30)
        retry_count (int): Number of retries (default: 3)
    """
    def __init__(self, config: dict = None):
        # Implementation
        pass
```

### 4. Validate Configuration

Validate configuration values early:

```python
from dataknobs_config import ConfigurableBase, ValidationError

class ValidatedClass(ConfigurableBase):
    @classmethod
    def from_config(cls, config: dict):
        # Validate required fields
        if "host" not in config:
            raise ValidationError("'host' is required in configuration")
        
        # Validate types
        if not isinstance(config.get("port"), int):
            raise ValidationError("'port' must be an integer")
        
        return cls(config)
```

### 5. Use Environment Variables for Secrets

Never hardcode secrets in configuration files:

```yaml
database:
  - name: production
    class: dataknobs_data.backends.postgres.PostgresDatabase
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

1. **Add ConfigurableBase inheritance**:
```python
# Before
class MyClass:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

# After
from dataknobs_config import ConfigurableBase

class MyClass(ConfigurableBase):
    def __init__(self, param1=None, param2=None, config=None):
        if config:
            self.param1 = config.get("param1", param1)
            self.param2 = config.get("param2", param2)
        else:
            self.param1 = param1
            self.param2 = param2
    
    @classmethod
    def from_config(cls, config: dict):
        return cls(config=config)
```

2. **Update tests** to verify both construction methods work
3. **Document** the configuration options in class docstring
4. **Add examples** showing configuration-based usage

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