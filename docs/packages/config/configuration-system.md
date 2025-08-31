# Configuration System Overview

The DataKnobs Config package provides a powerful, flexible configuration management system designed for complex applications with multiple environments and dynamic configuration needs.

## Core Concepts

### Configuration Structure

The configuration system organizes settings as dictionaries of lists by type. Each configuration type (e.g., "databases", "caches", "services") contains a list of configuration objects:

```yaml
databases:
  - name: primary
    host: localhost
    port: 5432
    
caches:
  - name: redis
    host: localhost
    port: 6379
```

### Key Features

- **Multi-format Support**: Load from YAML, JSON, or Python dictionaries
- **Environment Overrides**: Override any configuration value via environment variables
- **Factory Pattern**: Register and use factories for object construction
- **Cross-references**: Reference configurations across types using xref syntax
- **Variable Substitution**: Use environment variables within configuration files
- **Validation**: Built-in validation and normalization
- **Caching**: Intelligent caching of constructed objects

## Basic Usage

### Loading Configuration

```python
from dataknobs_config import Config

# Load from file
config = Config.from_file("config.yaml")

# Load from dictionary
config = Config({
    "databases": [
        {"name": "primary", "host": "localhost", "port": 5432}
    ]
})

# Load with environment overrides
config = Config.from_file("config.yaml", apply_env_overrides=True)
```

### Accessing Configuration

```python
# Get all databases
databases = config.get("databases")

# Get specific database by name
primary_db = config.get("databases", "primary")

# Get by index
first_db = config.get("databases", 0)

# Get with default
cache = config.get("caches", "redis", default={"host": "localhost"})
```

### Setting Configuration

```python
# Set entire type
config.set("caches", [{"name": "redis", "host": "localhost"}])

# Update specific item
config.set("databases", "primary", {"host": "prod.example.com"})

# Add new item
config.add("databases", {"name": "analytics", "host": "analytics.example.com"})
```

## Configuration Files

### YAML Format

```yaml
# config.yaml
databases:
  - name: primary
    host: ${DB_HOST:localhost}  # Variable substitution with default
    port: 5432
    pool_size: 20
    
  - name: analytics
    host: analytics.example.com
    port: 5432
    pool_size: 10

caches:
  - name: redis
    factory: "myapp.cache.RedisFactory"
    host: localhost
    port: 6379
    ttl: 3600

services:
  - name: api
    database: "xref:databases[primary]"  # Cross-reference
    cache: "xref:caches[redis]"
    port: 8000
```

### JSON Format

```json
{
  "databases": [
    {
      "name": "primary",
      "host": "localhost",
      "port": 5432
    }
  ],
  "caches": [
    {
      "name": "redis",
      "host": "localhost",
      "port": 6379
    }
  ]
}
```

## Cross-References

The xref syntax allows referencing configurations across types:

```yaml
services:
  - name: api
    # Reference by name
    database: "xref:databases[primary]"
    
    # Reference by index
    cache: "xref:caches[0]"
    
    # Reference first item (default)
    queue: "xref:queues"
```

References are resolved recursively and support circular reference detection.

## Variable Substitution

Configuration files can use environment variables:

```yaml
database:
  host: ${DB_HOST}               # Required variable
  port: ${DB_PORT:5432}          # With default value
  password: ${DB_PASS:-secret}   # Bash-style default
```

## Object Construction

The configuration system can automatically construct objects using registered factories or class references:

```python
# Register a factory
config.register_factory("database", DatabaseFactory())

# In configuration
databases:
  - name: primary
    factory: "database"  # Uses registered factory
    host: localhost
    
# Or use class directly
caches:
  - name: redis
    class: "myapp.cache.RedisCache"
    host: localhost

# Get constructed objects
db = config.construct("databases", "primary")
```

## Settings and Defaults

Configure global settings and type-specific defaults:

```python
from dataknobs_config import Settings

settings = Settings()

# Set type-specific defaults
settings.set_defaults("databases", {
    "port": 5432,
    "pool_size": 20
})

# Set global defaults
settings.set_global_defaults({
    "timeout": 30
})

# Apply to config
config = Config.from_file("config.yaml", settings=settings)
```

## Merging Configurations

Combine multiple configuration sources:

```python
# Load base configuration
base_config = Config.from_file("base.yaml")

# Load environment-specific overrides
env_config = Config.from_file("production.yaml")

# Merge configurations
base_config.merge(env_config)

# Or during construction
config = Config.from_file("base.yaml")
config.merge_file("production.yaml")
```

## Validation

The configuration system provides built-in validation:

```python
# Validate configuration
try:
    config.validate()
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")

# Custom validation
def validate_database(db_config):
    if db_config.get("port", 0) < 1024:
        raise ValueError("Port must be >= 1024")

config.add_validator("databases", validate_database)
```

## Best Practices

1. **Use Type Organization**: Group related configurations by type (databases, caches, services)
2. **Leverage Cross-References**: Avoid duplication by referencing shared configurations
3. **Environment-Specific Files**: Use separate files for different environments
4. **Variable Substitution**: Keep sensitive data in environment variables
5. **Factory Registration**: Register reusable factories for common object types
6. **Validation**: Add custom validators for critical configurations
7. **Defaults**: Use settings to define sensible defaults

## Advanced Features

### Dynamic Configuration

```python
# Reload configuration
config.reload()

# Watch for changes
config.watch(callback=on_config_change)

# Clear caches
config.clear_cache()
```

### Custom Factories

```python
from dataknobs_config import FactoryBase

class DatabaseFactory(FactoryBase):
    def create(self, **config):
        # Custom initialization logic
        config.setdefault("pool_size", 20)
        return Database(**config)

config.register_factory("database", DatabaseFactory())
```

### Configuration Export

```python
# Export to dictionary
config_dict = config.to_dict()

# Export to YAML
yaml_str = config.to_yaml()

# Export to JSON
json_str = config.to_json()
```

## Examples

For practical examples, see:
- [Database Configuration Example](../../examples/database-config.md)
- [Service Configuration Example](../../examples/service-config.md)
- [Multi-Environment Setup](../../examples/multi-environment.md)