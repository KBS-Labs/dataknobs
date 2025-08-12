# DataKnobs Config

A modular, reusable configuration system for composable settings with environment variable overrides, file loading, and optional object construction helpers.

## Features

- **Modular Configuration**: Organize configurations by type with atomic configuration units
- **Multiple Input Formats**: Load from YAML, JSON files, or Python dictionaries
- **Composable**: Reference other configurations and compose complex setups
- **Environment Overrides**: Override any configuration value via environment variables
- **Path Resolution**: Automatically resolve relative paths to absolute
- **Object Construction**: Optional helpers to build objects from configurations
- **Defaults Management**: Global and type-specific default values
- **Caching**: Cache constructed objects for efficiency

## Installation

```bash
pip install dataknobs-config
```

## Quick Start

```python
from dataknobs_config import Config

# Load from dictionary
config = Config({
    "database": [
        {"name": "primary", "host": "localhost", "port": 5432},
        {"name": "secondary", "host": "backup.local", "port": 5433}
    ],
    "cache": [
        {"name": "redis", "host": "localhost", "port": 6379}
    ]
})

# Access configurations
primary_db = config.get("database", "primary")
print(primary_db["host"])  # localhost

# Load from file
config = Config.from_file("config.yaml")

# Load from multiple sources
config = Config("base.yaml", "overrides.json", {"extra": [...]})
```

## Core Concepts

### Atomic Configurations

Each configuration is an "atomic" unit - a dictionary of settings for a single object:

```python
{
    "name": "primary",      # Optional, auto-generated if not provided
    "type": "database",     # Optional, inferred from parent key
    "host": "localhost",
    "port": 5432,
    # ... any other attributes
}
```

### Configuration Structure

Internally, configurations are organized by type:

```python
{
    "database": [           # Type name
        {...},              # Atomic config 1
        {...}               # Atomic config 2
    ],
    "cache": [
        {...}               # Atomic config
    ],
    "settings": {           # Special type for global settings
        "config_root": "/app/config",
        "default_timeout": 30
    }
}
```

## String References (xref)

Reference other configurations using the xref format:

```python
config = Config({
    "database": [
        {"name": "primary", "host": "db.example.com"}
    ],
    "api": [
        {
            "name": "main",
            "database": "xref:database[primary]"  # Reference
        }
    ]
})

# Resolve references
api = config.resolve_reference("xref:api[main]")
print(api["database"]["host"])  # db.example.com
```

### Reference Formats

- `xref:type[name]` - Reference by name
- `xref:type[0]` - Reference by index
- `xref:type[-1]` - Reference last item
- `xref:type` - Reference first/only item

## Environment Variable Overrides

Override any configuration value using environment variables:

```bash
export DATAKNOBS_DATABASE__PRIMARY__HOST=prod.example.com
export DATAKNOBS_DATABASE__PRIMARY__PORT=5433
export DATAKNOBS_CACHE__REDIS__TTL=7200
```

```python
config = Config({
    "database": [{"name": "primary", "host": "localhost", "port": 5432}],
    "cache": [{"name": "redis", "ttl": 3600}]
})

# Environment variables automatically override values
db = config.get("database", "primary")
print(db["host"])  # prod.example.com
print(db["port"])  # 5433 (converted to int)
```

### Environment Variable Format

- Pattern: `DATAKNOBS_<TYPE>__<NAME_OR_INDEX>__<ATTRIBUTE>`
- Nested attributes: `DATAKNOBS_DATABASE__0__CONNECTION__TIMEOUT`
- Automatic type conversion for integers, floats, and booleans

## File References

Reference external configuration files using the `@` prefix:

```yaml
# main.yaml
database:
  - "@database/primary.yaml"    # Load from file
  - "@database/secondary.yaml"

settings:
  config_root: /app/config       # Base path for relative references
```

## Global Settings and Defaults

Configure global settings and defaults in the special `settings` section:

```python
config = Config({
    "database": [{"name": "db1"}],
    "settings": {
        # Paths
        "config_root": "/app/config",           # Base path for "@"-prefixed config file references
        "global_root": "/app",                   # Base for path resolution (settings.path_resolution_attributes)
        "database.global_root": "/app/db",       # Type-specific base for path resolution
        
        # Path resolution (supports exact names and regex patterns)
        "path_resolution_attributes": [
            "config_path",                       # Exact match for all types
            "database.data_dir",                 # Exact match for database type only
            "/.*_path$/",                        # Regex: all attributes ending with "_path"
            "cache./.*_dir$/"                    # Regex: cache type attributes ending with "_dir"
        ],
        
        # Defaults
        "default_timeout": 30,                   # Global default
        "database.default_pool_size": 10        # Type-specific default
    }
})
```

## Path Resolution

Automatically resolve relative paths to absolute:

```python
config = Config({
    "database": [{
        "name": "db1",
        "data_dir": "./data",              # Relative path
        "backup_dir": "/abs/path"          # Absolute path unchanged
    }],
    "settings": {
        "global_root": "/app",              # Base for path resolution
        "path_resolution_attributes": ["data_dir", "backup_dir"]
    }
})

db = config.get("database", "db1")
print(db["data_dir"])     # /app/data (resolved)
print(db["backup_dir"])   # /abs/path (unchanged)
```

## Object Construction (Optional)

Build objects directly from configurations:

```python
# Using class attribute
config = Config({
    "database": [{
        "name": "primary",
        "class": "myapp.database.PostgreSQL",
        "host": "localhost",
        "port": 5432
    }]
})

# Build object
db = config.build_object("xref:database[primary]")
# Returns instance of myapp.database.PostgreSQL

# Using factory pattern
config = Config({
    "cache": [{
        "name": "redis",
        "factory": "myapp.cache.CacheFactory",
        "type": "redis",
        "host": "localhost"
    }]
})

cache = config.build_object("xref:cache[redis]")
```

### Implementing Configurable Classes

```python
from dataknobs_config import ConfigurableBase

class MyDatabase(ConfigurableBase):
    def __init__(self, host, port, **kwargs):
        self.host = host
        self.port = port
        
    @classmethod
    def from_config(cls, config):
        # Custom configuration logic
        return cls(**config)
```

### Implementing Factories

```python
from dataknobs_config import FactoryBase

class DatabaseFactory(FactoryBase):
    def create(self, **config):
        db_type = config.pop("type", "postgresql")
        if db_type == "postgresql":
            return PostgreSQL(**config)
        elif db_type == "mysql":
            return MySQL(**config)
```

### Lazy Factory Access

```python
# Configuration with factory
config = Config({
    "database": [{
        "name": "primary",
        "factory": "myapp.db.DatabaseFactory",
        "type": "postgresql",
        "host": "localhost"
    }]
})

# Get the factory instance (cached)
factory = config.get_factory("database", "primary")
db1 = factory.create(database="app1")
db2 = factory.create(database="app2")

# Or get an instance directly
db = config.get_instance("database", "primary", database="myapp")
```

## API Reference

### Config Class

```python
class Config:
    def __init__(self, *sources, use_env=True)
    def from_file(cls, path) -> Config
    def from_dict(cls, data) -> Config
    
    # Access
    def get_types() -> List[str]
    def get_count(type_name: str) -> int
    def get_names(type_name: str) -> List[str]
    def get(type_name: str, name_or_index: Union[str, int] = 0) -> dict
    def set(type_name: str, name_or_index: Union[str, int], config: dict)
    
    # References
    def resolve_reference(ref: str) -> dict
    def build_reference(type_name: str, name_or_index: Union[str, int]) -> str
    
    # Merging
    def merge(other: Config, precedence: str = "first")
    
    # Export
    def to_dict() -> dict
    def to_file(path: Path, format: str = None)
    
    # Object Construction
    def build_object(ref: str, cache: bool = True, **kwargs) -> Any
    def clear_object_cache(ref: str = None)
    
    # Lazy Factory Access
    def get_factory(type_name: str, name_or_index: Union[str, int] = 0) -> Any
    def get_instance(type_name: str, name_or_index: Union[str, int] = 0, **kwargs) -> Any
```

## Examples

### Multi-Environment Configuration

```python
# base.yaml
database:
  - name: primary
    host: localhost
    port: 5432

# production.yaml  
database:
  - name: primary
    host: prod.db.example.com
    pool_size: 50

# Load with overrides
config = Config("base.yaml", "production.yaml")
```

### Service Discovery Integration

```python
config = Config({
    "services": [
        {"name": "auth", "url": "http://auth:8000"},
        {"name": "api", "url": "http://api:8080"}
    ],
    "app": [{
        "name": "main",
        "auth_service": "xref:services[auth]",
        "api_service": "xref:services[api]"
    }]
})

app = config.resolve_reference("xref:app[main]")
# app["auth_service"]["url"] = "http://auth:8000"
```

### Dynamic Configuration with Environment

```python
# Development: export DATAKNOBS_DATABASE__PRIMARY__HOST=localhost
# Production:  export DATAKNOBS_DATABASE__PRIMARY__HOST=prod.db.aws.com

config = Config.from_file("config.yaml")
db = config.get("database", "primary")
# Automatically uses environment-appropriate host
```

## Best Practices

1. **Use Type Organization**: Group related configurations by type
2. **Leverage Defaults**: Define common values in settings to avoid repetition
3. **Environment Overrides**: Use for deployment-specific values (hosts, ports, credentials)
4. **File References**: Split large configurations into manageable files
5. **Path Resolution**: Use relative paths in configs for portability
6. **Object Caching**: Enable caching for expensive object construction

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.