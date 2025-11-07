# dataknobs-config API Reference

## Overview

The `dataknobs-config` package provides a modular configuration system with composable settings, environment variable overrides, and cross-references between configurations.

> **💡 Quick Links:**
> - [Complete API Documentation](complete-reference.md#dataknobs-config) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/config/src/dataknobs_config) - Browse on GitHub
> - [Package Guide](../packages/config/index.md) - Detailed documentation

## Main Classes

### Config

**Source:** [`config.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py)

The main configuration management class.

```python
from dataknobs_config import Config

# Load from file
config = Config("config.yaml")

# Load from dictionary
config = Config({"database": [{"name": "prod", "host": "localhost"}]})

# Load from multiple sources
config = Config("base.yaml", "overrides.yaml")
```

#### Key Methods

- [`get(type_name, name_or_index)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Get a specific configuration
- [`get_all(type_name)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Get all configurations of a type
- [`set(type_name, name_or_index, value)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Set a configuration value
- [`resolve_reference(ref)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Resolve a string reference
- [`build_object(ref)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Build an object from configuration
- [`to_dict()`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Export configuration as dictionary
- [`to_file(path)`](https://github.com/kbs-labs/dataknobs/blob/main/packages/config/src/dataknobs_config/config.py) - Save configuration to file

### Reference System

String references allow cross-referencing between configurations:

```yaml
database:
  - name: primary
    host: localhost
    port: 5432

app:
  - name: web
    database: "xref:database[primary]"  # References the primary database
```

### Environment Overrides

Override any configuration value using environment variables:

```bash
# Override database host
export DATAKNOBS_DATABASE__PRIMARY__HOST=prod.example.com

# Override with index
export DATAKNOBS_DATABASE__0__PORT=5433
```

### Settings Management

Global and type-specific settings:

```yaml
settings:
  global_root: /etc/myapp
  database.global_root: /var/lib/databases
  path_resolution_attributes: ["config_file", "data_dir"]
```

### Object Construction

Build objects from configurations:

```yaml
logger:
  - name: main
    class: logging.Logger
    level: INFO
    handlers: ["console"]
```

```python
# Build the logger object
logger = config.build_object("xref:logger[main]")
```

## Exceptions

- `ConfigError` - Base exception for configuration errors
- `ConfigNotFoundError` - Configuration not found
- `InvalidReferenceError` - Invalid reference format
- `ValidationError` - Configuration validation error
- `FileNotFoundError` - Configuration file not found

## Full Example

```python
from dataknobs_config import Config

# Create configuration
config = Config({
    "settings": {
        "global_root": "/app",
        "path_resolution_attributes": ["config_path"]
    },
    "database": [
        {
            "name": "primary",
            "host": "localhost",
            "port": 5432,
            "config_path": "configs/db.conf"  # Will be resolved to /app/configs/db.conf
        }
    ],
    "cache": [
        {
            "name": "redis",
            "host": "localhost",
            "port": 6379
        }
    ],
    "app": [
        {
            "name": "web",
            "database": "xref:database[primary]",
            "cache": "xref:cache[redis]"
        }
    ]
})

# Access configurations
db_config = config.get("database", "primary")
app_config = config.get("app", "web")

# The app_config will have resolved references:
# {
#     "name": "web",
#     "database": {"name": "primary", "host": "localhost", "port": 5432, "config_path": "/app/configs/db.conf"},
#     "cache": {"name": "redis", "host": "localhost", "port": 6379}
# }

# Environment overrides (if DATAKNOBS_DATABASE__PRIMARY__HOST=prod.db.com is set)
# The database host will be overridden to "prod.db.com"

# Save configuration
config.to_file("output.yaml")
```

For more details, see the [package documentation](../packages/config/index.md).