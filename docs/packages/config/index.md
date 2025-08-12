# dataknobs-config

A modular configuration system with composable settings, environment variable overrides, and cross-references.

## Overview

The `dataknobs-config` package provides a flexible and powerful configuration management system designed for complex applications. It supports:

- **Modular configurations** organized by type
- **Atomic configuration units** with validation
- **Cross-references** between configurations
- **Environment variable overrides** with bash-compatible naming
- **Path resolution** for relative paths
- **Object construction** from configurations
- **Global and type-specific settings**

## Installation

```bash
pip install dataknobs-config
```

## Quick Start

```python
from dataknobs_config import Config

# Load configuration from a YAML file
config = Config("config.yaml")

# Get a specific configuration
db_config = config.get("database", "production")

# Get all configurations of a type
all_databases = config.get_all("database")

# Use with environment overrides
# export DATAKNOBS_DATABASE__PRODUCTION__HOST=new-host.com
config = Config("config.yaml", use_env=True)
```

## Key Features

### 1. Modular Design

Organize configurations by type, with each type containing multiple atomic configurations:

```yaml
database:
  - name: primary
    host: localhost
    port: 5432
  - name: replica
    host: localhost
    port: 5433

cache:
  - name: redis
    host: localhost
    port: 6379
```

### 2. Cross-References

Link configurations using the `xref:` syntax:

```yaml
app:
  - name: web
    database: "xref:database[primary]"
    cache: "xref:cache[redis]"
```

### 3. Environment Overrides

Override any configuration value using environment variables:

```bash
export DATAKNOBS_DATABASE__PRIMARY__HOST=prod.db.com
export DATAKNOBS_CACHE__REDIS__PORT=6380
```

### 4. Path Resolution

Automatically resolve relative paths based on global or type-specific roots:

```yaml
settings:
  global_root: /etc/myapp
  path_resolution_attributes: ["config_file", "data_dir"]

service:
  - name: api
    config_file: configs/api.yaml  # Resolved to /etc/myapp/configs/api.yaml
```

### 5. Object Construction

Build objects directly from configurations:

```yaml
logger:
  - name: main
    class: logging.Logger
    level: INFO
```

```python
logger = config.build_object("xref:logger[main]")
```

## Configuration File Format

Configuration files can be in YAML or JSON format:

```yaml
# config.yaml
settings:
  global_root: /app
  path_resolution_attributes: ["path", "dir"]

database:
  - name: main
    host: localhost
    port: 5432
    
cache:
  - name: redis
    host: localhost
    port: 6379

app:
  - name: web
    database: "xref:database[main]"
    cache: "xref:cache[redis]"
```

## API Overview

### Config Class

The main configuration management class:

- `Config(*sources, use_env=True)` - Initialize with one or more sources
- `get(type_name, name_or_index)` - Get a specific configuration
- `get_all(type_name)` - Get all configurations of a type
- `set(type_name, name_or_index, value)` - Set a configuration value
- `resolve_reference(ref)` - Resolve a string reference
- `build_object(ref)` - Build an object from configuration
- `to_dict()` - Export as dictionary
- `to_file(path)` - Save to file

## Examples

### Basic Usage

```python
from dataknobs_config import Config

# Load from file
config = Config("config.yaml")

# Access configurations
db = config.get("database", "primary")
print(f"Connecting to {db['host']}:{db['port']}")

# Update configuration
config.set("database", "primary", {"host": "new-host.com", "port": 5432})

# Save changes
config.to_file("updated_config.yaml")
```

### Using References

```python
config = Config({
    "database": [{"name": "main", "host": "localhost"}],
    "app": [{"name": "web", "db": "xref:database[main]"}]
})

app_config = config.get("app", "web")
# app_config["db"] will contain the resolved database configuration
```

### Environment Overrides

```python
import os
os.environ["DATAKNOBS_DATABASE__MAIN__HOST"] = "prod.db.com"

config = Config("config.yaml", use_env=True)
db = config.get("database", "main")
print(db["host"])  # "prod.db.com"
```

## Best Practices

1. **Use meaningful type names** that reflect the configuration's purpose
2. **Keep atomic configurations small** and focused on a single concern
3. **Use references** to avoid duplication and maintain consistency
4. **Document your settings** section for team members
5. **Use environment overrides** for deployment-specific values
6. **Validate configurations** early in your application startup

## Further Reading

- [API Reference](api.md)
- [Advanced Usage Guide](../../user-guide/advanced-usage.md)
- [Migration Guide](../../migration-guide.md)