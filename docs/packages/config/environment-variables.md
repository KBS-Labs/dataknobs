# Environment Variables

The DataKnobs Config package provides comprehensive environment variable support for overriding configuration values at runtime. This enables secure management of sensitive data and environment-specific configuration without modifying files.

## Overview

Environment variables can override any configuration value using a structured naming convention. The system supports:

- Automatic type conversion
- Nested attribute access
- Default values
- Variable substitution in configuration files
- Both named and indexed access

## Naming Convention

Environment variables follow this pattern:

```
DATAKNOBS_<TYPE>__<NAME_OR_INDEX>__<ATTRIBUTE>
```

- **DATAKNOBS**: Default prefix (configurable)
- **TYPE**: Configuration type (e.g., DATABASE, CACHE, SERVICE)
- **NAME_OR_INDEX**: Item name or numeric index
- **ATTRIBUTE**: Configuration attribute (supports nesting)

### Examples

```bash
# Override database host by name
DATAKNOBS_DATABASE__PRIMARY__HOST=prod.example.com

# Override database port by index
DATAKNOBS_DATABASE__0__PORT=5433

# Override nested attribute
DATAKNOBS_DATABASE__PRIMARY__CONNECTION__TIMEOUT=60

# Override cache TTL
DATAKNOBS_CACHE__REDIS__TTL=7200
```

## Type Conversion

Values are automatically converted to appropriate types:

```bash
# String (default)
DATAKNOBS_DATABASE__PRIMARY__HOST=localhost

# Integer
DATAKNOBS_DATABASE__PRIMARY__PORT=5432

# Float
DATAKNOBS_SERVICE__API__TIMEOUT=30.5

# Boolean (true, false, yes, no, 1, 0)
DATAKNOBS_DATABASE__PRIMARY__SSL_ENABLED=true
DATAKNOBS_SERVICE__API__DEBUG=1
```

## Applying Environment Overrides

### During Configuration Load

```python
from dataknobs_config import Config

# Apply environment overrides automatically
config = Config.from_file("config.yaml", apply_env_overrides=True)

# Or apply manually
config = Config.from_file("config.yaml")
config.apply_env_overrides()
```

### Custom Prefix

```python
# Use custom prefix
config.apply_env_overrides(prefix="MYAPP_")

# Now use: MYAPP_DATABASE__PRIMARY__HOST=localhost
```

### Selective Application

```python
# Apply only to specific types
config.apply_env_overrides(types=["databases", "caches"])

# Apply with filter function
def filter_func(var_name, value):
    return not var_name.endswith("__PASSWORD")

config.apply_env_overrides(filter_func=filter_func)
```

## Variable Substitution in Files

Configuration files can reference environment variables directly:

### Basic Substitution

```yaml
database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  password: ${DB_PASSWORD}
```

### With Default Values

```yaml
database:
  # Colon syntax
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  
  # Bash-style syntax
  username: ${DB_USER:-postgres}
  password: ${DB_PASS:-}
```

### Nested Substitution

```yaml
database:
  connection_string: ${DB_PROTOCOL:postgresql}://${DB_HOST}:${DB_PORT}/${DB_NAME}
```

## Named vs Indexed Access

### Named Access

Use the configuration item's name:

```bash
# config.yaml:
# databases:
#   - name: primary
#     host: localhost

DATAKNOBS_DATABASE__PRIMARY__HOST=prod.example.com
```

### Indexed Access

Use numeric indices (0-based):

```bash
# First database
DATAKNOBS_DATABASE__0__HOST=prod.example.com

# Second database
DATAKNOBS_DATABASE__1__HOST=analytics.example.com

# Last database (negative indexing)
DATAKNOBS_DATABASE__-1__HOST=backup.example.com
```

## Nested Attributes

Access deeply nested configuration attributes:

```bash
# config.yaml:
# databases:
#   - name: primary
#     connection:
#       pool:
#         min_size: 5
#         max_size: 20

DATAKNOBS_DATABASE__PRIMARY__CONNECTION__POOL__MIN_SIZE=10
DATAKNOBS_DATABASE__PRIMARY__CONNECTION__POOL__MAX_SIZE=50
```

## Lists and Arrays

Override list values using indexed notation:

```bash
# config.yaml:
# service:
#   allowed_origins:
#     - http://localhost:3000
#     - http://localhost:8080

DATAKNOBS_SERVICE__API__ALLOWED_ORIGINS__0=https://app.example.com
DATAKNOBS_SERVICE__API__ALLOWED_ORIGINS__1=https://www.example.com
```

## Complex Examples

### Database Configuration

```bash
# Development
export DATAKNOBS_DATABASE__PRIMARY__HOST=localhost
export DATAKNOBS_DATABASE__PRIMARY__PORT=5432
export DATAKNOBS_DATABASE__PRIMARY__USERNAME=dev_user
export DATAKNOBS_DATABASE__PRIMARY__PASSWORD=dev_pass

# Production
export DATAKNOBS_DATABASE__PRIMARY__HOST=prod-db.example.com
export DATAKNOBS_DATABASE__PRIMARY__PORT=5432
export DATAKNOBS_DATABASE__PRIMARY__USERNAME=prod_user
export DATAKNOBS_DATABASE__PRIMARY__PASSWORD=${SECRET_DB_PASSWORD}
export DATAKNOBS_DATABASE__PRIMARY__SSL_ENABLED=true
export DATAKNOBS_DATABASE__PRIMARY__POOL_SIZE=50
```

### Service Configuration

```bash
# API Service
export DATAKNOBS_SERVICE__API__PORT=8000
export DATAKNOBS_SERVICE__API__HOST=0.0.0.0
export DATAKNOBS_SERVICE__API__DEBUG=false
export DATAKNOBS_SERVICE__API__LOG_LEVEL=INFO
export DATAKNOBS_SERVICE__API__RATE_LIMIT=1000

# Worker Service
export DATAKNOBS_SERVICE__WORKER__CONCURRENCY=10
export DATAKNOBS_SERVICE__WORKER__QUEUE_NAME=tasks
export DATAKNOBS_SERVICE__WORKER__RETRY_ATTEMPTS=3
```

### Cache Configuration

```bash
# Redis Cache
export DATAKNOBS_CACHE__REDIS__HOST=redis.example.com
export DATAKNOBS_CACHE__REDIS__PORT=6379
export DATAKNOBS_CACHE__REDIS__DB=0
export DATAKNOBS_CACHE__REDIS__TTL=3600
export DATAKNOBS_CACHE__REDIS__MAX_CONNECTIONS=100
```

## Docker and Container Usage

### Docker Compose

```yaml
version: '3.8'
services:
  app:
    image: myapp:latest
    environment:
      - DATAKNOBS_DATABASE__PRIMARY__HOST=db
      - DATAKNOBS_DATABASE__PRIMARY__PORT=5432
      - DATAKNOBS_DATABASE__PRIMARY__USERNAME=postgres
      - DATAKNOBS_DATABASE__PRIMARY__PASSWORD=${DB_PASSWORD}
      - DATAKNOBS_CACHE__REDIS__HOST=redis
      - DATAKNOBS_SERVICE__API__PORT=8000
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATAKNOBS_DATABASE__PRIMARY__HOST: "postgres-service"
  DATAKNOBS_DATABASE__PRIMARY__PORT: "5432"
  DATAKNOBS_CACHE__REDIS__HOST: "redis-service"
  DATAKNOBS_SERVICE__API__LOG_LEVEL: "INFO"
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  DATAKNOBS_DATABASE__PRIMARY__PASSWORD: "secret-password"
  DATAKNOBS_SERVICE__API__SECRET_KEY: "secret-api-key"
```

## .env File Support

Use .env files for local development:

```bash
# .env
DATAKNOBS_DATABASE__PRIMARY__HOST=localhost
DATAKNOBS_DATABASE__PRIMARY__PORT=5432
DATAKNOBS_DATABASE__PRIMARY__USERNAME=dev_user
DATAKNOBS_DATABASE__PRIMARY__PASSWORD=dev_password
DATAKNOBS_CACHE__REDIS__HOST=localhost
DATAKNOBS_CACHE__REDIS__PORT=6379
DATAKNOBS_SERVICE__API__DEBUG=true
DATAKNOBS_SERVICE__API__LOG_LEVEL=DEBUG
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
from dataknobs_config import Config

# Load .env file
load_dotenv()

# Apply environment overrides
config = Config.from_file("config.yaml", apply_env_overrides=True)
```

## Debugging Environment Variables

### List Applied Overrides

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

config = Config.from_file("config.yaml")
overrides = config.apply_env_overrides(return_applied=True)

print("Applied overrides:")
for key, value in overrides.items():
    print(f"  {key}: {value}")
```

### Validate Environment Variables

```python
def validate_env_overrides(config):
    """Validate that required environment variables are set."""
    required = [
        "DATAKNOBS_DATABASE__PRIMARY__PASSWORD",
        "DATAKNOBS_SERVICE__API__SECRET_KEY",
    ]
    
    missing = []
    for var in required:
        if var not in os.environ:
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

# Use before applying overrides
validate_env_overrides(config)
config.apply_env_overrides()
```

## Best Practices

### 1. Security

- Never commit sensitive environment variables to version control
- Use secrets management systems in production
- Validate that required secrets are set before starting

### 2. Naming

- Use consistent, descriptive names
- Group related variables with common prefixes
- Document all environment variables

### 3. Defaults

- Provide sensible defaults in configuration files
- Use environment variables for overrides, not base configuration
- Document which values are commonly overridden

### 4. Type Safety

```python
# Validate types after applying overrides
def validate_types(config):
    db_config = config.get("databases", "primary")
    assert isinstance(db_config["port"], int)
    assert isinstance(db_config["ssl_enabled"], bool)
```

### 5. Documentation

Create an environment variable reference:

```markdown
# Environment Variables Reference

## Database Configuration
- `DATAKNOBS_DATABASE__PRIMARY__HOST`: Database host (default: localhost)
- `DATAKNOBS_DATABASE__PRIMARY__PORT`: Database port (default: 5432)
- `DATAKNOBS_DATABASE__PRIMARY__USERNAME`: Database username (required)
- `DATAKNOBS_DATABASE__PRIMARY__PASSWORD`: Database password (required)

## Cache Configuration
- `DATAKNOBS_CACHE__REDIS__HOST`: Redis host (default: localhost)
- `DATAKNOBS_CACHE__REDIS__PORT`: Redis port (default: 6379)
```

## Troubleshooting

### Common Issues

1. **Variables Not Applied**: Ensure `apply_env_overrides=True` or call `apply_env_overrides()`
2. **Wrong Type**: Check automatic type conversion is working as expected
3. **Name Mismatch**: Verify configuration item names match environment variable names
4. **Case Sensitivity**: Environment variable names are case-sensitive

### Debug Mode

```python
# Enable detailed logging
config.apply_env_overrides(debug=True)

# Or set environment variable
os.environ["DATAKNOBS_DEBUG"] = "true"
```

## Advanced Usage

### Custom Override Logic

```python
from dataknobs_config import Config

class CustomConfig(Config):
    def apply_env_overrides(self, **kwargs):
        # Custom preprocessing
        self.preprocess_env_vars()
        
        # Apply standard overrides
        super().apply_env_overrides(**kwargs)
        
        # Custom postprocessing
        self.validate_overrides()
```

### Dynamic Environment Variables

```python
import os

def set_dynamic_env_vars(environment):
    """Set environment variables based on deployment environment."""
    if environment == "production":
        os.environ["DATAKNOBS_DATABASE__PRIMARY__POOL_SIZE"] = "50"
        os.environ["DATAKNOBS_SERVICE__API__WORKERS"] = "4"
    else:
        os.environ["DATAKNOBS_DATABASE__PRIMARY__POOL_SIZE"] = "10"
        os.environ["DATAKNOBS_SERVICE__API__WORKERS"] = "1"

set_dynamic_env_vars("production")
config = Config.from_file("config.yaml", apply_env_overrides=True)
```