# Environment-Aware Configuration

The `dataknobs-config` package provides an **environment-aware configuration system** for deploying the same application across different environments (development, staging, production) where infrastructure differs.

## Overview

### The Problem

Traditional configurations contain environment-specific details that cause issues when stored in databases or shared registries:

```yaml
# PROBLEMATIC: This config is not portable
database:
  backend: postgres
  connection_string: postgresql://localhost:5432/mydb  # Only works locally

vector_store:
  backend: faiss
  path: ~/data/vectors  # Local path doesn't exist in production
```

When these configs are stored and then loaded in production:
- Localhost URLs don't exist in production containers
- Local file paths don't exist in containerized environments
- Different environments need different backends

### The Solution

Use **logical resource references** (`$resource`) to separate behavior from infrastructure:

```yaml
# PORTABLE: This config works in any environment
bot:
  database:
    $resource: default
    type: databases

  vector_store:
    $resource: knowledge
    type: vector_stores
```

The logical names are resolved at **instantiation time** against environment-specific bindings:

```yaml
# config/environments/development.yaml
resources:
  databases:
    default:
      backend: sqlite
      path: ~/.local/share/myapp/dev.db

# config/environments/production.yaml
resources:
  databases:
    default:
      backend: postgres
      connection_string: ${DATABASE_URL}
```

## Key Concepts

### 1. Late Binding

Environment-specific values are resolved at the **latest possible moment**:

- **Config loading**: Keep placeholders intact (`${DATABASE_URL}`)
- **Config storage**: Store unresolved app config
- **Object instantiation**: Resolve environment bindings

### 2. Logical Resource References

Instead of hardcoded infrastructure, use logical names that map to environment-specific implementations:

```yaml
# In app config
database:
  $resource: conversations
  type: databases
  pool_size: 10  # Merged with resolved config
```

### 3. Capability Requirements

Resource references can declare required capabilities using `$requires`:

```yaml
llm:
  $resource: default
  type: llm_providers
  $requires: [function_calling]   # Validated against resource capabilities
```

If the resolved resource declares `capabilities` metadata, the system validates that all requirements are met. Missing capabilities raise a `ConfigError` at resolution time.

Environment configs declare resource capabilities as metadata:

```yaml
resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      capabilities: [chat, function_calling, streaming]
```

The `capabilities` field is stripped during resolution — it's validation metadata, not a provider parameter. The `$requires` field is also stripped and not passed through.

### 3. Environment Detection

The system automatically detects the current environment via:

1. **Explicit**: `DATAKNOBS_ENVIRONMENT=production`
2. **Cloud indicators**: AWS Lambda, ECS, Kubernetes, GCP Cloud Run, Azure Functions
3. **Default**: `development`

## Classes

### EnvironmentConfig

Manages environment-specific resource bindings.

```python
from dataknobs_config import EnvironmentConfig

# Auto-detect environment
env = EnvironmentConfig.load()

# Or specify explicitly
env = EnvironmentConfig.load("production", config_dir="config/environments")

# Get concrete config for a logical resource
db_config = env.get_resource("databases", "conversations")
# Returns: {"backend": "postgres", "connection_string": "..."}

# Check if resource exists
if env.has_resource("databases", "analytics"):
    analytics = env.get_resource("databases", "analytics")

# Get environment settings
log_level = env.get_setting("log_level", "INFO")
```

#### Environment File Format

```yaml
# config/environments/production.yaml
name: production
description: AWS production environment

settings:
  log_level: INFO
  enable_metrics: true

resources:
  databases:
    default:
      backend: postgres
      connection_string: ${DATABASE_URL}
      pool_size: 20

    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
      table: conversations

  vector_stores:
    default:
      backend: pgvector
      connection_string: ${DATABASE_URL}

    knowledge:
      backend: pgvector
      connection_string: ${DATABASE_URL}
      dimensions: 1536

  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
      capabilities: [chat, function_calling, streaming]
```

### EnvironmentAwareConfig

Configuration with environment-aware resource resolution.

```python
from dataknobs_config import EnvironmentAwareConfig

# Load app config with auto-detected environment
config = EnvironmentAwareConfig.load_app(
    "my-bot",
    app_dir="config/apps",
    env_dir="config/environments"
)

# Get resolved config for object building (late binding happens here)
resolved = config.resolve_for_build()

# Resolve specific section
bot_config = config.resolve_for_build("bot")

# Get portable config for storage (no env vars resolved)
portable = config.get_portable_config()
```

#### Application Config Format

```yaml
# config/apps/my-bot.yaml
name: my-bot
version: "1.0.0"

bot:
  llm:
    $resource: default
    type: llm_providers
    $requires: [function_calling]  # Optional: require specific capabilities
    temperature: 0.7               # Merged into resolved config

  conversation_storage:
    $resource: conversations
    type: databases

  knowledge_base:
    vector_store:
      $resource: knowledge
      type: vector_stores

  system_prompt: |
    You are a helpful assistant.
```

### ConfigBindingResolver

Resolves logical resource bindings to concrete instances using factories.

```python
from dataknobs_config import (
    EnvironmentConfig,
    ConfigBindingResolver,
    SimpleFactory,
    CallableFactory,
    AsyncCallableFactory,
)

# Load environment
env = EnvironmentConfig.load("production")

# Create resolver
resolver = ConfigBindingResolver(env)

# Register factories for resource types
resolver.register_factory("databases", SimpleFactory(DatabaseConnection))
resolver.register_factory("vector_stores", CallableFactory(create_vector_store))

# Resolve a logical reference to a concrete instance
db = resolver.resolve("databases", "conversations")

# With config overrides
db = resolver.resolve("databases", "conversations", pool_size=50)

# Async resolution (for async factories)
vector_store = await resolver.resolve_async("vector_stores", "knowledge")

# Skip cache for fresh instance
fresh_db = resolver.resolve("databases", "conversations", use_cache=False)

# Clear cache
resolver.clear_cache()  # All resources
resolver.clear_cache("databases")  # Specific type
```

#### Factory Types

**SimpleFactory** - Creates instances of a class:

```python
from dataknobs_config import SimpleFactory

resolver.register_factory(
    "databases",
    SimpleFactory(DatabaseConnection, timeout=30)  # Default kwargs
)
```

**CallableFactory** - Wraps a callable:

```python
from dataknobs_config import CallableFactory

def create_database(backend, connection_string, **kwargs):
    if backend == "postgres":
        return PostgresDB(connection_string, **kwargs)
    elif backend == "sqlite":
        return SQLiteDB(connection_string, **kwargs)

resolver.register_factory("databases", CallableFactory(create_database))
```

**AsyncCallableFactory** - Wraps an async callable:

```python
from dataknobs_config import AsyncCallableFactory

async def create_database(backend, connection_string, **kwargs):
    db = DatabaseConnection(backend, connection_string)
    await db.connect()
    return db

resolver.register_factory("databases", AsyncCallableFactory(create_database))

# Must use resolve_async
db = await resolver.resolve_async("databases", "conversations")
```

## Usage Patterns

### Pattern 1: Direct Environment Config

For simple resource lookup without object instantiation:

```python
from dataknobs_config import EnvironmentConfig

env = EnvironmentConfig.load()

# Get config for a resource
db_config = env.get_resource("databases", "conversations")

# Use config directly
connection = create_connection(**db_config)
```

### Pattern 2: App Config with Late Binding

For applications with portable configuration:

```python
from dataknobs_config import EnvironmentAwareConfig

# Load portable app config
config = EnvironmentAwareConfig.load_app("my-app")

# Store portable config (safe for database storage)
db.store(config.get_portable_config())

# At runtime, resolve for current environment
resolved = config.resolve_for_build()
app = MyApp.from_config(resolved)
```

### Pattern 3: Full Factory Resolution

For applications needing complete object lifecycle management:

```python
from dataknobs_config import (
    EnvironmentConfig,
    ConfigBindingResolver,
    AsyncCallableFactory,
)

async def create_bot_resources():
    env = EnvironmentConfig.load()
    resolver = ConfigBindingResolver(env)

    # Register factories
    resolver.register_factory("databases", AsyncCallableFactory(create_db))
    resolver.register_factory("llm_providers", AsyncCallableFactory(create_llm))

    # Resolve resources
    db = await resolver.resolve_async("databases", "conversations")
    llm = await resolver.resolve_async("llm_providers", "default")

    return {"db": db, "llm": llm}
```

## Environment Detection

The environment is automatically detected in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | `DATAKNOBS_ENVIRONMENT` | `export DATAKNOBS_ENVIRONMENT=production` |
| 2 | AWS Lambda/ECS | `AWS_EXECUTION_ENV` present |
| 3 | AWS ECS Fargate | `ECS_CONTAINER_METADATA_URI` present |
| 4 | Kubernetes | `KUBERNETES_SERVICE_HOST` present |
| 5 | Google Cloud Run | `K_SERVICE` present |
| 6 | Azure Functions | `FUNCTIONS_WORKER_RUNTIME` present |
| 7 | Default | `development` |

For cloud environments, the actual environment name (staging, production) is read from the `ENVIRONMENT` variable if present.

## Best Practices

### 1. Store Portable Configs

Only store configs with `$resource` references in databases and registries:

```python
# CORRECT: Store portable config
registry.store(domain_id, config.get_portable_config())

# At load time, resolve for current environment
stored = registry.load(domain_id)
config = EnvironmentAwareConfig(stored)
resolved = config.resolve_for_build()
```

### 2. Use Late Binding

Resolve environment variables at instantiation time, not load time:

```python
# Loading keeps placeholders
config = EnvironmentAwareConfig.load_app("my-app")
# config still has ${DATABASE_URL} placeholders

# Resolution substitutes env vars
resolved = config.resolve_for_build()
# resolved has actual connection strings
```

### 3. Consistent Resource Names

Use the same logical names across all environment configs:

```yaml
# development.yaml
resources:
  databases:
    conversations: ...  # Same name

# production.yaml
resources:
  databases:
    conversations: ...  # Same name
```

### 4. Separate Behavior from Infrastructure

Put behavioral settings and capability requirements in app configs, infrastructure and capability metadata in environment configs:

```yaml
# App config (portable)
llm:
  $resource: default
  type: llm_providers
  $requires: [function_calling]  # What the app needs
  temperature: 0.7               # Behavioral
  max_tokens: 2000               # Behavioral

# Environment config (per-environment)
llm_providers:
  default:
    provider: openai              # Infrastructure
    api_key: ${KEY}               # Infrastructure
    capabilities: [chat, function_calling, streaming]  # What it provides
```

## API Reference

### EnvironmentConfig

| Method | Description |
|--------|-------------|
| `load(environment, config_dir)` | Load environment config from file |
| `from_dict(data)` | Create from dictionary |
| `detect_environment()` | Detect current environment |
| `get_resource(type, name, defaults)` | Get resource config |
| `has_resource(type, name)` | Check if resource exists |
| `get_setting(key, default)` | Get environment setting |
| `get_resource_types()` | List all resource types |
| `get_resource_names(type)` | List resources of a type |
| `merge(other)` | Merge with another config |
| `to_dict()` | Export as dictionary |

### EnvironmentAwareConfig

| Method | Description |
|--------|-------------|
| `load_app(name, app_dir, env_dir, environment)` | Load app with environment |
| `from_dict(config, environment, env_dir)` | Create from dictionary |
| `resolve_for_build(key, resolve_resources, resolve_env_vars)` | Late-bind config |
| `get_portable_config()` | Get unresolved config |
| `get(key, default)` | Get config value |
| `with_environment(environment, env_dir)` | Create with different env |
| `get_resource(type, name, defaults)` | Direct resource access |
| `get_setting(key, default)` | Direct setting access |

### ConfigBindingResolver

| Method | Description |
|--------|-------------|
| `register_factory(type, factory)` | Register resource factory |
| `unregister_factory(type)` | Remove factory |
| `has_factory(type)` | Check if factory registered |
| `get_registered_types()` | List registered types |
| `resolve(type, name, use_cache, **overrides)` | Resolve to instance |
| `resolve_async(type, name, use_cache, **overrides)` | Async resolution |
| `get_cached(type, name)` | Get cached instance |
| `is_cached(type, name)` | Check if cached |
| `clear_cache(type)` | Clear cache |
| `cache_instance(type, name, instance)` | Manually cache |

## See Also

- [Configuration System](configuration-system.md) - Core Config class
- [Environment Variables](environment-variables.md) - Environment override system
- [Factory Registration](factory-registration.md) - Object construction patterns
