# Environment-Aware Configuration

DynaBot supports **environment-aware configuration** for deploying the same bot configuration across different environments (development, staging, production). This separates portable bot behavior from environment-specific infrastructure bindings.

## Overview

### The Problem

Traditional bot configurations contain environment-specific details:

```yaml
# PROBLEMATIC: This config is not portable
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434  # Only works locally

conversation_storage:
  backend: sqlite
  path: ~/.local/share/myapp/conversations.db  # Local path
```

When stored in a shared registry or database, these configs fail in production because:

- Localhost URLs don't exist in production
- Local file paths don't exist in containers
- Different environments need different backends

### The Solution

Use **logical resource references** (`$resource`) to separate behavior from infrastructure:

```yaml
# PORTABLE: This config works in any environment
bot:
  llm:
    $resource: default          # Logical name
    type: llm_providers         # Resource type
    temperature: 0.7            # Behavioral setting

  conversation_storage:
    $resource: conversations
    type: databases
```

The logical names are resolved at **instantiation time** against environment-specific bindings.

## Quick Start

### 1. Create Environment Config Files

```yaml
# config/environments/development.yaml
name: development
resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434
  databases:
    conversations:
      backend: memory
```

```yaml
# config/environments/production.yaml
name: production
resources:
  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
  databases:
    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
```

### 2. Create Portable Bot Config

```yaml
# config/bots/assistant.yaml
name: assistant
bot:
  llm:
    $resource: default
    type: llm_providers
    temperature: 0.7

  conversation_storage:
    $resource: conversations
    type: databases

  system_prompt: |
    You are a helpful assistant.
```

### 3. Resolve Resources in Code

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import BotResourceResolver

# Auto-detect environment from DATAKNOBS_ENVIRONMENT
env = EnvironmentConfig.load()

# Create resolver with all DynaBot factories
resolver = BotResourceResolver(env)

# Get initialized resources
llm = await resolver.get_llm("default")
db = await resolver.get_database("conversations")
```

## Environment Detection

The environment is determined in this order:

1. **Explicit**: `DATAKNOBS_ENVIRONMENT=production`
2. **Cloud indicators**: AWS Lambda, ECS, Kubernetes, GCP Cloud Run, Azure Functions
3. **Default**: `development`

```bash
# Set environment explicitly
export DATAKNOBS_ENVIRONMENT=production

# Or auto-detect based on cloud environment
# AWS Lambda: AWS_EXECUTION_ENV
# Kubernetes: KUBERNETES_SERVICE_HOST
# Cloud Run: K_SERVICE
# Azure Functions: FUNCTIONS_WORKER_RUNTIME
```

## Resource Reference Syntax

### Basic Syntax

```yaml
llm:
  $resource: default        # Logical resource name
  type: llm_providers       # Resource type
  $requires: [function_calling]  # Required capabilities (optional)
  temperature: 0.7          # Merged into resolved config
```

### Supported Resource Types

| Type | Description |
|------|-------------|
| `llm_providers` | LLM providers (OpenAI, Anthropic, Ollama) |
| `databases` | Database backends (memory, sqlite, postgres) |
| `vector_stores` | Vector store backends (memory, FAISS, pgvector) |
| `embedding_providers` | Embedding providers |

### Capability Requirements (`$requires`)

Bot configs can declare required capabilities using `$requires`:

```yaml
llm:
  $resource: default
  type: llm_providers
  $requires: [function_calling]
```

If the resolved resource declares `capabilities` metadata, the system validates that all requirements are met. Missing capabilities raise a `ConfigError` at resolution time.

Requirements are also **inferred** from bot structure — for example, a bot with `reasoning.strategy: react` and `tools` automatically requires `function_calling`. Explicit `$requires` is additive.

### Capability Metadata on Resources

Environment configs can declare what capabilities each resource provides:

```yaml
resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      capabilities: [chat, function_calling, streaming]
    fast:
      provider: ollama
      model: gemma3:4b
      capabilities: [chat, streaming]
```

The `capabilities` field is stripped during resolution — it's validation metadata, not passed to the provider constructor.

### Config Merging

Additional fields in a resource reference (except `$resource`, `type`, and `$requires`) are merged with the resolved config:

```yaml
# In bot config
llm:
  $resource: default
  type: llm_providers
  temperature: 0.9          # Overrides environment default

# If environment defines temperature: 0.7
# Resolved config will have temperature: 0.9
```

## BotResourceResolver

High-level resolver that automatically initializes resources.

### Basic Usage

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import BotResourceResolver

env = EnvironmentConfig.load()
resolver = BotResourceResolver(env)

# Get initialized LLM (calls initialize() automatically)
llm = await resolver.get_llm("default")

# Get connected database (calls connect() automatically)
db = await resolver.get_database("conversations")

# Get initialized vector store
vs = await resolver.get_vector_store("knowledge")

# Get initialized embedding provider
embedder = await resolver.get_embedding_provider("default")
```

### Config Overrides

```python
# Override temperature for this resolution
llm = await resolver.get_llm("default", temperature=0.9)

# Get fresh instance (skip cache)
llm = await resolver.get_llm("default", use_cache=False)
```

### Cache Management

```python
# Clear all cached resources
resolver.clear_cache()

# Clear only LLM providers
resolver.clear_cache("llm_providers")
```

## Low-Level Resolution

For more control, use `create_bot_resolver`:

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import create_bot_resolver

env = EnvironmentConfig.load("production")
resolver = create_bot_resolver(env)

# Resolve without auto-initialization
llm = resolver.resolve("llm_providers", "default")
await llm.initialize()  # Manual initialization

# Check registered factories
resolver.has_factory("llm_providers")  # True
resolver.get_registered_types()  # ['llm_providers', 'databases', ...]
```

### Custom Factory Registration

```python
from dataknobs_config import ConfigBindingResolver, EnvironmentConfig
from dataknobs_bots.config import (
    register_llm_factory,
    register_database_factory,
)

env = EnvironmentConfig.load()

# Create resolver without defaults
resolver = create_bot_resolver(env, register_defaults=False)

# Register only what you need
register_llm_factory(resolver)
register_database_factory(resolver)
```

## Environment Config Format

### Full Example

```yaml
# config/environments/production.yaml
name: production
description: Production environment

settings:
  log_level: INFO
  enable_metrics: true

resources:
  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
      temperature: 0.7
      max_tokens: 2000
      capabilities: [chat, function_calling, streaming]

    fast:
      provider: openai
      model: gpt-3.5-turbo
      api_key: ${OPENAI_API_KEY}
      capabilities: [chat, streaming]

  databases:
    default:
      backend: postgres
      connection_string: ${DATABASE_URL}

    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
      pool_size: 20

  vector_stores:
    default:
      backend: pgvector
      connection_string: ${DATABASE_URL}
      dimensions: 1536

    knowledge:
      backend: pgvector
      connection_string: ${DATABASE_URL}
      dimensions: 1536
      table: knowledge_vectors

  embedding_providers:
    default:
      provider: openai
      model: text-embedding-3-small
      api_key: ${OPENAI_API_KEY}
```

## Best Practices

### 1. Store Portable Configs

Only store configs with `$resource` references in databases and registries.

### 2. Use Late Binding

Resolve environment variables at instantiation time, not load time.

### 3. Define All Environments

Create config files for development, staging, and production.

### 4. Use Consistent Names

Use the same logical names across all environment configs.

### 5. Keep Behavior Separate

Put behavioral settings (temperature, max_tokens) in bot configs, infrastructure settings in environment configs.

## Integration with EnvironmentAwareConfig

For full config resolution including `$resource` references:

```python
from dataknobs_config import EnvironmentAwareConfig

# Load app config with environment bindings
config = EnvironmentAwareConfig.load_app(
    "assistant",
    app_dir="config/bots",
    env_dir="config/environments"
)

# Resolve for building (late binding)
resolved = config.resolve_for_build("bot")

# Get portable config for storage
portable = config.get_portable_config()
```

## Next Steps

- [Migration Guide](migration.md) - Migrate existing configs
- [Configuration Reference](configuration.md) - All configuration options
- [Examples](../examples/index.md) - Working examples
