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

The `$requires` field is stripped during resolution — it's validation
metadata, not passed to the provider constructor. `capabilities` is **not**
stripped: it is read to validate `$requires` and then passed through with the
rest of the resource config, so a provider that receives one must tolerate
the keyword.

A `$requires` naming a resource this environment does not define raises
`ResourceNotFoundError` — an absent resource satisfies no capability. See
[Missing Resources](../../config/environment-aware.md#4-missing-resources)
for the full policy and how to opt out with `$required: false`.

### Config Merging

Additional fields in a resource reference are **inline defaults**: they fill
keys the environment's resource does not set, and are discarded wherever it
does. The environment wins — a binding is the deployment's to decide.

The exceptions are the marker keys — `$resource`, `type`, `$requires` and
`$required` — which are the reference's own syntax and never reach a factory.
That set is closed: any other `$`-prefixed key is rejected as a malformed
reference rather than merged, and so is a `$required` / `$requires` on a
block with no `$resource`, which is how a misspelled `$resource` gives itself
away.

```yaml
# In bot config
llm:
  $resource: default
  type: llm_providers
  temperature: 0.9          # discarded if the environment sets it
  timeout: 30               # kept if the environment does not

# If environment defines temperature: 0.7
# Resolved config has temperature: 0.7 and timeout: 30
```

### When a Resource Is Missing

A reference naming a resource the environment does not define resolves to
the reference's inline defaults, with a warning — or to `{}` when it
declares none. That is usually right in development and usually wrong in
production: an empty config handed to a factory rarely fails, it produces
the factory's default. A degraded `conversation_storage` binding becomes an
in-memory database, which holds state perfectly until the process restarts.

[Missing Resources](../../config/environment-aware.md#4-missing-resources)
documents the full four-level precedence chain. Two of those levels live in
code, and every entry point here that resolves a config exposes them:

| Entry point | Default | Where to say otherwise |
|---|---|---|
| `DynaBot.from_environment_aware_config` | lenient — warn and degrade | `strict_resources=` on the call |
| `BotRegistry` / `InMemoryBotRegistry` | lenient | `strict_resources=` on the constructor |
| `BotManager` (deprecated) | lenient | `strict_resources=` on the constructor |
| `ConfigCachingManager` | **strict — raises** | `strict_resources=` on the constructor |
| `BotResourceResolver` | **strict — raises** | not configurable |

`ConfigCachingManager` and `BotResourceResolver` differ because raising is
what those two paths have always done; the default is preserved rather than
moved. Pass `strict_resources=None` to `ConfigCachingManager` to hand the
decision to the environment's own setting instead.

```python
from dataknobs_bots.bot import InMemoryBotRegistry

# Fail at startup on a binding production does not define, rather than
# discovering it as an empty conversation history after deployment
registry = InMemoryBotRegistry(
    environment="production",
    strict_resources=True,
)
```

The registries and the manager take the policy on the constructor rather
than per call because all three cache: an argument passed to one
`get_bot` / `get_or_create` would silently decide what every later caller
receives. For a per-resolution decision, call
`DynaBot.from_environment_aware_config` directly.

Leaving it unset changes nothing — `None` defers to the config's own
policy, then to the environment's `strict_resources` setting, then to
lenient. A reference's own `$required` overrides every level, so a single
binding can opt in or out regardless of what the code asked for:

```yaml
conversation_storage:
  $resource: conversations
  type: databases
  $required: true       # absent in this environment -> raise, do not degrade
```

`ResourceNotFoundError` subclasses `KeyError`, so code wrapping bot
creation in `except KeyError` for unrelated reasons will swallow it.

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
