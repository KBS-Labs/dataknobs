# Migration Guide: Environment-Aware Configuration

This guide helps you migrate existing DynaBot configurations to the new environment-aware configuration system.

## Overview

The new system separates **portable bot configuration** (behavior) from **environment-specific infrastructure** (LLM providers, databases, vector stores). This enables:

- Same bot config works in development, staging, and production
- No more local paths leaking into production databases
- Easier multi-tenant deployments
- Clear separation of concerns

## Quick Migration Checklist

1. [ ] Identify infrastructure settings in your bot configs
2. [ ] Create environment config files
3. [ ] Convert infrastructure settings to `$resource` references
4. [ ] Update code to use environment-aware loading
5. [ ] Test in each environment

## Step-by-Step Migration

### Step 1: Identify Infrastructure Settings

Look for environment-specific values in your bot configs:

**Problematic patterns to find:**
- File paths (`~/data/...`, `/home/...`, `/app/...`)
- Connection strings (`postgresql://...`)
- API endpoints (`http://localhost:...`)
- API keys (even if referenced via `${ENV_VAR}`)
- Backend selections that differ by environment

**Example - Before (non-portable):**
```yaml
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434  # Environment-specific!

conversation_storage:
  backend: sqlite
  path: ~/.local/share/myapp/conversations.db  # Environment-specific!

knowledge_base:
  enabled: true
  vector_store:
    backend: faiss
    persist_path: ~/data/vectors  # Environment-specific!
    dimensions: 768
  embedding_provider: ollama
  embedding_model: nomic-embed-text
```

### Step 2: Create Environment Config Files

Create a directory structure for environment configs:

```
config/
├── environments/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
└── bots/
    └── my-bot.yaml
```

**Development environment (`config/environments/development.yaml`):**
```yaml
name: development
description: Local development environment

settings:
  log_level: DEBUG
  enable_metrics: false

resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434

  databases:
    conversations:
      backend: memory
    # Or for persistent local development:
    # conversations:
    #   backend: sqlite
    #   path: ~/.local/share/myapp/conversations.db

  vector_stores:
    knowledge:
      backend: memory
      dimensions: 768
    # Or for persistent local development:
    # knowledge:
    #   backend: faiss
    #   persist_path: ~/.local/share/myapp/vectors
    #   dimensions: 768

  embedding_providers:
    default:
      provider: ollama
      model: nomic-embed-text
      base_url: http://localhost:11434
```

**Production environment (`config/environments/production.yaml`):**
```yaml
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

  databases:
    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
      pool_size: 20

  vector_stores:
    knowledge:
      backend: pgvector
      connection_string: ${DATABASE_URL}
      dimensions: 1536

  embedding_providers:
    default:
      provider: openai
      model: text-embedding-3-small
      api_key: ${OPENAI_API_KEY}
```

### Step 3: Convert Bot Config to Use Resource References

Replace infrastructure settings with `$resource` references:

**After (portable):**
```yaml
# config/bots/my-bot.yaml
name: my-bot
version: "1.0.0"

bot:
  llm:
    $resource: default
    type: llm_providers
    # Behavioral settings (portable) can stay here
    temperature: 0.7
    max_tokens: 2000

  conversation_storage:
    $resource: conversations
    type: databases

  memory:
    type: buffer
    max_messages: 20

  knowledge_base:
    enabled: true
    vector_store:
      $resource: knowledge
      type: vector_stores
    embedding_provider:
      $resource: default
      type: embedding_providers
    # Behavioral settings (portable)
    retrieval:
      top_k: 5
      score_threshold: 0.6

  system_prompt: |
    You are a helpful assistant...

  reasoning:
    strategy: react
    max_iterations: 5
```

### Step 4: Update Your Code

**Before (direct configuration):**
```python
import yaml

with open("config/my-bot.yaml") as f:
    config = yaml.safe_load(f)

bot = await DynaBot.from_config(config)
```

**After (environment-aware):**
```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import BotResourceResolver

# Load environment (auto-detects from DATAKNOBS_ENVIRONMENT)
env = EnvironmentConfig.load()

# Create resolver
resolver = BotResourceResolver(env)

# Get initialized resources
llm = await resolver.get_llm("default")
db = await resolver.get_database("conversations")
vs = await resolver.get_vector_store("knowledge")
embedder = await resolver.get_embedding_provider("default")
```

Or use `EnvironmentAwareConfig` for full config resolution:

```python
from dataknobs_config import EnvironmentAwareConfig

# Load app config with environment bindings
config = EnvironmentAwareConfig.load_app(
    "my-bot",
    app_dir="config/bots",
    env_dir="config/environments"
)

# Resolve for building (late binding happens here)
resolved = config.resolve_for_build("bot")

# Create bot with resolved config
bot = await DynaBot.from_config(resolved)
```

### Step 5: Set Environment Variable

```bash
# Development (default if not set)
export DATAKNOBS_ENVIRONMENT=development

# Production
export DATAKNOBS_ENVIRONMENT=production

# Staging
export DATAKNOBS_ENVIRONMENT=staging
```

## Migration Patterns

### Pattern: Mixed Migration

You can migrate incrementally. Use `$resource` for some settings while keeping others inline:

```yaml
bot:
  # Migrated to resource reference
  llm:
    $resource: default
    type: llm_providers
    temperature: 0.7

  # Not yet migrated (still works, but not portable)
  conversation_storage:
    backend: memory

  # Migrated
  knowledge_base:
    vector_store:
      $resource: knowledge
      type: vector_stores
```

### Pattern: Default Values

Include default values in resource references that apply regardless of environment:

```yaml
llm:
  $resource: default
  type: llm_providers
  temperature: 0.7      # Always applied
  max_tokens: 2000      # Always applied
```

These values are merged with the environment's config, with your values taking precedence.

### Pattern: Multiple Resource Names

Define multiple resources for different use cases:

**In environment config:**
```yaml
resources:
  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}

    fast:
      provider: openai
      model: gpt-3.5-turbo
      api_key: ${OPENAI_API_KEY}

    reasoning:
      provider: openai
      model: gpt-4-turbo
      api_key: ${OPENAI_API_KEY}
```

**In bot config:**
```yaml
bot:
  # Use fast model for chat
  llm:
    $resource: fast
    type: llm_providers

  # Use reasoning model for complex tasks
  reasoning_llm:
    $resource: reasoning
    type: llm_providers
```

## Common Issues

### Issue: Resource Not Found

```
ResourceNotFoundError: Resource 'conversations' of type 'databases' not found in environment 'production'
```

**Solution:** Ensure the resource is defined in your environment config file:

```yaml
# config/environments/production.yaml
resources:
  databases:
    conversations:  # This name must match the $resource value
      backend: postgres
      connection_string: ${DATABASE_URL}
```

### Issue: Environment Not Detected

**Solution:** Set the environment variable explicitly:

```bash
export DATAKNOBS_ENVIRONMENT=production
```

Or specify it in code:

```python
env = EnvironmentConfig.load("production")
```

### Issue: Environment Variables Not Resolved

Environment variables are resolved at **instantiation time** (late binding), not at config load time. If you see `${VARIABLE}` in your resolved config:

1. Ensure the environment variable is set
2. Ensure you're calling `resolve_for_build()` or using `BotResourceResolver`

```python
# This resolves env vars
resolved = config.resolve_for_build()

# Or use BotResourceResolver which handles this automatically
resolver = BotResourceResolver(env)
llm = await resolver.get_llm("default")
```

## Backward Compatibility

The environment-aware system is **fully backward compatible**:

- `DynaBot.from_config()` works unchanged with direct configs
- Configs without `$resource` references work as before
- You can migrate incrementally, one resource at a time
- Existing code continues to work

## Next Steps

- See [Configuration Reference](configuration.md) for complete configuration reference
- See [API Reference](../api/reference.md) for API reference
