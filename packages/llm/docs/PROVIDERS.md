# Provider Creation

Factory functions for creating and initializing LLM providers from configuration.

## Overview

The `dataknobs-llm` package provides two factory functions for creating providers:

| Function | Purpose |
|----------|---------|
| `create_llm_provider()` | Create a chat/completion provider |
| `create_embedding_provider()` | Create an embedding provider (initialized, mode forced) |

Both use `LLMProviderFactory` internally and support all registered provider
backends (Ollama, OpenAI, Anthropic, HuggingFace, Echo).

## create_llm_provider()

Create a chat/completion provider from configuration.

```python
from dataknobs_llm import create_llm_provider

provider = create_llm_provider({
    "provider": "ollama",
    "model": "llama3.2",
})

# Use the provider
response = await provider.complete(messages)
```

Accepts `LLMConfig`, `Config`, or `dict`. Returns an uninitialized provider —
call `await provider.initialize()` before use.

## create_embedding_provider()

Create and initialize an embedding provider from configuration. The provider is
returned ready for `embed()` calls with `CompletionMode.EMBEDDING` forced.

### Signature

```python
async def create_embedding_provider(
    config: dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Configuration dict (see formats below) |
| `default_provider` | `str` | `"ollama"` | Provider when not specified in config |
| `default_model` | `str` | `"nomic-embed-text"` | Model when not specified in config |

### Returns

Initialized `AsyncLLMProvider` with `CompletionMode.EMBEDDING` set.

### Configuration Formats

Two configuration formats are supported. The nested format is preferred.

**Nested format** (preferred):

```python
provider = await create_embedding_provider({
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimensions": 768,
        "api_base": "http://localhost:11434",
    },
})
```

All keys in the `embedding` sub-dict other than `provider` and `model` are
forwarded to the provider (e.g., `api_base`, `api_key`, `dimensions`).

**Legacy prefix format:**

```python
provider = await create_embedding_provider({
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "dimensions": 768,
    "api_base": "http://localhost:11434",
    "api_key": "...",
})
```

Only `api_base`, `api_key`, and `dimensions` are forwarded from the top level.
Other top-level keys (e.g., `backend`, `type`) are ignored.

When the nested format is present, it takes precedence over legacy keys.

### Embedding Mode

`CompletionMode.EMBEDDING` is always forced on the created provider, even if
the caller's config includes `"mode": "chat"`. This ensures the provider is
correctly configured for `embed()` calls.

### Examples

```python
from dataknobs_llm import create_embedding_provider

# Ollama with nomic-embed-text (default)
provider = await create_embedding_provider({
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
    },
})
embedding = await provider.embed("hello world")
await provider.close()

# OpenAI embeddings
provider = await create_embedding_provider({
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "sk-...",
        "dimensions": 1536,
    },
})

# Custom defaults (e.g., for testing)
provider = await create_embedding_provider(
    {},
    default_provider="echo",
    default_model="test-embed",
)
```

### Backward Compatibility

`create_embedding_provider()` is also available from `dataknobs_bots.providers`
for backward compatibility. The canonical import path is `dataknobs_llm`:

```python
# Preferred
from dataknobs_llm import create_embedding_provider

# Also works (backward compat)
from dataknobs_bots.providers import create_embedding_provider
```

## Provider Backends

Both factory functions support all registered providers:

| Provider | Key | Package |
|----------|-----|---------|
| Ollama | `"ollama"` | Built-in |
| OpenAI | `"openai"` | Built-in |
| Anthropic | `"anthropic"` | Built-in |
| HuggingFace | `"huggingface"` | Built-in |
| Echo | `"echo"` | Built-in (testing) |

## Testing

Use `EchoProvider` (via `"echo"` provider key) for tests:

```python
provider = await create_embedding_provider({
    "embedding": {"provider": "echo", "model": "test"},
})
embedding = await provider.embed("test input")
assert len(embedding) > 0
await provider.close()
```
