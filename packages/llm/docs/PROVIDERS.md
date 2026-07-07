# Provider Creation

Factory functions for creating and initializing LLM providers from configuration.

## Overview

The `dataknobs-llm` package provides two factory functions for creating providers:

| Function | Purpose |
|----------|---------|
| `create_llm_provider()` | Create a chat/completion provider |
| `create_embedding_provider()` | Create an embedding provider (initialized, mode forced) |

Both use `LLMProviderFactory` internally and support all registered provider
backends (Ollama, OpenAI, Anthropic, Amazon Bedrock, HuggingFace, Echo).

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
    config: LLMConfig | dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "nomic-embed-text",
) -> AsyncLLMProvider:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `LLMConfig \| dict` | — | A typed `LLMConfig` or a configuration dict (see formats below) |
| `default_provider` | `str` | `"ollama"` | Provider when not specified (dict path only) |
| `default_model` | `str` | `"nomic-embed-text"` | Model when not specified (dict path only) |

### Returns

Initialized `AsyncLLMProvider` with `CompletionMode.EMBEDDING` set.

### Configuration Formats

A typed `LLMConfig` or one of two dict formats is supported. An embedder
config **is** an `LLMConfig` — embedding providers ride the same provider
registry — so no separate config type is needed.

**Typed `LLMConfig`:**

```python
from dataknobs_llm import LLMConfig

provider = await create_embedding_provider(
    LLMConfig(provider="ollama", model="nomic-embed-text", dimensions=768)
)
```

`provider` / `model` are validated as required fields, and `mode` is forced to
`CompletionMode.EMBEDDING` (via `clone()` — `LLMConfig` is frozen, so the
caller's config is never mutated). `default_provider` / `default_model` are
unused on this path.

The two dict formats follow. The nested format is preferred.

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

### Config-lint validation

Because an embedder config is an `LLMConfig`, `dataknobs-llm` registers an
`"embedding"` resolver in `config_registries` (eager on import) that resolves
an `embedding` section to `LLMConfig` — the same resolver as the `"llm"`
binding. A consumer config that holds a nested `embedding` section
(currently `RAGKnowledgeBaseConfig` and `VectorMemoryConfig`) declares
`{"embedding": "embedding"}` in `_polymorphic_fields`, so
`config.validate()` dry-run-builds the embedder `LLMConfig` and surfaces an
unknown provider or bad field at config-parse time — without constructing a
provider. See the Structured Config guide in `dataknobs-common` for the
`validate()` / `_polymorphic_fields` mechanism.

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
| Amazon Bedrock | `"bedrock"` | Built-in (needs `[bedrock]` extra) |
| HuggingFace | `"huggingface"` | Built-in |
| Echo | `"echo"` | Built-in (testing) |

## Amazon Bedrock

Amazon Bedrock is registered as the `"bedrock"` provider. A single
`BedrockProvider` serves **both** chat/completion (via the unified Converse
API) and embeddings (Titan / Cohere via `invoke_model`).

**Authentication is via the AWS credential chain — there is no API key.**
Credentials resolve from the environment, the `~/.aws` shared config, or an
EC2/ECS instance or task IAM role. Region, endpoint, explicit credentials, and
Bedrock guardrail settings are supplied through `LLMConfig.options`.

Install the async Bedrock transport (`aioboto3`, pulled via the shared
`dataknobs-common[aws]` session factory):

```bash
pip install 'dataknobs-llm[bedrock]'
```

### Chat / completion

```python
from dataknobs_llm import create_llm_provider

provider = create_llm_provider({
    "provider": "bedrock",
    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "temperature": 0.7,
    "max_tokens": 1024,
    "options": {"region_name": "us-west-2"},  # credentials via the IAM chain
})
await provider.initialize()
response = await provider.complete("Explain quantum computing")
```

The model id is a Bedrock foundation-model id
(`anthropic.claude-3-5-sonnet-20240620-v1:0`) or a cross-region
inference-profile id (`us.anthropic.claude-3-5-sonnet-20240620-v1:0`).
Streaming (`stream_complete`) and tool use (`complete(tools=...)`) work as with
the other providers.

### `options` keys

| Key | Purpose |
|-----|---------|
| `region_name` (or `region`) | AWS region for the client |
| `endpoint_url` | Custom endpoint (PrivateLink / VPC endpoint). Bedrock's endpoint knob — `LLMConfig.api_base` is not consulted |
| `aws_access_key_id` / `aws_secret_access_key` / `aws_session_token` | Explicit credentials (omit to use the credential chain). A partial pair fails closed at construction |
| `normalize` | Titan embeddings only — L2-normalize the vector (default `True`) |
| `input_type` | Cohere embeddings only — `"search_document"` (default) or `"search_query"` at query time |
| `embed_max_concurrency` | Bound on Titan's per-text `invoke_model` fan-out (default: `max_pool_connections`, i.e. `10`) |
| `stream_read_timeout` | Per-socket-read (inter-chunk) timeout for `stream_complete`, in seconds (default: boto's `60`s). See the timeout note below |
| `guardrail_identifier` + `guardrail_version` | Applied to Converse requests when both are set (optional `guardrail_trace`) |

The `complete()` / `function_call()` socket read timeout is `LLMConfig.timeout`
(default `60`s); retry and connection-pool tuning follow the shared
`AwsSessionConfig` defaults. **Streaming is different:** botocore's
`read_timeout` is a per-read (inter-chunk) timeout, and there is no
total-stream-duration knob, so `LLMConfig.timeout` is *not* applied to
`stream_complete` — otherwise a long inter-token pause would kill the stream.
Streaming uses `stream_read_timeout` instead (default: boto's `60`s); raise it
for slow-thinking models.

### Embeddings

```python
from dataknobs_llm import create_embedding_provider

provider = await create_embedding_provider({
    "embedding": {
        "provider": "bedrock",
        "model": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "options": {"region_name": "us-west-2"},
    },
})
vector = await provider.embed("hello world")
```

Two embedding families are supported:

| Family | Model ids | Notes |
|--------|-----------|-------|
| Amazon Titan | `amazon.titan-embed-text-v2:0` | `dimensions` selects 256 / 512 / 1024 (default 1024); embeds one text per call, bounded by `embed_max_concurrency`; `normalize` via options |
| Cohere | `cohere.embed-english-v3`, `cohere.embed-multilingual-v3` | embeds the whole list in one call; `input_type` via options (`search_query` at query time) |

An unrecognized embedding-model id raises `ValueError` naming the two supported
families.

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
