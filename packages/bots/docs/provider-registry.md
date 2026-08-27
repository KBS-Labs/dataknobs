# Provider Registry

DynaBot's provider registry is a central catalog of all LLM and embedding providers
used by a bot instance.  It enables comprehensive resource shutdown, provider
enumeration for observability, and clean test injection.

## The Problem

A DynaBot instance may use up to five separate LLM/embedding providers:

| Subsystem | Provider Type | Example |
|-----------|--------------|---------|
| Primary LLM | Chat/completion | Ollama llama3.2 |
| Extraction | Chat/completion | Ollama qwen3-coder |
| Vector memory | Embedding | Ollama nomic-embed-text |
| Summary memory | Chat/completion | Ollama llama3.2 (or dedicated) |
| Knowledge base | Embedding | Ollama nomic-embed-text |

Without the registry, each subsystem stores its provider in a private attribute.
Code that needs to operate on "all providers" — shutdown, cost tracking,
capture/replay, testing — must hard-code knowledge of each subsystem's internals.

The registry solves this by giving every provider a **role** and exposing them
through a uniform API.

## Provider Roles

Roles are string constants importable from `dataknobs_bots.bot.base`:

```python
from dataknobs_bots.bot.base import (
    PROVIDER_ROLE_MAIN,             # "main"
    PROVIDER_ROLE_EXTRACTION,       # "extraction"
    PROVIDER_ROLE_MEMORY_EMBEDDING, # "memory_embedding"
    PROVIDER_ROLE_SUMMARY_LLM,     # "summary_llm"
    PROVIDER_ROLE_KB_EMBEDDING,    # "kb_embedding"
)
```

The `"main"` role is reserved for `bot.llm` and is always present.  Other roles
are registered only when the corresponding subsystem is configured.

## API

### `register_provider(role, provider)`

Register a provider under a given role.  The `"main"` role is reserved and
cannot be overwritten (a warning is logged if attempted).

```python
from dataknobs_llm import EchoProvider

bot.register_provider("memory_embedding", embedding_provider)
```

### `get_provider(role) -> AsyncLLMProvider | None`

Retrieve a provider by role.  Returns `None` for unregistered roles.

```python
extraction = bot.get_provider("extraction")
if extraction is not None:
    print(f"Extraction model: {extraction.config.model}")
```

### `all_providers -> dict[str, AsyncLLMProvider]`

Property returning a snapshot dict of all registered providers keyed by role.
Always includes `"main"`.

```python
for role, provider in bot.all_providers.items():
    print(f"{role}: {provider.config.provider}/{provider.config.model}")
```

## Automatic Registration

When using `DynaBot.from_config()`, subsystem providers are automatically
discovered and registered.  No manual registration is needed:

```python
config = {
    "llm": {"provider": "ollama", "model": "llama3.2"},
    "conversation_storage": {"backend": "memory"},
    "memory": {
        "type": "vector",
        "backend": "memory",
        "dimension": 768,
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
    },
    "reasoning": {
        "strategy": "wizard",
        "wizard_config": "wizards/onboarding.yaml",
        "extraction_config": {
            "provider": "ollama",
            "model": "qwen3-coder",
        },
    },
}

bot = await DynaBot.from_config(config)

# Providers are already registered:
print(bot.all_providers.keys())
# dict_keys(['main', 'memory_embedding', 'extraction'])
```

Detection rules for `_build_from_config()`:

Providers are registered by calling ``subsystem.providers()`` on each configured
subsystem (memory, knowledge_base, reasoning_strategy).  Each subsystem returns
a dict of role -> provider for the providers it owns.

- **memory_embedding**: Returned by ``VectorMemory.providers()``.
- **summary_llm**: Returned by ``SummaryMemory.providers()`` when a dedicated provider was created.
- **kb_embedding**: Returned by ``RAGKnowledgeBase.providers()``.
- **extraction**: Returned by ``WizardReasoning.providers()`` when extraction is configured.

## Resource Management

`bot.close()` closes each subsystem (memory, knowledge base, reasoning
strategy) and the main LLM provider.  Each subsystem closes the providers
it created (originator-owns-lifecycle).  The registry catalog is for
observability — it does not manage provider lifecycle.

```python
async with bot:
    response = await bot.chat("Hello", context)
# Subsystems + main provider closed via __aexit__ → close()
```

`AsyncLLMProvider.close()` should be idempotent so that overlapping
close calls from subsystem cleanup are safe.  Note that ``EchoProvider``
intentionally counts every ``close()`` call (via ``close_count``) for
test assertions, so test code should be precise about close expectations.

## Testing

### inject_providers()

The `inject_providers()` helper uses the registry for clean provider
replacement during tests:

```python
from dataknobs_llm import EchoProvider
from dataknobs_bots.testing import inject_providers

# Replace main and extraction providers
main_echo = EchoProvider({"provider": "echo", "model": "test-main"})
ext_echo = EchoProvider({"provider": "echo", "model": "test-extract"})
inject_providers(bot, main_provider=main_echo, extraction_provider=ext_echo)

# Replace any role-based provider via **kwargs
mem_echo = EchoProvider({"provider": "echo", "model": "test-embed"})
inject_providers(bot, memory_embedding=mem_echo)
```

### Direct registry access in tests

```python
# Verify a subsystem provider was registered
assert bot.get_provider("extraction") is not None

# Verify total provider count
assert len(bot.all_providers) == 3  # main + extraction + memory_embedding

# Enumerate for assertions
for role, provider in bot.all_providers.items():
    assert hasattr(provider, "complete") or hasattr(provider, "embed")
```

## Custom Roles

Consumers can register providers under custom roles for application-specific
subsystems:

```python
# Register a custom translation provider
bot.register_provider("translation", translation_provider)

# Later retrieve it
translator = bot.get_provider("translation")
```

Custom roles appear in `all_providers` for observability and enumeration.
The registry does not manage their lifecycle — the originator is
responsible for closing them.
