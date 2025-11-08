---
title: llm (curated)
---

# LLM Provider API

Core LLM provider interface for interacting with language models.

## Overview

The LLM package provides a unified interface for working with different LLM providers (OpenAI, Anthropic, etc.). The factory pattern allows for easy instantiation and configuration.

## LLM Factory

### create_llm_provider

::: dataknobs_llm.llm.create_llm_provider
    options:
      show_source: true
      heading_level: 4

### LLMProviderFactory

::: dataknobs_llm.llm.LLMProviderFactory
    options:
      show_source: true
      heading_level: 4

## Base LLM Provider

::: dataknobs_llm.llm.LLMProvider
    options:
      show_source: true
      heading_level: 3
      members:
        - complete
        - acomplete
        - stream
        - astream

## Response Types

::: dataknobs_llm.llm.LLMResponse
    options:
      show_source: true
      heading_level: 3

::: dataknobs_llm.llm.LLMStreamResponse
    options:
      show_source: true
      heading_level: 3

## Provider Implementations

### OpenAI

::: dataknobs_llm.llm.OpenAIProvider
    options:
      show_source: false
      heading_level: 4

### Anthropic

::: dataknobs_llm.llm.AnthropicProvider
    options:
      show_source: false
      heading_level: 4

## Usage Examples

### Basic Completion

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# Create LLM provider
config = LLMConfig(
    provider="openai",
    api_key="your-key"
)
llm = create_llm_provider("openai", config)

# Asynchronous completion (recommended)
response = await llm.acomplete("What is Python?")
print(response.content)
print(f"Tokens: {response.usage.total_tokens}")
```

### Streaming

```python
# Asynchronous streaming (recommended)
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Configuration

```python
from dataknobs_llm import create_llm_provider, LLMConfig

# With custom configuration
config = LLMConfig(
    provider="openai",
    api_key="your-key",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
llm = create_llm_provider("openai", config)

# Or using factory
from dataknobs_llm import LLMProviderFactory

factory = LLMProviderFactory()
llm = factory.create("openai", api_key="your-key", model="gpt-4")
```

### Multiple Messages

```python
from dataknobs_llm import LLMMessage

messages = [
    LLMMessage(role="system", content="You are a helpful assistant."),
    LLMMessage(role="user", content="What is Python?"),
    LLMMessage(role="assistant", content="Python is a programming language."),
    LLMMessage(role="user", content="Tell me more.")
]

response = await llm.acomplete(messages)
```

## Error Handling

```python
try:
    response = await llm.acomplete("What is Python?")
except Exception as e:
    print(f"LLM error: {e}")
```

## See Also

- [Prompts API](prompts.md) - Template and builder interface
- [Conversations API](conversations.md) - Conversation management
- [Quick Start](../quickstart.md) - Getting started guide
