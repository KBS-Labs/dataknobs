# DynaBot Quickstart Guide

Welcome to DynaBot! This guide will help you get started quickly.

## Installation

To install DynaBot, use pip:

```bash
pip install dataknobs-bots
```

## Basic Usage

Here's a simple example to create your first bot:

```python
from dataknobs_bots import DynaBot, BotContext

# Configure the bot
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4"
    },
    "conversation_storage": {
        "backend": "memory"
    }
}

# Create bot
bot = await DynaBot.from_config(config)

# Create context
context = BotContext(
    conversation_id="conv-1",
    client_id="my-app"
)

# Chat
response = await bot.chat("Hello!", context)
print(response)
```

## Configuration

DynaBot is highly configurable. You can customize:

- LLM provider and model
- Memory type (buffer or vector)
- Knowledge base for RAG
- Tools and middleware

### Memory Configuration

For buffer memory:

```yaml
memory:
  type: buffer
  max_messages: 10
```

For vector memory:

```yaml
memory:
  type: vector
  backend: faiss
  dimension: 1536
```

## Next Steps

- Check out the [Configuration Guide](configuration.md)
- Learn about [Memory Systems](memory.md)
- Explore [RAG Integration](rag.md)
