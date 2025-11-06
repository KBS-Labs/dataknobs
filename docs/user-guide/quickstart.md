# Quick Start Guide

Get up and running with Dataknobs packages quickly.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

Choose packages based on your needs:

```bash
# For AI applications
pip install dataknobs-bots dataknobs-llm

# For data processing
pip install dataknobs-fsm dataknobs-data dataknobs-config

# For core utilities
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

## Quick Examples

### AI Chatbot

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Configure and create bot
    bot = await DynaBot.from_config({
        "llm": {"provider": "ollama", "model": "gemma3:3b"},
        "conversation_storage": {"backend": "memory"},
        "memory": {"type": "buffer", "max_messages": 10}
    })

    # Create context
    context = BotContext(
        conversation_id="chat-001",
        client_id="my-app",
        user_id="user1"
    )

    # Chat with context
    response = await bot.chat("What's the capital of France?", context)
    print(response)  # "Paris"

    response = await bot.chat("What's its population?", context)
    # Bot remembers we're talking about Paris

asyncio.run(main())
```

[Learn more →](../packages/bots/quickstart.md)

### Data Pipeline

```python
from dataknobs_fsm import SimpleFSM, DataHandlingMode

# Define workflow
pipeline = SimpleFSM({
    "name": "processor",
    "states": [
        {"name": "load", "is_start": True},
        {"name": "transform"},
        {"name": "save", "is_end": True}
    ],
    "arcs": [
        {"from": "load", "to": "transform"},
        {"from": "transform", "to": "save"}
    ]
}, data_mode=DataHandlingMode.COPY)

# Process data
result = pipeline.process({"input": "data"})
```

[Learn more →](../packages/fsm/quickstart.md)

### Unified Data Access

```python
from dataknobs_config import Config
from dataknobs_data import database_factory, Record, Query

# Load config (supports environment variables)
config = Config("config.yaml")
config.register_factory("database", database_factory)

# Get database (PostgreSQL, Elasticsearch, S3, etc.)
db = config.get_instance("databases", "primary")

# Use same API across all backends
record = Record({"name": "Alice", "role": "engineer"})
record_id = db.create(record)
results = db.search(Query().filter("role", "=", "engineer"))
```

[Learn more →](../packages/data/index.md) | [Config →](../packages/config/index.md)

### LLM Integration

```python
from dataknobs_llm import create_llm_provider, LLMMessage

# Create LLM client
llm = create_llm_provider({
    "provider": "openai",
    "model": "gpt-4"
})

# Multi-turn conversation
messages = [
    LLMMessage(role="user", content="What is Python?")
]
response = await llm.generate(messages)

messages.append(LLMMessage(role="assistant", content=response.content))
messages.append(LLMMessage(role="user", content="What's it used for?"))
response = await llm.generate(messages)  # Maintains context
```

[Learn more →](../packages/llm/quickstart.md)

### Data Structures

```python
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Create tree
tree = Tree("root")
tree.add_child("chapter1").add_child("section1.1")

# Navigate JSON
data = {"user": {"name": "Alice", "age": 30}}
name = json_utils.get_value(data, "user.name")  # "Alice"

# Normalize text
text = "  Hello   WORLD!!!  "
clean = normalize.basic_normalization_fn(text)  # "hello world!"
```

[Learn more →](../packages/structures/index.md) | [Utils →](../packages/utils/index.md)

## Next Steps

Choose your path based on what you're building:

**Building AI Applications:**
- [AI Chatbots](../packages/bots/quickstart.md) - Multi-tenant bots with memory
- [LLM Integration](../packages/llm/quickstart.md) - Custom LLM workflows
- [Bots Examples](../packages/bots/examples/index.md) - Real-world bot implementations

**Data Engineering:**
- [FSM Workflows](../packages/fsm/quickstart.md) - Robust data pipelines
- [Data Abstraction](../packages/data/index.md) - Multi-backend access
- [FSM Examples](../packages/fsm/examples/index.md) - ETL and processing examples

**General Development:**
- [Basic Usage](basic-usage.md) - Core structures and utilities
- [Advanced Usage](advanced-usage.md) - Advanced patterns
- [Examples](../examples/index.md) - Practical use cases
