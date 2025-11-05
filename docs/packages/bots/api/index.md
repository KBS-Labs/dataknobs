# API Reference

Complete API reference for dataknobs-bots.

## Overview

The dataknobs-bots API is organized into several modules:

- **[Bot Classes](reference.md#dynabot-class)** - Core DynaBot and BotContext classes
- **[Bot Registry](reference.md#botregistry-class)** - Multi-tenant bot management
- **[Memory Systems](reference.md#memory-systems)** - Memory implementations for context management
- **[Knowledge Base](reference.md#ragknowledgebase-class)** - RAG implementation for document retrieval
- **[Reasoning Strategies](reference.md#reasoning-strategies)** - ReAct and simple reasoning
- **[Tools](reference.md#tools)** - Tool interface and built-in tools

## Quick Links

### Core Classes

- [`DynaBot`](reference.md#dynabot-class) - Main bot orchestrator
- [`BotContext`](reference.md#botcontext-dataclass) - Execution context
- [`BotRegistry`](reference.md#botregistry-class) - Multi-tenant registry

### Memory

- [`Memory`](reference.md#memory-base-class) - Memory interface
- [`BufferMemory`](reference.md#buffermemory-class) - Sliding window memory
- [`VectorMemory`](reference.md#vectormemory-class) - Semantic search memory

### Knowledge Base

- [`RAGKnowledgeBase`](reference.md#ragknowledgebase-class) - RAG implementation

### Reasoning

- [`ReasoningStrategy`](reference.md#reasoningstrategy-base-class) - Reasoning interface
- [`SimpleReasoning`](reference.md#simplereasoning-class) - Direct LLM response
- [`ReActReasoning`](reference.md#reactreasoning-class) - Reasoning + Acting pattern

### Tools

- [`Tool`](reference.md#tool-base-class) - Tool interface
- [`KnowledgeSearchTool`](reference.md#knowledgesearchtool) - Built-in knowledge search

### Factory Functions

- [`create_memory()`](reference.md#create_memory) - Create memory from configuration
- [`create_knowledge_base()`](reference.md#create_knowledge_base) - Create knowledge base from configuration
- [`create_reasoning_strategy()`](reference.md#create_reasoning_strategy) - Create reasoning strategy from configuration

## Complete Reference

See the [Complete API Reference](reference.md) for detailed documentation of all classes, methods, and functions.

## Usage Examples

### Creating a Bot

```python
from dataknobs_bots import DynaBot, BotContext

config = {
    "llm": {"provider": "ollama", "model": "gemma3:3b"},
    "conversation_storage": {"backend": "memory"}
}

bot = await DynaBot.from_config(config)
context = BotContext(
    conversation_id="conv-001",
    client_id="demo-client"
)

response = await bot.chat("Hello!", context)
```

### Using Bot Registry

```python
from dataknobs_bots import BotRegistry

registry = BotRegistry(config=base_config)

await registry.register_client("client-a", client_config)
bot = await registry.get_bot("client-a")
```

### Creating Memory

```python
from dataknobs_bots.memory import create_memory

memory = await create_memory({
    "type": "buffer",
    "max_messages": 10
})

await memory.add_message("Hello", "user")
context = await memory.get_context("How are you?")
```

### Using Knowledge Base

```python
from dataknobs_bots.knowledge import create_knowledge_base

kb = await create_knowledge_base({
    "enabled": True,
    "documents_path": "./docs",
    "vector_store": {"backend": "faiss", "dimension": 384}
})

results = await kb.query("What is the product?", k=5)
```

## Type Hints

All classes and functions in dataknobs-bots are fully typed with Python type hints. Import types from the package:

```python
from dataknobs_bots import DynaBot, BotContext
from dataknobs_bots.memory import Memory, BufferMemory
from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_llm.tools import Tool
```

## Error Handling

Errors are raised as standard Python exceptions:

```python
try:
    bot = await DynaBot.from_config(config)
    response = await bot.chat(message, context)
except ValueError as e:
    # Configuration or validation error
    print(f"Configuration error: {e}")
except RuntimeError as e:
    # Runtime error (LLM, storage, etc.)
    print(f"Runtime error: {e}")
```

## Async/Await

All main operations in dataknobs-bots are asynchronous:

```python
import asyncio

async def main():
    bot = await DynaBot.from_config(config)
    response = await bot.chat(message, context)

asyncio.run(main())
```

## Related Documentation

- [User Guide](../guides/user-guide.md) - Tutorials and how-to guides
- [Configuration Reference](../guides/configuration.md) - Configuration options
- [Examples](../examples/simple-chatbot.md) - Working examples
