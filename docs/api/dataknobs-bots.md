# dataknobs-bots API Reference

## Overview

The `dataknobs-bots` package provides a flexible framework for building AI chatbots with support for memory, RAG knowledge bases, reasoning strategies, and tool integration.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/bots.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/bots/src/dataknobs_bots) - Browse on GitHub
> - [Package Guide](../packages/bots/index.md) - Detailed documentation

## Main Classes

### DynaBot

**Source:** [`bot.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/bot.py)

The main bot orchestrator that coordinates conversations, memory, knowledge retrieval, and reasoning.

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

#### Key Methods

- `from_config(config)` - Create bot from configuration dictionary
- `chat(message, context, temperature, max_tokens, stream)` - Send a message and get response
- `get_conversation(conversation_id)` - Retrieve conversation history (returns ConversationState or None)
- `clear_conversation(conversation_id)` - Clear a conversation (returns True if deleted, False if not found)

### BotContext

**Source:** [`bot.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/bot.py)

Execution context for bot interactions containing conversation and client identifiers.

```python
from dataknobs_bots import BotContext

context = BotContext(
    conversation_id="conv-123",
    client_id="client-abc",
    user_id="user-456",  # Optional user identifier
    session_metadata={"session": "web-001"},
    request_metadata={"request_id": "req-789"}
)
```

### BotRegistry

**Source:** [`registry.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/registry.py)

Multi-tenant bot registry for managing multiple clients with shared or isolated configurations.

```python
from dataknobs_bots import BotRegistry

registry = BotRegistry(config=base_config)

await registry.register_client("client-a", client_config)
bot = await registry.get_bot("client-a")
```

#### Key Methods

- `register_client(client_id, config)` - Register a new client with specific configuration
- `get_bot(client_id)` - Get bot instance for a client
- `remove_client(client_id)` - Remove a client and cleanup resources

## Memory System

### BufferMemory

**Source:** [`memory/buffer.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/memory/buffer.py)

Sliding window memory that maintains recent conversation history.

```python
from dataknobs_bots.memory import create_memory

memory = await create_memory({
    "type": "buffer",
    "max_messages": 10
})

await memory.add_message("Hello", "user")
context = await memory.get_context("How are you?")
```

### VectorMemory

**Source:** [`memory/vector.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/memory/vector.py)

Semantic search-based memory using vector similarity.

```python
memory = await create_memory({
    "type": "vector",
    "vector_store": {"backend": "faiss", "dimension": 384},
    "top_k": 5
})
```

## Knowledge Base

### RAGKnowledgeBase

**Source:** [`knowledge/rag.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/knowledge/rag.py)

RAG (Retrieval-Augmented Generation) implementation for document-based question answering.

```python
from dataknobs_bots.knowledge import create_knowledge_base

kb = await create_knowledge_base({
    "enabled": True,
    "documents_path": "./docs",
    "vector_store": {"backend": "faiss", "dimension": 384}
})

results = await kb.query("What is the product?", k=5)
```

## Reasoning Strategies

### SimpleReasoning

**Source:** [`reasoning/simple.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/reasoning/simple.py)

Direct LLM response without additional reasoning steps.

```python
config = {
    "reasoning": {"type": "simple"}
}
```

### ReActReasoning

**Source:** [`reasoning/react.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/bots/src/dataknobs_bots/reasoning/react.py)

ReAct pattern (Reasoning + Acting) for tool-using agents.

```python
from dataknobs_llm.tools import Tool

def search_tool(query: str) -> str:
    return f"Results for: {query}"

config = {
    "reasoning": {
        "type": "react",
        "tools": [
            Tool(name="search", func=search_tool, description="Search the web")
        ]
    }
}
```

## Full Example

```python
from dataknobs_bots import DynaBot, BotContext
from dataknobs_llm.tools import Tool

# Define custom tools
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

def get_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

# Create bot with memory, knowledge base, and tools
config = {
    "llm": {
        "provider": "ollama",
        "model": "gemma3:3b"
    },
    "conversation_storage": {
        "backend": "memory"
    },
    "memory": {
        "type": "buffer",
        "max_messages": 10
    },
    "knowledge": {
        "enabled": True,
        "documents_path": "./docs",
        "vector_store": {"backend": "faiss"}
    },
    "reasoning": {
        "type": "react",
        "tools": [
            Tool(name="weather", func=get_weather, description="Get weather for location"),
            Tool(name="time", func=get_time, description="Get current time")
        ]
    }
}

# Create bot and chat
bot = await DynaBot.from_config(config)
context = BotContext(conversation_id="conv-001", client_id="user-123")

# Chat with memory and tool access
response = await bot.chat("What's the weather in Seattle?", context)
print(response)

# Follow-up using conversation memory
response = await bot.chat("What about the time?", context)
print(response)

# Retrieve conversation history
conversation_state = await bot.get_conversation("conv-001")
if conversation_state:
    print(f"Conversation has {len(conversation_state.message_tree)} messages")
    print(f"Metadata: {conversation_state.metadata}")

# Clear conversation when done
deleted = await bot.clear_conversation("conv-001")
print(f"Conversation cleared: {deleted}")
```

## Conversation Management

### Retrieving Conversations

```python
# Get conversation history
conversation_state = await bot.get_conversation("conv-123")

if conversation_state:
    # Access the full message tree
    messages = conversation_state.message_tree

    # Check conversation metadata
    print(f"Conversation ID: {conversation_state.conversation_id}")
    print(f"Client ID: {conversation_state.metadata.get('client_id')}")
    print(f"User ID: {conversation_state.metadata.get('user_id')}")
else:
    print("Conversation not found")
```

### Clearing Conversations

```python
# Clear a conversation (for privacy, reset, or cleanup)
deleted = await bot.clear_conversation("conv-123")

if deleted:
    print("Conversation successfully deleted")

    # Next chat with same ID will start fresh
    context = BotContext(conversation_id="conv-123", client_id="user-456")
    response = await bot.chat("Starting over!", context)
else:
    print("Conversation not found (may have already been deleted)")
```

## Usage Examples

For detailed usage examples, see:
- [Simple Chatbot](../packages/bots/examples/simple-chatbot.md)
- [Memory Chatbot](../packages/bots/examples/memory-chatbot.md)
- [RAG Chatbot](../packages/bots/examples/rag-chatbot.md)
- [ReAct Agent](../packages/bots/examples/react-agent.md)
- [Multi-tenant Bots](../packages/bots/examples/multi-tenant.md)
