# Conversations API

Tree-structured conversation management with branching and persistence.

> **📖 Also see:** [Auto-generated API Reference](../../../api/reference/llm.md) - Complete documentation from source code docstrings

---

## Overview

The conversations API provides a powerful system for managing multi-turn conversations with support for branching, time-travel, persistence, and FSM-based flows.

## ConversationManager

::: dataknobs_llm.conversations.ConversationManager
    options:
      show_source: true
      heading_level: 3
      members:
        - create
        - add_message
        - complete
        - switch_to_node
        - branch_from
        - get_tree_structure
        - get_rag_metadata
        - save
        - load

## Conversation Types

### ConversationNode

::: dataknobs_llm.conversations.ConversationNode
    options:
      show_source: true
      heading_level: 3

### ConversationState

::: dataknobs_llm.conversations.ConversationState
    options:
      show_source: true
      heading_level: 3

### LLMMessage

::: dataknobs_llm.llm.LLMMessage
    options:
      show_source: true
      heading_level: 3

## Storage

### Abstract Storage

::: dataknobs_llm.conversations.ConversationStorage
    options:
      show_source: true
      heading_level: 3
      members:
        - save_conversation
        - load_conversation
        - delete_conversation
        - list_conversations

### Implementations

#### DataknobsConversationStorage

::: dataknobs_llm.conversations.DataknobsConversationStorage
    options:
      show_source: true
      heading_level: 4

## Middleware

### Abstract Middleware

::: dataknobs_llm.conversations.ConversationMiddleware
    options:
      show_source: true
      heading_level: 3
      members:
        - before_add_message
        - after_add_message
        - before_complete
        - after_complete

### Built-in Middleware

#### LoggingMiddleware

::: dataknobs_llm.conversations.LoggingMiddleware
    options:
      show_source: true
      heading_level: 4

## Usage Examples

### Basic Conversation

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends import AsyncMemoryDatabase
from pathlib import Path

# Setup
config = LLMConfig(provider="openai", api_key="your-key")
llm = create_llm_provider(config)
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))
builder = AsyncPromptBuilder(library=library)

# Create storage
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Create conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

# Add user message
await manager.add_message(
    role="user",
    prompt_name="greeting",
    params={"name": "Alice"}
)

# Get assistant response
response = await manager.complete()
print(response.content)

# Continue conversation
await manager.add_message(
    role="user",
    content="Tell me about Python decorators"
)
response = await manager.complete()
```

### Branching Conversations

```python
# Create conversation
manager = await ConversationManager.create(llm=llm, prompt_builder=builder)

# Initial exchange
await manager.add_message(role="user", content="Help me design an API")
response1 = await manager.complete()

# Save checkpoint
checkpoint_node = manager.current_node

# Try REST approach
await manager.add_message(role="user", content="Use REST principles")
rest_response = await manager.complete(branch_name="rest_approach")

# Go back and try GraphQL
await manager.switch_to_node(checkpoint_node)
await manager.add_message(role="user", content="Use GraphQL instead")
graphql_response = await manager.complete(branch_name="graphql_approach")

# View conversation tree
tree = await manager.get_tree_structure()
print(tree)
```

### Persistence

```python
from dataknobs_llm.conversations import DataknobsConversationStorage
from dataknobs_data.backends import AsyncMemoryDatabase

# Create with persistence
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    conversation_id="user123-session1"
)

# Have conversation...
await manager.add_message(role="user", content="Hello")
await manager.complete()

# Automatically persisted on each operation

# Later, restore the same conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    conversation_id="user123-session1"  # Loads existing conversation
)

# Continue from where you left off
await manager.add_message(role="user", content="Continue...")
```

### RAG Caching

```python
# Enable RAG caching
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    cache_rag_results=True,      # Store RAG metadata in conversation
    reuse_rag_on_branch=True     # Reuse RAG when branching
)

# Add message with RAG-enabled prompt
await manager.add_message(
    role="user",
    prompt_name="code_question",  # Has RAG configured
    params={"language": "python"}
)
# RAG search executed, metadata cached

# Get LLM response
response = await manager.complete()

# Inspect RAG metadata
rag_info = await manager.get_rag_metadata()
print(f"Query: {rag_info['RAG_DOCS']['query']}")
print(f"Results: {len(rag_info['RAG_DOCS']['results'])}")

# Branch conversation - RAG is reused from cache
await manager.switch_to_node("0")
await manager.complete(branch_name="alternative")
# No RAG search needed, uses cached results
```

### Middleware

```python
from dataknobs_llm.conversations import LoggingMiddleware, ConversationMiddleware

# Custom middleware
class TokenCounterMiddleware(ConversationMiddleware):
    def __init__(self):
        self.total_tokens = 0

    async def after_complete(self, manager, response):
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return response

# Use middleware
token_counter = TokenCounterMiddleware()
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    middlewares=[
        LoggingMiddleware(),
        token_counter
    ]
)

# Have conversation...
await manager.add_message(role="user", content="Hello")
await manager.complete()

print(f"Total tokens used: {token_counter.total_tokens}")
```

### Multi-Turn with System Prompts

```python
# Create with system prompt
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    system_prompt_name="code_assistant",
    system_prompt_params={"language": "python"}
)

# All completions will use this system prompt
await manager.add_message(role="user", content="Review this code: def foo(): pass")
response = await manager.complete()

await manager.add_message(role="user", content="How can I improve it?")
response = await manager.complete()
```

### Navigation

```python
# Get conversation history
history = manager.get_history()
for msg in history:
    print(f"{msg.role}: {msg.content}")

# Get all nodes
nodes = await manager.list_nodes()
for node in nodes:
    print(f"Node {node.node_id}: {node.message.role if node.message else 'root'}")

# Navigate to specific node
await manager.switch_to_node("node-123")

# Branch from a node (navigate to its parent for sibling creation)
await manager.branch_from("node-123")
# Next add_message() or complete() creates a sibling of "node-123"

# Get parent node
parent = manager.get_parent_node()

# Get children nodes
children = manager.get_children_nodes()
```

## Advanced Features

### Tree Structure Analysis

```python
# Get tree visualization
tree = await manager.get_tree_structure()
print(tree)

# Output example:
# root (system)
# ├── node-1 (user): "Help me design..."
# │   ├── node-2 (assistant): "Here's a REST approach..."
# │   │   └── node-3 (user): "Use REST principles"
# │   │       └── node-4 (assistant): [rest_approach]
# │   └── node-5 (user): "Use GraphQL instead"
# │       └── node-6 (assistant): [graphql_approach]
```

### Conversation Metadata

```python
# Add metadata to messages
await manager.add_message(
    role="user",
    content="Hello",
    metadata={"source": "web", "user_id": "123"}
)

# Access metadata
current_node = manager.get_current_node()
print(current_node.metadata)
```

### Streaming in Conversations

```python
# Stream assistant response
await manager.add_message(role="user", content="Tell me a story")

# Stream chunks
async for chunk in manager.stream_complete():
    print(chunk.content, end="", flush=True)
```

## See Also

- [Conversation Management Guide](../guides/conversations.md) - Detailed guide
- [FSM-Based Flows](../guides/flows.md) - Workflow orchestration
- [Performance Guide](../guides/performance.md) - RAG caching details
- [Examples](../examples/conversation-flows.md) - Flow examples
