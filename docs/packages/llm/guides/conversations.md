# Conversation Management Guide

Tree-structured conversation history with branching, time-travel, and FSM integration.

## Overview

The conversation system provides:

- **Tree Structure**: Branch conversations to explore alternatives
- **Time Travel**: Navigate back through conversation history
- **Persistence**: Save and restore conversation state
- **RAG Caching**: Conversation-level RAG metadata caching
- **Middleware**: Pluggable middleware for cross-cutting concerns
- **FSM Integration**: State machine-based conversation flows

## Quick Start

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Create LLM provider
config = LLMConfig(provider="openai", api_key="key")
llm = create_llm_provider(config)

# Create storage
db = AsyncMemoryDatabase()
storage = DataknobsConversationStorage(db)

# Create conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

# Add messages
await manager.add_message(
    role="user",
    prompt_name="greeting",
    params={"name": "Alice"}
)

# Get LLM response
response = await manager.complete()

# Branch conversation
await manager.switch_to_node("node-id")
await manager.complete(branch_name="alternative")
```

## Key Features

### Tree Structure

```python
# View conversation tree
tree = await manager.get_tree_structure()
print(tree)

# Navigate to specific node
await manager.switch_to_node("earlier-node-id")

# Continue from that point
response = await manager.complete(branch_name="new_path")

# Branch from an existing node (create a sibling)
await manager.branch_from("node-id")  # Navigates to parent
response_alt = await manager.complete()  # New sibling node
```

### RAG Caching

```python
# Enable RAG caching
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    cache_rag_results=True,      # Store RAG metadata
    reuse_rag_on_branch=True     # Reuse when branching
)

# Inspect RAG metadata
rag_info = await manager.get_rag_metadata()
```

For complete RAG caching documentation, see:
**Location**: `packages/llm/docs/RAG_CACHING.md`

### Middleware System

```python
from dataknobs_llm.conversations import LoggingMiddleware, ValidationMiddleware

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    middlewares=[
        LoggingMiddleware(),
        ValidationMiddleware(rules=my_rules)
    ]
)
```

## Detailed Documentation

For comprehensive conversation management documentation:

**Location**: `packages/llm/docs/USER_GUIDE.md` (Conversation section)

Topics covered:
- Tree-based conversation structure
- Branching and navigation
- Persistence and restoration
- RAG caching configuration
- Middleware development
- Best practices

## Common Patterns

### Multi-Turn Conversation

```python
# Initialize
manager = await ConversationManager.create(llm=llm, prompt_builder=builder)

# Multiple exchanges
for user_input in user_inputs:
    await manager.add_message(
        role="user",
        content=user_input
    )
    response = await manager.complete()
    print(f"Assistant: {response.content}")
```

### Exploring Alternatives

```python
# Save checkpoint
checkpoint_node = manager.current_node

# Try option A
await manager.add_message(role="user", content="Let's try approach A")
response_a = await manager.complete(branch_name="approach_a")

# Go back and try option B
await manager.switch_to_node(checkpoint_node)
await manager.add_message(role="user", content="Let's try approach B")
response_b = await manager.complete(branch_name="approach_b")

# Compare results
```

### Branching from a Node

Use `branch_from()` to create a sibling of a specific node without manually
locating its parent:

```python
# response_a is at node "0.0" — branch from the same parent
await manager.branch_from("0.0")  # Positions at node "0"
response_b = await manager.complete()  # Creates node "0.1"
```

This is used internally by the wizard reasoning strategy when a stage is
revisited via back or restart navigation, creating sibling branches in the
conversation tree rather than chaining deeper.

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

# Later, restore
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    conversation_id="user123-session1"  # Restores state
)
```

## See Also

- [FSM-Based Flows](flows.md) - Workflow orchestration
- [Prompt Engineering](prompts.md) - Building prompts
- [Performance](performance.md) - RAG caching and optimization
- [API Reference](../api/conversations.md) - Full API documentation
