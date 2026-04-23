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
    middleware=[
        LoggingMiddleware(),
        ValidationMiddleware(rules=my_rules)
    ]
)
```

#### Per-Turn (Scoped) Middleware

For middleware whose internal state must be fresh for each turn (e.g. a
post-processor that holds that turn's retrieval candidates), attach it
via `ConversationManager.scoped_middleware()` rather than the permanent
`middleware=[...]` argument. The context manager appends on entry and
removes on exit — including on exception — preserving onion ordering
relative to the permanent stack.

```python
async with manager.scoped_middleware(citation_mw):
    response = await manager.complete()
# citation_mw is removed here, even if complete() raised.
```

Response mutations performed inside the scope flow to the persisted
assistant node, because `process_response` runs before the tree snapshot
in `_finalize_completion`. Not safe for concurrent use on the same
manager instance.

With `stream_complete`, `process_response` runs only after the stream
is fully drained. Consumers that `break` out of the stream early skip
`process_response` (the scoped middleware is still detached correctly);
use `complete()` or drain the stream if you rely on that behavior.

#### Persisting Middleware Audit Data

Middleware writes to `response.metadata` are **ephemeral by default** —
they live on the `LLMResponse` for this call but do not flow to the
persisted assistant conversation node. To opt in to persistence, write
into the `_persist` sub-dictionary:

```python
class CitationAuditMiddleware(ConversationMiddleware):
    async def process_response(self, response, state):
        if response.metadata is None:
            response.metadata = {}
        # `_persist` is the single opt-in gate for persistence.
        persist = response.metadata.setdefault("_persist", {})
        persist["citation_audit"] = self._outcome
        return response
```

Keys inside `_persist` are merged into the assistant node's metadata by
`ConversationManager._finalize_completion` — `_persist` itself is not
propagated. Canonical framework fields (`usage`, `model`, `provider`,
`finish_reason`, cost, config overrides) and the caller's `metadata=`
kwarg win on key conflict; non-dict `_persist` values are skipped with a
WARNING log.

To persist keys written as **flat** `response.metadata` entries by a
provider or an existing middleware (e.g. an Ollama-style provider's
`eval_duration`, `RateLimitMiddleware`'s `rate_limit_count`) without
modifying the writer's source, use `PromoteToPersistMiddleware` at
position `[0]` of the `middleware` list:

```python
from dataknobs_llm.conversations import (
    PromoteToPersistMiddleware,
    RateLimitMiddleware,
)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    middleware=[
        # Position-[0] = runs LAST on response, after other middleware
        # have written their flat keys. Place the promoter at position
        # [0] so it captures those writes. Provider writes are already
        # in response.metadata before any middleware runs, so position
        # is irrelevant for provider-sourced keys.
        PromoteToPersistMiddleware(keys=[
            "rate_limit_count",
            "eval_duration",
        ]),
        RateLimitMiddleware(max_requests=10, window_seconds=60),
    ],
)
```

The promoter uses `setdefault` into `_persist`, so a same-named key
already written by a native `_persist` writer takes precedence over
passive promotion. (Native-vs-native `_persist` collisions instead
follow onion ordering — the outer middleware's write wins.)

For one-shot promotion on a single call, register the promoter via
`manager.scoped_middleware(PromoteToPersistMiddleware(keys=[...]))`
instead of adding it to the permanent `middleware=[...]` list.

See `packages/llm/docs/USER_GUIDE.md` for the full contract.

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
