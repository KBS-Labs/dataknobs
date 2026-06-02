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

#### Citation Carry-Over Redaction (`HistoryRedactionMiddleware`)

Bots that emit structured citation tokens — bracketed headers like
`[bib:N · vendor · slug]`, bare `bib:N` references, footnote markers —
carry those tokens forward in the persisted conversation tree. On the
next turn, the model sees its own prior citations as part of conversation
history and reuses them, even when *this* turn's retrieval no longer
surfaces those sources. The result is a **citation carry-over leak**:
bibs cited inline that are not in the current kept set.

`HistoryRedactionMiddleware` rewrites assistant-role messages on their
way INTO the LLM. The persisted tree, exports, and the response of the
fresh turn are NOT mutated — only the in-memory message list this turn
forwards to the provider:

```python
from dataknobs_llm.conversations import HistoryRedactionMiddleware

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[
        HistoryRedactionMiddleware(
            redactions=[
                # Bracketed header MUST come first — it's the more
                # specific pattern. If the bare-token rule ran first
                # it would consume the `bib:N` inside the bracket and
                # leave a malformed `[ · vendor · …]` header.
                {"pattern": r"\[bib:\d+[^\]]*\]",
                 "replacement": "[prior citation]"},
                {"pattern": r"\bbib:\d+\b",
                 "replacement": "[prior citation]"},
            ],
        ),
    ],
)
```

Pass an `redact_roles=(...,)` override to widen the rewrite beyond the
assistant role (e.g. for tool-result messages that quote prior assistant
output). The default is `("assistant",)` — users rarely type citation
tokens, and system prompts may intentionally reference bibs for the
bibliography menu.

Each redaction spec is validated at construction time: a missing
`pattern` key, an empty pattern, or an invalid regex raises a clear
`ValueError` / `re.error` at the config-load boundary rather than
mid-loop on the first request. Non-content fields on the rewritten
assistant message — `tool_calls`, `tool_call_id`, `name`,
`function_call`, `metadata` — are preserved, so agent / tool-use loops
keep their invocation and pairing fields intact across the redaction.

DynaBot users do not configure this middleware on the
`ConversationManager` directly; they list it under
`DynaBotConfig.conversation_middleware` and the bot forwards it to every
manager it constructs. To inject a pre-built instance against an
otherwise config-driven bot, pass it via the
`conversation_middleware=` kwarg on `DynaBot.from_config(...)` — it
replaces the config-driven list symmetrically with the existing
`middleware=` kwarg.

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

### Metadata Propagation

`state.metadata` is mirrored into the underlying `Record.metadata` on save,
so backends with a dedicated metadata column (Postgres, Elasticsearch, etc.)
can index and filter on it. `list_conversations(filter_metadata={"key": value})`
and `count_conversations(filter_metadata=...)` translate to a `metadata.<key>`
query that SQL backends route to the metadata column directly. The
serialized state in the data column also still includes the metadata block,
so loading a conversation and round-tripping it is unaffected.

`state.metadata` is typed `Dict[str, Any]` and may contain JSON-serializable
nested values; the persistence layer deep-copies `state.metadata` on save,
so post-save mutations of nested values do not affect already-persisted
rows. SQL backends index top-level keys, so nested-value filtering is
outside the `filter_metadata` contract.

## See Also

- [FSM-Based Flows](flows.md) - Workflow orchestration
- [Prompt Engineering](prompts.md) - Building prompts
- [Performance](performance.md) - RAG caching and optimization
- [API Reference](../api/conversations.md) - Full API documentation
