# RAG Caching in Conversations

This document describes the RAG (Retrieval-Augmented Generation) caching feature in the dataknobs-llm package, which optimizes conversation performance by caching and reusing RAG search results.

## Overview

RAG caching reduces latency and costs by:
- **Storing RAG metadata** in conversation nodes for transparency and debugging
- **Reusing cached results** when the same RAG query is executed again
- **Supporting conversation branching** with intelligent cache lookup across the tree

## Table of Contents

1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Configuration](#configuration)
4. [Cache Matching](#cache-matching)
5. [Inspecting RAG Metadata](#inspecting-rag-metadata)
6. [Best Practices](#best-practices)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### Enable RAG Caching

```python
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

# Configure prompts with RAG
config = {
    "system": {
        "assistant": {"template": "You are a helpful coding assistant."}
    },
    "user": {
        "code_question": {
            "template": "Context: {{DOCS}}\n\nQuestion: {{question}}",
            "rag_configs": [{
                "adapter_name": "docs",
                "query": "{{language}} {{topic}}",
                "placeholder": "DOCS"
            }]
        }
    }
}

library = ConfigPromptLibrary(config)
builder = AsyncPromptBuilder(library=library, adapters={"docs": docs_adapter})

# Create manager with caching enabled
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="assistant",
    cache_rag_results=True,        # Store RAG metadata
    reuse_rag_on_branch=True       # Reuse cache when branching
)
```

### Use in Conversations

```python
# First message executes RAG search
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"question": "How do decorators work?", "language": "python", "topic": "decorators"}
)
await manager.complete()

# Branch back and add similar message
await manager.switch_to_node("0")

# This reuses cached RAG (same query parameters!)
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"question": "Explain decorators", "language": "python", "topic": "decorators"}
)
```

## How It Works

### 1. RAG Metadata Capture

When `cache_rag_results=True`, the prompt builder captures detailed metadata for each RAG search:

```python
{
    "DOCS": {  # Placeholder name
        "adapter_name": "docs",
        "query": "python decorators",  # Resolved query string
        "query_hash": "a1b2c3...",     # SHA256 hash for matching
        "k": 5,
        "timestamp": "2025-01-15T10:30:00",
        "results": [
            {"content": "...", "score": 0.95, "metadata": {...}},
            ...
        ],
        "formatted_content": "# Relevant Documentation\n\n- Python decorators..."
    }
}
```

This metadata is stored in the conversation node's `metadata` field.

### 2. Cache Lookup

When `reuse_rag_on_branch=True`, the conversation manager searches for cached RAG before executing new searches:

1. **Extract RAG configs** from the prompt template
2. **Render query templates** with current parameters
3. **Compute query hashes** for each RAG config
4. **Search conversation tree** for matching cached RAG
5. **Match on query hashes** to ensure same query parameters
6. **Reuse if found**, otherwise execute fresh search

### 3. Query Hash Matching

Cache matching uses SHA256 hashes of `adapter_name:resolved_query`:

```python
# These match (same resolved query):
query1 = "python decorators"  # from params: {language: "python", topic: "decorators"}
query2 = "python decorators"  # from params: {language: "python", topic: "decorators"}
# Hash: sha256("docs:python decorators") == sha256("docs:python decorators") ✓

# These don't match (different resolved queries):
query3 = "python async"       # from params: {language: "python", topic: "async"}
# Hash: sha256("docs:python async") != sha256("docs:python decorators") ✗
```

## Configuration

### cache_rag_results

**Type**: `bool`
**Default**: `False`
**When to use**: Enable for debugging, transparency, or when you plan to use `reuse_rag_on_branch`

**What it does**:
- Stores full RAG metadata in conversation nodes
- Enables inspection of what documents were retrieved
- Required for `reuse_rag_on_branch` to work

**Performance impact**: Minimal (only metadata storage overhead)

```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="assistant",
    cache_rag_results=True  # Enable metadata capture
)
```

### reuse_rag_on_branch

**Type**: `bool`
**Default**: `False`
**When to use**: Enable when conversations involve branching with repeated queries

**What it does**:
- Searches conversation tree for cached RAG results
- Reuses cache if query hashes match
- Skips expensive RAG searches when possible

**Performance impact**: Positive (reduces RAG search latency and costs)

**Requires**: `cache_rag_results=True`

```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="assistant",
    cache_rag_results=True,
    reuse_rag_on_branch=True  # Enable cache reuse
)
```

## Cache Matching

### When Cache is Reused

Cache is reused when **ALL** of the following match:

1. **Prompt name** matches (e.g., `"code_question"`)
2. **Role** matches (e.g., `"user"`)
3. **RAG query hashes** match for all placeholders

### When Cache is NOT Reused

Cache is skipped when:

- Different prompt name
- Different role
- Different query parameters (different hash)
- No matching cache exists in tree
- `reuse_rag_on_branch=False`

### Example: Cache Hit vs Miss

```python
# Message 1: Executes RAG search
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"language": "python", "topic": "decorators"}
)
# Query: "python decorators" → Hash: abc123...

# Message 2: Cache HIT (same query hash)
await manager.switch_to_node("0")
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"language": "python", "topic": "decorators"}  # Same!
)
# Query: "python decorators" → Hash: abc123... ✓ REUSES CACHE

# Message 3: Cache MISS (different query hash)
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"language": "python", "topic": "async"}  # Different!
)
# Query: "python async" → Hash: def456... ✗ EXECUTES NEW SEARCH
```

## Inspecting RAG Metadata

### Get RAG Metadata from Current Node

```python
# Add message with RAG
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"question": "How?", "language": "python", "topic": "async"}
)

# Inspect RAG metadata
metadata = manager.get_rag_metadata()

if metadata:
    for placeholder, rag_data in metadata.items():
        print(f"Placeholder: {placeholder}")
        print(f"  Query: {rag_data['query']}")
        print(f"  Adapter: {rag_data['adapter_name']}")
        print(f"  Results: {len(rag_data['results'])}")
        print(f"  Timestamp: {rag_data['timestamp']}")

        # Inspect individual results
        for i, result in enumerate(rag_data['results']):
            print(f"  Result {i+1}:")
            print(f"    Score: {result.get('score', 'N/A')}")
            print(f"    Content preview: {result['content'][:100]}...")
```

### Get RAG Metadata from Specific Node

```python
# Get metadata from a specific node
metadata = manager.get_rag_metadata(node_id="0.1.0")

if metadata:
    print(f"Node 0.1.0 retrieved {len(metadata)} RAG results")
```

### Analyze Cache Effectiveness

```python
# Track cache hits
from unittest.mock import Mock

adapter = Mock()
adapter.search = Mock(return_value=[...])

# Add first message
await manager.add_message(...)
initial_calls = adapter.search.call_count

# Branch and reuse
await manager.switch_to_node("0")
await manager.add_message(...)  # Should reuse cache

# Check if cache was hit
if adapter.search.call_count == initial_calls:
    print("✓ Cache HIT - no new search executed")
else:
    print("✗ Cache MISS - new search executed")
```

## Best Practices

### 1. Enable Caching for Debugging

Even if you don't plan to reuse cache, enabling `cache_rag_results` helps with debugging:

```python
manager = await ConversationManager.create(
    cache_rag_results=True,   # Good for development
    reuse_rag_on_branch=False  # Disable reuse in production if not needed
)
```

### 2. Use Descriptive RAG Query Templates

Make queries deterministic and cacheable:

```python
# Good: Deterministic query
"rag_configs": [{
    "query": "{{language}} {{topic}} documentation"
}]

# Avoid: Non-deterministic queries
"rag_configs": [{
    "query": "{{language}} help me with {{user_free_text}}"  # Varies too much
}]
```

### 3. Parameterize Wisely

Group similar queries under the same parameters:

```python
# Good: Reusable parameters
params = {"language": "python", "topic": "async"}

# Less good: Too specific
params = {"language": "python", "topic": "async/await syntax in Python 3.10"}
```

### 4. Monitor Cache Performance

```python
# Wrap adapter to track calls
class TrackedAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.call_count = 0

    async def search(self, **kwargs):
        self.call_count += 1
        return await self.adapter.search(**kwargs)

# Use tracked adapter
tracked_adapter = TrackedAdapter(real_adapter)
builder = AsyncPromptBuilder(adapters={"docs": tracked_adapter})

# After conversation
print(f"Total RAG searches: {tracked_adapter.call_count}")
```

### 5. Clear Cache When Needed

If your RAG index is updated, you may want to start fresh:

```python
# Option 1: Create new conversation
manager = await ConversationManager.create(...)

# Option 2: Disable reuse temporarily
manager.reuse_rag_on_branch = False
await manager.add_message(...)  # Forces fresh search
manager.reuse_rag_on_branch = True
```

## Advanced Usage

### Multiple RAG Adapters

Cache works with multiple RAG configs per prompt:

```python
"rag_configs": [
    {
        "adapter_name": "docs",
        "query": "{{language}} {{topic}} documentation",
        "placeholder": "DOCS"
    },
    {
        "adapter_name": "examples",
        "query": "{{language}} {{topic}} code examples",
        "placeholder": "EXAMPLES"
    }
]
```

Cache matches when **all** query hashes match:

- Both DOCS and EXAMPLES must have matching cached queries
- If only one matches, cache is NOT reused
- All searches execute together or none do

### Custom Cache Logic

For advanced use cases, you can implement custom caching:

```python
class CachingAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.cache = {}

    async def search(self, query, **kwargs):
        cache_key = f"{query}:{kwargs.get('k', 10)}"

        if cache_key in self.cache:
            print(f"✓ Adapter-level cache hit: {query}")
            return self.cache[cache_key]

        results = await self.adapter.search(query, **kwargs)
        self.cache[cache_key] = results
        return results
```

### Conversation Tree Analysis

Analyze RAG usage across the entire conversation tree:

```python
def analyze_rag_usage(manager):
    """Analyze RAG cache usage in conversation tree."""
    from collections import deque

    stats = {
        "total_nodes": 0,
        "nodes_with_rag": 0,
        "unique_queries": set(),
        "adapters_used": set()
    }

    queue = deque([manager.state.message_tree])

    while queue:
        node = queue.popleft()
        stats["total_nodes"] += 1

        rag_metadata = node.data.metadata.get("rag_metadata")
        if rag_metadata:
            stats["nodes_with_rag"] += 1

            for placeholder, rag_data in rag_metadata.items():
                stats["unique_queries"].add(rag_data.get("query"))
                stats["adapters_used"].add(rag_data.get("adapter_name"))

        if node.children:
            queue.extend(node.children)

    stats["unique_queries"] = len(stats["unique_queries"])
    stats["adapters_used"] = list(stats["adapters_used"])

    return stats

# Usage
stats = analyze_rag_usage(manager)
print(f"Total nodes: {stats['total_nodes']}")
print(f"Nodes with RAG: {stats['nodes_with_rag']}")
print(f"Unique queries: {stats['unique_queries']}")
print(f"Adapters used: {stats['adapters_used']}")
```

## Troubleshooting

### Cache not being reused?

Check these common issues:

1. **Is `cache_rag_results=True`?**
   ```python
   print(manager.cache_rag_results)  # Should be True
   ```

2. **Is `reuse_rag_on_branch=True`?**
   ```python
   print(manager.reuse_rag_on_branch)  # Should be True
   ```

3. **Are query parameters identical?**
   ```python
   metadata = manager.get_rag_metadata()
   print(metadata["DOCS"]["query"])  # Check actual resolved query
   print(metadata["DOCS"]["query_hash"])  # Check hash
   ```

4. **Does cached node exist in tree?**
   ```python
   # Search tree for matching prompts
   from collections import deque
   queue = deque([manager.state.message_tree])
   matches = []

   while queue:
       node = queue.popleft()
       if (node.data.prompt_name == "code_question" and
           node.data.message.role == "user"):
           matches.append(node.data.node_id)
       if node.children:
           queue.extend(node.children)

   print(f"Nodes with matching prompt: {matches}")
   ```

### Cache size concerns?

RAG metadata increases conversation storage:

```python
# Estimate cache size
import sys

metadata = manager.get_rag_metadata()
if metadata:
    size_bytes = sys.getsizeof(str(metadata))
    print(f"RAG metadata size: {size_bytes / 1024:.2f} KB")
```

Mitigation strategies:
- Use `cache_rag_results=False` in production if not needed
- Implement custom storage with compression
- Periodically prune old conversations

## See Also

- [USER_GUIDE.md](USER_GUIDE.md) - General conversation management
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - RAG configuration best practices
- [Prompt Builder Documentation](../src/dataknobs_llm/prompts/builders/README.md) - RAG configuration details
