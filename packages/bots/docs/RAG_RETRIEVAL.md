# RAG Retrieval Utilities

This module provides utilities for optimizing RAG retrieval results, including chunk merging and context formatting.

## Overview

After retrieving chunks from a vector store, raw results often need optimization:
- Adjacent chunks from the same section should be merged for coherence
- Context should be formatted intelligently for LLM consumption
- Headings should be included dynamically based on content size

## Quick Start

```python
from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    MergerConfig,
    FormatterConfig,
)

# Create merger and formatter
merger = ChunkMerger(MergerConfig(max_merged_size=2000))
formatter = ContextFormatter(FormatterConfig(include_source=True))

# After retrieval
results = await kb.query("How do I configure auth?", k=10)

# Merge adjacent chunks
merged = merger.merge(results)

# Format for LLM context
context = formatter.format_merged(merged)
wrapped = formatter.wrap_for_prompt(context)
```

## Chunk Merging

### The Problem

Without merging, retrieval returns fragmented context:
```
[0.92] Chunk 5: "...continued from above. The second step is..."
[0.88] Chunk 3: "The first step is to..."
[0.85] Chunk 7: "Finally, the third step..."
```

The LLM sees disconnected snippets in the wrong order.

### The Solution

`ChunkMerger` groups chunks by their heading path and source, then combines them:
```
[0.88] Authentication > Setup
The first step is to...

...continued from above. The second step is...

Finally, the third step...
```

### MergerConfig

```python
from dataknobs_bots.knowledge.retrieval import MergerConfig

config = MergerConfig(
    max_merged_size=2000,   # Maximum merged chunk size in chars
    preserve_order=True      # Maintain document order within groups
)
```

### ChunkMerger

```python
from dataknobs_bots.knowledge.retrieval import ChunkMerger, MergerConfig

merger = ChunkMerger(MergerConfig(max_merged_size=2000))

# Merge search results
results = await kb.query("authentication setup", k=10)
merged = merger.merge(results)

for chunk in merged:
    print(f"[{chunk.avg_similarity:.2f}] {chunk.heading_display}")
    print(f"Merged {len(chunk.chunks)} chunks")
    print(chunk.text)
    print()
```

### MergedChunk Fields

```python
merged_chunk.text           # Combined text content
merged_chunk.source         # Source file path
merged_chunk.heading_path   # ["Section", "Subsection"]
merged_chunk.heading_display # "Section > Subsection"
merged_chunk.chunks         # List of original chunks
merged_chunk.avg_similarity # Average similarity score
merged_chunk.content_length # Total content length
```

### Converting Back to Results

```python
# Convert merged chunks to standard result format
result_list = merger.to_result_list(merged)

for result in result_list:
    print(result["text"])
    print(result["similarity"])
    print(result["metadata"]["merged_count"])
```

## Context Formatting

### The Problem

Context window tokens are expensive. Including full heading paths for every chunk wastes tokens:
```
[1] Getting Started > Installation > System Requirements > Dependencies > Python
Just the content about Python requirements.
```

### The Solution

`ContextFormatter` applies dynamic heading inclusion based on content size:
- **Small chunks** (< 200 chars): Full heading path (need context)
- **Medium chunks** (< 800 chars): Last 2 heading levels
- **Large chunks** (> 800 chars): No headings (self-contained)

### FormatterConfig

```python
from dataknobs_bots.knowledge.retrieval import FormatterConfig

config = FormatterConfig(
    small_chunk_threshold=200,   # Full headings below this
    medium_chunk_threshold=800,  # Partial headings below this
    include_scores=False,        # Show similarity scores
    include_source=True,         # Show source file
    group_by_source=False,       # Group chunks by file
)
```

### ContextFormatter

```python
from dataknobs_bots.knowledge.retrieval import ContextFormatter, FormatterConfig

formatter = ContextFormatter(FormatterConfig(
    small_chunk_threshold=200,
    include_scores=True,
    include_source=True
))

# Format standard results
context = formatter.format(results)

# Format merged chunks
context = formatter.format_merged(merged_chunks)

# Wrap for prompt injection
wrapped = formatter.wrap_for_prompt(context, tag="knowledge_base")
```

### Output Example

```
<knowledge_base>
[1] [0.92] Authentication > OAuth 2.0
Configure OAuth by setting the client ID and secret in your
environment variables. The callback URL should point to your
application's auth endpoint.
(Source: docs/auth.md)

---

[2] [0.88] Getting Started
First, install the package using pip. Then configure your
API keys in the environment or config file.
(Source: docs/quickstart.md)
</knowledge_base>
```

### Grouping by Source

```python
formatter = ContextFormatter(FormatterConfig(group_by_source=True))
context = formatter.format(results)
```

Output:
```
## Source: docs/auth.md

[1] Authentication > OAuth 2.0
Content here...

[2] Authentication > API Keys
More content...

---

## Source: docs/setup.md

[3] Installation
Setup content...
```

## Complete Integration

### Basic RAG Pipeline

```python
from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    MergerConfig,
    FormatterConfig,
)

class RAGKnowledgeBase:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.merger = ChunkMerger(MergerConfig(max_merged_size=2000))
        self.formatter = ContextFormatter(FormatterConfig(
            include_scores=False,
            include_source=True
        ))

    async def query(self, query: str, k: int = 10):
        # Search vector store
        results = await self.vector_store.search(query, k=k)

        # Merge adjacent chunks
        merged = self.merger.merge(results)

        return merged

    def format_context(self, merged_chunks, wrap=True):
        context = self.formatter.format_merged(merged_chunks)
        if wrap:
            return self.formatter.wrap_for_prompt(context)
        return context
```

### In DynaBot Chat

```python
class EnhancedBot(DynaBot):
    async def _build_message_with_context(self, message, rag_query=None):
        if self.knowledge_base:
            query = rag_query or message
            results = await self.knowledge_base.query(query, k=10)
            context = self.knowledge_base.format_context(results)

            return f"{context}\n\nUser: {message}"

        return message
```

## API Reference

### MergerConfig

```python
@dataclass
class MergerConfig:
    max_merged_size: int = 2000   # Maximum merged chunk size
    preserve_order: bool = True    # Maintain document order
```

### MergedChunk

```python
@dataclass
class MergedChunk:
    text: str                      # Combined text
    source: str                    # Source file
    heading_path: list[str]        # Heading hierarchy
    heading_display: str           # Formatted display
    chunks: list[dict[str, Any]]   # Original chunks
    avg_similarity: float          # Average similarity
    content_length: int            # Total length
```

### ChunkMerger

```python
class ChunkMerger:
    def __init__(self, config: MergerConfig | None = None):
        """Initialize with optional configuration."""

    def merge(self, results: list[dict[str, Any]]) -> list[MergedChunk]:
        """Merge search results by shared heading path."""

    def to_result_list(self, merged: list[MergedChunk]) -> list[dict[str, Any]]:
        """Convert merged chunks back to result format."""
```

### FormatterConfig

```python
@dataclass
class FormatterConfig:
    small_chunk_threshold: int = 200    # Full headings below this
    medium_chunk_threshold: int = 800   # Partial headings below this
    include_scores: bool = False        # Show similarity scores
    include_source: bool = True         # Show source file
    group_by_source: bool = False       # Group by source file
```

### ContextFormatter

```python
class ContextFormatter:
    def __init__(self, config: FormatterConfig | None = None):
        """Initialize with optional configuration."""

    def format(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM context."""

    def format_merged(self, merged_chunks: list[MergedChunk]) -> str:
        """Format merged chunks for LLM context."""

    def wrap_for_prompt(self, context: str, tag: str = "knowledge_base") -> str:
        """Wrap context in XML tags for prompt injection."""
```

## Related

- [Query Processing](rag-query.md) - Query transformation and expansion
- [User Guide](user-guide.md) - Complete bot usage guide
