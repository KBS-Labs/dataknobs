# RAG Ingestion and Hybrid Search

This document covers the RAGKnowledgeBase methods for loading documents from directories and performing hybrid search queries.

## Overview

The RAGKnowledgeBase provides two powerful methods for working with document collections:

1. **`load_from_directory()`**: Batch load documents using KnowledgeBaseConfig
2. **`hybrid_query()`**: Search using combined text and vector similarity

## Quick Start

### Loading Documents

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    KnowledgeBaseConfig,
    FilePatternConfig,
)

# Create knowledge base
kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
})

# Load documents with configuration
config = KnowledgeBaseConfig(
    name="product-docs",
    patterns=[
        FilePatternConfig(pattern="api/**/*.json", text_fields=["title", "description"]),
        FilePatternConfig(pattern="**/*.md"),
    ],
    exclude_patterns=["**/drafts/**"],
)

results = await kb.load_from_directory("./docs", config=config)
print(f"Loaded {results['total_chunks']} chunks from {results['total_files']} files")
```

### Hybrid Search

```python
# Combine text and vector search
results = await kb.hybrid_query(
    "how to configure authentication",
    k=5,
    text_weight=0.4,
    vector_weight=0.6,
)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['heading_path']}")
    print(f"  text: {r['text_score']:.3f}, vector: {r['vector_score']:.3f}")
    print(r['text'][:100])
```

## load_from_directory()

Load documents from a directory using the xization DirectoryProcessor with configurable patterns, chunking, and metadata.

### Method Signature

```python
async def load_from_directory(
    self,
    directory: str | Path,
    config: KnowledgeBaseConfig | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `directory` | `str \| Path` | Directory path containing documents |
| `config` | `KnowledgeBaseConfig \| None` | Configuration for processing. If None, loads from `knowledge_base.json/yaml` in directory or uses defaults |
| `progress_callback` | `Callable \| None` | Optional callback `(file_path, num_chunks)` for progress tracking |

### Returns

Dictionary with loading statistics:

```python
{
    "total_files": 15,
    "total_chunks": 234,
    "files_by_type": {
        "markdown": 10,
        "json": 4,
        "jsonl": 1,
    },
    "errors": [],
    "documents": [
        {"source": "/path/to/file.md", "type": "markdown", "chunks": 12, "errors": []},
        # ...
    ]
}
```

### Examples

#### Basic Usage

```python
# Auto-load config from directory
results = await kb.load_from_directory("./docs")

print(f"Files: {results['total_files']}")
print(f"Chunks: {results['total_chunks']}")
print(f"Errors: {len(results['errors'])}")
```

#### With Explicit Configuration

```python
from dataknobs_bots.knowledge import KnowledgeBaseConfig, FilePatternConfig

config = KnowledgeBaseConfig(
    name="api-docs",
    default_chunking={
        "max_chunk_size": 500,
        "chunk_overlap": 50,
    },
    patterns=[
        FilePatternConfig(
            pattern="api/**/*.json",
            text_fields=["title", "description", "examples"],
        ),
        FilePatternConfig(
            pattern="guides/**/*.md",
            chunking={"max_chunk_size": 800},  # Override for guides
        ),
    ],
    exclude_patterns=[
        "**/drafts/**",
        "**/test/**",
        "*.tmp",
    ],
    default_metadata={
        "version": "2.0",
        "source": "documentation",
    },
)

results = await kb.load_from_directory("./docs", config=config)
```

#### With Progress Tracking

```python
def on_progress(file_path: str, num_chunks: int):
    print(f"Processed {file_path}: {num_chunks} chunks")

results = await kb.load_from_directory(
    "./docs",
    progress_callback=on_progress,
)
```

#### Processing Large Directories

```python
import asyncio

async def load_with_batching(kb, directory: str, batch_size: int = 10):
    """Load documents with batched progress output."""
    progress = {"files": 0, "chunks": 0}

    def on_progress(path: str, chunks: int):
        progress["files"] += 1
        progress["chunks"] += chunks
        if progress["files"] % batch_size == 0:
            print(f"Progress: {progress['files']} files, {progress['chunks']} chunks")

    results = await kb.load_from_directory(directory, progress_callback=on_progress)

    print(f"Complete: {results['total_files']} files, {results['total_chunks']} chunks")
    return results
```

## hybrid_query()

Query the knowledge base using hybrid search that combines keyword matching with semantic vector search.

### Method Signature

```python
async def hybrid_query(
    self,
    query: str,
    k: int = 5,
    text_weight: float = 0.5,
    vector_weight: float = 0.5,
    fusion_strategy: str = "rrf",
    text_fields: list[str] | None = None,
    filter_metadata: dict[str, Any] | None = None,
    min_similarity: float = 0.0,
    merge_adjacent: bool = False,
    max_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Query text to search for |
| `k` | `int` | 5 | Number of results to return |
| `text_weight` | `float` | 0.5 | Weight for text search (0.0-1.0) |
| `vector_weight` | `float` | 0.5 | Weight for vector search (0.0-1.0) |
| `fusion_strategy` | `str` | "rrf" | Fusion method: "rrf", "weighted_sum", or "native" |
| `text_fields` | `list[str] \| None` | None | Fields to search (default: ["text"]) |
| `filter_metadata` | `dict \| None` | None | Metadata filters to apply |
| `min_similarity` | `float` | 0.0 | Minimum combined score threshold |
| `merge_adjacent` | `bool` | False | Merge adjacent chunks from same section |
| `max_chunk_size` | `int \| None` | None | Max size for merged chunks |

### Returns

List of result dictionaries:

```python
[
    {
        "text": "Chunk content...",
        "source": "/path/to/file.md",
        "heading_path": "Section > Subsection",
        "similarity": 0.85,          # Combined score
        "text_score": 0.75,          # Text match score
        "vector_score": 0.92,        # Vector similarity score
        "metadata": {...},           # Full chunk metadata
    },
    # ...
]
```

### Examples

#### Basic Hybrid Search

```python
results = await kb.hybrid_query("database configuration", k=5)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['heading_path']}")
    print(r['text'][:200])
    print()
```

#### Weighted Toward Semantic Search

```python
# When exact keywords are less important than meaning
results = await kb.hybrid_query(
    "how do I set up user authentication",
    k=5,
    text_weight=0.3,
    vector_weight=0.7,
)
```

#### Weighted Toward Keyword Search

```python
# When exact terms matter (e.g., error codes, product names)
results = await kb.hybrid_query(
    "ERROR_CODE_AUTH_FAILED",
    k=5,
    text_weight=0.8,
    vector_weight=0.2,
)
```

#### Using Weighted Sum Fusion

```python
results = await kb.hybrid_query(
    "install dependencies",
    k=5,
    fusion_strategy="weighted_sum",
    text_weight=0.5,
    vector_weight=0.5,
)
```

#### With Metadata Filtering

```python
results = await kb.hybrid_query(
    "API endpoints",
    k=10,
    filter_metadata={"document_type": "api", "version": "2.0"},
)
```

#### With Chunk Merging

```python
# Merge adjacent chunks for more context
results = await kb.hybrid_query(
    "authentication flow",
    k=5,
    merge_adjacent=True,
    max_chunk_size=2000,
)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['heading_path']}")
    print(f"Content length: {len(r['text'])} chars")
```

#### Custom Text Fields

```python
# Search specific metadata fields
results = await kb.hybrid_query(
    "OAuth 2.0",
    k=5,
    text_fields=["text", "heading_path", "source"],
)
```

## Fusion Strategies

### RRF (Reciprocal Rank Fusion)

Default strategy. Combines based on rank position:

```
RRF_score(d) = sum(weight / (k + rank))
```

**Best for:**
- General-purpose hybrid search
- When score distributions differ between text and vector
- Robust default choice

```python
results = await kb.hybrid_query(query, fusion_strategy="rrf")
```

### Weighted Sum

Combines normalized scores directly:

```
combined = text_weight * norm(text_score) + vector_weight * norm(vector_score)
```

**Best for:**
- When scores are comparable
- Fine-tuned weight control needed

```python
results = await kb.hybrid_query(query, fusion_strategy="weighted_sum")
```

### Native

Uses backend's native hybrid search if available:

```python
results = await kb.hybrid_query(query, fusion_strategy="native")
# Falls back to RRF if backend doesn't support native
```

## Complete Example

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    KnowledgeBaseConfig,
    FilePatternConfig,
)

async def build_and_query_kb():
    # Create knowledge base
    kb = await RAGKnowledgeBase.from_config({
        "vector_store": {"backend": "faiss", "dimensions": 384, "persist_path": "./kb_index"},
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    })

    # Load documents
    config = KnowledgeBaseConfig(
        name="docs",
        patterns=[
            FilePatternConfig(pattern="**/*.md"),
            FilePatternConfig(pattern="api/*.json", text_fields=["title", "body"]),
        ],
    )

    results = await kb.load_from_directory("./docs", config=config)
    print(f"Loaded {results['total_chunks']} chunks")

    # Hybrid search
    search_results = await kb.hybrid_query(
        "How do I configure OAuth authentication?",
        k=5,
        text_weight=0.4,
        vector_weight=0.6,
        merge_adjacent=True,
    )

    # Format for LLM
    context = kb.format_context(search_results)
    print(context)

    # Save and close
    await kb.close()

# Run
import asyncio
asyncio.run(build_and_query_kb())
```

## Exported Types

The following types are re-exported from the knowledge module for convenience:

```python
from dataknobs_bots.knowledge import (
    # Main class
    RAGKnowledgeBase,

    # Ingestion types
    DirectoryProcessor,
    FilePatternConfig,
    KnowledgeBaseConfig,
    ProcessedDocument,

    # Hybrid search types
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
)
```

## Testing

```bash
cd packages/bots
uv run pytest tests/test_knowledge.py -v -k "load_from_directory or hybrid_query"
```

## Related

- [RAG Retrieval](RAG_RETRIEVAL.md) - Chunk merging and formatting
- [RAG Query](RAG_QUERY.md) - Query transformation and expansion
- [User Guide](USER_GUIDE.md) - Complete bot usage guide
