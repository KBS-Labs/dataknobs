# RAG Ingestion and Hybrid Search

This guide covers loading documents from directories and performing hybrid search queries with the RAGKnowledgeBase.

## Overview

The RAGKnowledgeBase provides two powerful methods:

1. **`load_from_directory()`**: Batch load documents using KnowledgeBaseConfig
2. **`hybrid_query()`**: Search using combined text and vector similarity

## Loading Documents

### Basic Usage

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase

kb = await RAGKnowledgeBase.from_config(config)

# Auto-load config from directory
results = await kb.load_from_directory("./docs")

print(f"Files: {results['total_files']}")
print(f"Chunks: {results['total_chunks']}")
```

### With Configuration

```python
from dataknobs_bots.knowledge import KnowledgeBaseConfig, FilePatternConfig

config = KnowledgeBaseConfig(
    name="product-docs",
    patterns=[
        FilePatternConfig(
            pattern="api/**/*.json",
            text_fields=["title", "description"],
        ),
        FilePatternConfig(pattern="**/*.md"),
    ],
    exclude_patterns=["**/drafts/**"],
)

results = await kb.load_from_directory("./docs", config=config)
```

### Progress Tracking

```python
def on_progress(file_path: str, num_chunks: int):
    print(f"Processed {file_path}: {num_chunks} chunks")

results = await kb.load_from_directory(
    "./docs",
    progress_callback=on_progress,
)
```

### Return Value

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
    ]
}
```

## Hybrid Search

### Basic Query

```python
results = await kb.hybrid_query("database configuration", k=5)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['heading_path']}")
    print(r['text'][:200])
```

### Custom Weights

```python
# Semantic-focused (natural language queries)
results = await kb.hybrid_query(
    "how do I set up user authentication",
    text_weight=0.3,
    vector_weight=0.7,
)

# Keyword-focused (exact terms, error codes)
results = await kb.hybrid_query(
    "ERROR_CODE_AUTH_FAILED",
    text_weight=0.8,
    vector_weight=0.2,
)
```

### Fusion Strategies

```python
# RRF (default) - robust, no score normalization needed
results = await kb.hybrid_query(query, fusion_strategy="rrf")

# Weighted Sum - direct score combination
results = await kb.hybrid_query(query, fusion_strategy="weighted_sum")

# Native - use backend's hybrid search if available
results = await kb.hybrid_query(query, fusion_strategy="native")
```

### With Chunk Merging

```python
results = await kb.hybrid_query(
    "authentication flow",
    k=5,
    merge_adjacent=True,
    max_chunk_size=2000,
)
```

### Return Value

```python
[
    {
        "text": "Chunk content...",
        "source": "/path/to/file.md",
        "heading_path": "Section > Subsection",
        "similarity": 0.85,          # Combined score
        "text_score": 0.75,          # Text match score
        "vector_score": 0.92,        # Vector similarity
        "metadata": {...},
    },
]
```

## Complete Example

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    KnowledgeBaseConfig,
    FilePatternConfig,
)

async def build_and_query():
    # Create knowledge base
    kb = await RAGKnowledgeBase.from_config({
        "vector_store": {"backend": "faiss", "dimensions": 384},
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    })

    # Load documents
    config = KnowledgeBaseConfig(
        name="docs",
        patterns=[
            FilePatternConfig(pattern="**/*.md"),
            FilePatternConfig(pattern="api/*.json", text_fields=["title"]),
        ],
    )

    results = await kb.load_from_directory("./docs", config=config)
    print(f"Loaded {results['total_chunks']} chunks")

    # Hybrid search
    search_results = await kb.hybrid_query(
        "How do I configure OAuth?",
        k=5,
        text_weight=0.4,
        vector_weight=0.6,
    )

    # Format for LLM
    context = kb.format_context(search_results)
    print(context)

    await kb.close()
```

## Automatic Ingestion with KnowledgeIngestionService

The `KnowledgeIngestionService` provides high-level ingestion management with automatic population checks and skip-if-populated behavior.

### Basic Usage

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    KnowledgeIngestionService,
)

service = KnowledgeIngestionService()
kb = await RAGKnowledgeBase.from_config(config)

# Ensure populated (skips if already has documents)
result = await service.ensure_ingested(kb, {
    "enabled": True,
    "documents_path": "path/to/docs",
})

if result.skipped:
    print(f"Skipped: {result.reason}")
else:
    print(f"Ingested {result.total_chunks} chunks")
```

### Using with Registry Managers

Use the `AutoIngestionMixin` to add auto-ingestion to bot managers:

```python
from dataknobs_bots.registry import CachingRegistryManager
from dataknobs_bots.knowledge import AutoIngestionMixin, get_ingestion_service

class MyBotManager(CachingRegistryManager[MyBot], AutoIngestionMixin):
    def __init__(self, auto_ingest: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._auto_ingest = auto_ingest
        self._ingestion_service = get_ingestion_service()

    async def register(self, domain_id, config, ingest=None):
        await super().register(domain_id, config)
        should_ingest = ingest if ingest is not None else self._auto_ingest
        if should_ingest:
            await self._ensure_knowledge_base_ingested(domain_id, config)
```

### Result Types

- **`IngestionResult`**: Lower-level, for file-backend-to-vector-store coordination
- **`EnsureIngestionResult`**: Higher-level, with `skipped`/`reason` for skip-if-populated semantics

## Exported Types

```python
from dataknobs_bots.knowledge import (
    # Main class
    RAGKnowledgeBase,

    # Ingestion types
    DirectoryProcessor,
    FilePatternConfig,
    KnowledgeBaseConfig,
    ProcessedDocument,

    # Low-level ingestion (file-backend to vector-store)
    KnowledgeIngestionManager,
    IngestionResult,

    # High-level ingestion service
    KnowledgeIngestionService,
    EnsureIngestionResult,
    get_ingestion_service,
    ensure_knowledge_base_ingested,
    AutoIngestionMixin,

    # Hybrid search types
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
)
```

## Related

- [RAG Retrieval](rag-retrieval.md) - Chunk merging and formatting
- [RAG Query](rag-query.md) - Query transformation and expansion
- [User Guide](user-guide.md) - Complete tutorials
