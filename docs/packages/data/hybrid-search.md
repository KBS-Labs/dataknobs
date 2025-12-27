# Hybrid Search

Hybrid search combines traditional keyword (text) search with semantic vector search to improve retrieval quality.

## Overview

- **Text Search**: BM25/TF-IDF keyword matching
- **Vector Search**: Semantic similarity via embeddings
- **Fusion**: Combine results using RRF or weighted scoring

## Quick Start

```python
from dataknobs_data.vector.hybrid import HybridSearchConfig, FusionStrategy

# Configure hybrid search
config = HybridSearchConfig(
    text_weight=0.4,
    vector_weight=0.6,
    fusion_strategy=FusionStrategy.RRF,
)

# Perform hybrid search
results = await backend.hybrid_search(
    query_text="machine learning classification",
    query_vector=embedding,
    text_fields=["title", "content"],
    k=10,
    config=config,
)

for result in results:
    print(f"Score: {result.combined_score:.4f}")
    print(f"  Text: {result.text_score}, Vector: {result.vector_score}")
```

## Configuration

### HybridSearchConfig

```python
from dataknobs_data.vector.hybrid import HybridSearchConfig, FusionStrategy

config = HybridSearchConfig(
    text_weight=0.5,       # Weight for text search (0-1)
    vector_weight=0.5,     # Weight for vector search (0-1)
    fusion_strategy=FusionStrategy.RRF,  # RRF, WEIGHTED_SUM, NATIVE
    rrf_k=60,              # RRF smoothing constant
    text_fields=["title", "content"],  # Fields to search
)
```

## Fusion Strategies

### RRF (Reciprocal Rank Fusion)

Default strategy. Combines based on rank position:

```
RRF_score(d) = sum(weight / (k + rank))
```

**Best for**: General-purpose hybrid search, different score distributions.

### Weighted Sum

Combines normalized scores directly:

```
combined = text_weight * norm(text_score) + vector_weight * norm(vector_score)
```

**Best for**: Comparable scores, fine-tuned weight control.

### Native

Uses backend's native hybrid search if available.

## Fusion Functions

```python
from dataknobs_data.vector.hybrid import (
    reciprocal_rank_fusion,
    weighted_score_fusion,
)

# RRF fusion
fused = reciprocal_rank_fusion(text_results, vector_results, k=60)

# Weighted sum fusion
fused = weighted_score_fusion(
    text_results, vector_results,
    text_weight=0.4, vector_weight=0.6,
)
```

## Backend Support

### Elasticsearch

Native RRF retriever (Elasticsearch 8.8+):

```python
results = await es_backend.hybrid_search(
    query_text="machine learning",
    query_vector=embedding,
    config=HybridSearchConfig(fusion_strategy=FusionStrategy.NATIVE),
)
```

### PostgreSQL

Uses tsvector + pgvector:

```python
results = await pg_backend.hybrid_search(
    query_text="database optimization",
    query_vector=embedding,
)
```

### Client-Side Fallback

Any VectorOperationsMixin backend automatically uses client-side fusion when native isn't available.

## API Reference

### HybridSearchResult

```python
@dataclass
class HybridSearchResult:
    record: Record              # Matched record
    combined_score: float       # Final fused score
    text_score: float | None    # Text match score
    vector_score: float | None  # Vector similarity
    text_rank: int | None       # Rank in text results
    vector_rank: int | None     # Rank in vector results
    metadata: dict[str, Any]    # Additional metadata
```

### VectorOperationsMixin.hybrid_search

```python
async def hybrid_search(
    self,
    query_text: str,
    query_vector: np.ndarray | list[float],
    text_fields: list[str] | None = None,
    vector_field: str = "embedding",
    k: int = 10,
    config: HybridSearchConfig | None = None,
    filter: Query | None = None,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> list[HybridSearchResult]: ...
```

## Use Cases

### RAG Knowledge Base

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase

results = await kb.hybrid_query(
    "how to configure authentication",
    k=5,
    text_weight=0.4,
    vector_weight=0.6,
)
```

### Search Application

```python
async def search(query: str, k: int = 10):
    embedding = await embed(query)
    return await backend.hybrid_search(
        query_text=query,
        query_vector=embedding,
        k=k,
    )
```

## Related

- [Backends Overview](backends.md) - Available backends
- [Elasticsearch Backend](elasticsearch-backend.md) - ES configuration
- [PostgreSQL Backend](postgres-backend.md) - PG configuration
