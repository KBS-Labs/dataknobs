# Hybrid Search

Hybrid search combines traditional keyword (text) search with semantic vector search to improve retrieval quality. This module provides types, fusion algorithms, and backend implementations for hybrid search operations.

## Overview

Pure vector search excels at semantic similarity but can miss exact keyword matches. Pure text search finds exact matches but misses semantically similar content. Hybrid search combines both approaches:

1. **Text Search**: BM25/TF-IDF keyword matching
2. **Vector Search**: Semantic similarity via embeddings
3. **Fusion**: Combine results using RRF or weighted scoring

## Key Features

- **Multiple fusion strategies**: RRF, Weighted Sum, or Native backend fusion
- **Configurable weights**: Balance text vs. vector importance
- **Backend-native support**: Elasticsearch RRF, PostgreSQL tsvector+pgvector
- **Client-side fallback**: Works with any vector store

## Installation

```bash
uv pip install dataknobs-data
```

## Quick Start

### Using VectorOperationsMixin

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
    print(f"  Record: {result.record.id}")
```

### Fusion Algorithms

```python
from dataknobs_data.vector.hybrid import (
    reciprocal_rank_fusion,
    weighted_score_fusion,
)

# Results from text search
text_results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]

# Results from vector search
vector_results = [("doc2", 0.95), ("doc1", 0.8), ("doc4", 0.6)]

# RRF fusion
fused = reciprocal_rank_fusion(text_results, vector_results, k=60)
# [("doc2", 0.033), ("doc1", 0.032), ("doc3", 0.016), ("doc4", 0.016)]

# Weighted sum fusion
fused = weighted_score_fusion(
    text_results, vector_results,
    text_weight=0.4, vector_weight=0.6,
    normalize_scores=True,
)
```

## Configuration

### HybridSearchConfig

```python
from dataknobs_data.vector.hybrid import HybridSearchConfig, FusionStrategy

config = HybridSearchConfig(
    # Weights (0.0 to 1.0)
    text_weight=0.5,      # Weight for text search scores
    vector_weight=0.5,    # Weight for vector search scores

    # Fusion strategy
    fusion_strategy=FusionStrategy.RRF,  # RRF, WEIGHTED_SUM, or NATIVE

    # RRF parameters
    rrf_k=60,             # RRF constant (higher = smoother ranking)

    # Text search
    text_fields=["title", "content"],  # Fields to search (None = all)
)

# Normalize weights for weighted sum
text_w, vector_w = config.normalize_weights()
# Ensures weights sum to 1.0
```

### Fusion Strategies

#### RRF (Reciprocal Rank Fusion)

Default strategy. Combines based on rank position, not scores:

```
RRF_score(d) = sum(weight / (k + rank)) for each ranking
```

**Advantages:**
- Works without score normalization
- Robust to different score distributions
- Good default choice

```python
config = HybridSearchConfig(
    fusion_strategy=FusionStrategy.RRF,
    rrf_k=60,  # Standard value
)
```

#### Weighted Sum

Combines normalized scores directly:

```
combined = text_weight * norm(text_score) + vector_weight * norm(vector_score)
```

**Use when:**
- Scores are comparable
- You want direct score combination

```python
config = HybridSearchConfig(
    fusion_strategy=FusionStrategy.WEIGHTED_SUM,
    text_weight=0.3,
    vector_weight=0.7,
)
```

#### Native

Uses backend's native hybrid search implementation:

```python
config = HybridSearchConfig(
    fusion_strategy=FusionStrategy.NATIVE,
)
# Falls back to RRF if backend doesn't support native
```

## API Reference

### HybridSearchConfig

```python
@dataclass
class HybridSearchConfig:
    text_weight: float = 0.5          # Weight for text search (0-1)
    vector_weight: float = 0.5        # Weight for vector search (0-1)
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60                   # RRF smoothing constant
    text_fields: list[str] | None = None  # Fields for text search

    def normalize_weights(self) -> tuple[float, float]:
        """Return weights normalized to sum to 1.0."""
```

### HybridSearchResult

```python
@dataclass
class HybridSearchResult:
    record: Record              # The matched record
    combined_score: float       # Final fused score
    text_score: float | None    # Score from text search
    vector_score: float | None  # Score from vector search
    text_rank: int | None       # Rank in text results
    vector_rank: int | None     # Rank in vector results
    metadata: dict[str, Any]    # Additional metadata
```

### Fusion Functions

```python
def reciprocal_rank_fusion(
    text_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    k: int = 60,
    text_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Combine results using RRF algorithm."""

def weighted_score_fusion(
    text_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    text_weight: float = 0.5,
    vector_weight: float = 0.5,
    normalize_scores: bool = True,
) -> list[tuple[str, float]]:
    """Combine results using weighted score sum."""
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
) -> list[HybridSearchResult]:
    """Perform hybrid search combining text and vector similarity."""
```

## Backend Implementations

### Elasticsearch

Uses native RRF retriever (Elasticsearch 8.8+):

```python
from dataknobs_data.backends import ElasticsearchAsyncBackend

backend = ElasticsearchAsyncBackend(...)
await backend.connect()

# Native hybrid search using RRF retriever
results = await backend.hybrid_search(
    query_text="machine learning",
    query_vector=embedding,
    text_fields=["title", "content"],
    k=10,
    config=HybridSearchConfig(fusion_strategy=FusionStrategy.NATIVE),
)
```

**Implementation:**
- Text: BM25 multi_match query
- Vector: kNN search on dense_vector field
- Fusion: Native RRF retriever

### PostgreSQL

Uses tsvector + pgvector combination:

```python
from dataknobs_data.backends import PostgresAsyncBackend

backend = PostgresAsyncBackend(...)
await backend.connect()

# Hybrid search using tsvector and pgvector
results = await backend.hybrid_search(
    query_text="database optimization",
    query_vector=embedding,
    text_fields=["title", "description"],
    k=10,
)
```

**Implementation:**
- Text: PostgreSQL full-text search with ts_rank_cd
- Vector: pgvector cosine similarity
- Fusion: SQL-based RRF in single query

### Client-Side Fallback

For backends without native hybrid search:

```python
# Any VectorOperationsMixin backend
results = await backend.hybrid_search(
    query_text="search query",
    query_vector=embedding,
    k=10,
    # Automatically uses client-side fusion
)
```

**Implementation:**
1. Text search using LIKE queries
2. Vector search using existing `vector_search()`
3. Client-side RRF/weighted fusion

## Use Cases

### RAG Knowledge Base

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase

kb = await RAGKnowledgeBase.from_config(config)
await kb.load_from_directory("./docs")

# Hybrid query
results = await kb.hybrid_query(
    "how to configure authentication",
    k=5,
    text_weight=0.4,
    vector_weight=0.6,
    fusion_strategy="rrf",
)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['heading_path']}")
    print(f"  text={r['text_score']:.3f}, vector={r['vector_score']:.3f}")
```

### Search Application

```python
async def search(query: str, k: int = 10):
    # Generate embedding
    embedding = await embed(query)

    # Hybrid search
    results = await backend.hybrid_search(
        query_text=query,
        query_vector=embedding,
        text_fields=["title", "content", "tags"],
        k=k,
        config=HybridSearchConfig(
            text_weight=0.5,
            vector_weight=0.5,
        ),
    )

    return [
        {
            "id": r.record.id,
            "score": r.combined_score,
            "title": r.record.data.get("title"),
        }
        for r in results
    ]
```

### Tuning Weights

```python
# Exact keyword matching important (e.g., product codes)
config = HybridSearchConfig(text_weight=0.7, vector_weight=0.3)

# Semantic understanding important (e.g., natural language)
config = HybridSearchConfig(text_weight=0.3, vector_weight=0.7)

# Balanced (default)
config = HybridSearchConfig(text_weight=0.5, vector_weight=0.5)
```

## Testing

```bash
cd packages/data
uv run pytest tests/test_hybrid_search.py -v
```

## Related

- [Vector Search](VECTOR_GETTING_STARTED.md) - Pure vector search
- [Query Operations](BOOLEAN_LOGIC_OPERATORS.md) - Query filtering
- [Elasticsearch Backend](../history/vector-implementation/VECTOR_IMPLEMENTATION_SUMMARY.md) - ES implementation details
