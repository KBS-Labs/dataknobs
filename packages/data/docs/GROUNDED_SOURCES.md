# Grounded Sources

The grounded source abstraction provides a uniform interface for queryable data sources in retrieval pipelines. Sources receive structured intent and translate it deterministically to native queries — the LLM never generates query syntax.

## Overview

Grounded sources solve the problem of querying heterogeneous data backends (vector stores, SQL databases, Elasticsearch, etc.) through a single retrieval pipeline. Each source:

- **Declares a schema** of queryable dimensions (optional)
- **Receives structured intent** (`RetrievalIntent`) from an intent resolution layer
- **Translates intent to native queries** in deterministic code
- **Returns normalized results** (`SourceResult`) regardless of backing store

This module lives in `dataknobs-data` (not `dataknobs-bots`) so any project using the data layer can define and query sources without depending on the LLM or bots packages.

## Quick Start

```python
from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
)

# Create an intent (typically produced by an LLM or config)
intent = RetrievalIntent(
    text_queries=["OAuth grant types", "authorization code flow"],
    scope="focused",
)

# Query a source
results = await source.query(intent, top_k=5, score_threshold=0.3)
for result in results:
    print(f"[{result.relevance:.2f}] {result.source_name}: {result.content[:80]}")
```

## Core Types

### RetrievalIntent

Source-agnostic structured intent for retrieval.

```python
from dataknobs_data.sources.base import RetrievalIntent

intent = RetrievalIntent(
    text_queries=["search phrase 1", "search phrase 2"],
    filters={
        "source_name": {"field": "value"},  # Keyed by source name
    },
    scope="focused",       # Retrieval breadth hint
    raw_data={},           # Full extraction dict for provenance
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `text_queries` | `list[str]` | Semantic search phrases. Always present. |
| `filters` | `dict[str, Any]` | Structured conditions keyed by source name. |
| `scope` | `str` | Retrieval breadth hint (e.g., `"focused"`, `"broad"`). |
| `raw_data` | `dict[str, Any]` | Full extraction dict, preserved for provenance. |

### SourceResult

Normalized result that all sources produce.

```python
from dataknobs_data.sources.base import SourceResult

result = SourceResult(
    content="The authorization code grant type is used to...",
    source_id="chunk_42",
    source_name="knowledge_base",
    source_type="vector_kb",
    relevance=0.92,
    metadata={"heading_path": "Section 4.1 > Authorization Code"},
)

# Convert to dict for compatibility with existing formatters
result_dict = result.to_dict()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `str` | Text content for inclusion in synthesis prompts. |
| `source_id` | `str` | Unique ID within the source (chunk ID, primary key, etc.). |
| `source_name` | `str` | Which `GroundedSource` produced this result. |
| `source_type` | `str` | Category string (`"vector_kb"`, `"database"`, etc.). |
| `relevance` | `float` | Score from 0.0 to 1.0. Default 1.0. |
| `metadata` | `dict[str, Any]` | Source-specific metadata (heading paths, field values, etc.). |

### SourceSchema

Schema fragment a source declares for intent extraction.

```python
from dataknobs_data.sources.base import SourceSchema

schema = SourceSchema(
    source_name="case_db",
    fields={
        "category": {
            "type": "string",
            "enum": ["security", "compliance", "operations"],
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
    },
    required_fields=["category"],
    description="Case study database with security incident reports",
)
```

## GroundedSource ABC

All sources implement the `GroundedSource` abstract base class:

```python
from dataknobs_data.sources.base import GroundedSource, RetrievalIntent, SourceResult

class MySource(GroundedSource):
    @property
    def name(self) -> str:
        return "my_source"

    @property
    def source_type(self) -> str:
        return "custom"

    def get_schema(self) -> SourceSchema | None:
        # Return None for text-only sources (default)
        # Return SourceSchema to declare filter dimensions
        return None

    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        # Translate intent to native query — deterministic code, not LLM
        results = []
        for query_text in intent.text_queries:
            # Execute against your backing store
            hits = await self._search(query_text, limit=top_k)
            results.extend([
                SourceResult(
                    content=hit.text,
                    source_id=str(hit.id),
                    source_name=self.name,
                    source_type=self.source_type,
                    relevance=hit.score,
                )
                for hit in hits
                if hit.score >= score_threshold
            ])
        return sorted(results, key=lambda r: r.relevance, reverse=True)[:top_k]

    async def close(self) -> None:
        # Release resources (optional, default no-op)
        pass
```

## Built-in Implementations

### VectorKnowledgeSource (in dataknobs-bots)

`VectorKnowledgeSource` wraps an existing `RAGKnowledgeBase` as a `GroundedSource`. It lives in `dataknobs-bots` (not `dataknobs-data`) because it depends on the bot-layer `KnowledgeBase` class.

```python
from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

source = VectorKnowledgeSource(knowledge_base)
results = await source.query(intent, top_k=5, score_threshold=0.3)
```

This is used automatically by the `GroundedReasoning` strategy when a bot has a configured knowledge base.

### DatabaseSource

Wraps any `AsyncDatabase` backend with text search across configured fields.

```python
from dataknobs_data.sources.database import DatabaseSource
from dataknobs_data.backends.memory import AsyncMemoryDatabase

db = AsyncMemoryDatabase()
source = DatabaseSource(
    db=db,
    name="case_studies",
    content_field="summary",
    text_search_fields=["title", "summary", "tags"],
)
results = await source.query(intent)
```

For each text query, `DatabaseSource` searches all configured `text_search_fields` for matching records. Structured filters from `intent.filters[source_name]` are also applied when present.

#### Relevance Scoring

`DatabaseSource` computes a term-coverage relevance score for each result rather than returning a flat 1.0. The score reflects what fraction of the query terms appear in the record's searchable fields:

- The **content field** receives 2x weight (it's the primary field users care about)
- **Secondary text search fields** receive 1x weight each
- Score = matched_weight / total_weight, with a minimum floor of 0.05

This makes `score_threshold` meaningful for database sources — a record matching only 1 of 3 query terms in a secondary field scores lower than one matching all terms in the content field. Results are returned sorted by relevance descending.

## Multi-Source Retrieval

When used with the `GroundedReasoning` strategy in `dataknobs-bots`, multiple sources are queried in parallel and results are merged via **weighted round-robin**. Each source has a configurable `weight` (default 1) that determines how many results it contributes per round-robin cycle. Results are deduplicated by `(source_name, source_id)` when enabled.

```python
# In GroundedReasoning, sources are queried and merged:
results_by_source = await strategy._retrieve_from_sources(intent)
merged = strategy._merge_source_results(results_by_source)
```

## Testing

Use `AsyncMemoryDatabase` for `DatabaseSource` tests and mock knowledge bases for `VectorKnowledgeSource` tests:

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.sources.database import DatabaseSource
from dataknobs_data.sources.base import RetrievalIntent

db = AsyncMemoryDatabase()
await db.create(Record({"title": "OAuth Overview", "summary": "OAuth 2.0 is..."}))

source = DatabaseSource(db=db, name="docs", content_field="summary",
                        text_search_fields=["title", "summary"])

intent = RetrievalIntent(text_queries=["OAuth"])
results = await source.query(intent)
assert len(results) > 0
```
