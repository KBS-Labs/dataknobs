# Content Deduplication

The `DedupChecker` provides content uniqueness checking by combining exact hash matching with optional semantic similarity via vector stores.

## Overview

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig

db = AsyncMemoryDatabase()
checker = DedupChecker(db=db, config=DedupConfig(hash_fields=["stem"]))

# Register existing content
await checker.register({"stem": "What is 2+2?"}, record_id="q-1")

# Check for duplicates
result = await checker.check({"stem": "What is 2+2?"})
assert result.is_exact_duplicate is True
assert result.exact_match_id == "q-1"

# New content is unique
result = await checker.check({"stem": "What is 3+3?"})
assert result.is_exact_duplicate is False
```

## DedupConfig

Configuration for dedup checking behavior.

```python
from dataknobs_data.dedup import DedupConfig

config = DedupConfig(
    hash_fields=["stem", "answer"],
    hash_algorithm="md5",
    semantic_check=False,
    semantic_fields=None,
    similarity_threshold=0.92,
    max_similar_results=5,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hash_fields` | `list[str]` | `["content"]` | Field names used for computing the content hash |
| `hash_algorithm` | `str` | `"md5"` | Hash algorithm (`"md5"` or `"sha256"`) |
| `semantic_check` | `bool` | `False` | Enable semantic similarity search |
| `semantic_fields` | `list[str] \| None` | `None` | Fields for embedding (defaults to `hash_fields`) |
| `similarity_threshold` | `float` | `0.92` | Minimum similarity score for a match |
| `max_similar_results` | `int` | `5` | Maximum similar items to return |

## DedupChecker

### Creating a Checker

The checker requires an `AsyncDatabase` for hash storage:

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig

checker = DedupChecker(
    db=AsyncMemoryDatabase(),
    config=DedupConfig(hash_fields=["stem"]),
)
```

Any `AsyncDatabase` backend works — use `AsyncMemoryDatabase` for in-session dedup, or a persistent backend (SQLite, PostgreSQL, etc.) for cross-session dedup.

### Registering Content

Register content to make it available for future duplicate checks:

```python
await checker.register(
    content={"stem": "What is photosynthesis?", "answer": "..."},
    record_id="q-42",
)
```

This stores:
- A hash record in the database (content hash → record ID)
- Optionally, an embedding in the vector store (if semantic check is enabled)

### Checking for Duplicates

```python
result = await checker.check({"stem": "What is photosynthesis?"})
```

The check proceeds in two steps:

1. **Exact hash match** — Computes a hash from the configured `hash_fields` and looks for a matching record in the database
2. **Semantic similarity** (optional) — If `semantic_check` is enabled and no exact match is found, searches the vector store for similar content

### Computing Hashes

The hash is computed deterministically from configured fields:

```python
content_hash = checker.compute_hash({"stem": "What is 2+2?", "answer": "4"})
```

Fields are joined with `|` to avoid collisions (e.g., `("a b", "c")` vs `("a", "b c")`). Missing fields are treated as empty strings.

### Accessing Config

```python
config = checker.config  # Returns the DedupConfig
```

## DedupResult

Returned by `check()`:

```python
from dataknobs_data.dedup import DedupResult

result = await checker.check(content)

if result.is_exact_duplicate:
    print(f"Exact match: {result.exact_match_id}")
elif result.similar_items:
    print(f"Found {len(result.similar_items)} similar items")
else:
    print("Content is unique")
```

| Field | Type | Description |
|-------|------|-------------|
| `is_exact_duplicate` | `bool` | Whether an exact hash match was found |
| `exact_match_id` | `str \| None` | Record ID of the exact match |
| `similar_items` | `list[SimilarItem]` | Semantically similar items (if semantic check enabled) |
| `recommendation` | `str` | One of `"unique"`, `"possible_duplicate"`, or `"exact_duplicate"` |
| `content_hash` | `str` | The computed hash of the checked content |

## SimilarItem

Represents a semantically similar record found during semantic dedup:

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `str` | The ID of the similar record |
| `score` | `float` | Similarity score (higher is more similar) |
| `matched_text` | `str` | The text that was matched against |

## Semantic Similarity

To enable semantic dedup, provide a vector store and embedding function:

```python
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.dedup import DedupChecker, DedupConfig

async def embed(text: str) -> list[float]:
    # Your embedding function
    ...

vector_store = MemoryVectorStore(dimensions=384)
await vector_store.initialize()

checker = DedupChecker(
    db=AsyncMemoryDatabase(),
    config=DedupConfig(
        hash_fields=["stem"],
        semantic_check=True,
        semantic_fields=["stem"],  # Fields to embed (defaults to hash_fields)
        similarity_threshold=0.92,
        max_similar_results=5,
    ),
    vector_store=vector_store,
    embedding_fn=embed,
)
```

With semantic checking enabled:
- `register()` stores both a hash record and an embedding vector
- `check()` first checks for exact hash match, then searches for semantically similar content above the threshold

## Integration with ArtifactCorpus

`DedupChecker` integrates with `ArtifactCorpus` from the `dataknobs-bots` package for collection-level dedup:

```python
from dataknobs_bots.artifacts import ArtifactCorpus
from dataknobs_bots.artifacts.corpus import CorpusConfig

corpus = await ArtifactCorpus.create(
    registry=registry,
    config=CorpusConfig(
        corpus_type="quiz_bank",
        item_type="quiz_question",
        name="Chapter 1 Quiz",
    ),
    dedup_checker=checker,
)

# Items are automatically checked and registered
artifact, result = await corpus.add_item(content={"stem": "What is 2+2?"})

# Pre-screen without adding
result = await corpus.check_dedup({"stem": "What is 2+2?"})
```

When a corpus is created with a dedup checker, the dedup configuration is stored in the corpus artifact content. `ArtifactCorpus.load()` reconstructs the checker and re-registers existing items so dedup works across session reloads.

See [Artifact Corpus](../bots/guides/artifact-corpus.md) for full documentation.

## Serialization

`DedupConfig` is a dataclass and can be serialized with `dataclasses.asdict()`:

```python
import dataclasses

config = DedupConfig(hash_fields=["stem"], hash_algorithm="sha256")
config_dict = dataclasses.asdict(config)
# {"hash_fields": ["stem"], "hash_algorithm": "sha256", ...}

# Reconstruct
restored = DedupConfig(**config_dict)
```

This is used internally by `ArtifactCorpus` to persist dedup configuration in the corpus artifact.

## Database Backend Considerations

The dedup checker stores hash records using `AsyncDatabase.create()` and `AsyncDatabase.search()`. Any backend works:

| Backend | Use Case |
|---------|----------|
| `AsyncMemoryDatabase` | In-session dedup (data lost on restart) |
| `AsyncSQLiteDatabase` | Persistent local dedup |
| `AsyncPostgresDatabase` | Shared/production dedup across services |

For cross-session dedup without `ArtifactCorpus.load()`, use a persistent backend. For in-session dedup (most common with `ArtifactCorpus`), `AsyncMemoryDatabase` is sufficient since `load()` re-registers items automatically.
