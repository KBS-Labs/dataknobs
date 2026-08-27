# Artifact Corpus

An `ArtifactCorpus` manages a collection of related artifacts (e.g., a quiz bank is a corpus of quiz questions). It provides corpus-level operations: add items with optional dedup, query items, get summaries, and finalize.

The corpus itself is stored as an artifact in the `ArtifactRegistry`, and child items link to it via `content["corpus_id"]`.

## Overview

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_bots.artifacts import ArtifactCorpus, ArtifactRegistry
from dataknobs_bots.artifacts.corpus import CorpusConfig

db = AsyncMemoryDatabase()
registry = ArtifactRegistry(db=db)

corpus = await ArtifactCorpus.create(
    registry=registry,
    config=CorpusConfig(
        corpus_type="quiz_bank",
        item_type="quiz_question",
        name="Chapter 1 Quiz",
    ),
)

artifact, dedup_result = await corpus.add_item(
    content={"stem": "What is 2+2?", "answer": "4"},
    tags=["math"],
)

print(corpus.id)          # "art_..."
print(artifact.id)        # "art_..."
print(await corpus.count())  # 1
```

## CorpusConfig

Configuration for creating a corpus.

```python
from dataknobs_bots.artifacts.corpus import CorpusConfig

config = CorpusConfig(
    corpus_type="quiz_bank",
    item_type="quiz_question",
    name="Chapter 1 Quiz",
    rubric_ids=["content_quality"],
    auto_review=False,
    dedup_config=None,
    metadata={"course": "MATH101"},
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `corpus_type` | `str` | required | Artifact type for the parent corpus (e.g., `"quiz_bank"`) |
| `item_type` | `str` | required | Artifact type for child items (e.g., `"quiz_question"`) |
| `name` | `str` | required | Human-readable corpus name |
| `rubric_ids` | `list[str]` | `[]` | Rubric IDs to apply to items |
| `auto_review` | `bool` | `False` | Automatically submit items for review on creation |
| `dedup_config` | `DedupConfig \| None` | `None` | Optional dedup configuration |
| `metadata` | `dict[str, Any]` | `{}` | Additional metadata stored on the corpus artifact |

## Creating a Corpus

Use `ArtifactCorpus.create()` to start a new corpus:

```python
corpus = await ArtifactCorpus.create(
    registry=registry,
    config=CorpusConfig(
        corpus_type="quiz_bank",
        item_type="quiz_question",
        name="Chapter 1 Quiz",
    ),
)
```

This creates a parent artifact in the registry with:
- Type set to `corpus_type`
- Name set to `config.name`
- Tag `"corpus"` applied automatically
- Content includes `item_type` and `metadata`

### With Dedup

To enable duplicate detection, pass a `DedupChecker`:

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig

dedup_checker = DedupChecker(
    db=AsyncMemoryDatabase(),
    config=DedupConfig(hash_fields=["stem"]),
)

corpus = await ArtifactCorpus.create(
    registry=registry,
    config=config,
    dedup_checker=dedup_checker,
)
```

When a dedup checker is provided, the dedup configuration is stored in the corpus artifact's content so it can be reconstructed on `load()`.

## Loading an Existing Corpus

Use `ArtifactCorpus.load()` to resume an existing corpus by ID:

```python
loaded = await ArtifactCorpus.load(registry, corpus_id="art_abc123")
```

`load()` reconstructs the full corpus state:

1. Loads the parent corpus artifact from the registry
2. Rebuilds `CorpusConfig` from stored artifact content
3. If the corpus was created with dedup, reconstructs a `DedupChecker` from the stored `dedup_config` with a fresh `AsyncMemoryDatabase`
4. Re-registers all existing items with the dedup checker so hash-based duplicate detection works across session reloads

!!! note "Semantic dedup not restored"
    Semantic dedup (vector store + embedding function) requires runtime infrastructure that cannot be serialized. Only hash-based exact matching is restored by `load()`.

## Adding Items

```python
artifact, dedup_result = await corpus.add_item(
    content={"stem": "What is 2+2?", "answer": "4"},
    tags=["math"],
)
```

Each item artifact is created with:
- Type set to the corpus `item_type`
- `content["corpus_id"]` set to the corpus ID (links item to corpus)
- Tags from the `tags` parameter

The return value is a tuple of `(Artifact, DedupResult | None)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `dict[str, Any]` | required | Item content dictionary |
| `provenance` | `ProvenanceRecord \| None` | `None` | Optional provenance record |
| `tags` | `list[str] \| None` | `None` | Tags for the item |
| `skip_dedup` | `bool` | `False` | Skip dedup checking even if configured |

### Dedup Behavior

When a dedup checker is configured:

- **New content**: Returns the new artifact and a `DedupResult` with `is_exact_duplicate=False`
- **Exact duplicate**: Returns the *existing* artifact (no new artifact created) and a `DedupResult` with `is_exact_duplicate=True`

```python
# First add — new
a1, result1 = await corpus.add_item(content={"stem": "What is 2+2?"})
assert result1.is_exact_duplicate is False

# Second add with same content — returns existing
a2, result2 = await corpus.add_item(content={"stem": "What is 2+2?"})
assert result2.is_exact_duplicate is True
assert a2.id == a1.id  # Same artifact returned
assert await corpus.count() == 1  # Only one item
```

## Pre-Screening with check_dedup

Check content for duplicates without adding it to the corpus:

```python
result = await corpus.check_dedup({"stem": "What is 2+2?"})
if result and result.is_exact_duplicate:
    print("Already exists!")
```

Returns `None` if no dedup checker is configured. This is useful for pre-screening content before showing it to a user or before committing to add it.

## Querying Items

### get_items

```python
# All items
items = await corpus.get_items()

# Filtered by status
draft_items = await corpus.get_items(status=ArtifactStatus.DRAFT)
approved_items = await corpus.get_items(status=ArtifactStatus.APPROVED)
```

Items are scoped to their corpus — each corpus only returns its own items.

### count

```python
total = await corpus.count()
draft_count = await corpus.count(status=ArtifactStatus.DRAFT)
```

## Removing Items

```python
await corpus.remove_item(artifact_id)
```

Sets the item's status to `ARCHIVED`. The item remains in storage but won't appear in status-filtered queries for `DRAFT`.

## Finalizing a Corpus

```python
finalized = await corpus.finalize()
```

Finalizing:
1. Collects all item IDs and count
2. Updates the corpus artifact content with `item_ids`, `item_count`, and `finalized: True`
3. Transitions the corpus artifact through `DRAFT` → `PENDING_REVIEW` → `IN_REVIEW` → `APPROVED`

```python
assert finalized.status == ArtifactStatus.APPROVED
assert finalized.content["item_count"] == 3
assert finalized.content["finalized"] is True
```

## Getting a Summary

```python
summary = await corpus.get_summary()
```

Returns a dictionary with corpus metadata and item status breakdown:

```python
{
    "corpus_id": "art_...",
    "corpus_name": "Chapter 1 Quiz",
    "corpus_type": "quiz_bank",
    "item_type": "quiz_question",
    "total_items": 5,
    "status_breakdown": {"draft": 3, "approved": 2},
    "corpus_status": "draft",
    "metadata": {},
}
```

## Wizard Transforms

Three pre-built transform functions integrate corpus operations into wizard workflows. Each operates on a wizard data dict and uses a `TransformContext`.

### create_corpus

Creates a new corpus and stores references in wizard data.

```python
from dataknobs_bots.artifacts import create_corpus, TransformContext

context = TransformContext(artifact_registry=registry)
data = {"topic": "Chapter 1 Quiz"}

await create_corpus(data, context, config={
    "corpus_type": "quiz_bank",
    "item_type": "quiz_question",
    "name_field": "topic",
})
```

Sets in `data`:
- `_corpus_id` — the corpus artifact ID
- `_corpus` — the `ArtifactCorpus` instance
- `_corpus_item_count` — initialized to 0

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `corpus_type` | `str` | `"corpus"` | Artifact type for the corpus |
| `item_type` | `str` | `"item"` | Artifact type for items |
| `name_template` | `str` | — | Jinja2 template for corpus name |
| `name_field` | `str` | `"name"` | Fallback: key in `data` for name |
| `rubric_ids` | `list[str]` | `[]` | Rubrics to apply to items |
| `auto_review` | `bool` | `False` | Auto-evaluate items on add |
| `dedup` | `dict` | — | DedupConfig settings (see below) |

**Dedup config keys:** `hash_fields`, `hash_algorithm`, `semantic_check`, `similarity_threshold`.

```yaml
# Example wizard YAML config
transforms:
  - create_corpus:
      corpus_type: quiz_bank
      item_type: quiz_question
      name_field: topic
      dedup:
        hash_fields: [stem]
```

### add_to_corpus

Adds content from wizard data to the corpus.

```python
from dataknobs_bots.artifacts import add_to_corpus

data["_current_item"] = {"stem": "What is 2+2?", "answer": "4"}
await add_to_corpus(data, context, config={
    "content_key": "_current_item",
    "tags": ["math"],
})
```

Sets in `data`:
- `_last_added_artifact_id` — the new item's artifact ID
- `_corpus_item_count` — updated count
- `_dedup_result` — dict with dedup info, or `None`

If the `_corpus` key is missing from data (e.g., after a session reload), the transform automatically reconstructs the corpus from `_corpus_id` using `ArtifactCorpus.load()`.

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `content_key` | `str` | `"_current_item"` | Key in `data` holding content dict |
| `corpus_key` | `str` | `"_corpus"` | Key in `data` holding the corpus |
| `tags` | `list[str]` | `[]` | Additional tags for the item |

### finalize_corpus

Finalizes the corpus and stores the summary.

```python
from dataknobs_bots.artifacts import finalize_corpus

await finalize_corpus(data, context)
```

Sets `data["_corpus_summary"]` with the result of `corpus.get_summary()`.

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `corpus_key` | `str` | `"_corpus"` | Key in `data` holding the corpus |

## Complete Example

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_bots.artifacts import (
    ArtifactCorpus,
    ArtifactRegistry,
    ArtifactStatus,
    TransformContext,
    create_corpus,
    add_to_corpus,
    finalize_corpus,
)

# Setup
db = AsyncMemoryDatabase()
registry = ArtifactRegistry(db=db)
context = TransformContext(artifact_registry=registry)

# Create corpus via transform
data = {"topic": "Biology Quiz"}
await create_corpus(data, context, config={
    "corpus_type": "quiz_bank",
    "item_type": "quiz_question",
    "name_field": "topic",
    "dedup": {"hash_fields": ["stem"]},
})

# Add items
for q in ["What is DNA?", "Explain mitosis.", "What is DNA?"]:
    data["_current_item"] = {"stem": q}
    await add_to_corpus(data, context)

# Third item was a duplicate — only 2 unique items
assert data["_corpus_item_count"] == 2

# Finalize
await finalize_corpus(data, context)
assert data["_corpus_summary"]["total_items"] == 2
assert data["_corpus_summary"]["corpus_status"] == "approved"
```

## Related Documentation

- [Artifact System](artifacts.md) — Core artifact models, registry, and lifecycle
- [Rubric Evaluation System](rubrics.md) — Evaluating artifacts with structured rubrics
- [Content Deduplication](../../data/dedup.md) — DedupChecker for hash and semantic dedup
