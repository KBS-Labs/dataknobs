# Vector Store Metadata Filter Semantics

`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`, and
`PgVectorStore` all accept a `filter: dict[str, Any] | None` argument
on `search()` and `count()`. Per-key match semantics are identical
across backends — consumers can runtime-swap the backing store without
behavioral surprises.

## Four-quadrant match table

A filter is a mapping of metadata key to filter value. Each filter
key is checked against the same key in the record's metadata. All
keys must match (AND across keys). Empty filter dict matches every
record. A key missing from the record's metadata fails the filter.

| Filter value | Metadata value | Match when |
|---|---|---|
| scalar | scalar | values are equal |
| scalar | list | scalar appears in the list |
| list | scalar | scalar is one of the filter elements |
| list | list | the two lists have a non-empty intersection |

Empty-list filter values (`{"key": []}`) never match — intersection
with the empty set is empty.

## Examples

```python
# Records:
#   A: {"type": "tension", "tags": ["urgent", "blocker"]}
#   B: {"type": "gap",     "tags": ["urgent"]}
#   C: {"type": "tension", "tags": ["later"]}
#   D: {"type": "gap",     "tags": []}
#   E: {"type": "terminology"}                      # no "tags" key

await store.search(q, k=10, filter={"type": "tension"})
# → A, C       (scalar/scalar EQ)

await store.search(q, k=10, filter={"tags": "urgent"})
# → A, B       (scalar filter, list metadata — "is in list")

await store.search(q, k=10, filter={"type": ["tension", "gap"]})
# → A, B, C, D (list filter, scalar metadata — IN)

await store.search(q, k=10, filter={"tags": ["urgent", "later"]})
# → A, B, C    (list filter, list metadata — intersection)

await store.search(q, k=10, filter={"type": "tension", "tags": "urgent"})
# → A         (AND across keys)

await store.search(q, k=10, filter={"missing_key": "value"})
# → ∅          (missing key fails the filter)
```

## Per-backend implementation notes

| Backend | Implementation |
|---|---|
| `MemoryVectorStore` / `FaissVectorStore` | Post-hoc Python filter via `VectorStoreBase._match_metadata_filter`. Applied after similarity ranking. |
| `ChromaVectorStore` | Native Chroma `$in` predicate for list filter values (pushed down for prefiltering); scalar filter values are post-filtered in Python because Chroma's `$eq` does not match list-valued metadata. Scalar/list-metadata fix is the gap this was designed to close. `count()` uses `collection.get(where=..., include=["metadatas"])` and post-filters. |
| `PgVectorStore` | JSONB-native via `jsonb_build_object` and the `@>` containment operator. For each filter element, two `@>` checks are emitted ORed together — one with the value as a scalar and one wrapped in an array — to cover both scalar-metadata and list-metadata in one SQL shape. Type-preserving (booleans stay booleans, numbers stay numbers); replaces the older text-cast `metadata->>'key' = '...'` translation, which silently returned zero rows for booleans, numbers, and lists. |

## Type safety (PgVector)

The `@>` translation preserves JSONB types. Filtering integer
metadata with `{"count": 5}` matches an integer-valued record;
filtering with `{"count": "5"}` does not (no implicit string
coercion). Boolean metadata works the same way:

```python
# Stored: {"active": True, "count": 5}
await store.count(filter={"active": True})    # → 1
await store.count(filter={"active": False})   # → 0
await store.count(filter={"count": 5})        # → 1
await store.count(filter={"count": "5"})      # → 0  (type-preserving)
```

## Constraints

- **Hashability.** List filter values and list metadata values are
  reduced to a `set` for the intersection check. Elements must be
  hashable. Nested dicts or lists in metadata array elements are
  unsupported; consumers storing such values should compose a separate
  filter source. The `TypeError` from unhashable elements only
  surfaces in the list/list quadrant — the other three quadrants do
  not build a set, so unhashable values pass through silently. Treat
  hashability as a global precondition rather than a quadrant-local
  one.
- **Filter shape is flat.** The current `filter` signature is
  `dict[str, Any] | None` with scalar or list values per key. Boolean
  composition (`$or`, `$not`), range predicates (`>=`, `BETWEEN`), and
  reusing `dataknobs_data.query.Filter` / `ComplexQuery` are
  follow-ups requiring a signature change. Until then, compose
  multiple `GroundedSource` implementations or pre-narrow at the
  knowledge layer.
- **Chroma post-filter over-fetch.** When a scalar filter is partially
  post-filtered, `ChromaVectorStore.search()` over-fetches `k * 4`
  candidates from Chroma, then narrows. High-cardinality tag filters
  with strict scalar gates may need a higher multiplier — currently a
  constant; configurability is a follow-up if a consumer hits it.
- **Chroma `count` materializes metadata.** Chroma has no first-class
  filtered-count API. The `count(filter=...)` path uses
  `collection.get(where=..., include=["metadatas"])` and post-filters
  in process. Memory-bound for very large collections.

## Background

This is a strict superset of the prior behavior. Scalar/scalar
equality is preserved exactly on every backend; cases that previously
silently returned zero rows (list metadata with a scalar filter on
Memory/FAISS/PgVector; boolean and numeric metadata on PgVector) now
match. No existing tests pinned the broken behavior, so the change is
additive. The `filter` signature is unchanged.
