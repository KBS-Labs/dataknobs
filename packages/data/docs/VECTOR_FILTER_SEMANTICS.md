# Vector Store Metadata Filter Semantics

`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`, and
`PgVectorStore` all accept a `filter: dict[str, Any] | None` argument
on `search()`, `count()`, `clear()`, and `update_metadata_where()`.
Per-key match semantics are identical across backends — consumers can
runtime-swap the backing store without behavioral surprises.

`update_metadata_where(filter, set_)` is the filter-keyed mutator
sibling of the id-keyed `update_metadata(ids, metadata)`. It selects
rows with the **same** four-quadrant `filter` shape as `clear` /
`count` / `search`, then *merges* `set_` into each matched row's
metadata (keys in `set_` overwrite, unrelated keys are preserved),
returning the affected row count. `filter=None` matches every vector
(parity with `clear()`). The ABC default raises `NotImplementedError`
— the contract for out-of-tree stores only; all four in-tree stores
implement it. It is the primitive behind `dataknobs-bots`'
`IngestSwapMode.TOMBSTONE` zero-downtime re-ingest (mark a generation
`_stale`, then un-mark on rollback).

`clear(filter=...)` removes only vectors whose metadata matches the
filter, leaving non-matching vectors intact. `clear()` (no filter)
preserves the historical unscoped behavior — every vector in the
store is removed. Backend-specific note: FAISS has no native
filtered delete; filtered clear iterates `metadata_store` to collect
matching IDs and delegates to `delete_vectors(ids)` (O(N) over
stored vectors). Workloads at scale where filtered clear is hot
should prefer pgvector or Chroma where filtered delete is native.

### Config-level `domain_id` scoping (all four backends)

All four backends accept a config-level `domain_id` that scopes
**every** read and write to the configured tenant. A store
constructed with `domain_id="x"`:

* defaults `domain_id="x"` into the metadata of vectors added
  without an explicit `domain_id` (Memory/FAISS/Chroma write it
  into the per-row metadata; PgVector writes it to its dedicated
  `domain_id` column and leaves the caller's JSONB metadata
  verbatim), and
* AND-composes `domain_id="x"` into the effective filter for
  `search()`, `count()`, `clear()`, and `update_metadata_where()`.
  So `clear()` (no filter) deletes only the configured tenant's
  rows — not a full-collection wipe — and `clear(filter={...})`
  AND-composes the explicit filter on top of the tenant scope.
  An explicit caller `domain_id` that is out of scope (e.g.
  `filter={"domain_id": "y"}` on a store scoped to `"x"`)
  resolves to an unsatisfiable filter and matches zero rows.

This makes the **runtime-swap promise hold for config-level
scoping**: a tenant-scoped store behaves identically under
unscoped `count()` / `search()` / `clear()` /
`update_metadata_where()` regardless of backend — each touches
only the configured tenant's rows.

#### One residual divergence: explicit `domain_id` filters on PgVector

Memory/FAISS/Chroma store the configured `domain_id` *inside*
each row's metadata, so an explicit in-scope `filter={"domain_id":
"x"}` is an ordinary metadata-key match and selects those rows.
PgVector stores the configured `domain_id` in a dedicated
**column**, not in the JSONB metadata, and an explicit
`filter={"domain_id": "x"}` is translated to a JSONB-containment
probe (`metadata @> {"domain_id": "x"}`). Rows whose tenant was
assigned only via config carry no `domain_id` *in JSONB*, so that
explicit filter selects zero rows on PgVector while selecting the
tenant's rows on the other three backends.

Practical guidance: rely on **config-level** scoping (omit
`domain_id` from the caller filter and let the store apply it) for
backend-portable multi-tenant isolation. Only pass an explicit
`{"domain_id": ...}` filter when every backend in play stores
`domain_id` in caller metadata (i.e. not PgVector, or PgVector
where the consumer also writes `domain_id` into the metadata
dict). The `KnowledgeIngestionManager` /
`RAGKnowledgeBase` / `VectorMemory` upper layers apply tenant
scope through this config-level path, so consumers driving
multi-tenant ingestion through them see consistent behavior across
all four backends.

### Optional `scalar_metadata_keys` push-down on `ChromaVectorStore`

By default, `ChromaVectorStore` post-filters every filter value
(scalar *and* list) in Python: chromadb's where-engine returns zero
rows for any predicate against list-valued metadata, so a pushed-down
predicate would silently drop matches for consumers whose metadata
stores tags or categories as lists. The post-filter is correct but
materializes matching metadata in process for `count()` and
over-fetches for `search()`.

Consumers whose metadata for a given key is **always scalar** (the
common multi-tenant scoping pattern) can declare those keys via
the `scalar_metadata_keys` config option:

```python
from dataknobs_data.vector.stores.chroma import ChromaVectorStore

store = ChromaVectorStore({
    "dimensions": 384,
    "collection_name": "kb",
    # Stored values for these keys are guaranteed scalar.
    "scalar_metadata_keys": ["domain_id", "tenant_id"],
})
```

For declared keys the partitioner pushes a Chroma-native predicate
(`$eq` for a scalar filter value, `$in` for a list filter value),
eliminating the post-filter. `count(filter={"domain_id": "x"})`
then fetches only IDs (no metadata) when the entire filter pushes
down, regardless of collection size. (Declaring a key scalar is a
contract that its stored values are never lists; pushing a native
predicate for a genuinely list-valued key would drop all matches.)

The declaration is opt-in and additive: keys not declared keep
the conservative post-filter behavior, so existing consumers see
no change.

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
| `MemoryVectorStore` / `FaissVectorStore` | Post-hoc Python filter via `VectorStoreBase._match_metadata_filter`. Applied after similarity ranking. `update_metadata_where` walks the in-process `metadata_store` (FAISS: the same side-car `search`/`clear` already post-filter — no FAISS index involvement) and `dict.update`s `set_` into each match. |
| `ChromaVectorStore` | Post-hoc Python filter via `VectorStoreBase._match_metadata_filter` by default. Chroma's where-engine returns zero rows for *any* predicate against list-valued metadata, so neither scalar nor list filter values are pushed down unless the key is declared in `scalar_metadata_keys` (then `$eq`/`$in` is pushed for that key). `count()` uses `collection.get(where=..., include=["metadatas"])` and post-filters. `update_metadata_where` fetches matched rows, merges `set_` in Python (Chroma `update` replaces a row's metadata wholesale), and writes them back. Metadata is encoded at the Chroma boundary since chromadb's store is scalar-only (empty dict → no-metadata; every list/dict value, including `[]`, → reversible JSON sentinel — chromadb otherwise silently corrupts non-scalar values, bleeding them across collections); reads decode back so the round-trip matches Memory/FAISS. |
| `PgVectorStore` | JSONB-native via `jsonb_build_object` and the `@>` containment operator. For each filter element, two `@>` checks are emitted ORed together — one with the value as a scalar and one wrapped in an array — to cover both scalar-metadata and list-metadata in one SQL shape. Type-preserving (booleans stay booleans, numbers stay numbers); replaces the older text-cast `metadata->>'key' = '...'` translation, which silently returned zero rows for booleans, numbers, and lists. `update_metadata_where` reuses this translation in a single `UPDATE ... SET metadata = metadata || $::jsonb` (JSONB merge, `updated_at` refreshed). |

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
