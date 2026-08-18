# Vector Store Metadata Filter Semantics

`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`, and
`PgVectorStore` all accept a `filter: dict[str, Any] | None` argument
on `search()`, `count()`, `clear()`, and `update_metadata_where()`.
Per-key match semantics are identical across backends — consumers can
runtime-swap the backing store without behavioral surprises. That
covers the result *count* as well as which rows match: a filtered
`search(k=...)` returns `k` rows on every backend whenever `k` rows
match, however far outside the unfiltered top-`k` they sit. All four
backends now hold that unconditionally; what each *pays* to hold it
differs, and is described under [Constraints](#constraints).

The one case where a backend can still under-return is a FAISS store
whose vector side-car is incomplete — a `.meta` pickle written before
that side-car existed. It is reported at `WARNING` and fixed by
re-ingesting; see the FAISS constraint below.

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
  resolves to an unsatisfiable filter and matches zero rows, and
* confines the **id-keyed** operations to the same scope.
  `get_vectors()`, `delete_vectors()` and `update_metadata()`
  address rows by id, so they build no filter; each instead checks
  the row it resolved against the configured scope. An
  out-of-domain id is answered exactly as an absent one — `(None,
  None)` from `get_vectors()`, and no contribution to the count
  returned by the other two — so a caller cannot tell "not here"
  from "not yours". `metadata_fields()` likewise unions keys only
  over in-scope rows.

  This half matters more than it looks. Scoping only the surfaces
  that happen to take a `filter` would make the tenant boundary a
  property of *how a caller asks* rather than of the store, and
  vector ids are routinely derived from content (`"<doc>_<chunk>"`),
  so they are guessable rather than secret. It also closes a write
  path: because a replacement carries the configured `domain_id`
  forward, an unscoped `update_metadata()` on another tenant's row
  would not merely edit that row, it would relabel it into the
  caller's own domain.

  `update_metadata()` preserving the configured `domain_id` is what
  keeps that replacement from having the opposite effect — dropping
  the scope key and leaving a row its own store can no longer see,
  count, or even `clear()`.

This makes the **runtime-swap promise hold for config-level
scoping**: a tenant-scoped store behaves identically under
unscoped `count()` / `search()` / `clear()` /
`update_metadata_where()` regardless of backend — each touches
only the configured tenant's rows, and `search()` returns as many
of them as it was asked for.

A scoped store is the case that exercises this hardest: the scope
is AND-composed into *every* call, so every search on such a store
is a filtered search. A backend that narrows only after its index
has truncated to `k` therefore under-returns on every query, and
the smaller the configured tenant is relative to its co-tenants,
the less it retrieves — while `count()` keeps reporting the full
number of rows it holds.

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

This is a **backend-conformance contract**, not an accident of the
Python matcher: `{key: []}` is an unsatisfiable predicate that MUST
match zero rows on *every* backend, including those that translate
filters natively (Chroma's `$in`/`$eq` push-down, pgvector's JSONB SQL,
which emits a literal `FALSE`). Consumers rely on it to express a
deliberate no-op — e.g. `VectorMemory.clear()` AND-merges a `{key: []}`
contradiction into a caller filter so a cross-tenant clear removes
nothing rather than another tenant's rows. A parametrized cross-backend
conformance test pins this so a future backend (or a refactor of an
existing one) cannot silently break it.

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
| `MemoryVectorStore` | Python filter via `VectorStoreBase._match_metadata_filter`, applied to candidates **before** similarity ranking: every stored row is matched, the survivors are scored, sorted, and truncated to `k`. `update_metadata_where` walks the in-process `metadata_store` and `dict.update`s `set_` into each match. |
| `FaissVectorStore` | Same `_match_metadata_filter`, and likewise **not** applied after ranking. An unfiltered search is answered by the FAISS index. A filtered one selects the matching rows from `metadata_store`, scores just those against the query from the vector side-car, sorts, and truncates to `k` — exact on every index type, and the same rows the index would have ranked, in the same order. (Rows scoring *equal* are ordered by insertion rather than by whatever internal order the index would have used; the ranking agrees, the tie-break need not.) `update_metadata_where` walks the same `metadata_store`, with no FAISS index involvement. |
| `ChromaVectorStore` | Post-hoc Python filter via `VectorStoreBase._match_metadata_filter` by default. Chroma's where-engine returns zero rows for *any* predicate against list-valued metadata, so neither scalar nor list filter values are pushed down unless the key is declared in `scalar_metadata_keys` (then `$eq`/`$in` is pushed for that key). Because that residual filter runs *after* Chroma has truncated to `n_results`, the query escalates — `k * POST_FILTER_OVERFETCH`, then doubling, bounded by `collection.count()` — until `k` rows survive the filter or the whole collection has been returned, at which point the answer is exact. `count()` uses `collection.get(where=..., include=["metadatas"])` and post-filters. `update_metadata_where` fetches matched rows, merges `set_` in Python and writes them back — the merge is done here rather than left to chromadb because the decoded row has to be re-encoded as a whole. (chromadb's own `update` *merges*: a key it is not given survives. That is why `update_metadata`, whose contract is a wholesale replacement, has to name each departing key with a `None` value to delete it — and why a `None` **value** is not storable on this backend, since dropping a key and setting it to `None` are the same request.) Row timestamps live in this collection too, under two reserved NUL-delimited keys that every read strips. Metadata is encoded at the Chroma boundary since chromadb's store is scalar-only (empty dict → no-metadata; every list/dict value, including `[]`, → reversible JSON sentinel — chromadb otherwise silently corrupts non-scalar values, bleeding them across collections); reads decode back so the round-trip matches Memory/FAISS. |
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

## Metadata ownership

A store and its caller never share a metadata object, in either
direction. Every backend takes a copy of what is written to it and
returns a copy of what is read from it, so:

```python
meta = {"type": "note", "tags": ["urgent"]}
await store.add_vectors(vectors, ids=["a"], metadata=[meta])

meta["tags"].append("later")        # does not reach the stored row

(_, stored), = await store.get_vectors(["a"])
stored["tags"].append("later")      # does not reach it either
```

The copies are **deep**. A shallow copy would leave the nested `tags`
list shared, which is where this is actually reached — the outer dict
is usually replaced wholesale rather than edited. Memory and FAISS copy
with `copy.deepcopy` rather than through a JSON round-trip, because they
persist by pickle and so accept values JSON cannot express: a tuple in
metadata survives the copy as a tuple. That is a property of the copy on
those two backends, not a portable one — Chroma and pgvector store
through JSON, where a tuple reads back as a list.

This applies to `add_vectors`, `update_metadata`,
`update_metadata_where`, `search` and `get_vectors`. The
`update_metadata_where` case is the one worth calling out: a single
`set_` is merged into every row the filter selects, and each row gets
its own copy, so the rows can diverge afterwards and a later edit to
`set_` reaches none of them.

Chroma and pgvector satisfy this by construction — both serialize
metadata at their boundary, so what they store and what they return are
already reconstructions. Memory and FAISS copy explicitly. The cost is
paid per stored row and per *returned* row, never per scored candidate,
so a filtered search over a large corpus does not copy the rows it
discards.

The only way to change a stored row is to call a mutator.

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
- **Chroma post-filters cost repeated queries.** When part of a filter
  cannot be pushed down, `ChromaVectorStore.search()` (and
  `search_documents()`) must narrow in Python *after* Chroma has already
  truncated to `n_results`. It compensates by escalating the fetch —
  `k * POST_FILTER_OVERFETCH`, then doubling, bounded by
  `collection.count()` — so a sparse filter costs several round-trips to
  Chroma plus one `count()`, rather than under-returning. The count is
  native and O(1); the extra queries are not free, and a filter matching
  very few rows in a large collection will walk most of the way to the
  full collection size before it is satisfied. Declaring the key in
  `scalar_metadata_keys` pushes the predicate down and avoids all of it.
  `POST_FILTER_OVERFETCH` is a module-level constant in
  `dataknobs_data.vector.stores.common`, shared by every backend that
  post-filters, and is not yet configurable per store.
- **Chroma's writes read first.** `add_vectors()` and `add_documents()`
  issue one extra `collection.get(...)` per call — per batch, not per
  row — to read each id's stored `created_at` before overwriting it, so
  that re-adding an existing id preserves the original creation date.
  `update_metadata()` reads for a second reason as well: the keys it
  must tombstone are the ones the row has and the caller's replacement
  does not, which cannot be known without the stored dict. For a bulk
  ingest this materializes every existing row's full metadata to read
  two floats, because chromadb has no way to project a subset of keys.
- **Chroma's read-then-write is not atomic.** chromadb offers no
  compare-and-swap, so both sequences above have a window. Two
  concurrent `add_vectors()` calls for the same new id each read
  "absent" and each stamp their own `created_at`; last write wins, and
  both values are within a moment of each other. More consequentially,
  a key written by a concurrent writer *between* `update_metadata()`'s
  read and its write is not in the snapshot, so it is not tombstoned
  and survives — the wholesale replacement quietly degrades to a
  partial one. Treat a row as having a single writer, or serialize at
  the application layer. This is the one place the note under
  *Persistence* below — that Chroma and pgvector leave concurrency to
  the backing service — does not fully hold: Postgres has the
  primitives to make this atomic and chromadb does not.
- **Every backend tracks timestamps, including Chroma.** All four
  answer `include_timestamps=True` on `search()` and `get_vectors()`
  from values they really hold. Where those values live differs —
  pgvector has real columns, Memory and FAISS keep a side-car keyed by
  row, and Chroma, whose only per-row storage is the metadata dict the
  consumer also owns, keeps them in-band under reserved keys stripped
  from every read path.

  `None` therefore means one thing everywhere: *this row has no tracked
  timestamp*. A pgvector row from before the timestamp migration, a
  Memory/FAISS pickle written before tracking existed, and a Chroma
  collection written before this backend tracked them all answer that
  way, and all repopulate on the row's next `add_vectors`. An
  `update_metadata` does **not** repopulate them on any backend: an
  update does not begin tracking a row that was never tracked, because
  the alternative is recording the update time as a creation date with
  nothing left to distinguish it from a real one.
- **FAISS filtered search leaves the index.** An unfiltered
  `FaissVectorStore.search()` uses the FAISS index and is as fast as
  the configured index type makes it. A filtered one does not: no FAISS
  index can express the filter, so the matching rows are scored
  directly instead — an O(N) walk of `metadata_store` plus O(M)
  similarity computations for the M rows that match. In **CPU** that is
  less work than `MemoryVectorStore` does for the same query, which
  scores every survivor of a full scan one row at a time in Python. In
  **peak memory** it is the other way round: the M matching vectors are
  stacked into one contiguous float32 array and the metric computed over
  it, so a filtered query transiently holds `M × dimensions × 4` bytes —
  doubled on the L2/euclidean path, which materializes the difference
  array as well — where `MemoryVectorStore` holds `O(dimensions)`. At
  500k matching rows and 768 dimensions that is 1.5 GB, or 3 GB under
  L2, on top of the index the store is no longer using. Size for it, or
  filter to a narrower M.

  A store configured with a `domain_id` filters on every call and
  therefore never uses the index at all; scope-heavy workloads at scale
  should prefer `pgvector`, where the scope is a native SQL predicate
  the index search runs under.
- **An incomplete FAISS side-car under-returns.** `_load_from_disk`
  reads the vector side-car with a default, so a `.meta` pickle written
  before it existed loads empty against a fully populated index. Rows
  with no stored vector are ranked from the index instead of scored
  directly, and merged with the exactly-scored ones — the two are on the
  same scale. An approximate index does not route to every row, so a
  filtered search on such a store can still return fewer than `k`. It is
  reported once per store at `WARNING`, and re-ingesting fixes it.
- **A file `persist_path` is single-writer.** This covers
  `FaissVectorStore` and `MemoryVectorStore` — both persist by
  serializing the instance's whole in-memory state over one file, which
  is what makes the hazard theirs and not the file format's. Two
  instances holding one path with overlapping lifetimes would each write
  a snapshot that never saw the other's rows, so a store raises
  `dataknobs_common.exceptions.ConcurrencyError` rather than overwrite a
  file that changed since it read it.

  Three consequences worth knowing:

  * `close()` persists only a store that was **mutated**. An instance
    opened to read writes nothing on teardown — necessarily, since that
    write would move the file's identity and make the real writer's save
    refuse.
  * The check is best-effort, not a lock: it compares modification time,
    size and inode, and two writes inside one filesystem timestamp tick
    that happen to produce the same size are indistinguishable. It
    catches the overwhelmingly common accident, not a determined race.
  * A refusal is recoverable with `save(force=True)`, which overwrites
    deliberately and accepts the loss of whatever the other writer
    persisted. Without it the refusal repeats forever, because what it
    compares against has not moved. It is only ever the right call
    against a *genuine* conflict — a refusal the store caused itself is
    a defect, not a case for `force`.

  Sequential lifetimes are unaffected and keep appending, since
  `initialize()` loads the file first. For genuinely concurrent writers,
  use `pgvector`.

  FAISS writes two files — the index and its `.meta` side-car — to
  scratch siblings and renames them into place only once both have been
  written, so a failed write leaves the previous state intact. Renaming
  is atomic per file but not across the pair, so a failure *between* the
  two renames can still leave a new index beside an old side-car; making
  that impossible needs a single-file format or a write-ahead log.
  Retrying is not blocked by it, though: the store re-reads the file's
  identity and stays dirty, so the next `save()` or `close()` writes the
  rows it still holds.
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
