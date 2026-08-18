# Vector Store Timestamp Exposure

`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`, and
`PgVectorStore` track `created_at` and `updated_at` timestamps per
vector and expose them on demand via `include_timestamps=True` on
`get_vectors()` and `search()`.

## Configuration

Timestamp exposure is configured via the shared `timestamps` block on
any `VectorStoreBase` subclass:

```yaml
vector_store:
  provider: memory  # or faiss, chroma, pgvector
  dimensions: 768
  timestamps:
    format: iso        # "iso" | "epoch" | "datetime" (default: "iso")
    created_key: _created_at   # default: "_created_at"
    updated_key: _updated_at   # default: "_updated_at"
```

Invalid `format` values raise `ValueError` at store construction.

| Key | Default | Values |
|-----|---------|--------|
| `timestamps.format` | `"iso"` | `"iso"`, `"epoch"`, `"datetime"` |
| `timestamps.created_key` | `"_created_at"` | Any string |
| `timestamps.updated_key` | `"_updated_at"` | Any string |

## Usage

```python
results = await store.get_vectors(["id1"], include_timestamps=True)
vector, meta = results[0]
# meta["_created_at"] == "2026-04-22T14:23:45.123456+00:00"
# meta["_updated_at"] == "2026-04-22T14:23:45.123456+00:00"

# Same kwarg on search:
hits = await store.search(query, k=5, include_timestamps=True)
for vec_id, score, meta in hits:
    print(meta["_created_at"], meta["_updated_at"])
```

`include_timestamps=True` requires `include_metadata=True` (the default).
When `include_metadata=False`, timestamp injection is silently skipped
— there is no metadata dict to inject into.

## Semantics

- `created_at` is set on first `add_vectors` for an ID and **preserved**
  on subsequent upserts (same-ID `add_vectors`).
- `updated_at` is **refreshed** on every upsert and on
  `update_metadata` — for a row the store already tracks. A row
  written before this backend tracked timestamps is not given one
  retroactively by an update; see [Null timestamps](#null-timestamps)
  for what an untracked row does instead.

## Where the values are stored

Per-backend, and it only matters in one case:

| Backend | Storage |
|---------|---------|
| `PgVectorStore` | real `created_at` / `updated_at` columns |
| `MemoryVectorStore`, `FaissVectorStore` | an in-process side-car keyed by row |
| `ChromaVectorStore` | **in-band**, in the collection's own metadata |

A Chroma collection is the only per-row storage that backend has, so
its timestamps live in the same metadata dict a consumer owns, under
two reserved NUL-delimited keys (the convention this store already
uses to keep non-scalar values inside chromadb's scalar-only
contract). They are stripped from every read — `get_vectors()`,
`search()`, `search_documents()`, `metadata_fields()`, and the
residual metadata filter behind `count()` / `clear()` — so nothing a
consumer can reach through the store surfaces them, and a filter
cannot match on them.

Two consequences worth knowing:

- **Inspecting a Chroma collection directly** — outside this store,
  through chromadb — will show the reserved keys. They are ours, not
  the row's.
- **Version skew on one collection.** A collection written by this
  version and read by an earlier one surfaces the reserved keys as
  ordinary metadata, because the earlier version has no strip. Read a
  collection with the version that wrote it, or later.

The stored value is an epoch float regardless of the configured
`format`; `format` applies on the way out. That is deliberate — a
store whose `timestamps.format` config changes can still read rows
written under the old one.

## Output formats

`_format_timestamp` maps the backend's stored timestamp to the
configured output:

| Format | Output type | Example |
|--------|-------------|---------|
| `iso` (default) | `str` (ISO-8601) | `"2026-04-22T14:23:45.123456+00:00"` |
| `epoch` | `float` (seconds since epoch) | `1745330625.123456` |
| `datetime` | `datetime` | `datetime(2026, 4, 22, 14, 23, 45, ...)` |

All three formats return `None` when the backend has no timestamp for
the row (see "Null timestamps" below).

## Clock sources

Timestamps are **backend-local** — compare within a store, not across:

| Backend | Clock source |
|---------|--------------|
| `MemoryVectorStore` | Python `datetime.now(UTC)` (aware UTC) |
| `FaissVectorStore` | Python `datetime.now(UTC)` (aware UTC) |
| `ChromaVectorStore` | Python `datetime.now(UTC)` (aware UTC) |
| `PgVectorStore` | Postgres server `NOW()` (naive `TIMESTAMP`) |

In the `epoch` format, naive datetimes (pgvector) are converted using
the system's local-time interpretation, while aware datetimes
(`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`) use
their embedded timezone.
Cross-backend epoch comparisons are therefore not meaningful — this is
by design, since the two clocks are already unsynchronised.

## Null timestamps

- **PgVectorStore pre-migration rows.** Existing rows at the moment the
  `updated_at` column was added have `NULL` in that column.
  `include_timestamps=True` surfaces `None` for those rows.
  Distinguish "never re-ingested since the column was added" from
  "current" via `meta["_updated_at"] is None`. The column is backfilled
  to `NOW()` on the next upsert or `update_metadata`.
- **MemoryVectorStore legacy pickles.** Pickle files saved before
  timestamp tracking was added have no tracked timestamps; existing
  rows return `None` for both `_created_at` and `_updated_at` on
  injection until the next `add_vectors` refresh populates the
  tracking dict.
- **FaissVectorStore legacy persisted indexes.** The same applies to
  FAISS sidecar pickles persisted before timestamp tracking was added:
  the timestamp side-car loads empty (`data.get("timestamps", {})`),
  so existing rows return `None` for both keys until the next
  `add_vectors` refresh repopulates them — `update_metadata` does not,
  because it guards on the row already having a side-car entry. An
  index *persisted before the stored-vector side-car was added* also
  has no `vectors` side-car (`data.get("vectors", {})` loads empty),
  so `get_vectors()` returns `None` for its ids until the vectors are
  re-added (or the corpus re-ingested) once; similarity `search` is
  unaffected because the FAISS index itself is restored normally.
- **ChromaVectorStore collections written before tracking.** Rows in a
  collection written by an earlier version carry no reserved timestamp
  keys and return `None` for both until their next `add_vectors` or
  `add_documents` repopulates them. Nothing is backfilled on read.

**One rule across all four:** a *write* establishes tracking; an
*update* does not. `add_vectors` (and `add_documents`) repopulates a
pre-tracking row's timestamps; `update_metadata` and
`update_metadata_where` leave `created_at` exactly as they found it,
including `None`. `update_vectors` is a write — it is an alias for
`add_vectors` — so it establishes tracking and, on a row already
tracked, preserves `created_at` like any other upsert.

The reason is that `None` here means *not known*, and there is no
honest value an update could put in its place. Stamping the update time
into `created_at` would make one `update_metadata_where(None, ...)`
migration sweep record every legacy row as having been created at the
moment of the sweep — and nothing afterwards could tell a fabricated
creation date from a real one. A retention or audit policy reading
those dates would be reading the sweep.

`updated_at` is the one asymmetry: pgvector refreshes it to `NOW()` on
`update_metadata` even for a pre-migration row, because it is a real
column with no side-car entry to be missing.

## Consumer metadata key collision

If a consumer's metadata dict already contains a key matching the
configured `created_key` or `updated_key`, the **consumer's value
wins** and the framework skips timestamp injection for that key. A
`WARNING` is logged once per process per `(store instance,
colliding key)` pair:

```
VectorStore timestamp injection skipped — consumer metadata already
contains key '_created_at'. Rename via timestamps.created_key /
timestamps.updated_key config to avoid collision.
```

To avoid collisions with consumer-owned keys, override the defaults in
config:

```yaml
timestamps:
  created_key: __dk_created_at
  updated_key: __dk_updated_at
```
