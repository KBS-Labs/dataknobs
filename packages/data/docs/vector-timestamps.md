# Vector Store Timestamp Exposure

`MemoryVectorStore` and `PgVectorStore` track `created_at` and
`updated_at` timestamps per vector and expose them on demand via
`include_timestamps=True` on `get_vectors()` and `search()`.

**Deferred backends:** `FaissVectorStore` and `ChromaVectorStore` do
**not** yet accept the `include_timestamps` kwarg — calling it on
those backends raises `TypeError`. The `VectorStoreBase` helpers
(`_format_timestamp`, `_inject_timestamps`) are in place so adding
support is purely additive; the work is tracked under Item 36's
deferred follow-ups and will ship when a consumer needs it.

## Configuration

Timestamp exposure is configured via the shared `timestamps` block on
any `VectorStoreBase` subclass:

```yaml
vector_store:
  provider: memory  # or pgvector
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
  `update_metadata`.

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
| `MemoryVectorStore` | Python `datetime.now(timezone.utc)` (aware UTC) |
| `PgVectorStore` | Postgres server `NOW()` (naive `TIMESTAMP`) |

In the `epoch` format, naive datetimes (pgvector) are converted using
the system's local-time interpretation, while aware datetimes
(MemoryVectorStore) use their embedded timezone. Cross-backend epoch
comparisons are therefore not meaningful — this is by design, since
the two clocks are already unsynchronised.

## Null timestamps

- **PgVectorStore pre-migration rows.** Existing rows at the moment the
  `updated_at` column was added have `NULL` in that column.
  `include_timestamps=True` surfaces `None` for those rows.
  Distinguish "never re-ingested since the column was added" from
  "current" via `meta["_updated_at"] is None`. The column is backfilled
  to `NOW()` on the next upsert or `update_metadata`.
- **MemoryVectorStore legacy pickles.** Pickle files saved before
  Item 36 have no tracked timestamps; existing rows return `None` for
  both `_created_at` and `_updated_at` on injection until the next
  `add_vectors` refresh populates the tracking dict.

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
