# DataKnobs Batch Processing Guide

## Overview

The DataKnobs data package provides two configuration systems for batch operations:

1. **StreamConfig** - For streaming database operations with real-time processing
2. **BatchConfig** - For pandas DataFrame batch operations with memory efficiency

## When to Use Each Configuration

### Use StreamConfig When:

- **Real-time streaming**: Processing records as they arrive from a source
- **Database operations**: Writing/reading records to/from database backends
- **Error resilience**: Need graceful degradation with per-record error handling
- **Progress tracking**: Want detailed statistics on success/failure rates
- **Large datasets**: Processing data that doesn't fit in memory all at once

**Example Use Cases:**
- Ingesting sensor readings in real-time
- Processing log files line by line
- Migrating data between databases
- Handling webhook events

```python
from dataknobs_data import StreamConfig, StreamResult

# Configure streaming with error handling
config = StreamConfig(
    batch_size=1000,
    prefetch=2,  # Prefetch 2 batches for performance
    on_error=lambda exc, record: True  # Continue on error
)

# Stream records to database
result = db.stream_write(records, config)
print(f"Processed {result.total_processed} records")
print(f"Success rate: {result.success_rate:.1f}%")
print(f"Failed indices: {result.failed_indices}")
```

#### Conflict policy (insert / upsert / skip)

`StreamConfig.on_conflict` selects how a write resolves an id that already
exists in the target. It applies to every `stream_write` backend and to the
`Migrator` methods that stream. The default preserves strict insert behavior.

| Policy | Behavior on a colliding id |
|--------|----------------------------|
| `"insert"` (default) | Fail closed — the id is recorded as a failure. Correct for a virgin target. |
| `"upsert"` | Overwrite the existing row so the target matches the source. Idempotent; a colliding id cannot fail. |
| `"skip"` | Leave the existing row untouched and count the id as skipped (`StreamResult.skipped`). An idempotent top-up. |

```python
from dataknobs_data import ConflictPolicy, StreamConfig

# Re-run a migration into a populated target, overwriting existing rows.
config = StreamConfig(batch_size=1000, on_conflict="upsert")
result = db.stream_write(records, config)

# Or migrate only what is not already present, leaving existing rows as-is.
config = StreamConfig(on_conflict=ConflictPolicy.SKIP)
result = db.stream_write(records, config)
print(f"Skipped {result.skipped} already-present records")
```

The `"insert"` fast-path uses the backend's native `create_batch` (or an atomic
`_write_batch` on PostgreSQL) with a per-record `create()` fallback so a
collision is attributed to the specific id; the non-transactional backends (S3,
Elasticsearch-sync, async-Elasticsearch) write INSERT per-record via `create()`
directly. `"upsert"`
uses the native `upsert_batch` bulk verb (see below) with a per-record `upsert`
fallback; `"skip"` writes one record at a time (a whole-batch verb cannot skip
individual duplicates while inserting the rest). An unknown policy value is
rejected when the `StreamConfig` is constructed, rather than silently falling
back to insert.

Streaming `"insert"` fails closed on a colliding id — recording it as a failure
and preserving the source id — across **every** backend. SQLite / DuckDB reach
this through the fail-closed bulk `create_batch` plus a per-record `create()`
fallback; PostgreSQL through its atomic `_write_batch` fast-path; S3,
Elasticsearch-sync, and async-Elasticsearch through per-record `create()` (their
non-transactional bulk write would otherwise re-write already-written rows under
the fallback). Use `"upsert"` / `"skip"` for idempotent re-runs into a populated
target.

#### Batch write verbs: `create_batch` vs `upsert_batch`

Every backend exposes two batch write verbs:

| Verb | Semantics on a colliding id | Honors `record.id` |
|------|-----------------------------|--------------------|
| `create_batch(records)` | Insert-only — a colliding id (existing row or a within-batch duplicate) **fails closed** with `DuplicateRecordError`, uniform across every backend, matching single `create()` | yes |
| `upsert_batch(records)` | **Overwrite** — never raised, never skipped | yes (minted only when absent) |

`upsert_batch` is the batch sibling of a per-record `upsert` loop: it inserts
new records and overwrites existing ones in one call, returns the ids in input
order, and carries no version check (batch compare-and-set is not supported — a
whole batch cannot carry a single version token). It uses the backend's native
bulk verb where one exists — a single `INSERT ... ON CONFLICT (id) DO UPDATE` on
SQLite / DuckDB / PostgreSQL, a bulk index-by-id on Elasticsearch, a single
file-rewrite (file) or single-lock pass (memory) — and a per-record loop on S3
(per-key PUT). It is the batch verb the streaming `"upsert"` policy routes
through.

```python
from dataknobs_data import Record

# Insert two new rows and overwrite one existing row, in a single call.
ids = db.upsert_batch([
    Record({"name": "ACME"}, id="k1"),   # overwrites k1 if present
    Record({"name": "beta"}, id="k2"),   # inserts k2
])
assert ids == ["k1", "k2"]
```

### Use BatchConfig When:

- **DataFrame operations**: Working with pandas DataFrames
- **Memory efficiency**: Processing large CSVs or DataFrames in chunks
- **Parallel processing**: Need multi-threaded batch processing
- **Data analysis**: Performing aggregations or transformations

**Example Use Cases:**
- Loading large CSV files into DataFrames
- Bulk data transformations
- Parallel DataFrame processing
- Memory-efficient data analysis

```python
from dataknobs_data.pandas import BatchConfig, ChunkedProcessor

# Configure batch processing for DataFrames
config = BatchConfig(
    chunk_size=1000,
    parallel=True,
    max_workers=4,
    error_handling="skip"  # Skip failed chunks
)

# Process DataFrame in chunks
processor = ChunkedProcessor(config.chunk_size)
results = processor.process_dataframe(
    df,
    processor=lambda chunk: chunk.mean(),
    combine=lambda results: pd.concat(results)
)
```

## Key Differences

| Feature | StreamConfig | BatchConfig |
|---------|-------------|-------------|
| **Primary Use** | Database streaming | DataFrame operations |
| **Data Format** | Record objects | pandas DataFrames |
| **Error Handling** | Per-record callbacks | Chunk-level strategies |
| **Performance** | Prefetching support | Parallel processing |
| **Memory Model** | Stream-based | Chunk-based |
| **Progress Tracking** | Detailed StreamResult | Progress callbacks |

## Unified Features (Added in v1.0)

Both configurations now support:

- **Total batch tracking**: `StreamResult.total_batches` shows how many batches were processed
- **Failed indices**: `StreamResult.failed_indices` lists indices of failed records
- **Adaptive processing**: Graceful fallback from batch to individual processing

## Conversion Utilities

When you need to bridge between list-based and stream-based APIs:

```python
from dataknobs_data.streaming import StreamProcessor

# Convert list to async iterator for async streaming
records = [record1, record2, record3]
async_iter = StreamProcessor.list_to_async_iterator(records)
result = await async_db.stream_write(async_iter, config)

# Convert sync iterator to async iterator
sync_iter = iter(records)
async_iter = StreamProcessor.iterator_to_async_iterator(sync_iter)

# Batch a stream of records
batched = StreamProcessor.batch_iterator(record_iterator, batch_size=100)
```

## Best Practices

### For StreamConfig:

1. **Set appropriate batch sizes**: 
   - Smaller batches (100-500) for real-time responsiveness
   - Larger batches (1000-5000) for bulk operations

2. **Use error handlers for resilience**:
   ```python
   def continue_on_error(exc, record):
       log.warning(f"Failed to process {record.id}: {exc}")
       return True  # Continue processing
   
   config = StreamConfig(on_error=continue_on_error)
   ```

3. **Monitor progress with StreamResult**:
   ```python
   result = db.stream_write(records, config)
   if result.failed > 0:
       log.error(f"Failed records at indices: {result.failed_indices}")
   ```

### For BatchConfig:

1. **Choose chunk size based on memory**:
   ```python
   # For 1GB DataFrame, use smaller chunks
   config = BatchConfig(
       chunk_size=10000,  # Process 10k rows at a time
       memory_efficient=True
   )
   ```

2. **Use parallel processing for CPU-intensive operations**:
   ```python
   config = BatchConfig(
       parallel=True,
       max_workers=cpu_count() - 1  # Leave one CPU free
   )
   ```

3. **Handle errors appropriately**:
   - `"raise"`: Stop on first error (default)
   - `"skip"`: Skip failed chunks and continue
   - `"log"`: Log errors but include failed chunks in results

## Migration from Legacy APIs

If you're using older batch APIs, here's how to migrate:

### Old Way:
```python
# Ambiguous - is this streaming or batch?
db.batch_create(records, batch_size=1000)
```

### New Way:
```python
# Clear streaming intent
config = StreamConfig(batch_size=1000)
result = db.stream_write(records, config)

# Or clear DataFrame batch intent
config = BatchConfig(chunk_size=1000)
processor.process_dataframe(df, config)
```

## Summary

- Use **StreamConfig** for database operations and real-time streaming
- Use **BatchConfig** for pandas DataFrame operations and memory-efficient processing
- Both provide detailed error tracking with `failed_indices` and `total_batches`
- Conversion utilities help bridge between list and stream-based APIs
- Choose configuration based on your data format and processing needs