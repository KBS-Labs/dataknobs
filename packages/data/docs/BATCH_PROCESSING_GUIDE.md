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