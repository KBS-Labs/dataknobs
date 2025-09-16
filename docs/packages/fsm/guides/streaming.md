# Streaming Guide

## Overview

The DataKnobs FSM package provides native streaming support for processing large datasets that don't fit in memory. The streaming module includes support for file and database streaming with backpressure management, parallel processing, and comprehensive metrics.

## Understanding Streaming

Streaming in FSM allows you to:
- Process files larger than available memory
- Handle continuous data streams
- Implement backpressure for flow control
- Process data in parallel chunks
- Track processing metrics and progress

### Key Concepts

- **StreamChunk**: Data is processed in manageable chunks
- **Backpressure**: Automatic flow control prevents memory overflow
- **Parallelism**: Process multiple chunks concurrently
- **Metrics**: Track throughput, errors, and performance

## Core Components

### StreamConfig

Configuration for stream processing:

```python
from dataknobs_fsm.streaming.core import StreamConfig

config = StreamConfig(
    chunk_size=1000,           # Items per chunk
    buffer_size=10000,         # Maximum items to buffer
    parallelism=4,             # Number of parallel workers
    memory_limit_mb=512,       # Memory limit in MB
    backpressure_threshold=5000,  # Queue size triggering backpressure
    timeout_seconds=300,       # Maximum processing time
    enable_metrics=True,       # Collect metrics
    retry_on_error=True,       # Retry failed chunks
    max_retries=3             # Maximum retry attempts
)
```

### StreamChunk

The basic unit of streaming data:

```python
from dataknobs_fsm.streaming.core import StreamChunk
import time

chunk = StreamChunk(
    data=[item1, item2, item3],  # Chunk data
    chunk_id="chunk_001",         # Unique identifier
    sequence_number=0,            # Position in stream
    metadata={"source": "file"},  # Additional metadata
    timestamp=time.time(),        # Creation time
    is_last=False                # Last chunk indicator
)
```

### StreamMetrics

Track streaming performance:

```python
from dataknobs_fsm.streaming.core import StreamMetrics

metrics = StreamMetrics()
print(f"Chunks processed: {metrics.chunks_processed}")
print(f"Items processed: {metrics.items_processed}")
print(f"Throughput: {metrics.throughput_items_per_second()} items/sec")
print(f"Duration: {metrics.duration_seconds()}s")
print(f"Errors: {metrics.errors_count}")
```

## Stream Sources

### File Stream Source

Stream data from files with automatic format detection:

```python
from dataknobs_fsm.streaming.file_stream import FileStreamSource, FileFormat

# Auto-detect format from extension
source = FileStreamSource(
    file_path="data.csv",
    chunk_size=1000
)

# Explicit format specification
source = FileStreamSource(
    file_path="data.jsonl",
    format=FileFormat.JSONL,
    chunk_size=5000,
    encoding="utf-8"
)

# With compression
source = FileStreamSource(
    file_path="data.csv.gz",
    compression="gzip",
    chunk_size=1000
)

# Iterate over chunks
for chunk in source:
    print(f"Processing chunk {chunk.chunk_id} with {len(chunk.data)} items")
```

#### Supported File Formats

```python
from dataknobs_fsm.streaming.file_stream import FileFormat

FileFormat.JSON   # JSON files (entire file loaded)
FileFormat.JSONL  # JSON Lines (streaming)
FileFormat.CSV    # CSV/TSV files
FileFormat.TEXT   # Plain text files
FileFormat.BINARY # Binary files
```

### Database Stream Source

Stream query results from databases:

```python
from dataknobs_fsm.streaming.db_stream import DatabaseStreamSource

# Create database stream
source = DatabaseStreamSource(
    database=db_connection,  # dataknobs_data database
    query="SELECT * FROM large_table",
    chunk_size=5000,
    fetch_size=10000  # Database fetch size
)

# Process database records
for chunk in source:
    records = chunk.data  # List of Record objects
    print(f"Processing {len(records)} records")
```

### Memory Stream Source

For testing and small datasets:

```python
from dataknobs_fsm.streaming.core import MemoryStreamSource

# Create in-memory source
data = [{"id": i, "value": i*2} for i in range(10000)]
source = MemoryStreamSource(data, chunk_size=100)

# Use like any other source
for chunk in source:
    process(chunk)
```

## Stream Sinks

### File Stream Sink

Write processed data to files:

```python
from dataknobs_fsm.streaming.file_stream import FileStreamSink

# Create file sink
sink = FileStreamSink(
    file_path="output.jsonl",
    format=FileFormat.JSONL,
    mode="w",  # Write mode (w, a)
    encoding="utf-8"
)

# Write chunks
for chunk in processed_chunks:
    success = sink.write_chunk(chunk)
    if not success:
        handle_error()

# Always flush and close
sink.flush()
sink.close()
```

### Memory Stream Sink

Collect results in memory:

```python
from dataknobs_fsm.streaming.core import MemoryStreamSink

sink = MemoryStreamSink()

# Process and collect
for chunk in source:
    processed = transform(chunk)
    sink.write_chunk(processed)

# Access collected data
all_records = sink.records
all_chunks = sink.chunks
```

## Stream Processing

### Basic Stream Processing

Use `BasicStreamProcessor` for simple streaming:

```python
from dataknobs_fsm.streaming.core import BasicStreamProcessor
from dataknobs_fsm.streaming.file_stream import FileStreamSource, FileStreamSink

# Create processor
processor = BasicStreamProcessor(
    source=FileStreamSource("input.csv"),
    sink=FileStreamSink("output.json"),
    transform_func=lambda chunk: transform_data(chunk),
    buffer_size=1000
)

# Process stream
results = processor.process()
print(f"Processed {results['processed_chunks']} chunks")
print(f"Total records: {results['processed_records']}")
print(f"Duration: {results['duration']}s")
print(f"Errors: {results['errors']}")
```

### Stream Context

Advanced streaming with parallelism and backpressure:

```python
from dataknobs_fsm.streaming.core import StreamContext, StreamConfig

# Configure streaming
config = StreamConfig(
    chunk_size=1000,
    parallelism=4,
    backpressure_threshold=5000
)

# Create context
context = StreamContext(config)

# Add processors
context.add_processor(validate_chunk)
context.add_processor(transform_chunk)
context.add_processor(enrich_chunk)

# Stream with context
source = FileStreamSource("input.csv")
sink = FileStreamSink("output.json")

metrics = context.stream(
    source=source,
    sink=sink,
    transform=optional_transform
)

print(f"Throughput: {metrics.throughput_mb_per_second()} MB/s")
```

### Async Stream Processing

For async/await environments:

```python
from dataknobs_fsm.streaming.core import AsyncStreamContext
import asyncio

async def process_stream_async():
    context = AsyncStreamContext(config)

    # Async source iterator
    async def source_iterator():
        for chunk in source:
            yield chunk

    # Sink function
    def sink_function(chunk):
        return sink.write_chunk(chunk)

    # Process asynchronously
    metrics = await context.stream_async(
        source=source_iterator(),
        sink=sink_function,
        transform=transform_func
    )

    return metrics

# Run async processing
metrics = asyncio.run(process_stream_async())
```

## Backpressure Management

### Understanding Backpressure

Backpressure prevents memory overflow when processing can't keep up with input:

```python
config = StreamConfig(
    buffer_size=10000,           # Maximum buffer size
    backpressure_threshold=5000  # Trigger backpressure at 50% full
)

context = StreamContext(config)

# Backpressure is handled automatically:
# - Input is paused when buffers exceed threshold
# - Processing resumes when buffers drain
# - Status changes to PAUSED during backpressure
```

### Manual Chunk Management

For fine-grained control:

```python
context = StreamContext()

# Add chunks manually
success = context.add_chunk(chunk)
if not success:
    # Queue is full, handle backpressure
    time.sleep(0.1)

# Or add data directly
success = context.add_data(
    data=record_list,
    chunk_id="custom_001",
    is_last=False
)

# Get processed chunks
processed_chunk = context.get_next_chunk()
```

## Parallel Processing

### Configure Parallelism

Process multiple chunks simultaneously:

```python
config = StreamConfig(
    parallelism=4,      # 4 worker threads
    chunk_size=1000,    # Each worker processes 1000 items
    buffer_size=20000   # Buffer for parallel processing
)

context = StreamContext(config)

# Workers process chunks in parallel
# Results maintain order if needed
```

### Worker Thread Pattern

The StreamContext uses worker threads internally:

```python
# StreamContext creates worker threads that:
# 1. Get chunks from input queue
# 2. Process through registered processors
# 3. Put results in output queue
# 4. Handle errors and retries
# 5. Track metrics

# This happens automatically when you call:
metrics = context.stream(source, sink)
```

## Error Handling

### Retry Configuration

Configure automatic retries for failed chunks:

```python
config = StreamConfig(
    retry_on_error=True,
    max_retries=3
)

# Failed chunks are retried automatically
# Metrics track retry counts
```

### Error Collection

Track errors during processing:

```python
processor = BasicStreamProcessor(source, sink)
results = processor.process()

if results['errors']:
    for error in results['errors']:
        logger.error(f"Processing error: {error}")

# Check success
if not results['success']:
    handle_failure()
```

## Stream Patterns in FSM

### File Processing Pattern

The file processing pattern uses streaming internally:

```python
from dataknobs_fsm.patterns.file_processing import FileProcessor, ProcessingMode

# Stream mode for large files
processor = FileProcessor(config)
processor.config.mode = ProcessingMode.STREAM  # Uses streaming

# This automatically:
# - Creates FileStreamSource
# - Processes in chunks
# - Manages memory efficiently
```

### Database ETL Pattern

Stream database records:

```python
from dataknobs_fsm.patterns.etl import ETLProcessor

# ETL with streaming
etl = ETLProcessor(
    source_db=source_connection,
    target_db=target_connection,
    chunk_size=10000
)

# Streams from source to target
etl.process()
```

## Complete Examples

### Example 1: CSV to JSON Conversion

```python
from dataknobs_fsm.streaming.core import BasicStreamProcessor
from dataknobs_fsm.streaming.file_stream import (
    FileStreamSource, FileStreamSink, FileFormat
)

def csv_to_json():
    """Convert CSV file to JSON Lines format."""

    # Define transformation
    def transform_chunk(chunk):
        # chunk.data contains list of dict from CSV
        for record in chunk.data:
            # Clean and transform each record
            record['processed'] = True
            record['timestamp'] = chunk.timestamp

        return chunk

    # Create processor
    processor = BasicStreamProcessor(
        source=FileStreamSource("input.csv", format=FileFormat.CSV),
        sink=FileStreamSink("output.jsonl", format=FileFormat.JSONL),
        transform_func=transform_chunk
    )

    # Process
    results = processor.process()

    print(f"Conversion complete:")
    print(f"  Chunks: {results['processed_chunks']}")
    print(f"  Records: {results['processed_records']}")
    print(f"  Duration: {results['duration']:.2f}s")

    return results

csv_to_json()
```

### Example 2: Parallel Stream Processing

```python
from dataknobs_fsm.streaming.core import StreamContext, StreamConfig
from dataknobs_fsm.streaming.file_stream import FileStreamSource, FileStreamSink
import time

def parallel_processing():
    """Process large file with parallel workers."""

    # Configure parallel processing
    config = StreamConfig(
        chunk_size=5000,
        parallelism=4,
        buffer_size=50000,
        backpressure_threshold=25000,
        enable_metrics=True
    )

    context = StreamContext(config)

    # Add processing pipeline
    def validate(chunk):
        # Validate records
        valid_data = [r for r in chunk.data if validate_record(r)]
        chunk.data = valid_data
        return chunk

    def enrich(chunk):
        # Enrich with additional data
        for record in chunk.data:
            record['enriched'] = lookup_data(record['id'])
        return chunk

    def transform(chunk):
        # Transform format
        chunk.data = [transform_record(r) for r in chunk.data]
        return chunk

    context.add_processor(validate)
    context.add_processor(enrich)
    context.add_processor(transform)

    # Process stream
    source = FileStreamSource("large_dataset.csv")
    sink = FileStreamSink("processed_data.jsonl")

    start = time.time()
    metrics = context.stream(source, sink)
    duration = time.time() - start

    print(f"Parallel processing complete:")
    print(f"  Workers: {config.parallelism}")
    print(f"  Chunks: {metrics.chunks_processed}")
    print(f"  Items: {metrics.items_processed}")
    print(f"  Throughput: {metrics.throughput_items_per_second():.0f} items/sec")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Errors: {metrics.errors_count}")
    print(f"  Retries: {metrics.retries_count}")

parallel_processing()
```

### Example 3: Database to File Export

```python
from dataknobs_fsm.streaming.db_stream import DatabaseStreamSource
from dataknobs_fsm.streaming.file_stream import FileStreamSink, FileFormat
from dataknobs_fsm.streaming.core import StreamContext

async def export_database_to_file(db_connection):
    """Export large table to compressed JSON file."""

    # Create database source
    source = DatabaseStreamSource(
        database=db_connection,
        query="""
            SELECT id, name, email, created_at
            FROM users
            WHERE active = true
            ORDER BY created_at
        """,
        chunk_size=10000,
        fetch_size=50000
    )

    # Create compressed file sink
    sink = FileStreamSink(
        file_path="users_export.jsonl.gz",
        format=FileFormat.JSONL,
        compression="gzip"
    )

    # Stream with progress tracking
    context = StreamContext()

    def transform_records(chunk):
        # Convert Records to dicts
        chunk.data = [record.to_dict() for record in chunk.data]
        return chunk

    context.add_processor(transform_records)

    # Process
    metrics = context.stream(source, sink)

    print(f"Export complete:")
    print(f"  Records exported: {metrics.items_processed}")
    print(f"  File size: {sink.bytes_written / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {sink.compression_ratio:.2f}")

    return metrics
```

## Monitoring and Metrics

### Stream Status

Track stream processing status:

```python
from dataknobs_fsm.streaming.core import StreamStatus

context = StreamContext()

# Check status during processing
if context.status == StreamStatus.ACTIVE:
    print("Processing...")
elif context.status == StreamStatus.PAUSED:
    print("Backpressure active")
elif context.status == StreamStatus.ERROR:
    print("Error occurred")
elif context.status == StreamStatus.COMPLETED:
    print("Processing complete")
```

### Metrics Analysis

Analyze streaming performance:

```python
metrics = context.stream(source, sink)

# Performance metrics
print(f"Duration: {metrics.duration_seconds():.2f}s")
print(f"Chunks/sec: {metrics.chunks_processed / metrics.duration_seconds():.2f}")
print(f"Items/sec: {metrics.throughput_items_per_second():.0f}")
print(f"MB/sec: {metrics.throughput_mb_per_second():.2f}")

# Memory metrics
print(f"Peak memory: {metrics.peak_memory_mb:.2f} MB")

# Error metrics
print(f"Error rate: {metrics.errors_count / metrics.chunks_processed * 100:.2f}%")
print(f"Retry rate: {metrics.retries_count / metrics.chunks_processed * 100:.2f}%")
```

## Best Practices

### 1. Choose Appropriate Chunk Size

```python
# Small chunks for low latency
config = StreamConfig(chunk_size=100)

# Large chunks for throughput
config = StreamConfig(chunk_size=10000)

# Balance based on:
# - Memory constraints
# - Processing complexity
# - Network/IO latency
```

### 2. Configure Buffers Properly

```python
# High-throughput configuration
config = StreamConfig(
    buffer_size=100000,
    backpressure_threshold=50000
)

# Memory-constrained configuration
config = StreamConfig(
    buffer_size=1000,
    backpressure_threshold=500
)
```

### 3. Handle Errors Gracefully

```python
# Always check results
results = processor.process()
if not results['success']:
    logger.error(f"Processing failed: {results['errors']}")
    # Implement recovery logic
```

### 4. Monitor Memory Usage

```python
# Set memory limits
config = StreamConfig(
    memory_limit_mb=512,
    chunk_size=1000  # Adjust based on memory
)
```

### 5. Clean Up Resources

```python
# Use context managers
with context.streaming_context() as ctx:
    ctx.stream(source, sink)
# Automatic cleanup

# Or manual cleanup
try:
    context.stream(source, sink)
finally:
    context.close()
    source.close()
    sink.close()
```

## Common Pitfalls

### 1. Not Closing Resources

```python
# ❌ Bad - resources not closed
processor.process()

# ✅ Good - proper cleanup
try:
    results = processor.process()
finally:
    # process() closes source and sink automatically
    pass
```

### 2. Ignoring Backpressure

```python
# ❌ Bad - no buffer limits
config = StreamConfig(buffer_size=None)  # Unlimited!

# ✅ Good - reasonable limits
config = StreamConfig(
    buffer_size=10000,
    backpressure_threshold=5000
)
```

### 3. Wrong Chunk Size

```python
# ❌ Bad - chunk too large for memory
config = StreamConfig(chunk_size=1000000)  # 1M items!

# ✅ Good - reasonable chunk size
config = StreamConfig(chunk_size=1000)
```

## Troubleshooting

### Common Issues

1. **Memory Growth**
   - Reduce chunk_size
   - Lower buffer_size
   - Enable backpressure

2. **Slow Processing**
   - Increase parallelism
   - Optimize transform functions
   - Increase chunk_size

3. **Backpressure Triggered**
   - Increase buffer_size
   - Optimize processing speed
   - Add more workers

4. **File Format Errors**
   - Verify file format
   - Check encoding
   - Validate data structure

## Conclusion

The DataKnobs FSM streaming capabilities enable:

- **Scalability**: Process datasets of any size
- **Efficiency**: Optimal memory usage with backpressure
- **Performance**: Parallel processing support
- **Reliability**: Error handling and retries
- **Flexibility**: Multiple formats and sources

Use streaming when dealing with large datasets or continuous data flows.

## Next Steps

- [File Processing Pattern](../patterns/file-processing.md) - File processing with streaming
- [ETL Pattern](../patterns/etl.md) - Database streaming ETL
- [CLI Guide](../guides/cli.md) - Command-line interface guide
- [API Reference](../api/index.md) - Streaming API documentation

