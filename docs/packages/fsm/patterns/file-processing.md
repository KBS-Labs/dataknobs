# File Processing Pattern

## Overview

The File Processing pattern provides a pre-configured FSM for processing files of various formats including CSV, JSON, XML, Parquet, and text files. It supports streaming, batch, and whole-file processing modes with built-in validation, transformation, filtering, and aggregation capabilities.

## Core Components

### FileProcessingConfig

Configuration dataclass for file processing:

```python
from dataknobs_fsm.patterns.file_processing import (
    FileProcessingConfig,
    FileFormat,
    ProcessingMode
)

config = FileProcessingConfig(
    input_path="data/input.csv",
    output_path="data/output.json",
    format=FileFormat.CSV,  # Auto-detected if not specified
    mode=ProcessingMode.STREAM,
    chunk_size=1000,
    parallel_chunks=4,
    encoding="utf-8",
    validation_schema=schema_dict,
    transformations=[transform_func1, transform_func2],
    filters=[filter_func1, filter_func2],
    aggregations={"sum": sum_func, "count": count_func}
)
```

`FileProcessingConfig` is a frozen
[`StructuredConfig`](../../common/structured-config.md) subclass: it has a
`from_dict()` / `to_dict()` pair and is **immutable** (derive a modified copy
with `dataclasses.replace(...)`). When `format` (or `output_format`) is left
unset, the format is auto-detected from the file extension by the
`FileProcessor` and stored on the processor — the config itself keeps the
caller-supplied value (`None`, meaning "auto-detect"); it is never mutated.

### File Formats

```python
from dataknobs_fsm.patterns.file_processing import FileFormat

FileFormat.JSON     # JSON and JSON Lines files
FileFormat.CSV      # CSV and TSV files
FileFormat.XML      # XML files
FileFormat.PARQUET  # Parquet files
FileFormat.TEXT     # Plain text files
FileFormat.BINARY   # Binary files
```

### Processing Modes

```python
from dataknobs_fsm.patterns.file_processing import ProcessingMode

ProcessingMode.STREAM  # Process file as stream (uses REFERENCE mode)
ProcessingMode.BATCH   # Process in batches (uses COPY mode)
ProcessingMode.WHOLE   # Load entire file (uses DIRECT mode)
```

## Basic Usage

### FileProcessor Class

```python
from dataknobs_fsm.patterns.file_processing import FileProcessor, FileProcessingConfig

# Create configuration
config = FileProcessingConfig(
    input_path="data.csv",
    output_path="processed.json",
    mode=ProcessingMode.BATCH
)

# Create processor
processor = FileProcessor(config)

# Process file
import asyncio
results = asyncio.run(processor.process())

# BATCH / WHOLE report these keys; STREAM reports total_processed /
# successful / failed instead (see Metrics).
print(f"Records processed: {results['records_processed']}")
print(f"Records written: {results['records_written']}")
print(f"Errors: {results['errors']}")
```

## Factory Functions

The pattern provides convenient factory functions for common use cases:

### CSV Processing

```python
from dataknobs_fsm.patterns.file_processing import create_csv_processor

# Create CSV processor
processor = create_csv_processor(
    input_file="data.csv",
    output_file="output.json",
    transformations=[
        lambda row: {**row, "processed": True},
        lambda row: {k: v.strip() for k, v in row.items()}
    ],
    filters=[
        lambda row: row.get("status") == "active",
        lambda row: float(row.get("value", 0)) > 100
    ]
)

# Process CSV
results = asyncio.run(processor.process())
```

### JSON Stream Processing

```python
from dataknobs_fsm.patterns.file_processing import create_json_stream_processor

# Validation schema
schema = {
    "id": {"required": True, "type": "int"},
    "name": {"required": True, "type": "str"},
    "email": {"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"}
}

# Create JSON lines processor
processor = create_json_stream_processor(
    input_file="data.jsonl",
    output_file="validated.jsonl",
    validation_schema=schema,
    chunk_size=5000
)

results = asyncio.run(processor.process())
```

### Log Analysis

```python
from dataknobs_fsm.patterns.file_processing import create_log_analyzer

# Define patterns to extract. Each pattern's named groups are merged into the
# record (a transformation step), so a matching line gains `timestamp`,
# `level`, and `message` fields.
patterns = [
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    r'(?P<level>ERROR|WARNING|INFO)',
    r'(?P<message>.*)'
]

# Create log analyzer (pattern extraction runs as a per-record transform).
processor = create_log_analyzer(
    log_file="app.log",
    output_file="analysis.json",
    patterns=patterns,
)

results = asyncio.run(processor.process())
```

> `create_log_analyzer` also accepts `aggregations=`, but remember aggregation
> is **per-record** (see [Aggregations](#aggregations)) — each function receives
> a single record, not a collection across lines. Cross-line roll-ups (e.g. a
> total error count for the whole file) are not provided by this pattern.

## Validation

### Schema-Based Validation

The validation schema supports various constraints:

```python
schema = {
    "field_name": {
        "required": True,           # Field must exist
        "type": "str",              # Python type name
        "min": 0,                   # Minimum value
        "max": 100,                 # Maximum value
        "pattern": r"^\d{3}-\d{4}$" # Regex pattern
    }
}

config = FileProcessingConfig(
    input_path="data.csv",
    validation_schema=schema
)
```

### Custom Validation

Implement custom validation through filter functions:

```python
def validate_email(record):
    """Custom email validation."""
    email = record.get("email", "")
    return "@" in email and "." in email.split("@")[1]

def validate_age(record):
    """Age validation."""
    age = record.get("age", 0)
    return 0 <= age <= 120

config = FileProcessingConfig(
    input_path="users.csv",
    filters=[validate_email, validate_age]
)
```

## Transformations

### Sequential Transformations

Transformations are applied in sequence:

```python
def normalize(record):
    """Normalize field names."""
    return {k.lower().replace(" ", "_"): v for k, v in record.items()}

def enrich(record):
    """Add computed fields."""
    record["full_name"] = f"{record.get('first_name', '')} {record.get('last_name', '')}"
    return record

def clean(record):
    """Clean data."""
    return {k: v.strip() if isinstance(v, str) else v
            for k, v in record.items()}

config = FileProcessingConfig(
    input_path="data.csv",
    transformations=[normalize, enrich, clean]
)
```

## Filtering

### Multiple Filters

All filters must pass for a record to be processed:

```python
def is_active(record):
    return record.get("status") == "active"

def has_email(record):
    return bool(record.get("email"))

def above_threshold(record):
    return float(record.get("value", 0)) > 1000

config = FileProcessingConfig(
    input_path="data.csv",
    filters=[is_active, has_email, above_threshold]
)
```

## Aggregations

### Aggregation Functions

Aggregation here is **per-record**: each aggregation callable receives the
current record dict and returns a value, and the record is replaced by
`{name: fn(record) for name, fn in aggregations.items()}`. It is a map that
produces a summary dict from each record's own fields — not a cross-record
reduce (the FSM processes records independently, so true cross-record
aggregation is not provided by this pattern).

```python
def total(record):
    # Reduce over a list field on the record.
    return sum(float(v) for v in record.get("values", []))

def item_count(record):
    return len(record.get("items", []))

config = FileProcessingConfig(
    input_path="data.csv",
    aggregations={
        "total": total,
        "item_count": item_count,
    }
)
# A record {"values": [1, 2, 3], "items": ["a", "b"]} becomes
# {"total": 6.0, "item_count": 2}.
```

## Processing Modes

### Stream Processing

Best for large files:

```python
config = FileProcessingConfig(
    input_path="large_file.csv",
    output_path="output.json",
    mode=ProcessingMode.STREAM,
    chunk_size=10000  # Process 10000 records at a time
)

processor = FileProcessor(config)
results = asyncio.run(processor.process())
```

### Batch Processing

For parallel processing with isolation:

```python
config = FileProcessingConfig(
    input_path="data.csv",
    mode=ProcessingMode.BATCH,
    chunk_size=1000,
    parallel_chunks=4  # Process 4 batches in parallel
)
```

### Whole File Processing

For small files that fit in memory:

```python
config = FileProcessingConfig(
    input_path="small_file.json",
    mode=ProcessingMode.WHOLE
)
```

## FSM Structure

The processor wires an FSM from only the **enabled** stages, connected into a
single chain that always reaches `write → complete` — so a record never
dead-ends. The always-present stages are `read` (start), `parse`, `write`, and
`complete` (end); the middle stages appear only when their config section is
set:

| Stage | Present when | Role |
|---|---|---|
| `read` | always (start) | Entry point |
| `parse` | always | Pass-through (the reader/stream executor already parses lines into records) |
| `validate` | `validation_schema` set | Gate: valid records continue; invalid route to `error` |
| `filter` | `filters` set | Gate: passing records continue; the rest route to `filtered` |
| `transform` | `transformations` set | Apply the transformations in order |
| `aggregate` | `aggregations` set | Reduce each record to a summary dict |
| `write` | always | Records here are emitted to the output |
| `complete` | always (end) | Emitting terminal — record is part of the output |
| `filtered` | `filters` set (end) | **Non-emitting** terminal — filtered records are excluded from the output |
| `error` | `validation_schema` set (end) | **Non-emitting** terminal — invalid records are excluded from the output |

A pure passthrough config therefore flows `read → parse → write → complete`,
and every record reaches the output. Disabled stages are simply absent from
the chain (never a dead-end).

The `filtered` and `error` terminals set `emit_output=False`
(`StateDefinition.emit_output`, the `emit_output:` state-config key). That flag
drives the *same* exclusion decision in every processing mode: the streaming
sink skips non-emitting terminals just as the batch/whole writers only write
records that reached `complete`. So filtering and validation behave identically
whether a record flows through the batch, whole-file, or streaming path —
**all three modes execute on the same async engine.**

The per-record functions are wired through the FSM's `custom_functions=`
channel and referenced from each state's `functions` block (transform /
aggregate) or arc `condition` (filter / validate) — the supported idiom; a
top-level `config['functions']` dict is silently dropped by `FSMConfig`.

## Complete Examples

### Example 1: CSV to JSON with Validation

```python
import asyncio
from dataknobs_fsm.patterns.file_processing import (
    FileProcessor,
    FileProcessingConfig,
    FileFormat,
    ProcessingMode
)

async def process_csv_to_json():
    # Define validation schema
    schema = {
        "id": {"required": True, "type": "int"},
        "name": {"required": True, "type": "str"},
        "age": {"type": "int", "min": 0, "max": 150},
        "email": {"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"}
    }

    # Define transformations
    def clean_data(record):
        return {k: v.strip() if isinstance(v, str) else v
                for k, v in record.items()}

    def add_timestamp(record):
        from datetime import datetime
        record["processed_at"] = datetime.now().isoformat()
        return record

    # Create configuration. BATCH (or WHOLE) mode populates the
    # records_processed / records_written / skipped / errors metrics; STREAM
    # mode reports total_processed / successful / failed instead (see Metrics).
    config = FileProcessingConfig(
        input_path="users.csv",
        output_path="users.json",
        format=FileFormat.CSV,
        output_format=FileFormat.JSON,
        mode=ProcessingMode.BATCH,
        validation_schema=schema,
        transformations=[clean_data, add_timestamp],
        filters=[lambda r: r.get("active") == "true"]
    )

    # Process file
    processor = FileProcessor(config)
    results = await processor.process()

    print("Processing complete:")
    print(f"  Records read: {results['lines_read']}")
    print(f"  Records processed: {results['records_processed']}")
    print(f"  Records written: {results['records_written']}")  # reached `complete`
    print(f"  Errors: {results['errors']}")                    # invalid records
    print(f"  Skipped: {results['skipped']}")                  # filtered records

    return results

# Run the processor
asyncio.run(process_csv_to_json())
```

### Example 2: Log Analysis (pattern extraction)

`create_log_analyzer` extracts named regex groups into each record as a
per-record transformation, so every matching line becomes a structured record.

```python
import asyncio
from dataknobs_fsm.patterns.file_processing import create_log_analyzer

async def analyze_logs():
    # Each pattern's named groups are merged into the record.
    patterns = [
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        r'\[(?P<level>\w+)\]',
        r'(?P<module>\w+):',
        r'(?P<message>.*)'
    ]

    # Create and run analyzer. The output contains one structured record per
    # matching line, e.g. {"timestamp": ..., "level": "ERROR", "module": ...,
    # "message": ...}. (Aggregation is per-record; cross-line roll-ups such as
    # a total error count are not provided by this pattern.)
    analyzer = create_log_analyzer(
        log_file="application.log",
        output_file="log_analysis.json",
        patterns=patterns,
    )

    results = await analyzer.process()
    return results

asyncio.run(analyze_logs())
```

### Example 3: Batch File Processing

```python
import asyncio
from pathlib import Path
from dataknobs_fsm.patterns.file_processing import (
    FileProcessor,
    FileProcessingConfig,
    ProcessingMode
)

async def batch_process_files():
    # Find all CSV files
    csv_files = list(Path("data/").glob("*.csv"))

    results = []
    for csv_file in csv_files:
        # Create processor for each file
        config = FileProcessingConfig(
            input_path=str(csv_file),
            output_path=str(csv_file.with_suffix('.json')),
            mode=ProcessingMode.BATCH,
            chunk_size=500,
            parallel_chunks=2
        )

        processor = FileProcessor(config)
        result = await processor.process()
        results.append({
            "file": csv_file.name,
            "processed": result['records_processed'],
            "errors": result['errors']
        })

    # Print summary
    for r in results:
        print(f"{r['file']}: {r['processed']} processed, {r['errors']} errors")

    return results

asyncio.run(batch_process_files())
```

## Metrics

`process()` returns a metrics dict. The keys it populates depend on the mode:

**BATCH and WHOLE:**

```python
metrics = {
    'lines_read': 0,         # Total lines read (BATCH only; WHOLE reads at once)
    'records_processed': 0,  # Records that reached a clean terminal (complete or filtered)
    'records_written': 0,    # Records that reached `complete` (in the output)
    'errors': 0,             # Invalid (validation `error`) + failed-transform records
    'skipped': 0,            # Filtered records (reached the `filtered` terminal)
}
```

**STREAM** merges the streaming executor's statistics instead:

```python
metrics = {
    'lines_read': 0,        # (unused in stream mode)
    'total_processed': 0,   # Records pulled through the stream
    'successful': 0,        # Records that finished without error (includes filtered)
    'failed': 0,            # Records that errored
    'duration': 0.0,        # Wall-clock seconds
    'throughput': 0.0,      # Records per second
    ...                     # plus the initial BATCH-style keys, left at 0
}
```

In every mode, only records that reach `complete` are written to the output;
filtered and invalid records are excluded (see [FSM Structure](#fsm-structure)).

## Error Handling

The file processor routes records by outcome:

1. **Parse errors** (malformed JSON lines while reading) are counted in `errors`
   during the read step.
2. **Validation failures** route the record to the non-emitting `error`
   terminal — excluded from the output and counted in `errors`.
3. **Filter rejections** route the record to the non-emitting `filtered`
   terminal — excluded from the output and counted in `skipped`.
4. **Transform errors** (a transformation raising, or returning a non-dict)
   mark the record as failed; it is not written and is counted in `errors`.
5. **Compressed output** is rejected up front with `NotImplementedError` (no
   execution path writes compressed output).

## Performance Considerations

1. **Mode Selection**:
   - Use STREAM mode for large files
   - Use BATCH mode for parallel processing
   - Use WHOLE mode only for small files

2. **Chunk Size**:
   - Larger chunks = better throughput
   - Smaller chunks = lower memory usage

3. **Parallel Processing**:
   - Set `parallel_chunks` for concurrent batch processing
   - Balance between CPU cores and I/O capacity

## Next Steps

- [ETL Pattern](etl.md) - Database ETL workflows
- [Streaming Guide](../guides/streaming.md) - Advanced streaming
- [API Reference](../../../api/dataknobs-fsm.md) - FSM configuration
- [Examples](../examples/file-processor.md) - More file processing examples