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
    mode=ProcessingMode.STREAM
)

# Create processor
processor = FileProcessor(config)

# Process file
import asyncio
results = asyncio.run(processor.process())

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

# Define patterns to extract
patterns = [
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    r'(?P<level>ERROR|WARNING|INFO)',
    r'(?P<message>.*)'
]

# Define aggregations
aggregations = {
    "error_count": lambda errors: len([e for e in errors if e == "ERROR"]),
    "warning_count": lambda warnings: len([w for w in warnings if w == "WARNING"])
}

# Create log analyzer
processor = create_log_analyzer(
    log_file="app.log",
    output_file="analysis.json",
    patterns=patterns,
    aggregations=aggregations
)

results = asyncio.run(processor.process())
```

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

Aggregate data during processing:

```python
def sum_values(values):
    return sum(float(v) for v in values if v)

def average(values):
    nums = [float(v) for v in values if v]
    return sum(nums) / len(nums) if nums else 0

def count_unique(values):
    return len(set(values))

config = FileProcessingConfig(
    input_path="data.csv",
    aggregations={
        "total": sum_values,
        "average": average,
        "unique_count": count_unique
    }
)
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

The file processor creates an FSM with the following states:

1. **read** (start) - Read input file
2. **parse** - Parse file format
3. **validate** - Validate against schema (optional)
4. **filter** - Apply filters (optional)
5. **transform** - Apply transformations (optional)
6. **aggregate** - Perform aggregations (optional)
7. **write** - Write output
8. **complete** (end) - Processing complete
9. **error** (end) - Error state

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

    # Create configuration
    config = FileProcessingConfig(
        input_path="users.csv",
        output_path="users.json",
        format=FileFormat.CSV,
        output_format=FileFormat.JSON,
        mode=ProcessingMode.STREAM,
        validation_schema=schema,
        transformations=[clean_data, add_timestamp],
        filters=[lambda r: r.get("active") == "true"]
    )

    # Process file
    processor = FileProcessor(config)
    results = await processor.process()

    print(f"Processing complete:")
    print(f"  Records read: {results['lines_read']}")
    print(f"  Records processed: {results['records_processed']}")
    print(f"  Records written: {results['records_written']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Skipped: {results['skipped']}")

    return results

# Run the processor
asyncio.run(process_csv_to_json())
```

### Example 2: Log Analysis with Aggregation

```python
import asyncio
import re
from dataknobs_fsm.patterns.file_processing import create_log_analyzer

async def analyze_logs():
    # Define patterns for log parsing
    patterns = [
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        r'\[(?P<level>\w+)\]',
        r'(?P<module>\w+):',
        r'(?P<message>.*)'
    ]

    # Define aggregation functions
    def count_by_level(levels):
        from collections import Counter
        return dict(Counter(levels))

    def extract_errors(messages):
        return [msg for msg in messages if "error" in msg.lower()]

    aggregations = {
        "level_counts": count_by_level,
        "error_messages": extract_errors
    }

    # Create and run analyzer
    analyzer = create_log_analyzer(
        log_file="application.log",
        output_file="log_analysis.json",
        patterns=patterns,
        aggregations=aggregations
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

The processor tracks the following metrics:

```python
metrics = {
    'lines_read': 0,        # Total lines read from file
    'records_processed': 0,  # Records successfully processed
    'records_written': 0,    # Records written to output
    'errors': 0,            # Processing errors
    'skipped': 0            # Records skipped (filtered out)
}
```

## Error Handling

The file processor handles errors at multiple levels:

1. **Parse Errors**: Records that fail parsing are counted in errors
2. **Validation Errors**: Invalid records go to error state
3. **Filter Rejections**: Filtered records are counted as skipped
4. **Transform Errors**: Transformation failures are logged
5. **Write Errors**: Output errors are tracked

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
- [API Reference](../api/index.md) - FSM configuration
- [Examples](../examples/file-processor.md) - More file processing examples