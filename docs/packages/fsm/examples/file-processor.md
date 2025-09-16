# Large File Processing Example

This example demonstrates how to process large files efficiently using the FSM framework with REFERENCE mode and streaming capabilities.

## Overview

The example showcases:

- **REFERENCE mode** for handling large files without loading into memory
- **Streaming processing** for chunks of data
- **Parallel processing** of file chunks
- **Progress tracking** and statistics collection
- **Error handling** for partial failures

## Source Code

The complete example is available at: [`packages/fsm/examples/large_file_processor.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/fsm/examples/large_file_processor.py)

## Implementation Details

### FSM Configuration

The example uses SimpleFSM with REFERENCE mode for memory efficiency:

```python
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

def create_file_processor_fsm() -> SimpleFSM:
    """Create and configure the file processor FSM."""
    config = {
        "name": "large_file_processor",
        "states": [
            {"name": "start", "initial": True},
            {"name": "initialize"},
            {"name": "read_chunk"},
            {"name": "process_chunk"},
            {"name": "aggregate_results"},
            {"name": "write_output"},
            {"name": "complete", "terminal": True},
            {"name": "error", "terminal": True}
        ],
        "arcs": [...]
    }

    # Use REFERENCE mode for large files
    fsm = SimpleFSM(
        config,
        data_mode=DataHandlingMode.REFERENCE,
        custom_functions={
            "initialize_processing": initialize_processing,
            "read_next_chunk": read_next_chunk,
            "process_chunk_data": process_chunk_data,
            "aggregate_chunk_results": aggregate_chunk_results,
            "write_results": write_results,
            "finalize_processing": finalize_processing
        }
    )

    return fsm
```

### Key Functions

#### Initialize Processing

Sets up the file processing context with streaming support:

```python
def initialize_processing(state) -> Dict[str, Any]:
    """Initialize the file processing context."""
    data = state.data.copy()

    # Get file reference
    file_ref = data.get('file_reference')

    # Set up processing context
    data['processing'] = {
        'total_lines': 0,
        'processed_lines': 0,
        'failed_lines': 0,
        'chunks_processed': 0,
        'errors': [],
        'statistics': {
            'min_value': None,
            'max_value': None,
            'sum': 0,
            'count': 0
        }
    }

    # Configure chunk size
    data['chunk_size'] = data.get('chunk_size', 1000)
    data['current_offset'] = 0

    return data
```

#### Stream Processing

Processes the file in chunks to handle large datasets:

```python
def read_next_chunk(state) -> Dict[str, Any]:
    """Read the next chunk of data from the file."""
    data = state.data.copy()
    file_ref = data['file_reference']
    chunk_size = data['chunk_size']

    # Stream the file chunk by chunk
    with open(file_ref.file_path, 'r') as f:
        # Seek to current offset
        f.seek(data['current_offset'])

        # Read chunk
        chunk_data = []
        for _ in range(chunk_size):
            line = f.readline()
            if not line:
                data['end_of_file'] = True
                break
            chunk_data.append(line.strip())

        data['current_chunk'] = chunk_data
        data['current_offset'] = f.tell()

    return data
```

## Usage

```python
import asyncio
from file_reference import FileReference

# Create file reference (not loaded into memory)
file_ref = FileReference(
    file_path="large_dataset.csv",
    file_type="csv",
    metadata={"encoding": "utf-8"}
)

# Create and run the processor
fsm = create_file_processor_fsm()
result = fsm.process({
    "file_reference": file_ref,
    "chunk_size": 10000,
    "output_file": "processed_results.json"
})

print(f"Processed {result['data']['processing']['processed_lines']} lines")
print(f"Statistics: {result['data']['processing']['statistics']}")
```

## Benefits of This Approach

1. **Memory Efficiency**: Files are never fully loaded into memory
2. **Scalability**: Can handle files of any size
3. **Resilience**: Partial failures don't affect the entire process
4. **Performance**: Chunks can be processed in parallel
5. **Flexibility**: Works with various file formats (CSV, JSONL, etc.)