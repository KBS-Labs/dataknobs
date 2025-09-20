# End-to-End Streaming Example

This example demonstrates how the FSM framework supports **end-to-end streaming**, where data flows through the state machine network with transformations applied at each state, all while maintaining memory efficiency for large datasets.

## Overview

The FSM's streaming capabilities enable:

- **Memory-efficient processing**: Data is processed in chunks without loading entire datasets into memory
- **Real-time data streams**: Process data from APIs, message queues, or sensors as it arrives
- **Pipeline composition**: Chain multiple FSMs together in streaming pipelines
- **Backpressure management**: Automatic flow control to prevent memory overflow

## Example Code

### FSM Configuration

The streaming FSM processes data through multiple states:

```python
def create_streaming_fsm_config():
    """
    Create an FSM configuration that processes streaming data.

    Processing stages:
    1. Input → Validate: Check data validity
    2. Validate → Enrich/Error: Route based on validation
    3. Enrich → Categorize: Add calculated fields
    4. Categorize → Output: Final transformation and classification
    """
    config = {
        'name': 'StreamingProcessor',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {
                    'name': 'input',
                    'is_start': True
                },
                {
                    'name': 'validate',
                    'functions': {
                        'transform': {
                            'type': 'inline',
                            'code': """lambda state: {
                                **state.data,
                                'valid': state.data.get('value') is not None
                                        and state.data.get('value') >= 0
                            }"""
                        }
                    }
                },
                {
                    'name': 'enrich',
                    'functions': {
                        'transform': {
                            'type': 'inline',
                            'code': """lambda state: {
                                'id': state.data['id'],
                                'original_value': state.data['value'],
                                'doubled_value': state.data['value'] * 2,
                                'squared_value': state.data['value'] ** 2,
                                'category': state.data['category'],
                                'status': 'enriched'
                            }"""
                        }
                    }
                },
                {
                    'name': 'categorize',
                    'functions': {
                        'transform': {
                            'type': 'inline',
                            'code': """lambda state: {
                                **state.data,
                                'value_tier': 'high' if state.data['original_value'] > 5000
                                            else 'medium' if state.data['original_value'] > 1000
                                            else 'low',
                                'status': 'processed',
                                'risk_score': min(100, state.data['squared_value'] / 100)
                            }"""
                        }
                    }
                },
                {
                    'name': 'output',
                    'is_end': True
                },
                {
                    'name': 'error',
                    'is_end': True
                }
            ],
            'arcs': [
                {'from': 'input', 'to': 'validate'},
                {
                    'from': 'validate',
                    'to': 'enrich',
                    'condition': {
                        'type': 'inline',
                        'code': "lambda state: state.data.get('valid', True)"
                    }
                },
                {
                    'from': 'validate',
                    'to': 'error',
                    'condition': {
                        'type': 'inline',
                        'code': "lambda state: not state.data.get('valid', True)"
                    }
                },
                {'from': 'enrich', 'to': 'categorize'},
                {'from': 'categorize', 'to': 'output'}
            ]
        }]
    }

    return config
```

### File-to-File Streaming

Process large files without loading them entirely into memory:

```python
async def example_file_to_file_streaming():
    """Stream from file to file with FSM processing."""

    # Create FSM configuration
    config = create_streaming_fsm_config()

    # Initialize FSM
    fsm = AsyncSimpleFSM(config)

    # Process with streaming enabled
    results = await fsm.process_stream(
        source='input.jsonl',
        sink='output.jsonl',
        chunk_size=10,  # Process 10 records at a time
        use_streaming=True,  # Enable memory-efficient streaming
        on_progress=lambda p: print(f"Processed {p.records_processed} records...")
    )

    print(f"Streaming complete!")
    print(f"Total records: {results.get('total_records', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
```

### Real-time Stream Processing

Process data streams from APIs, queues, or sensors:

```python
async def generate_streaming_data(count: int = 1000, chunk_size: int = 100):
    """
    Simulate a streaming data source.

    In production, this could be:
    - Real-time API data
    - Message queue consumption
    - IoT sensor streams
    - Database change streams
    """
    for i in range(0, count, chunk_size):
        chunk_data = []
        for j in range(min(chunk_size, count - i)):
            record = {
                'id': i + j,
                'value': (i + j) * 10,
                'category': f'cat_{(i + j) % 5}',
                'status': 'pending'
            }
            chunk_data.append(record)

        # Yield records one by one for streaming
        for record in chunk_data:
            yield record

        # Simulate processing delay
        await asyncio.sleep(0.01)


async def example_generator_to_file_streaming():
    """Process real-time data stream."""

    config = create_streaming_fsm_config()
    fsm = AsyncSimpleFSM(config)

    # Process streaming data
    results = await fsm.process_stream(
        source=generate_streaming_data(count=50, chunk_size=10),
        sink='realtime_output.jsonl',
        chunk_size=5,
        on_progress=lambda p: print(f"Streamed {p.records_processed} records...")
    )

    print(f"Real-time processing complete!")
```

### Multi-Stage Pipeline

Chain multiple FSMs in a streaming pipeline:

```python
async def example_pipeline_streaming():
    """Multi-stage pipeline with streaming."""

    # Stage 1: Data cleaning
    stage1_config = {
        'name': 'DataCleaner',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {'name': 'input', 'is_start': True},
                {
                    'name': 'clean',
                    'functions': {
                        'transform': {
                            'type': 'inline',
                            'code': """lambda state: {
                                'id': state.data['id'],
                                'value': max(0, state.data.get('value', 0)),
                                'category': state.data.get('category', '').upper(),
                                'timestamp': __import__('time').time()
                            }"""
                        }
                    }
                },
                {'name': 'output', 'is_end': True}
            ],
            'arcs': [
                {'from': 'input', 'to': 'clean'},
                {'from': 'clean', 'to': 'output'}
            ]
        }]
    }

    # Stage 2: Processing FSM
    stage2_config = create_streaming_fsm_config()

    # Stage 1: Clean data
    stage1_fsm = AsyncSimpleFSM(stage1_config)
    stage1_results = await stage1_fsm.process_stream(
        source=generate_streaming_data(count=30, chunk_size=10),
        sink='stage1_output.jsonl',
        chunk_size=5
    )

    # Stage 2: Process cleaned data
    stage2_fsm = AsyncSimpleFSM(stage2_config)
    stage2_results = await stage2_fsm.process_stream(
        source='stage1_output.jsonl',
        sink='final_output.jsonl',
        chunk_size=5,
        use_streaming=True
    )

    print(f"Pipeline complete!")
```

## Key Features

### 1. Memory Efficiency

The streaming implementation processes data in chunks, maintaining a constant memory footprint regardless of input size:

```python
# Process a 10GB file with only 100MB of memory
results = await fsm.process_stream(
    source='huge_file.jsonl',
    sink='output.jsonl',
    chunk_size=100,  # Small chunks
    use_streaming=True
)
```

### 2. Backpressure Management

Automatic flow control prevents memory overflow when processing speed varies:

```python
stream_config = CoreStreamConfig(
    chunk_size=100,
    parallelism=4,
    memory_limit_mb=1024,  # Max 1GB memory
    backpressure_threshold=5000  # Pause at 5000 pending items
)
```

### 3. Progress Tracking

Monitor processing progress in real-time:

```python
def track_progress(progress):
    print(f"Chunks: {progress.chunks_processed}")
    print(f"Records: {progress.records_processed}")
    print(f"Rate: {progress.records_per_second:.2f} rec/s")
    print(f"Errors: {len(progress.errors)}")

results = await fsm.process_stream(
    source=data_source,
    sink=output_sink,
    on_progress=track_progress
)
```

### 4. Error Handling

The streaming pipeline handles errors gracefully:

- Invalid records are routed to error states
- Malformed data is skipped without crashing
- Partial failures don't stop the entire stream

## Use Cases

This streaming pattern is ideal for:

1. **ETL Pipelines**: Process large datasets without loading into memory
2. **Real-time Analytics**: Process streaming data as it arrives
3. **Log Processing**: Analyze log files of any size
4. **Data Migration**: Transform and move large amounts of data
5. **Event Processing**: Handle continuous event streams
6. **IoT Data**: Process sensor data in real-time

## Testing

The example includes comprehensive unit tests:

```python
# Test file-to-file streaming
async def test_file_to_file_streaming_basic():
    """Test basic file streaming with transformations."""
    # Create test data
    input_data = [
        {'id': i, 'value': i * 100, 'category': f'cat_{i % 3}'}
        for i in range(10)
    ]

    # Process through FSM
    config = create_streaming_fsm_config()
    fsm = AsyncSimpleFSM(config)

    results = await fsm.process_stream(
        source=input_file,
        sink=output_file,
        chunk_size=5,
        use_streaming=True
    )

    # Verify transformations
    with open(output_file, 'r') as f:
        output_records = [json.loads(line) for line in f]

    for record in output_records:
        assert 'original_value' in record
        assert 'doubled_value' in record
        assert 'value_tier' in record
        assert record['status'] == 'processed'
```

## Performance Considerations

1. **Chunk Size**: Balance between memory usage and processing efficiency
   - Small chunks (10-100): Lower memory, more overhead
   - Large chunks (1000-10000): Higher memory, better throughput

2. **Parallelism**: Use multiple workers for CPU-intensive transformations
   ```python
   stream_config = CoreStreamConfig(parallelism=4)
   ```

3. **Buffer Size**: Control memory usage with buffer limits
   ```python
   stream_config = CoreStreamConfig(buffer_size=10000)
   ```

## Complete Example

The full example with all patterns is available at:
`packages/fsm/examples/end_to_end_streaming.py`

Run it with:
```bash
cd packages/fsm
uv run python examples/end_to_end_streaming.py
```

Run the tests:
```bash
uv run pytest tests/test_end_to_end_streaming_example.py -v
```

## Related Documentation

- [FSM Configuration Guide](../guides/configuration.md) - Complete guide to FSM configuration
- [SimpleFSM API](../api/simple.md) - Simple FSM API documentation
- [AdvancedFSM API](../api/advanced.md) - Advanced FSM with debugging capabilities
- [API Index](../api/index.md) - Complete API documentation

## Summary

The FSM's end-to-end streaming capability enables efficient processing of large datasets and real-time streams. Data flows through the state machine network with transformations applied at each state, maintaining memory efficiency through chunk-based processing and automatic backpressure management.