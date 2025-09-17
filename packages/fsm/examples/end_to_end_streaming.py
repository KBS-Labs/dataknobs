"""
End-to-End Streaming Example for FSM

This example demonstrates how to stream data through an FSM network where:
1. Input is streamed from a source (file, database, or custom iterator)
2. Data flows through FSM states with transformations applied
3. Processed output is streamed to a sink (file, database, etc.)

The streaming is memory-efficient, processing data in chunks without loading
the entire dataset into memory at once.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, Any

from dataknobs_fsm import SimpleFSM, AsyncSimpleFSM


async def generate_streaming_data(count: int = 1000, chunk_size: int = 100) -> AsyncIterator[Dict[str, Any]]:
    """
    Simulate a streaming data source.

    This could be replaced with:
    - Real-time API data
    - Database query results
    - Large file reading
    - Message queue consumption
    """
    for i in range(0, count, chunk_size):
        # Simulate chunk of records
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

        # Simulate some processing delay
        await asyncio.sleep(0.01)


def create_streaming_fsm_config():
    """
    Create an FSM configuration that processes streaming data.

    This FSM:
    1. Validates incoming records
    2. Enriches data with calculations
    3. Categorizes based on value thresholds
    4. Formats output for downstream systems
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
                                'valid': state.data.get('value') is not None and state.data.get('value') >= 0
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
                                'value_tier': 'high' if state.data['original_value'] > 5000 else 'medium' if state.data['original_value'] > 1000 else 'low',
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
                {
                    'from': 'input',
                    'to': 'validate'
                },
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
                {
                    'from': 'enrich',
                    'to': 'categorize'
                },
                {
                    'from': 'categorize',
                    'to': 'output'
                }
            ]
        }]
    }

    return config


async def example_file_to_file_streaming():
    """
    Example 1: Stream from file to file with FSM processing.

    This demonstrates the most common use case: processing large files
    without loading them entirely into memory.
    """
    print("\n" + "="*60)
    print("Example 1: File-to-File Streaming with FSM Processing")
    print("="*60)

    # Create temporary input file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
        input_path = Path(input_file.name)

        # Write test data (simulating a large file)
        for i in range(100):
            record = {
                'id': i,
                'value': i * 100,
                'category': f'cat_{i % 3}',
                'status': 'raw'
            }
            input_file.write(json.dumps(record) + '\n')

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        # Create FSM configuration
        config = create_streaming_fsm_config()

        # Initialize FSM
        fsm = AsyncSimpleFSM(config)

        # Process with streaming enabled
        print(f"Processing {input_path} -> {output_path}")
        print("Streaming mode: ENABLED (memory-efficient)")

        results = await fsm.process_stream(
            source=str(input_path),
            sink=str(output_path),
            chunk_size=10,  # Process 10 records at a time
            use_streaming=True,  # Enable memory-efficient streaming
            on_progress=lambda p: print(f"  Processed {p.records_processed} records...")
        )

        print(f"\nStreaming complete!")
        print(f"  Total records: {results.get('total_records', 0)}")
        print(f"  Successful: {results.get('successful', 0)}")
        print(f"  Failed: {results.get('failed', 0)}")

        # Read and display a few output records
        print("\nSample output records:")
        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 records
                    record = json.loads(line)
                    print(f"  {json.dumps(record, indent=2)}")
                elif i == 3:
                    print("  ...")
                    break

    finally:
        # Cleanup
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


async def example_generator_to_file_streaming():
    """
    Example 2: Stream from async generator to file.

    This demonstrates processing real-time data streams like:
    - API responses
    - Message queues
    - Database cursors
    - Live sensor data
    """
    print("\n" + "="*60)
    print("Example 2: Real-time Stream Processing (Generator to File)")
    print("="*60)

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        # Create FSM configuration
        config = create_streaming_fsm_config()

        # Initialize FSM
        fsm = AsyncSimpleFSM(config)

        print("Processing real-time data stream...")
        print("Simulating data from API/Queue/Sensor...")

        # Process streaming data
        results = await fsm.process_stream(
            source=generate_streaming_data(count=50, chunk_size=10),
            sink=str(output_path),
            chunk_size=5,  # Process 5 records at a time
            on_progress=lambda p: print(f"  Streamed {p.records_processed} records...")
        )

        print(f"\nReal-time processing complete!")
        print(f"  Total records: {results.get('total_records', 0)}")
        print(f"  Processing time: {results.get('processing_time', 0):.2f}s")

        # Verify output
        with open(output_path, 'r') as f:
            lines = f.readlines()
            print(f"  Output file contains {len(lines)} processed records")

    finally:
        # Cleanup
        output_path.unlink(missing_ok=True)


async def example_pipeline_streaming():
    """
    Example 3: Multi-stage pipeline with streaming.

    This demonstrates how to chain multiple FSMs in a streaming pipeline,
    where output from one FSM streams directly into the next.
    """
    print("\n" + "="*60)
    print("Example 3: Multi-Stage Streaming Pipeline")
    print("="*60)

    # Stage 1 FSM: Data cleaning and normalization
    stage1_config = {
        'name': 'DataCleaner',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {
                    'name': 'input',
                    'is_start': True
                },
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
                {
                    'name': 'output',
                    'is_end': True
                }
            ],
            'arcs': [
                {'from': 'input', 'to': 'clean'},
                {'from': 'clean', 'to': 'output'}
            ]
        }]
    }

    # Stage 2 FSM: Use our main processing FSM
    stage2_config = create_streaming_fsm_config()

    # Create temporary files for intermediate results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as stage1_output:
        stage1_path = Path(stage1_output.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as final_output:
        final_path = Path(final_output.name)

    try:
        print("Stage 1: Data Cleaning")
        stage1_fsm = AsyncSimpleFSM(stage1_config)

        # Process stage 1
        stage1_results = await stage1_fsm.process_stream(
            source=generate_streaming_data(count=30, chunk_size=10),
            sink=str(stage1_path),
            chunk_size=5
        )
        print(f"  Cleaned {stage1_results.get('total_records', 0)} records")

        print("\nStage 2: Data Processing")
        stage2_fsm = AsyncSimpleFSM(stage2_config)

        # Stream stage 1 output to stage 2
        stage2_results = await stage2_fsm.process_stream(
            source=str(stage1_path),
            sink=str(final_path),
            chunk_size=5,
            use_streaming=True
        )
        print(f"  Processed {stage2_results.get('total_records', 0)} records")

        print("\nPipeline complete!")
        print(f"  Final output: {final_path}")

        # Show sample final output
        with open(final_path, 'r') as f:
            first_line = f.readline()
            if first_line:
                record = json.loads(first_line)
                print("\nSample final record:")
                print(json.dumps(record, indent=2))

    finally:
        # Cleanup
        stage1_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


async def main():
    """Run all streaming examples."""
    print("\n" + "#"*60)
    print("# FSM End-to-End Streaming Examples")
    print("#"*60)
    print("\nThese examples demonstrate how data can be streamed through")
    print("an FSM network with transformations applied at each state,")
    print("all while maintaining memory efficiency for large datasets.")

    # Run examples
    await example_file_to_file_streaming()
    await example_generator_to_file_streaming()
    await example_pipeline_streaming()

    print("\n" + "#"*60)
    print("# All Examples Complete!")
    print("#"*60)
    print("\nKey Features Demonstrated:")
    print("- Memory-efficient chunk-based processing")
    print("- Real-time data stream handling")
    print("- Multi-stage pipeline composition")
    print("- Progress tracking and monitoring")
    print("- Automatic backpressure management")


if __name__ == "__main__":
    asyncio.run(main())