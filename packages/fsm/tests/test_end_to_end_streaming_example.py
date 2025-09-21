"""
Unit tests for end-to-end streaming example.

These tests verify that the streaming example functions correctly,
including file-to-file streaming, generator-based streaming, and
multi-stage pipeline processing.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, AsyncIterator
import pytest

from dataknobs_fsm import AsyncSimpleFSM
from examples.end_to_end_streaming import (
    generate_streaming_data,
    create_streaming_fsm_config
)


class TestStreamingConfiguration:
    """Test FSM configuration creation and validation."""

    def test_create_streaming_fsm_config(self):
        """Test that the streaming FSM config is valid and complete."""
        config = create_streaming_fsm_config()

        # Verify top-level structure
        assert config['name'] == 'StreamingProcessor'
        assert config['main_network'] == 'main'
        assert 'networks' in config
        assert len(config['networks']) == 1

        # Verify network structure
        network = config['networks'][0]
        assert network['name'] == 'main'
        assert 'states' in network
        assert 'arcs' in network

        # Verify states
        state_names = {state['name'] for state in network['states']}
        expected_states = {'input', 'validate', 'enrich', 'categorize', 'output', 'error'}
        assert state_names == expected_states

        # Verify initial and final states
        initial_states = [s for s in network['states'] if s.get('is_start')]
        final_states = [s for s in network['states'] if s.get('is_end')]
        assert len(initial_states) == 1
        assert initial_states[0]['name'] == 'input'
        assert len(final_states) == 2
        assert set(s['name'] for s in final_states) == {'output', 'error'}

        # Verify arcs
        assert len(network['arcs']) >= 5
        arc_pairs = [(arc['from'], arc['to']) for arc in network['arcs']]
        assert ('input', 'validate') in arc_pairs
        assert ('enrich', 'categorize') in arc_pairs
        assert ('categorize', 'output') in arc_pairs

    def test_fsm_initialization_with_config(self):
        """Test that FSM can be initialized with the streaming config."""
        config = create_streaming_fsm_config()

        # Should not raise any exceptions
        fsm = AsyncSimpleFSM(config)

        # Verify FSM is properly initialized
        states = fsm.get_states()
        assert 'input' in states
        assert 'validate' in states
        assert 'enrich' in states
        assert 'categorize' in states
        assert 'output' in states
        assert 'error' in states


class TestDataGeneration:
    """Test streaming data generation."""

    @pytest.mark.asyncio
    async def test_generate_streaming_data(self):
        """Test the streaming data generator."""
        records = []
        async for record in generate_streaming_data(count=10, chunk_size=5):
            records.append(record)

        # Verify we got all records
        assert len(records) == 10

        # Verify record structure
        for i, record in enumerate(records):
            assert 'id' in record
            assert 'value' in record
            assert 'category' in record
            assert 'status' in record
            assert record['id'] == i
            assert record['value'] == i * 10
            assert record['status'] == 'pending'
            assert record['category'].startswith('cat_')

    @pytest.mark.asyncio
    async def test_generate_streaming_data_chunking(self):
        """Test that data generation respects chunk size."""
        # Track timing to verify chunking
        chunk_boundaries = []
        record_count = 0

        async for record in generate_streaming_data(count=20, chunk_size=10):
            if record_count % 10 == 0:
                chunk_boundaries.append(record['id'])
            record_count += 1

        # Should have records at chunk boundaries
        assert chunk_boundaries == [0, 10]
        assert record_count == 20


class TestFileToFileStreaming:
    """Test file-to-file streaming with FSM processing."""

    @pytest.mark.asyncio
    async def test_file_to_file_streaming_basic(self):
        """Test basic file-to-file streaming."""
        # Create input file with test data
        input_data = [
            {'id': i, 'value': i * 100, 'category': f'cat_{i % 3}', 'status': 'raw'}
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            for record in input_data:
                input_file.write(json.dumps(record) + '\n')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            # Process the file
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            results = await fsm.process_stream(
                source=str(input_path),
                sink=str(output_path),
                chunk_size=5,
                use_streaming=True
            )

            # Verify processing results
            assert results.get('successful', 0) > 0
            assert results.get('failed', 0) == 0

            # Verify output file contents
            with open(output_path, 'r') as f:
                output_records = [json.loads(line) for line in f]

            assert len(output_records) == len(input_data)

            # Verify transformations were applied
            for i, record in enumerate(output_records):
                assert 'original_value' in record
                assert 'doubled_value' in record
                assert 'squared_value' in record
                assert 'value_tier' in record
                assert 'risk_score' in record
                assert record['status'] == 'processed'

                # Verify calculations
                expected_value = i * 100
                assert record['original_value'] == expected_value
                assert record['doubled_value'] == expected_value * 2
                assert record['squared_value'] == expected_value ** 2

                # Verify categorization
                if expected_value > 5000:
                    assert record['value_tier'] == 'high'
                elif expected_value > 1000:
                    assert record['value_tier'] == 'medium'
                else:
                    assert record['value_tier'] == 'low'

        finally:
            # Cleanup
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_file_streaming_with_invalid_data(self):
        """Test streaming with invalid data records."""
        # Create input with mix of valid and invalid data
        input_data = [
            {'id': 0, 'value': 100, 'category': 'cat_0'},  # Valid
            {'id': 1, 'value': -50, 'category': 'cat_1'},  # Invalid (negative)
            {'id': 2, 'category': 'cat_2'},                # Invalid (missing value)
            {'id': 3, 'value': 300, 'category': 'cat_0'},  # Valid
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            for record in input_data:
                input_file.write(json.dumps(record) + '\n')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            results = await fsm.process_stream(
                source=str(input_path),
                sink=str(output_path),
                chunk_size=2,
                use_streaming=True
            )

            # Read output to see what was processed
            with open(output_path, 'r') as f:
                output_records = [json.loads(line) for line in f]

            # Should have processed valid records
            # The exact behavior depends on FSM error handling
            assert len(output_records) >= 2  # At least the valid records

            # Valid records should be fully processed
            valid_records = [r for r in output_records if r.get('status') == 'processed']
            assert len(valid_records) >= 2

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class TestGeneratorStreaming:
    """Test streaming from async generators."""

    @pytest.mark.asyncio
    async def test_generator_to_file_streaming(self):
        """Test streaming from generator to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            # Use the generator as source
            results = await fsm.process_stream(
                source=generate_streaming_data(count=20, chunk_size=5),
                sink=str(output_path),
                chunk_size=5
            )

            # Verify output
            with open(output_path, 'r') as f:
                output_records = [json.loads(line) for line in f]

            assert len(output_records) == 20

            # Verify all records were processed correctly
            for i, record in enumerate(output_records):
                assert record['status'] == 'processed'
                assert 'original_value' in record
                assert 'doubled_value' in record
                assert 'value_tier' in record

        finally:
            output_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_generator_with_progress_callback(self):
        """Test streaming with progress tracking."""
        progress_updates = []

        def track_progress(progress):
            progress_updates.append({
                'records': progress.records_processed,
                'chunks': progress.chunks_processed
            })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            results = await fsm.process_stream(
                source=generate_streaming_data(count=15, chunk_size=5),
                sink=str(output_path),
                chunk_size=3,
                on_progress=track_progress
            )

            # Should have received progress updates
            assert len(progress_updates) > 0

            # Progress should increase monotonically
            for i in range(1, len(progress_updates)):
                assert progress_updates[i]['records'] >= progress_updates[i-1]['records']

        finally:
            output_path.unlink(missing_ok=True)


class TestMultiStagePipeline:
    """Test multi-stage pipeline processing."""

    @pytest.mark.asyncio
    async def test_two_stage_pipeline(self):
        """Test a two-stage processing pipeline."""
        # Stage 1: Cleaning config
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
                                    **state.data,
                                    'value': max(0, state.data.get('value', 0)),
                                    'cleaned': True
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

        # Create test data with some negative values
        test_data = [
            {'id': 0, 'value': 100, 'category': 'A'},
            {'id': 1, 'value': -50, 'category': 'B'},  # Negative - should be cleaned
            {'id': 2, 'value': 200, 'category': 'C'},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as stage1_input:
            input_path = Path(stage1_input.name)
            for record in test_data:
                stage1_input.write(json.dumps(record) + '\n')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as stage1_output:
            stage1_path = Path(stage1_output.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as final_output:
            final_path = Path(final_output.name)

        try:
            # Stage 1: Clean data
            stage1_fsm = AsyncSimpleFSM(stage1_config)
            stage1_results = await stage1_fsm.process_stream(
                source=str(input_path),
                sink=str(stage1_path),
                chunk_size=5
            )

            # Verify stage 1 output
            with open(stage1_path, 'r') as f:
                cleaned_records = [json.loads(line) for line in f]

            assert len(cleaned_records) == 3
            assert all('cleaned' in r for r in cleaned_records)
            assert cleaned_records[1]['value'] == 0  # Negative value was cleaned to 0

            # Stage 2: Process with main FSM
            stage2_config = create_streaming_fsm_config()
            stage2_fsm = AsyncSimpleFSM(stage2_config)

            stage2_results = await stage2_fsm.process_stream(
                source=str(stage1_path),
                sink=str(final_path),
                chunk_size=5,
                use_streaming=True
            )

            # Verify final output
            with open(final_path, 'r') as f:
                final_records = [json.loads(line) for line in f]

            assert len(final_records) == 3

            # Verify transformations from stage 2 were applied
            # Note: The FSM processes each record independently, so stage 1 metadata
            # may not persist through stage 2 unless explicitly preserved
            for record in final_records:
                assert 'status' in record  # From stage 2
                assert 'value_tier' in record  # From stage 2
                assert record['original_value'] >= 0  # Cleaning effect should be visible

        finally:
            input_path.unlink(missing_ok=True)
            stage1_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_pipeline_with_different_chunk_sizes(self):
        """Test pipeline with different chunk sizes at each stage."""
        # Generate test data
        test_records = 25

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as stage1_output:
            stage1_path = Path(stage1_output.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as final_output:
            final_path = Path(final_output.name)

        try:
            # Stage 1: Process with small chunks
            config1 = create_streaming_fsm_config()
            fsm1 = AsyncSimpleFSM(config1)

            results1 = await fsm1.process_stream(
                source=generate_streaming_data(count=test_records, chunk_size=5),
                sink=str(stage1_path),
                chunk_size=3  # Small chunks
            )

            # Stage 2: Process with larger chunks
            config2 = create_streaming_fsm_config()
            fsm2 = AsyncSimpleFSM(config2)

            results2 = await fsm2.process_stream(
                source=str(stage1_path),
                sink=str(final_path),
                chunk_size=10,  # Larger chunks
                use_streaming=True
            )

            # Verify all records made it through
            with open(final_path, 'r') as f:
                final_records = [json.loads(line) for line in f]

            assert len(final_records) == test_records

            # Verify records are properly processed
            for record in final_records:
                assert 'status' in record
                assert 'value_tier' in record

        finally:
            stage1_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)


class TestStreamingPerformance:
    """Test streaming performance and memory efficiency."""

    @pytest.mark.asyncio
    async def test_large_file_streaming(self):
        """Test that streaming handles large files efficiently."""
        # Create a "large" file (1000 records)
        record_count = 1000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            for i in range(record_count):
                record = {
                    'id': i,
                    'value': i * 10,
                    'category': f'cat_{i % 10}',
                    'data': 'x' * 100  # Add some bulk to each record
                }
                input_file.write(json.dumps(record) + '\n')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            # Process with streaming enabled
            results = await fsm.process_stream(
                source=str(input_path),
                sink=str(output_path),
                chunk_size=50,  # Process in reasonable chunks
                use_streaming=True
            )

            # Verify all records were processed
            with open(output_path, 'r') as f:
                line_count = sum(1 for _ in f)

            assert line_count == record_count

            # Spot check some records
            with open(output_path, 'r') as f:
                # Check first record (id=0, value=0)
                first = json.loads(f.readline())
                assert first['original_value'] == 0
                assert first['status'] == 'processed'

                # Skip to middle (record 499, value=4990)
                for _ in range(498):
                    f.readline()
                middle = json.loads(f.readline())
                assert middle['original_value'] == 4990

                # Skip to last record (record 999, value=9990)
                for _ in range(499):
                    f.readline()
                last = json.loads(f.readline())
                assert last['original_value'] == 9990

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class TestErrorHandling:
    """Test error handling in streaming scenarios."""

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON in input files."""
        # Create input file with some malformed JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            input_file.write('{"id": 0, "value": 100}\n')  # Valid
            input_file.write('not valid json\n')           # Invalid
            input_file.write('{"id": 2, "value": 300}\n')  # Valid

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            # Should handle malformed JSON gracefully
            results = await fsm.process_stream(
                source=str(input_path),
                sink=str(output_path),
                chunk_size=5,
                use_streaming=True
            )

            # Check that valid records were still processed
            with open(output_path, 'r') as f:
                output_records = [json.loads(line) for line in f if line.strip()]

            # Should have processed at least the valid records
            assert len(output_records) >= 2

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_file_handling(self):
        """Test handling of empty input files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_path = Path(input_file.name)
            # File is empty

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            config = create_streaming_fsm_config()
            fsm = AsyncSimpleFSM(config)

            # Should handle empty file gracefully
            results = await fsm.process_stream(
                source=str(input_path),
                sink=str(output_path),
                chunk_size=5,
                use_streaming=True
            )

            # Output should also be empty
            assert output_path.stat().st_size == 0

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])