"""Tests for the large file processor example.

This test validates that the large file processor example works correctly,
including REFERENCE mode handling, chunk processing, and statistics aggregation.
"""

import pytest
import sys
import json
import csv
import tempfile
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from large_file_processor import (
    initialize_processing,
    process_chunk,
    aggregate_results,
    check_initialization_success,
    check_processing_complete,
    mark_success,
    mark_failure,
    calculate_file_hash,
    create_sample_file,
    simulate_chunked_reading,
    update_statistics,
    config
)
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


class TestLargeFileProcessor:
    """Test the large file processor example."""
    
    @pytest.fixture
    def fsm(self):
        """Create FSM with custom processing functions."""
        return SimpleFSM(
            config,
            data_mode=DataHandlingMode.REFERENCE,
            custom_functions={
                'initialize_processing': initialize_processing,
                'process_chunk': process_chunk,
                'aggregate_results': aggregate_results,
                'check_initialization_success': check_initialization_success,
                'check_processing_complete': check_processing_complete,
                'mark_success': mark_success,
                'mark_failure': mark_failure
            }
        )
    
    @pytest.fixture
    def temp_jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(100):
                record = {'id': i, 'value': i * 2.5, 'name': f'item_{i}'}
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['value', 'name', 'category'])
            for i in range(100):
                writer.writerow([i * 2.5, f'item_{i}', f'cat_{i % 5}'])
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_initialize_processing(self, temp_jsonl_file):
        """Test initialization of file processing."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        state = MockState({'file_reference': str(temp_jsonl_file)})
        result = initialize_processing(state)
        
        assert 'processing' in result
        assert result['file_type'] == 'jsonl'
        assert result['file_path'] == str(temp_jsonl_file)
        assert result['file_name'] == temp_jsonl_file.name
        assert result['file_size'] > 0
        assert result['processing']['total_lines'] == 0
        assert result['processing']['processed_lines'] == 0
    
    def test_process_jsonl_chunk(self):
        """Test processing a JSONL chunk."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        chunk_data = [
            '{"id": 1, "value": 10}',
            '{"id": 2, "value": 20}',
            '{"id": 3, "value": 30}'
        ]
        
        state = MockState({
            'file_type': 'jsonl',
            'chunk_data': chunk_data,
            'processing': {
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
        })
        
        result = process_chunk(state)
        
        assert result['processing']['processed_lines'] == 3
        assert result['processing']['failed_lines'] == 0
        assert result['processing']['chunks_processed'] == 1
        assert result['processing']['statistics']['count'] == 3
        assert result['processing']['statistics']['sum'] == 60
        assert result['processing']['statistics']['min_value'] == 10
        assert result['processing']['statistics']['max_value'] == 30
    
    def test_process_csv_chunk(self):
        """Test processing a CSV chunk."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        chunk_data = [
            ['10', 'item_1', 'cat_1'],
            ['20', 'item_2', 'cat_2'],
            ['30', 'item_3', 'cat_3']
        ]
        
        state = MockState({
            'file_type': 'csv',
            'chunk_data': chunk_data,
            'processing': {
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
        })
        
        result = process_chunk(state)
        
        assert result['processing']['processed_lines'] == 3
        assert result['processing']['failed_lines'] == 0
        assert result['processing']['statistics']['sum'] == 60
    
    def test_process_chunk_with_errors(self):
        """Test chunk processing with errors."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        chunk_data = [
            '{"id": 1, "value": 10}',
            'invalid json',
            '{"id": 3, "value": 30}'
        ]
        
        state = MockState({
            'file_type': 'jsonl',
            'chunk_data': chunk_data,
            'processing': {
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
        })
        
        result = process_chunk(state)
        
        assert result['processing']['processed_lines'] == 2
        assert result['processing']['failed_lines'] == 1
        assert len(result['processing']['errors']) > 0
    
    def test_aggregate_results(self):
        """Test results aggregation."""
        class MockState:
            def __init__(self, data):
                self.data = data
        
        state = MockState({
            'file_name': 'test.jsonl',
            'file_path': '/tmp/test.jsonl',
            'file_size': 1000,
            'file_type': 'jsonl',
            'processing': {
                'processed_lines': 95,
                'failed_lines': 5,
                'chunks_processed': 10,
                'errors': ['Error 1', 'Error 2'],
                'statistics': {
                    'min_value': 0,
                    'max_value': 100,
                    'sum': 5000,
                    'count': 95
                }
            }
        })
        
        result = aggregate_results(state)
        
        assert 'summary' in result
        summary = result['summary']
        assert summary['total_lines'] == 100
        assert summary['processed_lines'] == 95
        assert summary['failed_lines'] == 5
        assert summary['success_rate'] == '95.00%'
        assert summary['statistics']['avg'] == pytest.approx(52.63, rel=0.01)
    
    def test_update_statistics(self):
        """Test statistics update function."""
        stats = {
            'min_value': None,
            'max_value': None,
            'sum': 0,
            'count': 0
        }
        
        # First value
        update_statistics(stats, 10)
        assert stats['min_value'] == 10
        assert stats['max_value'] == 10
        assert stats['sum'] == 10
        assert stats['count'] == 1
        
        # Add more values
        update_statistics(stats, 5)
        assert stats['min_value'] == 5
        assert stats['max_value'] == 10
        
        update_statistics(stats, 20)
        assert stats['min_value'] == 5
        assert stats['max_value'] == 20
        assert stats['sum'] == 35
        assert stats['count'] == 3
    
    def test_check_initialization_success(self):
        """Test initialization success check function."""
        # Successful initialization
        data = {'initialization_failed': False}
        assert check_initialization_success(data, None) is True
        
        # Failed initialization
        data = {'initialization_failed': True}
        assert check_initialization_success(data, None) is False
        
        # Default (no flag set)
        data = {}
        assert check_initialization_success(data, None) is True
    
    def test_check_processing_complete(self):
        """Test completion check function."""
        # No chunks processed
        data = {'processing': {'chunks_processed': 0}}
        assert check_processing_complete(data, None) is False
        
        # Some chunks processed
        data = {'processing': {'chunks_processed': 5}}
        assert check_processing_complete(data, None) is True
    
    def test_calculate_file_hash(self, temp_jsonl_file):
        """Test file hash calculation."""
        hash1 = calculate_file_hash(str(temp_jsonl_file))
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
        
        # Same file should produce same hash
        hash2 = calculate_file_hash(str(temp_jsonl_file))
        assert hash1 == hash2
    
    def test_create_and_read_sample_files(self):
        """Test sample file creation and chunked reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test JSONL file
            jsonl_path = tmpdir / "test.jsonl"
            create_sample_file(jsonl_path, 'jsonl', num_lines=100)
            assert jsonl_path.exists()
            
            chunks = simulate_chunked_reading(jsonl_path, 'jsonl', chunk_size=30)
            assert len(chunks) == 4  # 100 lines / 30 per chunk = 4 chunks
            assert len(chunks[0]) == 30
            assert len(chunks[-1]) == 10  # Last chunk has remainder
            
            # Test CSV file
            csv_path = tmpdir / "test.csv"
            create_sample_file(csv_path, 'csv', num_lines=100)
            assert csv_path.exists()
            
            chunks = simulate_chunked_reading(csv_path, 'csv', chunk_size=25)
            assert len(chunks) == 4
            
            # Test text file
            text_path = tmpdir / "test.txt"
            create_sample_file(text_path, 'text', num_lines=50)
            assert text_path.exists()
            
            chunks = simulate_chunked_reading(text_path, 'text', chunk_size=20)
            assert len(chunks) == 3
    
    def test_fsm_jsonl_processing(self, fsm, temp_jsonl_file):
        """Test FSM processing of JSONL file."""
        # Read first chunk
        chunks = simulate_chunked_reading(temp_jsonl_file, 'jsonl', chunk_size=50)
        
        result = fsm.process({
            'file_reference': str(temp_jsonl_file),
            'chunk_data': chunks[0] if chunks else []
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'success'
        assert result['data']['status'] == 'SUCCESS'
        
        summary = result['data'].get('summary')
        assert summary is not None
        assert summary['file_type'] == 'jsonl'
        assert summary['processed_lines'] == 50
        assert summary['success_rate'] == '100.00%'
    
    def test_fsm_csv_processing(self, fsm, temp_csv_file):
        """Test FSM processing of CSV file."""
        # Read first chunk
        chunks = simulate_chunked_reading(temp_csv_file, 'csv', chunk_size=50)
        
        result = fsm.process({
            'file_reference': str(temp_csv_file),
            'chunk_data': chunks[0] if chunks else []
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'success'
        assert result['data']['status'] == 'SUCCESS'
        
        summary = result['data'].get('summary')
        assert summary is not None
        assert summary['file_type'] == 'csv'
        assert summary['processed_lines'] == 50
    
    def test_fsm_processing_path(self, fsm, temp_jsonl_file):
        """Test that FSM follows the correct processing path."""
        chunks = simulate_chunked_reading(temp_jsonl_file, 'jsonl', chunk_size=50)
        
        result = fsm.process({
            'file_reference': str(temp_jsonl_file),
            'chunk_data': chunks[0]
        })
        
        expected_path = [
            'start',
            'initialize',
            'process_chunks',
            'aggregate',
            'success'
        ]
        assert result['path'] == expected_path
    
    def test_fsm_failure_path(self, fsm):
        """Test FSM failure path when no file reference provided."""
        result = fsm.process({})
        
        # The FSM should now properly route to failure state
        assert result['success'] is True  # FSM execution succeeds
        assert result['final_state'] == 'failure'  # But ends in failure state
        assert result['data']['status'] == 'FAILED'
        assert 'No file reference provided' in result['data']['error_message']
    
    def test_reference_mode_benefits(self, fsm, temp_jsonl_file):
        """Test that REFERENCE mode maintains file references."""
        chunks = simulate_chunked_reading(temp_jsonl_file, 'jsonl', chunk_size=50)
        
        initial_data = {
            'file_reference': str(temp_jsonl_file),
            'chunk_data': chunks[0]
        }
        
        result = fsm.process(initial_data)
        
        # Verify file reference is maintained
        assert result['data']['file_path'] == str(temp_jsonl_file)
        assert result['data']['file_reference'] == str(temp_jsonl_file)
        
        # Verify chunk data was processed
        assert result['data']['processing']['processed_lines'] > 0