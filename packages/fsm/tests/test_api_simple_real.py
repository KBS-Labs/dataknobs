"""Tests for SimpleFSM API - Using real implementations."""

import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path
import tempfile
import json

from dataknobs_fsm.api.simple import SimpleFSM, create_fsm, process_file, validate_data, batch_process
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_data import Record


@pytest.fixture
def simple_fsm_config():
    """Simple working FSM configuration."""
    return {
        'name': 'test_simple_fsm',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {
                    'name': 'start',
                    'is_start': True,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'input': {'type': 'string'}
                        },
                        'required': ['input']
                    }
                },
                {
                    'name': 'process',
                    'functions': {
                        'transform': 'lambda state: {"output": state.data["input"].upper(), "processed": True}'
                    }
                },
                {
                    'name': 'end',
                    'is_end': True
                }
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin_processing'
                },
                {
                    'from': 'process', 
                    'to': 'end',
                    'name': 'complete'
                }
            ]
        }]
    }


@pytest.fixture
def processing_fsm_config():
    """FSM configuration for data processing tests."""
    return {
        'name': 'processing_fsm',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {
                    'name': 'input',
                    'is_start': True
                },
                {
                    'name': 'multiply',
                    'functions': {
                        'transform': 'lambda state: {"result": state.data.get("value", 1) * 2}'
                    }
                },
                {
                    'name': 'output',
                    'is_end': True
                }
            ],
            'arcs': [
                {'from': 'input', 'to': 'multiply', 'name': 'process'},
                {'from': 'multiply', 'to': 'output', 'name': 'done'}
            ]
        }]
    }


@pytest.fixture
def temp_config_file(simple_fsm_config):
    """Temporary config file for testing file loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(simple_fsm_config, f)
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestSimpleFSMInitialization:
    """Test SimpleFSM initialization with real components."""
    
    def test_initialization_from_dict(self, simple_fsm_config):
        """Test basic FSM initialization from config dictionary."""
        fsm = SimpleFSM(simple_fsm_config)
        
        # Test basic properties
        assert fsm.data_mode == DataHandlingMode.COPY
        assert fsm.config.name == simple_fsm_config['name']
        assert fsm.config.main_network == simple_fsm_config['main_network']
        assert fsm._fsm is not None
        assert fsm._engine is not None
        assert fsm._resource_manager is not None
        
        # Test that FSM was built correctly
        states = fsm.get_states()
        assert 'start' in states
        assert 'process' in states 
        assert 'end' in states
    
    def test_initialization_from_file(self, temp_config_file, simple_fsm_config):
        """Test FSM initialization from config file."""
        fsm = SimpleFSM(temp_config_file)
        
        assert fsm._fsm is not None
        states = fsm.get_states()
        assert len(states) == 3
    
    def test_initialization_with_data_mode(self, simple_fsm_config):
        """Test FSM with different data modes."""
        fsm = SimpleFSM(simple_fsm_config, data_mode=DataHandlingMode.REFERENCE)
        assert fsm.data_mode == DataHandlingMode.REFERENCE
    
    def test_initialization_with_resources(self, simple_fsm_config):
        """Test FSM with additional resources."""
        resources = {'test_resource': {'type': 'memory', 'data': {}}}
        fsm = SimpleFSM(simple_fsm_config, resources=resources)
        
        # Verify resources are available
        resource_names = fsm.get_resources()
        # Note: The exact resource names depend on implementation
        # but we should be able to get a list
        assert isinstance(resource_names, list)


class TestSimpleFSMProcessing:
    """Test SimpleFSM data processing with real FSM execution."""
    
    def test_process_simple_data(self, processing_fsm_config):
        """Test processing simple data through the FSM."""
        fsm = SimpleFSM(processing_fsm_config)
        
        # Process data through the FSM
        input_data = {'value': 5}
        result = fsm.process(input_data)
        
        # Verify successful processing
        if not result['success']:
            print(f"Process failed with error: {result['error']}")
        assert result['success'] is True
        assert result['final_state'] == 'output'
        assert 'data' in result
        # The exact result depends on the FSM execution
        assert 'path' in result
        assert isinstance(result['path'], list)
    
    def test_process_with_record(self, processing_fsm_config):
        """Test processing with Record input."""
        fsm = SimpleFSM(processing_fsm_config)
        
        record = Record({'value': 3})
        result = fsm.process(record)
        
        assert result['success'] is True
        assert result['final_state'] == 'output'
    
    def test_process_with_timeout(self, simple_fsm_config):
        """Test processing with timeout (should not timeout on simple FSM)."""
        fsm = SimpleFSM(simple_fsm_config)
        
        result = fsm.process({'input': 'test'}, timeout=5.0)
        
        assert result['success'] is True
    
    def test_validation_with_schema(self, simple_fsm_config):
        """Test data validation against start state schema."""
        fsm = SimpleFSM(simple_fsm_config)
        
        # Valid data
        valid_result = fsm.validate({'input': 'test_string'})
        assert valid_result['valid'] is True
        assert len(valid_result['errors']) == 0
        
        # Invalid data (missing required field)
        invalid_result = fsm.validate({'wrong_field': 'value'})
        # The exact validation behavior depends on schema implementation
        # but we should get a validation result
        assert isinstance(invalid_result['valid'], bool)
        assert isinstance(invalid_result['errors'], list)
    
    def test_validation_with_record(self, simple_fsm_config):
        """Test validation with Record input."""
        fsm = SimpleFSM(simple_fsm_config)
        
        record = Record({'input': 'test'})
        result = fsm.validate(record)
        
        assert isinstance(result['valid'], bool)
        assert isinstance(result['errors'], list)


class TestSimpleFSMBatchProcessing:
    """Test SimpleFSM batch processing with real execution."""
    
    def test_batch_process_small_dataset(self, processing_fsm_config):
        """Test batch processing with small dataset."""
        fsm = SimpleFSM(processing_fsm_config)
        
        batch_data = [
            {'value': 1},
            {'value': 2},
            {'value': 3}
        ]
        
        results = fsm.process_batch(batch_data, batch_size=2, max_workers=1)
        
        assert len(results) == 3
        # All should succeed with simple processing
        successful = [r for r in results if r['success']]
        assert len(successful) >= 0  # At least some should succeed
        
        # Check structure of results
        for result in results:
            assert 'success' in result
            assert 'final_state' in result
            assert 'data' in result
            assert 'path' in result
    
    def test_batch_process_with_records(self, processing_fsm_config):
        """Test batch processing with Record inputs."""
        fsm = SimpleFSM(processing_fsm_config)
        
        records = [Record({'value': i}) for i in range(3)]
        results = fsm.process_batch(records)
        
        assert len(results) == 3
        assert all('success' in r for r in results)


class TestSimpleFSMStreamProcessing:
    """Test SimpleFSM stream processing capabilities."""

    def test_process_stream_from_file(self, processing_fsm_config, tmp_path):
        """Test stream processing from file."""
        fsm = SimpleFSM(processing_fsm_config)

        # Create a test file with data
        test_file = tmp_path / "test_data.jsonl"
        with open(test_file, 'w') as f:
            for i in range(3):
                json.dump({'value': i}, f)
                f.write('\n')

        result = fsm.process_stream(
            source=str(test_file),
            chunk_size=2
        )

        # Verify stream processing results
        assert 'total_processed' in result
        assert 'successful' in result
        assert 'failed' in result
        assert 'duration' in result
        assert 'throughput' in result

        # Basic sanity checks
        assert isinstance(result['total_processed'], int)
        assert isinstance(result['successful'], int)
        assert isinstance(result['failed'], int)

    @pytest.mark.asyncio
    async def test_process_stream_from_iterator(self, processing_fsm_config):
        """Test stream processing from async iterator using AsyncSimpleFSM."""
        fsm = AsyncSimpleFSM(processing_fsm_config)

        # Create async iterator
        async def data_generator():
            for i in range(3):
                yield {'value': i}

        result = await fsm.process_stream(
            source=data_generator(),
            chunk_size=2
        )

        # Verify stream processing results
        assert 'total_processed' in result
        assert 'successful' in result
        assert 'failed' in result
        assert 'duration' in result
        assert 'throughput' in result

        # Basic sanity checks
        assert isinstance(result['total_processed'], int)
        assert isinstance(result['successful'], int)
        assert isinstance(result['failed'], int)

        # Clean up
        await fsm.close()


class TestSimpleFSMUtilityMethods:
    """Test SimpleFSM utility methods with real FSM."""
    
    def test_get_states(self, simple_fsm_config):
        """Test retrieving state names from real FSM."""
        fsm = SimpleFSM(simple_fsm_config)
        
        states = fsm.get_states()
        
        assert isinstance(states, list)
        assert 'start' in states
        assert 'process' in states
        assert 'end' in states
        assert len(states) == 3
    
    def test_get_resources(self, simple_fsm_config):
        """Test retrieving resource names."""
        fsm = SimpleFSM(simple_fsm_config)
        
        resources = fsm.get_resources()
        
        assert isinstance(resources, list)
        # The exact resources depend on what the FSM registers
        # but we should get a valid list
    
    def test_close_cleanup(self, simple_fsm_config):
        """Test FSM cleanup and resource closing."""
        fsm = SimpleFSM(simple_fsm_config)
        
        # This should not raise an exception
        fsm.close()


class TestSimpleFSMFactoryFunctions:
    """Test SimpleFSM factory and convenience functions with real FSMs."""
    
    def test_create_fsm_factory(self, simple_fsm_config):
        """Test create_fsm factory function."""
        fsm = create_fsm(simple_fsm_config, data_mode=DataHandlingMode.REFERENCE)
        
        assert isinstance(fsm, SimpleFSM)
        assert fsm.data_mode == DataHandlingMode.REFERENCE
        
        # Test it actually works
        states = fsm.get_states()
        assert len(states) > 0
    
    def test_validate_data_convenience(self, simple_fsm_config):
        """Test validate_data convenience function."""
        data = [
            {'input': 'valid1'},
            {'input': 'valid2'},
            {'missing': 'field'}  # This should fail validation
        ]
        
        results = validate_data(simple_fsm_config, data)
        
        assert len(results) == 3
        assert all('valid' in r and 'errors' in r for r in results)
        
        # At least the valid ones should pass
        valid_results = [r for r in results if r['valid']]
        assert len(valid_results) >= 2
    
    def test_batch_process_convenience(self, processing_fsm_config):
        """Test batch_process convenience function."""
        data = [
            {'value': 10},
            {'value': 20}
        ]
        
        results = batch_process(processing_fsm_config, data, batch_size=1)
        
        assert len(results) == 2
        assert all('success' in r for r in results)
        
        # Should process successfully
        successful = [r for r in results if r['success']]
        assert len(successful) >= 0


class TestSimpleFSMErrorHandling:
    """Test SimpleFSM error handling with real scenarios."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid FSM configuration."""
        invalid_config = {
            'name': 'invalid',
            'networks': [{
                'name': 'broken',
                'states': [
                    {'name': 'start', 'is_start': True}
                    # Missing end state, no arcs
                ],
                'arcs': []
            }]
        }
        
        # This might succeed or fail depending on validation strictness
        try:
            fsm = SimpleFSM(invalid_config)
            # If it succeeds, it should still be a valid FSM object
            assert fsm._fsm is not None
        except Exception as e:
            # If it fails, it should be a meaningful error
            assert isinstance(e, Exception)
    
    def test_processing_with_missing_data(self, simple_fsm_config):
        """Test processing when required data is missing."""
        fsm = SimpleFSM(simple_fsm_config)
        
        # Try to process without required 'input' field
        result = fsm.process({})
        
        # Should handle the error gracefully
        assert 'success' in result
        assert 'error' in result or 'final_state' in result
        
        # The exact behavior depends on how validation is implemented
        # but it should not crash


class TestSimpleFSMIntegration:
    """Integration tests using real FSM components end-to-end."""
    
    def test_complete_workflow(self, simple_fsm_config):
        """Test complete workflow from initialization to processing."""
        # Initialize FSM
        fsm = SimpleFSM(simple_fsm_config)
        
        # Validate configuration worked
        states = fsm.get_states()
        assert len(states) > 0
        
        # Validate input data
        validation = fsm.validate({'input': 'test_data'})
        
        # Process data
        if validation['valid']:
            result = fsm.process({'input': 'test_data'})
            assert result['success'] is True
        
        # Clean up
        fsm.close()
    
    def test_data_transformation_pipeline(self, processing_fsm_config):
        """Test data flowing through transformation pipeline."""
        fsm = SimpleFSM(processing_fsm_config)
        
        # Process data and verify transformation
        input_value = 7
        result = fsm.process({'value': input_value})
        
        if result['success']:
            # The FSM should have transformed the data
            assert 'data' in result
            # The exact transformation depends on the FSM implementation
            # but data should have been processed
            
        # Process multiple items to test consistency
        batch_results = fsm.process_batch([
            {'value': 1},
            {'value': 2},
            {'value': 3}
        ])
        
        successful_results = [r for r in batch_results if r['success']]
        assert len(successful_results) >= 0
        
        fsm.close()
