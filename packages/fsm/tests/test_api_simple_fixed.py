"""Tests for SimpleFSM API - Fixed to match actual implementation."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
from pathlib import Path
import tempfile
import json

from dataknobs_fsm.api.simple import SimpleFSM, create_fsm, process_file, validate_data, batch_process
from dataknobs_fsm.core.data_modes import DataMode
from dataknobs_data import Record


@pytest.fixture
def basic_config():
    """Basic FSM configuration that matches actual schema."""
    return {
        'name': 'test_fsm',
        'networks': [{
            'name': 'main',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'middle'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'middle', 'name': 'proceed'},
                {'from': 'middle', 'to': 'end', 'name': 'finish'}
            ]
        }]
    }


@pytest.fixture  
def config_with_functions():
    """FSM configuration with state functions."""
    return {
        'name': 'function_fsm',
        'networks': [{
            'name': 'main',
            'states': [
                {'name': 'process', 'is_start': True, 'is_end': True}
            ]
        }],
        'functions': {
            'process': 'lambda state: {"result": state.data.get("input", 0) * 2}'
        }
    }


@pytest.fixture
def temp_config_file(basic_config):
    """Temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(basic_config, f)
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestSimpleFSMInitialization:
    """Test SimpleFSM initialization."""
    
    def test_basic_initialization(self, basic_config):
        """Test basic FSM initialization."""
        fsm = SimpleFSM(basic_config)
        assert fsm.data_mode == DataMode.COPY
        assert fsm._config == basic_config
        assert fsm._fsm is not None
        assert fsm._engine is not None
        assert fsm._resource_manager is not None
    
    def test_initialization_with_data_mode(self, basic_config):
        """Test FSM initialization with specific data mode."""
        fsm = SimpleFSM(basic_config, data_mode=DataMode.REFERENCE)
        assert fsm.data_mode == DataMode.REFERENCE
    
    def test_initialization_with_resources(self, basic_config):
        """Test FSM initialization with resources."""
        resources = {'db': {'type': 'memory'}}
        fsm = SimpleFSM(basic_config, resources=resources)
        assert fsm._resources == resources
    
    def test_initialization_from_file(self, temp_config_file):
        """Test initialization from config file."""
        fsm = SimpleFSM(temp_config_file)
        assert fsm._fsm is not None


class TestSimpleFSMProcessing:
    """Test SimpleFSM data processing methods."""
    
    def test_process_single_dict(self, basic_config):
        """Test processing a single dictionary."""
        fsm = SimpleFSM(basic_config)
        
        # Mock the execution to avoid complex FSM execution
        with patch.object(fsm, '_execute_async', new_callable=AsyncMock) as mock_execute:
            mock_result = Mock()
            mock_result.state = Mock()
            mock_result.state.definition.name = 'end'
            mock_result.state.data = {'processed': True}
            mock_result.path = [Mock(), Mock()]
            mock_result.path[0].definition.name = 'start'
            mock_result.path[1].definition.name = 'end'
            mock_execute.return_value = mock_result
            
            result = fsm.process({'test': 'data'})
            
            assert result['success'] is True
            assert result['final_state'] == 'end'
            assert result['data'] == {'processed': True}
            assert result['path'] == ['start', 'end']
            assert result['error'] is None
    
    def test_process_single_record(self, basic_config):
        """Test processing a single Record."""
        fsm = SimpleFSM(basic_config)
        record = Record({'test': 'data'})
        
        with patch.object(fsm, '_execute_async', new_callable=AsyncMock) as mock_execute:
            mock_result = Mock()
            mock_result.state = Mock()
            mock_result.state.definition.name = 'end'
            mock_result.state.data = {'processed': True}
            mock_result.path = []
            mock_execute.return_value = mock_result
            
            result = fsm.process(record)
            
            assert result['success'] is True
            assert result['final_state'] == 'end'
    
    def test_process_with_error(self, basic_config):
        """Test processing with execution error."""
        fsm = SimpleFSM(basic_config)
        
        with patch.object(fsm, '_execute_async', side_effect=Exception("Test error")):
            # Mock current_state for error handling
            fsm._engine = Mock()
            mock_context = Mock()
            mock_context.current_state = Mock()
            mock_context.current_state.definition.name = 'start'
            mock_context.current_state.data = {'test': 'data'}
            
            with patch('dataknobs_fsm.api.simple.ExecutionContext', return_value=mock_context):
                result = fsm.process({'test': 'data'})
            
            assert result['success'] is False
            assert result['error'] == 'Test error'
    
    def test_process_with_timeout(self, basic_config):
        """Test processing with timeout."""
        fsm = SimpleFSM(basic_config)
        
        with patch.object(fsm, '_execute_async', new_callable=AsyncMock) as mock_execute:
            mock_result = Mock()
            mock_result.state = Mock()
            mock_result.state.definition.name = 'end'
            mock_result.state.data = {'test': True}
            mock_result.path = []
            mock_execute.return_value = mock_result
            
            result = fsm.process({'test': 'data'}, timeout=1.0)
            
            assert result['success'] is True
    
    def test_process_batch(self, basic_config):
        """Test batch processing."""
        fsm = SimpleFSM(basic_config)
        batch_data = [{'id': i} for i in range(3)]
        
        # Mock BatchExecutor
        with patch('dataknobs_fsm.api.simple.BatchExecutor') as mock_batch_executor:
            mock_executor = mock_batch_executor.return_value
            mock_results = []
            
            for i in range(3):
                mock_result = Mock()
                mock_result.success = True
                mock_result.final_state = Mock()
                mock_result.final_state.definition.name = 'end'
                mock_result.final_state.data = {'id': i, 'processed': True}
                mock_result.path = []
                mock_results.append(mock_result)
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = mock_results
                
                results = fsm.process_batch(batch_data, batch_size=2, max_workers=2)
                
                assert len(results) == 3
                assert all(r['success'] for r in results)
                assert all(r['final_state'] == 'end' for r in results)
    
    @pytest.mark.asyncio
    async def test_process_stream(self, basic_config):
        """Test stream processing."""
        fsm = SimpleFSM(basic_config)
        
        # Mock StreamExecutor
        with patch('dataknobs_fsm.api.simple.StreamExecutor') as mock_stream_executor:
            mock_executor = mock_stream_executor.return_value
            mock_result = Mock()
            mock_result.total_processed = 100
            mock_result.successful = 95
            mock_result.failed = 5
            mock_result.duration = 2.5
            mock_result.throughput = 40.0
            
            mock_executor.execute_stream = AsyncMock(return_value=mock_result)
            
            result = await fsm.process_stream("input.json", "output.json", chunk_size=50)
            
            assert result['total_processed'] == 100
            assert result['successful'] == 95
            assert result['failed'] == 5
            assert result['duration'] == 2.5
            assert result['throughput'] == 40.0


class TestSimpleFSMValidation:
    """Test SimpleFSM validation methods."""
    
    def test_validate_with_schema(self, basic_config):
        """Test validation with schema."""
        fsm = SimpleFSM(basic_config)
        
        # Mock start state with schema
        mock_start_state = Mock()
        mock_schema = Mock()
        mock_validation_result = Mock()
        mock_validation_result.valid = True
        mock_validation_result.errors = []
        mock_schema.validate.return_value = mock_validation_result
        mock_start_state.schema = mock_schema
        
        fsm._fsm.get_start_state = Mock(return_value=mock_start_state)
        
        result = fsm.validate({'test': 'data'})
        
        assert result['valid'] is True
        assert result['errors'] == []
    
    def test_validate_without_schema(self, basic_config):
        """Test validation without schema."""
        fsm = SimpleFSM(basic_config)
        
        # Mock start state without schema
        mock_start_state = Mock()
        mock_start_state.schema = None
        fsm._fsm.get_start_state = Mock(return_value=mock_start_state)
        
        result = fsm.validate({'test': 'data'})
        
        assert result['valid'] is True
        assert result['errors'] == []
    
    def test_validate_with_record(self, basic_config):
        """Test validation with Record input."""
        fsm = SimpleFSM(basic_config)
        record = Record({'test': 'data'})
        
        mock_start_state = Mock()
        mock_start_state.schema = None
        fsm._fsm.get_start_state = Mock(return_value=mock_start_state)
        
        result = fsm.validate(record)
        
        assert result['valid'] is True


class TestSimpleFSMUtilityMethods:
    """Test SimpleFSM utility methods."""
    
    def test_get_states(self, basic_config):
        """Test getting state names."""
        fsm = SimpleFSM(basic_config)
        
        # Mock FSM states
        mock_state1 = Mock()
        mock_state1.name = 'start'
        mock_state2 = Mock()
        mock_state2.name = 'middle'
        mock_state3 = Mock()
        mock_state3.name = 'end'
        
        fsm._fsm.states = {
            'start': mock_state1,
            'middle': mock_state2,
            'end': mock_state3
        }
        
        states = fsm.get_states()
        
        assert 'start' in states
        assert 'middle' in states
        assert 'end' in states
        assert len(states) == 3
    
    def test_get_resources(self, basic_config):
        """Test getting resource names."""
        fsm = SimpleFSM(basic_config)
        
        # Mock resource manager
        fsm._resource_manager._resources = {'db': Mock(), 'cache': Mock()}
        
        resources = fsm.get_resources()
        
        assert 'db' in resources
        assert 'cache' in resources
        assert len(resources) == 2
    
    def test_close(self, basic_config):
        """Test cleanup/close method."""
        fsm = SimpleFSM(basic_config)
        
        with patch('asyncio.run') as mock_run:
            fsm._resource_manager.cleanup = AsyncMock()
            
            fsm.close()
            
            mock_run.assert_called_once()


class TestSimpleFSMFactoryFunctions:
    """Test SimpleFSM factory and convenience functions."""
    
    def test_create_fsm(self, basic_config):
        """Test create_fsm factory function."""
        fsm = create_fsm(basic_config, data_mode=DataMode.REFERENCE)
        
        assert isinstance(fsm, SimpleFSM)
        assert fsm.data_mode == DataMode.REFERENCE
    
    def test_process_file(self, basic_config, temp_config_file):
        """Test process_file convenience function."""
        with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
            mock_fsm = Mock()
            mock_fsm.process_stream = AsyncMock(return_value={'total_processed': 10})
            mock_fsm.close = Mock()
            mock_create.return_value = mock_fsm
            
            with patch('asyncio.run', return_value={'total_processed': 10}) as mock_run:
                result = process_file(
                    fsm_config=temp_config_file,
                    input_file='input.json',
                    output_file='output.json',
                    chunk_size=500
                )
                
                assert result['total_processed'] == 10
                mock_fsm.close.assert_called_once()
    
    def test_validate_data(self, basic_config):
        """Test validate_data convenience function."""
        with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
            mock_fsm = Mock()
            mock_fsm.validate.side_effect = [
                {'valid': True, 'errors': []},
                {'valid': False, 'errors': ['Missing field']}
            ]
            mock_fsm.close = Mock()
            mock_create.return_value = mock_fsm
            
            data = [{'valid': 'data'}, {'invalid': 'data'}]
            results = validate_data(basic_config, data)
            
            assert len(results) == 2
            assert results[0]['valid'] is True
            assert results[1]['valid'] is False
            mock_fsm.close.assert_called_once()
    
    def test_batch_process(self, basic_config):
        """Test batch_process convenience function."""
        with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
            mock_fsm = Mock()
            mock_fsm.process_batch.return_value = [
                {'success': True, 'final_state': 'end'},
                {'success': True, 'final_state': 'end'}
            ]
            mock_fsm.close = Mock()
            mock_create.return_value = mock_fsm
            
            data = [{'id': 1}, {'id': 2}]
            results = batch_process(basic_config, data, batch_size=2, max_workers=2)
            
            assert len(results) == 2
            assert all(r['success'] for r in results)
            mock_fsm.close.assert_called_once()


class TestSimpleFSMIntegration:
    """Integration tests for SimpleFSM."""
    
    def test_end_to_end_simple_workflow(self, basic_config):
        """Test simple end-to-end workflow."""
        # This would be a more complex test with real FSM execution
        # For now, just test that the components work together
        fsm = SimpleFSM(basic_config)
        
        # Verify initialization worked
        assert fsm._fsm is not None
        assert fsm._engine is not None
        assert fsm._resource_manager is not None
        
        # Verify we can get basic info
        states = fsm.get_states()
        assert isinstance(states, list)
        
        resources = fsm.get_resources()
        assert isinstance(resources, list)