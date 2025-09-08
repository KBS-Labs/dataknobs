"""Tests for SimpleFSM API."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.state import State
from dataknobs_fsm.core.data_modes import DataMode
from dataknobs_fsm.core.exceptions import InvalidConfigurationError, StateExecutionError


@pytest.fixture
def basic_config():
    """Basic FSM configuration for testing."""
    return {
        'name': 'test_fsm',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'middle'},
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'middle', 'name': 'proceed'},
            {'from': 'middle', 'to': 'end', 'name': 'finish'}
        ]
    }


@pytest.fixture
def fsm_with_functions():
    """FSM configuration with functions."""
    return {
        'name': 'function_fsm',
        'states': [
            {'name': 'process', 'is_start': True, 'is_end': True}
        ],
        'functions': {
            'process': lambda state: {'result': state.data.get('input', 0) * 2}
        }
    }


class TestSimpleFSMInitialization:
    """Test SimpleFSM initialization."""
    
    def test_basic_initialization(self, basic_config):
        """Test basic FSM initialization."""
        fsm = SimpleFSM(basic_config)
        assert fsm.name == 'test_fsm'
        assert len(fsm._fsm.states) == 3
        assert len(fsm._fsm.arcs) == 2
    
    def test_initialization_with_data_mode(self, basic_config):
        """Test FSM initialization with specific data mode."""
        fsm = SimpleFSM(basic_config, data_mode=DataMode.COPY)
        assert fsm.data_mode == DataMode.COPY
    
    def test_initialization_with_resources(self, basic_config):
        """Test FSM initialization with resources."""
        config = {**basic_config, 'resources': {'db': {'type': 'postgres'}}}
        fsm = SimpleFSM(config)
        assert 'db' in fsm._resources
    
    def test_invalid_configuration(self):
        """Test initialization with invalid configuration."""
        with pytest.raises(InvalidConfigurationError):
            SimpleFSM({'name': 'invalid'})  # Missing states


class TestSimpleFSMExecution:
    """Test SimpleFSM execution methods."""
    
    @pytest.mark.asyncio
    async def test_execute_single(self, basic_config):
        """Test single execution."""
        fsm = SimpleFSM(basic_config)
        result = await fsm.execute({'test': 'data'})
        
        assert result['success'] is True
        assert result['final_state'] == 'end'
        assert 'execution_time' in result
        assert 'state_history' in result
    
    @pytest.mark.asyncio
    async def test_execute_with_function(self, fsm_with_functions):
        """Test execution with function."""
        fsm = SimpleFSM(fsm_with_functions)
        result = await fsm.execute({'input': 5})
        
        assert result['success'] is True
        assert result['data']['result'] == 10
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, basic_config):
        """Test execution with error."""
        config = {
            **basic_config,
            'functions': {
                'middle': lambda state: 1/0  # Will raise ZeroDivisionError
            }
        }
        fsm = SimpleFSM(config)
        
        result = await fsm.execute({'test': 'data'})
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_execute_batch(self, basic_config):
        """Test batch execution."""
        fsm = SimpleFSM(basic_config)
        batch_data = [{'id': i} for i in range(5)]
        
        results = await fsm.execute_batch(batch_data, batch_size=2)
        
        assert len(results) == 5
        assert all(r['success'] for r in results)
        assert all(r['final_state'] == 'end' for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_stream(self, basic_config):
        """Test stream execution."""
        fsm = SimpleFSM(basic_config)
        
        async def data_generator():
            for i in range(3):
                yield {'id': i}
        
        results = []
        async for result in fsm.execute_stream(data_generator()):
            results.append(result)
        
        assert len(results) == 3
        assert all(r['success'] for r in results)


class TestSimpleFSMValidation:
    """Test SimpleFSM validation methods."""
    
    def test_validate_valid_config(self, basic_config):
        """Test validation of valid configuration."""
        fsm = SimpleFSM(basic_config)
        errors = fsm.validate()
        assert len(errors) == 0
    
    def test_validate_disconnected_states(self):
        """Test validation with disconnected states."""
        config = {
            'name': 'disconnected',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'orphan'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'end', 'name': 'skip'}
            ]
        }
        fsm = SimpleFSM(config)
        errors = fsm.validate()
        assert len(errors) > 0
        assert any('orphan' in str(e) for e in errors)
    
    def test_validate_no_start_state(self):
        """Test validation with no start state."""
        config = {
            'name': 'no_start',
            'states': [
                {'name': 'middle'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'middle', 'to': 'end', 'name': 'finish'}
            ]
        }
        fsm = SimpleFSM(config)
        errors = fsm.validate()
        assert len(errors) > 0
        assert any('start' in str(e).lower() for e in errors)


class TestSimpleFSMVisualization:
    """Test SimpleFSM visualization methods."""
    
    def test_get_visualization(self, basic_config):
        """Test getting visualization data."""
        fsm = SimpleFSM(basic_config)
        viz = fsm.get_visualization()
        
        assert 'states' in viz
        assert 'transitions' in viz
        assert len(viz['states']) == 3
        assert len(viz['transitions']) == 2
    
    def test_visualization_with_current_state(self, basic_config):
        """Test visualization with current state highlighted."""
        fsm = SimpleFSM(basic_config)
        viz = fsm.get_visualization(current_state='middle')
        
        middle_state = next(s for s in viz['states'] if s['id'] == 'middle')
        assert middle_state.get('is_current') is True


class TestSimpleFSMStateManagement:
    """Test SimpleFSM state management."""
    
    @pytest.mark.asyncio
    async def test_get_state(self, basic_config):
        """Test getting current state."""
        fsm = SimpleFSM(basic_config)
        await fsm.execute({'test': 'data'})
        
        state = fsm.get_state()
        assert state is not None
        assert 'current_state' in state
        assert 'data' in state
    
    @pytest.mark.asyncio
    async def test_reset_state(self, basic_config):
        """Test resetting FSM state."""
        fsm = SimpleFSM(basic_config)
        await fsm.execute({'test': 'data'})
        
        fsm.reset()
        state = fsm.get_state()
        assert state['current_state'] == 'start'  # Should be back at start
    
    @pytest.mark.asyncio
    async def test_checkpoint_restore(self, basic_config):
        """Test checkpoint and restore functionality."""
        fsm = SimpleFSM(basic_config)
        
        # Execute partially
        await fsm._fsm.execute({'test': 'data'})
        await fsm._fsm.transition('proceed')
        
        # Create checkpoint
        checkpoint = fsm.checkpoint()
        assert 'state' in checkpoint
        assert 'data' in checkpoint
        
        # Reset and restore
        fsm.reset()
        fsm.restore(checkpoint)
        
        state = fsm.get_state()
        assert state['current_state'] == checkpoint['state']


class TestSimpleFSMResourceManagement:
    """Test SimpleFSM resource management."""
    
    @pytest.mark.asyncio
    async def test_resource_acquisition(self, basic_config):
        """Test resource acquisition during execution."""
        config = {
            **basic_config,
            'resources': {'db': {'type': 'test'}},
            'states': [
                {'name': 'start', 'is_start': True, 'resources': ['db']},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'end', 'name': 'done'}
            ]
        }
        
        with patch('dataknobs_fsm.core.resource.ResourceManager') as mock_rm:
            mock_manager = Mock()
            mock_rm.return_value = mock_manager
            mock_manager.acquire = AsyncMock()
            mock_manager.release = AsyncMock()
            
            fsm = SimpleFSM(config)
            await fsm.execute({'test': 'data'})
            
            # Verify resource was acquired and released
            assert mock_manager.acquire.called
            assert mock_manager.release.called


class TestSimpleFSMErrorHandling:
    """Test SimpleFSM error handling."""
    
    @pytest.mark.asyncio
    async def test_function_error_handling(self):
        """Test handling of function errors."""
        config = {
            'name': 'error_fsm',
            'states': [
                {'name': 'process', 'is_start': True, 'is_end': True}
            ],
            'functions': {
                'process': lambda state: {'error': 'test'}
            },
            'error_handler': lambda error: {'handled': True}
        }
        
        fsm = SimpleFSM(config)
        result = await fsm.execute({'test': 'data'})
        
        # Should handle error gracefully
        assert 'error' in result or result['success'] is False
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, basic_config):
        """Test timeout handling."""
        config = {
            **basic_config,
            'functions': {
                'middle': lambda state: asyncio.sleep(10)  # Long running
            }
        }
        
        fsm = SimpleFSM(config)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                fsm.execute({'test': 'data'}),
                timeout=0.1
            )


class TestSimpleFSMDataModes:
    """Test SimpleFSM data modes."""
    
    @pytest.mark.asyncio
    async def test_reference_mode(self, basic_config):
        """Test REFERENCE data mode."""
        fsm = SimpleFSM(basic_config, data_mode=DataMode.REFERENCE)
        original_data = {'test': 'data', 'list': [1, 2, 3]}
        
        result = await fsm.execute(original_data)
        
        # Data should be same reference
        assert result['data'] is original_data
    
    @pytest.mark.asyncio
    async def test_copy_mode(self, basic_config):
        """Test COPY data mode."""
        fsm = SimpleFSM(basic_config, data_mode=DataMode.COPY)
        original_data = {'test': 'data', 'list': [1, 2, 3]}
        
        result = await fsm.execute(original_data)
        
        # Data should be a copy
        assert result['data'] is not original_data
        assert result['data'] == original_data
    
    @pytest.mark.asyncio
    async def test_stream_mode(self):
        """Test STREAM data mode."""
        config = {
            'name': 'stream_fsm',
            'data_mode': 'STREAM',
            'states': [
                {'name': 'process', 'is_start': True, 'is_end': True}
            ]
        }
        
        fsm = SimpleFSM(config, data_mode=DataMode.STREAM)
        
        async def stream_generator():
            for i in range(3):
                yield {'chunk': i}
        
        chunks = []
        async for result in fsm.execute_stream(stream_generator()):
            chunks.append(result)
        
        assert len(chunks) == 3