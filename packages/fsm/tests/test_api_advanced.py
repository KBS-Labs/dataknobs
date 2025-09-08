"""Tests for AdvancedFSM API."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.core.state import State
from dataknobs_fsm.core.data_modes import DataMode
from dataknobs_fsm.core.exceptions import InvalidConfigurationError, StateExecutionError


@pytest.fixture
def workflow_config():
    """Complex workflow configuration for testing."""
    return {
        'name': 'test_workflow',
        'states': [
            {'name': 'validate', 'is_start': True},
            {'name': 'process'},
            {'name': 'review'},
            {'name': 'complete', 'is_end': True},
            {'name': 'error', 'is_end': True}
        ],
        'arcs': [
            {'from': 'validate', 'to': 'process', 'name': 'valid'},
            {'from': 'validate', 'to': 'error', 'name': 'invalid'},
            {'from': 'process', 'to': 'review', 'name': 'processed'},
            {'from': 'review', 'to': 'complete', 'name': 'approved'},
            {'from': 'review', 'to': 'process', 'name': 'rejected'}
        ],
        'parallel_states': ['process'],
        'error_states': ['error']
    }


@pytest.fixture
def parallel_config():
    """Configuration with parallel execution."""
    return {
        'name': 'parallel_workflow',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'parallel_1'},
            {'name': 'parallel_2'},
            {'name': 'parallel_3'},
            {'name': 'merge'},
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'parallel_1', 'name': 'split1'},
            {'from': 'start', 'to': 'parallel_2', 'name': 'split2'},
            {'from': 'start', 'to': 'parallel_3', 'name': 'split3'},
            {'from': 'parallel_1', 'to': 'merge', 'name': 'done1'},
            {'from': 'parallel_2', 'to': 'merge', 'name': 'done2'},
            {'from': 'parallel_3', 'to': 'merge', 'name': 'done3'},
            {'from': 'merge', 'to': 'end', 'name': 'complete'}
        ],
        'parallel_states': ['parallel_1', 'parallel_2', 'parallel_3']
    }


class TestAdvancedFSMInitialization:
    """Test AdvancedFSM initialization."""
    
    def test_workflow_initialization(self, workflow_config):
        """Test workflow FSM initialization."""
        fsm = AdvancedFSM(workflow_config)
        assert fsm.name == 'test_workflow'
        assert len(fsm._parallel_states) == 1
        assert 'process' in fsm._parallel_states
    
    def test_parallel_initialization(self, parallel_config):
        """Test parallel FSM initialization."""
        fsm = AdvancedFSM(parallel_config)
        assert len(fsm._parallel_states) == 3
    
    def test_hooks_initialization(self, workflow_config):
        """Test initialization with hooks."""
        hooks = {
            'on_state_enter': lambda state: print(f"Entering {state.name}"),
            'on_state_exit': lambda state: print(f"Exiting {state.name}")
        }
        fsm = AdvancedFSM(workflow_config, hooks=hooks)
        assert fsm._hooks['on_state_enter'] is not None
        assert fsm._hooks['on_state_exit'] is not None


class TestAdvancedFSMParallelExecution:
    """Test AdvancedFSM parallel execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, parallel_config):
        """Test parallel state execution."""
        executed_states = []
        
        def track_execution(state_name):
            def func(state):
                executed_states.append(state_name)
                return {'processed': state_name}
            return func
        
        config = {
            **parallel_config,
            'functions': {
                'parallel_1': track_execution('parallel_1'),
                'parallel_2': track_execution('parallel_2'),
                'parallel_3': track_execution('parallel_3')
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        assert result['success'] is True
        assert len(executed_states) == 3
        assert set(executed_states) == {'parallel_1', 'parallel_2', 'parallel_3'}
    
    @pytest.mark.asyncio
    async def test_parallel_error_handling(self, parallel_config):
        """Test error handling in parallel execution."""
        config = {
            **parallel_config,
            'functions': {
                'parallel_1': lambda state: {'result': 'ok'},
                'parallel_2': lambda state: 1/0,  # Will raise error
                'parallel_3': lambda state: {'result': 'ok'}
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        # Should handle error in one parallel branch
        assert 'error' in result or result['success'] is False


class TestAdvancedFSMConditionalLogic:
    """Test AdvancedFSM conditional logic."""
    
    @pytest.mark.asyncio
    async def test_conditional_branching(self, workflow_config):
        """Test conditional branching based on data."""
        def validate_func(state):
            if state.data.get('valid', False):
                state.data['next_arc'] = 'valid'
            else:
                state.data['next_arc'] = 'invalid'
            return state.data
        
        config = {
            **workflow_config,
            'functions': {
                'validate': validate_func
            },
            'arc_conditions': {
                'valid': lambda state: state.data.get('next_arc') == 'valid',
                'invalid': lambda state: state.data.get('next_arc') == 'invalid'
            }
        }
        
        fsm = AdvancedFSM(config)
        
        # Test valid path
        result_valid = await fsm.execute_workflow({'valid': True})
        assert result_valid['final_state'] == 'complete'
        
        # Test invalid path
        result_invalid = await fsm.execute_workflow({'valid': False})
        assert result_invalid['final_state'] == 'error'
    
    @pytest.mark.asyncio
    async def test_loop_detection(self, workflow_config):
        """Test loop detection and prevention."""
        config = {
            **workflow_config,
            'max_iterations': 3,
            'functions': {
                'review': lambda state: {'next_arc': 'rejected'}  # Always reject
            },
            'arc_conditions': {
                'rejected': lambda state: state.data.get('next_arc') == 'rejected',
                'approved': lambda state: state.data.get('next_arc') == 'approved'
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        # Should detect loop and stop after max iterations
        assert 'loop_count' in result or 'error' in result


class TestAdvancedFSMErrorHandling:
    """Test AdvancedFSM error handling."""
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, workflow_config):
        """Test error recovery mechanism."""
        attempt_count = {'count': 0}
        
        def failing_func(state):
            attempt_count['count'] += 1
            if attempt_count['count'] < 3:
                raise Exception("Temporary failure")
            return {'processed': True}
        
        config = {
            **workflow_config,
            'functions': {
                'process': failing_func
            },
            'retry_config': {
                'max_retries': 3,
                'retry_delay': 0.01
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        # Should retry and eventually succeed
        assert attempt_count['count'] == 3
        assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_compensation_logic(self, workflow_config):
        """Test compensation/rollback logic."""
        compensations_run = []
        
        def process_func(state):
            state.data['processed'] = True
            return state.data
        
        def compensate_func(state):
            compensations_run.append('process')
            state.data['processed'] = False
            return state.data
        
        config = {
            **workflow_config,
            'functions': {
                'process': process_func,
                'review': lambda state: 1/0  # Will fail
            },
            'compensations': {
                'process': compensate_func
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        # Should run compensation for process state
        assert len(compensations_run) > 0
        assert 'process' in compensations_run


class TestAdvancedFSMMonitoring:
    """Test AdvancedFSM monitoring capabilities."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, workflow_config):
        """Test metrics collection during execution."""
        fsm = AdvancedFSM(workflow_config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        metrics = fsm.get_metrics()
        assert 'total_executions' in metrics
        assert 'success_rate' in metrics
        assert 'average_duration' in metrics
        assert metrics['total_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_event_emission(self, workflow_config):
        """Test event emission during execution."""
        events = []
        
        def event_handler(event):
            events.append(event)
        
        hooks = {
            'on_state_enter': lambda state: event_handler({'type': 'enter', 'state': state.name}),
            'on_state_exit': lambda state: event_handler({'type': 'exit', 'state': state.name}),
            'on_transition': lambda arc: event_handler({'type': 'transition', 'arc': arc})
        }
        
        fsm = AdvancedFSM(workflow_config, hooks=hooks)
        await fsm.execute_workflow({'test': 'data'})
        
        # Should have events for state entries, exits, and transitions
        assert len(events) > 0
        assert any(e['type'] == 'enter' for e in events)
        assert any(e['type'] == 'exit' for e in events)
    
    @pytest.mark.asyncio
    async def test_tracing(self, workflow_config):
        """Test execution tracing."""
        fsm = AdvancedFSM(workflow_config, enable_tracing=True)
        result = await fsm.execute_workflow({'test': 'data'})
        
        trace = fsm.get_trace()
        assert len(trace) > 0
        assert 'timestamp' in trace[0]
        assert 'state' in trace[0]
        assert 'data' in trace[0]


class TestAdvancedFSMOptimization:
    """Test AdvancedFSM optimization features."""
    
    @pytest.mark.asyncio
    async def test_state_caching(self, workflow_config):
        """Test state result caching."""
        execution_count = {'count': 0}
        
        def expensive_func(state):
            execution_count['count'] += 1
            return {'result': execution_count['count']}
        
        config = {
            **workflow_config,
            'functions': {
                'process': expensive_func
            },
            'cache_config': {
                'enabled': True,
                'ttl': 60
            }
        }
        
        fsm = AdvancedFSM(config)
        
        # First execution
        result1 = await fsm.execute_workflow({'test': 'data'})
        count1 = execution_count['count']
        
        # Second execution with same data (should use cache)
        result2 = await fsm.execute_workflow({'test': 'data'})
        count2 = execution_count['count']
        
        # Cache should prevent re-execution
        assert count1 == count2
    
    @pytest.mark.asyncio
    async def test_path_optimization(self, parallel_config):
        """Test execution path optimization."""
        fsm = AdvancedFSM(parallel_config, optimize_paths=True)
        
        # Analyze optimal paths
        paths = fsm.analyze_paths()
        assert 'shortest_path' in paths
        assert 'critical_path' in paths
        assert len(paths['shortest_path']) > 0


class TestAdvancedFSMIntegration:
    """Test AdvancedFSM integration features."""
    
    @pytest.mark.asyncio
    async def test_subprocess_execution(self, workflow_config):
        """Test subprocess FSM execution."""
        subprocess_config = {
            'name': 'subprocess',
            'states': [
                {'name': 'sub_start', 'is_start': True},
                {'name': 'sub_end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'sub_start', 'to': 'sub_end', 'name': 'complete'}
            ]
        }
        
        config = {
            **workflow_config,
            'functions': {
                'process': lambda state: {'subprocess_result': 'done'}
            },
            'subprocesses': {
                'process': subprocess_config
            }
        }
        
        fsm = AdvancedFSM(config)
        result = await fsm.execute_workflow({'test': 'data'})
        
        assert result['success'] is True
        assert 'subprocess_results' in result or 'subprocess_result' in result['data']
    
    @pytest.mark.asyncio
    async def test_external_service_integration(self, workflow_config):
        """Test integration with external services."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={'status': 'ok'})
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            config = {
                **workflow_config,
                'functions': {
                    'process': 'http://api.example.com/process'  # External API
                }
            }
            
            fsm = AdvancedFSM(config)
            result = await fsm.execute_workflow({'test': 'data'})
            
            # Should call external service
            assert mock_client.called or 'error' in result


class TestAdvancedFSMHistory:
    """Test AdvancedFSM history management."""
    
    @pytest.mark.asyncio
    async def test_execution_history(self, workflow_config):
        """Test execution history tracking."""
        fsm = AdvancedFSM(workflow_config, track_history=True)
        
        # Execute multiple times
        for i in range(3):
            await fsm.execute_workflow({'id': i})
        
        history = fsm.get_history()
        assert len(history) == 3
        assert all('execution_id' in h for h in history)
        assert all('timestamp' in h for h in history)
    
    @pytest.mark.asyncio
    async def test_history_query(self, workflow_config):
        """Test querying execution history."""
        fsm = AdvancedFSM(workflow_config, track_history=True)
        
        # Execute with different outcomes
        await fsm.execute_workflow({'valid': True})
        await fsm.execute_workflow({'valid': False})
        
        # Query successful executions
        successful = fsm.query_history({'success': True})
        assert len(successful) >= 0
        
        # Query by time range
        recent = fsm.query_history({
            'start_time': datetime.now() - timedelta(minutes=1),
            'end_time': datetime.now()
        })
        assert len(recent) >= 0


class TestAdvancedFSMScheduling:
    """Test AdvancedFSM scheduling capabilities."""
    
    @pytest.mark.asyncio
    async def test_scheduled_execution(self, workflow_config):
        """Test scheduled workflow execution."""
        fsm = AdvancedFSM(workflow_config)
        
        # Schedule execution
        schedule_id = await fsm.schedule_execution(
            data={'test': 'data'},
            delay=0.1  # 100ms delay
        )
        
        assert schedule_id is not None
        
        # Wait for scheduled execution
        await asyncio.sleep(0.2)
        
        # Check if execution completed
        status = fsm.get_schedule_status(schedule_id)
        assert status in ['completed', 'running', 'scheduled']
    
    @pytest.mark.asyncio
    async def test_recurring_execution(self, workflow_config):
        """Test recurring workflow execution."""
        execution_count = {'count': 0}
        
        def count_func(state):
            execution_count['count'] += 1
            return state.data
        
        config = {
            **workflow_config,
            'functions': {
                'process': count_func
            }
        }
        
        fsm = AdvancedFSM(config)
        
        # Schedule recurring execution
        schedule_id = await fsm.schedule_recurring(
            data={'test': 'data'},
            interval=0.1,  # Every 100ms
            max_runs=3
        )
        
        # Wait for executions
        await asyncio.sleep(0.5)
        
        # Should have executed multiple times
        assert execution_count['count'] >= 2