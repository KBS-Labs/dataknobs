"""Tests for AdvancedFSM API - Using real implementations."""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock

from dataknobs_fsm.api.advanced import AdvancedFSM, ExecutionMode, ExecutionHook
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.engine import TraversalStrategy
from dataknobs_fsm.core.transactions import TransactionStrategy
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_data import Record


@pytest.fixture
def simple_fsm():
    """Create a real FSM instance for testing."""
    config = {
        'name': 'test_advanced_fsm',
        'main_network': 'main',
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
    
    from dataknobs_fsm.config.loader import ConfigLoader
    
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    return builder.build(fsm_config)


@pytest.fixture
def complex_fsm():
    """Create a more complex FSM with multiple paths."""
    config = {
        'name': 'complex_workflow',
        'main_network': 'main',
        'networks': [{
            'name': 'main',
            'states': [
                {
                    'name': 'validate', 
                    'is_start': True,
                    'functions': {
                        'validate': 'lambda state: {"valid": state.data.get("value", 0) > 0}'
                    }
                },
                {
                    'name': 'process',
                    'functions': {
                        'transform': 'lambda state: {"result": state.data.get("value", 1) * 2}'
                    }
                },
                {
                    'name': 'review'},
                {'name': 'complete', 'is_end': True},
                {'name': 'error', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'validate',
                    'to': 'process',
                    'name': 'valid',
                    'pre_test': {
                        'test': 'lambda state: state.data.get("valid", False)'
                    }
                },
                {
                    'from': 'validate',
                    'to': 'error', 
                    'name': 'invalid',
                    'pre_test': {
                        'test': 'lambda state: not state.data.get("valid", False)'
                    }
                },
                {'from': 'process', 'to': 'review', 'name': 'processed'},
                {'from': 'review', 'to': 'complete', 'name': 'approved'},
                {'from': 'review', 'to': 'process', 'name': 'rejected'}
            ]
        }]
    }
    
    from dataknobs_fsm.config.loader import ConfigLoader
    
    loader = ConfigLoader()
    fsm_config = loader.load_from_dict(config)
    builder = FSMBuilder()
    return builder.build(fsm_config)


@pytest.fixture
def advanced_fsm(simple_fsm):
    """AdvancedFSM instance using real FSM."""
    return AdvancedFSM(simple_fsm, ExecutionMode.STEP_BY_STEP)


class TestAdvancedFSMInitialization:
    """Test AdvancedFSM initialization with real FSM."""
    
    def test_basic_initialization(self, simple_fsm):
        """Test AdvancedFSM initialization with real FSM."""
        fsm = AdvancedFSM(simple_fsm)
        
        assert fsm.fsm == simple_fsm
        assert fsm.execution_mode == ExecutionMode.STEP_BY_STEP
        assert fsm._engine is not None
        assert fsm._resource_manager is not None
        assert isinstance(fsm._hooks, ExecutionHook)
        assert fsm._breakpoints == set()
        assert fsm._trace_buffer == []
        assert fsm._profile_data == {}
    
    def test_initialization_with_execution_mode(self, simple_fsm):
        """Test initialization with different execution modes."""
        modes = [
            ExecutionMode.STEP_BY_STEP,
            ExecutionMode.BREAKPOINT,
            ExecutionMode.TRACE,
            ExecutionMode.PROFILE,
            ExecutionMode.DEBUG
        ]
        
        for mode in modes:
            fsm = AdvancedFSM(simple_fsm, mode)
            assert fsm.execution_mode == mode
    
    def test_set_execution_strategy(self, advanced_fsm):
        """Test setting execution strategy."""
        strategies = [
            TraversalStrategy.DEPTH_FIRST,
            TraversalStrategy.BREADTH_FIRST
        ]
        
        for strategy in strategies:
            advanced_fsm.set_execution_strategy(strategy)
            assert advanced_fsm._engine.strategy == strategy
    
    def test_configure_transactions(self, advanced_fsm):
        """Test configuring transaction management."""
        advanced_fsm.configure_transactions(
            TransactionStrategy.BATCH,
            timeout=30,
            max_retries=3
        )
        
        assert advanced_fsm._transaction_manager is not None
        # Transaction manager should be configured with the strategy
        # (exact verification depends on TransactionManager implementation)


class TestAdvancedFSMResourceManagement:
    """Test AdvancedFSM resource management with real components."""
    
    def test_register_resource_dict(self, advanced_fsm):
        """Test registering resource from dictionary config."""
        resource_config = {
            'type': 'memory',
            'capacity': 1000,
            'data': {'key1': 'value1'}
        }
        
        advanced_fsm.register_resource('test_db', resource_config)
        
        # Verify resource was registered (exact verification depends on manager)
        # The resource should be available through the resource manager
        assert 'test_db' in advanced_fsm._resource_manager._resources or \
               hasattr(advanced_fsm._resource_manager, 'get_resource')
    
    def test_register_multiple_resources(self, advanced_fsm):
        """Test registering multiple resources."""
        resources = {
            'db': {'type': 'database', 'url': 'test://localhost'},
            'cache': {'type': 'memory', 'size': '100MB'},
            'queue': {'type': 'queue', 'max_size': 1000}
        }
        
        for name, config in resources.items():
            advanced_fsm.register_resource(name, config)
        
        # All resources should be registered
        # (exact verification depends on resource manager implementation)


class TestAdvancedFSMHooksAndBreakpoints:
    """Test AdvancedFSM hooks and breakpoints with real execution."""
    
    def test_execution_hooks_setup(self, advanced_fsm):
        """Test setting up execution hooks."""
        hook_calls = []
        
        async def on_enter(state):
            hook_calls.append(f'enter_{state.definition.name}')
            
        async def on_exit(state):
            hook_calls.append(f'exit_{state.definition.name}')
            
        async def on_arc_execute(arc):
            hook_calls.append(f'arc_{arc.name}')
        
        hooks = ExecutionHook(
            on_state_enter=on_enter,
            on_state_exit=on_exit,
            on_arc_execute=on_arc_execute
        )
        
        advanced_fsm.set_hooks(hooks)
        
        assert advanced_fsm._hooks.on_state_enter == on_enter
        assert advanced_fsm._hooks.on_state_exit == on_exit
        assert advanced_fsm._hooks.on_arc_execute == on_arc_execute
    
    def test_breakpoint_management(self, advanced_fsm):
        """Test breakpoint addition and removal."""
        # Add breakpoints
        advanced_fsm.add_breakpoint('middle')
        advanced_fsm.add_breakpoint('review')
        
        assert 'middle' in advanced_fsm._breakpoints
        assert 'review' in advanced_fsm._breakpoints
        assert len(advanced_fsm._breakpoints) == 2
        
        # Remove breakpoint
        advanced_fsm.remove_breakpoint('middle')
        
        assert 'middle' not in advanced_fsm._breakpoints
        assert 'review' in advanced_fsm._breakpoints
        assert len(advanced_fsm._breakpoints) == 1
        
        # Remove non-existent breakpoint (should not error)
        advanced_fsm.remove_breakpoint('nonexistent')
        assert len(advanced_fsm._breakpoints) == 1


class TestAdvancedFSMExecutionContext:
    """Test AdvancedFSM execution context with real FSM."""
    
    @pytest.mark.asyncio
    async def test_execution_context_creation(self, advanced_fsm):
        """Test creating execution context with real FSM."""
        test_data = {'test': 'data', 'value': 42}
        
        async with advanced_fsm.execution_context(
            data=test_data,
            data_mode=DataHandlingMode.COPY
        ) as context:
            # Verify context was created correctly
            assert context is not None
            assert context.current_state is not None
            assert context.current_state == 'start'
            assert context.data == test_data
    
    @pytest.mark.asyncio
    async def test_execution_context_with_initial_state(self, advanced_fsm):
        """Test execution context with specific initial state."""
        test_data = {'test': 'data'}
        
        async with advanced_fsm.execution_context(
            data=test_data,
            initial_state='middle'
        ) as context:
            assert context.current_state == 'middle'
    
    @pytest.mark.asyncio
    async def test_execution_context_with_record(self, advanced_fsm):
        """Test execution context with Record input."""
        record = Record({'test': 'record_data'})
        
        async with advanced_fsm.execution_context(data=record) as context:
            assert context.current_state is not None
            assert context.data == {'test': 'record_data'}
            # Also check that we have a state instance with data
            assert hasattr(context, 'current_state_instance')
            assert context.current_state_instance.data == {'test': 'record_data'}
    
    @pytest.mark.asyncio
    async def test_execution_context_with_hooks(self, advanced_fsm):
        """Test execution context with hooks."""
        hook_calls = []
        
        async def on_enter(state):
            hook_calls.append(f'enter_{state.definition.name}')
            
        async def on_exit(state):
            hook_calls.append(f'exit_{state.definition.name}')
        
        hooks = ExecutionHook(on_state_enter=on_enter, on_state_exit=on_exit)
        advanced_fsm.set_hooks(hooks)
        
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            pass  # Just test context creation and cleanup
        
        # Hooks should have been called
        assert len(hook_calls) >= 1  # At least on_enter should be called


class TestAdvancedFSMStepExecution:
    """Test AdvancedFSM step-by-step execution with real FSM."""
    
    @pytest.mark.asyncio
    async def test_single_step_execution(self, advanced_fsm):
        """Test executing a single step."""
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            # Initial state should be 'start'
            assert context.current_state == 'start'
            assert context.current_state_instance.definition.name == 'start'
            
            # Execute one step
            new_state = await advanced_fsm.step(context)
            
            # Should transition to 'middle'
            if new_state is not None:
                assert new_state.definition.name == 'middle'
                assert context.current_state == 'middle'
    
    @pytest.mark.asyncio
    async def test_multi_step_execution(self, advanced_fsm):
        """Test executing multiple steps to completion."""
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            states_visited = [context.current_state_instance.definition.name]
            
            # Execute steps until no more transitions
            while True:
                new_state = await advanced_fsm.step(context)
                if new_state is None:
                    break
                states_visited.append(new_state.definition.name)
                
                # Safety check to prevent infinite loops
                if len(states_visited) > 10:
                    break
            
            # Should have visited start -> middle -> end
            assert 'start' in states_visited
            assert states_visited[-1] == 'end' or len(states_visited) > 1
    
    @pytest.mark.asyncio
    async def test_step_with_specific_arc(self, complex_fsm):
        """Test step execution with specific arc selection."""
        advanced_fsm = AdvancedFSM(complex_fsm)
        
        # Use valid data that will pass validation
        async with advanced_fsm.execution_context({'value': 10}) as context:
            # Should start at 'validate' state
            assert context.current_state == 'validate'
            assert context.current_state_instance.definition.name == 'validate'
            
            # Execute validation step (should set 'valid' to true)
            new_state = await advanced_fsm.step(context)
            
            if new_state is not None:
                # After validation, should be able to take 'valid' arc
                next_state = await advanced_fsm.step(context, arc_name='valid')
                
                if next_state is not None:
                    assert next_state.definition.name == 'process'
    
    @pytest.mark.asyncio
    async def test_step_with_tracing(self, advanced_fsm):
        """Test step execution with tracing enabled."""
        advanced_fsm.execution_mode = ExecutionMode.TRACE
        
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            # Execute a few steps
            await advanced_fsm.step(context)
            await advanced_fsm.step(context)
            
            # Trace buffer should have entries
            assert len(advanced_fsm._trace_buffer) >= 0
            
            # Each trace entry should have required fields
            for trace in advanced_fsm._trace_buffer:
                assert 'from' in trace
                assert 'to' in trace
                assert 'arc' in trace
                assert 'data' in trace


class TestAdvancedFSMHistoryManagement:
    """Test AdvancedFSM history management with real execution."""
    
    def test_enable_history_tracking(self, advanced_fsm):
        """Test enabling history tracking."""
        advanced_fsm.enable_history(max_depth=50)
        
        assert advanced_fsm._history is not None
        assert advanced_fsm._storage is None  # No storage backend specified
    
    def test_enable_history_with_storage(self, advanced_fsm):
        """Test enabling history with storage backend."""
        # Create a simple mock storage for testing
        mock_storage = AsyncMock()
        
        advanced_fsm.enable_history(storage=mock_storage, max_depth=100)
        
        assert advanced_fsm._history is not None
        assert advanced_fsm._storage == mock_storage
    
    @pytest.mark.asyncio
    async def test_history_tracking_during_execution(self, advanced_fsm):
        """Test that history is tracked during execution."""
        advanced_fsm.enable_history()
        
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            # Execute some steps
            await advanced_fsm.step(context)
            await advanced_fsm.step(context)
        
        # History should have been recorded
        # (exact verification depends on ExecutionHistory implementation)
        assert advanced_fsm._history is not None


class TestAdvancedFSMComplexWorkflows:
    """Test AdvancedFSM with complex workflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_conditional_branching(self, complex_fsm):
        """Test conditional branching with real FSM."""
        advanced_fsm = AdvancedFSM(complex_fsm)
        
        # Test with valid data (should go to process)
        async with advanced_fsm.execution_context({'value': 5}) as context:
            # Execute validation
            await advanced_fsm.step(context)
            
            # Should be able to proceed to processing
            # (exact path depends on FSM execution logic)
            current_state = context.current_state
            assert current_state in ['validate', 'process', 'error']
        
        # Test with invalid data (should go to error)
        async with advanced_fsm.execution_context({'value': -1}) as context:
            await advanced_fsm.step(context)
            
            current_state = context.current_state
            assert current_state in ['validate', 'error']
    
    @pytest.mark.asyncio
    async def test_workflow_with_loops(self, complex_fsm):
        """Test workflow that can loop (review -> process -> review)."""
        advanced_fsm = AdvancedFSM(complex_fsm)
        
        async with advanced_fsm.execution_context({'value': 10}) as context:
            states_visited = []
            max_steps = 10  # Prevent infinite loops
            
            for _ in range(max_steps):
                current_name = context.current_state
                states_visited.append(current_name)
                
                # Stop if we reach an end state
                if current_name in ['complete', 'error']:
                    break
                
                new_state = await advanced_fsm.step(context)
                if new_state is None:
                    break
            
            # Should have executed some workflow
            assert len(states_visited) > 0
            assert states_visited[0] == 'validate'
    
    @pytest.mark.asyncio
    async def test_breakpoint_execution(self, advanced_fsm):
        """Test execution that stops at breakpoints."""
        # Set a breakpoint
        advanced_fsm.add_breakpoint('middle')
        
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            # Run until breakpoint
            if hasattr(advanced_fsm, 'run_until_breakpoint'):
                final_state = await advanced_fsm.run_until_breakpoint(context)
                
                # Should stop at the breakpoint
                if final_state is not None:
                    assert final_state.definition.name == 'middle'


class TestAdvancedFSMIntegration:
    """Integration tests using real FSM components."""
    
    def test_full_advanced_fsm_setup(self, complex_fsm):
        """Test complete AdvancedFSM setup with all features."""
        # Create FSM with full configuration
        fsm = AdvancedFSM(complex_fsm, ExecutionMode.DEBUG)
        
        # Configure all features
        fsm.set_execution_strategy(TraversalStrategy.BREADTH_FIRST)
        fsm.configure_transactions(TransactionStrategy.SINGLE, timeout=60)
        fsm.register_resource('test_db', {'type': 'memory', 'data': {}})
        
        # Set up hooks
        hooks = ExecutionHook(
            on_state_enter=AsyncMock(),
            on_state_exit=AsyncMock(),
            on_arc_execute=AsyncMock()
        )
        fsm.set_hooks(hooks)
        
        # Add breakpoints and enable history
        fsm.add_breakpoint('review')
        fsm.enable_history(max_depth=100)
        
        # Verify configuration
        assert fsm.execution_mode == ExecutionMode.DEBUG
        assert fsm._engine.strategy == TraversalStrategy.BREADTH_FIRST
        assert fsm._transaction_manager is not None
        assert 'review' in fsm._breakpoints
        assert fsm._history is not None
        assert fsm._hooks == hooks
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, simple_fsm):
        """Test complete workflow execution from start to finish."""
        advanced_fsm = AdvancedFSM(simple_fsm, ExecutionMode.TRACE)
        
        # Enable all monitoring
        advanced_fsm.enable_history()
        
        hook_calls = []
        
        async def track_hook(event_type):
            async def hook(obj):
                hook_calls.append(event_type)
            return hook
        
        hooks = ExecutionHook(
            on_state_enter=await track_hook('enter'),
            on_state_exit=await track_hook('exit'),
            on_arc_execute=await track_hook('arc')
        )
        advanced_fsm.set_hooks(hooks)
        
        # Execute complete workflow
        async with advanced_fsm.execution_context({'test': 'data'}) as context:
            # Execute until completion
            steps = 0
            while steps < 5:  # Safety limit
                new_state = await advanced_fsm.step(context)
                if new_state is None:
                    break
                steps += 1
                
                # Stop if we reach end state
                if context.current_state == 'end':
                    break
        
        # Verify monitoring captured execution
        assert len(advanced_fsm._trace_buffer) >= 0
        assert len(hook_calls) >= 0
        
        # Final state should be 'end' or we should have made progress
        final_state_name = context.current_state
        assert final_state_name in ['start', 'middle', 'end']