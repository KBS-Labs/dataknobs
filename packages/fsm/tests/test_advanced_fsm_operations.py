"""Unit tests for AdvancedFSM operations demonstrated in advanced_debugging_simple.py."""

import pytest
import asyncio
from dataknobs_fsm.api.advanced import (
    AdvancedFSM,
    ExecutionMode,
    ExecutionHook,
    create_advanced_fsm
)


class TestAdvancedFSMOperations:
    """Test all AdvancedFSM operations from the example."""
    
    @pytest.fixture
    def workflow_config(self):
        """The same workflow configuration from the example."""
        return {
            "name": "SimpleWorkflow",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True
                    },
                    {
                        "name": "initialize",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "initialize_workflow"
                            }
                        }
                    },
                    {
                        "name": "validate",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "validate_input"
                            }
                        }
                    },
                    {
                        "name": "process",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "process_action"
                            }
                        }
                    },
                    {
                        "name": "finalize",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "finalize_workflow"
                            }
                        }
                    },
                    {
                        "name": "success",
                        "is_end": True
                    },
                    {
                        "name": "validation_failed",
                        "is_end": True
                    }
                ],
                "arcs": [
                    {"from": "start", "to": "initialize"},
                    {"from": "initialize", "to": "validate"},
                    {
                        "from": "validate",
                        "to": "process",
                        "condition": {
                            "type": "registered",
                            "name": "check_validation"
                        }
                    },
                    {
                        "from": "validate",
                        "to": "validation_failed",
                        "condition": {
                            "type": "inline",
                            "code": "not data.get('is_valid', False)"
                        }
                    },
                    {"from": "process", "to": "finalize"},
                    {"from": "finalize", "to": "success"}
                ]
            }]
        }
    
    @pytest.fixture
    def custom_functions(self):
        """The custom functions from the example."""
        def initialize_workflow(state):
            """Initialize the workflow."""
            data = state.data.copy()
            data['workflow_id'] = "WF-TEST"
            data['initialized'] = True
            data['steps_completed'] = []
            return data
        
        def validate_input(state):
            """Validate input data."""
            data = state.data.copy()
            
            required = ['user_id', 'action']
            missing = [f for f in required if f not in data]
            
            if missing:
                data['is_valid'] = False
                data['validation_errors'] = f"Missing: {missing}"
            else:
                data['is_valid'] = True
            
            data['steps_completed'].append('validate')
            return data
        
        def process_action(state):
            """Process the action."""
            data = state.data.copy()
            
            action = data.get('action', 'unknown')
            
            if action == 'create':
                data['result'] = {'id': 123, 'status': 'created'}
            elif action == 'update':
                data['result'] = {'id': data.get('id', 0), 'status': 'updated'}
            elif action == 'delete':
                data['result'] = {'id': data.get('id', 0), 'status': 'deleted'}
            else:
                data['result'] = {'error': 'Unknown action'}
            
            data['processed'] = True
            data['steps_completed'].append('process')
            return data
        
        def finalize_workflow(state):
            """Finalize the workflow."""
            data = state.data.copy()
            
            data['completed_at'] = "2025-01-01T00:00:00"
            data['status'] = 'completed'
            data['steps_completed'].append('finalize')
            
            return data
        
        def check_validation(data, context):
            """Check if validation passed."""
            return data.get('is_valid', False)
        
        return {
            'initialize_workflow': initialize_workflow,
            'validate_input': validate_input,
            'process_action': process_action,
            'finalize_workflow': finalize_workflow,
            'check_validation': check_validation
        }
    
    def test_create_advanced_fsm_with_step_by_step_mode(self, workflow_config, custom_functions):
        """Test creating AdvancedFSM with STEP_BY_STEP execution mode."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions,
            execution_mode=ExecutionMode.STEP_BY_STEP
        )
        
        assert isinstance(fsm, AdvancedFSM)
        assert fsm.execution_mode == ExecutionMode.STEP_BY_STEP
        assert fsm.fsm is not None
        assert fsm.fsm.name == "SimpleWorkflow"
    
    def test_add_and_remove_breakpoints(self, workflow_config, custom_functions):
        """Test adding and removing breakpoints."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions,
            execution_mode=ExecutionMode.STEP_BY_STEP
        )
        
        # Add breakpoints
        fsm.add_breakpoint('validate')
        fsm.add_breakpoint('finalize')
        
        assert 'validate' in fsm.breakpoints
        assert 'finalize' in fsm.breakpoints
        
        # Remove a breakpoint
        fsm.remove_breakpoint('validate')
        
        assert 'validate' not in fsm.breakpoints
        assert 'finalize' in fsm.breakpoints
        
        # Clear all breakpoints
        fsm.clear_breakpoints()
        
        assert len(fsm.breakpoints) == 0
    
    def test_inspect_state(self, workflow_config, custom_functions):
        """Test inspecting state information."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        # Inspect start state
        start_info = fsm.inspect_state('start')
        assert start_info['name'] == 'start'
        assert start_info['is_start'] == True
        assert start_info['is_end'] == False
        assert start_info['has_transform'] == False
        assert len(start_info['arcs']) == 1
        
        # Inspect validate state
        validate_info = fsm.inspect_state('validate')
        assert validate_info['name'] == 'validate'
        assert validate_info['is_start'] == False
        assert validate_info['is_end'] == False
        assert validate_info['has_transform'] == True
        assert len(validate_info['arcs']) == 2
        
        # Inspect process state
        process_info = fsm.inspect_state('process')
        assert process_info['name'] == 'process'
        assert process_info['has_transform'] == True
        
        # Inspect success state
        success_info = fsm.inspect_state('success')
        assert success_info['is_end'] == True
        assert len(success_info['arcs']) == 0
    
    def test_get_available_transitions(self, workflow_config, custom_functions):
        """Test getting available transitions from states."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        # Get transitions from start
        start_transitions = fsm.get_available_transitions('start')
        assert len(start_transitions) == 1
        assert start_transitions[0]['target'] == 'initialize'
        assert start_transitions[0]['has_pre_test'] == False
        
        # Get transitions from validate
        validate_transitions = fsm.get_available_transitions('validate')
        assert len(validate_transitions) == 2
        
        # Find the transition to process
        process_transition = next(t for t in validate_transitions if t['target'] == 'process')
        assert process_transition['has_pre_test'] == True
        
        # Find the transition to validation_failed
        failed_transition = next(t for t in validate_transitions if t['target'] == 'validation_failed')
        assert failed_transition['has_pre_test'] == True
        
        # Get transitions from success (should be empty)
        success_transitions = fsm.get_available_transitions('success')
        assert len(success_transitions) == 0
    
    def test_visualize_fsm(self, workflow_config, custom_functions):
        """Test FSM visualization."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        viz = fsm.visualize_fsm()
        
        # Check that visualization contains expected elements
        assert 'digraph FSM' in viz
        assert 'rankdir=LR' in viz
        assert 'start [style=filled,fillcolor=green]' in viz
        assert 'success [shape=doublecircle,style=filled,fillcolor=red]' in viz
        assert 'validation_failed [shape=doublecircle,style=filled,fillcolor=red]' in viz
        assert 'start -> initialize' in viz
        assert 'validate -> process' in viz
        assert 'validate -> validation_failed' in viz
        assert 'finalize -> success' in viz
    
    def test_set_hooks(self, workflow_config, custom_functions):
        """Test setting execution hooks."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        # Track hook calls
        hook_calls = {
            'state_enter': [],
            'state_exit': [],
            'error': []
        }
        
        def on_state_enter(state):
            hook_calls['state_enter'].append(state)
        
        def on_state_exit(state):
            hook_calls['state_exit'].append(state)
        
        def on_error(error, state, data):
            hook_calls['error'].append((error, state, data))
        
        hooks = ExecutionHook(
            on_state_enter=on_state_enter,
            on_state_exit=on_state_exit,
            on_error=on_error
        )
        
        fsm.set_hooks(hooks)
        
        # Verify hooks are set
        assert fsm.hooks is not None
        assert fsm.hooks.on_state_enter is not None
        assert fsm.hooks.on_state_exit is not None
        assert fsm.hooks.on_error is not None
    
    def test_enable_history(self, workflow_config, custom_functions):
        """Test enabling history tracking."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        # Enable history with max depth
        fsm.enable_history(max_depth=50)
        
        assert fsm.history_enabled == True
        assert fsm.max_history_depth == 50
        assert fsm.execution_history is not None
        assert len(fsm.execution_history) == 0
        
        # Disable history
        fsm.disable_history()
        
        assert fsm.history_enabled == False
    
    @pytest.mark.asyncio
    async def test_trace_execution(self, workflow_config, custom_functions):
        """Test trace execution mode."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions,
            execution_mode=ExecutionMode.TRACE
        )
        
        # Test data with valid input
        test_data = {
            'user_id': 'USER-123',
            'action': 'create',
            'data': {'name': 'Test Item'}
        }
        
        trace = await fsm.trace_execution(test_data)
        
        # Verify trace contains expected transitions
        assert len(trace) > 0
        
        # Check that trace contains the expected path
        trace_path = [f"{t['from']} → {t['to']}" for t in trace]
        expected_transitions = [
            "start → initialize",
            "initialize → validate",
            "validate → process",
            "process → finalize",
            "finalize → success"
        ]
        
        for expected in expected_transitions:
            assert any(expected in path for path in trace_path), f"Missing transition: {expected}"
    
    @pytest.mark.asyncio
    async def test_profile_execution(self, workflow_config, custom_functions):
        """Test profile execution mode."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions,
            execution_mode=ExecutionMode.PROFILE
        )
        
        # Test data with valid input
        test_data = {
            'user_id': 'USER-123',
            'action': 'create',
            'data': {'name': 'Test Item'}
        }
        
        profile = await fsm.profile_execution(test_data)
        
        # Verify profile contains expected metrics
        assert 'total_time' in profile
        assert 'transitions' in profile
        assert 'avg_transition_time' in profile
        assert 'state_times' in profile
        
        assert profile['total_time'] >= 0
        assert profile['transitions'] > 0
        assert profile['avg_transition_time'] >= 0
        
        # Check state timings
        state_times = profile['state_times']
        expected_states = ['start', 'initialize', 'validate', 'process', 'finalize']
        
        for state in expected_states:
            assert state in state_times
            assert 'avg' in state_times[state]
            assert 'count' in state_times[state]
            assert state_times[state]['count'] > 0
            assert state_times[state]['avg'] >= 0
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_failure(self, workflow_config, custom_functions):
        """Test execution when validation fails."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions,
            execution_mode=ExecutionMode.TRACE
        )
        
        # Test data missing required fields
        test_data = {
            'data': {'name': 'Test Item'}
            # Missing user_id and action
        }
        
        trace = await fsm.trace_execution(test_data)
        
        # Verify it goes to validation_failed
        trace_path = [f"{t['from']} → {t['to']}" for t in trace]
        
        # Should go through initialize and validate, then to validation_failed
        assert any("validate → validation_failed" in path for path in trace_path)
        # Should NOT go to process
        assert not any("validate → process" in path for path in trace_path)
    
    @pytest.mark.asyncio
    async def test_different_actions(self, workflow_config, custom_functions):
        """Test different action types (create, update, delete)."""
        fsm = create_advanced_fsm(
            workflow_config,
            custom_functions=custom_functions
        )
        
        # Test create action
        create_data = {
            'user_id': 'USER-123',
            'action': 'create',
            'data': {'name': 'New Item'}
        }
        
        create_trace = await fsm.trace_execution(create_data)
        assert len(create_trace) > 0
        
        # Test update action
        update_data = {
            'user_id': 'USER-123',
            'action': 'update',
            'id': 456,
            'data': {'name': 'Updated Item'}
        }
        
        update_trace = await fsm.trace_execution(update_data)
        assert len(update_trace) > 0
        
        # Test delete action
        delete_data = {
            'user_id': 'USER-123',
            'action': 'delete',
            'id': 789
        }
        
        delete_trace = await fsm.trace_execution(delete_data)
        assert len(delete_trace) > 0
        
        # All should complete successfully
        for trace in [create_trace, update_trace, delete_trace]:
            trace_path = [f"{t['from']} → {t['to']}" for t in trace]
            assert any("finalize → success" in path for path in trace_path)