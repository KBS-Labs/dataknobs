"""Tests for StateTransform execution behavior.

This test suite ensures that StateTransforms are executed exactly once
when entering a state, and verifies the proper separation of concerns
between ArcTransforms and StateTransforms.
"""

import pytest
from dataknobs_fsm.api.simple import SimpleFSM


class TestStateTransformExecution:
    """Test StateTransform execution behavior."""

    def test_state_transform_executed_once_per_state_entry(self):
        """Test that StateTransforms are executed exactly once when entering a state."""
        
        # Track how many times the transform is called
        call_count = 0
        call_data_history = []
        
        def counting_transform(state):
            """Transform that tracks its invocations."""
            nonlocal call_count, call_data_history
            call_count += 1
            call_data_history.append(state.data.copy())
            
            # Double the value if it exists, otherwise set to 100
            if 'value' in state.data:
                result = {'result': state.data['value'] * 2}
            else:
                result = {'result': 100}
            
            return result
        
        # Create FSM config with StateTransform
        config = {
            'name': 'test_transform_fsm',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'input',
                        'is_start': True
                    },
                    {
                        'name': 'transform_state',
                        'functions': {
                            'transform': 'counting_transform'
                        }
                    },
                    {
                        'name': 'output',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'input', 'to': 'transform_state', 'name': 'to_transform'},
                    {'from': 'transform_state', 'to': 'output', 'name': 'to_output'}
                ]
            }]
        }
        
        # Create FSM with the custom function
        try:
            fsm = SimpleFSM(config, custom_functions={
                'counting_transform': counting_transform
            })
            result = fsm.process({'value': 42})
            
            # Verify the transform was called exactly once
            assert call_count == 1, f"Transform should be called exactly once, but was called {call_count} times"
            
            # Verify the transform received the correct input data
            assert len(call_data_history) == 1
            assert call_data_history[0] == {'value': 42}
            
            # Verify the final result is correct
            assert result['success'] is True
            assert result['data'] == {'result': 84}  # 42 * 2
            assert result['final_state'] == 'output'
            assert result['path'] == ['input', 'transform_state', 'output']
            
        except Exception as e:
            raise e

    def test_state_transform_with_chained_transformations(self):
        """Test that StateTransforms work correctly in a chain of transformations."""
        
        transform_calls = []
        
        def first_transform(state):
            transform_calls.append(('first', state.data.copy()))
            return {'step1': state.data.get('value', 0) + 10}
        
        def second_transform(state):
            transform_calls.append(('second', state.data.copy()))
            return {'step2': state.data.get('step1', 0) * 3}
        
        config = {
            'name': 'chained_transform_fsm',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True
                    },
                    {
                        'name': 'first_transform_state',
                        'functions': {
                            'transform': 'first_transform'
                        }
                    },
                    {
                        'name': 'second_transform_state', 
                        'functions': {
                            'transform': 'second_transform'
                        }
                    },
                    {
                        'name': 'end',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'start', 'to': 'first_transform_state'},
                    {'from': 'first_transform_state', 'to': 'second_transform_state'},
                    {'from': 'second_transform_state', 'to': 'end'}
                ]
            }]
        }
        
        # Create FSM with the custom functions
        try:
            fsm = SimpleFSM(config, custom_functions={
                'first_transform': first_transform,
                'second_transform': second_transform
            })
            result = fsm.process({'value': 5})
            
            # Verify each transform was called exactly once with correct data
            assert len(transform_calls) == 2
            
            # First transform should receive initial data
            assert transform_calls[0] == ('first', {'value': 5})
            
            # Second transform should receive result of first transform
            assert transform_calls[1] == ('second', {'step1': 15})  # 5 + 10
            
            # Final result should be correct
            assert result['success'] is True
            assert result['data'] == {'step2': 45}  # (5 + 10) * 3 = 45
            assert result['path'] == ['start', 'first_transform_state', 'second_transform_state', 'end']

        except Exception as e:
            raise e

    def test_state_transform_vs_arc_transform_separation(self):
        """Test that StateTransforms and ArcTransforms are properly separated."""
        
        execution_log = []
        
        def state_transform(state):
            execution_log.append(('state_transform', state.data.copy()))
            return {'state_result': state.data.get('value', 0) * 10}
        
        def arc_transform(data, context):
            execution_log.append(('arc_transform', data.copy() if isinstance(data, dict) else data))
            if isinstance(data, dict) and 'value' in data:
                return {'arc_result': data['value'] + 100}
            return data
        
        config = {
            'name': 'mixed_transform_fsm',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True
                    },
                    {
                        'name': 'transform_state',
                        'functions': {
                            'transform': 'state_transform'
                        },
                        'arcs': [{
                            'target': 'end',
                            'transform': {
                                'type': 'inline',
                                'code': 'arc_transform'
                            }
                        }]
                    },
                    {
                        'name': 'end',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'start', 'to': 'transform_state'}
                ]
            }]
        }
        
        # Create FSM with the custom functions
        try:
            fsm = SimpleFSM(config, custom_functions={
                'state_transform': state_transform,
                'arc_transform': arc_transform
            })
            result = fsm.process({'value': 7})
            
            # Verify execution order and data flow
            assert len(execution_log) == 2
            
            # StateTransform should be called first when entering the state
            assert execution_log[0] == ('state_transform', {'value': 7})
            
            # ArcTransform should be called during arc traversal with state transform result
            assert execution_log[1] == ('arc_transform', {'state_result': 70})  # 7 * 10
            
            # Note: In the current implementation, if both state and arc transforms exist,
            # the arc transform would be applied during transition, but since our test 
            # has the arc going from transform_state to end, and the state transform
            # changes the data, the arc transform gets the state-transformed data.
            
            assert result['success'] is True

        except Exception as e:
            raise e

    def test_state_transform_not_executed_on_state_validation(self):
        """Test that StateTransforms are not executed during state validation phase."""
        
        transform_call_count = 0
        validator_call_count = 0
        
        def test_validator(data):
            nonlocal validator_call_count
            validator_call_count += 1
            return True  # Always pass validation
        
        def test_transform(state):
            nonlocal transform_call_count
            transform_call_count += 1
            return {'transformed': True}
        
        config = {
            'name': 'validation_transform_fsm',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True
                    },
                    {
                        'name': 'validated_state',
                        'validators': [{
                            'type': 'inline',
                            'code': 'test_validator'
                        }],
                        'transforms': [{
                            'type': 'inline', 
                            'code': 'test_transform'
                        }]
                    },
                    {
                        'name': 'end',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'start', 'to': 'validated_state'},
                    {'from': 'validated_state', 'to': 'end'}
                ]
            }]
        }
        
        # Register the functions with SimpleFSM
        try:
            fsm = SimpleFSM(config, custom_functions={
                'test_validator': test_validator,
                'test_transform': test_transform
            })
            result = fsm.process({'test': 'data'})
            
            # Validator should be called during state evaluation
            assert validator_call_count >= 1, "Validator should be called"
            
            # Transform should be called exactly once when entering the state
            assert transform_call_count == 1, f"Transform should be called exactly once, got {transform_call_count}"
            
            assert result['success'] is True
            assert result['data'] == {'transformed': True}

        except Exception as e:
            raise e