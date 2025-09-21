"""Test specifically for the duplicate StateTransform execution fix.

This test ensures that StateTransforms are not executed twice during FSM execution,
which was the bug we discovered and fixed where transforms were being called both
in _execute_state_functions and _execute_state_transforms.
"""

import pytest
from dataknobs_fsm.api.simple import SimpleFSM


class TestDuplicateStateTransformFix:
    """Test for the specific duplicate StateTransform execution bug fix."""

    def test_state_transform_not_executed_twice(self):
        """
        Test that StateTransforms are executed exactly once, not twice.
        
        This test reproduces the original bug where StateTransforms were called
        both by _execute_state_functions (before transition evaluation) and
        _execute_state_transforms (after entering the state).
        
        The bug caused transforms to be applied twice:
        1. First call: {'value': 5} -> {'result': 10}
        2. Second call: {'result': 10} -> {'result': 2} (using default value 1)
        """
        
        # Track every call to the transform function
        transform_calls = []
        
        def tracking_transform(state):
            """Transform that tracks all its invocations."""
            call_data = state.data.copy()
            transform_calls.append(call_data)
            
            # This mimics the original lambda from our debug test:
            # lambda state: {"result": state.data.get("value", 1) * 2}
            result_value = state.data.get("value", 1) * 2
            return {"result": result_value}
        
        # Create FSM config identical to our debug test
        config = {
            'name': 'duplicate_transform_test',
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
                            'transform': 'tracking_transform'
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
        
        # Execute the FSM with custom function
        fsm = SimpleFSM(config, custom_functions={
            'tracking_transform': tracking_transform
        })
        result = fsm.process({'value': 5})

        # CRITICAL ASSERTION: Transform should be called exactly once
        assert len(transform_calls) == 1, (
            f"StateTransform should be called exactly once, but was called "
            f"{len(transform_calls)} times with data: {transform_calls}"
        )

        # Verify the single call received the correct input
        assert transform_calls[0] == {'value': 5}, (
            f"Transform should receive original input {{'value': 5}}, "
            f"but received {transform_calls[0]}"
        )

        # Verify the final result is correct (5 * 2 = 10, not 1 * 2 = 2)
        assert result['success'] is True
        assert result['data'] == {'result': 10}, (
            f"Final result should be {{'result': 10}} (5 * 2), "
            f"but got {result['data']}"
        )

        # Verify full execution path
        assert result['final_state'] == 'output'
        assert result['path'] == ['input', 'multiply', 'output']

    def test_state_transform_with_various_input_values(self):
        """
        Test that StateTransforms work correctly with different input values.
        
        This ensures our fix works for various inputs, not just the specific
        test case that revealed the bug.
        """
        
        transform_calls = []
        
        def value_doubler(state):
            transform_calls.append(state.data.copy())
            return {"doubled": state.data.get("number", 0) * 2}
        
        config = {
            'name': 'value_doubler_test',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True
                    },
                    {
                        'name': 'doubler',
                        'functions': {
                            'transform': 'value_doubler'
                        }
                    },
                    {
                        'name': 'end',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'start', 'to': 'doubler'},
                    {'from': 'doubler', 'to': 'end'}
                ]
            }]
        }
        
        # Create FSM with custom function
        fsm = SimpleFSM(config, custom_functions={
            'value_doubler': value_doubler
        })
        # Test with different input values
        test_cases = [
            ({'number': 1}, {'doubled': 2}),
            ({'number': 10}, {'doubled': 20}),
            ({'number': 0}, {'doubled': 0}),
            ({'number': -5}, {'doubled': -10}),
            ({'other': 'data'}, {'doubled': 0}),  # Missing 'number' key
        ]

        for i, (input_data, expected_output) in enumerate(test_cases):
            # Reset call tracking for each test
            transform_calls.clear()

            result = fsm.process(input_data)

            # Verify single call for each execution
            assert len(transform_calls) == 1, (
                f"Test case {i}: Transform should be called once, "
                f"got {len(transform_calls)} calls"
            )

            # Verify correct input received
            assert transform_calls[0] == input_data, (
                f"Test case {i}: Transform should receive {input_data}, "
                f"got {transform_calls[0]}"
            )

            # Verify correct output
            assert result['success'] is True
            assert result['data'] == expected_output, (
                f"Test case {i}: Expected {expected_output}, "
                f"got {result['data']}"
            )

    def test_multiple_states_with_transforms(self):
        """
        Test that multiple states each execute their transforms exactly once.
        
        This ensures the fix works correctly in a more complex FSM with
        multiple transform states.
        """
        
        all_transform_calls = []
        
        def first_transform(state):
            all_transform_calls.append(('first', state.data.copy()))
            return {"step1": state.data.get("value", 0) + 100}
        
        def second_transform(state):
            all_transform_calls.append(('second', state.data.copy()))
            return {"step2": state.data.get("step1", 0) * 3}
        
        def third_transform(state):
            all_transform_calls.append(('third', state.data.copy()))
            return {"final": state.data.get("step2", 0) - 50}
        
        config = {
            'name': 'multi_transform_test',
            'main_network': 'main',
            'networks': [{
                'name': 'main',
                'states': [
                    {
                        'name': 'start',
                        'is_start': True
                    },
                    {
                        'name': 'transform1',
                        'functions': {'transform': 'first_transform'}
                    },
                    {
                        'name': 'transform2',
                        'functions': {'transform': 'second_transform'}
                    },
                    {
                        'name': 'transform3',
                        'functions': {'transform': 'third_transform'}
                    },
                    {
                        'name': 'end',
                        'is_end': True
                    }
                ],
                'arcs': [
                    {'from': 'start', 'to': 'transform1'},
                    {'from': 'transform1', 'to': 'transform2'},
                    {'from': 'transform2', 'to': 'transform3'},
                    {'from': 'transform3', 'to': 'end'}
                ]
            }]
        }
        
        # Create FSM with custom functions
        fsm = SimpleFSM(config, custom_functions={
            'first_transform': first_transform,
            'second_transform': second_transform,
            'third_transform': third_transform
        })
        result = fsm.process({'value': 10})

        # Verify exactly 3 transform calls (one per transform state)
        assert len(all_transform_calls) == 3, (
            f"Should have exactly 3 transform calls, got {len(all_transform_calls)}: "
            f"{all_transform_calls}"
        )

        # Verify each transform was called with correct data
        expected_calls = [
            ('first', {'value': 10}),
            ('second', {'step1': 110}),  # 10 + 100
            ('third', {'step2': 330})    # 110 * 3
        ]

        assert all_transform_calls == expected_calls, (
            f"Transform calls don't match expected sequence.\n"
            f"Expected: {expected_calls}\n"
            f"Actual: {all_transform_calls}"
        )

        # Verify final result
        assert result['success'] is True
        assert result['data'] == {'final': 280}  # (10 + 100) * 3 - 50 = 280
        assert result['path'] == ['start', 'transform1', 'transform2', 'transform3', 'end']