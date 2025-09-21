"""Test custom initial state specification in push arcs."""

import pytest
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    PushArcConfig,
    FunctionReference,
)
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.network import NetworkExecutor
from dataknobs_fsm.core.modes import ProcessingMode


class TestPushArcCustomInitial:
    """Test custom initial state in push arcs using network:state syntax."""

    def test_push_to_specific_state(self):
        """Test pushing to a specific non-initial state in a subnetwork."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[
                                # Push to specific state in subnetwork
                                PushArcConfig(
                                    target="end",
                                    target_network="sub:middle",  # Skip 'sub_start', go directly to 'middle'
                                    return_state="end"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['main_start'] = True
    context.variables['execution_path'] = ['main_start']
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="end",
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['main_end'] = True
    context.variables['execution_path'].append('main_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="sub_start",
                            is_start=True,  # This is the default initial state
                            arcs=[
                                {"target": "middle"}
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    # This should be skipped when we push to sub:middle
    data['sub_start'] = True
    context.variables['execution_path'].append('sub_start')
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="middle",
                            arcs=[
                                {"target": "sub_end"}
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['middle'] = True
    context.variables['execution_path'].append('middle')
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="sub_end",
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['sub_end'] = True
    context.variables['execution_path'].append('sub_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        # Build and execute
        builder = FSMBuilder()
        fsm = builder.build(config)
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        executor = NetworkExecutor(fsm)

        success, result = executor.execute_network("main", context, {})
        assert success, f"Execution failed: {result}"

        # Verify execution path
        execution_path = context.variables.get('execution_path', [])
        print(f"\nExecution path: {execution_path}")

        # Key assertions
        assert 'main_start' in execution_path, "Main start should execute"
        assert 'sub_start' not in execution_path, "Sub start should be skipped"
        assert 'middle' in execution_path, "Middle state should execute"
        assert 'sub_end' in execution_path, "Sub end should execute"
        assert 'main_end' in execution_path, "Main end should execute"

        # Verify data
        assert result.get('main_start') == True
        assert result.get('sub_start') != True, "sub_start should not have executed"
        assert result.get('middle') == True
        assert result.get('sub_end') == True
        assert result.get('main_end') == True

    def test_push_to_nonexistent_state_fails(self):
        """Test that pushing to a non-existent state fails gracefully."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[
                                PushArcConfig(
                                    target="end",
                                    target_network="sub:nonexistent",  # This state doesn't exist
                                    return_state="end"
                                )
                            ]
                        ),
                        StateConfig(
                            name="end",
                            is_end=True
                        )
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="only_state",
                            is_start=True,
                            is_end=True
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        executor = NetworkExecutor(fsm)

        success, result = executor.execute_network("main", context, {})
        assert not success, "Should fail when pushing to non-existent state"
        assert "not found" in str(result).lower() or "no valid transition" in str(result).lower()

    def test_multiple_push_arcs_different_initial_states(self):
        """Test multiple push arcs targeting different initial states in the same subnetwork."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[
                                {
                                    "target": "path1",
                                    "condition": FunctionReference(
                                        type="inline",
                                        code="lambda data, context: data.get('path') == 1"
                                    )
                                },
                                {
                                    "target": "path2",
                                    "condition": FunctionReference(
                                        type="inline",
                                        code="lambda data, context: data.get('path') == 2"
                                    )
                                }
                            ]
                        ),
                        StateConfig(
                            name="path1",
                            arcs=[
                                PushArcConfig(
                                    target="end",
                                    target_network="sub:state_a",  # Enter at state_a
                                    return_state="end"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="def transform(data, context): data['took_path'] = 'path1'; return data"
                                )
                            ]
                        ),
                        StateConfig(
                            name="path2",
                            arcs=[
                                PushArcConfig(
                                    target="end",
                                    target_network="sub:state_b",  # Enter at state_b
                                    return_state="end"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="def transform(data, context): data['took_path'] = 'path2'; return data"
                                )
                            ]
                        ),
                        StateConfig(
                            name="end",
                            is_end=True
                        )
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="state_a",
                            is_start=True,
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="def transform(data, context): data['entered_at'] = 'state_a'; return data"
                                )
                            ]
                        ),
                        StateConfig(
                            name="state_b",
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="def transform(data, context): data['entered_at'] = 'state_b'; return data"
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        # Test path 1
        context1 = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        executor1 = NetworkExecutor(fsm)
        success1, result1 = executor1.execute_network("main", context1, {"path": 1})
        print(f"Path 1 result: {result1}")
        assert success1, f"Path 1 failed: {result1}"
        assert result1.get('entered_at') == 'state_a', f"Path 1 should enter at state_a, got {result1.get('entered_at')}"

        # Test path 2
        context2 = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        executor2 = NetworkExecutor(fsm)
        success2, result2 = executor2.execute_network("main", context2, {"path": 2})
        print(f"Path 2 result: {result2}")
        assert success2, f"Path 2 failed: {result2}"
        assert result2.get('entered_at') == 'state_b', f"Path 2 should enter at state_b, got {result2.get('entered_at')}"

    def test_backward_compatibility(self):
        """Test that the old syntax (network name only) still works."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[
                                PushArcConfig(
                                    target="end",
                                    target_network="sub",  # Old syntax - no colon
                                    return_state="end"
                                )
                            ]
                        ),
                        StateConfig(
                            name="end",
                            is_end=True
                        )
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="default_initial",
                            is_start=True,
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="def transform(data, context): data['used_default'] = True; return data"
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        executor = NetworkExecutor(fsm)

        success, result = executor.execute_network("main", context, {})
        assert success
        assert result.get('used_default') == True, "Should use default initial state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])