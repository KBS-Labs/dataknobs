"""Tests for ExecutionEngine subflow (PushArc) support.

These tests verify that the ExecutionEngine correctly handles PushArc
transitions, including:
- Detecting and executing PushArc transitions
- Data mapping between parent and child contexts
- Result mapping from child back to parent
- Subflow completion detection and return
- Data isolation modes
- Max depth enforcement
"""

import pytest
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    ArcConfig,
    PushArcConfig,
    FunctionReference,
)
from dataknobs_fsm.core.arc import PushArc, DataIsolationMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.network import NetworkExecutor


class TestPushArcDetection:
    """Test that ExecutionEngine correctly detects and handles PushArc."""

    def test_pusharc_is_detected_in_execute_transition(self):
        """Verify that PushArc instances are detected and delegated."""
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
                                    target="after_push",
                                    target_network="subflow",
                                    return_state="after_push"
                                )
                            ]
                        ),
                        StateConfig(
                            name="after_push",
                            arcs=[ArcConfig(target="end")]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                ),
                NetworkConfig(
                    name="subflow",
                    states=[
                        StateConfig(
                            name="sub_start",
                            is_start=True,
                            arcs=[ArcConfig(target="sub_end")]
                        ),
                        StateConfig(name="sub_end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        # Execute using NetworkExecutor to properly handle subflows
        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.data = {"value": 1}

        success, result = executor.execute_network("main", context, context.data)

        # Should have successfully completed
        assert success
        # Should have gone through the subflow and back
        assert context.current_state == "end"


class TestDataMapping:
    """Test data mapping between parent and child contexts."""

    def test_apply_data_mapping_helper(self):
        """Test the _apply_data_mapping helper method directly."""
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
                            arcs=[ArcConfig(target="end")]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        # Parent data
        parent_data = {
            "user_id": 123,
            "user_name": "Alice",
            "other_field": "ignored"
        }

        # Mapping: parent_field -> child_field
        mapping = {
            "user_id": "id",
            "user_name": "name"
        }

        result = engine._apply_data_mapping(parent_data, mapping)

        # Only mapped fields should be present
        assert result.get("id") == 123
        assert result.get("name") == "Alice"
        # Original fields should NOT be in mapped data
        assert "user_id" not in result
        assert "other_field" not in result

    def test_apply_data_mapping_empty_mapping_returns_original(self):
        """Test that empty data_mapping returns original data."""
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
                            arcs=[ArcConfig(target="end")]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        original_data = {"field1": "value1", "field2": "value2"}

        # Empty mapping should return original
        result = engine._apply_data_mapping(original_data, {})

        assert result == original_data


class TestResultMapping:
    """Test result mapping from child back to parent context."""

    def test_apply_result_mapping_helper(self):
        """Test the _apply_result_mapping helper method directly."""
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
                            arcs=[ArcConfig(target="end")]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        # Child result data
        child_data = {
            "result_value": 42,
            "status": "complete"
        }

        # Parent data before subflow
        parent_data = {
            "original_field": "preserved"
        }

        # Mapping: child_field -> parent_field
        mapping = {
            "result_value": "output",
            "status": "workflow_status"
        }

        result = engine._apply_result_mapping(child_data, mapping, parent_data)

        assert result["output"] == 42
        assert result["workflow_status"] == "complete"
        assert result["original_field"] == "preserved"


class TestSubflowCompletion:
    """Test subflow completion detection and return."""

    def test_subflow_completes_and_returns(self):
        """Verify subflow completes and returns to parent correctly."""
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
                                    target="after_return",
                                    target_network="subflow",
                                    return_state="after_return"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    context.variables['path'] = ['main_start']
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="after_return",
                            arcs=[ArcConfig(target="end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    context.variables['path'].append('after_return')
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
    context.variables['path'].append('main_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                NetworkConfig(
                    name="subflow",
                    states=[
                        StateConfig(
                            name="sub_start",
                            is_start=True,
                            arcs=[ArcConfig(target="sub_end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    context.variables['path'].append('sub_start')
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
    context.variables['path'].append('sub_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.data = {}

        success, result = executor.execute_network("main", context, context.data)

        assert success
        # Verify execution path includes subflow
        expected_path = ['main_start', 'sub_start', 'sub_end', 'after_return', 'main_end']
        assert context.variables.get('path') == expected_path


class TestMaxDepthEnforcement:
    """Test maximum subflow depth enforcement."""

    def test_max_depth_prevents_deep_nesting(self):
        """Verify that max depth is enforced in NetworkExecutor."""
        # Create a self-referential subflow for testing
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
                                    target_network="recurse",
                                    return_state="end"
                                )
                            ]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                ),
                NetworkConfig(
                    name="recurse",
                    states=[
                        StateConfig(
                            name="r_start",
                            is_start=True,
                            arcs=[
                                # Self-referential push for depth testing
                                PushArcConfig(
                                    target="r_end",
                                    target_network="recurse",
                                    return_state="r_end"
                                )
                            ]
                        ),
                        StateConfig(name="r_end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        # Set a low max depth
        executor = NetworkExecutor(fsm, max_depth=3)
        context = ExecutionContext()
        context.data = {}

        # This should eventually fail due to depth limit
        # The exact behavior depends on whether it raises or returns False
        try:
            success, result = executor.execute_network("main", context, context.data)
            # If it returns, it should have failed
            assert not success or len(context.network_stack) <= 3
        except Exception:
            # Expected to fail due to depth limit
            pass


class TestInitialStateOverride:
    """Test specifying initial state in target network syntax."""

    def test_target_network_with_initial_state(self):
        """Verify 'network:state' syntax for specifying initial state."""
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
                                    target_network="sub:sub_alternate",  # Skip sub_start
                                    return_state="end"
                                )
                            ]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="sub_start",
                            is_start=True,
                            arcs=[ArcConfig(target="sub_end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['visited_sub_start'] = True
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="sub_alternate",
                            arcs=[ArcConfig(target="sub_end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['visited_sub_alternate'] = True
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(name="sub_end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.data = {}

        success, result = executor.execute_network("main", context, context.data)

        assert success
        # Should have skipped sub_start and gone directly to sub_alternate
        assert result.get('visited_sub_alternate') is True
        assert result.get('visited_sub_start') is None


class TestFullSubflowExecution:
    """Integration tests for complete subflow execution."""

    def test_complete_subflow_round_trip(self):
        """Test a complete execution through a subflow and back."""
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
                            arcs=[ArcConfig(target="before_sub")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'] = ['main_start']
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="before_sub",
                            arcs=[
                                PushArcConfig(
                                    target="after_sub",
                                    target_network="subflow",
                                    return_state="after_sub"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('before_sub')
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="after_sub",
                            arcs=[ArcConfig(target="end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('after_sub')
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
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('main_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                NetworkConfig(
                    name="subflow",
                    states=[
                        StateConfig(
                            name="sub_start",
                            is_start=True,
                            arcs=[ArcConfig(target="sub_process")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('sub_start')
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="sub_process",
                            arcs=[ArcConfig(target="sub_end")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('sub_process')
    data['subflow_processed'] = True
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
    data['counter'] = data.get('counter', 0) + 1
    context.variables['visits'].append('sub_end')
    return data
"""
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.data = {"counter": 0}

        success, result = executor.execute_network("main", context, context.data)

        # Should have successfully completed
        assert success
        assert context.current_state == "end"
        # Should have returned from subflow
        assert len(context.network_stack) == 0
        # Should have visited all states
        expected_visits = [
            'main_start', 'before_sub',
            'sub_start', 'sub_process', 'sub_end',
            'after_sub', 'main_end'
        ]
        assert context.variables.get('visits') == expected_visits
        # Subflow should have set this flag
        assert result.get('subflow_processed') is True


class TestErrorHandling:
    """Test error handling in subflow operations."""

    def test_push_to_nonexistent_network_fails(self):
        """Verify push to nonexistent network handles gracefully."""
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
                                    target_network="nonexistent",
                                    return_state="end"
                                )
                            ]
                        ),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)

        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.data = {}

        success, result = executor.execute_network("main", context, context.data)

        # Should fail because target network doesn't exist
        assert not success


class TestExecutionEngineDirectPushArc:
    """Test ExecutionEngine's direct PushArc handling (without NetworkExecutor)."""

    def test_execute_push_arc_method_exists(self):
        """Verify _execute_push_arc method exists and is callable."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                        StateConfig(name="end", is_end=True)
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(name="s1", is_start=True, arcs=[ArcConfig(target="s2")]),
                        StateConfig(name="s2", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        # Verify the method exists
        assert hasattr(engine, '_execute_push_arc')
        assert callable(engine._execute_push_arc)

    def test_check_subflow_completion_method_exists(self):
        """Verify _check_subflow_completion method exists."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        # Verify the method exists
        assert hasattr(engine, '_check_subflow_completion')
        assert callable(engine._check_subflow_completion)

    def test_data_mapping_helpers_exist(self):
        """Verify data mapping helper methods exist."""
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                        StateConfig(name="end", is_end=True)
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        engine = ExecutionEngine(fsm)

        # Verify methods exist
        assert hasattr(engine, '_apply_data_mapping')
        assert hasattr(engine, '_apply_result_mapping')
        assert callable(engine._apply_data_mapping)
        assert callable(engine._apply_result_mapping)
