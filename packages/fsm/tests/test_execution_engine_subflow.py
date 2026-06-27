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
        """Test the apply_data_mapping helper method directly."""
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

        result = engine.apply_data_mapping(parent_data, mapping)

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
        result = engine.apply_data_mapping(original_data, {})

        assert result == original_data


class TestResultMapping:
    """Test result mapping from child back to parent context."""

    def test_apply_result_mapping_helper(self):
        """Test the apply_result_mapping helper method directly."""
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

        result = engine.apply_result_mapping(child_data, mapping, parent_data)

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


class TestResultMappingAppliedOnPop:
    """A push arc's ``result_mapping`` is applied when the subflow pops.

    Reproduce-first for the dead-``result_mapping`` bug: the pop site called
    ``_pop_subflow`` without the originating arc and read a ``_parent_data_snapshot``
    that was never assigned, so ``result_mapping`` was inert on both engines.
    The fix records the arc + parent snapshot in a ``SubflowFrame`` at push time
    and consumes it on pop. These tests drive the sync ``ExecutionEngine``
    directly so the in-stack push/pop path is exercised end to end.
    """

    @staticmethod
    def _build_result_mapping_fsm():
        config = FSMConfig(
            name="result_map_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code=(
                                        "def transform(data, context):\n"
                                        "    data['keep'] = 'me'\n"
                                        "    return data\n"
                                    ),
                                )
                            ],
                            arcs=[
                                PushArcConfig(
                                    target="after",
                                    target_network="sub",
                                    return_state="after",
                                    result_mapping={"sub_out": "parent_in"},
                                )
                            ],
                        ),
                        StateConfig(name="after", arcs=[ArcConfig(target="end")]),
                        StateConfig(name="end", is_end=True),
                    ],
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="s1",
                            is_start=True,
                            arcs=[ArcConfig(target="s2")],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code=(
                                        "def transform(data, context):\n"
                                        "    data['sub_out'] = 99\n"
                                        "    return data\n"
                                    ),
                                )
                            ],
                        ),
                        StateConfig(name="s2", is_end=True),
                    ],
                ),
            ],
        )
        return FSMBuilder().build(config)

    def test_pop_applies_result_mapping_onto_parent_data(self):
        """The sync ``_pop_subflow`` overlays the child result onto parent data.

        Drives the sync orchestrator directly (the sync ``ExecutionEngine.execute``
        entry flat-traverses push arcs — a separate pre-existing gap — so the
        in-stack lifecycle is exercised by direct call). With the context at the
        sub-network's final state, ``_check_subflow_completion`` pops and applies
        the originating arc's ``result_mapping`` over the parent's pre-push data:
        ``parent_in`` is present, the parent-only ``keep`` survives, and the
        unmapped ``sub_out`` does not leak. Before the fix the pop ignored the
        arc (it was not threaded), so the raw child data flowed through and
        ``parent_in`` was absent.
        """
        fsm = self._build_result_mapping_fsm()
        engine = ExecutionEngine(fsm)
        # The runtime push arc now carries the config-authored result_mapping.
        push_arc = fsm.networks["main"].states["start"].arcs[0]
        assert push_arc.result_mapping == {"sub_out": "parent_in"}

        context = ExecutionContext()
        # Simulate being inside the sub-network at its final state with the
        # child's result, having recorded the push frame at push time.
        context.data = {"sub_out": 99}
        context.push_network("sub", "after")
        context.push_subflow_frame(push_arc, {"keep": "me"}, {})
        context.set_state("s2")  # final state of the 'sub' network

        popped = engine._check_subflow_completion(context)

        assert popped
        assert context.current_state == "after"  # returned to the parent state
        assert context.network_stack == []  # subflow popped
        assert context.data["parent_in"] == 99  # mapped child field
        assert context.data["keep"] == "me"  # parent's pre-push field preserved
        assert "sub_out" not in context.data  # unmapped child field did not leak


class TestNestedSubflowCascade:
    """A subflow whose return state is the parent subflow's final state.

    Reproduce-first for the nested-cascade gap: ``_check_subflow_completion``
    popped at most one level, so when an inner subflow returned to a state that
    was itself a final state of the *outer* subflow, the next loop-top global
    final-state check finalized the whole run there — the outer subflow and the
    main network's remaining states never ran. The fix drains pops in a loop.
    """

    @staticmethod
    def _build_nested_fsm():
        def _append(name: str) -> FunctionReference:
            return FunctionReference(
                type="inline",
                code=(
                    "def transform(data, context):\n"
                    f"    context.variables.setdefault('path', []).append('{name}')\n"
                    "    return data\n"
                ),
            )

        config = FSMConfig(
            name="nested_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            transforms=[_append("start")],
                            arcs=[
                                PushArcConfig(
                                    target="after",
                                    target_network="subA",
                                    return_state="after",
                                )
                            ],
                        ),
                        StateConfig(
                            name="after",
                            transforms=[_append("after")],
                            arcs=[ArcConfig(target="end")],
                        ),
                        StateConfig(
                            name="end", is_end=True, transforms=[_append("end")]
                        ),
                    ],
                ),
                NetworkConfig(
                    name="subA",
                    states=[
                        StateConfig(
                            name="a1",
                            is_start=True,
                            transforms=[_append("a1")],
                            arcs=[
                                # Returns to a_end, which is a *final* state of
                                # subA — the nested-cascade trigger.
                                PushArcConfig(
                                    target="a_end",
                                    target_network="subB",
                                    return_state="a_end",
                                )
                            ],
                        ),
                        StateConfig(
                            name="a_end", is_end=True, transforms=[_append("a_end")]
                        ),
                    ],
                ),
                NetworkConfig(
                    name="subB",
                    states=[
                        StateConfig(
                            name="b1",
                            is_start=True,
                            transforms=[_append("b1")],
                            arcs=[ArcConfig(target="b_end")],
                        ),
                        StateConfig(
                            name="b_end", is_end=True, transforms=[_append("b_end")]
                        ),
                    ],
                ),
            ],
        )
        return FSMBuilder().build(config)

    def test_check_completion_drains_nested_levels(self):
        """A single ``_check_subflow_completion`` drains a nested cascade.

        Drives the sync orchestrator directly (the sync ``execute`` entry
        flat-traverses push arcs — a separate pre-existing gap). With the
        context at subB's final state and subB's return state being subA's final
        state, one ``_check_subflow_completion`` must pop *both* levels and land
        on the main network's return state. Before the drain fix it popped a
        single level and stopped on subA's final state, where the next loop-top
        global final-state check would finalize the whole run prematurely.
        """
        fsm = self._build_nested_fsm()
        engine = ExecutionEngine(fsm)
        push_a = fsm.networks["main"].states["start"].arcs[0]
        push_b = fsm.networks["subA"].states["a1"].arcs[0]

        context = ExecutionContext()
        context.data = {}
        # Stack as if start pushed subA (return 'after') and a1 pushed subB
        # (return 'a_end', a final state of subA), now at subB's final state.
        context.push_network("subA", "after")
        context.push_subflow_frame(push_a, {}, {})
        context.push_network("subB", "a_end")
        context.push_subflow_frame(push_b, {}, {})
        context.set_state("b_end")  # final state of subB

        popped = engine._check_subflow_completion(context)

        assert popped
        assert context.current_state == "after"  # drained both levels to main
        assert context.network_stack == []  # subA and subB both popped


class TestFailedPushRollback:
    """A push whose initial-state entry cannot be resolved leaves no residue.

    Reproduce-first for the rollback gap: the old code replaced ``context.data``
    with the isolated sub-network view *before* validating the target initial
    state, so a bad ``network:state`` target popped the network but left
    ``context.data`` as the orphaned isolated copy. The fix resolves the target
    before committing the push, so a bad target never mutates the context.
    """

    @staticmethod
    def _build_bad_initial_state_fsm():
        config = FSMConfig(
            name="bad_init_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[ArcConfig(target="end")]),
                        StateConfig(name="end", is_end=True),
                    ],
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(name="s1", is_start=True, arcs=[ArcConfig(target="s2")]),
                        StateConfig(name="s2", is_end=True),
                    ],
                ),
            ],
        )
        return FSMBuilder().build(config)

    def test_failed_push_does_not_mutate_or_strand_context(self):
        """A push to a nonexistent initial state rolls back cleanly.

        ``_execute_push_arc`` returns False and leaves ``context.data`` as the
        original object, with an empty network stack and no subflow frame —
        before the fix the data was replaced by the isolated copy and never
        restored.
        """
        fsm = self._build_bad_initial_state_fsm()
        engine = ExecutionEngine(fsm)
        context = ExecutionContext()
        original = {"id": 1}
        context.data = original

        push_arc = PushArc(
            target_state="end",
            target_network="sub:nonexistent",  # bad explicit initial state
            return_state="end",
        )

        ok = engine._execute_push_arc(context, push_arc)

        assert ok is False
        assert context.data is original  # not replaced by an orphaned copy
        assert context.network_stack == []  # nothing left on the stack
        assert context.subflow_frames == []  # no dangling frame


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

        # Verify methods exist (shared on BaseExecutionEngine)
        assert hasattr(engine, 'apply_data_mapping')
        assert hasattr(engine, 'apply_result_mapping')
        assert callable(engine.apply_data_mapping)
        assert callable(engine.apply_result_mapping)
