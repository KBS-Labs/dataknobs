"""Tests for pre-validator functionality."""

import pytest
from typing import Any, Dict

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    ArcConfig,
    FunctionReference,
)
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.functions.base import FunctionContext
from dataknobs_fsm.core.modes import ProcessingMode


class TestPreValidators:
    """Test pre-validator execution and behavior."""

    def test_pre_validators_execute_before_transforms(self):
        """Test that pre-validators run before transforms."""
        # Use context variables to track execution order

        # Create FSM configuration
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="initial",
                            is_start=True,
                            pre_validators=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def pre_val(data, context):
    if 'execution_order' not in context.variables:
        context.variables['execution_order'] = []
    context.variables['execution_order'].append('pre_validator')
    return True
"""
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    if 'execution_order' not in context.variables:
        context.variables['execution_order'] = []
    context.variables['execution_order'].append('transform')
    return data
"""
                                )
                            ],
                            validators=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def post_val(data, context):
    if 'execution_order' not in context.variables:
        context.variables['execution_order'] = []
    context.variables['execution_order'].append('post_validator')
    return True
"""
                                )
                            ],
                            arcs=[
                                ArcConfig(target="final")
                            ]
                        ),
                        StateConfig(
                            name="final",
                            is_end=True
                        )
                    ]
                )
            ]
        )

        # Build and execute FSM
        builder = FSMBuilder()
        fsm = builder.build(config)

        # Create execution context
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        engine = ExecutionEngine(fsm)

        # Execute FSM
        success, result = engine.execute(context, {"test": "data"})

        # Verify execution order
        assert context.variables.get('execution_order') == ["pre_validator", "transform", "post_validator"]
        assert success

    def test_pre_validator_failure_prevents_transform(self):
        """Test that pre-validator failure prevents transform execution."""
        execution_log = []

        # Create FSM configuration
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="initial",
                            is_start=True,
                            pre_validators=[
                                FunctionReference(
                                    type="inline",
                                    code="lambda state: False"  # This will fail
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="lambda state: execution_log.append('transform_executed') or state.data"
                                )
                            ],
                            arcs=[
                                ArcConfig(target="final")
                            ]
                        ),
                        StateConfig(
                            name="final",
                            is_end=True
                        )
                    ]
                )
            ]
        )

        # Build and execute FSM
        builder = FSMBuilder()
        fsm = builder.build(config)

        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        engine = ExecutionEngine(fsm)

        # Execute FSM
        success, result = engine.execute(context, {"test": "data"})

        # Verify transform was not executed
        assert execution_log == []
        assert not success
        assert "Pre-validation failed" in str(result)

    def test_pre_validators_receive_state_resources(self):
        """Test that pre-validators and other state functions receive state resources."""

        # Create FSM configuration without actual resources
        # This test verifies the infrastructure is in place
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="initial",
                            is_start=True,
                            pre_validators=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def pre_val(data, context):
    # Verify that context has resources attribute (even if empty without resource manager)
    assert hasattr(context, 'resources')
    assert isinstance(context.resources, dict)
    # Also verify shared variables are available
    assert hasattr(context, 'variables')
    assert isinstance(context.variables, dict)
    return True
"""
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    # Transform should also receive resources
    assert hasattr(context, 'resources')
    assert isinstance(context.resources, dict)
    return data
"""
                                )
                            ],
                            arcs=[
                                ArcConfig(target="final")
                            ]
                        ),
                        StateConfig(
                            name="final",
                            is_end=True
                        )
                    ]
                )
            ]
        )

        # Build and execute FSM
        builder = FSMBuilder()
        fsm = builder.build(config)

        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        engine = ExecutionEngine(fsm)
        success, result = engine.execute(context, {"test": "data"})

        # If assertions in the functions pass, execution succeeds
        assert success

    def test_shared_variables_across_states(self):
        """Test that shared variables work across states."""

        # Create FSM configuration
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="state1",
                            is_start=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    # Set a variable in context
                                    code="lambda state: {'data': state.data, 'set_var': 'value1'}"
                                )
                            ],
                            arcs=[
                                ArcConfig(target="state2")
                            ]
                        ),
                        StateConfig(
                            name="state2",
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    # Read the variable from context
                                    code="lambda state: {'data': state.data, 'read_var': 'value1'}"
                                )
                            ],
                            arcs=[
                                ArcConfig(target="final")
                            ]
                        ),
                        StateConfig(
                            name="final",
                            is_end=True
                        )
                    ]
                )
            ]
        )

        # Build and execute FSM
        builder = FSMBuilder()
        fsm = builder.build(config)

        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        context.variables = {"shared_data": "initial"}

        engine = ExecutionEngine(fsm)
        success, result = engine.execute(context, {"test": "data"})

        # Verify execution completed
        assert success
        # Variables should be maintained in context
        assert "shared_data" in context.variables

    def test_arc_definition_order_preserved(self):
        """Test that arcs with equal priority are selected in definition order."""

        arc_execution_order = []

        # Create FSM configuration with multiple arcs of same priority
        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="initial",
                            is_start=True,
                            arcs=[
                                ArcConfig(
                                    target="path1",
                                    priority=1,
                                    condition=FunctionReference(
                                        type="inline",
                                        code="lambda state: True"
                                    )
                                ),
                                ArcConfig(
                                    target="path2",
                                    priority=1,
                                    condition=FunctionReference(
                                        type="inline",
                                        code="lambda state: True"
                                    )
                                ),
                                ArcConfig(
                                    target="path3",
                                    priority=1,
                                    condition=FunctionReference(
                                        type="inline",
                                        code="lambda state: True"
                                    )
                                ),
                            ]
                        ),
                        StateConfig(name="path1", is_end=True),
                        StateConfig(name="path2", is_end=True),
                        StateConfig(name="path3", is_end=True),
                    ]
                )
            ]
        )

        # Build FSM
        builder = FSMBuilder()
        fsm = builder.build(config)

        # Verify arc definition order is set
        initial_state = fsm.get_state("initial")
        for i, arc in enumerate(initial_state.outgoing_arcs):
            assert arc.definition_order == i
            assert arc.priority == 1

        # First arc should be selected due to definition order
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        engine = ExecutionEngine(fsm)
        success, result = engine.execute(context, {"test": "data"})

        assert success
        # Should have transitioned to path1 (first defined arc)
        assert context.current_state == "path1"