"""Test synchronous debugging functionality for FSMs."""

import pytest
from dataknobs_fsm import (
    AdvancedFSM,
    FSMDebugger,
    StepResult,
    create_advanced_fsm
)


def test_synchronous_context_creation():
    """Test that we can create execution context synchronously."""
    # Create simple FSM config with correct format
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "process"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "process"},
                    {"from": "process", "to": "end"}
                ]
            }
        ]
    }

    # Create FSM
    fsm = create_advanced_fsm(config)

    # Create context synchronously
    context = fsm.create_context({"value": 42})

    assert context is not None
    assert context.data == {"value": 42}
    assert context.current_state == "start"


def test_execute_step_sync():
    """Test synchronous step execution."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "process"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "process",
                        "name": "begin_processing"
                    },
                    {
                        "from": "process",
                        "to": "end",
                        "name": "finish"
                    }
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)
    context = fsm.create_context({"value": 42})

    # Execute first step
    result = fsm.execute_step_sync(context)

    assert isinstance(result, StepResult)
    assert result.success
    assert result.from_state == "start"
    assert result.to_state == "process"
    assert result.transition == "begin_processing"
    assert not result.is_complete

    # Execute second step
    result = fsm.execute_step_sync(context)

    assert result.success
    assert result.from_state == "process"
    assert result.to_state == "end"
    assert result.transition == "finish"
    assert result.is_complete


def test_fsm_debugger():
    """Test FSMDebugger functionality."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "validate"},
                    {"name": "process"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "validate"},
                    {"from": "validate", "to": "process"},
                    {"from": "process", "to": "end"}
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)
    debugger = FSMDebugger(fsm)

    # Start debugging session
    debugger.start({"counter": 0, "valid": True})

    # Step through execution
    result = debugger.step()
    assert result.to_state == "validate"

    # Inspect data
    counter = debugger.inspect("counter")
    assert counter == 0

    # Add watch
    debugger.watch("counter_watch", "counter")

    # Continue stepping
    result = debugger.step()
    assert result.to_state == "process"

    result = debugger.step()
    assert result.to_state == "end"
    assert result.is_complete


def test_breakpoints():
    """Test breakpoint functionality."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "step1"},
                    {"name": "step2"},
                    {"name": "step3"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "step1"},
                    {"from": "step1", "to": "step2"},
                    {"from": "step2", "to": "step3"},
                    {"from": "step3", "to": "end"}
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)

    # Add breakpoint at step2
    fsm.add_breakpoint("step2")

    context = fsm.create_context({"value": 1})

    # Run until breakpoint
    state = fsm.run_until_breakpoint_sync(context)

    assert context.current_state == "step2"
    assert state is not None

    # Continue to completion
    fsm.clear_breakpoints()
    state = fsm.run_until_breakpoint_sync(context)

    assert context.current_state == "end"
    assert context.is_complete()


def test_trace_execution_sync():
    """Test synchronous trace execution."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "process"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "process"},
                    {"from": "process", "to": "end"}
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)

    # Execute with tracing
    trace = fsm.trace_execution_sync({"value": 100})

    assert len(trace) == 2  # Two transitions
    assert trace[0]["from_state"] == "start"
    assert trace[0]["to_state"] == "process"
    assert trace[1]["from_state"] == "process"
    assert trace[1]["to_state"] == "end"


def test_profile_execution_sync():
    """Test synchronous profile execution."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "process"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "process"},
                    {"from": "process", "to": "end"}
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)

    # Execute with profiling
    profile = fsm.profile_execution_sync({"value": 100})

    assert "total_time" in profile
    assert "transitions" in profile
    assert profile["transitions"] == 2
    assert "states_visited" in profile
    assert profile["final_state"] == "end"


def test_context_helper_methods():
    """Test ExecutionContext helper methods."""
    config = {
        "name": "test_fsm",
        "version": "1.0",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "end"}
                ]
            }
        ]
    }

    fsm = create_advanced_fsm(config)
    context = fsm.create_context({"test": "value", "count": 5})

    # Test get_current_state
    assert context.get_current_state() == "start"

    # Test get_data_snapshot
    snapshot = context.get_data_snapshot()
    assert snapshot == {"test": "value", "count": 5}

    # Test is_complete
    assert not context.is_complete()

    # Execute to end
    fsm.execute_step_sync(context)
    assert context.is_complete()

    # Test get_current_state_instance
    instance = context.get_current_state_instance()
    assert instance is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])