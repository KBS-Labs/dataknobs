"""Tests for example conversation flows.

This module tests the flows defined in examples/conversation_flow_example.py
to ensure they are correctly structured and function as expected.
"""

import pytest
import sys
from pathlib import Path

# Add examples directory to path so we can import the example
examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from conversation_flow_example import create_support_flow, create_sales_flow


def test_create_support_flow():
    """Test that support flow is created correctly."""
    flow = create_support_flow()

    # Basic structure
    assert flow.name == "customer_support"
    assert flow.initial_state == "greeting"
    assert len(flow.states) == 10

    # Verify all expected states exist
    expected_states = {
        "greeting", "collect_issue", "tech_support", "billing_support",
        "account_support", "clarify_issue", "end_browsing", "end_resolved",
        "end_escalate", "end_unclear"
    }
    assert set(flow.states.keys()) == expected_states

    # Verify initial state
    greeting_state = flow.states["greeting"]
    assert greeting_state.prompt_name == "support_greeting"
    assert len(greeting_state.transitions) == 2
    assert "need_help" in greeting_state.transitions
    assert "just_browsing" in greeting_state.transitions

    # Verify loop detection is configured
    collect_issue_state = flow.states["collect_issue"]
    assert collect_issue_state.max_loops == 2

    clarify_state = flow.states["clarify_issue"]
    assert clarify_state.max_loops == 1


def test_support_flow_terminal_states():
    """Test that terminal states are properly configured."""
    flow = create_support_flow()

    # Terminal states should have no transitions
    terminal_states = ["end_browsing", "end_resolved", "end_escalate", "end_unclear"]

    for state_name in terminal_states:
        state = flow.states[state_name]
        assert len(state.transitions) == 0, f"{state_name} should have no transitions"
        assert len(state.transition_conditions) == 0


def test_support_flow_validation():
    """Test flow validation.

    Terminal states will trigger warnings about no exit transitions,
    but this is expected behavior, not an error.
    """
    flow = create_support_flow()
    warnings = flow.validate_flow()

    # We expect warnings about terminal states (this is normal)
    # Filter out expected warnings about terminal states
    terminal_state_warnings = [
        w for w in warnings
        if "no exit transitions" in w.lower() and any(
            term in w for term in ["end_browsing", "end_resolved", "end_escalate", "end_unclear"]
        )
    ]

    # All warnings should be about terminal states
    assert len(warnings) == len(terminal_state_warnings), \
        "Flow has unexpected warnings beyond terminal state warnings"

    # Should have 4 warnings (one per terminal state)
    assert len(terminal_state_warnings) == 4


def test_support_flow_transitions():
    """Test that all transitions point to valid states."""
    flow = create_support_flow()

    for state_name, state in flow.states.items():
        for target_state in state.transitions.values():
            assert target_state in flow.states, \
                f"State '{state_name}' transitions to unknown state '{target_state}'"


def test_support_flow_reachability():
    """Test that all states except terminal ones are reachable."""
    flow = create_support_flow()

    # Get reachable states
    reachable = set([flow.initial_state])
    to_visit = [flow.initial_state]

    while to_visit:
        current = to_visit.pop()
        for next_state in flow.get_reachable_states(current):
            if next_state not in reachable:
                reachable.add(next_state)
                to_visit.append(next_state)

    # All states should be reachable
    assert set(flow.states.keys()) == reachable, \
        f"Unreachable states: {set(flow.states.keys()) - reachable}"


def test_create_sales_flow():
    """Test that sales flow is created correctly."""
    flow = create_sales_flow()

    # Basic structure
    assert flow.name == "sales_qualification"
    assert flow.initial_state == "introduce"
    assert len(flow.states) == 10

    # Verify all expected states exist
    expected_states = {
        "introduce", "qualify_needs", "present_solution", "handle_objection",
        "pricing_discussion", "end_no_interest", "end_poor_fit",
        "end_follow_up", "end_think", "end_close"
    }
    assert set(flow.states.keys()) == expected_states


def test_sales_flow_terminal_states():
    """Test that sales flow terminal states are properly configured."""
    flow = create_sales_flow()

    terminal_states = [
        "end_no_interest", "end_poor_fit", "end_follow_up",
        "end_think", "end_close"
    ]

    for state_name in terminal_states:
        state = flow.states[state_name]
        assert len(state.transitions) == 0
        assert len(state.transition_conditions) == 0


def test_sales_flow_loop_detection():
    """Test that sales flow has loop detection configured."""
    flow = create_sales_flow()

    handle_objection_state = flow.states["handle_objection"]
    assert handle_objection_state.max_loops == 2


def test_sales_flow_validation():
    """Test sales flow validation."""
    flow = create_sales_flow()
    warnings = flow.validate_flow()

    # Filter out expected warnings about terminal states
    terminal_state_warnings = [
        w for w in warnings
        if "no exit transitions" in w.lower()
    ]

    # All warnings should be about terminal states
    assert len(warnings) == len(terminal_state_warnings)

    # Should have 5 warnings (one per terminal state)
    assert len(terminal_state_warnings) == 5


def test_both_flows_have_max_total_loops():
    """Test that both flows have reasonable total loop limits."""
    support_flow = create_support_flow()
    sales_flow = create_sales_flow()

    assert support_flow.max_total_loops == 15
    assert sales_flow.max_total_loops == 20


def test_condition_coverage():
    """Test that all transitions have corresponding conditions."""
    support_flow = create_support_flow()
    sales_flow = create_sales_flow()

    for flow in [support_flow, sales_flow]:
        for state_name, state in flow.states.items():
            # Every transition must have a condition
            for cond_name in state.transitions.keys():
                assert cond_name in state.transition_conditions, \
                    f"State '{state_name}' has transition '{cond_name}' without condition"


@pytest.mark.asyncio
async def test_example_main_runs():
    """Test that the example's main function runs without errors."""
    from conversation_flow_example import example_usage

    # This should complete without raising exceptions
    await example_usage()


def test_flow_descriptions():
    """Test that flows have proper descriptions."""
    support_flow = create_support_flow()
    sales_flow = create_sales_flow()

    assert support_flow.description is not None
    assert len(support_flow.description) > 0

    assert sales_flow.description is not None
    assert len(sales_flow.description) > 0


def test_prompt_names_are_consistent():
    """Test that prompt names follow a consistent pattern."""
    support_flow = create_support_flow()

    for state_name, state in support_flow.states.items():
        # Prompt names should be non-empty strings
        assert isinstance(state.prompt_name, str)
        assert len(state.prompt_name) > 0

        # Terminal states should have goodbye-related or end-related prompts
        if state_name.startswith("end_"):
            assert any(
                keyword in state.prompt_name
                for keyword in ["goodbye", "escalation", "unclear"]
            )
