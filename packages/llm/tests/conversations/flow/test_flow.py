"""Tests for ConversationFlow and FlowState."""

import pytest
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    keyword_condition,
    always,
)


def test_flow_state_creation():
    """Test creating a FlowState."""
    state = FlowState(
        prompt_name="test_prompt",
        transitions={"next": "other_state"},
        transition_conditions={"next": always()}
    )

    assert state.prompt_name == "test_prompt"
    assert state.transitions == {"next": "other_state"}
    assert "next" in state.transition_conditions


def test_flow_state_validation():
    """Test FlowState validation."""
    # Should raise error if transition has no condition
    with pytest.raises(ValueError, match="no corresponding condition"):
        FlowState(
            prompt_name="test",
            transitions={"next": "other"},
            transition_conditions={}  # Missing condition!
        )


def test_conversation_flow_creation():
    """Test creating a ConversationFlow."""
    flow = ConversationFlow(
        name="test_flow",
        initial_state="start",
        states={
            "start": FlowState(
                prompt_name="greeting",
                transitions={"continue": "middle"},
                transition_conditions={"continue": always()}
            ),
            "middle": FlowState(
                prompt_name="question",
                transitions={"done": "end"},
                transition_conditions={"done": always()}
            ),
            "end": FlowState(
                prompt_name="goodbye",
                transitions={},
                transition_conditions={}
            )
        }
    )

    assert flow.name == "test_flow"
    assert flow.initial_state == "start"
    assert len(flow.states) == 3


def test_conversation_flow_validation():
    """Test ConversationFlow validation."""
    # Missing initial state
    with pytest.raises(ValueError, match="initial_state.*not found"):
        ConversationFlow(
            name="test",
            initial_state="nonexistent",
            states={
                "start": FlowState(
                    prompt_name="test",
                    transitions={},
                    transition_conditions={}
                )
            }
        )

    # Invalid transition target
    with pytest.raises(ValueError, match="transitions to unknown state"):
        ConversationFlow(
            name="test",
            initial_state="start",
            states={
                "start": FlowState(
                    prompt_name="test",
                    transitions={"next": "nonexistent"},
                    transition_conditions={"next": always()}
                )
            }
        )


def test_get_state():
    """Test getting a state by name."""
    flow = ConversationFlow(
        name="test",
        initial_state="start",
        states={
            "start": FlowState(
                prompt_name="test",
                transitions={},
                transition_conditions={}
            )
        }
    )

    state = flow.get_state("start")
    assert state.prompt_name == "test"

    with pytest.raises(KeyError):
        flow.get_state("nonexistent")


def test_get_reachable_states():
    """Test getting reachable states."""
    flow = ConversationFlow(
        name="test",
        initial_state="start",
        states={
            "start": FlowState(
                prompt_name="test",
                transitions={"a": "state_a", "b": "state_b"},
                transition_conditions={"a": always(), "b": always()}
            ),
            "state_a": FlowState(
                prompt_name="a",
                transitions={},
                transition_conditions={}
            ),
            "state_b": FlowState(
                prompt_name="b",
                transitions={},
                transition_conditions={}
            )
        }
    )

    reachable = flow.get_reachable_states("start")
    assert set(reachable) == {"state_a", "state_b"}


def test_validate_flow():
    """Test flow validation."""
    # Flow with unreachable state
    flow = ConversationFlow(
        name="test",
        initial_state="start",
        states={
            "start": FlowState(
                prompt_name="test",
                transitions={},
                transition_conditions={}
            ),
            "unreachable": FlowState(
                prompt_name="test2",
                transitions={},
                transition_conditions={}
            )
        }
    )

    warnings = flow.validate_flow()
    assert len(warnings) > 0
    assert any("unreachable" in w.lower() for w in warnings)


def test_flow_with_max_loops():
    """Test flow state with loop detection."""
    state = FlowState(
        prompt_name="test",
        transitions={"loop": "test"},
        transition_conditions={"loop": always()},
        max_loops=3
    )

    assert state.max_loops == 3
