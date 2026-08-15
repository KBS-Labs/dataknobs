"""Tests for FSM conversation example.

These tests verify the conversation example functionality.
Migrated from dataknobs-fsm package test_llm_conversation_example.py.
"""


def test_example_exists():
    """Test that the conversation example can be imported."""
    # The example is a script, not a module, but we can verify it exists
    from pathlib import Path

    example_path = Path(__file__).parent / "../../examples/fsm_conversation.py"
    assert example_path.exists(), f"Example not found at {example_path}"


# Coverage gap, tracked: the assertion above checks that a path exists, so it
# passes when the example is broken and fails only when someone renames it. The
# 475-line `test_llm_conversation_example.py` that ran the example was deleted by
# the FSM -> LLM migration (`eb1b4c2c`) and is recoverable from `eb1b4c2c^`.
# Either execute the example here or drop this file — a path check dressed as a
# test is worse than none, because the directory then looks covered.
