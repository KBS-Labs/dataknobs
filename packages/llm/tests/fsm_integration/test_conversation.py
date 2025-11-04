"""Tests for FSM conversation example.

These tests verify the conversation example functionality.
Migrated from dataknobs-fsm package test_llm_conversation_example.py.
"""

import pytest


def test_example_exists():
    """Test that the conversation example can be imported."""
    # The example is a script, not a module, but we can verify it exists
    import os
    example_path = os.path.join(
        os.path.dirname(__file__),
        "../../examples/fsm_conversation.py"
    )
    assert os.path.exists(example_path), f"Example not found at {example_path}"


# TODO: Add more comprehensive tests for the conversation example
# The original tests from FSM package should be migrated here
