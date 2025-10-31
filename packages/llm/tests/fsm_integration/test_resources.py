"""Tests for FSM LLM resources.

These tests verify the LLM resource providers for FSM.
Migrated from dataknobs-fsm package.
"""

import pytest
from dataknobs_llm.fsm_integration import (
    LLMProvider,
    LLMSession,
    LLMResource,
)


def test_resource_imports():
    """Test that resource classes can be imported."""
    assert LLMProvider is not None
    assert LLMSession is not None
    assert LLMResource is not None


def test_llm_provider_enum():
    """Test LLMProvider enum values."""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.ANTHROPIC.value == "anthropic"
    assert LLMProvider.OLLAMA.value == "ollama"
    assert LLMProvider.HUGGINGFACE.value == "huggingface"


# TODO: Add more comprehensive tests for resources
# The original tests from FSM package should be migrated here
