"""Tests for FSM LLM functions.

These tests verify the LLM function library for FSM.
Migrated from dataknobs-fsm package.
"""

import pytest
from dataknobs_llm.fsm_integration import (
    PromptBuilder,
    LLMCaller,
    ResponseValidator,
    FunctionCaller,
    ConversationManager,
    EmbeddingGenerator,
    build_prompt,
    call_llm,
    validate_response,
    call_function,
    manage_conversation,
    generate_embeddings,
)


def test_function_class_imports():
    """Test that function classes can be imported."""
    assert PromptBuilder is not None
    assert LLMCaller is not None
    assert ResponseValidator is not None
    assert FunctionCaller is not None
    assert ConversationManager is not None
    assert EmbeddingGenerator is not None


def test_function_factory_imports():
    """Test that function factories can be imported."""
    assert build_prompt is not None
    assert call_llm is not None
    assert validate_response is not None
    assert call_function is not None
    assert manage_conversation is not None
    assert generate_embeddings is not None


# TODO: Add more comprehensive tests for functions
# The original tests from FSM package should be migrated here
