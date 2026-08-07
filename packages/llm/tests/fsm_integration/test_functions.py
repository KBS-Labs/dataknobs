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


class TestSchemaValidationErrorShape:
    """Which fields failed, in the field for it — not pydantic's whole dump.

    ``str(pydantic_error)`` is a multi-line blob carrying each field's
    ``input_value`` and a versioned docs URL. Relaying it made the message
    unreadable and left ``validation_errors`` — the parameter this class has
    for precisely this list — empty.

    Note this is ``dataknobs_fsm.functions.base.ValidationError``, which
    derives from a plain ``Exception``, *not* from ``DataknobsError``. So the
    ``dataknobs-bots`` API layer never mapped it and this was never an HTTP
    disclosure; it is a message-quality fix.
    """

    def test_the_pydantic_rendering_does_not_reach_the_message(self):
        from dataknobs_fsm.functions.base import ValidationError

        validator = ResponseValidator(
            format="json",
            schema={"account": (str, ...), "balance": (int, ...)},
        )

        with pytest.raises(ValidationError) as excinfo:
            validator.validate(
                {"llm_response": '{"account": 42, "balance": "not-a-number"}'}
            )

        message = str(excinfo.value)
        assert "not-a-number" not in message
        assert "errors.pydantic.dev" not in message
        # The actionable half survives, and lands where callers look for it.
        assert "2 field(s)" in message
        assert excinfo.value.validation_errors == ["account", "balance"]

    def test_the_rendering_survives_on_the_cause(self):
        """Moved to where it belongs, not deleted."""
        from dataknobs_fsm.functions.base import ValidationError

        validator = ResponseValidator(format="json", schema={"account": (str, ...)})

        with pytest.raises(ValidationError) as excinfo:
            validator.validate({"llm_response": '{"account": 42}'})

        assert excinfo.value.__cause__ is not None
        assert "input_value" in str(excinfo.value.__cause__)


# TODO: Add more comprehensive tests for functions
# The original tests from FSM package should be migrated here
