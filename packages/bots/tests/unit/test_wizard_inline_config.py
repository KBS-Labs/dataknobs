"""Tests for inline dict wizard config support.

Verifies that WizardReasoning.from_config() and
DynaBotConfigBuilder.set_reasoning_wizard() both accept inline dict
wizard configs (not just file paths).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.reasoning.wizard import WizardReasoning


@pytest.fixture
def conversation_wizard_dict() -> dict[str, Any]:
    """Minimal conversation-start wizard config as a dict."""
    return {
        "name": "inline-test",
        "version": "1.0",
        "settings": {
            "tool_reasoning": "react",
            "max_tool_iterations": 3,
        },
        "stages": [
            {
                "name": "conversation",
                "is_start": True,
                "mode": "conversation",
                "prompt": "Have a natural conversation.",
                "tools": ["knowledge_search"],
                "transitions": [],
            },
        ],
    }


@pytest.fixture
def structured_wizard_dict() -> dict[str, Any]:
    """Structured multi-stage wizard config as a dict."""
    return {
        "name": "structured-inline",
        "version": "1.0",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "prompt": "Hello, what is your name?",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "transitions": [
                    {"target": "done", "condition": "data.get('name')"},
                ],
            },
            {
                "name": "done",
                "is_end": True,
                "prompt": "Goodbye!",
                "transitions": [],
            },
        ],
    }


def _minimal_builder() -> DynaBotConfigBuilder:
    """Create a builder with the minimum required components for build()."""
    return (
        DynaBotConfigBuilder()
        .set_llm("ollama", model="llama3.2")
        .set_conversation_storage("memory")
    )


class TestWizardReasoningInlineConfig:
    """Test WizardReasoning.from_config() with inline dict wizard_config."""

    def test_from_config_with_dict(
        self, conversation_wizard_dict: dict[str, Any]
    ) -> None:
        """from_config() loads wizard when wizard_config is a dict."""
        config = {"wizard_config": conversation_wizard_dict}
        reasoning = WizardReasoning.from_config(config)

        assert reasoning is not None
        # Internal FSM attribute is _fsm
        assert reasoning._fsm is not None

    def test_from_config_with_dict_structured(
        self, structured_wizard_dict: dict[str, Any]
    ) -> None:
        """from_config() loads structured wizard from inline dict."""
        config = {"wizard_config": structured_wizard_dict}
        reasoning = WizardReasoning.from_config(config)

        assert reasoning is not None
        assert reasoning._fsm is not None

    def test_from_config_dict_preserves_settings(
        self, conversation_wizard_dict: dict[str, Any]
    ) -> None:
        """Wizard-level settings survive inline dict loading."""
        config = {"wizard_config": conversation_wizard_dict}
        reasoning = WizardReasoning.from_config(config)

        # The reasoning object should have been created successfully
        assert reasoning._fsm is not None

    def test_from_config_raises_when_missing(self) -> None:
        """from_config() raises ValueError when wizard_config is missing."""
        with pytest.raises(ValueError, match="wizard_config is required"):
            WizardReasoning.from_config({})

    def test_from_config_raises_when_none(self) -> None:
        """from_config() raises ValueError when wizard_config is None."""
        with pytest.raises(ValueError, match="wizard_config is required"):
            WizardReasoning.from_config({"wizard_config": None})

    def test_roundtrip_builder_to_reasoning(self) -> None:
        """WizardConfigBuilder → to_dict() → from_config() roundtrip."""
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        wiz = WizardConfigBuilder.conversation_start(
            name="roundtrip-test",
            prompt="Chat with the student about math.",
            tools=["knowledge_search"],
            tool_reasoning="react",
            max_tool_iterations=3,
        )
        wizard_config = wiz.build()
        wizard_dict = wizard_config.to_dict()

        config = {"wizard_config": wizard_dict}
        reasoning = WizardReasoning.from_config(config)

        assert reasoning is not None
        assert reasoning._fsm is not None


class TestBuilderSetReasoningWizardDict:
    """Test DynaBotConfigBuilder.set_reasoning_wizard() with dict."""

    def test_set_reasoning_wizard_with_dict(
        self, conversation_wizard_dict: dict[str, Any]
    ) -> None:
        """set_reasoning_wizard() accepts a dict and stores it inline."""
        builder = _minimal_builder()
        builder.set_reasoning_wizard(conversation_wizard_dict)

        config = builder.build()
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == conversation_wizard_dict

    def test_set_reasoning_wizard_with_dict_and_kwargs(
        self, conversation_wizard_dict: dict[str, Any]
    ) -> None:
        """set_reasoning_wizard() forwards kwargs with dict config."""
        builder = _minimal_builder()
        builder.set_reasoning_wizard(
            conversation_wizard_dict,
            strict_validation=True,
        )

        config = builder.build()
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == conversation_wizard_dict
        assert config["reasoning"]["strict_validation"] is True

    def test_set_reasoning_wizard_with_str(self) -> None:
        """set_reasoning_wizard() still works with string paths."""
        builder = _minimal_builder()
        builder.set_reasoning_wizard("configs/wizards/test.yaml")

        config = builder.build()
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == "configs/wizards/test.yaml"

    def test_set_reasoning_wizard_with_wizard_config_object(self) -> None:
        """set_reasoning_wizard() still works with WizardConfig objects."""
        from dataknobs_bots.config.wizard_builder import WizardConfigBuilder

        wiz = WizardConfigBuilder.conversation_start(
            name="obj-test",
            prompt="Test prompt.",
        )
        wizard_config = wiz.build()

        builder = _minimal_builder()
        builder.set_reasoning_wizard(wizard_config)

        config = builder.build()
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == "obj-test"


class TestBuilderToReasoningIntegration:
    """Integration: builder with inline dict → WizardReasoning.from_config()."""

    def test_builder_dict_config_loads_in_reasoning(
        self, conversation_wizard_dict: dict[str, Any]
    ) -> None:
        """Config built with inline dict loads successfully in WizardReasoning."""
        builder = _minimal_builder()
        builder.set_reasoning_wizard(conversation_wizard_dict)

        config = builder.build()
        reasoning_config = config["reasoning"]

        reasoning = WizardReasoning.from_config(reasoning_config)
        assert reasoning is not None
        assert reasoning._fsm is not None
