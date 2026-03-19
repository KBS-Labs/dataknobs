"""Tests for wizard null extraction handling and has() condition helper.

Two fixes:

1. Null extraction values (LLM returns ``null`` for a field) must not be stored
   in wizard_state.data.  Some models return null instead of omitting the field;
   both cases should be treated identically (key absent = not yet provided).

2. The ``has(key)`` helper is available in transition condition expressions as
   a shorthand for ``data.get(key) is not None``.  This is preferred for
   boolean/numeric/list fields where falsy values (False, 0, []) are legitimate.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


WIZARD_CONFIG: dict[str, Any] = {
    "name": "null-test",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me your intent and provider.",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "domain_name": {"type": "string"},
                    "llm_provider": {"type": "string"},
                },
                "required": ["intent", "domain_name", "llm_provider"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('intent') and data.get('domain_name') "
                        "and data.get('llm_provider')"
                    ),
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All done!",
        },
    ],
}


WIZARD_WITH_DEFAULTS_CONFIG: dict[str, Any] = {
    "name": "null-defaults-test",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me your intent.",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "level": {"type": "string", "default": "beginner"},
                },
                "required": ["intent", "level"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('intent') and data.get('level')"
                    ),
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All done!",
        },
    ],
}


def _build_wizard(
    config: dict[str, Any],
    extractor: ConfigurableExtractor,
    **kwargs: Any,
) -> WizardReasoning:
    """Build a WizardReasoning with a ConfigurableExtractor."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
        **kwargs,
    )


def _get_wizard_state_data(manager: ConversationManager) -> dict[str, Any]:
    """Extract wizard_state.data from manager metadata."""
    wizard_meta = manager.metadata.get("wizard", {})
    fsm_state = wizard_meta.get("fsm_state", {})
    return fsm_state.get("data", {})


def _get_wizard_stage(manager: ConversationManager) -> str:
    """Extract current wizard stage from manager metadata."""
    wizard_meta = manager.metadata.get("wizard", {})
    fsm_state = wizard_meta.get("fsm_state", {})
    return fsm_state.get("current_stage", "")


# ---------------------------------------------------------------------------
# Tests: null extraction handling
# ---------------------------------------------------------------------------


class TestNullExtractionHandling:
    """Null values from extraction must not be stored in wizard state."""

    @pytest.mark.asyncio
    async def test_null_values_dropped_from_extraction(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Extraction returning null fields should not store None in data."""
        manager, conv_provider = conversation_manager_pair

        # Extraction returns intent + two null fields
        extractor = ConfigurableExtractor([
            SimpleExtractionResult(
                data={"intent": "create", "domain_name": None, "llm_provider": None},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(WIZARD_CONFIG, extractor)

        await manager.add_message(role="user", content="I want to create a bot")
        conv_provider.set_responses(["Please provide domain and provider."])

        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("intent") == "create", (
            "Non-null value must be stored"
        )
        assert "domain_name" not in state_data, (
            "Null extraction must not store key in wizard state"
        )
        assert "llm_provider" not in state_data, (
            "Null extraction must not store key in wizard state"
        )

    @pytest.mark.asyncio
    async def test_null_values_do_not_block_schema_defaults(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Null extraction should allow schema defaults to apply."""
        manager, conv_provider = conversation_manager_pair

        # Extract intent (non-null) + level (null).
        # Schema has default "beginner" for level.
        extractor = ConfigurableExtractor([
            SimpleExtractionResult(
                data={"intent": "learn", "level": None},
                confidence=0.9,
            ),
        ])
        reasoning = _build_wizard(WIZARD_WITH_DEFAULTS_CONFIG, extractor)

        await manager.add_message(role="user", content="I want to learn")
        conv_provider.set_responses(["Great!"])

        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("intent") == "learn"
        assert state_data.get("level") == "beginner", (
            "Schema default must apply when null extraction leaves key absent"
        )

    @pytest.mark.asyncio
    async def test_null_values_do_not_overwrite_existing(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Null extraction for an existing key must not clear it."""
        manager, conv_provider = conversation_manager_pair

        # Turn 1: extract intent + domain_name
        extractor = ConfigurableExtractor([
            SimpleExtractionResult(
                data={"intent": "create", "domain_name": "test.com"},
                confidence=0.5,
            ),
            SimpleExtractionResult(
                data={"intent": None, "domain_name": None, "llm_provider": "ollama"},
                confidence=0.9,
            ),
        ])
        reasoning = _build_wizard(WIZARD_CONFIG, extractor)

        # Turn 1
        await manager.add_message(role="user", content="create bot for test.com")
        conv_provider.set_responses(["Which LLM provider?"])
        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("intent") == "create"
        assert state_data.get("domain_name") == "test.com"

        # Turn 2: null for intent and domain_name should not overwrite
        await manager.add_message(role="user", content="use ollama")
        conv_provider.set_responses(["All set!"])
        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("intent") == "create", (
            "Null extraction must not overwrite existing value"
        )
        assert state_data.get("domain_name") == "test.com", (
            "Null extraction must not overwrite existing value"
        )
        assert state_data.get("llm_provider") == "ollama"

    @pytest.mark.asyncio
    async def test_mixed_null_and_values_merged_correctly(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Only non-null values from extraction are merged."""
        manager, conv_provider = conversation_manager_pair

        extractor = ConfigurableExtractor([
            SimpleExtractionResult(
                data={
                    "intent": "create",
                    "domain_name": None,
                    "llm_provider": "ollama",
                },
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(WIZARD_CONFIG, extractor)

        await manager.add_message(role="user", content="create with ollama")
        conv_provider.set_responses(["What domain?"])
        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("intent") == "create"
        assert "domain_name" not in state_data
        assert state_data.get("llm_provider") == "ollama"


# ---------------------------------------------------------------------------
# Tests: has() condition helper
# ---------------------------------------------------------------------------


class TestHasHelper:
    """The has() helper checks field presence (not truthiness)."""

    @pytest.fixture
    def reasoning(self) -> WizardReasoning:
        """Create a minimal WizardReasoning instance."""
        config = {
            "name": "test",
            "stages": [{"name": "start", "is_start": True, "is_end": True}],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        return WizardReasoning(wizard_fsm=fsm)

    def test_has_absent_key(self, reasoning: WizardReasoning) -> None:
        """Absent key returns False."""
        assert not reasoning._evaluate_condition("has('x')", {})

    def test_has_none_value(self, reasoning: WizardReasoning) -> None:
        """None value returns False (same as absent)."""
        assert not reasoning._evaluate_condition("has('x')", {"x": None})

    def test_has_false_value(self, reasoning: WizardReasoning) -> None:
        """False is a legitimate value -- has() returns True."""
        assert reasoning._evaluate_condition("has('x')", {"x": False})

    def test_has_zero_value(self, reasoning: WizardReasoning) -> None:
        """0 is a legitimate value -- has() returns True."""
        assert reasoning._evaluate_condition("has('x')", {"x": 0})

    def test_has_empty_list(self, reasoning: WizardReasoning) -> None:
        """[] is a legitimate value -- has() returns True."""
        assert reasoning._evaluate_condition("has('x')", {"x": []})

    def test_has_empty_string(self, reasoning: WizardReasoning) -> None:
        """Empty string is present -- has() returns True."""
        assert reasoning._evaluate_condition("has('x')", {"x": ""})

    def test_has_with_value(self, reasoning: WizardReasoning) -> None:
        """Normal value returns True."""
        assert reasoning._evaluate_condition("has('x')", {"x": "hello"})

    def test_has_in_compound_condition(self, reasoning: WizardReasoning) -> None:
        """has() works in compound conditions with falsy values."""
        assert reasoning._evaluate_condition(
            "has('a') and has('b')",
            {"a": False, "b": 0},
        )
        assert not reasoning._evaluate_condition(
            "has('a') and has('b')",
            {"a": False},
        )

    def test_has_combined_with_data_get(
        self, reasoning: WizardReasoning
    ) -> None:
        """has() can be combined with data.get() in the same condition."""
        assert reasoning._evaluate_condition(
            "has('flag') and data.get('name')",
            {"flag": False, "name": "Alice"},
        )
        assert not reasoning._evaluate_condition(
            "has('flag') and data.get('name')",
            {"flag": False, "name": ""},
        )
