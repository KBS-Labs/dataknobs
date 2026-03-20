"""Tests for wizard null extraction handling and has() condition helper.

Two fixes:

1. Null extraction values (LLM returns ``null`` for a field) must not be stored
   in wizard_state.data.  Some models return null instead of omitting the field;
   both cases should be treated identically (key absent = not yet provided).

2. The ``has(key)`` helper is available in transition condition expressions as
   a shorthand for ``data.get(key) is not None``.  This is preferred for
   boolean/numeric/list fields where falsy values (False, 0, []) are legitimate.
"""

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Shared wizard configs
# ---------------------------------------------------------------------------


def _null_test_config() -> dict:
    """Gather stage: intent, domain_name, llm_provider (all required)."""
    return (
        WizardConfigBuilder("null-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your intent and provider.",
        )
        .field("intent", field_type="string", required=True)
        .field("domain_name", field_type="string", required=True)
        .field("llm_provider", field_type="string", required=True)
        .transition(
            "done",
            "data.get('intent') and data.get('domain_name') "
            "and data.get('llm_provider')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


def _defaults_config() -> dict:
    """Gather stage: intent + level (with default "beginner")."""
    return (
        WizardConfigBuilder("null-defaults-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your intent.",
        )
        .field("intent", field_type="string", required=True)
        .field("level", field_type="string", required=True, default="beginner")
        .transition(
            "done",
            "data.get('intent') and data.get('level')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


def _boolean_config() -> dict:
    """Gather stage: intent + kb_enabled (boolean, uses has())."""
    return (
        WizardConfigBuilder("null-boolean-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your intent and whether KB is enabled.",
        )
        .field("intent", field_type="string", required=True)
        .field("kb_enabled", field_type="boolean", required=True)
        .transition(
            "done",
            "data.get('intent') and has('kb_enabled')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


# ---------------------------------------------------------------------------
# Tests: null extraction handling
# ---------------------------------------------------------------------------


class TestNullExtractionHandling:
    """Null values from extraction must not be stored in wizard state."""

    @pytest.mark.asyncio
    async def test_null_values_dropped_from_extraction(self) -> None:
        """Extraction returning null fields should not store None in data."""
        async with await BotTestHarness.create(
            wizard_config=_null_test_config(),
            main_responses=["Please provide domain and provider."],
            extraction_results=[
                [{"intent": "create", "domain_name": None, "llm_provider": None}],
            ],
        ) as harness:
            await harness.chat("I want to create a bot")
            assert harness.wizard_data.get("intent") == "create", (
                "Non-null value must be stored"
            )
            assert "domain_name" not in harness.wizard_data, (
                "Null extraction must not store key in wizard state"
            )
            assert "llm_provider" not in harness.wizard_data, (
                "Null extraction must not store key in wizard state"
            )

    @pytest.mark.asyncio
    async def test_null_values_do_not_block_schema_defaults(self) -> None:
        """Null extraction should allow schema defaults to apply."""
        async with await BotTestHarness.create(
            wizard_config=_defaults_config(),
            main_responses=["Great!"],
            extraction_results=[
                [{"intent": "learn", "level": None}],
            ],
        ) as harness:
            await harness.chat("I want to learn")
            assert harness.wizard_data.get("intent") == "learn"
            assert harness.wizard_data.get("level") == "beginner", (
                "Schema default must apply when null extraction leaves key absent"
            )

    @pytest.mark.asyncio
    async def test_null_values_do_not_overwrite_existing(self) -> None:
        """Null extraction for an existing key must not clear it."""
        async with await BotTestHarness.create(
            wizard_config=_null_test_config(),
            main_responses=["Which LLM provider?", "All set!"],
            extraction_results=[
                [{"intent": "create", "domain_name": "test.com"}],
                [{"intent": None, "domain_name": None, "llm_provider": "ollama"}],
            ],
        ) as harness:
            # Turn 1: establish intent + domain_name
            await harness.chat("create bot for test.com")
            assert harness.wizard_data.get("intent") == "create"
            assert harness.wizard_data.get("domain_name") == "test.com"

            # Turn 2: null for intent and domain_name should not overwrite
            await harness.chat("use ollama")
            assert harness.wizard_data.get("intent") == "create", (
                "Null extraction must not overwrite existing value"
            )
            assert harness.wizard_data.get("domain_name") == "test.com", (
                "Null extraction must not overwrite existing value"
            )
            assert harness.wizard_data.get("llm_provider") == "ollama"

    @pytest.mark.asyncio
    async def test_mixed_null_and_values_merged_correctly(self) -> None:
        """Only non-null values from extraction are merged."""
        async with await BotTestHarness.create(
            wizard_config=_null_test_config(),
            main_responses=["What domain?"],
            extraction_results=[
                [{"intent": "create", "domain_name": None, "llm_provider": "ollama"}],
            ],
        ) as harness:
            await harness.chat("create with ollama")
            assert harness.wizard_data.get("intent") == "create"
            assert "domain_name" not in harness.wizard_data
            assert harness.wizard_data.get("llm_provider") == "ollama"

    @pytest.mark.asyncio
    async def test_null_boolean_field_not_coerced_to_false(self) -> None:
        """Null for a boolean-typed field must not be coerced to False.

        _normalize_extracted_data coerces string->bool for boolean fields.
        A null value must be dropped (not stored), not coerced to False.
        """
        async with await BotTestHarness.create(
            wizard_config=_boolean_config(),
            main_responses=["Should KB be enabled?"],
            extraction_results=[
                [{"intent": "create", "kb_enabled": None}],
            ],
        ) as harness:
            await harness.chat("I want to create a bot")
            assert harness.wizard_data.get("intent") == "create"
            assert "kb_enabled" not in harness.wizard_data, (
                "Null boolean extraction must not be coerced to False"
            )


# ---------------------------------------------------------------------------
# Tests: has() condition helper
# ---------------------------------------------------------------------------


class TestHasHelper:
    """The has() helper checks field presence (not truthiness).

    These tests exercise _evaluate_condition() directly — this is a
    legitimate unit test of internal condition evaluation logic.
    """

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
