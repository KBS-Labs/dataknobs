"""Tests for schema-driven extraction grounding and selective merge.

The grounding check prevents extraction models from overwriting existing
wizard state data with ungrounded values --- values that don't appear in
the user's message.  Only grounded values (or first-time extractions with
no existing data to protect) are merged.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_grounding import (
    MergeFilter,
    SchemaGroundingFilter,
    _has_negation,
    significant_words,
)
from dataknobs_bots.reasoning.wizard_utils import word_in_text
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm import LLMConfig
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.conversations.storage import DataknobsConversationStorage
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.prompts import ConfigPromptLibrary
from dataknobs_llm.prompts.builders import AsyncPromptBuilder
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult

# ---------------------------------------------------------------------------
# Unit tests for SchemaGroundingFilter
# ---------------------------------------------------------------------------


class TestSchemaGroundingFilterStrings:
    """String grounding checks."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter(overlap_threshold=0.5)

    def test_string_grounded_when_value_in_message(self) -> None:
        assert self.f.should_merge(
            "subject", "history", None,
            "I want to study history",
            {"type": "string"},
        )

    def test_string_not_grounded_no_existing_merges(self) -> None:
        """No existing value -> merge regardless (benefit of the doubt)."""
        assert self.f.should_merge(
            "subject", "history", None,
            "make it a tutor",
            {"type": "string"},
        )

    def test_string_not_grounded_blocks_overwrite(self) -> None:
        """Existing data protected from ungrounded overwrite."""
        assert not self.f.should_merge(
            "subject", "", "history",
            "make it a tutor instead",
            {"type": "string"},
        )

    def test_string_grounded_allows_overwrite(self) -> None:
        assert self.f.should_merge(
            "subject", "math", "history",
            "actually change the subject to math",
            {"type": "string"},
        )

    def test_string_word_overlap_at_threshold(self) -> None:
        # "History Quizzer" -> words {history, quizzer}
        # Message has "history" -> 50% overlap -> meets default threshold
        assert self.f.should_merge(
            "domain_name", "History Quizzer", None,
            "I want history content",
            {"type": "string"},
        )

    def test_string_word_overlap_below_threshold(self) -> None:
        # "Advanced World History" -> 3 significant words
        # Message has 0 overlap -> blocked
        assert not self.f.should_merge(
            "domain_name", "Advanced World History", "My Bot",
            "make it a tutor",
            {"type": "string"},
        )

    def test_string_all_stopwords_trusts_extraction(self) -> None:
        """A value composed entirely of stopwords is trusted."""
        assert self.f.should_merge(
            "prefix", "the", None,
            "completely unrelated",
            {"type": "string"},
        )


class TestSchemaGroundingFilterEnums:
    """Enum grounding checks."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_enum_grounded_when_value_in_message(self) -> None:
        assert self.f.should_merge(
            "intent", "tutor", "quiz",
            "make it a tutor instead",
            {"type": "string", "enum": ["tutor", "quiz", "custom"]},
        )

    def test_enum_not_grounded_blocks_overwrite(self) -> None:
        assert not self.f.should_merge(
            "intent", "custom", "quiz",
            "keep the same settings",
            {"type": "string", "enum": ["tutor", "quiz", "custom"]},
        )

    def test_enum_case_insensitive(self) -> None:
        assert self.f.should_merge(
            "intent", "Tutor", "quiz",
            "I want a TUTOR bot",
            {"type": "string", "enum": ["tutor", "quiz"]},
        )


class TestSchemaGroundingFilterBooleans:
    """Boolean grounding checks."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_boolean_grounded_when_field_keyword_in_message(self) -> None:
        assert self.f.should_merge(
            "kb_enabled", False, None,
            "no knowledge base please",
            {"type": "boolean", "description": "Whether knowledge base is enabled"},
        )

    def test_boolean_not_grounded_blocks_overwrite(self) -> None:
        assert not self.f.should_merge(
            "kb_enabled", False, True,
            "make it a tutor instead",
            {"type": "boolean", "description": "Whether knowledge base is enabled"},
        )

    def test_boolean_grounded_by_field_name(self) -> None:
        """Field name itself provides keywords when no description."""
        assert self.f.should_merge(
            "hints_enabled", True, False,
            "enable hints please",
            {"type": "boolean"},
        )


class TestSchemaGroundingFilterNumbers:
    """Number grounding checks."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_number_grounded_when_literal_in_message(self) -> None:
        assert self.f.should_merge(
            "max_hints", 2, None,
            "give me 2 hints max",
            {"type": "integer"},
        )

    def test_number_not_grounded_blocks_overwrite(self) -> None:
        assert not self.f.should_merge(
            "max_hints", 5, 2,
            "keep the same settings",
            {"type": "integer"},
        )

    def test_float_grounded(self) -> None:
        assert self.f.should_merge(
            "threshold", 0.8, None,
            "set threshold to 0.8",
            {"type": "number"},
        )


class TestSchemaGroundingFilterArrays:
    """Array grounding checks."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_array_grounded_when_element_in_message(self) -> None:
        assert self.f.should_merge(
            "tools", ["search", "calculator"], None,
            "I want search and calculator tools",
            {"type": "array"},
        )

    def test_empty_array_not_grounded_blocks_overwrite(self) -> None:
        assert not self.f.should_merge(
            "tools", [], ["search"],
            "make it a tutor",
            {"type": "array"},
        )

    def test_array_partial_grounding(self) -> None:
        """At least one element present is sufficient."""
        assert self.f.should_merge(
            "tools", ["search", "unknown_tool"], None,
            "enable search",
            {"type": "array"},
        )


class TestSchemaGroundingFilterEmptyStrings:
    """Empty string grounding."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_empty_string_grounded_with_negation(self) -> None:
        """Empty string is grounded when user uses negation + field keyword."""
        assert self.f.should_merge(
            "description", "", "A great bot",
            "no description needed",
            {"type": "string", "description": "Brief description of the bot"},
        )

    def test_empty_string_not_grounded_without_negation(self) -> None:
        assert not self.f.should_merge(
            "description", "", "A great bot",
            "make it a tutor",
            {"type": "string", "description": "Brief description of the bot"},
        )

    def test_empty_string_with_empty_allowed(self) -> None:
        """x-extraction.empty_allowed: true allows empty overwrite."""
        assert self.f.should_merge(
            "description", "", "A great bot",
            "make it a tutor",  # No negation keyword, but empty_allowed=true
            {"type": "string", "x-extraction": {"empty_allowed": True}},
        )


class TestSchemaGroundingFilterXExtraction:
    """Per-field x-extraction hint overrides."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def test_grounding_skip_always_merges(self) -> None:
        assert self.f.should_merge(
            "tone", "formal", "casual",
            "completely unrelated message",
            {"type": "string", "x-extraction": {"grounding": "skip"}},
        )

    def test_grounding_exact_requires_literal_match(self) -> None:
        assert not self.f.should_merge(
            "domain_id", "my-bot", "old-bot",
            "I said My Bot",  # "my-bot" not literally in message
            {"type": "string", "x-extraction": {"grounding": "exact"}},
        )
        assert self.f.should_merge(
            "domain_id", "my-bot", "old-bot",
            "set the id to my-bot",
            {"type": "string", "x-extraction": {"grounding": "exact"}},
        )

    def test_grounding_fuzzy_always_trusts(self) -> None:
        assert self.f.should_merge(
            "tone", "professional", "casual",
            "unrelated message",
            {"type": "string", "x-extraction": {"grounding": "fuzzy"}},
        )

    def test_per_field_overlap_threshold(self) -> None:
        """x-extraction.overlap_threshold overrides the global default."""
        # "Advanced World History" -> 3 words, msg has "history" -> 33%
        # Default threshold 0.5 would block, but per-field 0.3 allows it
        assert self.f.should_merge(
            "domain_name", "Advanced World History", "Old Name",
            "I want a history bot",
            {"type": "string", "x-extraction": {"overlap_threshold": 0.3}},
        )


class TestSignificantWords:
    """Test the significant_words helper."""

    def test_filters_stopwords(self) -> None:
        result = significant_words("the quick brown fox is a very fast animal")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_filters_short_words(self) -> None:
        result = significant_words("I am ok to go do it")
        # All <= 2 chars -> empty (or just stopwords)
        assert "am" not in result
        assert "ok" not in result

    def test_empty_string(self) -> None:
        assert significant_words("") == set()


# ---------------------------------------------------------------------------
# Integration tests: merge loop with grounding
# ---------------------------------------------------------------------------

GROUNDING_WIZARD_CONFIG: dict[str, Any] = {
    "name": "grounding-test",
    "version": "1.0",
    "settings": {
        "extraction_grounding": True,
    },
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me about your bot.",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["tutor", "quiz", "custom"],
                    },
                    "subject": {"type": "string"},
                    "domain_id": {"type": "string"},
                    "domain_name": {"type": "string"},
                    "llm_provider": {"type": "string"},
                },
                "required": [
                    "intent", "subject", "domain_id",
                    "domain_name", "llm_provider",
                ],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('intent') and data.get('subject') "
                        "and data.get('domain_id') and data.get('domain_name') "
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


GROUNDING_DISABLED_CONFIG: dict[str, Any] = {
    "name": "grounding-disabled-test",
    "version": "1.0",
    "settings": {
        "extraction_grounding": False,
    },
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me about your bot.",
            "schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["subject", "tone"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": "data.get('subject') and data.get('tone')",
                },
            ],
        },
        {"name": "done", "is_end": True, "prompt": "Done!"},
    ],
}


PER_STAGE_GROUNDING_CONFIG: dict[str, Any] = {
    "name": "per-stage-grounding-test",
    "version": "1.0",
    "settings": {
        "extraction_grounding": True,
    },
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me about your bot.",
            "extraction_grounding": False,  # Per-stage override
            "schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["subject", "tone"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": "data.get('subject') and data.get('tone')",
                },
            ],
        },
        {"name": "done", "is_end": True, "prompt": "Done!"},
    ],
}


async def _create_manager() -> tuple[ConversationManager, EchoProvider]:
    """Create a ConversationManager + EchoProvider pair for testing."""
    config = LLMConfig(
        provider="echo",
        model="echo-test",
        options={"echo_prefix": ""},
    )
    provider = EchoProvider(config)
    library = ConfigPromptLibrary({
        "system": {
            "assistant": {
                "template": "You are a helpful assistant.",
            },
        },
    })
    builder = AsyncPromptBuilder(library=library)
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    manager = await ConversationManager.create(
        llm=provider,
        prompt_builder=builder,
        storage=storage,
    )
    return manager, provider


def _build_reasoning(
    config: dict[str, Any],
    extractor: ConfigurableExtractor,
    **kwargs: Any,
) -> WizardReasoning:
    """Build a WizardReasoning from a config dict with injected extractor."""
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
    settings = config.get("settings", {})
    extraction_grounding = settings.get("extraction_grounding", True)
    grounding_overlap_threshold = settings.get("grounding_overlap_threshold", 0.5)
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
        extraction_grounding=extraction_grounding,
        grounding_overlap_threshold=grounding_overlap_threshold,
        **kwargs,
    )


class TestCorrectionScenario:
    """The primary motivating scenario: multi-turn corrections."""

    @pytest.mark.asyncio
    async def test_correction_preserves_existing_data(self) -> None:
        """Turn 3: 'make it a tutor instead, keep the same name/subject'.

        The extraction model returns empty strings for subject/domain_id,
        but grounding prevents the overwrite.
        """
        # Turn 2 extraction: user provides all fields
        turn2_data = {
            "intent": "quiz",
            "subject": "history",
            "domain_id": "history-quizzer",
            "domain_name": "History Quizzer",
        }
        # Turn 3 extraction: model hallucinated empty fields
        turn3_data = {
            "intent": "tutor",
            "subject": "",
            "domain_id": "",
            "domain_name": "History Quizzer",
        }

        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(data=turn2_data, confidence=0.9),
                SimpleExtractionResult(data=turn3_data, confidence=0.9),
            ],
        )
        reasoning = _build_reasoning(GROUNDING_WIZARD_CONFIG, extractor)
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!", "Updated!"])

        # Turn 2: all fields filled for the first time
        await manager.add_message(role="user", content=
            "I want a history quiz bot called History Quizzer, "
            "ID history-quizzer"
        )
        await reasoning.generate(manager, provider)

        # Verify turn 2 state
        ws = reasoning._get_wizard_state(manager)
        assert ws.data["intent"] == "quiz"
        assert ws.data["subject"] == "history"
        assert ws.data["domain_id"] == "history-quizzer"

        # Turn 3: correction — only intent should change
        await manager.add_message(role="user", content=
            "Actually, make it a tutor instead. "
            "Keep the same name and subject."
        )
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        assert ws.data["intent"] == "tutor", "Intent should be updated"
        assert ws.data["subject"] == "history", "Subject should be preserved"
        assert ws.data["domain_id"] == "history-quizzer", (
            "Domain ID should be preserved"
        )

    @pytest.mark.asyncio
    async def test_first_turn_all_fields_merge(self) -> None:
        """First turn: no existing data, everything merges."""
        first_data = {
            "intent": "quiz",
            "subject": "history",
            "domain_id": "history-quizzer",
            "domain_name": "History Quizzer",
        }
        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(data=first_data, confidence=0.9),
            ],
        )
        reasoning = _build_reasoning(GROUNDING_WIZARD_CONFIG, extractor)
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!"])

        await manager.add_message(role="user", content=
            "I want a history quiz bot"
        )
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        # All fields should merge — no existing data to protect
        assert ws.data["intent"] == "quiz"
        assert ws.data["subject"] == "history"
        assert ws.data["domain_id"] == "history-quizzer"
        assert ws.data["domain_name"] == "History Quizzer"


class TestGroundingConfig:
    """Grounding on/off via configuration."""

    @pytest.mark.asyncio
    async def test_grounding_disabled_allows_overwrite(self) -> None:
        """extraction_grounding: false allows ungrounded overwrites."""
        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(
                    data={"subject": "history", "tone": "casual"},
                    confidence=0.9,
                ),
                SimpleExtractionResult(
                    data={"subject": "", "tone": "formal"},
                    confidence=0.9,
                ),
            ],
        )
        reasoning = _build_reasoning(GROUNDING_DISABLED_CONFIG, extractor)
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!", "Updated!"])

        # Turn 1
        await manager.add_message(role="user", content="history casual")
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        assert ws.data["subject"] == "history"

        # Turn 2: ungrounded overwrite allowed
        await manager.add_message(role="user", content="make it formal")
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        assert ws.data["subject"] == "", "Grounding disabled: overwrite allowed"

    @pytest.mark.asyncio
    async def test_per_stage_grounding_override(self) -> None:
        """Per-stage extraction_grounding: false overrides wizard-level true."""
        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(
                    data={"subject": "history", "tone": "casual"},
                    confidence=0.9,
                ),
                SimpleExtractionResult(
                    data={"subject": "", "tone": "formal"},
                    confidence=0.9,
                ),
            ],
        )
        reasoning = _build_reasoning(PER_STAGE_GROUNDING_CONFIG, extractor)
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!", "Updated!"])

        # Turn 1
        await manager.add_message(role="user", content="history casual")
        await reasoning.generate(manager, provider)

        # Turn 2: stage has grounding disabled
        await manager.add_message(role="user", content="change tone to formal")
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        # Per-stage grounding disabled -> empty string overwrites
        assert ws.data["subject"] == ""


class TestMergeFilterProtocol:
    """Custom MergeFilter replaces built-in grounding."""

    def test_custom_filter_used(self) -> None:
        """A custom MergeFilter that always allows merge."""

        class AlwaysMerge:
            def should_merge(
                self,
                field: str,
                new_value: Any,
                existing_value: Any,
                user_message: str,
                schema_property: dict[str, Any],
            ) -> bool:
                return True

        f = AlwaysMerge()
        assert isinstance(f, MergeFilter)

        # Verify it would allow overwrite that grounding would block
        assert f.should_merge(
            "subject", "", "history", "unrelated", {"type": "string"},
        )

    def test_custom_filter_blocks(self) -> None:
        """A custom MergeFilter that blocks all overwrites."""

        class NeverOverwrite:
            def should_merge(
                self,
                field: str,
                new_value: Any,
                existing_value: Any,
                user_message: str,
                schema_property: dict[str, Any],
            ) -> bool:
                return existing_value is None

        f = NeverOverwrite()
        assert isinstance(f, MergeFilter)
        assert f.should_merge("x", "val", None, "msg", {})
        assert not f.should_merge("x", "new", "old", "msg", {})


class TestGroundingInit:
    """Test WizardReasoning init with grounding params."""

    def test_grounding_enabled_creates_filter(self) -> None:
        """Default: extraction_grounding=True creates SchemaGroundingFilter."""
        reasoning = _build_reasoning(
            GROUNDING_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
        )
        assert reasoning._merge_filter is not None
        assert isinstance(reasoning._merge_filter, SchemaGroundingFilter)

    def test_grounding_disabled_no_filter(self) -> None:
        """extraction_grounding: false -> no filter."""
        reasoning = _build_reasoning(
            GROUNDING_DISABLED_CONFIG,
            ConfigurableExtractor(results=[]),
        )
        assert reasoning._merge_filter is None

    def test_custom_filter_overrides_grounding(self) -> None:
        """Custom merge_filter takes precedence over extraction_grounding."""

        class CustomFilter:
            def should_merge(
                self,
                field: str,
                new_value: Any,
                existing_value: Any,
                user_message: str,
                schema_property: dict[str, Any],
            ) -> bool:
                return True

        custom = CustomFilter()
        reasoning = _build_reasoning(
            GROUNDING_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            merge_filter=custom,
        )
        assert reasoning._merge_filter is custom


class TestFromConfigRoundTrip:
    """Tests that exercise the from_config code path."""

    def test_from_config_with_grounding_enabled(self) -> None:
        """from_config creates SchemaGroundingFilter when enabled."""
        config: dict[str, Any] = {
            "wizard_config": GROUNDING_WIZARD_CONFIG,
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning._merge_filter is not None
        assert isinstance(reasoning._merge_filter, SchemaGroundingFilter)

    def test_from_config_with_grounding_disabled(self) -> None:
        """from_config creates no filter when disabled."""
        config: dict[str, Any] = {
            "wizard_config": GROUNDING_DISABLED_CONFIG,
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning._merge_filter is None

    def test_from_config_invalid_merge_filter_path(self) -> None:
        """Invalid merge_filter dotted path raises ConfigurationError."""
        from dataknobs_common.exceptions import ConfigurationError

        bad_config = {
            **GROUNDING_WIZARD_CONFIG,
            "settings": {
                **GROUNDING_WIZARD_CONFIG.get("settings", {}),
                "merge_filter": "nonexistent.module.BadFilter",
            },
        }
        config: dict[str, Any] = {
            "wizard_config": bad_config,
        }
        with pytest.raises(ConfigurationError, match="Cannot import"):
            WizardReasoning.from_config(config)


class TestPerStageReEnable:
    """Per-stage extraction_grounding: true re-enables when wizard is off."""

    @pytest.mark.asyncio
    async def test_per_stage_reenable_grounding(self) -> None:
        """Per-stage true creates a filter even when wizard-level is false."""
        reenable_config: dict[str, Any] = {
            "name": "reenable-test",
            "version": "1.0",
            "settings": {
                "extraction_grounding": False,  # Wizard-level OFF
            },
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Tell me.",
                    "extraction_grounding": True,  # Per-stage ON
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "tone": {"type": "string"},
                            "extra": {"type": "string"},
                        },
                        "required": ["subject", "tone", "extra"],
                    },
                    "transitions": [
                        {
                            "target": "done",
                            "condition": (
                                "data.get('subject') and data.get('tone') "
                                "and data.get('extra')"
                            ),
                        },
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }

        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(
                    data={"subject": "history", "tone": "casual"},
                    confidence=0.9,
                ),
                SimpleExtractionResult(
                    data={"subject": "", "tone": "formal", "extra": "val"},
                    confidence=0.9,
                ),
            ],
        )
        reasoning = _build_reasoning(reenable_config, extractor)
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!", "Updated!"])

        # Turn 1
        await manager.add_message(role="user", content="history casual")
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        assert ws.data["subject"] == "history"

        # Turn 2: grounding re-enabled per-stage should protect subject
        await manager.add_message(
            role="user", content="change tone to formal",
        )
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        assert ws.data["subject"] == "history", (
            "Per-stage re-enabled grounding should protect existing data"
        )
        assert ws.data["tone"] == "formal"


class TestAdditionalEdgeCases:
    """Additional edge case tests from code review."""

    def test_array_non_empty_ungrounded_blocks_overwrite(self) -> None:
        """Non-empty array not in message should not overwrite existing."""
        f = SchemaGroundingFilter()
        assert not f.should_merge(
            "tools", ["calculator"], ["search"],
            "no tools needed",
            {"type": "array"},
        )

    def test_number_word_boundary_no_false_positive(self) -> None:
        """Number 5 should not match in '15' or '50'."""
        f = SchemaGroundingFilter()
        assert not f.should_merge(
            "count", 5, 3,
            "I want 15 items and 50 results",
            {"type": "integer"},
        )

    def test_number_word_boundary_matches_standalone(self) -> None:
        """Number 5 should match when standalone."""
        f = SchemaGroundingFilter()
        assert f.should_merge(
            "count", 5, 3,
            "I want 5 items",
            {"type": "integer"},
        )


# ---------------------------------------------------------------------------
# P2: Word-boundary matching (replaces substring `in` checks)
# ---------------------------------------------------------------------------


class TestWordBoundaryMatching:
    """All grounding checks use word-boundary matching, not substring."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()

    def testword_in_text_helper_matches_whole_word(self) -> None:
        assert word_in_text("base", "the base is ready")
        assert not word_in_text("base", "use a database")

    def testword_in_text_helper_case_sensitive(self) -> None:
        assert word_in_text("tutor", "find a tutor")
        assert not word_in_text("tutor", "find a Tutor")  # case matters

    def test_boolean_base_not_in_database(self) -> None:
        """'base' from description should not match 'database'."""
        assert not self.f.should_merge(
            "kb_enabled", True, False,
            "use a database for storage",
            {"type": "boolean", "description": "Whether knowledge base is enabled"},
        )

    def test_boolean_log_not_in_catalog(self) -> None:
        """'log' from 'logging' should not match 'catalog'."""
        assert not self.f.should_merge(
            "logging_enabled", True, False,
            "add to catalog",
            {"type": "boolean", "description": "Enable logging"},
        )

    def test_enum_tutor_not_in_tutored(self) -> None:
        """Enum 'tutor' should not match 'tutored'."""
        assert not self.f.should_merge(
            "intent", "tutor", "quiz",
            "I was tutored yesterday",
            {"type": "string", "enum": ["tutor", "quiz"]},
        )

    def test_enum_tutor_matches_standalone(self) -> None:
        """Enum 'tutor' should match when standalone."""
        assert self.f.should_merge(
            "intent", "tutor", "quiz",
            "make it a tutor",
            {"type": "string", "enum": ["tutor", "quiz"]},
        )

    def test_empty_string_name_not_in_rename(self) -> None:
        """'name' from description should not match 'rename'."""
        assert not self.f.should_merge(
            "domain_name", "", "My Bot",
            "rename the project, skip the rest",
            {"type": "string", "description": "The domain name"},
        )

    def test_array_element_word_boundary(self) -> None:
        """Array element 'search' should not match 'researching'."""
        assert not self.f.should_merge(
            "tools", ["search"], ["calculator"],
            "I was researching options",
            {"type": "array"},
        )

    def test_array_element_matches_standalone(self) -> None:
        """Array element 'search' should match when standalone."""
        assert self.f.should_merge(
            "tools", ["search"], ["calculator"],
            "enable the search tool",
            {"type": "array"},
        )


# ---------------------------------------------------------------------------
# P3: Boolean value-direction checking
# ---------------------------------------------------------------------------


class TestBooleanValueDirection:
    """Boolean grounding checks extracted value against negation signals."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()
        self.kb_prop: dict[str, Any] = {
            "type": "boolean",
            "description": "Whether knowledge base is enabled",
        }

    # -- Basic direction checking --

    def test_false_with_negation_is_grounded(self) -> None:
        """False + negation keyword → grounded (user said 'no KB')."""
        assert self.f.should_merge(
            "kb_enabled", False, True,
            "no knowledge base please",
            self.kb_prop,
        )

    def test_true_with_negation_is_not_grounded(self) -> None:
        """True + negation keyword → NOT grounded (hallucinated True)."""
        assert not self.f.should_merge(
            "kb_enabled", True, False,
            "no knowledge base please",
            self.kb_prop,
        )

    def test_true_without_negation_is_grounded(self) -> None:
        """True + no negation → grounded (user affirmed the field)."""
        assert self.f.should_merge(
            "kb_enabled", True, False,
            "enable the knowledge base",
            self.kb_prop,
        )

    def test_false_without_negation_is_not_grounded(self) -> None:
        """False + no negation → NOT grounded (hallucinated False)."""
        assert not self.f.should_merge(
            "kb_enabled", False, True,
            "enable the knowledge base",
            self.kb_prop,
        )

    def test_field_not_mentioned_is_not_grounded(self) -> None:
        """Field keywords absent → not grounded regardless of value."""
        assert not self.f.should_merge(
            "kb_enabled", True, False,
            "make it a tutor instead",
            self.kb_prop,
        )

    # -- check_direction: false disables direction checking --

    def test_check_direction_false_allows_any_value(self) -> None:
        """check_direction: false means field-mention is sufficient."""
        prop: dict[str, Any] = {
            **self.kb_prop,
            "x-extraction": {"check_direction": False},
        }
        # True even though there's negation — direction not checked
        assert self.f.should_merge(
            "kb_enabled", True, False,
            "no knowledge base please",
            prop,
        )

    # -- Custom negation keywords --

    def test_custom_negation_keywords(self) -> None:
        """Custom negation_keywords override the default set."""
        prop: dict[str, Any] = {
            **self.kb_prop,
            "x-extraction": {"negation_keywords": ["nope", "nah"]},
        }
        # "no" is in default set but NOT in custom set → not negation
        assert self.f.should_merge(
            "kb_enabled", True, False,
            "no knowledge base please",
            prop,
        )
        # "nah" IS in custom set → negation detected
        assert self.f.should_merge(
            "kb_enabled", False, True,
            "nah, skip the knowledge base",
            prop,
        )

    # -- Negation proximity --

    def test_negation_proximity_blocks_distant_negation(self) -> None:
        """Proximity check: 'no' far from field keyword should not count."""
        prop: dict[str, Any] = {
            **self.kb_prop,
            "x-extraction": {"negation_proximity": 2},
        }
        # "no" is 6+ words from "knowledge" → not within proximity
        assert not self.f.should_merge(
            "kb_enabled", False, True,
            "no I really do want a knowledge base",
            prop,
        )

    def test_negation_proximity_allows_nearby_negation(self) -> None:
        """Proximity check: 'no' near field keyword should count."""
        prop: dict[str, Any] = {
            **self.kb_prop,
            "x-extraction": {"negation_proximity": 2},
        }
        # "no" is 1 word from "knowledge" → within proximity=2
        assert self.f.should_merge(
            "kb_enabled", False, True,
            "no knowledge base",
            prop,
        )

    def test_negation_proximity_zero_means_anywhere(self) -> None:
        """Proximity 0 (default) means negation anywhere counts."""
        prop: dict[str, Any] = {
            **self.kb_prop,
            "x-extraction": {"negation_proximity": 0},
        }
        # "no" is far from "knowledge" but proximity=0 → still counts
        assert self.f.should_merge(
            "kb_enabled", False, True,
            "no I really do want a knowledge base",
            prop,
        )


# ---------------------------------------------------------------------------
# P4: Empty array clearing via negation keywords
# ---------------------------------------------------------------------------


class TestEmptyArrayClearing:
    """Empty arrays can be grounded via negation keywords."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter()
        self.tools_prop: dict[str, Any] = {
            "type": "array",
            "description": "Available tools",
        }

    def test_empty_array_grounded_with_negation(self) -> None:
        """'no tools' → empty array is grounded."""
        assert self.f.should_merge(
            "tools", [], ["search", "calculator"],
            "no tools needed",
            self.tools_prop,
        )

    def test_empty_array_not_grounded_without_negation(self) -> None:
        """Field keyword present but no negation → not grounded."""
        assert not self.f.should_merge(
            "tools", [], ["search"],
            "I want better tools",
            self.tools_prop,
        )

    def test_empty_array_not_grounded_no_field_keyword(self) -> None:
        """No field keyword at all → not grounded."""
        assert not self.f.should_merge(
            "tools", [], ["search"],
            "make it a tutor",
            self.tools_prop,
        )

    def test_empty_array_grounded_with_field_name_negation(self) -> None:
        """Field name keyword + negation → grounded."""
        assert self.f.should_merge(
            "tools_enabled", [], ["search"],
            "remove all tools please",
            {"type": "array"},
        )

    def test_empty_array_with_empty_allowed(self) -> None:
        """empty_allowed: true bypasses negation check for arrays."""
        prop: dict[str, Any] = {
            **self.tools_prop,
            "x-extraction": {"empty_allowed": True},
        }
        assert self.f.should_merge(
            "tools", [], ["search"],
            "make it a tutor",  # No negation, no field keyword
            prop,
        )

    def test_empty_array_custom_negation_keywords(self) -> None:
        """Custom negation_keywords for array clearing."""
        prop: dict[str, Any] = {
            **self.tools_prop,
            "x-extraction": {"negation_keywords": ["drop", "ditch"]},
        }
        # "no" is NOT in custom set → should not ground
        assert not self.f.should_merge(
            "tools", [], ["search"],
            "no tools needed",
            prop,
        )
        # "drop" IS in custom set → should ground
        assert self.f.should_merge(
            "tools", [], ["search"],
            "drop the tools",
            prop,
        )

    def test_empty_array_with_proximity(self) -> None:
        """Proximity check applies to empty array negation."""
        prop: dict[str, Any] = {
            **self.tools_prop,
            "x-extraction": {"negation_proximity": 2},
        }
        # "no" within 2 words of "tools"
        assert self.f.should_merge(
            "tools", [], ["search"],
            "no tools needed",
            prop,
        )
        # "no" far from "tools"
        assert not self.f.should_merge(
            "tools", [], ["search"],
            "no I actually want some tools",
            prop,
        )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHasNegation:
    """Tests for the _has_negation helper."""

    def test_negation_present_no_proximity(self) -> None:
        assert _has_negation("no thanks", frozenset({"no", "skip"}))

    def test_negation_absent(self) -> None:
        assert not _has_negation("yes please", frozenset({"no", "skip"}))

    def test_proximity_nearby(self) -> None:
        assert _has_negation(
            "no knowledge base",
            frozenset({"no"}),
            field_keywords={"knowledge"},
            proximity=2,
        )

    def test_proximity_too_far(self) -> None:
        assert not _has_negation(
            "no I really do want a knowledge base",
            frozenset({"no"}),
            field_keywords={"knowledge"},
            proximity=2,
        )

    def test_proximity_zero_ignores_distance(self) -> None:
        assert _has_negation(
            "no I really do want a knowledge base",
            frozenset({"no"}),
            field_keywords={"knowledge"},
            proximity=0,
        )

    def test_proximity_with_no_field_keywords_falls_back(self) -> None:
        """field_keywords=None with proximity>0 falls back to presence."""
        assert _has_negation(
            "no thanks for that",
            frozenset({"no"}),
            field_keywords=None,
            proximity=5,
        )
