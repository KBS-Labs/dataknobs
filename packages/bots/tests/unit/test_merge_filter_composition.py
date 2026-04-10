"""Tests for MergeFilter protocol enhancements and clarification grouping.

Covers:
- MergeDecision dataclass construction and semantics
- CompositeMergeFilter chaining (grounding + custom)
- Custom filter receiving wizard_data
- skip_builtin_grounding configuration
- Clarification field grouping and templates
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_grounding import (
    CompositeMergeFilter,
    MergeDecision,
    MergeFilter,
    SchemaGroundingFilter,
)
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_derivations import parse_derivation_rules
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
# MergeDecision dataclass
# ---------------------------------------------------------------------------


class TestMergeDecision:
    """Unit tests for MergeDecision construction."""

    def test_accept(self) -> None:
        d = MergeDecision.accept()
        assert d.action == "accept"
        assert d.value is None
        assert d.reason is None

    def test_accept_with_reason(self) -> None:
        d = MergeDecision.accept(reason="grounded")
        assert d.action == "accept"
        assert d.reason == "grounded"

    def test_reject(self) -> None:
        d = MergeDecision.reject()
        assert d.action == "reject"
        assert d.value is None

    def test_reject_with_reason(self) -> None:
        d = MergeDecision.reject(reason="ungrounded value")
        assert d.action == "reject"
        assert d.reason == "ungrounded value"

    def test_transform(self) -> None:
        d = MergeDecision.transform("UPPER")
        assert d.action == "transform"
        assert d.value == "UPPER"

    def test_transform_with_reason(self) -> None:
        d = MergeDecision.transform("normalized", reason="case fix")
        assert d.action == "transform"
        assert d.value == "normalized"
        assert d.reason == "case fix"

    def test_frozen(self) -> None:
        d = MergeDecision.accept()
        with pytest.raises(AttributeError):
            d.action = "reject"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SchemaGroundingFilter returns MergeDecision
# ---------------------------------------------------------------------------


class TestSchemaGroundingFilterReturnsDecision:
    """Verify SchemaGroundingFilter.filter() returns MergeDecision."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter(overlap_threshold=0.5)

    def test_grounded_returns_accept(self) -> None:
        decision = self.f.filter(
            "subject", "history", None,
            "I want to study history",
            {"type": "string"}, {},
        )
        assert decision.action == "accept"
        assert decision.reason is not None

    def test_ungrounded_overwrite_returns_reject(self) -> None:
        decision = self.f.filter(
            "subject", "math", "history",
            "completely unrelated message",
            {"type": "string"}, {},
        )
        assert decision.action == "reject"
        assert decision.reason is not None

    def test_ungrounded_no_existing_returns_accept(self) -> None:
        decision = self.f.filter(
            "subject", "math", None,
            "completely unrelated message",
            {"type": "string"}, {},
        )
        assert decision.action == "accept"
        assert "no existing" in (decision.reason or "")

    def test_skip_grounding_returns_accept(self) -> None:
        decision = self.f.filter(
            "tone", "formal", "casual",
            "unrelated",
            {"type": "string", "x-extraction": {"grounding": "skip"}},
            {},
        )
        assert decision.action == "accept"
        assert "skip" in (decision.reason or "")


# ---------------------------------------------------------------------------
# CompositeMergeFilter
# ---------------------------------------------------------------------------


class _AcceptFilter:
    """Test filter that always accepts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        self.calls.append((field, new_value))
        return MergeDecision.accept()


class _RejectFilter:
    """Test filter that always rejects."""

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        return MergeDecision.reject(reason="domain constraint")


class _TransformFilter:
    """Test filter that transforms values to uppercase."""

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        if isinstance(new_value, str):
            return MergeDecision.transform(
                new_value.upper(), reason="uppercased",
            )
        return MergeDecision.accept()


class _WizardDataInspector:
    """Test filter that records wizard_data for inspection."""

    def __init__(self) -> None:
        self.received_data: dict[str, Any] | None = None

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        self.received_data = dict(wizard_data)
        return MergeDecision.accept()


class TestCompositeMergeFilter:
    """Tests for CompositeFilter chaining."""

    def test_empty_chain_accepts(self) -> None:
        composite = CompositeMergeFilter([])
        d = composite.filter("f", "v", None, "msg", {}, {})
        assert d.action == "accept"

    def test_single_filter(self) -> None:
        f = _AcceptFilter()
        composite = CompositeMergeFilter([f])
        d = composite.filter("f", "v", None, "msg", {}, {})
        assert d.action == "accept"
        assert len(f.calls) == 1

    def test_reject_short_circuits(self) -> None:
        reject = _RejectFilter()
        accept = _AcceptFilter()
        composite = CompositeMergeFilter([reject, accept])
        d = composite.filter("f", "v", None, "msg", {}, {})
        assert d.action == "reject"
        assert d.reason == "domain constraint"
        # Second filter should NOT have been called
        assert len(accept.calls) == 0

    def test_transform_flows_through(self) -> None:
        transform = _TransformFilter()
        accept = _AcceptFilter()
        composite = CompositeMergeFilter([transform, accept])
        d = composite.filter("f", "hello", None, "msg", {}, {})
        assert d.action == "transform"
        assert d.value == "HELLO"
        # Second filter should see the transformed value
        assert accept.calls == [("f", "HELLO")]

    def test_grounding_then_custom(self) -> None:
        """Grounding rejects ungrounded → custom never called."""
        grounding = SchemaGroundingFilter()
        custom = _AcceptFilter()
        composite = CompositeMergeFilter([grounding, custom])

        # Ungrounded overwrite: grounding rejects, custom not called
        d = composite.filter(
            "subject", "math", "history",
            "completely unrelated",
            {"type": "string"}, {},
        )
        assert d.action == "reject"
        assert len(custom.calls) == 0

    def test_grounding_accepts_then_custom_runs(self) -> None:
        """Grounded value passes through grounding → custom sees it."""
        grounding = SchemaGroundingFilter()
        custom = _AcceptFilter()
        composite = CompositeMergeFilter([grounding, custom])

        d = composite.filter(
            "subject", "math", "history",
            "change the subject to math",
            {"type": "string"}, {},
        )
        assert d.action == "accept"
        assert len(custom.calls) == 1

    def test_grounding_then_custom_transform(self) -> None:
        """Grounded value → custom transforms it."""
        grounding = SchemaGroundingFilter()
        transform = _TransformFilter()
        composite = CompositeMergeFilter([grounding, transform])

        d = composite.filter(
            "subject", "math", "history",
            "change the subject to math",
            {"type": "string"}, {},
        )
        assert d.action == "transform"
        assert d.value == "MATH"

    def test_wizard_data_passed_through(self) -> None:
        """Custom filter receives wizard_data snapshot."""
        inspector = _WizardDataInspector()
        composite = CompositeMergeFilter([inspector])
        wizard_data = {"intent": "quiz", "subject": "history"}

        composite.filter("tone", "formal", None, "msg", {}, wizard_data)
        assert inspector.received_data == wizard_data

    def test_satisfies_merge_filter_protocol(self) -> None:
        """CompositeMergeFilter satisfies MergeFilter protocol."""
        composite = CompositeMergeFilter([])
        assert isinstance(composite, MergeFilter)


# ---------------------------------------------------------------------------
# WizardReasoning composition model
# ---------------------------------------------------------------------------


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
    grounding_overlap_threshold = settings.get(
        "grounding_overlap_threshold", 0.5,
    )
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        extractor=extractor,
        strict_validation=False,
        extraction_scope="current_message",
        extraction_grounding=extraction_grounding,
        grounding_overlap_threshold=grounding_overlap_threshold,
        **kwargs,
    )


BASIC_WIZARD_CONFIG: dict[str, Any] = {
    "name": "test-wizard",
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
                    "domain_name": {
                        "type": "string",
                        "description": "Display name for the bot",
                    },
                },
                "required": [
                    "intent", "subject", "domain_id", "domain_name",
                ],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('intent') and data.get('subject') "
                        "and data.get('domain_id') "
                        "and data.get('domain_name')"
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


class TestCompositionInit:
    """Test WizardReasoning filter composition via __init__."""

    def test_grounding_only(self) -> None:
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
        )
        assert isinstance(reasoning._extraction._merge_filter, SchemaGroundingFilter)

    def test_custom_only_grounding_disabled(self) -> None:
        config = {
            **BASIC_WIZARD_CONFIG,
            "settings": {"extraction_grounding": False},
        }
        custom = _AcceptFilter()
        reasoning = _build_reasoning(
            config, ConfigurableExtractor(results=[]),
            merge_filter=custom,
        )
        assert reasoning._extraction._merge_filter is custom

    def test_grounding_plus_custom_creates_composite(self) -> None:
        custom = _AcceptFilter()
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            merge_filter=custom,
        )
        assert isinstance(reasoning._extraction._merge_filter, CompositeMergeFilter)

    def test_skip_builtin_grounding_with_custom(self) -> None:
        custom = _AcceptFilter()
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            merge_filter=custom,
            skip_builtin_grounding=True,
        )
        assert reasoning._extraction._merge_filter is custom

    def test_skip_builtin_grounding_without_custom(self) -> None:
        """skip_builtin_grounding with no custom filter → no filter."""
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            skip_builtin_grounding=True,
        )
        assert reasoning._extraction._merge_filter is None

    def test_no_grounding_no_custom(self) -> None:
        config = {
            **BASIC_WIZARD_CONFIG,
            "settings": {"extraction_grounding": False},
        }
        reasoning = _build_reasoning(
            config, ConfigurableExtractor(results=[]),
        )
        assert reasoning._extraction._merge_filter is None


# ---------------------------------------------------------------------------
# Clarification grouping
# ---------------------------------------------------------------------------


class TestBuildClarificationGroups:
    """Tests for _build_clarification_groups helper."""

    def _make_reasoning(
        self,
        groups: list[dict[str, Any]] | None = None,
        exclude_derivable: bool = False,
        derivations: list[dict[str, Any]] | None = None,
    ) -> WizardReasoning:
        field_derivations = None
        if derivations:
            field_derivations = parse_derivation_rules(derivations)
        return _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            clarification_groups=groups,
            clarification_exclude_derivable=exclude_derivable,
            field_derivations=field_derivations,
        )

    def test_no_groups_configured(self) -> None:
        """No groups → ungrouped fields get individual questions."""
        reasoning = self._make_reasoning()
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(
            {"intent", "subject"}, stage,
        )
        assert len(groups) == 2
        questions = {g["question"] for g in groups}
        assert any("intent" in q for q in questions)
        assert any("subject" in q for q in questions)

    def test_configured_groups(self) -> None:
        """Configured groups map missing fields to questions."""
        reasoning = self._make_reasoning(
            groups=[
                {
                    "fields": ["domain_id", "domain_name"],
                    "question": "What would you like to call your bot?",
                },
            ],
        )
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(
            {"domain_id", "domain_name", "intent"}, stage,
        )
        # domain_id + domain_name grouped; intent individual
        assert len(groups) == 2
        grouped_q = next(
            g for g in groups if len(g["fields"]) == 2
        )
        assert grouped_q["question"] == (
            "What would you like to call your bot?"
        )
        assert set(grouped_q["fields"]) == {"domain_id", "domain_name"}

    def test_partial_group_overlap(self) -> None:
        """Only the missing fields from a group are included."""
        reasoning = self._make_reasoning(
            groups=[
                {
                    "fields": ["domain_id", "domain_name"],
                    "question": "What would you like to call your bot?",
                },
            ],
        )
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        # Only domain_name is missing, not domain_id
        groups = reasoning._build_clarification_groups(
            {"domain_name"}, stage,
        )
        assert len(groups) == 1
        assert groups[0]["fields"] == ["domain_name"]

    def test_exclude_derivable(self) -> None:
        """Derivable fields excluded when exclude_derivable=True."""
        from dataknobs_bots.reasoning.wizard import WizardState

        reasoning = self._make_reasoning(
            exclude_derivable=True,
            derivations=[
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
        )
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        # domain_name is derivable from domain_id
        ws = WizardState(
            current_stage="gather",
            data={"domain_id": "my-bot"},
        )
        groups = reasoning._build_clarification_groups(
            {"domain_name", "intent"}, stage, wizard_state=ws,
        )
        # domain_name should be excluded (derivable from domain_id)
        all_fields = {f for g in groups for f in g["fields"]}
        assert "domain_name" not in all_fields
        assert "intent" in all_fields

    def test_empty_missing_fields(self) -> None:
        """No missing fields → empty list."""
        reasoning = self._make_reasoning()
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(set(), stage)
        assert groups == []

    def test_ungrouped_fields_use_schema_description(self) -> None:
        """Ungrouped fields derive question from schema description."""
        reasoning = self._make_reasoning()
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(
            {"domain_name"}, stage,
        )
        assert len(groups) == 1
        # domain_name has description "Display name for the bot"
        assert "Display name for the bot" in groups[0]["question"]

    def test_ungrouped_fields_fallback_to_field_name(self) -> None:
        """Fields without description use field name as fallback."""
        reasoning = self._make_reasoning()
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(
            {"domain_id"}, stage,
        )
        assert len(groups) == 1
        # domain_id has no description → falls back to "domain id"
        assert "domain id" in groups[0]["question"]

    def test_exclude_derivable_both_source_and_target_missing(self) -> None:
        """When both source and target are missing, target IS excluded.

        domain_name is derivable from domain_id.  Even though domain_id
        is also missing, the clarification will ask for domain_id — once
        provided, derivation fills domain_name automatically.  So
        domain_name is still excluded from clarification.
        """
        from dataknobs_bots.reasoning.wizard import WizardState

        reasoning = self._make_reasoning(
            exclude_derivable=True,
            derivations=[
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
        )
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        # Both source (domain_id) and target (domain_name) are missing
        ws = WizardState(
            current_stage="gather",
            data={},
        )
        groups = reasoning._build_clarification_groups(
            {"domain_id", "domain_name"}, stage, wizard_state=ws,
        )
        all_fields = {f for g in groups for f in g["fields"]}
        # domain_id appears (user must provide it)
        assert "domain_id" in all_fields
        # domain_name excluded — derivable once domain_id is provided
        assert "domain_name" not in all_fields

    def test_exclude_derivable_alone_preserves_issues(self) -> None:
        """exclude_derivable without groups should not replace issues."""
        reasoning = self._make_reasoning(
            exclude_derivable=True,
            derivations=[
                {
                    "source": "domain_id",
                    "target": "domain_name",
                    "transform": "title_case",
                },
            ],
        )
        # No groups configured → _build_clarification_groups returns
        # ungrouped individual questions, but the main method should
        # not even enter the grouping path without configured groups.
        assert reasoning._response._clarification_groups == []
        assert reasoning._response._clarification_exclude_derivable is True


class TestClarificationTemplate:
    """Tests for clarification template rendering."""

    def test_custom_template(self) -> None:
        """Custom Jinja2 template renders field groups."""
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            ConfigurableExtractor(results=[]),
            clarification_groups=[
                {
                    "fields": ["domain_id", "domain_name"],
                    "question": "What should we call the bot?",
                },
            ],
            clarification_template=(
                "Please provide:\n"
                "{% for group in field_groups %}"
                "* {{ group.question }}\n"
                "{% endfor %}"
            ),
        )
        # The template rendering is tested through the helper
        stage = BASIC_WIZARD_CONFIG["stages"][0]
        groups = reasoning._build_clarification_groups(
            {"domain_id", "domain_name"}, stage,
        )
        # Verify groups are built correctly for template consumption
        assert len(groups) == 1
        assert groups[0]["question"] == "What should we call the bot?"


class TestFromConfigClarification:
    """Test clarification config loading via from_config."""

    def test_from_config_loads_clarification_groups(self) -> None:
        wizard_config = {
            **BASIC_WIZARD_CONFIG,
            "settings": {
                **BASIC_WIZARD_CONFIG.get("settings", {}),
                "recovery": {
                    "clarification": {
                        "groups": [
                            {
                                "fields": ["domain_id", "domain_name"],
                                "question": "What's the bot name?",
                            },
                        ],
                        "exclude_derivable": True,
                        "template": "Custom: {{ field_groups }}",
                    },
                },
            },
        }
        config: dict[str, Any] = {"wizard_config": wizard_config}
        reasoning = WizardReasoning.from_config(config)
        assert len(reasoning._response._clarification_groups) == 1
        assert reasoning._response._clarification_exclude_derivable is True
        assert reasoning._response._clarification_template == (
            "Custom: {{ field_groups }}"
        )

    def test_from_config_defaults(self) -> None:
        config: dict[str, Any] = {
            "wizard_config": BASIC_WIZARD_CONFIG,
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning._response._clarification_groups == []
        assert reasoning._response._clarification_exclude_derivable is False
        assert reasoning._response._clarification_template is None


class TestFromConfigSkipBuiltinGrounding:
    """Test skip_builtin_grounding config loading."""

    def test_from_config_skip_builtin_grounding(self) -> None:
        wizard_config = {
            **BASIC_WIZARD_CONFIG,
            "settings": {
                **BASIC_WIZARD_CONFIG.get("settings", {}),
                "skip_builtin_grounding": True,
            },
        }
        config: dict[str, Any] = {"wizard_config": wizard_config}
        reasoning = WizardReasoning.from_config(config)
        # No custom filter + skip_builtin_grounding → no filter at all
        assert reasoning._extraction._merge_filter is None

    def test_from_config_skip_builtin_grounding_default_false(self) -> None:
        config: dict[str, Any] = {
            "wizard_config": BASIC_WIZARD_CONFIG,
        }
        reasoning = WizardReasoning.from_config(config)
        # Default: grounding enabled → grounding filter is present
        assert reasoning._extraction._merge_filter is not None


# ---------------------------------------------------------------------------
# End-to-end: transform action through generate()
# ---------------------------------------------------------------------------


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


class _UpperCaseFilter:
    """Test filter that transforms string values to uppercase."""

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        if isinstance(new_value, str) and new_value:
            return MergeDecision.transform(
                new_value.upper(), reason="uppercased",
            )
        return MergeDecision.accept()


class TestTransformEndToEnd:
    """End-to-end test: transform filter through generate()."""

    @pytest.mark.asyncio
    async def test_transform_filter_stores_transformed_value(self) -> None:
        """A custom filter that transforms values should result in
        the transformed value being stored in wizard_state.data."""
        extractor = ConfigurableExtractor(
            results=[
                SimpleExtractionResult(
                    data={
                        "intent": "quiz",
                        "subject": "history",
                        "domain_id": "my-bot",
                        "domain_name": "My Bot",
                    },
                    confidence=0.9,
                ),
            ],
        )
        custom_filter = _UpperCaseFilter()
        reasoning = _build_reasoning(
            BASIC_WIZARD_CONFIG,
            extractor,
            merge_filter=custom_filter,
            skip_builtin_grounding=True,
        )
        manager, provider = await _create_manager()
        provider.set_responses(["Got it!"])

        await manager.add_message(
            role="user",
            content="I want a history quiz bot called My Bot",
        )
        await reasoning.generate(manager, provider)

        ws = reasoning._get_wizard_state(manager)
        # All string values should be uppercased by the transform filter
        assert ws.data["intent"] == "QUIZ"
        assert ws.data["subject"] == "HISTORY"
        assert ws.data["domain_id"] == "MY-BOT"
        assert ws.data["domain_name"] == "MY BOT"
