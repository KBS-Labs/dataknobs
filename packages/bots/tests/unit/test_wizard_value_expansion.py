"""Tests for value expansion integration in SchemaGroundingFilter.

Verifies that the grounding filter expands partial string extractions
to the full compound phrase from the user's message, using
``MergeDecision.transform()`` to deliver the expanded value.

Expansion is **opt-in** via ``x-extraction.expand_from_message: true``
on string fields where compound phrases are expected (e.g., tone,
style).  This avoids false positives on identity fields (names, IDs)
where adjacent words often belong to different fields.
"""

from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard_grounding import (
    CompositeMergeFilter,
    MergeDecision,
    SchemaGroundingFilter,
)
from dataknobs_bots.testing import BotTestHarness

# Schema property with expansion enabled — used by tests that exercise
# the expansion path.
_EXPANDABLE: dict[str, Any] = {
    "type": "string",
    "x-extraction": {"expand_from_message": True},
}


# ---------------------------------------------------------------------------
# Unit tests — SchemaGroundingFilter expansion
# ---------------------------------------------------------------------------


class TestSchemaGroundingFilterExpansion:
    """Filter returns transform(expanded) for opted-in string fields."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter(overlap_threshold=0.5)

    def _filter(
        self,
        field: str,
        value: Any,
        message: str,
        schema_property: dict[str, Any] | None = None,
    ) -> MergeDecision:
        return self.f.filter(
            field, value, None, message,
            schema_property or _EXPANDABLE, {},
        )

    def test_partial_value_expanded(self) -> None:
        """'formal' → transform('formal and academic')."""
        decision = self._filter(
            "tone", "formal",
            "Set the tone to formal and academic",
        )
        assert decision.action == "transform"
        assert decision.value == "formal and academic"

    def test_complete_value_accepted(self) -> None:
        """Full value already present → accept (no transform)."""
        decision = self._filter(
            "tone", "formal and academic",
            "Set the tone to formal and academic",
        )
        assert decision.action == "accept"

    def test_value_at_sentence_end_accepted(self) -> None:
        """No expansion possible → accept as-is."""
        decision = self._filter(
            "tone", "formal",
            "The tone should be formal.",
        )
        assert decision.action == "accept"

    def test_or_conjunction_expanded(self) -> None:
        decision = self._filter(
            "tone", "formal",
            "Set the tone to formal or academic",
        )
        assert decision.action == "transform"
        assert decision.value == "formal or academic"

    def test_expansion_reason_set(self) -> None:
        decision = self._filter(
            "tone", "formal",
            "Set the tone to formal and academic",
        )
        assert decision.reason == "expanded partial extraction"


# ---------------------------------------------------------------------------
# Skip cases — expansion should NOT be attempted
# ---------------------------------------------------------------------------


class TestExpansionSkipCases:
    """Expansion is gated by opt-in, type, and enum."""

    def setup_method(self) -> None:
        self.f = SchemaGroundingFilter(overlap_threshold=0.5)

    def _filter(
        self,
        field: str,
        value: Any,
        message: str,
        schema_property: dict[str, Any],
    ) -> MergeDecision:
        return self.f.filter(
            field, value, None, message, schema_property, {},
        )

    def test_default_off_no_expansion(self) -> None:
        """Without expand_from_message, grounded values are accepted as-is.

        This is the motivating case for opt-in: fields like ``name``
        should not expand ``"Alice"`` to ``"Alice and math"`` when the
        user says ``"Alice and math"`` and ``"math"`` belongs to a
        different field.
        """
        decision = self._filter(
            "name", "Alice",
            "Alice and math",
            {"type": "string"},
        )
        assert decision.action == "accept"
        assert decision.reason == "grounded"

    def test_enum_field_not_expanded(self) -> None:
        """Enum fields have constrained values — skip expansion."""
        decision = self._filter(
            "tone", "formal",
            "Set the tone to formal and academic",
            {
                "type": "string",
                "enum": ["formal", "casual", "academic"],
                "x-extraction": {"expand_from_message": True},
            },
        )
        # Should accept (grounded) but NOT transform
        assert decision.action == "accept"

    def test_non_string_type_not_expanded(self) -> None:
        """Integer fields should not be expanded."""
        decision = self._filter(
            "count", "5",
            "Set count to 5 and enable logging",
            {"type": "integer", "x-extraction": {"expand_from_message": True}},
        )
        assert decision.action != "transform"

    def test_explicit_opt_out(self) -> None:
        """x-extraction.expand_from_message: false disables expansion."""
        decision = self._filter(
            "tone", "formal",
            "Set the tone to formal and academic",
            {
                "type": "string",
                "x-extraction": {"expand_from_message": False},
            },
        )
        assert decision.action == "accept"
        assert decision.reason == "grounded"

    def test_non_string_value_not_expanded(self) -> None:
        """Non-string values (e.g. int) skip expansion."""
        decision = self._filter(
            "count", 5,
            "Set count to 5 and 10",
            {"type": "string", "x-extraction": {"expand_from_message": True}},
        )
        # Integer value — _try_expand returns None, falls through
        assert decision.action != "transform"


# ---------------------------------------------------------------------------
# Composite filter chain
# ---------------------------------------------------------------------------


class TestCompositeMergeFilterWithExpansion:
    """Expansion transform propagates through composite chain."""

    def test_expansion_transform_propagates(self) -> None:
        """Grounding filter expands, second filter sees expanded value."""
        seen_values: list[Any] = []

        class TrackingFilter:
            def filter(
                self, field, new_value, existing_value,
                user_message, schema_property, wizard_data,
            ):
                seen_values.append(new_value)
                return MergeDecision.accept(reason="tracking")

        composite = CompositeMergeFilter([
            SchemaGroundingFilter(overlap_threshold=0.5),
            TrackingFilter(),
        ])

        decision = composite.filter(
            "tone", "formal", None,
            "Set the tone to formal and academic",
            _EXPANDABLE, {},
        )

        assert decision.action == "transform"
        assert decision.value == "formal and academic"
        # The tracking filter received the expanded value
        assert seen_values == ["formal and academic"]


# ---------------------------------------------------------------------------
# End-to-end with BotTestHarness
# ---------------------------------------------------------------------------


EXPANSION_WIZARD_CONFIG: dict[str, Any] = {
    "name": "expansion-test",
    "version": "1.0",
    "settings": {
        "extraction_grounding": True,
    },
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me about your preferences.",
            "schema": {
                "type": "object",
                "properties": {
                    "tone": {
                        "type": "string",
                        "description": "Writing tone",
                        "x-extraction": {"expand_from_message": True},
                    },
                    "style": {
                        "type": "string",
                        "description": "Writing style",
                    },
                },
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": "data.get('tone') and data.get('style')",
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All set!",
        },
    ],
}


class TestEndToEnd:
    """BotTestHarness with partial extraction → wizard data gets expanded value."""

    @pytest.mark.asyncio
    async def test_partial_extraction_expanded_in_wizard_data(self) -> None:
        """LLM extracts 'formal' but user said 'formal and academic'.

        Only ``tone`` has ``expand_from_message: true``; ``style`` does not.
        """
        async with await BotTestHarness.create(
            wizard_config=EXPANSION_WIZARD_CONFIG,
            main_responses=["Got it!"],
            extraction_results=[
                [{"tone": "formal", "style": "narrative"}],
            ],
        ) as harness:
            await harness.chat(
                "Set the tone to formal and academic, "
                "and the style to narrative"
            )
            # Expansion should have transformed "formal" → "formal and academic"
            assert harness.wizard_data["tone"] == "formal and academic"
            # "style" has no expand_from_message → accepted as-is
            assert harness.wizard_data["style"] == "narrative"

    @pytest.mark.asyncio
    async def test_field_without_opt_in_not_expanded_e2e(self) -> None:
        """Fields without expand_from_message keep their extracted value.

        Prevents false positives where adjacent field values would be
        absorbed — e.g., ``"Alice and math"`` should NOT expand
        ``name: "Alice"`` to ``"Alice and math"``.
        """
        config: dict[str, Any] = {
            "name": "no-expansion-test",
            "version": "1.0",
            "settings": {"extraction_grounding": True},
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Name and topic?",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "topic": {"type": "string"},
                        },
                    },
                    "transitions": [
                        {
                            "target": "done",
                            "condition": "data.get('name') and data.get('topic')",
                        },
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
            ],
        ) as harness:
            await harness.chat("Alice and math")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_data["topic"] == "math"

    @pytest.mark.asyncio
    async def test_enum_field_not_expanded_e2e(self) -> None:
        """Enum fields skip expansion even with expand_from_message."""
        enum_config: dict[str, Any] = {
            "name": "enum-expansion-test",
            "version": "1.0",
            "settings": {"extraction_grounding": True},
            "stages": [
                {
                    "name": "gather",
                    "is_start": True,
                    "prompt": "Pick a tone.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tone": {
                                "type": "string",
                                "enum": ["formal", "casual"],
                                "description": "Tone",
                                "x-extraction": {"expand_from_message": True},
                            },
                        },
                    },
                    "transitions": [
                        {"target": "done", "condition": "data.get('tone')"},
                    ],
                },
                {"name": "done", "is_end": True, "prompt": "Done!"},
            ],
        }
        async with await BotTestHarness.create(
            wizard_config=enum_config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"tone": "formal"}],
            ],
        ) as harness:
            await harness.chat("Set the tone to formal and academic")
            # Enum field — expansion skipped, original value preserved
            assert harness.wizard_data["tone"] == "formal"
