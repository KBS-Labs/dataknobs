"""Tests for wizard partial data accumulation across turns.

Bug: When extraction confidence was below threshold (< 0.8) and not all required
fields could be satisfied, the wizard returned a clarification response WITHOUT
merging the extracted data into wizard_state.data. This prevented multi-turn
data gathering — wizard_state.data stayed {} forever because partial data was
discarded on every low-confidence turn.

Fix: Restructured extraction result handling so normalize + merge + defaults
happen unconditionally before the confidence check. The confidence check now
only gates the clarification response, not the merge.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.llm.providers.echo import EchoProvider


# ---------------------------------------------------------------------------
# Test extractor that returns a sequence of results
# ---------------------------------------------------------------------------


@dataclass
class SimpleExtractionResult:
    """Simple extraction result for testing."""

    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8 and not self.errors


class SequenceExtractor:
    """Extractor that returns a sequence of configured results.

    Each call to extract() returns the next result in the sequence.
    After the sequence is exhausted, returns the last result repeatedly.
    """

    def __init__(self, results: list[SimpleExtractionResult]) -> None:
        self.results = results
        self.call_index = 0
        self.extract_calls: list[dict[str, Any]] = []

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> SimpleExtractionResult:
        """Return next result in sequence and record the call."""
        self.extract_calls.append({
            "text": text,
            "schema": schema,
            "context": context,
            "model": model,
        })
        idx = min(self.call_index, len(self.results) - 1)
        self.call_index += 1
        return self.results[idx]


# ---------------------------------------------------------------------------
# Wizard config: gather stage with 3 required fields
# ---------------------------------------------------------------------------


GATHER_WIZARD_CONFIG: dict[str, Any] = {
    "name": "gather-test",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me your name, topic, and level.",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "topic": {"type": "string"},
                    "level": {"type": "string"},
                },
                "required": ["name", "topic", "level"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('name') and data.get('topic') "
                        "and data.get('level')"
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


GATHER_WITH_DEFAULTS_CONFIG: dict[str, Any] = {
    "name": "gather-defaults-test",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me your name and topic.",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "topic": {"type": "string"},
                    "level": {"type": "string", "default": "beginner"},
                },
                "required": ["name", "topic", "level"],
            },
            "transitions": [
                {
                    "target": "done",
                    "condition": (
                        "data.get('name') and data.get('topic') "
                        "and data.get('level')"
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
    extractor: SequenceExtractor,
    **kwargs: Any,
) -> WizardReasoning:
    """Build a WizardReasoning with a SequenceExtractor."""
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


def _get_clarification_attempts(manager: ConversationManager) -> int:
    """Extract clarification_attempts from manager metadata."""
    wizard_meta = manager.metadata.get("wizard", {})
    fsm_state = wizard_meta.get("fsm_state", {})
    return fsm_state.get("clarification_attempts", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPartialDataAccumulation:
    """Verify that partial data accumulates across low-confidence turns."""

    @pytest.mark.asyncio
    async def test_partial_data_preserved_on_low_confidence(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Bug: extraction data was discarded on low-confidence turns.

        When confidence < 0.8 and not all required fields are present,
        the wizard returned a clarification response without merging
        the extracted data. This test verifies that partial data IS
        merged even when confidence is low.
        """
        manager, conv_provider = conversation_manager_pair

        # Turn 1: extract 2/3 fields with low confidence
        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        await manager.add_message(role="user", content="I'm Alice, topic is math")
        conv_provider.set_responses(["Could you also tell me the level?"])

        response = await reasoning.generate(manager, llm=None)
        assert response is not None

        # The key assertion: partial data must be preserved in state
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice", (
            "Partial data 'name' must be preserved on low-confidence turn"
        )
        assert state_data.get("topic") == "math", (
            "Partial data 'topic' must be preserved on low-confidence turn"
        )
        # Still on gather stage (not enough data to transition)
        assert _get_wizard_stage(manager) == "gather"

    @pytest.mark.asyncio
    async def test_accumulated_data_enables_advancement(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After partial data accumulates, a subsequent turn providing
        the remaining fields should allow the wizard to advance.
        """
        manager, conv_provider = conversation_manager_pair

        # Turn 1: 2/3 fields, low confidence → clarification
        # Turn 2: 3rd field, low confidence → now all satisfied → advance
        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
            SimpleExtractionResult(
                data={"level": "advanced"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        # Turn 1
        await manager.add_message(role="user", content="I'm Alice, topic is math")
        conv_provider.set_responses(["Could you tell me the level?"])
        await reasoning.generate(manager, llm=None)

        # Verify partial data preserved
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "math"

        # Turn 2
        await manager.add_message(role="assistant", content="Could you tell me the level?")
        await manager.add_message(role="user", content="Advanced level")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)

        # All 3 fields present, wizard should have advanced to "done"
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "math"
        assert state_data.get("level") == "advanced"
        assert _get_wizard_stage(manager) == "done"

    @pytest.mark.asyncio
    async def test_clarification_still_fires_when_fields_missing(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Even though data is merged, clarification must still fire
        when confidence is low and required fields are missing.
        """
        manager, conv_provider = conversation_manager_pair

        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        await manager.add_message(role="user", content="I'm Alice")
        conv_provider.set_responses(["Could you tell me more?"])

        response = await reasoning.generate(manager, llm=None)

        # Response should be a clarification (not advancement)
        assert response is not None
        assert _get_wizard_stage(manager) == "gather"
        assert _get_clarification_attempts(manager) == 1

        # But data should still be preserved
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"

    @pytest.mark.asyncio
    async def test_clarification_attempts_reset_on_satisfaction(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """After partial data accumulates and all required fields are
        satisfied, clarification_attempts should reset to 0.
        """
        manager, conv_provider = conversation_manager_pair

        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
            SimpleExtractionResult(
                data={"level": "advanced"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        # Turn 1: partial → clarification (attempts = 1)
        await manager.add_message(role="user", content="I'm Alice, topic is math")
        conv_provider.set_responses(["What level?"])
        await reasoning.generate(manager, llm=None)
        assert _get_clarification_attempts(manager) == 1

        # Turn 2: remaining field → satisfied → attempts reset
        await manager.add_message(role="assistant", content="What level?")
        await manager.add_message(role="user", content="Advanced")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)
        assert _get_clarification_attempts(manager) == 0

    @pytest.mark.asyncio
    async def test_schema_defaults_applied_before_confidence_check(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """Schema defaults should be applied before can_satisfy is
        evaluated, so fields with defaults don't block advancement.
        """
        manager, conv_provider = conversation_manager_pair

        # Extract 2/3 required fields; 3rd has a schema default ("beginner")
        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WITH_DEFAULTS_CONFIG, extractor)

        await manager.add_message(role="user", content="I'm Alice, topic is math")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)

        # level has default "beginner" — should be applied, satisfying all
        # required fields and allowing advancement despite low confidence
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "math"
        assert state_data.get("level") == "beginner"
        assert _get_wizard_stage(manager) == "done"

    @pytest.mark.asyncio
    async def test_conflict_detection_with_accumulated_data(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When turn 2's extraction conflicts with turn 1's accumulated
        data, latest_wins should apply and the new value should persist.
        """
        manager, conv_provider = conversation_manager_pair

        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
            SimpleExtractionResult(
                data={"topic": "science", "level": "advanced"},
                confidence=0.5,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        # Turn 1: partial data
        await manager.add_message(role="user", content="Alice, math")
        conv_provider.set_responses(["What level?"])
        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("topic") == "math"

        # Turn 2: correction (topic changes) + remaining field
        await manager.add_message(role="assistant", content="What level?")
        await manager.add_message(role="user", content="Actually science, advanced")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)

        # topic should be overridden to "science" (latest_wins)
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "science"
        assert state_data.get("level") == "advanced"
        assert _get_wizard_stage(manager) == "done"

    @pytest.mark.asyncio
    async def test_high_confidence_extraction_unchanged(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """High-confidence extractions should still work exactly as before
        — merge and proceed without hitting the clarification path.
        """
        manager, conv_provider = conversation_manager_pair

        extractor = SequenceExtractor([
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math", "level": "advanced"},
                confidence=0.95,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        await manager.add_message(role="user", content="Alice, math, advanced")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)

        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "math"
        assert state_data.get("level") == "advanced"
        assert _get_wizard_stage(manager) == "done"
        assert _get_clarification_attempts(manager) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_no_new_data_advances_when_state_complete(
        self,
        conversation_manager_pair: tuple[ConversationManager, EchoProvider],
    ) -> None:
        """When accumulated state already satisfies all required fields
        and a subsequent turn extracts nothing new with low confidence,
        the wizard should still advance (not ask for clarification).

        This tests the can_satisfy=True branch under is_confident=False:
        state was populated by prior turns, new extraction is empty,
        but can_satisfy checks wizard_state.data which already has
        everything needed.
        """
        manager, conv_provider = conversation_manager_pair

        extractor = SequenceExtractor([
            # Turn 1: 2/3 fields, low confidence → clarification
            SimpleExtractionResult(
                data={"name": "Alice", "topic": "math"},
                confidence=0.5,
            ),
            # Turn 2: remaining field, low confidence → accumulated
            #         state now has 3/3 → can_satisfy=True → advance
            SimpleExtractionResult(
                data={"level": "advanced"},
                confidence=0.3,
            ),
        ])
        reasoning = _build_wizard(GATHER_WIZARD_CONFIG, extractor)

        # Turn 1: partial data, clarification
        await manager.add_message(role="user", content="I'm Alice, math")
        conv_provider.set_responses(["What level?"])
        await reasoning.generate(manager, llm=None)
        assert _get_wizard_stage(manager) == "gather"
        assert _get_clarification_attempts(manager) == 1

        # Turn 2: remaining field with very low confidence — but
        # accumulated state (name, topic from turn 1 + level from turn 2)
        # satisfies all required fields → should advance, not clarify
        await manager.add_message(role="assistant", content="What level?")
        await manager.add_message(role="user", content="advanced")
        conv_provider.set_responses(["All done!"])
        await reasoning.generate(manager, llm=None)

        assert _get_wizard_stage(manager) == "done"
        assert _get_clarification_attempts(manager) == 0
        state_data = _get_wizard_state_data(manager)
        assert state_data.get("name") == "Alice"
        assert state_data.get("topic") == "math"
        assert state_data.get("level") == "advanced"
