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

import pytest

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder


# ---------------------------------------------------------------------------
# Wizard configs
# ---------------------------------------------------------------------------


def _gather_config() -> dict:
    """Gather stage with 3 required fields: name, topic, level."""
    return (
        WizardConfigBuilder("gather-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your name, topic, and level.",
        )
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .field("level", field_type="string", required=True)
        .transition(
            "done",
            "data.get('name') and data.get('topic') "
            "and data.get('level')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


def _gather_with_defaults_config() -> dict:
    """Gather stage: name, topic required; level has default 'beginner'."""
    return (
        WizardConfigBuilder("gather-defaults-test")
        .stage(
            "gather",
            is_start=True,
            prompt="Tell me your name and topic.",
        )
        .field("name", field_type="string", required=True)
        .field("topic", field_type="string", required=True)
        .field("level", field_type="string", required=True, default="beginner")
        .transition(
            "done",
            "data.get('name') and data.get('topic') "
            "and data.get('level')",
        )
        .stage("done", is_end=True, prompt="All done!")
        .build()
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPartialDataAccumulation:
    """Verify that partial data accumulates across low-confidence turns."""

    @pytest.mark.asyncio
    async def test_partial_data_preserved_on_low_confidence(self) -> None:
        """Bug: extraction data was discarded on low-confidence turns.

        When confidence < 0.8 and not all required fields are present,
        the wizard returned a clarification response without merging
        the extracted data. This test verifies that partial data IS
        merged even when confidence is low.
        """
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["Could you also tell me the level?"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
            ],
        ) as harness:
            result = await harness.chat("I'm Alice, topic is math")
            assert result.response is not None
            assert harness.wizard_data.get("name") == "Alice", (
                "Partial data 'name' must be preserved on low-confidence turn"
            )
            assert harness.wizard_data.get("topic") == "math", (
                "Partial data 'topic' must be preserved on low-confidence turn"
            )
            assert harness.wizard_stage == "gather"

    @pytest.mark.asyncio
    async def test_accumulated_data_enables_advancement(self) -> None:
        """After partial data accumulates, the remaining fields allow
        the wizard to advance."""
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["Could you tell me the level?", "All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
                [{"level": "advanced"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice, topic is math")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"

            await harness.chat("Advanced level")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"
            assert harness.wizard_data.get("level") == "advanced"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_clarification_still_fires_when_fields_missing(self) -> None:
        """Even though data is merged, clarification must still fire
        when confidence is low and required fields are missing."""
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["Could you tell me more?"],
            extraction_results=[
                [{"name": "Alice"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice")
            assert harness.wizard_stage == "gather"
            assert harness.wizard_data.get("name") == "Alice"
            # Clarification was triggered (still on gather, not done)
            assert harness.wizard_state is not None
            assert harness.wizard_state.get("data", {}).get("name") == "Alice"

    @pytest.mark.asyncio
    async def test_clarification_then_satisfaction_advances(self) -> None:
        """After clarification on partial data, providing the remaining
        field allows the wizard to advance (proves attempts reset).

        Behavioral equivalent of the old
        ``test_clarification_attempts_reset_on_satisfaction``: turn 1
        triggers clarification (stays on gather); turn 2 satisfies all
        fields and advances to done.
        """
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["What level?", "All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
                [{"level": "advanced"}],
            ],
        ) as harness:
            # Turn 1: partial → clarification (still on gather)
            await harness.chat("I'm Alice, topic is math")
            assert harness.wizard_stage == "gather"

            # Turn 2: remaining field → satisfied → advances to done
            # (would NOT advance if clarification_attempts weren't reset)
            await harness.chat("Advanced")
            assert harness.wizard_stage == "done"
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"
            assert harness.wizard_data.get("level") == "advanced"

    @pytest.mark.asyncio
    async def test_schema_defaults_applied_before_confidence_check(self) -> None:
        """Schema defaults should be applied before can_satisfy is
        evaluated, so fields with defaults don't block advancement."""
        async with await BotTestHarness.create(
            wizard_config=_gather_with_defaults_config(),
            main_responses=["All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice, topic is math")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"
            assert harness.wizard_data.get("level") == "beginner"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_conflict_detection_with_accumulated_data(self) -> None:
        """When turn 2's extraction conflicts with turn 1's accumulated
        data, latest_wins should apply and the new value should persist."""
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["What level?", "All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
                [{"topic": "science", "level": "advanced"}],
            ],
        ) as harness:
            await harness.chat("Alice, math")
            assert harness.wizard_data.get("topic") == "math"

            await harness.chat("Actually science, advanced")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "science"
            assert harness.wizard_data.get("level") == "advanced"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_high_confidence_extraction_unchanged(self) -> None:
        """High-confidence extractions should still work exactly as before."""
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math", "level": "advanced"}],
            ],
        ) as harness:
            await harness.chat("Alice, math, advanced")
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"
            assert harness.wizard_data.get("level") == "advanced"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_low_confidence_no_new_data_advances_when_state_complete(
        self,
    ) -> None:
        """When accumulated partial data + new low-confidence extraction
        satisfy all required fields, the wizard should advance."""
        async with await BotTestHarness.create(
            wizard_config=_gather_config(),
            main_responses=["What level?", "All done!"],
            extraction_results=[
                [{"name": "Alice", "topic": "math"}],
                [{"level": "advanced"}],
            ],
        ) as harness:
            await harness.chat("I'm Alice, math")
            assert harness.wizard_stage == "gather"

            await harness.chat("advanced")
            assert harness.wizard_stage == "done"
            assert harness.wizard_data.get("name") == "Alice"
            assert harness.wizard_data.get("topic") == "math"
            assert harness.wizard_data.get("level") == "advanced"
