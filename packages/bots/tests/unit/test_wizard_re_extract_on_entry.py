"""Tests for re_extract_on_entry stage option (Item 89).

When a wizard transitions from stage A to stage B mid-turn and stage B
has ``re_extract_on_entry: true``, the wizard re-extracts from the
user's original message against stage B's schema.  This enables
single-turn edit-back flows where a message like "change the tone to
formal" both routes (via the source stage) and captures the value (via
the target stage).
"""

import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import WizardReasoning, WizardState
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm import EchoProvider
from dataknobs_llm.testing import ConfigurableExtractor, SimpleExtractionResult


def _edit_back_config(
    *,
    re_extract: bool | None = True,
    auto_advance_filled: bool = True,
    target_has_schema: bool = True,
    target_auto_advance: bool | None = None,
) -> dict[str, Any]:
    """Build a wizard config for edit-back re-extraction tests.

    source (start) → target → done (end)

    ``source`` has a routing field; ``target`` has a value field.
    The user's message at ``source`` may contain data for both schemas.
    """
    builder = (
        WizardConfigBuilder("test-re-extract")
        .settings(auto_advance_filled_stages=auto_advance_filled)
        .stage("source", is_start=True, prompt="Enter routing info.")
        .field("routing_field", field_type="string", required=True)
        .transition("target", "data.get('routing_field')")
        .stage(
            "target",
            prompt="Enter value.",
            re_extract_on_entry=re_extract,
            auto_advance=target_auto_advance,
        )
    )
    if target_has_schema:
        builder = builder.field(
            "value_field", field_type="string", required=True,
        )
    builder = (
        builder
        .transition("done", "data.get('value_field')")
        .stage("done", is_end=True, prompt="Complete!")
    )
    return builder.build()


class TestReExtractOnEntry:
    """Integration tests: re_extract_on_entry through BotTestHarness."""

    @pytest.mark.asyncio
    async def test_re_extract_captures_data_from_transition_message(
        self,
    ) -> None:
        """Stage with re_extract_on_entry extracts from the triggering message.

        The user's single message provides data for both the source stage's
        routing field and the target stage's value field.  Without
        re-extraction, value_field would be lost.
        """
        config = _edit_back_config(re_extract=True)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # Re-extraction at target: extracts value_field
                [{"value_field": "formal"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("routing_field") == "go"
            assert harness.wizard_data.get("value_field") == "formal"
            # Single turn: source → target (re-extract) → done (auto-advance)
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_re_extract_explicit_false_no_re_extraction(self) -> None:
        """Stages with re_extract_on_entry=False do NOT re-extract.

        Explicit False disables re-extraction; the wizard lands on the
        target stage without extracting value_field.
        """
        config = _edit_back_config(re_extract=False)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Enter value."],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # No re-extraction — next extraction is on turn 2
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("routing_field") == "go"
            # value_field was NOT extracted — still at target
            assert harness.wizard_data.get("value_field") is None
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_re_extract_absent_no_re_extraction(self) -> None:
        """Stages with re_extract_on_entry absent (None) do NOT re-extract.

        When re_extract_on_entry is not configured at all, the stage
        behaves the same as explicit False — no re-extraction.
        """
        config = _edit_back_config(re_extract=None)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Enter value."],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # No re-extraction — next extraction is on turn 2
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("routing_field") == "go"
            assert harness.wizard_data.get("value_field") is None
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_re_extract_no_op_without_user_message(self) -> None:
        """Greet (no user message) does not crash with re_extract_on_entry.

        re_extract_on_entry is a no-op when there's no user message to
        re-extract from.
        """
        config = _edit_back_config(re_extract=True)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!"],
        ) as harness:
            result = await harness.greet()
            # Greet should work fine — no crash
            assert result is not None
            assert harness.wizard_stage == "source"

    @pytest.mark.asyncio
    async def test_re_extract_with_no_schema_is_skipped(self) -> None:
        """Target stage with no schema silently skips re-extraction."""
        config = _edit_back_config(
            re_extract=True, target_has_schema=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Enter value."],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # No re-extraction because target has no schema
            ],
        ) as harness:
            await harness.chat("Route to target")
            assert harness.wizard_data.get("routing_field") == "go"
            # Landed on target — no re-extraction, no auto-advance
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_re_extract_empty_extraction_stays_at_target(self) -> None:
        """Re-extraction that finds no data leaves wizard at target stage."""
        config = _edit_back_config(re_extract=True)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Enter value."],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # Re-extraction at target: finds nothing relevant
                [{}],
            ],
        ) as harness:
            await harness.chat("Route to target")
            assert harness.wizard_data.get("routing_field") == "go"
            assert harness.wizard_data.get("value_field") is None
            # Empty re-extraction → stays at target
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_re_extract_second_turn_extracts_normally(
        self,
    ) -> None:
        """After re-extraction, the next turn extracts normally.

        Re-extraction at the target stage captures ``value_field`` but not
        ``extra_field`` — the wizard stays at target.  The second turn
        must extract normally and capture ``extra_field``, advancing to
        done.  This verifies re-extraction doesn't interfere with the
        next turn's extraction lifecycle.
        """
        # source → target (re_extract, but auto-advance won't fire because
        # extra_field is also required).  The second turn at target must
        # extract normally.
        config = (
            WizardConfigBuilder("test-skip-clear")
            .settings(auto_advance_filled_stages=True)
            .stage("source", is_start=True, prompt="Start.")
            .field("routing_field", field_type="string", required=True)
            .transition("target", "data.get('routing_field')")
            .stage("target", prompt="Enter value.",
                   re_extract_on_entry=True)
            .field("value_field", field_type="string", required=True)
            .field("extra_field", field_type="string", required=True)
            .transition(
                "done",
                "data.get('value_field') and data.get('extra_field')",
            )
            .stage("done", is_end=True, prompt="Complete!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Almost there!", "Done!"],
            extraction_results=[
                # Turn 1 at source: extracts routing_field
                [{"routing_field": "go"}],
                # Re-extraction at target: extracts value_field only
                [{"value_field": "formal"}],
                # Turn 2 at target: normal extraction gets extra_field.
                # This only works if skip_extraction was cleared.
                [{"extra_field": "details"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            # Re-extraction captured value_field, but extra_field is
            # still missing → stays at target
            assert harness.wizard_data.get("value_field") == "formal"
            assert harness.wizard_stage == "target"

            # Second turn: extraction runs normally (not skipped)
            await harness.chat("extra details here")
            assert harness.wizard_data.get("extra_field") == "details"
            assert harness.wizard_stage == "done"


class TestReExtractAdvancePath:
    """Unit tests: re_extract_on_entry via the advance() API.

    These exercise the skip_extraction clearance code path that is
    only reachable through advance() — in the conversational path,
    process_input clears skip_extraction before re-extraction runs.
    """

    @pytest.mark.asyncio
    async def test_advance_re_extraction_clears_skip_extraction(
        self,
    ) -> None:
        """advance() re-extraction clears skip_extraction on the state.

        When a prior advance() triggered auto-advance, skip_extraction
        is left True on the state.  If the next advance() transitions to
        a re_extract_on_entry stage, re-extraction clears the flag so
        the following turn's extraction is not skipped.
        """
        config = _edit_back_config(re_extract=True, auto_advance_filled=False)
        # Force LLM extraction on all schema stages so the
        # ConfigurableExtractor is used instead of verbatim capture
        # (same override BotTestHarness applies automatically).
        for stage_def in config.get("stages", []):
            if stage_def.get("schema"):
                stage_def["capture_mode"] = "extract"
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm, strict_validation=False,
        )

        # Set up extractor: first call for source extraction, second
        # for re-extraction at target.
        extractor = ConfigurableExtractor(results=[
            SimpleExtractionResult(
                data={"routing_field": "go"}, confidence=0.9,
            ),
            SimpleExtractionResult(
                data={"value_field": "formal"}, confidence=0.9,
            ),
        ])
        reasoning.set_extractor(extractor)

        provider = EchoProvider({"provider": "echo", "model": "test"})
        state = WizardState(
            current_stage="source",
            data={},
            history=["source"],
            stage_entry_time=time.time(),
        )

        # Simulate a prior auto-advance leaving skip_extraction=True
        state.skip_extraction = True

        result = await reasoning.advance(
            "Route and set value to formal", state, llm=provider,
        )

        # Transitioned source → target, re-extraction captured value_field,
        # then auto-advance chained target → done (Item 92: re-extraction
        # relaxes the auto_advance gate).
        assert result.transitioned is True
        assert state.data.get("routing_field") == "go"
        assert state.data.get("value_field") == "formal"
        assert state.current_stage == "done"
        assert state.completed is True
        # skip_extraction is True because auto-advance set it — harmless
        # since the wizard is complete.  The re-extraction DID clear it
        # (allowing extraction to run), then auto-advance re-set it.
        assert state.skip_extraction is True


class TestReExtractAutoAdvanceFalse:
    """Item 92: re-extraction should chain-advance past auto_advance: false.

    ``auto_advance: false`` means "don't auto-advance during normal turn
    processing" — NOT "never advance under any circumstances".  After
    re-extraction captures data satisfying the transition condition, the
    wizard should advance.
    """

    @pytest.mark.asyncio
    async def test_re_extract_advances_past_auto_advance_false(self) -> None:
        """Bug 92: re-extraction should chain-advance past auto_advance: false.

        auto_advance: false means 'don't auto-advance during normal flow',
        not 'never advance'.  After re-extraction captures data satisfying
        the transition condition, the wizard should advance.
        """
        config = _edit_back_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                # Turn 1 at source: extracts routing_field → transitions
                [{"routing_field": "go"}],
                # Re-extraction at target: extracts value_field
                [{"value_field": "formal"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("routing_field") == "go"
            assert harness.wizard_data.get("value_field") == "formal"
            # Should chain-advance: source → target (re-extract) → done
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_re_extract_no_data_stays_at_auto_advance_false(
        self,
    ) -> None:
        """auto_advance: false stage stays when re-extraction produces no data."""
        config = _edit_back_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Please enter a value."],
            extraction_results=[
                [{"routing_field": "go"}],
                # Re-extraction at target: no data captured
                [{}],
            ],
        ) as harness:
            await harness.chat("Route to target")
            assert harness.wizard_stage == "target"
            assert harness.wizard_data.get("value_field") is None

    @pytest.mark.asyncio
    async def test_auto_advance_false_blocks_without_re_extraction(
        self,
    ) -> None:
        """auto_advance: false blocks auto-advance when data is pre-filled.

        Regression guard: auto_advance: false must block the auto-advance
        loop regardless of how the data was filled.  Here, value_field is
        captured at the source stage (in state.data via extraction), but
        auto-advance at target should still be blocked because no
        re-extraction occurred.
        """
        config = _edit_back_config(
            re_extract=False,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Enter value."],
            extraction_results=[
                # Source extraction captures both fields
                [{"routing_field": "go", "value_field": "formal"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("routing_field") == "go"
            assert harness.wizard_data.get("value_field") == "formal"
            # auto_advance: false blocks — stays at target
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_re_extract_advances_global_off_no_explicit_stage(
        self,
    ) -> None:
        """Re-extraction advances even when global auto_advance_filled=False
        and stage has no explicit auto_advance setting.

        This exercises the second gate in can_auto_advance (global off +
        no stage-level setting) — distinct from the first gate (explicit
        auto_advance: false).
        """
        config = _edit_back_config(
            re_extract=True,
            auto_advance_filled=False,
            target_auto_advance=None,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"routing_field": "go"}],
                [{"value_field": "formal"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("value_field") == "formal"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_re_extract_with_explicit_auto_advance_true(self) -> None:
        """Explicit auto_advance: true + re-extraction chains as before."""
        config = _edit_back_config(
            re_extract=True,
            target_auto_advance=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"routing_field": "go"}],
                [{"value_field": "formal"}],
            ],
        ) as harness:
            await harness.chat("Route and set value to formal")
            assert harness.wizard_data.get("value_field") == "formal"
            assert harness.wizard_stage == "done"
