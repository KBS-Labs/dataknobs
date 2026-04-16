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
    async def test_advance_re_extraction_chains_to_completion(
        self,
    ) -> None:
        """advance() re-extraction clears skip_extraction, allowing
        extraction to run, then auto-advance chains to completion.

        When a prior advance() triggered auto-advance, skip_extraction
        is left True on the state.  If the next advance() transitions to
        a re_extract_on_entry stage, re-extraction clears the flag so
        extraction runs.  After re-extraction captures data, auto-advance
        chains through to done (Item 92 gate relaxation).
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
        # Verify extraction ran twice: once at source (routing_field),
        # once via re-extraction at target (value_field).  This confirms
        # skip_extraction was cleared — if it hadn't been, the second
        # extraction would have been skipped.
        assert len(extractor.extract_calls) == 2
        # skip_extraction is True because auto-advance re-set it after
        # re-extraction cleared it — harmless since the wizard is complete.
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


def _configbot_like_config(
    *,
    re_extract: bool | None = True,
    auto_advance_filled: bool = True,
    target_auto_advance: bool | None = False,
    target_required: list[str] | None = None,
) -> dict[str, Any]:
    """Build a wizard config mimicking the ConfigBot scenario.

    source (start) → target (9 optional fields, required: []) → done (end)

    The target stage has multiple optional fields with mixed types,
    similar to ConfigBot's configure_options stage.
    """
    builder = (
        WizardConfigBuilder("test-92b-configbot-like")
        .settings(auto_advance_filled_stages=auto_advance_filled)
        .stage("source", is_start=True, prompt="What would you like to change?")
        .field("edit_section", field_type="string", required=True)
        .transition("target", "data.get('edit_section') == 'options'")
        .stage(
            "target",
            prompt="Configure your options.",
            re_extract_on_entry=re_extract,
            auto_advance=target_auto_advance,
        )
        # Multiple optional fields (none required=True)
        .field("tone", field_type="string")
        .field("llm_model", field_type="string")
        .field("llm_provider", field_type="string")
        .field("kb_enabled", field_type="boolean")
        .field("tools_enabled", field_type="boolean")
        .field("hints_enabled", field_type="boolean")
        .field("max_hints", field_type="integer")
        .field("domain_name", field_type="string")
        .field("domain_id", field_type="string")
        .transition("done", "data.get('tone')")
        .stage("done", is_end=True, prompt="Configuration complete!")
    )
    config = builder.build()

    # Override required field list on the target stage.
    # WizardConfigBuilder starts with required: [] and only adds fields
    # that have required=True.  Since none do, required is already [].
    # But allow explicit override for testing variants.
    if target_required is not None:
        for s in config["stages"]:
            if s["name"] == "target":
                s["schema"]["required"] = target_required

    return config


class TestReExtractRequiredFieldsGate:
    """Item 92b: required-fields gate must be skipped after re-extraction.

    When a stage has ``required: []`` (all fields optional), the
    required-fields fallback treats ALL properties as required.
    Empty optional fields (e.g., ``llm_model = ''``) then block
    auto-advance even with ``after_re_extraction=True``.
    """

    @pytest.mark.asyncio
    async def test_required_empty_list_advances_after_re_extraction(
        self,
    ) -> None:
        """Core 92b bug: required: [] with empty optional fields blocks
        auto-advance after re-extraction.

        Mimics ConfigBot: source extracts edit_section='options',
        transitions to target.  Re-extraction captures tone='formal'.
        Pre-existing data includes llm_model='' (empty string).
        Transition condition data.get('tone') is satisfied.
        Expected: auto-advance fires, wizard reaches 'done'.
        Actual (before fix): stays at 'target' because llm_model='' fails
        the empty-string check in Gate 2.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                # Turn 1 at source: routes to target
                [{"edit_section": "options"}],
                # Re-extraction at target: captures tone
                [{"tone": "formal"}],
            ],
        ) as harness:
            await harness.greet()
            # Pre-populate data simulating prior visit to configure_options
            harness.seed_wizard_data({
                "llm_model": "",
                "kb_enabled": True,
                "llm_provider": "ollama",
            })

            await harness.chat("Change the tone to formal")

            assert harness.wizard_data.get("edit_section") == "options"
            assert harness.wizard_data.get("tone") == "formal"
            # Must advance past target despite llm_model=''
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_required_empty_list_none_optional_field_advances(
        self,
    ) -> None:
        """None value in optional field doesn't block after re-extraction.

        Similar to core bug but with None instead of empty string.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"edit_section": "options"}],
                [{"tone": "formal"}],
            ],
        ) as harness:
            await harness.greet()
            harness.seed_wizard_data({
                "llm_model": None,
                "domain_name": None,
            })

            await harness.chat("Change the tone to formal")
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_required_empty_list_missing_optional_fields_advances(
        self,
    ) -> None:
        """Missing optional fields (not in data at all) don't block after
        re-extraction.

        Most optional fields have never been set — they don't appear in
        wizard_state.data.  This is the most common case for first-visit
        edit-back.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"edit_section": "options"}],
                [{"tone": "formal"}],
            ],
        ) as harness:
            # No pre-populated data — all optional fields are absent
            await harness.chat("Change the tone to formal")
            assert harness.wizard_data.get("tone") == "formal"
            assert harness.wizard_stage == "done"

    # --- 3b: Gate 3 enforcement (must still block when transition unsatisfied) ---

    @pytest.mark.asyncio
    async def test_required_empty_list_no_transition_stays(self) -> None:
        """Gate 3 still enforced: no satisfied transition → no auto-advance.

        Re-extraction captures data but the transition condition
        (data.get('tone')) is not satisfied because a different field
        was extracted.  After re-extraction, Gate 2 is skipped but
        Gate 3 blocks because no transition condition fires.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!", "Please configure options."],
            extraction_results=[
                [{"edit_section": "options"}],
                # Re-extraction captures kb_enabled, NOT tone
                [{"kb_enabled": True}],
            ],
        ) as harness:
            await harness.greet()
            await harness.chat("Enable the knowledge base")
            # Transition requires data.get('tone') — not satisfied
            assert harness.wizard_stage == "target"

    @pytest.mark.asyncio
    async def test_required_empty_list_empty_re_extraction_stays(
        self,
    ) -> None:
        """Empty re-extraction → after_re_extraction=False → Gate 2 enforced.

        When re-extraction produces no data, after_re_extraction is False
        and all gates remain enforced.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Please configure options."],
            extraction_results=[
                [{"edit_section": "options"}],
                # Re-extraction finds nothing
                [{}],
            ],
        ) as harness:
            await harness.chat("Go to options")
            assert harness.wizard_stage == "target"

    # --- 3c: Regression guards (existing behavior preserved) ---

    @pytest.mark.asyncio
    async def test_required_empty_list_no_re_extract_stays(self) -> None:
        """Without re-extraction, auto_advance: false still blocks.

        Regression guard: pre-filled data satisfying the transition
        condition does NOT trigger auto-advance when re-extraction
        is disabled.
        """
        config = _configbot_like_config(
            re_extract=False,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Please configure options."],
            extraction_results=[
                # Source extraction captures edit_section AND tone
                [{"edit_section": "options", "tone": "formal"}],
            ],
        ) as harness:
            await harness.chat("Change to formal tone in options")
            # auto_advance: false blocks without re-extraction
            assert harness.wizard_stage == "target"
            # Data is present but auto-advance didn't fire
            assert harness.wizard_data.get("tone") == "formal"

    @pytest.mark.asyncio
    async def test_normal_auto_advance_with_required_fields_unaffected(
        self,
    ) -> None:
        """Normal auto-advance (explicit required fields) is unaffected.

        Regression guard: the existing path where required fields are
        explicitly listed and all filled still works correctly.
        Uses the original _edit_back_config (single required field).
        """
        config = _edit_back_config(
            re_extract=True,
            target_auto_advance=False,
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
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_normal_flow_required_empty_list_blocks_when_unfilled(
        self,
    ) -> None:
        """Normal flow (no re-extraction): required: [] → all fields treated
        as required, unfilled fields block auto-advance.

        Regression guard: the required-fields fallback behavior in
        normal flow is unchanged.
        """
        config = _configbot_like_config(
            re_extract=False,
            auto_advance_filled=True,
            target_auto_advance=True,  # Explicitly enabled
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Please configure options."],
            extraction_results=[
                [{"edit_section": "options"}],
            ],
        ) as harness:
            await harness.chat("Go to options")
            # auto_advance: true but required-fields fallback blocks
            # because no optional fields are filled
            assert harness.wizard_stage == "target"

    # --- 3d: Edge case tests ---

    @pytest.mark.asyncio
    async def test_required_empty_list_all_fields_empty_string(
        self,
    ) -> None:
        """All optional fields are empty strings — still advances after
        re-extraction if transition is satisfied.

        Worst case: every field in wizard_state.data is '' except the
        one captured by re-extraction.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"edit_section": "options"}],
                [{"tone": "casual"}],
            ],
        ) as harness:
            await harness.greet()
            # Pre-populate ALL optional fields with empty strings
            harness.seed_wizard_data({
                field: ""
                for field in [
                    "llm_model", "llm_provider", "kb_enabled",
                    "tools_enabled", "hints_enabled", "max_hints",
                    "domain_name", "domain_id",
                ]
            })

            await harness.chat("Set tone to casual")
            assert harness.wizard_data.get("tone") == "casual"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_explicit_required_field_with_optional_empty_advances(
        self,
    ) -> None:
        """Stage with explicit required=[tone] and other optional empty fields.

        When required fields are explicit (not the fallback-to-all path),
        only those fields are checked.  After re-extraction fills the
        required field, Gate 2 is skipped entirely — but even without
        the 92b fix, the explicit required path would pass because only
        'tone' is checked.

        This test confirms the fix doesn't regress explicit-required behavior.
        """
        config = _configbot_like_config(
            re_extract=True,
            target_auto_advance=False,
            target_required=["tone"],
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"edit_section": "options"}],
                [{"tone": "formal"}],
            ],
        ) as harness:
            await harness.greet()
            harness.seed_wizard_data({"llm_model": ""})
            await harness.chat("Change tone to formal")
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_chain_through_multiple_stages_after_re_extraction(
        self,
    ) -> None:
        """Re-extraction at target chains through an intermediate stage
        to the final stage.

        source → target (re-extract, auto_advance: false) → intermediate
        (auto_advance: true) → done

        Verifies after_re_extraction only applies to the first
        auto-advance iteration (count == 0), not subsequent stages.
        """
        builder = (
            WizardConfigBuilder("test-92b-chain")
            .settings(auto_advance_filled_stages=True)
            .stage("source", is_start=True, prompt="Route.")
            .field("edit_section", field_type="string", required=True)
            .transition("target", "data.get('edit_section') == 'options'")
            .stage(
                "target",
                prompt="Configure.",
                re_extract_on_entry=True,
                auto_advance=False,
            )
            .field("tone", field_type="string")
            .transition(
                "intermediate", "data.get('tone')",
            )
            .stage(
                "intermediate",
                prompt="Review.",
                auto_advance=True,
            )
            .field("confirmed", field_type="boolean")
            .transition("done", "data.get('confirmed')")
            .stage("done", is_end=True, prompt="Done!")
        )
        config = builder.build()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"edit_section": "options"}],
                [{"tone": "formal"}],
            ],
        ) as harness:
            await harness.greet()
            # Pre-populate confirmed so intermediate can chain to done
            harness.seed_wizard_data({"confirmed": True})

            await harness.chat("Change tone to formal")
            # Should chain: source → target → intermediate → done
            assert harness.wizard_stage == "done"


# --- 3e: Scenario-diverse helper configs ---


def _multi_section_form_config() -> dict[str, Any]:
    """Multi-section form with jump-to-section edit-back.

    review (start) → preferences (re-extract, auto_advance: false) → review_done (end)
    """
    builder = (
        WizardConfigBuilder("test-92b-multi-section")
        .settings(auto_advance_filled_stages=True)
        .stage("review", is_start=True, prompt="Review your settings.")
        .field("jump_to", field_type="string", required=True)
        .transition("preferences", "data.get('jump_to') == 'preferences'")
        .stage(
            "preferences",
            prompt="Set your preferences.",
            re_extract_on_entry=True,
            auto_advance=False,
        )
        .field("theme", field_type="string")
        .field("language", field_type="string")
        .field("notifications", field_type="boolean")
        .field("font_size", field_type="integer")
        .field("timezone", field_type="string")
        .transition("review_done", "data.get('theme')")
        .stage("review_done", is_end=True, prompt="Preferences updated!")
    )
    return builder.build()


def _progressive_disclosure_config() -> dict[str, Any]:
    """Progressive disclosure with unconditional transition.

    intake (start) → details (re-extract, auto_advance: false) → complete (end)
    """
    builder = (
        WizardConfigBuilder("test-92b-progressive")
        .settings(auto_advance_filled_stages=True)
        .stage("intake", is_start=True, prompt="Describe the issue.")
        .field("section", field_type="string", required=True)
        .transition("details", "data.get('section') == 'details'")
        .stage(
            "details",
            prompt="Add optional details.",
            re_extract_on_entry=True,
            auto_advance=False,
        )
        .field("priority", field_type="string")
        .field("assignee", field_type="string")
        .field("tags", field_type="string")
        .field("description", field_type="string")
        # Unconditional transition — no condition expression
        .transition("complete")
        .stage("complete", is_end=True, prompt="Issue filed!")
    )
    return builder.build()


def _confirmation_with_overrides_config() -> dict[str, Any]:
    """Confirmation stage with inline edit overrides.

    collect (start) → confirm (re-extract, auto_advance: false) → done (end)
    """
    builder = (
        WizardConfigBuilder("test-92b-confirm-override")
        .settings(auto_advance_filled_stages=True)
        .stage("collect", is_start=True, prompt="Enter initial data.")
        .field("section", field_type="string", required=True)
        .transition("confirm", "data.get('section') == 'review'")
        .stage(
            "confirm",
            prompt="Confirm your choices.",
            re_extract_on_entry=True,
            auto_advance=False,
        )
        .field("confirmed", field_type="boolean")
        .field("name_override", field_type="string")
        .field("email_override", field_type="string")
        .field("notes", field_type="string")
        .transition("done", "data.get('confirmed')")
        .stage("done", is_end=True, prompt="Confirmed!")
    )
    return builder.build()


def _non_string_optional_fields_config(
    *,
    settings_auto_advance: bool = False,
) -> dict[str, Any]:
    """Stage with only boolean/integer optional fields.

    route (start) → settings (re-extract) → done (end)

    Args:
        settings_auto_advance: ``auto_advance`` value for the settings
            stage.  Defaults to ``False`` (re-extraction path).
    """
    builder = (
        WizardConfigBuilder("test-92b-non-string")
        .settings(auto_advance_filled_stages=True)
        .stage("route", is_start=True, prompt="Route.")
        .field("section", field_type="string", required=True)
        .transition("settings", "data.get('section') == 'settings'")
        .stage(
            "settings",
            prompt="Configure settings.",
            re_extract_on_entry=True,
            auto_advance=settings_auto_advance,
        )
        .field("enabled", field_type="boolean")
        .field("verbose", field_type="boolean")
        .field("max_retries", field_type="integer")
        .field("timeout", field_type="integer")
        .field("log_level", field_type="string")
        .transition("done", "data.get('enabled') is not None")
        .stage("done", is_end=True, prompt="Settings saved!")
    )
    return builder.build()


class TestReExtractDiverseScenarios:
    """Exercise the Gate 2 bypass across structurally distinct wizard patterns.

    Each test uses a different wizard design to ensure the fix is correct
    for the general case, not just the ConfigBot shape.
    """

    @pytest.mark.asyncio
    async def test_multi_section_jump_to_preferences(self) -> None:
        """Multi-section form: jump back to preferences, change theme.

        Preferences has partially-filled optional fields from first visit.
        Re-extraction captures theme, transition on data.get('theme').
        """
        config = _multi_section_form_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"jump_to": "preferences"}],
                [{"theme": "dark"}],
            ],
        ) as harness:
            await harness.greet()
            # Simulate prior visit: language set, others partial/empty
            harness.seed_wizard_data({
                "language": "en",
                "notifications": None,
                "font_size": 14,
                "timezone": "",
            })

            await harness.chat("Go to preferences and set theme to dark")
            assert harness.wizard_data.get("theme") == "dark"
            # timezone='' does not block after re-extraction
            assert harness.wizard_stage == "review_done"

    @pytest.mark.asyncio
    async def test_multi_section_jump_no_transition_stays(self) -> None:
        """Multi-section form: jump back, re-extract different field.

        Re-extraction captures language instead of theme.  Transition
        requires data.get('theme') — not satisfied.
        """
        config = _multi_section_form_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                "Welcome!", "Got it!", "Set your preferences.",
            ],
            extraction_results=[
                [{"jump_to": "preferences"}],
                [{"language": "fr"}],
            ],
        ) as harness:
            await harness.greet()
            await harness.chat(
                "Go to preferences and change language to French",
            )
            assert harness.wizard_data.get("language") == "fr"
            # Transition requires theme — not satisfied
            assert harness.wizard_stage == "preferences"

    @pytest.mark.asyncio
    async def test_progressive_disclosure_unconditional_transition(
        self,
    ) -> None:
        """Progressive disclosure: unconditional transition fires after
        re-extraction even with missing optional fields.

        The transition has no condition — the ``else: return True``
        branch in Gate 3.
        """
        config = _progressive_disclosure_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"section": "details"}],
                # Re-extraction captures only priority
                [{"priority": "high"}],
            ],
        ) as harness:
            # assignee, tags, description all absent
            await harness.chat("Go to details and set priority to high")
            assert harness.wizard_data.get("priority") == "high"
            # Unconditional transition → advances despite missing fields
            assert harness.wizard_stage == "complete"

    @pytest.mark.asyncio
    async def test_progressive_disclosure_empty_re_extraction_stays(
        self,
    ) -> None:
        """Progressive disclosure: empty re-extraction still blocks.

        Even with an unconditional transition, if re-extraction produces
        no data then after_re_extraction=False and the normal gates
        apply.
        """
        config = _progressive_disclosure_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Add optional details."],
            extraction_results=[
                [{"section": "details"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("Go to details")
            # Empty re-extraction → all gates enforced → stays
            assert harness.wizard_stage == "details"

    @pytest.mark.asyncio
    async def test_confirmation_with_inline_override(self) -> None:
        """Confirmation stage: 'yes, but change the name'.

        Re-extraction captures confirmed=True and name_override='Alice'.
        Transition fires on data.get('confirmed').
        """
        config = _confirmation_with_overrides_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"section": "review"}],
                [{"confirmed": True, "name_override": "Alice"}],
            ],
        ) as harness:
            # email_override and notes never set
            await harness.chat("Yes, but change the name to Alice")
            assert harness.wizard_data.get("confirmed") is True
            assert harness.wizard_data.get("name_override") == "Alice"
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_confirmation_rejected_stays(self) -> None:
        """Confirmation stage: 'no, go back'.

        Re-extraction captures confirmed=False.  Transition requires
        data.get('confirmed') which is falsy.  Must stay at confirm.
        """
        config = _confirmation_with_overrides_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Confirm your choices."],
            extraction_results=[
                [{"section": "review"}],
                [{"confirmed": False}],
            ],
        ) as harness:
            await harness.chat("No, I want to change something")
            assert harness.wizard_data.get("confirmed") is False
            # data.get('confirmed') returns False which is falsy
            assert harness.wizard_stage == "confirm"

    @pytest.mark.asyncio
    async def test_non_string_fields_with_none_values(self) -> None:
        """Non-string optional fields: None booleans/integers don't block
        after re-extraction.

        Before the fix, Gate 2 would check ``value is None`` and block
        on any unset boolean/integer.
        """
        config = _non_string_optional_fields_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!"],
            extraction_results=[
                [{"section": "settings"}],
                [{"enabled": True}],
            ],
        ) as harness:
            # verbose, max_retries, timeout, log_level all absent (None)
            await harness.chat("Go to settings and enable it")
            assert harness.wizard_data.get("enabled") is True
            assert harness.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_non_string_fields_false_and_zero_are_legitimate(
        self,
    ) -> None:
        """Non-string optional fields: False and 0 are legitimate values.

        Pre-populate with enabled=False, max_retries=0.  These are real
        values that satisfy Gate 2 even in normal flow.  Regression guard
        for the fix not changing non-string type behavior.
        """
        config = _non_string_optional_fields_config(
            settings_auto_advance=True,
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Welcome!", "Got it!"],
            extraction_results=[
                [{"section": "settings"}],
            ],
        ) as harness:
            await harness.greet()
            # Pre-populate ALL fields with legitimate "zero" values
            harness.seed_wizard_data({
                "enabled": False,
                "verbose": False,
                "max_retries": 0,
                "timeout": 0,
                "log_level": "INFO",
            })

            await harness.chat("Go to settings")
            # All fields filled with legitimate values, auto_advance: true
            # data.get('enabled') returns False (falsy) — transition
            # condition uses ``is not None`` so it's satisfied
            assert harness.wizard_stage == "done"


# --- 4: capture_only mode ---


def _capture_only_config(
    *,
    unconditional: bool = False,
) -> dict[str, Any]:
    """Stage with ``re_extract_on_entry: "capture_only"``.

    route (start) → details (capture_only, auto_advance: false) → done (end)

    Args:
        unconditional: If ``True``, the details→done transition has no
            condition.  If ``False``, uses ``data.get('priority')``.
    """
    builder = (
        WizardConfigBuilder("test-capture-only")
        .settings(auto_advance_filled_stages=True)
        .stage("route", is_start=True, prompt="Route.")
        .field("section", field_type="string", required=True)
        .transition("details", "data.get('section') == 'details'")
        .stage(
            "details",
            prompt="Add details.",
            re_extract_on_entry="capture_only",
            auto_advance=False,
        )
        .field("priority", field_type="string")
        .field("assignee", field_type="string")
        .field("description", field_type="string")
    )
    if unconditional:
        builder = builder.transition("done")
    else:
        builder = builder.transition("done", "data.get('priority')")
    builder = builder.stage("done", is_end=True, prompt="Filed!")
    return builder.build()


class TestCaptureOnlyMode:
    """``re_extract_on_entry: "capture_only"`` captures data but does NOT
    relax auto-advance gates.

    The distinction from ``True``:

    - ``True`` → extract + relax Gate 1 (auto_advance) and Gate 2
      (required fields).  Gate 3 (transition condition) decides.
    - ``"capture_only"`` → extract but keep all gates enforced.
      The stage behaves as if re-extraction did not happen with
      respect to auto-advance.
    """

    @pytest.mark.asyncio
    async def test_capture_only_extracts_data(self) -> None:
        """Data IS captured at the landing stage even though gates
        are not relaxed."""
        config = _capture_only_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Add details."],
            extraction_results=[
                [{"section": "details"}],
                [{"priority": "high"}],
            ],
        ) as harness:
            await harness.chat("Go to details and set priority to high")
            # Data was captured...
            assert harness.wizard_data.get("priority") == "high"
            # ...but stage did NOT advance (gates not relaxed)
            assert harness.wizard_stage == "details"

    @pytest.mark.asyncio
    async def test_capture_only_unconditional_transition_stays(self) -> None:
        """Even with an unconditional transition, capture_only does NOT
        auto-advance.

        This is the key behavioral difference from ``True``: with
        ``re_extract_on_entry: true`` and an unconditional transition,
        the stage would advance.  With ``"capture_only"``, it stays.
        """
        config = _capture_only_config(unconditional=True)

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Add details."],
            extraction_results=[
                [{"section": "details"}],
                [{"priority": "high"}],
            ],
        ) as harness:
            await harness.chat("Go to details and set priority to high")
            assert harness.wizard_data.get("priority") == "high"
            # Unconditional transition BUT capture_only → stays
            assert harness.wizard_stage == "details"

    @pytest.mark.asyncio
    async def test_capture_only_empty_extraction_stays(self) -> None:
        """Empty re-extraction with capture_only also stays."""
        config = _capture_only_config()

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "Add details."],
            extraction_results=[
                [{"section": "details"}],
                [{}],
            ],
        ) as harness:
            await harness.chat("Go to details")
            assert harness.wizard_stage == "details"

    @pytest.mark.asyncio
    async def test_capture_only_vs_true_contrast(self) -> None:
        """Demonstrate the behavioral difference: same config structure,
        ``True`` advances but ``"capture_only"`` stays.

        Uses the progressive disclosure config (unconditional transition)
        to show the contrast with the existing
        ``test_progressive_disclosure_unconditional_transition`` test.
        """
        # "capture_only" with unconditional transition → stays
        config_capture = _capture_only_config(unconditional=True)

        async with await BotTestHarness.create(
            wizard_config=config_capture,
            main_responses=["Got it!", "Add details."],
            extraction_results=[
                [{"section": "details"}],
                [{"priority": "high"}],
            ],
        ) as harness:
            await harness.chat("Go to details and set priority to high")
            assert harness.wizard_stage == "details"  # stays

        # Same shape but re_extract_on_entry: true → advances
        config_true = _progressive_disclosure_config()

        async with await BotTestHarness.create(
            wizard_config=config_true,
            main_responses=["Got it!"],
            extraction_results=[
                [{"section": "details"}],
                [{"priority": "high"}],
            ],
        ) as harness:
            await harness.chat("Go to details and set priority to high")
            assert harness.wizard_stage == "complete"  # advances
