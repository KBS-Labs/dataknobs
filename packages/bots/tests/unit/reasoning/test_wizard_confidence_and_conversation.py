"""Diagnostic tests for dk-40 (confidence gate) and dk-41 (conversation template).

Tier 1 tests reproduce reported bugs — expected to FAIL before fix.
Tier 2 tests probe hypotheses — their pass/fail pattern reveals the actual
root cause of dk-40 (which is not fully determined from code analysis alone).

These tests MUST be run before implementing any fixes.

dk-40: Wizard stages with optional-only fields (required=[]) get stuck in
    a clarification loop when extraction produces ambiguous results.

dk-41: Conversation-mode stages with response_template re-render the
    template on every turn instead of using LLM mode after the greeting.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm.testing import text_response


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Reproduce reported bugs
# ═══════════════════════════════════════════════════════════════════════════


class TestTier1ReproduceBugs:
    """Tests that reproduce the reported dk-40 and dk-41 bugs.

    These MUST fail before the fix is applied. If they pass on unfixed
    code, the reproduction is wrong and needs adjustment.
    """

    @pytest.mark.asyncio
    async def test_dk41_conversation_template_renders_only_once(self) -> None:
        """dk-41: Conversation stages must use template only for greeting.

        EXPECTED TO FAIL before fix: _generate_stage_response()
        unconditionally enters template mode when response_template is
        set, so every turn returns the greeting instead of an LLM
        response.
        """
        config = (
            WizardConfigBuilder("dk41-template-loop")
            .stage(
                "chat",
                is_start=True,
                prompt="Chat with the user about python.",
                mode="conversation",
                response_template="Hello! I'm here to help with python.",
            )
            .transition("end", "data.get('_intent') == 'quit'")
            .stage("end", is_end=True, prompt="Goodbye!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Closures capture variables from the enclosing scope..."),
            ],
        ) as harness:
            # First turn: greet → should render the template
            greet_result = await harness.greet()
            assert "Hello" in greet_result.response

            # Second turn: user asks a question → should get LLM response
            chat_result = await harness.chat("Tell me about closures")
            # The bug: this returns the template greeting again
            assert "Hello" not in chat_result.response, (
                "Conversation-mode stage re-rendered the greeting template "
                "instead of using LLM mode on the second turn"
            )

    @pytest.mark.asyncio
    async def test_dk40_optional_fields_stage_advances(self) -> None:
        """dk-40: Stage with optional-only fields should advance on
        ambiguous input via always-true transition.

        Reproduces the ConfigBot configure_options scenario: stage has
        schema with properties but required=[], user gives a declining
        response ("Later"), wizard should advance via fallback transition.

        EXPECTED TO FAIL before fix.
        """
        config = (
            WizardConfigBuilder("dk40-optional-fields")
            .stage(
                "options",
                is_start=True,
                prompt="Configure optional settings.",
                response_template="What would you like to configure?",
                confirm_first_render=False,
            )
            .field("feature_a", field_type="boolean", required=False)
            .field("feature_b", field_type="string", required=False)
            .transition("done", "True")  # Always-true fallback
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What would you like to configure?"),
                text_response("All done!"),
            ],
            extraction_results=[[{}]],  # Empty extraction — ambiguous input
        ) as harness:
            # Greet renders the template
            await harness.greet()

            # User declines — should advance to done via fallback transition
            result = await harness.chat("Later")
            assert result.wizard_stage == "done", (
                f"Stage stuck at '{result.wizard_stage}' instead of "
                f"advancing to 'done' — confidence gate or first-render "
                f"confirmation is blocking transition evaluation"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: Hypothesis probes
# ═══════════════════════════════════════════════════════════════════════════


class TestTier2NoSchemaPath:
    """Probe: Do no-schema stages bypass the confidence gate?

    _extract_data() returns confidence=1.0 for no-schema stages (line 4299).
    These tests verify that the no-schema shortcut works correctly.
    """

    @pytest.mark.asyncio
    async def test_no_schema_stage_bypasses_confidence_gate(self) -> None:
        """No-schema stages should never trigger the confidence gate."""
        config = (
            WizardConfigBuilder("no-schema-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
            )
            # No .field() calls — no schema
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Options available."),
                text_response("All done!"),
            ],
        ) as harness:
            await harness.greet()
            result = await harness.chat("Anything")
            assert result.wizard_stage == "done", (
                "No-schema stage should bypass confidence gate entirely "
                "(extraction returns confidence=1.0)"
            )


class TestTier2VacuousTruthOverride:
    """Probe: Does the vacuous-truth override fire for required=[]?

    When schema exists with required=[], all(... for f in []) is True,
    so is_confident should be overridden to True. This tests the override
    in isolation from other factors (confirm_first_render disabled).
    """

    @pytest.mark.asyncio
    async def test_vacuous_truth_override_empty_required(self) -> None:
        """required=[] with confirm_first_render=false should advance."""
        config = (
            WizardConfigBuilder("vacuous-truth-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                confirm_first_render=False,
            )
            .field("optional_a", field_type="boolean", required=False)
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Options."),
                text_response("All done!"),
            ],
            extraction_results=[[{}]],  # Empty extraction
        ) as harness:
            await harness.greet()
            result = await harness.chat("Later")
            assert result.wizard_stage == "done", (
                "Vacuous-truth override should fire for required=[] — "
                "all(... for f in []) is True"
            )

    @pytest.mark.asyncio
    async def test_vacuous_truth_with_extraction_errors(self) -> None:
        """required=[] should advance even when extraction has errors.

        The override at line 3276 checks can_satisfy (vacuous True for []),
        not extraction.errors. Errors should not block the override.
        """
        config = (
            WizardConfigBuilder("vacuous-errors-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                confirm_first_render=False,
            )
            .field("optional_a", field_type="string", required=False)
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Options."),
                text_response("All done!"),
            ],
            # Empty extraction simulates failed/ambiguous parse
            extraction_results=[[{}]],
        ) as harness:
            await harness.greet()
            result = await harness.chat("sdkjfhskdjfh")  # Gibberish
            assert result.wizard_stage == "done", (
                "required=[] should override low confidence even with "
                "extraction errors"
            )


class TestTier2FirstRenderConfirmation:
    """Probe: Does first-render confirmation cause a blocking loop?

    When extraction produces new_data_keys AND stage has response_template
    AND render_count == 0, the confirmation path at lines 2508-2557 fires,
    re-rendering the template and returning BEFORE transition evaluation.
    """

    @pytest.mark.asyncio
    async def test_first_render_confirmation_with_optional_fields(self) -> None:
        """Probe whether first-render confirmation blocks advancement
        when extraction returns data for optional fields.

        If T1 (test_dk40_optional_fields_stage_advances) fails but T2
        (test_vacuous_truth_override_empty_required) passes, this test
        reveals whether first-render confirmation is the actual blocker.
        """
        config = (
            WizardConfigBuilder("confirm-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                response_template="Settings: {{ feature_a | default('unset') }}",
                # Default confirm_first_render=True — may block!
            )
            .field("feature_a", field_type="boolean", required=False)
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Settings: unset"),
                text_response("All done!"),
            ],
            extraction_results=[[{"feature_a": True}]],  # Extracts a value
        ) as harness:
            await harness.greet()

            result = await harness.chat("Enable feature A")
            # If confirmation fires: stays at 'options' (template re-render)
            # If confirmation doesn't fire: advances to 'done'
            # Either outcome is diagnostic — record which happens.
            if result.wizard_stage == "options":
                # Confirmation fired — this is the suspected blocking path.
                # The stage has required=[], confirmation_first_render=True
                # (default), and new_data_keys is non-empty.
                # This is expected behavior for stages WITH required fields,
                # but for optional-only stages it creates a loop.
                pass  # Diagnostic: confirmation IS the blocker
            else:
                assert result.wizard_stage == "done"

    @pytest.mark.asyncio
    async def test_confirm_first_render_false_unblocks_optional_fields(
        self,
    ) -> None:
        """confirm_first_render=false should prevent the confirmation loop
        for optional-fields stages.
        """
        config = (
            WizardConfigBuilder("confirm-false-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                response_template="Settings: {{ feature_a | default('unset') }}",
                confirm_first_render=False,
            )
            .field("feature_a", field_type="boolean", required=False)
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Settings: unset"),
                text_response("All done!"),
            ],
            extraction_results=[[{"feature_a": True}]],
        ) as harness:
            await harness.greet()

            result = await harness.chat("Enable feature A")
            assert result.wizard_stage == "done", (
                "With confirm_first_render=false, optional-fields stage "
                "should advance immediately"
            )

    @pytest.mark.asyncio
    async def test_empty_extraction_skips_confirmation(self) -> None:
        """When extraction returns no data, new_data_keys is empty, so
        confirmation should NOT fire even with confirm_first_render=True.
        """
        config = (
            WizardConfigBuilder("empty-extraction-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                response_template="What would you like?",
                # Default confirm_first_render=True
            )
            .field("feature_a", field_type="boolean", required=False)
            .transition("done", "True")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("What would you like?"),
                text_response("All done!"),
            ],
            extraction_results=[[{}]],  # Empty — no new_data_keys
        ) as harness:
            await harness.greet()

            result = await harness.chat("Later")
            assert result.wizard_stage == "done", (
                "Empty extraction should not trigger confirmation — "
                "new_data_keys is empty"
            )


class TestTier2RenderCountAfterGreet:
    """Probe: Is render count correctly incremented after greet()?

    greet() increments render_count (line 2119). So when generate() runs
    for the user's first message, render_count should be >= 1, and the
    first-render confirmation path at line 2509 should NOT fire.
    """

    @pytest.mark.asyncio
    async def test_render_count_nonzero_after_greet(self) -> None:
        """After greet() on a stage with response_template, render count
        must be >= 1 so generate() doesn't re-trigger first-render
        confirmation.
        """
        config = (
            WizardConfigBuilder("render-count-probe")
            .stage(
                "gather",
                is_start=True,
                prompt="What's your name?",
                response_template="Welcome! Tell me your name.",
            )
            .field("name", field_type="string", required=True)
            .transition("done", "data.get('name')")
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("All done!"),
            ],
            extraction_results=[[{"name": "Alice"}]],
        ) as harness:
            greet_result = await harness.greet()
            assert "Welcome" in greet_result.response

            # After greet, render count should be >= 1
            # The user's first message should advance, not re-confirm
            result = await harness.chat("My name is Alice")
            assert result.wizard_stage == "done", (
                "After greet() renders a template, render_count should be "
                ">= 1 so the first chat() doesn't trigger confirmation"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3: Regression guards
# ═══════════════════════════════════════════════════════════════════════════


class TestTier3RegressionGuards:
    """Guards against unintended side effects of the dk-41 fix.

    The template rendering gate must NOT change behavior for structured
    (non-conversation) stages where template rendering every turn is
    the correct behavior.
    """

    @pytest.mark.asyncio
    async def test_structured_stage_template_renders_every_turn(self) -> None:
        """Non-conversation stages must render template on every turn.

        Structured stages use the template as the response (e.g. review
        pages showing accumulated data). The dk-41 fix must not break
        this — only conversation-mode stages should skip the template
        after the first render.
        """
        config = (
            WizardConfigBuilder("structured-template-guard")
            .stage(
                "review",
                is_start=True,
                prompt="Review your data.",
                response_template="Data: {{ name | default('none') }}",
                confirm_first_render=False,
            )
            .field("name", field_type="string", required=True)
            .transition("done", "False")  # Never transitions — stays at review
            .stage("done", is_end=True, prompt="Done!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("Data: Alice"),
                text_response("Data: Bob"),
            ],
            extraction_results=[
                [{"name": "Alice"}],
                [{"name": "Bob"}],
            ],
        ) as harness:
            # First message — template renders
            result1 = await harness.chat("Alice")
            assert "Data:" in result1.response

            # Second message — template MUST still render for structured stages
            result2 = await harness.chat("Bob")
            assert "Data:" in result2.response, (
                "Structured stages must render template on every turn"
            )

    @pytest.mark.asyncio
    async def test_conversation_mode_greet_uses_template(self) -> None:
        """Conversation-mode greet() should use the template for the
        initial greeting — the fix only skips template on subsequent turns.
        """
        config = (
            WizardConfigBuilder("conversation-greet-guard")
            .stage(
                "chat",
                is_start=True,
                prompt="Chat about things.",
                mode="conversation",
                response_template="Welcome! How can I help?",
            )
            .transition("end", "data.get('_intent') == 'quit'")
            .stage("end", is_end=True, prompt="Goodbye!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[],
        ) as harness:
            result = await harness.greet()
            assert "Welcome" in result.response, (
                "Conversation-mode greet() must render the template greeting"
            )

    @pytest.mark.asyncio
    async def test_conversation_mode_without_template_uses_llm(self) -> None:
        """Conversation stages without a template should always use LLM mode."""
        config = (
            WizardConfigBuilder("conversation-no-template")
            .stage(
                "chat",
                is_start=True,
                prompt="Chat about things.",
                mode="conversation",
            )
            .transition("end", "data.get('_intent') == 'quit'")
            .stage("end", is_end=True, prompt="Goodbye!")
            .build()
        )

        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=[
                text_response("I'm an LLM greeting."),
                text_response("Here's info about closures..."),
            ],
        ) as harness:
            greet_result = await harness.greet()
            assert "LLM greeting" in greet_result.response

            chat_result = await harness.chat("Tell me about closures")
            assert "closures" in chat_result.response
