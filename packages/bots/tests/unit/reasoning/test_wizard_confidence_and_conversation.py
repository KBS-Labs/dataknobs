"""Tests for dk-40 (confidence gate) and dk-41 (conversation template).

Covers:
- StageSchema value object: construction, required-field queries, properties
- dk-41: Conversation-mode stages render template only on first turn,
  then use LLM mode for subsequent turns
- dk-40: Confidence gate correctly handles stages with no schema or
  with optional-only fields (required=[])
- First-render confirmation behavior with optional fields
- Render count lifecycle after greet()
- Regression guards for structured-stage template rendering
"""

from __future__ import annotations

import pytest

from dataknobs_bots.reasoning.wizard import StageSchema
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder
from dataknobs_llm.testing import text_response


# ═══════════════════════════════════════════════════════════════════════════
# StageSchema unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStageSchema:
    """Unit tests for the StageSchema value object."""

    def test_from_stage_no_schema(self) -> None:
        """Stage without schema key → empty StageSchema."""
        ss = StageSchema.from_stage({"name": "test"})
        assert not ss.exists
        assert ss.required_fields == []
        assert not ss.has_required_fields
        assert ss.properties == {}
        assert ss.raw == {}

    def test_from_stage_none_schema(self) -> None:
        """Stage with schema=None → same as no schema."""
        ss = StageSchema.from_stage({"name": "test", "schema": None})
        assert not ss.exists

    def test_from_stage_empty_schema(self) -> None:
        """Stage with empty schema dict → exists but empty."""
        ss = StageSchema.from_stage({"name": "test", "schema": {}})
        assert not ss.exists  # Empty dict is falsy
        assert ss.required_fields == []
        assert ss.properties == {}

    def test_from_stage_with_properties(self) -> None:
        """Stage with populated schema."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            },
        })
        assert ss.exists
        assert ss.required_fields == ["name"]
        assert ss.has_required_fields
        assert ss.property_names == {"name", "age"}
        assert ss.field_type("name") == "string"
        assert ss.field_type("age") == "integer"
        assert ss.field_type("missing") is None

    def test_can_satisfy_required_no_schema(self) -> None:
        """No schema → vacuously True."""
        ss = StageSchema.from_stage({})
        assert ss.can_satisfy_required({}) is True
        assert ss.can_satisfy_required({"anything": "value"}) is True

    def test_can_satisfy_required_empty_required(self) -> None:
        """required: [] → vacuously True."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {"opt": {"type": "string"}},
                "required": [],
            },
        })
        assert ss.can_satisfy_required({}) is True

    def test_can_satisfy_required_all_present(self) -> None:
        """All required fields present → True."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        })
        assert ss.can_satisfy_required({"name": "Alice"}) is True

    def test_can_satisfy_required_missing(self) -> None:
        """Required field missing → False."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        })
        assert ss.can_satisfy_required({}) is False
        assert ss.can_satisfy_required({"name": None}) is False

    def test_missing_required_no_schema(self) -> None:
        """No schema → empty set."""
        ss = StageSchema.from_stage({})
        assert ss.missing_required({}) == set()

    def test_missing_required_some_missing(self) -> None:
        """Some required fields missing."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                },
                "required": ["a", "b"],
            },
        })
        assert ss.missing_required({"a": "val"}) == {"b"}
        assert ss.missing_required({"a": "v", "b": "v"}) == set()
        assert ss.missing_required({}) == {"a", "b"}

    def test_get_property_returns_empty_for_missing(self) -> None:
        """get_property for unknown field → empty dict."""
        ss = StageSchema.from_stage({
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        })
        assert ss.get_property("name") == {"type": "string"}
        assert ss.get_property("missing") == {}


# ═══════════════════════════════════════════════════════════════════════════
# Bug reproduction tests (dk-40 and dk-41)
# ═══════════════════════════════════════════════════════════════════════════


class TestBugReproduction:
    """Tests covering the dk-40 and dk-41 bug scenarios."""

    @pytest.mark.asyncio
    async def test_dk41_conversation_template_renders_only_once(self) -> None:
        """dk-41: Conversation stages use template only for greeting,
        then LLM mode for subsequent turns.
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
        """dk-40: Stage with optional-only fields advances on ambiguous input.

        ConfigBot scenario: stage has schema with properties but required=[],
        user gives a declining response ("Later"), wizard advances via
        fallback transition.
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
# Confidence gate and schema handling
# ═══════════════════════════════════════════════════════════════════════════


class TestNoSchemaConfidenceGate:
    """No-schema stages bypass the confidence gate.

    ``_extract_data()`` returns confidence=1.0 for no-schema stages,
    so the confidence gate should never fire.
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


class TestVacuousTruthOverride:
    """Vacuous-truth override fires correctly for required=[].

    ``StageSchema.can_satisfy_required()`` returns ``True`` for empty
    required lists via Python's ``all(... for f in [])`` semantics.
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
        """required=[] advances even when extraction has errors.

        ``can_satisfy_required()`` checks field presence, not extraction
        errors — errors do not block the override.
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


class TestFirstRenderConfirmation:
    """First-render confirmation interaction with optional-fields stages.

    When ``greet()`` has already rendered the template (render_count >= 1),
    first-render confirmation does not fire on the user's first message,
    so transitions evaluate normally.
    """

    @pytest.mark.asyncio
    async def test_first_render_confirmation_with_optional_fields(self) -> None:
        """After greet() renders the template, first-render confirmation
        does not re-fire — optional-fields stage advances normally.
        """
        config = (
            WizardConfigBuilder("confirm-probe")
            .stage(
                "options",
                is_start=True,
                prompt="Configure options.",
                response_template="Settings: {{ feature_a | default('unset') }}",
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
                text_response("Settings: unset"),
                text_response("All done!"),
            ],
            extraction_results=[[{"feature_a": True}]],  # Extracts a value
        ) as harness:
            await harness.greet()

            # greet() already rendered the template (render_count >= 1),
            # so first-render confirmation does not fire here.
            result = await harness.chat("Enable feature A")
            assert result.wizard_stage == "done", (
                "After greet() renders the template, the user's first "
                "message should advance — not re-trigger confirmation"
            )

    @pytest.mark.asyncio
    async def test_confirm_first_render_false_unblocks_optional_fields(
        self,
    ) -> None:
        """confirm_first_render=false skips confirmation for optional-fields
        stages regardless of render count.
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


class TestRenderCountAfterGreet:
    """Render count is correctly incremented after greet().

    ``greet()`` increments render_count so that ``generate()`` on the
    user's first message does not re-trigger first-render confirmation.
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
# Conversation-mode regression guards
# ═══════════════════════════════════════════════════════════════════════════


class TestConversationModeRegressionGuards:
    """Guards for the conversation-mode template gate (dk-41 fix).

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
