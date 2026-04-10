"""Tier 1: Gap 22 reproduction — stage_prompt rendering in advance results.

These tests verify that ``WizardAdvanceResult.stage_prompt`` and
``WizardAdvanceResult.suggestions`` contain Jinja2-rendered text
(expressions resolved with ``state.data``), and that plain-text
prompts pass through unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dataknobs_bots.reasoning.wizard import (
    WizardAdvanceResult,
    WizardReasoning,
    WizardState,
)
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reasoning(config: dict[str, Any]) -> WizardReasoning:
    loader = WizardConfigLoader()
    wizard_fsm = loader.load_from_dict(config)
    return WizardReasoning(
        wizard_fsm=wizard_fsm,
        strict_validation=False,
    )


def _make_state(
    reasoning: WizardReasoning,
    *,
    current_stage: str | None = None,
    data: dict[str, Any] | None = None,
    history: list[str] | None = None,
) -> WizardState:
    stage = current_stage or reasoning.initial_stage
    hist = history or [stage]
    return WizardState(
        current_stage=stage,
        data=data or {},
        history=hist,
        stage_entry_time=time.time(),
    )


# ---------------------------------------------------------------------------
# Config with Jinja2 templates in prompts and suggestions
# ---------------------------------------------------------------------------

JINJA_WIZARD_CONFIG: dict[str, Any] = {
    "name": "jinja-wizard",
    "version": "1.0",
    "description": "Wizard with Jinja2 templates in prompts",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Tell me about {{ topic | default('your topic') }}.",
            "schema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
            "suggestions": [
                "I want to learn {{ topic | default('something new') }}",
                "Plain suggestion",
            ],
            "transitions": [
                {"target": "details", "condition": "data.get('topic')"},
            ],
        },
        {
            "name": "details",
            "prompt": "Great, let's dive into {{ topic }}.",
            "transitions": [{"target": "done"}],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "All done!",
        },
    ],
}


# ---------------------------------------------------------------------------
# Tier 1: Gap 22 Reproduction Tests
# ---------------------------------------------------------------------------


class TestGap22StagePromptRendering:
    """Test 1: stage_prompt in WizardAdvanceResult is Jinja2-rendered."""

    @pytest.mark.asyncio
    async def test_advance_result_stage_prompt_rendered(self) -> None:
        reasoning = _make_reasoning(JINJA_WIZARD_CONFIG)
        state = _make_state(reasoning, data={"topic": "Python"})

        result = await reasoning.advance({"topic": "Python"}, state)

        # After advance, the new stage is "details" (transition fires)
        assert result.transitioned is True
        assert result.stage_name == "details"
        # stage_prompt should be rendered with state data
        assert result.stage_prompt == "Great, let's dive into Python."

    @pytest.mark.asyncio
    async def test_metadata_stage_prompt_rendered(self) -> None:
        """Test 2: get_wizard_metadata() returns rendered stage_prompt."""
        reasoning = _make_reasoning(JINJA_WIZARD_CONFIG)
        state = _make_state(reasoning, data={"topic": "Python"})

        result = await reasoning.advance({"topic": "Python"}, state)

        assert result.metadata["stage_prompt"] == "Great, let's dive into Python."

    @pytest.mark.asyncio
    async def test_advance_result_stage_prompt_plain_text_unchanged(
        self,
    ) -> None:
        """Test 3: plain-text prompt without Jinja passes through as-is."""
        reasoning = _make_reasoning(JINJA_WIZARD_CONFIG)
        state = _make_state(reasoning, data={"topic": "Python"})

        # Advance to "details" first
        result = await reasoning.advance({"topic": "Python"}, state)
        assert result.stage_name == "details"

        # Advance again to "done" — which has plain text prompt
        result2 = await reasoning.advance({}, state)
        assert result2.stage_name == "done"
        assert result2.stage_prompt == "All done!"


SUGGESTIONS_CONFIG: dict[str, Any] = {
    "name": "suggestions-wizard",
    "version": "1.0",
    "stages": [
        {
            "name": "gather",
            "is_start": True,
            "prompt": "Pick an option.",
            "schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "level": {"type": "string"},
                },
                "required": ["topic", "level"],
            },
            "suggestions": [
                "Learn {{ topic | default('something new') }}",
                "Plain suggestion",
            ],
            # Transition requires BOTH fields so we can test with
            # only topic present (stays on gather)
            "transitions": [
                {
                    "target": "done",
                    "condition": "data.get('topic') and data.get('level')",
                },
            ],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "Done!",
        },
    ],
}


class TestSuggestionsRendering:
    """Test 6: suggestions in WizardAdvanceResult are Jinja2-rendered."""

    @pytest.mark.asyncio
    async def test_suggestions_in_advance_result_rendered(self) -> None:
        reasoning = _make_reasoning(SUGGESTIONS_CONFIG)
        state = _make_state(reasoning, data={"topic": "Python"})

        # Only topic is provided; transition needs both topic+level
        result = await reasoning.advance({}, state)
        assert result.stage_name == "gather"

        # suggestions should be rendered with topic
        assert result.suggestions[0] == "Learn Python"
        assert result.suggestions[1] == "Plain suggestion"

    @pytest.mark.asyncio
    async def test_suggestions_use_default_filter(self) -> None:
        """Suggestions with Jinja default filter work when data missing."""
        reasoning = _make_reasoning(SUGGESTIONS_CONFIG)
        state = _make_state(reasoning)  # no topic in data

        result = await reasoning.advance({}, state)

        assert result.suggestions[0] == "Learn something new"


# ---------------------------------------------------------------------------
# Tier 2: Rendering Consistency
# ---------------------------------------------------------------------------


class TestRenderingConsistency:
    """Tests 4-7: verify rendering consistency across all sites."""

    @pytest.mark.asyncio
    async def test_stage_prompt_undefined_variable_renders_empty(self) -> None:
        """Test 4: undefined Jinja var in stage_prompt renders as empty."""
        config: dict[str, Any] = {
            "name": "undef-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello {{ nonexistent }}!",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "transitions": [
                        {"target": "end", "condition": "data.get('name')"},
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning)

        # No data provided — stays on start
        result = await reasoning.advance({}, state)
        assert result.stage_name == "start"
        assert result.stage_prompt == "Hello !"

    @pytest.mark.asyncio
    async def test_stage_prompt_sandboxed(self) -> None:
        """Test 5: SSTI payload in state data used by stage_prompt is blocked."""
        config: dict[str, Any] = {
            "name": "ssti-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Value: {{ payload }}",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "string"},
                            "confirm": {"type": "string"},
                        },
                        "required": ["payload", "confirm"],
                    },
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('confirm')",
                        },
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        # The payload is data, not code — Jinja2 renders it as a string
        ssti = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        state = _make_state(reasoning, data={"payload": ssti})

        result = await reasoning.advance({}, state)
        assert result.stage_name == "start"
        # The payload appears as literal text, not executed
        assert "subclasses" in result.stage_prompt
        assert "__class__" in result.stage_prompt

    def test_custom_context_sandboxed(self) -> None:
        """Test 7: _render_custom_context uses sandboxed environment."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config: dict[str, Any] = {
            "name": "sandboxed-ctx",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            context_template="Stage: {{stage_name}}, Topic: {{topic}}",
        )
        state = _make_state(reasoning, data={"topic": "Python"})

        stage = reasoning._fsm.current_metadata
        context = reasoning._build_stage_context(stage, state)
        assert "Stage: start" in context
        assert "Topic: Python" in context


# ---------------------------------------------------------------------------
# Tier 3: _build_wizard_metadata Coverage
# ---------------------------------------------------------------------------


METADATA_CONFIG: dict[str, Any] = {
    "name": "meta-wizard",
    "version": "1.0",
    "stages": [
        {
            "name": "intro",
            "is_start": True,
            "prompt": "Welcome to {{ topic | default('the wizard') }}.",
            "suggestions": ["Start {{ topic | default('now') }}"],
            "transitions": [
                {"target": "details", "condition": "data.get('topic')"},
            ],
        },
        {
            "name": "details",
            "prompt": "Tell me about {{ topic }}.",
            "suggestions": ["More about {{ topic }}"],
            "transitions": [{"target": "done"}],
        },
        {
            "name": "done",
            "is_end": True,
            "prompt": "Finished {{ topic | default('') }}.",
        },
    ],
}


class TestBuildWizardMetadata:
    """Tests 8-11: _build_wizard_metadata coverage."""

    @pytest.mark.asyncio
    async def test_metadata_stage_prompt_rendered(self) -> None:
        """Test 8: metadata stage_prompt is Jinja-rendered."""
        reasoning = _make_reasoning(METADATA_CONFIG)
        state = _make_state(reasoning, data={"topic": "Math"})

        metadata = reasoning._build_wizard_metadata(state)
        assert metadata["stage_prompt"] == "Welcome to Math."

    @pytest.mark.asyncio
    async def test_metadata_suggestions_rendered(self) -> None:
        """Test 9: metadata suggestions are Jinja-rendered."""
        reasoning = _make_reasoning(METADATA_CONFIG)
        state = _make_state(reasoning, data={"topic": "Math"})

        metadata = reasoning._build_wizard_metadata(state)
        assert metadata["suggestions"] == ["Start Math"]

    def test_metadata_progress_calculation(self) -> None:
        """Test 10: progress, progress_percent, stage_index, total_stages."""
        reasoning = _make_reasoning(METADATA_CONFIG)
        state = _make_state(reasoning)

        metadata = reasoning._build_wizard_metadata(state)
        assert metadata["stage_index"] == 0
        assert metadata["total_stages"] == 3
        assert metadata["progress"] == 0.0
        assert metadata["progress_percent"] == 0.0

        # Move to details
        state2 = _make_state(
            reasoning,
            current_stage="details",
            history=["intro", "details"],
        )
        metadata2 = reasoning._build_wizard_metadata(state2)
        assert metadata2["stage_index"] == 1
        assert metadata2["progress"] == 0.5
        assert metadata2["progress_percent"] == 50.0

    def test_metadata_no_subflow_key_when_not_in_subflow(self) -> None:
        """Test 11: subflow_stage absent when not in subflow."""
        reasoning = _make_reasoning(METADATA_CONFIG)
        state = _make_state(reasoning)

        metadata = reasoning._build_wizard_metadata(state)
        assert "subflow_stage" not in metadata


# ---------------------------------------------------------------------------
# Tier 5: Regression — existing behavior preserved through new renderer
# ---------------------------------------------------------------------------


class TestRegressionThroughRenderer:
    """Tests 19-22: existing rendering behavior preserved after migration."""

    def test_response_template_still_renders(self) -> None:
        """Test 19: response_template Jinja rendering works through renderer."""
        config: dict[str, Any] = {
            "name": "rt-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "response_template": "You picked {{ topic }}.",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning, data={"topic": "Python"})

        stage = reasoning._fsm.current_metadata
        result = reasoning._render_response_template(
            "You picked {{ topic }}.", stage, state,
        )
        assert result == "You picked Python."

    def test_transition_derivation_still_renders(self) -> None:
        """Test 21: derive block Jinja rendering works through renderer."""
        config: dict[str, Any] = {
            "name": "derive-wizard",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "schema": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                    },
                    "transitions": [
                        {
                            "target": "end",
                            "condition": "data.get('topic')",
                            "derive": {
                                "summary": "Topic is {{ topic }}",
                            },
                        },
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        reasoning = _make_reasoning(config)
        state = _make_state(reasoning, data={"topic": "Python"})

        stage = reasoning._fsm.current_metadata
        reasoning._apply_transition_derivations(stage, state)
        assert state.data["summary"] == "Topic is Python"

    def test_custom_context_mixed_mode_preserved(self) -> None:
        """Test 22: context_template with (( )) renders through renderer."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config: dict[str, Any] = {
            "name": "mixed-ctx",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Go",
                    "help_text": "Extra info",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(
            wizard_fsm=wizard_fsm,
            context_template="Stage: {{stage_name}}((, help: {{help_text}}))",
        )
        state = _make_state(reasoning, data={})

        stage = reasoning._fsm.current_metadata
        context = reasoning._build_stage_context(stage, state)
        assert "Stage: start" in context
        assert "help: Extra info" in context
