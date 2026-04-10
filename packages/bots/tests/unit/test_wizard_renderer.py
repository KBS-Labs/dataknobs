"""Tests for WizardRenderer — Tier 4 unit tests.

Tests the WizardRenderer class in isolation: context construction,
template rendering, error handling, sandboxing, mixed-mode (( ))
preprocessing, and injection resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from jinja2 import TemplateSyntaxError, UndefinedError

from dataknobs_bots.reasoning.wizard_renderer import WizardRenderer


# ---------------------------------------------------------------------------
# Minimal WizardState stub for unit tests — avoids importing the full
# WizardReasoning module.  Matches the fields accessed by WizardRenderer.
# ---------------------------------------------------------------------------
@dataclass
class _StubState:
    current_stage: str = "gather"
    data: dict[str, Any] = field(default_factory=dict)
    transient: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=lambda: ["gather"])
    completed: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def renderer() -> WizardRenderer:
    return WizardRenderer()


@pytest.fixture()
def stage() -> dict[str, Any]:
    return {
        "name": "details",
        "label": "Details",
        "prompt": "Tell me about your goals.",
        "help_text": "Be specific.",
        "suggestions": ["Option A", "Option B"],
    }


@pytest.fixture()
def state() -> _StubState:
    return _StubState(
        current_stage="details",
        data={"topic": "Python", "level": "beginner", "_internal": "hidden"},
        transient={"_temp": "ephemeral"},
        history=["intro", "details"],
        completed=False,
    )


# ===================================================================
# Test 12: build_context includes all canonical keys
# ===================================================================
class TestBuildContext:
    def test_includes_all_keys(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        ctx = renderer.build_context(stage, state)
        # Author-controlled keys from build_template_params
        assert ctx["stage_name"] == "details"
        assert ctx["stage_label"] == "Details"
        assert ctx["stage_prompt"] == "Tell me about your goals."
        assert ctx["help_text"] == "Be specific."
        assert ctx["suggestions"] == ["Option A", "Option B"]
        assert ctx["completed"] is False
        assert ctx["history"] == ["intro", "details"]
        # User data as top-level variables
        assert ctx["topic"] == "Python"
        assert ctx["level"] == "beginner"
        assert ctx["_internal"] == "hidden"  # present in all_data
        assert ctx["_temp"] == "ephemeral"  # transient merged
        # Filtered and unfiltered dicts
        assert "_internal" not in ctx["collected_data"]
        assert ctx["collected_data"]["topic"] == "Python"
        assert ctx["all_data"]["_internal"] == "hidden"
        assert ctx["raw_data"]["_internal"] == "hidden"
        # Defaults
        assert ctx["bank"] is None
        assert ctx["artifact"] is None

    def test_extra_context_merged(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        bank_obj = object()
        ctx = renderer.build_context(
            stage, state, extra_context={"bank": bank_obj, "custom_key": 42}
        )
        assert ctx["bank"] is bank_obj
        assert ctx["custom_key"] == 42


# ===================================================================
# Test 13: get_collected_data excludes internal keys
# ===================================================================
class TestCollectedData:
    def test_excludes_internal(self, state: _StubState) -> None:
        collected = WizardRenderer.get_collected_data(state)
        assert "topic" in collected
        assert "level" in collected
        assert "_internal" not in collected


# ===================================================================
# Tests 14-15: render with/without fallback
# ===================================================================
class TestRenderFallback:
    def test_with_fallback_returns_fallback_on_error(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        bad_template = "{{ broken | nonexistent_filter }}"
        result = renderer.render(
            bad_template, stage, state, fallback="safe value"
        )
        assert result == "safe value"

    def test_without_fallback_propagates_exception(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        bad_template = "{% for %}"  # syntax error
        with pytest.raises(TemplateSyntaxError):
            renderer.render(bad_template, stage, state)

    def test_none_fallback_is_valid(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        bad_template = "{% for %}"
        result = renderer.render(bad_template, stage, state, fallback=None)
        assert result is None


# ===================================================================
# Tests 16-17: render_list
# ===================================================================
class TestRenderList:
    def test_per_item_fallback(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        items = [
            "Hello {{ topic }}",
            "{% for %}",  # broken
            "Plain text",
        ]
        result = renderer.render_list(items, stage, state)
        assert result[0] == "Hello Python"
        assert result[1] == "{% for %}"  # returned as-is
        assert result[2] == "Plain text"

    def test_skips_plain_text(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        items = ["No templates here", "Just text"]
        result = renderer.render_list(items, stage, state)
        assert result == items

    def test_empty_list(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        assert renderer.render_list([], stage, state) == []


# ===================================================================
# Test 18: sandboxed environment
# ===================================================================
class TestSandboxing:
    def test_attribute_traversal_blocked(
        self, renderer: WizardRenderer, stage: dict, state: _StubState
    ) -> None:
        """SSTI payload attempting RCE via attribute traversal is blocked."""
        payload = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        with pytest.raises(Exception):  # SecurityError from sandbox
            renderer.render(payload, stage, state)


# ===================================================================
# Tests 18a-18g: mixed mode (( )) preprocessing
# ===================================================================
class TestMixedMode:
    def test_conditional_present(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18a: (( )) section retained when author-controlled vars have values."""
        stage = {
            "name": "details",
            "help_text": "Extra help here",
            "suggestions": [],
        }
        template = "Hello {{stage_name}}((, help: {{help_text}}))"
        result = renderer.render(template, stage, state, mixed_mode=True)
        assert result == "Hello details, help: Extra help here"

    def test_conditional_absent(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18b: (( )) section removed when author-controlled vars missing/empty."""
        stage = {
            "name": "details",
            "help_text": "",  # empty
            "suggestions": [],
        }
        template = "Hello {{stage_name}}((, help: {{help_text}}))"
        result = renderer.render(template, stage, state, mixed_mode=True)
        assert result == "Hello details"

    def test_with_jinja_features(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18c: (( )) + Jinja2 features work together."""
        stage = {"name": "details", "help_text": "tip", "suggestions": []}
        # Use {{topic}} without spaces — in mixed mode, the preprocessor
        # preserves whitespace around unmatched vars which can add spaces.
        template = (
            "Stage: {{stage_name}}"
            "((, help: {{help_text}}))"
            "{% if topic %}, topic={{topic}}{% endif %}"
        )
        result = renderer.render(template, stage, state, mixed_mode=True)
        assert "Stage: details" in result
        assert ", help: tip" in result
        assert "topic=Python" in result

    def test_false_ignores_conditionals(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18d: mixed_mode=False does NOT preprocess (( )) syntax."""
        stage = {"name": "details", "suggestions": []}
        template = "Hello ((world))"
        result = renderer.render(template, stage, state, mixed_mode=False)
        # (( )) passes through as literal text
        assert result == "Hello ((world))"

    def test_extra_context_author_key_flows_to_preprocessor(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """Extra-context author-controlled keys enrich (( )) preprocessor."""
        stage: dict[str, Any] = {"name": "details", "suggestions": []}
        # (( )) conditional needs {{var}} inside to trigger removal logic
        template = "Go((, skip={{can_skip}}))"
        # With default can_skip=False, section removed (falsy var)
        result = renderer.render(template, stage, state, mixed_mode=True)
        assert result == "Go"
        # With can_skip=True via extra_context, section retained
        result2 = renderer.render(
            template, stage, state,
            extra_context={"can_skip": True},
            mixed_mode=True,
        )
        assert result2 == "Go, skip=True"

    def test_user_data_not_substituted_by_preprocessor(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18e: User data survives (( )) preprocessing and resolves in Jinja2."""
        stage = {"name": "details", "suggestions": []}
        template = "Stage: {{stage_name}}, Topic: {{topic}}"
        result = renderer.render(template, stage, state, mixed_mode=True)
        # stage_name resolved by preprocessor, topic resolved by Jinja2
        assert result == "Stage: details, Topic: Python"

    def test_injection_via_user_data_blocked(
        self, renderer: WizardRenderer
    ) -> None:
        """18f: User-entered Jinja2 control flow is NOT interpreted as code."""
        malicious_value = "{% for i in range(999999999) %}x{% endfor %}"
        state = _StubState(
            data={"topic": malicious_value},
        )
        stage = {"name": "gather", "suggestions": []}
        # Use {{topic}} without spaces to avoid preprocessor whitespace
        template = "Your topic: {{topic}}"
        result = renderer.render(template, stage, state, mixed_mode=True)
        # The malicious payload appears as literal text, not executed
        assert result == f"Your topic: {malicious_value}"

    def test_user_data_in_conditional_section_removed(
        self, renderer: WizardRenderer, state: _StubState
    ) -> None:
        """18g: (( )) referencing user-data var is removed (not in template_params)."""
        stage = {"name": "details", "suggestions": []}
        template = "Hello((, your topic is {{topic}}))"
        result = renderer.render(template, stage, state, mixed_mode=True)
        # topic is not in template_params, so the conditional section is removed
        assert result == "Hello"


# ===================================================================
# render_simple
# ===================================================================
class TestRenderSimple:
    def test_basic_rendering(self, renderer: WizardRenderer) -> None:
        result = renderer.render_simple(
            "Hello {{ name }}", {"name": "World"}
        )
        assert result == "Hello World"

    def test_fallback_on_error(self, renderer: WizardRenderer) -> None:
        result = renderer.render_simple(
            "{% for %}", {}, fallback="default"
        )
        assert result == "default"

    def test_propagates_without_fallback(
        self, renderer: WizardRenderer
    ) -> None:
        with pytest.raises(TemplateSyntaxError):
            renderer.render_simple("{% for %}", {})

    def test_empty_template(self, renderer: WizardRenderer) -> None:
        assert renderer.render_simple("", {}) == ""


# ===================================================================
# Strict mode
# ===================================================================
class TestStrictMode:
    def test_strict_raises_on_undefined(self) -> None:
        renderer = WizardRenderer(strict=True)
        stage: dict[str, Any] = {"name": "test", "suggestions": []}
        state = _StubState()
        with pytest.raises(UndefinedError):
            renderer.render("{{ nonexistent_var }}", stage, state)

    def test_non_strict_renders_empty(
        self, renderer: WizardRenderer
    ) -> None:
        stage: dict[str, Any] = {"name": "test", "suggestions": []}
        state = _StubState()
        result = renderer.render("Hello {{ nonexistent_var }}!", stage, state)
        assert result == "Hello !"
