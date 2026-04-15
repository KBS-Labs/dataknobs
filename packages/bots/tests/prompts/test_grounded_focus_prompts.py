"""Tests for grounded synthesis and focus guard prompt organization.

Verifies:
- All grounded prompt keys are registered
- All focus prompt keys are registered
- Grounded synthesis meta-prompt renders with correct conditional logic
- Focus guidance meta-prompt renders with conditional fields
- Focus drift meta-prompt renders with tangent count branching
"""

import pytest

from dataknobs_llm.prompts import ConfigPromptLibrary, TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

from dataknobs_bots.prompts.grounded import GROUNDED_PROMPT_KEYS
from dataknobs_bots.prompts.focus import FOCUS_PROMPT_KEYS


def _make_library() -> ConfigPromptLibrary:
    all_keys = {**GROUNDED_PROMPT_KEYS, **FOCUS_PROMPT_KEYS}
    return ConfigPromptLibrary(config={"system": all_keys})


# ============================================================================
# Grounded prompt key tests
# ============================================================================

class TestGroundedPromptKeys:

    def test_all_synthesis_keys_present(self) -> None:
        expected = [
            "grounded.synthesis",
            "grounded.synthesis.base_instruction",
            "grounded.synthesis.citation_section",
            "grounded.synthesis.citation_source",
            "grounded.synthesis.bridge",
            "grounded.synthesis.strict",
            "grounded.synthesis.supplement",
            "grounded.synthesis.kb_wrapper",
        ]
        for key in expected:
            assert key in GROUNDED_PROMPT_KEYS, f"Missing key: {key}"

    def test_provenance_template_present(self) -> None:
        assert "grounded.provenance_template" in GROUNDED_PROMPT_KEYS


class TestGroundedSynthesisMetaPrompt:

    def test_strict_mode(self) -> None:
        """When allow_parametric is False, strict fragment is included."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("grounded.synthesis")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "require_citations": False,
                "allow_parametric": False,
                "citation_format": "source",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "Base your response" in result.content
        assert "explicitly state what is missing" in result.content
        assert "supplement" not in result.content.lower()

    def test_bridge_mode(self) -> None:
        """When allow_parametric is 'bridge', bridge fragment is included."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("grounded.synthesis")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "require_citations": False,
                "allow_parametric": "bridge",
                "citation_format": "source",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "synthesize concepts across" in result.content

    def test_supplement_mode(self) -> None:
        """When allow_parametric is True, supplement fragment is included."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("grounded.synthesis")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "require_citations": False,
                "allow_parametric": True,
                "citation_format": "source",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "supplement with general knowledge" in result.content

    def test_with_section_citations(self) -> None:
        """With require_citations + section format, section citation included."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("grounded.synthesis")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "require_citations": True,
                "allow_parametric": False,
                "citation_format": "section",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "section heading" in result.content

    def test_with_source_citations(self) -> None:
        """With require_citations + source format, source citation included."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("grounded.synthesis")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "require_citations": True,
                "allow_parametric": True,
                "citation_format": "source",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "source file" in result.content


# ============================================================================
# Focus prompt key tests
# ============================================================================

class TestFocusPromptKeys:

    def test_all_guidance_keys_present(self) -> None:
        expected = [
            "focus.guidance",
            "focus.guidance.header",
            "focus.guidance.goal",
            "focus.guidance.task",
            "focus.guidance.needed",
            "focus.guidance.collected",
            "focus.guidance.instructions",
        ]
        for key in expected:
            assert key in FOCUS_PROMPT_KEYS, f"Missing key: {key}"

    def test_all_drift_keys_present(self) -> None:
        expected = [
            "focus.drift",
            "focus.drift.header",
            "focus.drift.issue",
            "focus.drift.redirect",
            "focus.drift.firm_message",
            "focus.drift.gentle_message",
        ]
        for key in expected:
            assert key in FOCUS_PROMPT_KEYS, f"Missing key: {key}"


class TestFocusGuidanceMetaPrompt:

    def test_minimal_context(self) -> None:
        """With only primary_goal, optional fields are excluded."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("focus.guidance")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "primary_goal": "Complete the form",
                "current_task": "",
                "required_fields": "",
                "collected": "",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "## Focus Guidance" in result.content
        assert "Complete the form" in result.content
        assert "Stay focused" in result.content
        # Optional fields should not appear
        assert "Current Task" not in result.content
        assert "Still Needed" not in result.content
        assert "Already Have" not in result.content

    def test_full_context(self) -> None:
        """With all fields populated, all sections appear."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("focus.guidance")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "primary_goal": "Complete the form",
                "current_task": "Enter your name",
                "required_fields": "name, email",
                "collected": "age, city",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "Complete the form" in result.content
        assert "Enter your name" in result.content
        assert "name, email" in result.content
        assert "age, city" in result.content


class TestFocusDriftMetaPrompt:

    def test_gentle_correction(self) -> None:
        """Below max_tangent_depth, gentle message is used."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("focus.drift")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "reason": "User asked about weather",
                "suggested_redirect": "Back to the form",
                "tangent_count": 1,
                "max_tangent_depth": 3,
            },
            mode=TemplateMode.JINJA2,
        )
        assert "Focus Correction Needed" in result.content
        assert "weather" in result.content
        assert "gently steer" in result.content

    def test_firm_correction(self) -> None:
        """At or above max_tangent_depth, firm message is used."""
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("focus.drift")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "reason": "Persistent off-topic",
                "suggested_redirect": "",
                "tangent_count": 3,
                "max_tangent_depth": 3,
            },
            mode=TemplateMode.JINJA2,
        )
        assert "IMPORTANT" in result.content
        assert "firmly redirect" in result.content
