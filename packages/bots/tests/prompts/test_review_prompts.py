"""Tests for review persona prompt organization.

Verifies:
- All persona prompt keys are registered
- Shared fragments exist
- Meta-prompts compose correctly via prompt_ref
- All 5 built-in personas have complete key sets
"""

import pytest

from dataknobs_llm.prompts import ConfigPromptLibrary, TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

from dataknobs_bots.prompts.review import REVIEW_PROMPT_KEYS


BUILT_IN_PERSONAS = ["adversarial", "skeptical", "insightful", "minimalist", "downstream"]


def _make_library() -> ConfigPromptLibrary:
    return ConfigPromptLibrary(config={"system": REVIEW_PROMPT_KEYS})


class TestReviewSharedFragments:

    def test_response_format_present(self) -> None:
        assert "review.format.response" in REVIEW_PROMPT_KEYS

    def test_artifact_section_present(self) -> None:
        assert "review.artifact_section" in REVIEW_PROMPT_KEYS

    def test_response_format_contains_json_structure(self) -> None:
        template = REVIEW_PROMPT_KEYS["review.format.response"]["template"]
        assert '"passed"' in template
        assert '"score"' in template
        assert '"issues"' in template

    def test_artifact_section_has_placeholders(self) -> None:
        template = REVIEW_PROMPT_KEYS["review.artifact_section"]["template"]
        assert "{artifact_type}" in template
        assert "{artifact_name}" in template
        assert "{artifact_content}" in template


class TestPersonaKeyCompleteness:

    @pytest.mark.parametrize("persona", BUILT_IN_PERSONAS)
    def test_persona_has_all_keys(self, persona: str) -> None:
        expected = [
            f"review.persona.{persona}",
            f"review.persona.{persona}.role",
            f"review.persona.{persona}.focus",
            f"review.persona.{persona}.instructions",
        ]
        for key in expected:
            assert key in REVIEW_PROMPT_KEYS, f"Missing key: {key}"

    @pytest.mark.parametrize("persona", BUILT_IN_PERSONAS)
    def test_persona_meta_is_jinja2(self, persona: str) -> None:
        meta = REVIEW_PROMPT_KEYS[f"review.persona.{persona}"]
        assert meta.get("template_syntax") == "jinja2"

    @pytest.mark.parametrize("persona", BUILT_IN_PERSONAS)
    def test_persona_fragments_are_format(self, persona: str) -> None:
        for suffix in ["role", "focus", "instructions"]:
            key = f"review.persona.{persona}.{suffix}"
            assert REVIEW_PROMPT_KEYS[key].get("template_syntax") == "format", (
                f"{key} should be FORMAT syntax"
            )


class TestPersonaMetaPromptComposition:

    @pytest.mark.parametrize("persona", BUILT_IN_PERSONAS)
    def test_persona_meta_renders(self, persona: str) -> None:
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt(f"review.persona.{persona}")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "artifact_type": "code",
                "artifact_name": "test.py",
                "artifact_purpose": "Unit test",
                "artifact_content": "def test_example(): pass",
            },
            mode=TemplateMode.JINJA2,
        )

        # Every persona should include its role, focus, artifact, instructions, and format
        assert "## Your Focus" in result.content
        assert "## Artifact to Review" in result.content
        assert "test.py" in result.content
        assert "## Instructions" in result.content
        assert '"passed"' in result.content  # response format

    def test_fragment_override_changes_composition(self) -> None:
        """Overriding a single fragment changes the composed output."""
        custom_keys = dict(REVIEW_PROMPT_KEYS)
        custom_keys["review.persona.adversarial.role"] = {
            "template": "You are a CUSTOM adversarial reviewer.",
            "template_syntax": "format",
        }

        library = ConfigPromptLibrary(config={"system": custom_keys})
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("review.persona.adversarial")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "artifact_type": "doc",
                "artifact_name": "README.md",
                "artifact_purpose": "Documentation",
                "artifact_content": "# Hello",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "CUSTOM adversarial reviewer" in result.content
        # Other fragments still present
        assert "## Your Focus" in result.content
        assert "## Instructions" in result.content
