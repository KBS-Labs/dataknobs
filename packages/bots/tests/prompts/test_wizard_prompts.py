"""Tests for wizard + utility prompt organization.

Verifies:
- All wizard prompt keys are registered
- Fragment constants have correct syntax annotations
- Meta-prompts compose fragments correctly via prompt_ref
- Memory and rubric prompt keys are registered
- Backward-compatible DEFAULT_SUMMARY_PROMPT matches fragment
"""

from dataknobs_llm.prompts import ConfigPromptLibrary, TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode

from dataknobs_bots.prompts.wizard import WIZARD_PROMPT_KEYS
from dataknobs_bots.prompts.memory import MEMORY_PROMPT_KEYS, DEFAULT_SUMMARY_PROMPT
from dataknobs_bots.prompts.rubric import RUBRIC_PROMPT_KEYS


def _make_library() -> ConfigPromptLibrary:
    """Create a library from all wizard + utility prompt keys."""
    all_keys = {**WIZARD_PROMPT_KEYS, **MEMORY_PROMPT_KEYS, **RUBRIC_PROMPT_KEYS}
    return ConfigPromptLibrary(config={"system": all_keys})


class TestWizardPromptKeys:

    def test_all_clarification_keys_present(self) -> None:
        expected = [
            "wizard.clarification",
            "wizard.clarification.header",
            "wizard.clarification.preamble",
            "wizard.clarification.issues",
            "wizard.clarification.goal",
            "wizard.clarification.instructions",
        ]
        for key in expected:
            assert key in WIZARD_PROMPT_KEYS, f"Missing key: {key}"

    def test_all_validation_keys_present(self) -> None:
        expected = [
            "wizard.validation",
            "wizard.validation.header",
            "wizard.validation.issues",
            "wizard.validation.goal",
            "wizard.validation.instructions",
        ]
        for key in expected:
            assert key in WIZARD_PROMPT_KEYS, f"Missing key: {key}"

    def test_all_transform_error_keys_present(self) -> None:
        expected = [
            "wizard.transform_error",
            "wizard.transform_error.header",
            "wizard.transform_error.detail",
            "wizard.transform_error.instructions",
        ]
        for key in expected:
            assert key in WIZARD_PROMPT_KEYS, f"Missing key: {key}"

    def test_all_restart_offer_keys_present(self) -> None:
        expected = [
            "wizard.restart_offer",
            "wizard.restart_offer.header",
            "wizard.restart_offer.status",
            "wizard.restart_offer.options",
            "wizard.restart_offer.instructions",
        ]
        for key in expected:
            assert key in WIZARD_PROMPT_KEYS, f"Missing key: {key}"

    def test_fragment_syntax_annotations(self) -> None:
        """All non-meta fragments should be FORMAT syntax."""
        for key, template in WIZARD_PROMPT_KEYS.items():
            if "." in key and key.count(".") == 2:
                # This is a fragment (e.g. wizard.clarification.header)
                assert template.get("template_syntax") == "format", (
                    f"Fragment {key} should be FORMAT syntax"
                )

    def test_meta_prompt_syntax_annotations(self) -> None:
        """All meta-prompts should be JINJA2 syntax."""
        meta_keys = [
            "wizard.clarification",
            "wizard.validation",
            "wizard.transform_error",
            "wizard.restart_offer",
        ]
        for key in meta_keys:
            assert WIZARD_PROMPT_KEYS[key].get("template_syntax") == "jinja2", (
                f"Meta-prompt {key} should be JINJA2 syntax"
            )


class TestWizardMetaPromptComposition:

    def test_clarification_meta_renders(self) -> None:
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("wizard.clarification")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "issue_list": "- Response was ambiguous",
                "stage_prompt": "What is your name?",
                "suggestions_text": "",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "## Clarification Needed" in result.content
        assert "Response was ambiguous" in result.content
        assert "What is your name?" in result.content
        assert "Be conversational and helpful" in result.content

    def test_validation_meta_renders(self) -> None:
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("wizard.validation")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "error_list": "- Name is required",
                "stage_prompt": "Please provide your name.",
            },
            mode=TemplateMode.JINJA2,
        )
        assert "## Validation Required" in result.content
        assert "Name is required" in result.content
        assert "Please provide your name." in result.content

    def test_transform_error_meta_renders(self) -> None:
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("wizard.transform_error")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {"stage_name": "configure", "error": "Timeout reached"},
            mode=TemplateMode.JINJA2,
        )
        assert "## Processing Error" in result.content
        assert "configure" in result.content
        assert "Timeout reached" in result.content

    def test_restart_offer_meta_renders(self) -> None:
        library = _make_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("wizard.restart_offer")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {"stage_name": "gather_info", "stage_prompt": "Tell me about yourself"},
            mode=TemplateMode.JINJA2,
        )
        assert "## Multiple Clarification Attempts" in result.content
        assert "gather_info" in result.content
        assert "Tell me about yourself" in result.content
        assert "restart" in result.content


class TestMemoryPromptKeys:

    def test_summary_key_present(self) -> None:
        assert "memory.summary" in MEMORY_PROMPT_KEYS

    def test_backward_compat_constant(self) -> None:
        """DEFAULT_SUMMARY_PROMPT matches the fragment template text."""
        assert DEFAULT_SUMMARY_PROMPT == MEMORY_PROMPT_KEYS["memory.summary"]["template"]

    def test_summary_format_renders(self) -> None:
        """The summary prompt renders with .format() correctly."""
        rendered = DEFAULT_SUMMARY_PROMPT.format(
            existing_summary="(none)",
            new_messages="user: Hello\nassistant: Hi there!",
        )
        assert "conversation summarizer" in rendered
        assert "(none)" in rendered
        assert "user: Hello" in rendered


class TestRubricPromptKeys:

    def test_all_keys_present(self) -> None:
        expected = [
            "rubric.feedback_summary.system",
            "rubric.feedback_summary.user",
            "rubric.classification",
        ]
        for key in expected:
            assert key in RUBRIC_PROMPT_KEYS, f"Missing key: {key}"

    def test_classification_format_renders(self) -> None:
        """The classification prompt renders with .format() correctly."""
        template = RUBRIC_PROMPT_KEYS["rubric.classification"]["template"]
        rendered = template.format(
            criterion_name="Clarity",
            criterion_description="How clear is the writing",
            level_descriptions="- high: Very clear\n- low: Unclear",
            valid_ids='"high", "low"',
            example_level_id="high",
        )
        assert "Clarity" in rendered
        assert "How clear is the writing" in rendered
        assert '{"level_id": "high"}' in rendered
