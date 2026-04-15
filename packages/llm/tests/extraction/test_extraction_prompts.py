"""Tests for extraction prompt organization.

Verifies:
- Backward-compatible flat constants match the original prompts exactly
- Fragment decomposition: each fragment renders to produce the correct section
- Meta-prompt composition: composing fragments via prompt_ref produces output
  equivalent to the original flat prompt
- ExtractionPromptLibrary: all keys are registered and resolvable
"""

from dataknobs_llm.extraction.prompts import (
    DEFAULT_EXTRACTION_PROMPT,
    EXTRACTION_WITH_ASSUMPTIONS_PROMPT,
    get_extraction_prompt_library,
)
from dataknobs_llm.prompts import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode


class TestBackwardCompatibleConstants:
    """The flat prompt constants must produce identical output to the originals."""

    def test_default_prompt_format_renders(self) -> None:
        """DEFAULT_EXTRACTION_PROMPT.format() produces valid output."""
        rendered = DEFAULT_EXTRACTION_PROMPT.format(
            schema='{"type": "object"}',
            context="Stage: test",
            text="My name is Alice",
        )
        assert "## Schema" in rendered
        assert '{"type": "object"}' in rendered
        assert "## Context" in rendered
        assert "Stage: test" in rendered
        assert "## Instructions" in rendered
        assert "## User Message" in rendered
        assert "My name is Alice" in rendered
        assert "## Extracted Data (JSON only):" in rendered
        # Literal braces should render as {}
        assert "return an empty object {}" in rendered

    def test_assumptions_prompt_format_renders(self) -> None:
        """EXTRACTION_WITH_ASSUMPTIONS_PROMPT.format() produces valid output."""
        rendered = EXTRACTION_WITH_ASSUMPTIONS_PROMPT.format(
            schema='{"type": "object"}',
            context="None",
            text="Test message",
        )
        assert "## Schema" in rendered
        assert "## Instructions" in rendered
        assert "## Example Output" in rendered
        assert "## User Message" in rendered
        assert "## Extracted Data and Assumptions (JSON only):" in rendered
        # JSON example with literal braces should render correctly
        assert '"data":' in rendered
        assert '"assumptions":' in rendered


class TestExtractionPromptLibrary:
    """The prompt library should contain all extraction prompt keys."""

    def test_all_default_keys_present(self) -> None:
        library = get_extraction_prompt_library()
        expected_keys = [
            "extraction.default",
            "extraction.default.schema_section",
            "extraction.default.context_section",
            "extraction.default.instructions",
            "extraction.default.message_section",
        ]
        for key in expected_keys:
            template = library.get_system_prompt(key)
            assert template is not None, f"Missing key: {key}"
            assert "template" in template, f"Key {key} has no template"

    def test_all_assumptions_keys_present(self) -> None:
        library = get_extraction_prompt_library()
        expected_keys = [
            "extraction.with_assumptions",
            "extraction.with_assumptions.instructions",
            "extraction.with_assumptions.example",
            "extraction.with_assumptions.message_section",
        ]
        for key in expected_keys:
            template = library.get_system_prompt(key)
            assert template is not None, f"Missing key: {key}"
            assert "template" in template, f"Key {key} has no template"

    def test_fragments_have_format_syntax(self) -> None:
        """All fragment templates should be annotated as FORMAT."""
        library = get_extraction_prompt_library()
        fragment_keys = [
            "extraction.default.schema_section",
            "extraction.default.context_section",
            "extraction.default.instructions",
            "extraction.default.message_section",
            "extraction.with_assumptions.instructions",
            "extraction.with_assumptions.example",
            "extraction.with_assumptions.message_section",
        ]
        for key in fragment_keys:
            template = library.get_system_prompt(key)
            assert template is not None, f"Missing key: {key}"
            assert template.get("template_syntax") == "format", (
                f"Key {key} should have template_syntax='format'"
            )

    def test_meta_prompts_have_jinja2_syntax(self) -> None:
        """Meta-prompts should be annotated as JINJA2."""
        library = get_extraction_prompt_library()
        meta_keys = [
            "extraction.default",
            "extraction.with_assumptions",
        ]
        for key in meta_keys:
            template = library.get_system_prompt(key)
            assert template is not None, f"Missing key: {key}"
            assert template.get("template_syntax") == "jinja2", (
                f"Key {key} should have template_syntax='jinja2'"
            )


class TestMetaPromptComposition:
    """Meta-prompts should compose fragments to match original prompt output."""

    def test_default_meta_prompt_renders(self) -> None:
        """The default meta-prompt should produce output matching the flat prompt."""
        library = get_extraction_prompt_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("extraction.default")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "schema": '{"type": "object"}',
                "context": "Stage: test",
                "text": "My name is Alice",
            },
            mode=TemplateMode.JINJA2,
        )

        # Verify all expected sections are present in the composed output
        assert "Extract structured data from the user's message." in result.content
        assert "## Schema" in result.content
        assert '{"type": "object"}' in result.content
        assert "## Context" in result.content
        assert "Stage: test" in result.content
        assert "## Instructions" in result.content
        assert "## User Message" in result.content
        assert "My name is Alice" in result.content
        assert "## Extracted Data (JSON only):" in result.content
        # Literal braces from the instructions fragment
        assert "return an empty object {}" in result.content

    def test_assumptions_meta_prompt_renders(self) -> None:
        """The assumptions meta-prompt should produce output with all sections."""
        library = get_extraction_prompt_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("extraction.with_assumptions")
        assert meta is not None

        result = renderer.render(
            meta["template"],
            {
                "schema": '{"type": "object"}',
                "context": "None",
                "text": "Test message",
            },
            mode=TemplateMode.JINJA2,
        )

        # Verify all expected sections
        assert "identify any assumptions made" in result.content
        assert "## Schema" in result.content
        assert "## Instructions" in result.content
        assert "## Example Output" in result.content
        assert "## User Message" in result.content
        assert "## Extracted Data and Assumptions (JSON only):" in result.content

    def test_assumptions_reuses_default_schema_section(self) -> None:
        """The assumptions meta-prompt reuses the default schema_section fragment."""
        library = get_extraction_prompt_library()
        meta = library.get_system_prompt("extraction.with_assumptions")
        assert meta is not None
        # The meta-prompt should reference the default schema_section
        assert 'extraction.default.schema_section' in meta["template"]


class TestFlatFragmentConsistency:
    """Verify flat backward-compat prompts produce identical output to composed fragments.

    The flat constants (DEFAULT_EXTRACTION_PROMPT, EXTRACTION_WITH_ASSUMPTIONS_PROMPT)
    and the fragment decomposition must stay in sync. This test catches drift.
    """

    def test_default_extraction_flat_matches_composed(self) -> None:
        """DEFAULT_EXTRACTION_PROMPT.format() == composed meta-prompt render."""
        test_vars = {
            "schema": '{"type": "object", "properties": {"name": {"type": "string"}}}',
            "context": "Stage: gather_info",
            "text": "My name is Alice and I am 30 years old",
        }

        flat_output = DEFAULT_EXTRACTION_PROMPT.format(**test_vars)

        library = get_extraction_prompt_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("extraction.default")
        assert meta is not None
        composed_output = renderer.render(
            meta["template"], test_vars, mode=TemplateMode.JINJA2,
        ).content

        assert flat_output == composed_output, (
            f"Flat and composed outputs diverged.\n"
            f"--- Flat ---\n{flat_output}\n"
            f"--- Composed ---\n{composed_output}"
        )

    def test_assumptions_flat_matches_composed(self) -> None:
        """EXTRACTION_WITH_ASSUMPTIONS_PROMPT.format() == composed meta-prompt render."""
        test_vars = {
            "schema": '{"type": "object", "properties": {"age": {"type": "integer"}}}',
            "context": "Collecting demographics",
            "text": "I think I'm about thirty",
        }

        flat_output = EXTRACTION_WITH_ASSUMPTIONS_PROMPT.format(**test_vars)

        library = get_extraction_prompt_library()
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        meta = library.get_system_prompt("extraction.with_assumptions")
        assert meta is not None
        composed_output = renderer.render(
            meta["template"], test_vars, mode=TemplateMode.JINJA2,
        ).content

        assert flat_output == composed_output, (
            f"Flat and composed outputs diverged.\n"
            f"--- Flat ---\n{flat_output}\n"
            f"--- Composed ---\n{composed_output}"
        )
