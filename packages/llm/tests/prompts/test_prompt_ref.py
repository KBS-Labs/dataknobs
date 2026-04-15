"""Tests for prompt_ref() meta-prompt resolution.

Tests cover:
- Basic prompt_ref() resolution from a prompt library
- Recursive resolution (prompt_ref within a prompt_ref'd template)
- Cycle detection
- Missing key behavior (graceful empty string)
- No library set (warning + empty string)
- Extra params override referenced template defaults
- template_syntax normalization within prompt_ref()
- Integration with PromptBuilder
"""

import pytest

from dataknobs_llm.prompts import (
    TemplateRenderer,
    ConfigPromptLibrary,
    CompositePromptLibrary,
    PromptBuilder,
)
from dataknobs_llm.prompts.base.types import TemplateMode


# ============================================================================
# Helpers
# ============================================================================

def _make_library(prompts: dict) -> ConfigPromptLibrary:
    """Create a ConfigPromptLibrary from a dict of {name: template_str_or_dict}."""
    system_prompts = {}
    for name, value in prompts.items():
        if isinstance(value, str):
            system_prompts[name] = {"template": value}
        else:
            system_prompts[name] = value
    return ConfigPromptLibrary(config={"system": system_prompts})


# ============================================================================
# Basic prompt_ref resolution
# ============================================================================

class TestPromptRefBasic:

    def test_simple_resolution(self) -> None:
        """prompt_ref resolves a key from the library and returns its content."""
        library = _make_library({
            "greeting.header": "# Welcome",
            "greeting": '{{ prompt_ref("greeting.header") }}\nHello!',
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("greeting") }}', {}, mode=TemplateMode.JINJA2
        )
        assert "# Welcome" in result.content
        assert "Hello!" in result.content

    def test_fragment_with_variables(self) -> None:
        """prompt_ref'd templates can use variables from extra_params."""
        library = _make_library({
            "greet": "Hello {{ name }}!",
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("greet", name="Alice") }}',
            {},
            mode=TemplateMode.JINJA2,
        )
        assert result.content == "Hello Alice!"

    def test_fragment_with_defaults(self) -> None:
        """Referenced template defaults are used when extra_params don't override."""
        library = _make_library({
            "greet": {
                "template": "Hello {{ name }}!",
                "defaults": {"name": "World"},
            },
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("greet") }}', {}, mode=TemplateMode.JINJA2
        )
        assert result.content == "Hello World!"

    def test_extra_params_override_defaults(self) -> None:
        """Extra params in prompt_ref() override the referenced template's defaults."""
        library = _make_library({
            "greet": {
                "template": "Hello {{ name }}!",
                "defaults": {"name": "World"},
            },
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("greet", name="Alice") }}',
            {},
            mode=TemplateMode.JINJA2,
        )
        assert result.content == "Hello Alice!"


# ============================================================================
# Recursive resolution
# ============================================================================

class TestPromptRefRecursive:

    def test_two_level_resolution(self) -> None:
        """prompt_ref can resolve templates that themselves contain prompt_ref."""
        library = _make_library({
            "inner": "INNER",
            "middle": 'before-{{ prompt_ref("inner") }}-after',
            "outer": '{{ prompt_ref("middle") }}',
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("outer") }}', {}, mode=TemplateMode.JINJA2
        )
        assert result.content == "before-INNER-after"

    def test_meta_prompt_composition(self) -> None:
        """Real-world pattern: meta-prompt composes multiple fragments."""
        library = _make_library({
            "wizard.clarification.header": "## Clarification Needed",
            "wizard.clarification.preamble": "I need more info.",
            "wizard.clarification.instructions": "Please clarify.",
            "wizard.clarification": (
                '{{ prompt_ref("wizard.clarification.header") }}\n\n'
                '{{ prompt_ref("wizard.clarification.preamble") }}\n\n'
                '{{ prompt_ref("wizard.clarification.instructions") }}'
            ),
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("wizard.clarification") }}',
            {},
            mode=TemplateMode.JINJA2,
        )
        assert "## Clarification Needed" in result.content
        assert "I need more info." in result.content
        assert "Please clarify." in result.content


# ============================================================================
# Cycle detection
# ============================================================================

class TestPromptRefCycleDetection:

    def test_direct_self_reference(self) -> None:
        """A template referencing itself raises ValueError."""
        library = _make_library({
            "loop": '{{ prompt_ref("loop") }}',
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        with pytest.raises(ValueError, match="Circular prompt reference"):
            renderer.render(
                '{{ prompt_ref("loop") }}', {}, mode=TemplateMode.JINJA2
            )

    def test_indirect_cycle(self) -> None:
        """A -> B -> A cycle raises ValueError."""
        library = _make_library({
            "a": '{{ prompt_ref("b") }}',
            "b": '{{ prompt_ref("a") }}',
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        with pytest.raises(ValueError, match="Circular prompt reference"):
            renderer.render(
                '{{ prompt_ref("a") }}', {}, mode=TemplateMode.JINJA2
            )


# ============================================================================
# Missing key / no library
# ============================================================================

class TestPromptRefMissingKey:

    def test_missing_key_returns_empty(self) -> None:
        """A missing key returns empty string (not an error)."""
        library = _make_library({})
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("nonexistent") }}', {}, mode=TemplateMode.JINJA2
        )
        assert result.content == ""

    def test_no_library_returns_empty(self) -> None:
        """prompt_ref with no library set returns empty string."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        # No set_prompt_library call

        result = renderer.render(
            '{{ prompt_ref("any.key") }}', {}, mode=TemplateMode.JINJA2
        )
        assert result.content == ""


# ============================================================================
# Syntax normalization within prompt_ref
# ============================================================================

class TestPromptRefSyntaxNormalization:

    def test_format_syntax_in_referenced_template(self) -> None:
        """A referenced template with template_syntax='format' is normalized."""
        library = _make_library({
            "fragment": {
                "template": "Hello {name}!",
                "template_syntax": "format",
            },
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("fragment", name="Alice") }}',
            {},
            mode=TemplateMode.JINJA2,
        )
        assert result.content == "Hello Alice!"

    def test_format_syntax_with_literal_braces(self) -> None:
        """FORMAT template with literal braces works via prompt_ref."""
        library = _make_library({
            "fragment": {
                "template": "Return {{}}\nSchema: {schema}",
                "template_syntax": "format",
            },
        })
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        renderer.set_prompt_library(library)

        result = renderer.render(
            '{{ prompt_ref("fragment", schema="test") }}',
            {},
            mode=TemplateMode.JINJA2,
        )
        assert "Return {}" in result.content
        assert "Schema: test" in result.content


# ============================================================================
# Integration with PromptBuilder
# ============================================================================

class TestPromptRefBuilderIntegration:

    def test_builder_wires_library_to_renderer(self) -> None:
        """PromptBuilder sets prompt library on its renderer."""
        library = _make_library({
            "fragment": "FRAGMENT_CONTENT",
            "main": '{{ prompt_ref("fragment") }} plus more',
        })
        builder = PromptBuilder(library=library)

        result = builder.render_system_prompt("main")
        assert "FRAGMENT_CONTENT" in result.content
        assert "plus more" in result.content

    def test_builder_meta_prompt_with_params(self) -> None:
        """PromptBuilder renders meta-prompts that pass params to fragments."""
        library = _make_library({
            "greet.name": {
                "template": "Hello {{ name }}!",
            },
            "greet.farewell": "Goodbye!",
            "greet": (
                '{{ prompt_ref("greet.name") }}\n'
                '{{ prompt_ref("greet.farewell") }}'
            ),
        })
        builder = PromptBuilder(library=library)

        result = builder.render_system_prompt("greet", params={"name": "Alice"})
        # Note: prompt_ref'd templates get extra_params from the prompt_ref call,
        # not from the parent render context. The parent's params are not
        # automatically inherited by prompt_ref'd templates.
        # "greet.name" needs name passed explicitly via prompt_ref or defaults.
        assert "Goodbye!" in result.content
