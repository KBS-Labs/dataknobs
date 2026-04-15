"""Tests for template syntax annotation and conversion utilities.

Tests cover:
- TemplateSyntax enum creation and parsing
- format_to_jinja2() conversion with edge cases
- jinja2_to_format() conversion with rejection of unsupported features
- detect_syntax() heuristic accuracy
- normalize_to_jinja2() dispatch
- Round-trip conversions where possible
- Integration with TemplateRenderer via template_syntax field on PromptTemplateDict
"""

import pytest

from dataknobs_llm.prompts.syntax import (
    TemplateSyntax,
    format_to_jinja2,
    jinja2_to_format,
    detect_syntax,
    normalize_to_jinja2,
)
from dataknobs_llm.prompts import TemplateRenderer
from dataknobs_llm.prompts.base.types import TemplateMode


# ============================================================================
# TemplateSyntax enum
# ============================================================================

class TestTemplateSyntax:

    def test_values(self) -> None:
        assert TemplateSyntax.FORMAT.value == "format"
        assert TemplateSyntax.JINJA2.value == "jinja2"

    def test_from_string(self) -> None:
        assert TemplateSyntax.from_string("format") == TemplateSyntax.FORMAT
        assert TemplateSyntax.from_string("jinja2") == TemplateSyntax.JINJA2

    def test_from_string_case_insensitive(self) -> None:
        assert TemplateSyntax.from_string("FORMAT") == TemplateSyntax.FORMAT
        assert TemplateSyntax.from_string("Jinja2") == TemplateSyntax.JINJA2

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid template syntax"):
            TemplateSyntax.from_string("mako")


# ============================================================================
# format_to_jinja2
# ============================================================================

class TestFormatToJinja2:

    def test_empty_string(self) -> None:
        assert format_to_jinja2("") == ""

    def test_no_placeholders(self) -> None:
        text = "Hello, world! This is plain text."
        assert format_to_jinja2(text) == text

    def test_simple_variable(self) -> None:
        assert format_to_jinja2("{name}") == "{{ name }}"

    def test_multiple_variables(self) -> None:
        result = format_to_jinja2("Hello {name}, you are {age} years old.")
        assert result == "Hello {{ name }}, you are {{ age }} years old."

    def test_variable_with_underscores(self) -> None:
        assert format_to_jinja2("{my_var_name}") == "{{ my_var_name }}"

    def test_variable_with_format_spec_dropped(self) -> None:
        # Format specs are dropped since Jinja2 uses filters instead
        assert format_to_jinja2("{name:>10}") == "{{ name }}"

    def test_variable_with_conversion_flag_dropped(self) -> None:
        assert format_to_jinja2("{name!s}") == "{{ name }}"

    def test_literal_open_brace(self) -> None:
        # {{ in .format() means literal {
        assert format_to_jinja2("JSON: {{") == 'JSON: {{ "{" }}'

    def test_literal_close_brace(self) -> None:
        # }} in .format() means literal }
        assert format_to_jinja2("JSON: }}") == 'JSON: {{ "}" }}'

    def test_literal_braces_around_content(self) -> None:
        # {{key}} in .format() means literal { + key + literal }
        # This is NOT a variable — it's literal braces around text
        result = format_to_jinja2('{{key}}')
        assert result == '{{ "{" }}key{{ "}" }}'

    def test_json_example_in_prompt(self) -> None:
        """Real-world case: extraction prompts contain JSON examples with literal braces."""
        template = (
            "Return ONLY a valid JSON object:\n"
            '{{"name": "John", "age": 30}}'
        )
        result = format_to_jinja2(template)
        assert '{{ "{" }}' in result
        assert '{{ "}" }}' in result
        # Should not contain any variable references for "name" or "age"
        assert "{{ name }}" not in result

    def test_mixed_variables_and_literal_braces(self) -> None:
        """The extraction prompts have both {var} and {{literal}} patterns."""
        template = "Schema: {schema}\nExample: {{}}"
        result = format_to_jinja2(template)
        assert "{{ schema }}" in result
        assert '{{ "{" }}{{ "}" }}' in result

    def test_extraction_prompt_json_example(self) -> None:
        """Test the specific pattern from EXTRACTION_WITH_ASSUMPTIONS_PROMPT."""
        template = '{{"data": {{"name": "John"}}, "assumptions": []}}'
        result = format_to_jinja2(template)
        # All {{ and }} should become literal brace expressions
        assert "{schema}" not in result  # no format vars
        # The result should be renderable by Jinja2 to produce the original JSON
        from jinja2 import Environment
        env = Environment()
        rendered = env.from_string(result).render()
        assert rendered == '{"data": {"name": "John"}, "assumptions": []}'

    def test_multiline_template(self) -> None:
        template = "## Schema\n{schema}\n\n## Text\n{text}"
        result = format_to_jinja2(template)
        assert result == "## Schema\n{{ schema }}\n\n## Text\n{{ text }}"


# ============================================================================
# jinja2_to_format
# ============================================================================

class TestJinja2ToFormat:

    def test_empty_string(self) -> None:
        assert jinja2_to_format("") == ""

    def test_no_placeholders(self) -> None:
        text = "Hello, world!"
        assert jinja2_to_format(text) == text

    def test_simple_variable(self) -> None:
        assert jinja2_to_format("{{ name }}") == "{name}"

    def test_variable_no_spaces(self) -> None:
        assert jinja2_to_format("{{name}}") == "{name}"

    def test_multiple_variables(self) -> None:
        result = jinja2_to_format("Hello {{ name }}, age {{ age }}.")
        assert result == "Hello {name}, age {age}."

    def test_rejects_block_tags(self) -> None:
        with pytest.raises(ValueError, match="block tags"):
            jinja2_to_format("{% if x %}yes{% endif %}")

    def test_rejects_comments(self) -> None:
        with pytest.raises(ValueError, match="comments"):
            jinja2_to_format("{# a comment #}")

    def test_rejects_filters(self) -> None:
        with pytest.raises(ValueError, match="filters"):
            jinja2_to_format("{{ name | upper }}")

    def test_rejects_function_calls(self) -> None:
        with pytest.raises(ValueError, match="function calls"):
            jinja2_to_format('{{ prompt_ref("key") }}')

    def test_rejects_complex_expressions(self) -> None:
        with pytest.raises(ValueError, match="complex Jinja2 expression"):
            jinja2_to_format("{{ a + b }}")

    def test_literal_brace_expression_to_format_escape(self) -> None:
        assert jinja2_to_format('{{ "{" }}') == "{{"
        assert jinja2_to_format('{{ "}" }}') == "}}"


# ============================================================================
# detect_syntax
# ============================================================================

class TestDetectSyntax:

    def test_empty_string(self) -> None:
        assert detect_syntax("") == TemplateSyntax.FORMAT

    def test_plain_text(self) -> None:
        assert detect_syntax("Hello world") == TemplateSyntax.FORMAT

    def test_format_single_brace_var(self) -> None:
        assert detect_syntax("Hello {name}!") == TemplateSyntax.FORMAT

    def test_format_with_spec(self) -> None:
        assert detect_syntax("{value:>10}") == TemplateSyntax.FORMAT

    def test_jinja2_block_tag(self) -> None:
        assert detect_syntax("{% if x %}yes{% endif %}") == TemplateSyntax.JINJA2

    def test_jinja2_comment(self) -> None:
        assert detect_syntax("{# comment #}") == TemplateSyntax.JINJA2

    def test_jinja2_filter(self) -> None:
        assert detect_syntax("{{ name | upper }}") == TemplateSyntax.JINJA2

    def test_jinja2_function_call(self) -> None:
        assert detect_syntax('{{ prompt_ref("key") }}') == TemplateSyntax.JINJA2

    def test_jinja2_prompt_ref_keyword(self) -> None:
        assert detect_syntax("Uses prompt_ref for composition") == TemplateSyntax.JINJA2

    def test_jinja2_simple_var(self) -> None:
        assert detect_syntax("Hello {{ name }}") == TemplateSyntax.JINJA2

    def test_real_extraction_prompt_is_format(self) -> None:
        """The real extraction prompt uses {schema}, {context}, {text}."""
        template = (
            "Extract structured data.\n"
            "## Schema\n{schema}\n"
            "## Context\n{context}\n"
            "## User Message\n{text}"
        )
        assert detect_syntax(template) == TemplateSyntax.FORMAT

    def test_meta_prompt_is_jinja2(self) -> None:
        """Meta-prompts use prompt_ref() and are Jinja2."""
        template = (
            '{{ prompt_ref("wizard.clarification.header") }}\n\n'
            '{{ prompt_ref("wizard.clarification.preamble") }}'
        )
        assert detect_syntax(template) == TemplateSyntax.JINJA2


# ============================================================================
# normalize_to_jinja2
# ============================================================================

class TestNormalizeToJinja2:

    def test_jinja2_passthrough(self) -> None:
        template = "{{ name }}"
        assert normalize_to_jinja2(template, TemplateSyntax.JINJA2) == template

    def test_format_converts(self) -> None:
        assert normalize_to_jinja2("{name}", TemplateSyntax.FORMAT) == "{{ name }}"

    def test_empty_passthrough(self) -> None:
        assert normalize_to_jinja2("", TemplateSyntax.FORMAT) == ""
        assert normalize_to_jinja2("", TemplateSyntax.JINJA2) == ""


# ============================================================================
# Round-trip tests
# ============================================================================

class TestRoundTrip:

    def test_simple_variables_round_trip(self) -> None:
        """format → jinja2 → format should produce original for simple templates."""
        original = "Hello {name}, you are {age} years old."
        jinja2 = format_to_jinja2(original)
        back = jinja2_to_format(jinja2)
        assert back == original

    def test_plain_text_round_trip(self) -> None:
        original = "No variables here."
        assert jinja2_to_format(format_to_jinja2(original)) == original

    def test_literal_braces_round_trip(self) -> None:
        """Literal braces round-trip: format → jinja2 → format."""
        original = "JSON: {{}}"
        jinja2 = format_to_jinja2(original)
        back = jinja2_to_format(jinja2)
        assert back == original


# ============================================================================
# Integration: TemplateRenderer with template_syntax
# ============================================================================

class TestRendererSyntaxIntegration:

    def test_format_syntax_renders_correctly(self) -> None:
        """A PromptTemplateDict with template_syntax='format' renders correctly."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        prompt = {
            "template": "Hello {name}, welcome to {place}!",
            "template_syntax": "format",
        }
        result = renderer.render_prompt_template(prompt, {"name": "Alice", "place": "Wonderland"})
        assert result.content == "Hello Alice, welcome to Wonderland!"

    def test_jinja2_syntax_renders_correctly(self) -> None:
        """A PromptTemplateDict with template_syntax='jinja2' renders as-is."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        prompt = {
            "template": "Hello {{ name }}, welcome to {{ place }}!",
            "template_syntax": "jinja2",
        }
        result = renderer.render_prompt_template(prompt, {"name": "Alice", "place": "Wonderland"})
        assert result.content == "Hello Alice, welcome to Wonderland!"

    def test_no_syntax_annotation_renders_as_before(self) -> None:
        """Without template_syntax, behavior is unchanged — uses template_mode logic."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        # When no template_syntax is set, the template uses template_mode.
        # Explicitly set template_mode to jinja2 to get pure Jinja2 rendering.
        prompt = {
            "template": "Hello {{ name }}!",
            "template_mode": "jinja2",
        }
        result = renderer.render_prompt_template(prompt, {"name": "Alice"})
        assert result.content == "Hello Alice!"

    def test_format_with_literal_braces(self) -> None:
        """FORMAT template with literal braces (JSON example) renders correctly."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        prompt = {
            "template": "Return JSON: {{}}\nSchema: {schema}",
            "template_syntax": "format",
        }
        result = renderer.render_prompt_template(prompt, {"schema": "test_schema"})
        assert result.content == "Return JSON: {}\nSchema: test_schema"

    def test_format_extraction_prompt_pattern(self) -> None:
        """Test the exact pattern from the extraction prompts."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        prompt = {
            "template": (
                "Extract data.\n"
                "Schema: {schema}\n"
                "If empty, return {{}}\n"
                "Text: {text}"
            ),
            "template_syntax": "format",
        }
        result = renderer.render_prompt_template(
            prompt, {"schema": '{"type": "object"}', "text": "John is 30"}
        )
        assert 'Schema: {"type": "object"}' in result.content
        assert "return {}" in result.content
        assert "Text: John is 30" in result.content
