"""Tests for Jinja2 integration with template renderer.

This module tests:
- Jinja2 filters (upper, truncate, default, etc.)
- Jinja2 conditionals ({% if/elif/else %})
- Mixed mode (both (( )) and Jinja2 syntax)
- Validation (no Jinja2 inside (( )) blocks)
- Template mode configuration
- Custom filters
- Loops and advanced features
- Backward compatibility
"""

import pytest
from dataknobs_llm.prompts.rendering.template_renderer import TemplateRenderer
from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary
from dataknobs_llm.prompts.base.types import TemplateMode, ValidationLevel


class TestJinja2Filters:
    """Test Jinja2 built-in and custom filters."""

    def test_upper_filter(self):
        """Test uppercase filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name|upper}}",
            {"name": "alice"},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "ALICE"

    def test_lower_filter(self):
        """Test lowercase filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name|lower}}",
            {"name": "ALICE"},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "alice"

    def test_capitalize_filter(self):
        """Test capitalize filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name|capitalize}}",
            {"name": "alice"},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "Alice"

    def test_default_filter(self):
        """Test default filter with missing value."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name|default('Guest')}}",
            {},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "Guest"

    def test_length_filter(self):
        """Test length filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{items|length}}",
            {"items": [1, 2, 3, 4, 5]},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "5"

    def test_join_filter(self):
        """Test join filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{items|join(', ')}}",
            {"items": ["apple", "banana", "cherry"]},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "apple, banana, cherry"

    def test_custom_filter(self):
        """Test registering and using custom filter."""
        renderer = TemplateRenderer()

        # Register custom filter
        renderer.add_custom_filter('double', lambda x: x * 2)

        result = renderer.render(
            "{{count|double}}",
            {"count": 5},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "10"

    def test_builtin_format_code_filter(self):
        """Test built-in format_code filter."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{code|format_code('python')}}",
            {"code": "print('hello')"},
            mode=TemplateMode.JINJA2
        )

        assert "```python" in result.content
        assert "print('hello')" in result.content
        assert "```" in result.content

    def test_chained_filters(self):
        """Test chaining multiple filters."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name|lower|capitalize}}",
            {"name": "ALICE"},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "Alice"


class TestJinja2Conditionals:
    """Test Jinja2 {% if %} conditionals."""

    def test_simple_if(self):
        """Test simple if block."""
        renderer = TemplateRenderer()

        template = "{% if age >= 18 %}Adult{% endif %}"

        result1 = renderer.render(template, {"age": 20}, mode=TemplateMode.JINJA2)
        assert result1.content.strip() == "Adult"

        result2 = renderer.render(template, {"age": 15}, mode=TemplateMode.JINJA2)
        assert result2.content.strip() == ""

    def test_if_else(self):
        """Test if/else blocks."""
        renderer = TemplateRenderer()

        template = "{% if age >= 18 %}Adult{% else %}Minor{% endif %}"

        result1 = renderer.render(template, {"age": 20}, mode=TemplateMode.JINJA2)
        assert result1.content.strip() == "Adult"

        result2 = renderer.render(template, {"age": 15}, mode=TemplateMode.JINJA2)
        assert result2.content.strip() == "Minor"

    def test_if_elif_else(self):
        """Test if/elif/else blocks."""
        renderer = TemplateRenderer()

        template = """
        {% if age < 13 %}
        Child
        {% elif age < 18 %}
        Teen
        {% else %}
        Adult
        {% endif %}
        """

        result1 = renderer.render(template, {"age": 10}, mode=TemplateMode.JINJA2)
        assert "Child" in result1.content

        result2 = renderer.render(template, {"age": 15}, mode=TemplateMode.JINJA2)
        assert "Teen" in result2.content

        result3 = renderer.render(template, {"age": 25}, mode=TemplateMode.JINJA2)
        assert "Adult" in result3.content

    def test_boolean_operators(self):
        """Test boolean operators in conditionals."""
        renderer = TemplateRenderer()

        # AND
        template = "{% if verified and age >= 18 %}Approved{% endif %}"
        result1 = renderer.render(template, {"verified": True, "age": 20}, mode=TemplateMode.JINJA2)
        assert "Approved" in result1.content

        result2 = renderer.render(template, {"verified": False, "age": 20}, mode=TemplateMode.JINJA2)
        assert "Approved" not in result2.content

        # OR
        template2 = "{% if admin or moderator %}Access{% endif %}"
        result3 = renderer.render(template2, {"admin": False, "moderator": True}, mode=TemplateMode.JINJA2)
        assert "Access" in result3.content


class TestMixedMode:
    """Test mixed mode: both (( )) and Jinja2 syntax."""

    def test_filters_outside_conditionals(self):
        """Test filters outside (( )) blocks work in mixed mode."""
        renderer = TemplateRenderer()

        # Filters outside (( )) - should work
        template = "{{name|upper}}((, age {{age}}))"

        result1 = renderer.render(
            template,
            {"name": "alice", "age": 30},
            mode=TemplateMode.MIXED
        )
        assert result1.content == "ALICE, age 30"

        result2 = renderer.render(
            template,
            {"name": "alice", "age": None},
            mode=TemplateMode.MIXED
        )
        assert result2.content == "ALICE"

    def test_jinja_blocks_outside_conditionals(self):
        """Test Jinja2 {% %} blocks outside (( )) work in mixed mode."""
        renderer = TemplateRenderer()

        template = "{{name}}((, age {{age}})){% if verified %} ✓{% endif %}"

        result = renderer.render(
            template,
            {"name": "Alice", "verified": True},
            mode=TemplateMode.MIXED
        )

        assert "Alice" in result.content
        assert "✓" in result.content
        assert "age" not in result.content  # Conditional removed (age is None)

    def test_mixed_conditionals_and_jinja_if(self):
        """Test combining (( )) and {% if %} in same template."""
        renderer = TemplateRenderer()

        template = """
        Hello {{name}}((, from {{city}}))
        {% if premium %}⭐ Premium member{% endif %}
        """

        result = renderer.render(
            template,
            {"name": "Alice", "city": "NYC", "premium": True},
            mode=TemplateMode.MIXED
        )

        assert "Alice" in result.content
        assert "NYC" in result.content
        assert "⭐ Premium" in result.content


class TestSyntaxValidation:
    """Test validation of Jinja2 syntax inside (( )) blocks."""

    def test_filter_inside_conditional_raises_error(self):
        """Test that filters inside (( )) raise error in mixed mode."""
        renderer = TemplateRenderer()

        # Filters inside (( )) - should error
        template = "((Hello {{name|upper}}))"

        with pytest.raises(ValueError, match="filters.*not allowed inside"):
            renderer.render(
                template,
                {"name": "alice"},
                mode=TemplateMode.MIXED
            )

    def test_jinja_block_inside_conditional_raises_error(self):
        """Test that {% %} blocks inside (( )) raise error in mixed mode."""
        renderer = TemplateRenderer()

        # {% %} inside (( )) - should error
        template = "(({% if age %}age {{age}}{% endif %}))"

        with pytest.raises(ValueError, match="block syntax.*not allowed inside"):
            renderer.render(
                template,
                {"age": 30},
                mode=TemplateMode.MIXED
            )

    def test_filters_allowed_in_jinja2_mode(self):
        """Test that filters work anywhere in pure Jinja2 mode."""
        renderer = TemplateRenderer()

        # In JINJA2 mode, no (( )) preprocessing, so this is just text
        template = "Hello {{name|upper}}"

        result = renderer.render(
            template,
            {"name": "alice"},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "Hello ALICE"


class TestTemplateMode:
    """Test template mode configuration."""

    def test_default_mode_is_mixed(self):
        """Test that default mode is MIXED."""
        renderer = TemplateRenderer()
        assert renderer._default_mode == TemplateMode.MIXED

    def test_explicit_jinja2_mode(self):
        """Test explicit JINJA2 mode."""
        renderer = TemplateRenderer(default_mode=TemplateMode.JINJA2)
        assert renderer._default_mode == TemplateMode.JINJA2

    def test_mode_from_config(self):
        """Test template mode specified in config."""
        config = {
            "system": {
                "modern": {
                    "template": "{{name|upper}}",
                    "template_mode": "jinja2"
                },
                "legacy": {
                    "template": "(({{name}}))",
                    "template_mode": "mixed"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # Modern template uses Jinja2 mode
        modern = library.get_system_prompt("modern")
        assert modern is not None
        assert modern["template_mode"] == "jinja2"

        # Legacy template uses mixed mode
        legacy = library.get_system_prompt("legacy")
        assert legacy is not None
        assert legacy["template_mode"] == "mixed"

    def test_mode_override_in_render(self):
        """Test mode override at render time."""
        config = {
            "system": {
                "test": {
                    "template": "{{name}}",
                    "template_mode": "mixed"
                }
            }
        }

        library = ConfigPromptLibrary(config)
        renderer = TemplateRenderer()

        template = library.get_system_prompt("test")

        # Override to JINJA2 mode at render time
        result = renderer.render_prompt_template(
            template,
            {"name": "Alice"},
            mode_override=TemplateMode.JINJA2
        )

        assert result.metadata["template_mode"] == "jinja2"


class TestJinja2Loops:
    """Test Jinja2 {% for %} loops."""

    def test_simple_for_loop(self):
        """Test simple for loop."""
        renderer = TemplateRenderer()

        template = """
        {% for item in items %}
        {{ loop.index }}. {{ item }}
        {% endfor %}
        """

        result = renderer.render(
            template,
            {"items": ["apple", "banana", "cherry"]},
            mode=TemplateMode.JINJA2
        )

        assert "1. apple" in result.content
        assert "2. banana" in result.content
        assert "3. cherry" in result.content

    def test_loop_with_object_properties(self):
        """Test loop accessing object properties."""
        renderer = TemplateRenderer()

        template = """
        {% for user in users %}
        - {{ user.name }} ({{ user.role }})
        {% endfor %}
        """

        result = renderer.render(
            template,
            {
                "users": [
                    {"name": "Alice", "role": "admin"},
                    {"name": "Bob", "role": "user"}
                ]
            },
            mode=TemplateMode.JINJA2
        )

        assert "Alice (admin)" in result.content
        assert "Bob (user)" in result.content

    def test_loop_with_conditional(self):
        """Test loop with conditional inside."""
        renderer = TemplateRenderer()

        template = """
        {% for user in users %}
        {{ user.name }}{% if user.premium %} ⭐{% endif %}
        {% endfor %}
        """

        result = renderer.render(
            template,
            {
                "users": [
                    {"name": "Alice", "premium": True},
                    {"name": "Bob", "premium": False}
                ]
            },
            mode=TemplateMode.JINJA2
        )

        assert "Alice ⭐" in result.content
        assert "Bob ⭐" not in result.content


class TestBackwardCompatibility:
    """Test that existing (( )) templates still work."""

    def test_existing_conditional_syntax(self):
        """Test that (( )) syntax still works in mixed mode."""
        renderer = TemplateRenderer()

        # Old template
        template = "Hello {{name}}((, age {{age}}))"

        # With age
        result1 = renderer.render(
            template,
            {"name": "Alice", "age": 30},
            mode=TemplateMode.MIXED
        )
        assert result1.content == "Hello Alice, age 30"

        # Without age
        result2 = renderer.render(
            template,
            {"name": "Bob"},
            mode=TemplateMode.MIXED
        )
        assert result2.content == "Hello Bob"

    def test_nested_conditionals(self):
        """Test nested (( )) conditionals."""
        renderer = TemplateRenderer()

        template = "{{name}}((, from {{city}}((, {{country}}))))"

        # All params
        result1 = renderer.render(
            template,
            {"name": "Alice", "city": "NYC", "country": "USA"},
            mode=TemplateMode.MIXED
        )
        assert result1.content == "Alice, from NYC, USA"

        # Missing country
        result2 = renderer.render(
            template,
            {"name": "Alice", "city": "NYC"},
            mode=TemplateMode.MIXED
        )
        assert result2.content == "Alice, from NYC"

        # Missing city and country
        result3 = renderer.render(
            template,
            {"name": "Alice"},
            mode=TemplateMode.MIXED
        )
        assert result3.content == "Alice"

    def test_default_mode_backward_compatible(self):
        """Test that default renderer mode is backward compatible."""
        renderer = TemplateRenderer()  # No mode specified

        # Should default to MIXED mode
        template = "{{name}}((, age {{age}}))"
        result = renderer.render(template, {"name": "Alice"})

        assert result.content == "Alice"
        assert result.metadata["template_mode"] == "mixed"


class TestRenderResultMetadata:
    """Test render result metadata includes mode information."""

    def test_metadata_includes_mode(self):
        """Test that render result includes template mode."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name}}",
            {"name": "Alice"},
            mode=TemplateMode.JINJA2
        )

        assert "template_mode" in result.metadata
        assert result.metadata["template_mode"] == "jinja2"

    def test_metadata_with_mixed_mode(self):
        """Test metadata with mixed mode."""
        renderer = TemplateRenderer()

        result = renderer.render(
            "{{name}}",
            {"name": "Alice"},
            mode=TemplateMode.MIXED
        )

        assert result.metadata["template_mode"] == "mixed"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_jinja2_syntax_error(self):
        """Test that Jinja2 syntax errors are caught and reported."""
        renderer = TemplateRenderer()

        # Invalid Jinja2 syntax
        template = "{% if age %} Missing endif"

        with pytest.raises(ValueError, match="Template syntax error"):
            renderer.render(
                template,
                {"age": 30},
                mode=TemplateMode.JINJA2
            )

    def test_undefined_variable_in_filter(self):
        """Test undefined variable in filter."""
        renderer = TemplateRenderer()

        # Jinja2 by default returns empty string for undefined variables
        result = renderer.render(
            "{{undefined_var|default('N/A')}}",
            {},
            mode=TemplateMode.JINJA2
        )

        assert result.content == "N/A"

    def test_empty_template(self):
        """Test rendering empty template."""
        renderer = TemplateRenderer()

        result = renderer.render("", {}, mode=TemplateMode.JINJA2)
        assert result.content == ""

    def test_template_with_only_whitespace(self):
        """Test template with only whitespace."""
        renderer = TemplateRenderer()

        result = renderer.render("   \n  \n  ", {}, mode=TemplateMode.JINJA2)
        assert result.content.strip() == ""
