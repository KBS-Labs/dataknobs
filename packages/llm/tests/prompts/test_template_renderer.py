"""Unit tests for template renderer with validation."""

import pytest
from dataknobs_llm.prompts import (
    TemplateRenderer,
    ValidationLevel,
    ValidationConfig,
    render_template,
    render_template_strict,
)


class TestTemplateRenderer:
    """Test suite for TemplateRenderer class."""

    def test_basic_rendering(self):
        """Test basic template rendering with all parameters provided."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}",
            {"name": "Alice"}
        )
        assert result.content == "Hello Alice"
        assert result.params_used == {"name": "Alice"}
        assert result.params_missing == []

    def test_conditional_rendering_with_value(self):
        """Test conditional sections when variables have values."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}((, you are {{age}} years old))",
            {"name": "Alice", "age": 30}
        )
        assert result.content == "Hello Alice, you are 30 years old"

    def test_conditional_rendering_without_value(self):
        """Test conditional sections when variables are missing."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}((, you are {{age}} years old))",
            {"name": "Bob"}
        )
        assert result.content == "Hello Bob"

    def test_validation_error_level(self):
        """Test ERROR validation level raises exception for missing required params."""
        renderer = TemplateRenderer()
        validation = ValidationConfig(
            level=ValidationLevel.ERROR,
            required_params=["name", "age"]
        )

        with pytest.raises(ValueError, match="Missing required parameters: age"):
            renderer.render(
                "Hello {{name}}, age: {{age}}",
                {"name": "Alice"},
                validation=validation
            )

    def test_validation_warn_level(self):
        """Test WARN validation level logs warnings but continues."""
        renderer = TemplateRenderer()
        validation = ValidationConfig(
            level=ValidationLevel.WARN,
            required_params=["name", "age"]
        )

        result = renderer.render(
            "Hello {{name}}, age: {{age}}",
            {"name": "Alice"},
            validation=validation
        )

        assert result.content == "Hello Alice, age: {{age}}"
        assert result.params_missing == ["age"]
        assert len(result.validation_warnings) == 1
        assert "age" in result.validation_warnings[0]

    def test_validation_ignore_level(self):
        """Test IGNORE validation level silently ignores missing params."""
        renderer = TemplateRenderer()
        validation = ValidationConfig(
            level=ValidationLevel.IGNORE,
            required_params=["name", "age"]
        )

        result = renderer.render(
            "Hello {{name}}, age: {{age}}",
            {"name": "Alice"},
            validation=validation
        )

        assert result.content == "Hello Alice, age: {{age}}"
        assert result.params_missing == ["age"]
        assert result.validation_warnings == []

    def test_default_validation_level(self):
        """Test that default validation level is applied."""
        renderer = TemplateRenderer(default_validation=ValidationLevel.ERROR)

        # Should use ERROR level by default
        with pytest.raises(ValueError):
            renderer.render(
                "Hello {{name}}",
                {},
                validation=ValidationConfig(required_params=["name"])
            )

    def test_extract_variables(self):
        """Test variable extraction from templates."""
        variables = TemplateRenderer._extract_variables(
            "Hello {{name}}, you are {{age}} years old((, living in {{city}}))"
        )
        assert variables == {"name", "age", "city"}

    def test_extract_variables_with_whitespace(self):
        """Test variable extraction handles whitespace."""
        variables = TemplateRenderer._extract_variables(
            "Hello {{ name }}, age: {{age}}"
        )
        assert variables == {"name", "age"}

    def test_validate_template_syntax_valid(self):
        """Test syntax validation passes for valid templates."""
        errors = TemplateRenderer.validate_template_syntax(
            "Hello {{name}}((, you are {{age}} years old))"
        )
        assert errors == []

    def test_validate_template_syntax_unmatched_braces(self):
        """Test syntax validation catches unmatched braces."""
        errors = TemplateRenderer.validate_template_syntax(
            "Hello {name}"
        )
        assert len(errors) > 0
        assert "unmatched brace" in errors[0].lower()

    def test_validate_template_syntax_unmatched_conditionals(self):
        """Test syntax validation catches unmatched conditional sections."""
        errors = TemplateRenderer.validate_template_syntax(
            "Hello ((name"
        )
        assert len(errors) > 0
        # Check for either space or underscore version (new format uses underscore)
        assert "unmatched" in errors[0].lower() and "conditional" in errors[0].lower()

    def test_render_prompt_template_with_defaults(self):
        """Test rendering a PromptTemplateDict with default values."""
        renderer = TemplateRenderer()
        prompt_template = {
            "template": "Hello {{name}}, age: {{age}}",
            "defaults": {"age": 25},
        }

        result = renderer.render_prompt_template(
            prompt_template,
            {"name": "Alice"}
        )

        assert result.content == "Hello Alice, age: 25"

    def test_render_prompt_template_params_override_defaults(self):
        """Test that provided params override template defaults."""
        renderer = TemplateRenderer()
        prompt_template = {
            "template": "Hello {{name}}, age: {{age}}",
            "defaults": {"name": "Unknown", "age": 0},
        }

        result = renderer.render_prompt_template(
            prompt_template,
            {"name": "Bob", "age": 30}
        )

        assert result.content == "Hello Bob, age: 30"

    def test_render_prompt_template_with_validation_override(self):
        """Test runtime validation level override."""
        renderer = TemplateRenderer()
        prompt_template = {
            "template": "Hello {{name}}",
            "defaults": {},
            "validation": ValidationConfig(
                level=ValidationLevel.WARN,
                required_params=["name"]
            ),
        }

        # Override to ERROR level at runtime
        with pytest.raises(ValueError):
            renderer.render_prompt_template(
                prompt_template,
                {},
                validation_override=ValidationLevel.ERROR
            )

    def test_batch_render(self):
        """Test batch rendering of multiple templates."""
        renderer = TemplateRenderer()
        templates = [
            "Hello {{name}}",
            "Goodbye {{name}}",
            "Welcome {{name}}"
        ]
        params = {"name": "Alice"}

        results = renderer.batch_render(templates, params)

        assert len(results) == 3
        assert results[0].content == "Hello Alice"
        assert results[1].content == "Goodbye Alice"
        assert results[2].content == "Welcome Alice"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_render_template(self):
        """Test render_template convenience function."""
        result = render_template(
            "Hello {{name}}((, age {{age}}))",
            {"name": "Alice", "age": 30}
        )
        assert result == "Hello Alice, age 30"

    def test_render_template_with_validation_level(self):
        """Test render_template with custom validation level."""
        result = render_template(
            "Hello {{name}}",
            {"name": "Alice"},
            validation_level=ValidationLevel.IGNORE
        )
        assert result == "Hello Alice"

    def test_render_template_strict(self):
        """Test render_template_strict raises on missing params."""
        with pytest.raises(ValueError, match="Missing required parameters: age"):
            render_template_strict(
                "Hello {{name}}, age: {{age}}",
                {"name": "Alice"},
                required_params=["name", "age"]
            )

    def test_render_template_strict_success(self):
        """Test render_template_strict succeeds with all params."""
        result = render_template_strict(
            "Hello {{name}}, age: {{age}}",
            {"name": "Alice", "age": 30},
            required_params=["name", "age"]
        )
        assert result == "Hello Alice, age: 30"


class TestComplexScenarios:
    """Test suite for complex rendering scenarios."""

    def test_nested_conditionals(self):
        """Test nested conditional sections."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}((, from {{city}}((, {{country}}))))",
            {"name": "Alice", "city": "Paris", "country": "France"}
        )
        assert result.content == "Hello Alice, from Paris, France"

    def test_nested_conditionals_partial(self):
        """Test nested conditionals with partial data."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}((, from {{city}}((, {{country}}))))",
            {"name": "Alice", "city": "Paris"}
        )
        assert result.content == "Hello Alice, from Paris"

    def test_multiple_variables_in_conditional(self):
        """Test conditional with multiple variables.

        Note: If ANY variable in a conditional has a value, the conditional is kept.
        Missing variables inside conditionals become empty strings.
        """
        renderer = TemplateRenderer()
        template = "User: {{name}}((, {{age}} years old, from {{city}}))"

        # All variables present
        result1 = renderer.render(template, {"name": "Alice", "age": 30, "city": "NYC"})
        assert result1.content == "User: Alice, 30 years old, from NYC"

        # Some variables missing - conditional is kept because age has a value
        # Missing city becomes empty string
        result2 = renderer.render(template, {"name": "Bob", "age": 25})
        assert result2.content == "User: Bob, 25 years old, from "

        # All conditional variables missing - conditional is removed
        result3 = renderer.render(template, {"name": "Charlie"})
        assert result3.content == "User: Charlie"

    def test_whitespace_preservation(self):
        """Test that whitespace is preserved correctly."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{ name }}, age: {{ age }}",
            {"name": "Alice", "age": 30}
        )
        assert result.content == "Hello  Alice , age:  30 "

    def test_params_used_tracking(self):
        """Test that params_used correctly tracks which params were in template."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "Hello {{name}}",
            {"name": "Alice", "age": 30, "city": "NYC"}
        )
        # Only 'name' is in the template
        assert result.params_used == {"name": "Alice"}
        assert "age" not in result.params_used
        assert "city" not in result.params_used
