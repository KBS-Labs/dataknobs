"""Tests for improved template validation error messages."""

import pytest
from dataknobs_llm.prompts import (
    TemplateRenderer,
    TemplateSyntaxError
)


class TestImprovedValidationErrors:
    """Test improved error messages with line/column information."""

    def test_unmatched_opening_brace_single_line(self):
        """Test error message for unmatched braces on single line."""
        template = "Hello {name}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        # Both { and } are unmatched, so we get 2 errors
        assert len(errors) == 2
        assert all(isinstance(error, TemplateSyntaxError) for error in errors)
        assert all(error.error_type == "unmatched_brace" for error in errors)
        assert all(error.line == 1 for error in errors)

        # Check that both braces are reported
        columns = [error.column for error in errors]
        assert 7 in columns  # Opening brace
        assert 12 in columns  # Closing brace

        # Check snippets contain error markers
        assert all("⮜HERE⮞" in error.snippet for error in errors)

    def test_unmatched_closing_brace_multi_line(self):
        """Test error message for unmatched closing brace on multiple lines."""
        template = """Line 1
Line 2
Line 3 with } error
Line 4"""

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "unmatched_brace"
        assert error.line == 3  # Third line
        assert "⮜HERE⮞" in error.snippet

    def test_unmatched_conditional_opening(self):
        """Test error for opening (( without closing ))."""
        template = "Hello {{name}}((, age {{age}}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "unmatched_conditional"
        assert "Opening '((' without matching closing '))'." in error.message
        assert error.line == 1
        assert "⮜HERE⮞" in error.snippet

    def test_unmatched_conditional_closing(self):
        """Test error for closing )) without opening ((."""
        template = "Hello {{name}}, age {{age}}))"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "unmatched_conditional"
        assert "Closing ')) without matching opening '(('." in error.message

    def test_malformed_variable_special_chars(self):
        """Test error for variable with special characters."""
        template = "Hello {{user-name}}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "malformed_variable"
        assert "Malformed variable" in error.message
        assert "user-name" in error.message
        assert "letters, numbers, and underscores" in error.message

    def test_empty_variable(self):
        """Test error for empty variable {{}}."""
        template = "Hello {{}}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "malformed_variable"
        assert "Empty variable" in error.message
        assert error.line == 1

    def test_multiple_errors_different_types(self):
        """Test template with multiple different errors."""
        template = "Hello {name with {{age.foo}} and (( unclosed"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        # Should have at least unmatched brace, malformed variable, and unclosed conditional
        assert len(errors) >= 3

        error_types = {error.error_type for error in errors}
        assert "unmatched_brace" in error_types
        assert "malformed_variable" in error_types
        assert "unmatched_conditional" in error_types

    def test_error_string_formatting(self):
        """Test that error string formatting includes location."""
        template = "Bad {variable"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        error_str = str(errors[0])
        assert "line 1" in error_str
        assert "column" in error_str
        assert "⮜HERE⮞" in error_str

    def test_multiline_error_location(self):
        """Test line/column calculation for multiline templates."""
        template = """First line
Second line
Third line with {{invalid-var}}
Fourth line"""

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.line == 3  # Third line
        assert error.error_type == "malformed_variable"

    def test_snippet_context_size(self):
        """Test that snippet shows reasonable context around error."""
        long_text = "A" * 50
        template = f"{long_text}{{{{bad-var}}}}{long_text}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        error = errors[0]
        # Snippet should be limited (not show entire long_text)
        assert len(error.snippet) < len(template)
        # But should show the error
        assert "bad-var" in error.snippet or "⮜HERE⮞" in error.snippet

    def test_backward_compatible_validate_syntax(self):
        """Test that old validate_template_syntax still works."""
        template = "Bad {variable"

        errors = TemplateRenderer.validate_template_syntax(template)

        # Should return list of strings
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert isinstance(errors[0], str)

        # String should contain useful information
        error_str = errors[0]
        assert "line" in error_str.lower()
        assert "column" in error_str.lower()

    def test_nested_conditionals_error(self):
        """Test error detection in nested conditionals."""
        template = "Hello ((name is {{name}} ((and age is {{age}}))"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        # Missing one closing ))
        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "unmatched_conditional"

    def test_valid_template_no_errors(self):
        """Test that valid templates produce no errors."""
        valid_templates = [
            "Hello {{name}}",
            "Hello {{name}}((, you are {{age}} years old))",
            "{{greeting}} {{name}}!",
            "Text with ((optional {{var}}))",
            "Multiple {{var1}} and {{var2}}",
        ]

        for template in valid_templates:
            errors = TemplateRenderer.validate_template_syntax_detailed(template)
            assert len(errors) == 0, f"Valid template should have no errors: {template}"

    def test_complex_multiline_template(self):
        """Test error detection in complex multiline template."""
        template = """# Header

This is {{valid_var}}.

But this has {error} here.

And this has {{bad-var}} too.

Finally ((unclosed conditional"""

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        # Should find: unmatched brace, malformed variable, unclosed conditional
        assert len(errors) >= 3

        # Check line numbers are correct
        error_lines = {error.line for error in errors}
        assert 5 in error_lines  # {error} line
        assert 7 in error_lines  # {{bad-var}} line

    def test_error_snippet_replaces_newlines(self):
        """Test that newlines in snippets are escaped."""
        template = """Line 1 with error {
Line 2"""

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        error = errors[0]
        # Newlines should be replaced with \n for display
        assert "\\n" in error.snippet

    def test_variable_with_spaces(self):
        """Test error for variable with spaces."""
        template = "Hello {{ user name }}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == "malformed_variable"
        assert "user name" in error.message.lower() or "malformed" in error.message.lower()

    def test_consecutive_errors(self):
        """Test handling of consecutive errors."""
        template = "{{bad-1}} {{bad-2}} {{bad-3}}"

        errors = TemplateRenderer.validate_template_syntax_detailed(template)

        assert len(errors) == 3
        # All should be malformed variables
        assert all(e.error_type == "malformed_variable" for e in errors)

        # Line numbers should all be 1
        assert all(e.line == 1 for e in errors)

        # Column numbers should be different
        columns = [e.column for e in errors]
        assert len(set(columns)) == 3  # All different


class TestErrorDataclass:
    """Test TemplateSyntaxError dataclass."""

    def test_create_error_object(self):
        """Test creating TemplateSyntaxError object."""
        error = TemplateSyntaxError(
            message="Test error",
            line=10,
            column=5,
            snippet="some ⮜HERE⮞ text",
            error_type="test_error"
        )

        assert error.message == "Test error"
        assert error.line == 10
        assert error.column == 5
        assert error.snippet == "some ⮜HERE⮞ text"
        assert error.error_type == "test_error"

    def test_error_str_representation(self):
        """Test string representation of error."""
        error = TemplateSyntaxError(
            message="Something went wrong",
            line=5,
            column=12,
            snippet="context ⮜HERE⮞ text",
            error_type="syntax_error"
        )

        error_str = str(error)

        assert "line 5" in error_str
        assert "column 12" in error_str
        assert "Something went wrong" in error_str
        assert "⮜HERE⮞" in error_str
        assert "syntax_error" in error_str
