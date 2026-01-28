"""Tests for template variable substitution (enhancement 2a).

Tests the substitute_template_vars function for recursively substituting
{{var}} placeholders in nested configuration structures.
"""

import pytest

from dataknobs_config.template_vars import substitute_template_vars


class TestSubstituteSimpleValues:
    """Tests for simple string substitution."""

    def test_substitute_simple_string(self) -> None:
        """Test simple string substitution."""
        data = "Hello {{name}}"
        variables = {"name": "World"}
        result = substitute_template_vars(data, variables)
        assert result == "Hello World"

    def test_substitute_multiple_placeholders(self) -> None:
        """Test multiple placeholders in one string."""
        data = "{{greeting}}, {{name}}!"
        variables = {"greeting": "Hello", "name": "World"}
        result = substitute_template_vars(data, variables)
        assert result == "Hello, World!"

    def test_substitute_repeated_placeholder(self) -> None:
        """Test same placeholder appearing multiple times."""
        data = "{{name}} and {{name}} again"
        variables = {"name": "Test"}
        result = substitute_template_vars(data, variables)
        assert result == "Test and Test again"


class TestSubstituteNestedStructures:
    """Tests for nested dict and list structures."""

    def test_substitute_nested_dict(self) -> None:
        """Test substitution in nested dict structure."""
        data = {
            "level1": {
                "level2": "{{value}}",
                "list": ["{{item1}}", "{{item2}}"],
            }
        }
        variables = {"value": "nested", "item1": "a", "item2": "b"}
        result = substitute_template_vars(data, variables)
        assert result == {
            "level1": {
                "level2": "nested",
                "list": ["a", "b"],
            }
        }

    def test_substitute_list_of_dicts(self) -> None:
        """Test substitution in list of dicts."""
        data = [{"name": "{{name1}}"}, {"name": "{{name2}}"}]
        variables = {"name1": "Alice", "name2": "Bob"}
        result = substitute_template_vars(data, variables)
        assert result == [{"name": "Alice"}, {"name": "Bob"}]

    def test_deeply_nested_structure(self) -> None:
        """Test substitution in deeply nested structures."""
        data = {"a": {"b": {"c": ["{{x}}", {"d": "{{y}}"}]}}}
        variables = {"x": 1, "y": 2}
        result = substitute_template_vars(data, variables)
        assert result["a"]["b"]["c"][0] == 1
        assert result["a"]["b"]["c"][1]["d"] == 2


class TestTypePreservation:
    """Tests for type preservation with entire-value placeholders."""

    def test_type_preservation_int(self) -> None:
        """Test that entire-value placeholders preserve int type."""
        data = {"count": "{{count}}"}
        variables = {"count": 42}
        result = substitute_template_vars(data, variables)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_type_preservation_float(self) -> None:
        """Test that entire-value placeholders preserve float type."""
        data = {"rate": "{{rate}}"}
        variables = {"rate": 3.14}
        result = substitute_template_vars(data, variables)
        assert result["rate"] == 3.14
        assert isinstance(result["rate"], float)

    def test_type_preservation_bool(self) -> None:
        """Test that entire-value placeholders preserve bool type."""
        data = {"enabled": "{{enabled}}"}
        variables = {"enabled": True}
        result = substitute_template_vars(data, variables)
        assert result["enabled"] is True

    def test_type_preservation_false(self) -> None:
        """Test that False is preserved (not converted to empty string)."""
        data = {"disabled": "{{disabled}}"}
        variables = {"disabled": False}
        result = substitute_template_vars(data, variables)
        assert result["disabled"] is False

    def test_type_preservation_list(self) -> None:
        """Test that entire-value placeholders preserve list type."""
        data = {"items": "{{items}}"}
        variables = {"items": ["a", "b", "c"]}
        result = substitute_template_vars(data, variables)
        assert result["items"] == ["a", "b", "c"]
        assert isinstance(result["items"], list)

    def test_type_preservation_dict(self) -> None:
        """Test that entire-value placeholders preserve dict type."""
        data = {"config": "{{config}}"}
        variables = {"config": {"key": "value", "nested": {"x": 1}}}
        result = substitute_template_vars(data, variables)
        assert result["config"] == {"key": "value", "nested": {"x": 1}}
        assert isinstance(result["config"], dict)

    def test_type_preservation_zero(self) -> None:
        """Test that zero is preserved (not treated as missing)."""
        data = {"zero": "{{zero}}"}
        variables = {"zero": 0}
        result = substitute_template_vars(data, variables)
        assert result["zero"] == 0
        assert isinstance(result["zero"], int)

    def test_type_preservation_disabled(self) -> None:
        """Test that type_cast=False returns strings."""
        data = {"count": "{{count}}"}
        variables = {"count": 42}
        result = substitute_template_vars(data, variables, type_cast=False)
        assert result["count"] == "42"
        assert isinstance(result["count"], str)

    def test_mixed_content_returns_string(self) -> None:
        """Test that mixed placeholder/text always returns string."""
        data = "Has {{count}} items"
        variables = {"count": 10}
        result = substitute_template_vars(data, variables)
        assert result == "Has 10 items"
        assert isinstance(result, str)


class TestMissingVariables:
    """Tests for handling missing variables."""

    def test_preserve_missing_default(self) -> None:
        """Test that missing variables are preserved by default."""
        data = "Hello {{name}}, you have {{count}} items"
        variables = {"name": "Alice"}
        result = substitute_template_vars(data, variables)
        assert result == "Hello Alice, you have {{count}} items"

    def test_preserve_missing_entire_value(self) -> None:
        """Test that entire-value missing placeholders are preserved."""
        data = {"value": "{{missing}}"}
        variables = {}
        result = substitute_template_vars(data, variables)
        assert result["value"] == "{{missing}}"

    def test_preserve_missing_false(self) -> None:
        """Test that missing variables become empty when preserve_missing=False."""
        data = "Hello {{name}}"
        variables = {}
        result = substitute_template_vars(data, variables, preserve_missing=False)
        assert result == "Hello "

    def test_preserve_missing_false_entire_value(self) -> None:
        """Test entire-value placeholder with preserve_missing=False."""
        data = {"value": "{{missing}}"}
        variables = {}
        result = substitute_template_vars(data, variables, preserve_missing=False)
        assert result["value"] == ""


class TestNoneValues:
    """Tests for handling None values."""

    def test_none_value_in_mixed_content(self) -> None:
        """Test handling of None values in mixed content."""
        data = "Value: {{maybe}}"
        variables = {"maybe": None}
        result = substitute_template_vars(data, variables)
        assert result == "Value: "

    def test_entire_value_none(self) -> None:
        """Test entire-value placeholder with None."""
        data = {"value": "{{maybe}}"}
        variables = {"maybe": None}
        result = substitute_template_vars(data, variables)
        assert result["value"] is None

    def test_entire_value_none_type_cast_false(self) -> None:
        """Test entire-value None with type_cast=False."""
        data = {"value": "{{maybe}}"}
        variables = {"maybe": None}
        result = substitute_template_vars(data, variables, type_cast=False)
        assert result["value"] == ""


class TestWhitespaceHandling:
    """Tests for whitespace handling in placeholders."""

    def test_whitespace_in_placeholder(self) -> None:
        """Test that whitespace inside {{}} is handled."""
        data = "{{ name }}"
        variables = {"name": "test"}
        result = substitute_template_vars(data, variables)
        assert result == "test"

    def test_whitespace_around_entire_placeholder(self) -> None:
        """Test whitespace around entire-value placeholder."""
        data = {"value": "  {{ count }}  "}
        variables = {"count": 42}
        result = substitute_template_vars(data, variables)
        # Whitespace-padded entire value still gets type preservation
        assert result["value"] == 42

    def test_multiple_with_whitespace(self) -> None:
        """Test multiple placeholders with internal whitespace."""
        data = "{{ a }} and {{  b  }}"
        variables = {"a": "first", "b": "second"}
        result = substitute_template_vars(data, variables)
        assert result == "first and second"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dict(self) -> None:
        """Test with empty dict."""
        result = substitute_template_vars({}, {"name": "test"})
        assert result == {}

    def test_empty_list(self) -> None:
        """Test with empty list."""
        result = substitute_template_vars([], {"name": "test"})
        assert result == []

    def test_empty_string(self) -> None:
        """Test with empty string."""
        result = substitute_template_vars("", {"name": "test"})
        assert result == ""

    def test_no_placeholders(self) -> None:
        """Test string with no placeholders."""
        result = substitute_template_vars("Hello World", {"name": "test"})
        assert result == "Hello World"

    def test_primitive_passthrough_int(self) -> None:
        """Test that int values pass through unchanged."""
        result = substitute_template_vars(42, {"name": "test"})
        assert result == 42

    def test_primitive_passthrough_bool(self) -> None:
        """Test that bool values pass through unchanged."""
        result = substitute_template_vars(True, {"name": "test"})
        assert result is True

    def test_primitive_passthrough_none(self) -> None:
        """Test that None passes through unchanged."""
        result = substitute_template_vars(None, {"name": "test"})
        assert result is None

    def test_original_data_unchanged(self) -> None:
        """Test that original data structure is not modified."""
        data = {"value": "{{x}}", "nested": {"inner": "{{y}}"}}
        original_value = data["value"]
        original_nested = data["nested"]["inner"]
        substitute_template_vars(data, {"x": "replaced", "y": "also"})
        assert data["value"] == original_value  # Original unchanged
        assert data["nested"]["inner"] == original_nested

    def test_empty_variables_dict(self) -> None:
        """Test with empty variables dict."""
        data = "Hello {{name}}"
        result = substitute_template_vars(data, {})
        assert result == "Hello {{name}}"  # Preserved by default


class TestImportFromPackage:
    """Test that the function is properly exported from the package."""

    def test_import_from_package(self) -> None:
        """Test import from main package."""
        from dataknobs_config import substitute_template_vars as pkg_func

        result = pkg_func({"x": "{{y}}"}, {"y": 42})
        assert result == {"x": 42}
