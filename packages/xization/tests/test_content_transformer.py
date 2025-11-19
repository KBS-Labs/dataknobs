"""Tests for the ContentTransformer class."""

import tempfile
from pathlib import Path

import pytest

from dataknobs_xization import (
    ContentTransformer,
    csv_to_markdown,
    json_to_markdown,
    yaml_to_markdown,
)


class TestContentTransformerInit:
    """Test ContentTransformer initialization."""

    def test_default_init(self):
        """Test default initialization values."""
        transformer = ContentTransformer()
        assert transformer.base_heading_level == 2
        assert transformer.include_field_labels is True
        assert "example" in transformer.code_block_fields
        assert "steps" in transformer.list_fields
        assert transformer.schemas == {}

    def test_custom_init(self):
        """Test custom initialization values."""
        transformer = ContentTransformer(
            base_heading_level=3,
            include_field_labels=False,
            code_block_fields=["custom_code"],
            list_fields=["custom_list"],
        )
        assert transformer.base_heading_level == 3
        assert transformer.include_field_labels is False
        assert "custom_code" in transformer.code_block_fields
        assert "custom_list" in transformer.list_fields


class TestSchemaRegistration:
    """Test schema registration functionality."""

    def test_register_schema(self):
        """Test registering a custom schema."""
        transformer = ContentTransformer()
        schema = {
            "title_field": "name",
            "sections": [
                {"field": "description", "heading": "Description"},
            ],
        }
        transformer.register_schema("test", schema)
        assert "test" in transformer.schemas
        assert transformer.schemas["test"] == schema

    def test_register_multiple_schemas(self):
        """Test registering multiple schemas."""
        transformer = ContentTransformer()
        transformer.register_schema("schema1", {"title_field": "name"})
        transformer.register_schema("schema2", {"title_field": "title"})
        assert len(transformer.schemas) == 2
        assert "schema1" in transformer.schemas
        assert "schema2" in transformer.schemas


class TestJSONTransformation:
    """Test JSON to markdown transformation."""

    def test_simple_dict(self):
        """Test transforming a simple dictionary."""
        transformer = ContentTransformer()
        data = {"name": "Test", "description": "A test item"}
        result = transformer.transform_json(data)

        assert "## Test" in result
        assert "**Description**: A test item" in result

    def test_dict_with_title(self):
        """Test transforming with document title."""
        transformer = ContentTransformer()
        data = {"name": "Item", "value": "123"}
        result = transformer.transform_json(data, title="My Document")

        assert "# My Document" in result
        assert "## Item" in result

    def test_list_of_dicts(self):
        """Test transforming a list of dictionaries."""
        transformer = ContentTransformer()
        data = [
            {"name": "First", "value": "1"},
            {"name": "Second", "value": "2"},
        ]
        result = transformer.transform_json(data)

        assert "## First" in result
        assert "## Second" in result
        assert result.count("---") == 2

    def test_nested_dict(self):
        """Test transforming nested dictionaries."""
        transformer = ContentTransformer()
        data = {
            "name": "Parent",
            "nested": {
                "child_key": "child_value",
            },
        }
        result = transformer.transform_json(data)

        assert "## Parent" in result
        assert "### Nested" in result
        assert "child_value" in result

    def test_list_field(self):
        """Test transforming a list field as bullet points."""
        transformer = ContentTransformer()
        data = {
            "name": "Task",
            "steps": ["Step 1", "Step 2", "Step 3"],
        }
        result = transformer.transform_json(data)

        assert "### Steps" in result
        assert "- Step 1" in result
        assert "- Step 2" in result
        assert "- Step 3" in result

    def test_code_block_field(self):
        """Test transforming a code block field."""
        transformer = ContentTransformer()
        data = {
            "name": "Example",
            "example": "print('hello')",
        }
        result = transformer.transform_json(data)

        assert "### Example" in result
        assert "```" in result
        assert "print('hello')" in result

    def test_empty_values_skipped(self):
        """Test that empty and None values are skipped."""
        transformer = ContentTransformer()
        data = {
            "name": "Test",
            "empty": "",
            "null": None,
            "valid": "value",
        }
        result = transformer.transform_json(data)

        assert "**Valid**: value" in result
        assert "Empty" not in result
        assert "Null" not in result

    def test_custom_heading_level(self):
        """Test custom base heading level."""
        transformer = ContentTransformer(base_heading_level=3)
        data = {"name": "Test", "description": "Desc"}
        result = transformer.transform_json(data)

        assert "### Test" in result

    def test_without_field_labels(self):
        """Test transformation without field labels."""
        transformer = ContentTransformer(include_field_labels=False)
        data = {"name": "Test", "description": "A description"}
        result = transformer.transform_json(data)

        # Should have value without bold label
        assert "A description" in result
        assert "**Description**:" not in result


class TestSchemaBasedTransformation:
    """Test transformation using custom schemas."""

    def test_basic_schema(self):
        """Test basic schema transformation."""
        transformer = ContentTransformer()
        transformer.register_schema("pattern", {
            "title_field": "name",
            "description_field": "description",
            "sections": [
                {"field": "use_case", "heading": "When to Use"},
            ],
        })

        data = {
            "name": "Chain of Thought",
            "description": "Step by step reasoning",
            "use_case": "Complex problems",
        }
        result = transformer.transform_json(data, schema="pattern")

        assert "## Chain of Thought" in result
        assert "Step by step reasoning" in result
        assert "### When to Use" in result
        assert "Complex problems" in result

    def test_schema_with_code_format(self):
        """Test schema with code format sections."""
        transformer = ContentTransformer()
        transformer.register_schema("example", {
            "title_field": "name",
            "sections": [
                {"field": "code", "heading": "Code", "format": "code", "language": "python"},
            ],
        })

        data = {
            "name": "Sample",
            "code": "def hello():\n    print('hi')",
        }
        result = transformer.transform_json(data, schema="example")

        assert "```python" in result
        assert "def hello():" in result

    def test_schema_with_list_format(self):
        """Test schema with list format sections."""
        transformer = ContentTransformer()
        transformer.register_schema("task", {
            "title_field": "name",
            "sections": [
                {"field": "items", "heading": "Items", "format": "list"},
            ],
        })

        data = {
            "name": "Shopping",
            "items": ["Milk", "Eggs", "Bread"],
        }
        result = transformer.transform_json(data, schema="task")

        assert "### Items" in result
        assert "- Milk" in result
        assert "- Eggs" in result

    def test_schema_with_metadata_fields(self):
        """Test schema with metadata fields."""
        transformer = ContentTransformer()
        transformer.register_schema("pattern", {
            "title_field": "name",
            "metadata_fields": ["category", "difficulty"],
        })

        data = {
            "name": "Test",
            "category": "prompting",
            "difficulty": "easy",
        }
        result = transformer.transform_json(data, schema="pattern")

        assert "**Category**: prompting" in result
        assert "**Difficulty**: easy" in result

    def test_schema_with_subsections(self):
        """Test schema with subsections format."""
        transformer = ContentTransformer()
        transformer.register_schema("config", {
            "title_field": "name",
            "sections": [
                {"field": "settings", "heading": "Settings", "format": "subsections"},
            ],
        })

        data = {
            "name": "App Config",
            "settings": {
                "theme": "dark",
                "language": "en",
            },
        }
        result = transformer.transform_json(data, schema="config")

        assert "### Settings" in result
        assert "**Theme**: dark" in result
        assert "**Language**: en" in result

    def test_list_of_items_with_schema(self):
        """Test transforming a list of items with schema."""
        transformer = ContentTransformer()
        transformer.register_schema("pattern", {
            "title_field": "name",
            "description_field": "description",
        })

        data = [
            {"name": "First", "description": "First desc"},
            {"name": "Second", "description": "Second desc"},
        ]
        result = transformer.transform_json(data, schema="pattern")

        assert "## First" in result
        assert "## Second" in result
        assert "First desc" in result
        assert "Second desc" in result


class TestYAMLTransformation:
    """Test YAML to markdown transformation."""

    def test_yaml_string(self):
        """Test transforming YAML string."""
        transformer = ContentTransformer()
        yaml_content = """
name: Test Item
description: A test description
"""
        result = transformer.transform_yaml(yaml_content)

        assert "## Test Item" in result
        assert "A test description" in result

    def test_yaml_file(self, tmp_path):
        """Test transforming YAML file."""
        transformer = ContentTransformer()

        # Create temp YAML file
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("name: FromFile\nvalue: 123")

        result = transformer.transform_yaml(yaml_file)

        assert "## FromFile" in result
        assert "**Value**: 123" in result

    def test_yaml_with_schema(self):
        """Test YAML transformation with schema."""
        transformer = ContentTransformer()
        transformer.register_schema("item", {"title_field": "name"})

        yaml_content = "name: Schema Test\ndetails: Some details"
        result = transformer.transform_yaml(yaml_content, schema="item")

        assert "## Schema Test" in result


class TestCSVTransformation:
    """Test CSV to markdown transformation."""

    def test_csv_string(self):
        """Test transforming CSV string."""
        transformer = ContentTransformer()
        csv_content = "name,value,description\nItem1,100,First item\nItem2,200,Second item"
        result = transformer.transform_csv(csv_content)

        assert "## Item1" in result
        assert "## Item2" in result
        assert "**Value**: 100" in result
        assert "**Description**: First item" in result

    def test_csv_file(self, tmp_path):
        """Test transforming CSV file."""
        transformer = ContentTransformer()

        # Create temp CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nFromFile,999")

        result = transformer.transform_csv(csv_file)

        assert "## FromFile" in result
        assert "**Value**: 999" in result

    def test_csv_with_title_field(self):
        """Test CSV transformation with custom title field."""
        transformer = ContentTransformer()
        csv_content = "id,title,desc\n1,Custom Title,Description"
        result = transformer.transform_csv(csv_content, title_field="title")

        assert "## Custom Title" in result
        assert "**Id**: 1" in result

    def test_csv_with_document_title(self):
        """Test CSV transformation with document title."""
        transformer = ContentTransformer()
        csv_content = "name,value\nItem,100"
        result = transformer.transform_csv(csv_content, title="My Data")

        assert "# My Data" in result

    def test_empty_csv(self):
        """Test transforming empty CSV."""
        transformer = ContentTransformer()
        csv_content = "name,value"  # Headers only
        result = transformer.transform_csv(csv_content)

        # Should handle gracefully
        assert result == "" or "---" not in result


class TestTransformMethod:
    """Test the generic transform method."""

    def test_transform_json_format(self):
        """Test transform with JSON format."""
        transformer = ContentTransformer()
        data = {"name": "Test", "value": "123"}
        result = transformer.transform(data, format="json")

        assert "## Test" in result

    def test_transform_yaml_format(self):
        """Test transform with YAML format."""
        transformer = ContentTransformer()
        content = "name: Test"
        result = transformer.transform(content, format="yaml")

        assert "## Test" in result

    def test_transform_csv_format(self):
        """Test transform with CSV format."""
        transformer = ContentTransformer()
        content = "name,value\nTest,123"
        result = transformer.transform(content, format="csv")

        assert "## Test" in result

    def test_transform_invalid_format(self):
        """Test transform with invalid format raises error."""
        transformer = ContentTransformer()

        with pytest.raises(ValueError, match="Unsupported format"):
            transformer.transform({}, format="xml")

    def test_transform_json_from_file(self, tmp_path):
        """Test transform JSON from file path."""
        transformer = ContentTransformer()

        # Create temp JSON file
        json_file = tmp_path / "test.json"
        json_file.write_text('{"name": "FromJSON", "value": 42}')

        result = transformer.transform(json_file, format="json")

        assert "## FromJSON" in result


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_json_to_markdown(self):
        """Test json_to_markdown function."""
        data = {"name": "Quick", "desc": "Test"}
        result = json_to_markdown(data)

        assert "## Quick" in result

    def test_json_to_markdown_with_options(self):
        """Test json_to_markdown with options."""
        data = {"name": "Test"}
        result = json_to_markdown(data, title="Doc", base_heading_level=3)

        assert "# Doc" in result
        assert "### Test" in result

    def test_yaml_to_markdown(self):
        """Test yaml_to_markdown function."""
        content = "name: YAMLTest"
        result = yaml_to_markdown(content)

        assert "## YAMLTest" in result

    def test_csv_to_markdown(self):
        """Test csv_to_markdown function."""
        content = "name,value\nCSVTest,100"
        result = csv_to_markdown(content)

        assert "## CSVTest" in result


class TestFieldNameFormatting:
    """Test field name formatting."""

    def test_snake_case_formatting(self):
        """Test snake_case field name formatting."""
        transformer = ContentTransformer()
        data = {"name": "Test", "some_field_name": "value"}
        result = transformer.transform_json(data)

        assert "**Some Field Name**:" in result

    def test_camel_case_formatting(self):
        """Test camelCase field name formatting."""
        transformer = ContentTransformer()
        data = {"name": "Test", "someFieldName": "value"}
        result = transformer.transform_json(data)

        # Should split camelCase
        assert "Field" in result

    def test_hyphenated_formatting(self):
        """Test hyphenated field name formatting."""
        transformer = ContentTransformer()
        data = {"name": "Test", "some-field": "value"}
        result = transformer.transform_json(data)

        assert "Some Field" in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_deeply_nested_structure(self):
        """Test deeply nested data structure."""
        transformer = ContentTransformer()
        data = {
            "name": "Root",
            "level1": {
                "level2": {
                    "level3": "deep value",
                },
            },
        }
        result = transformer.transform_json(data)

        assert "## Root" in result
        assert "deep value" in result

    def test_mixed_list_items(self):
        """Test list with mixed item types."""
        transformer = ContentTransformer()
        data = {
            "name": "Mixed",
            "items": [
                {"name": "Dict Item", "description": "A dict"},
                "String item",
            ],
        }
        result = transformer.transform_json(data)

        assert "Dict Item" in result or "dict" in result.lower()
        assert "String item" in result

    def test_special_characters_in_values(self):
        """Test special characters in values."""
        transformer = ContentTransformer()
        data = {
            "name": "Special *chars* `code` [link]",
            "value": "Line1\nLine2",
        }
        result = transformer.transform_json(data)

        # Should preserve special characters
        assert "*chars*" in result or "chars" in result

    def test_unicode_content(self):
        """Test unicode content."""
        transformer = ContentTransformer()
        data = {
            "name": "Unicode Test",
            "content": "日本語 中文 한국어 emoji: 🎉",
        }
        result = transformer.transform_json(data)

        assert "日本語" in result
        assert "🎉" in result

    def test_empty_dict(self):
        """Test empty dictionary."""
        transformer = ContentTransformer()
        result = transformer.transform_json({})

        # Should not raise error
        assert isinstance(result, str)

    def test_empty_list(self):
        """Test empty list."""
        transformer = ContentTransformer()
        result = transformer.transform_json([])

        # Should not raise error
        assert isinstance(result, str)

    def test_schema_not_found(self):
        """Test that non-existent schema falls back to generic."""
        transformer = ContentTransformer()
        data = {"name": "Test"}
        result = transformer.transform_json(data, schema="nonexistent")

        # Should use generic transformation
        assert "## Test" in result

    def test_list_with_complex_dicts(self):
        """Test list field with complex dict items."""
        transformer = ContentTransformer()
        data = {
            "name": "Complex",
            "items": [
                {"name": "Item 1", "description": "First"},
                {"name": "Item 2", "description": "Second"},
            ],
        }
        result = transformer.transform_json(data)

        assert "Item 1" in result
        assert "Item 2" in result
