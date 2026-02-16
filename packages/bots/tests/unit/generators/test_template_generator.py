"""Tests for TemplateGenerator."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.generators.template_generator import TemplateGenerator


# --- Fixtures ---

SIMPLE_TEMPLATE = """\
greeting: Hello {{ name }}!
message: Welcome to {{ place }}.
"""

PARAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "place"],
    "properties": {
        "name": {"type": "string"},
        "place": {"type": "string"},
    },
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["greeting", "message"],
    "properties": {
        "greeting": {"type": "string"},
        "message": {"type": "string"},
    },
}

JSON_TEMPLATE = '{"greeting": "Hello {{ name }}!", "count": {{ count }}}'

JSON_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["greeting", "count"],
    "properties": {
        "greeting": {"type": "string"},
        "count": {"type": "integer"},
    },
}


def _make_template_generator(
    template: str = SIMPLE_TEMPLATE,
    param_schema: dict[str, Any] = PARAM_SCHEMA,
    output_schema: dict[str, Any] = OUTPUT_SCHEMA,
    output_format: str = "yaml",
    generator_id: str = "test_tpl",
    version: str = "1.0.0",
) -> TemplateGenerator:
    return TemplateGenerator(
        generator_id=generator_id,
        version=version,
        template=template,
        parameter_schema=param_schema,
        output_schema=output_schema,
        output_format=output_format,
    )


# --- Generation Tests ---


class TestTemplateGeneration:
    async def test_simple_yaml_generation(self) -> None:
        gen = _make_template_generator()
        output = await gen.generate({"name": "Alice", "place": "Wonderland"})

        assert output.content["greeting"] == "Hello Alice!"
        assert output.content["message"] == "Welcome to Wonderland."
        assert output.validation_errors == []

    async def test_json_output_format(self) -> None:
        gen = _make_template_generator(
            template=JSON_TEMPLATE,
            param_schema={
                "type": "object",
                "required": ["name", "count"],
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
            output_schema=JSON_OUTPUT_SCHEMA,
            output_format="json",
        )
        output = await gen.generate({"name": "Bob", "count": 5})

        assert output.content["greeting"] == "Hello Bob!"
        assert output.content["count"] == 5

    async def test_parameter_validation_failure_raises(self) -> None:
        gen = _make_template_generator()

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await gen.generate({"name": "Alice"})  # missing "place"

    async def test_output_validation_errors_recorded(self) -> None:
        bad_output_template = "greeting: Hello {{ name }}!"
        gen = _make_template_generator(
            template=bad_output_template,
            param_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            output_schema=OUTPUT_SCHEMA,  # requires "message" too
        )
        output = await gen.generate({"name": "Alice"})

        assert output.content["greeting"] == "Hello Alice!"
        assert len(output.validation_errors) > 0
        assert "message" in output.validation_errors[0]


# --- Provenance Tests ---


class TestTemplateProvenance:
    async def test_provenance_records_generator_info(self) -> None:
        gen = _make_template_generator(generator_id="my_gen", version="2.0.0")
        output = await gen.generate({"name": "Alice", "place": "Wonderland"})

        prov = output.provenance
        assert prov.created_by == "system:generator:my_gen"
        assert prov.creation_method == "generator"
        assert prov.creation_context["generator_id"] == "my_gen"
        assert prov.creation_context["generator_version"] == "2.0.0"
        assert prov.creation_context["parameters"] == {
            "name": "Alice",
            "place": "Wonderland",
        }

    async def test_provenance_includes_tool_chain(self) -> None:
        gen = _make_template_generator(generator_id="quiz_gen", version="1.1.0")
        output = await gen.generate({"name": "Bob", "place": "School"})

        assert len(output.provenance.tool_chain) == 1
        tool = output.provenance.tool_chain[0]
        assert tool.tool_name == "generator:quiz_gen"
        assert tool.tool_version == "1.1.0"
        assert tool.parameters == {"name": "Bob", "place": "School"}

    async def test_metadata_includes_generator_info(self) -> None:
        gen = _make_template_generator(generator_id="content_gen", version="3.0.0")
        output = await gen.generate({"name": "Charlie", "place": "Office"})

        assert output.metadata["generator_id"] == "content_gen"
        assert output.metadata["generator_version"] == "3.0.0"
        assert output.metadata["output_format"] == "yaml"


# --- Template Rendering Edge Cases ---


class TestTemplateRendering:
    async def test_template_with_loop(self) -> None:
        template = """\
items:
{% for item in items %}  - name: {{ item }}
{% endfor %}"""
        gen = _make_template_generator(
            template=template,
            param_schema={
                "type": "object",
                "required": ["items"],
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            output_schema={
                "type": "object",
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            },
        )
        output = await gen.generate({"items": ["apple", "banana"]})

        assert len(output.content["items"]) == 2
        assert output.content["items"][0]["name"] == "apple"
        assert output.content["items"][1]["name"] == "banana"

    async def test_missing_template_variable_raises(self) -> None:
        gen = _make_template_generator(
            template="greeting: Hello {{ missing_var }}!",
            param_schema={"type": "object", "properties": {}},
        )

        with pytest.raises(ValueError, match="missing_var"):
            await gen.generate({})

    async def test_yaml_parse_failure_raises(self) -> None:
        gen = _make_template_generator(
            template="{{ raw_content }}",
            param_schema={
                "type": "object",
                "required": ["raw_content"],
                "properties": {"raw_content": {"type": "string"}},
            },
        )

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            await gen.generate({"raw_content": "just a plain string"})

    async def test_json_parse_failure_raises(self) -> None:
        gen = _make_template_generator(
            template="not valid json {{ name }}",
            param_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            output_format="json",
        )

        with pytest.raises(ValueError, match="Failed to parse template output as JSON"):
            await gen.generate({"name": "test"})


# --- From Config Tests ---


class TestTemplateGeneratorFromConfig:
    def test_from_config_creates_generator(self) -> None:
        config = {
            "id": "cfg_gen",
            "version": "1.0.0",
            "template": "result: {{ value }}",
            "parameter_schema": {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        }
        gen = TemplateGenerator.from_config(config)

        assert gen.id == "cfg_gen"
        assert gen.version == "1.0.0"

    def test_from_config_with_output_format(self) -> None:
        config = {
            "id": "json_gen",
            "version": "1.0.0",
            "template": '{"result": "{{ value }}"}',
            "parameter_schema": {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
            "output_format": "json",
        }
        gen = TemplateGenerator.from_config(config)
        assert gen._output_format == "json"

    def test_from_config_missing_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required keys"):
            TemplateGenerator.from_config({"id": "incomplete"})

    async def test_from_config_generates_correctly(self) -> None:
        config = {
            "id": "cfg_gen",
            "version": "1.0.0",
            "template": "title: {{ title }}\nauthor: {{ author }}",
            "parameter_schema": {
                "type": "object",
                "required": ["title", "author"],
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                },
            },
            "output_schema": {
                "type": "object",
                "required": ["title", "author"],
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                },
            },
        }
        gen = TemplateGenerator.from_config(config)
        output = await gen.generate({"title": "My Book", "author": "Jane"})

        assert output.content["title"] == "My Book"
        assert output.content["author"] == "Jane"
