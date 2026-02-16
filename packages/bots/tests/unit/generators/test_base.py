"""Tests for generator base module: Generator ABC, GeneratorContext, GeneratorOutput."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.artifacts.provenance import ProvenanceRecord, create_provenance
from dataknobs_bots.generators.base import (
    Generator,
    GeneratorContext,
    GeneratorOutput,
)


# --- Concrete test implementation ---


class SimpleGenerator(Generator):
    """Minimal concrete generator for testing."""

    @property
    def id(self) -> str:
        return "test_gen"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameter_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["greeting"],
            "properties": {
                "greeting": {"type": "string"},
            },
        }

    async def generate(
        self,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        name = parameters["name"]
        count = parameters.get("count", 1)
        greeting = f"Hello {name}!" * count

        return GeneratorOutput(
            content={"greeting": greeting},
            provenance=create_provenance(
                created_by="system:generator:test_gen",
                creation_method="generator",
            ),
        )


# --- Generator ABC Tests ---


class TestGeneratorABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Generator()  # type: ignore[abstract]

    def test_concrete_implementation_properties(self) -> None:
        gen = SimpleGenerator()
        assert gen.id == "test_gen"
        assert gen.version == "1.0.0"
        assert "name" in gen.parameter_schema["required"]
        assert "greeting" in gen.output_schema["required"]

    async def test_generate_produces_output(self) -> None:
        gen = SimpleGenerator()
        output = await gen.generate({"name": "World"})

        assert output.content == {"greeting": "Hello World!"}
        assert output.provenance.created_by == "system:generator:test_gen"
        assert output.provenance.creation_method == "generator"

    async def test_generate_with_optional_parameter(self) -> None:
        gen = SimpleGenerator()
        output = await gen.generate({"name": "World", "count": 3})

        assert output.content == {"greeting": "Hello World!Hello World!Hello World!"}


# --- GeneratorContext Tests ---


class TestGeneratorContext:
    def test_default_context(self) -> None:
        ctx = GeneratorContext()
        assert ctx.db is None
        assert ctx.llm is None
        assert ctx.vector_store is None
        assert ctx.artifact_registry is None
        assert ctx.config == {}

    def test_context_with_dependencies(self) -> None:
        ctx = GeneratorContext(
            db="mock_db",
            llm="mock_llm",
            config={"key": "value"},
        )
        assert ctx.db == "mock_db"
        assert ctx.llm == "mock_llm"
        assert ctx.config == {"key": "value"}


# --- GeneratorOutput Tests ---


class TestGeneratorOutput:
    def test_create_output(self) -> None:
        provenance = create_provenance("system", "generator")
        output = GeneratorOutput(
            content={"key": "value"},
            provenance=provenance,
        )
        assert output.content == {"key": "value"}
        assert output.validation_errors == []
        assert output.metadata == {}

    def test_output_with_errors(self) -> None:
        provenance = create_provenance("system", "generator")
        output = GeneratorOutput(
            content={"key": "value"},
            provenance=provenance,
            validation_errors=["missing required field 'title'"],
        )
        assert len(output.validation_errors) == 1

    def test_output_serialization_roundtrip(self) -> None:
        provenance = create_provenance(
            "system:generator:test",
            "generator",
            creation_context={"param": "val"},
        )
        output = GeneratorOutput(
            content={"greeting": "hello"},
            provenance=provenance,
            validation_errors=["warning1"],
            metadata={"gen_id": "test"},
        )

        data = output.to_dict()
        assert data["content"] == {"greeting": "hello"}
        assert data["validation_errors"] == ["warning1"]
        assert data["metadata"] == {"gen_id": "test"}
        assert data["provenance"]["created_by"] == "system:generator:test"

        restored = GeneratorOutput.from_dict(data)
        assert restored.content == output.content
        assert restored.validation_errors == output.validation_errors
        assert restored.metadata == output.metadata
        assert restored.provenance.created_by == output.provenance.created_by


# --- Parameter Validation Tests ---


class TestParameterValidation:
    async def test_valid_parameters(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_parameters({"name": "World"})
        assert errors == []

    async def test_missing_required_parameter(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_parameters({})
        assert len(errors) == 1
        assert "name" in errors[0]

    async def test_wrong_type_parameter(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_parameters({"name": "World", "count": "not_int"})
        assert len(errors) == 1
        assert "not_int" in errors[0] or "string" in errors[0] or "type" in errors[0].lower()

    async def test_constraint_violation(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_parameters({"name": "World", "count": 0})
        assert len(errors) == 1


# --- Output Validation Tests ---


class TestOutputValidation:
    async def test_valid_output(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_output({"greeting": "Hello"})
        assert errors == []

    async def test_missing_required_output_field(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_output({})
        assert len(errors) == 1
        assert "greeting" in errors[0]

    async def test_wrong_type_output_field(self) -> None:
        gen = SimpleGenerator()
        errors = await gen.validate_output({"greeting": 123})
        assert len(errors) == 1
