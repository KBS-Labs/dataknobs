"""Tests for GeneratorRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase

from dataknobs_bots.artifacts.provenance import create_provenance
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.generators.base import Generator, GeneratorContext, GeneratorOutput
from dataknobs_bots.generators.registry import GeneratorRegistry
from dataknobs_bots.generators.template_generator import TemplateGenerator


# --- Test generator ---


class CountGenerator(Generator):
    """Simple generator that creates content with a count."""

    @property
    def id(self) -> str:
        return "count_gen"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameter_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["count"],
            "properties": {
                "count": {"type": "integer", "minimum": 1},
            },
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

    async def generate(
        self,
        parameters: dict[str, Any],
        context: GeneratorContext | None = None,
    ) -> GeneratorOutput:
        count = parameters["count"]
        return GeneratorOutput(
            content={"items": [f"item_{i}" for i in range(count)]},
            provenance=create_provenance(
                created_by="system:generator:count_gen",
                creation_method="generator",
            ),
            metadata={"artifact_type": "generated_content"},
        )


# --- Registration and Retrieval Tests ---


class TestGeneratorRegistration:
    async def test_register_and_get(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)
        gen = CountGenerator()

        gen_id = await registry.register(gen)

        assert gen_id == "count_gen"
        retrieved = await registry.get("count_gen")
        assert retrieved is gen

    async def test_get_nonexistent_returns_none(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        result = await registry.get("nonexistent")
        assert result is None

    async def test_list_all_generators(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        gen = CountGenerator()
        await registry.register(gen)

        ids = await registry.list_all()
        assert ids == ["count_gen"]

    async def test_list_all_empty(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        ids = await registry.list_all()
        assert ids == []

    async def test_metadata_stored_in_database(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)
        gen = CountGenerator()

        await registry.register(gen)

        record = await db.read("gen:count_gen")
        assert record is not None
        assert record.data["generator_id"] == "count_gen"
        assert record.data["version"] == "1.0.0"
        assert "parameter_schema" in record.data
        assert "output_schema" in record.data


# --- Generation Tests ---


class TestGeneratorRegistryGenerate:
    async def test_generate_with_registered_generator(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)
        gen = CountGenerator()
        await registry.register(gen)

        output = await registry.generate("count_gen", {"count": 3})

        assert output.content["items"] == ["item_0", "item_1", "item_2"]

    async def test_generate_nonexistent_raises(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        with pytest.raises(ValueError, match="not found"):
            await registry.generate("nonexistent", {"count": 1})

    async def test_generate_invalid_params_raises(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)
        gen = CountGenerator()
        await registry.register(gen)

        with pytest.raises(ValueError, match="Parameter validation failed"):
            await registry.generate("count_gen", {"count": 0})

    async def test_generate_with_template_generator(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        tpl_gen = TemplateGenerator(
            generator_id="greeting_gen",
            version="1.0.0",
            template="greeting: Hello {{ name }}!",
            parameter_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "required": ["greeting"],
                "properties": {"greeting": {"type": "string"}},
            },
        )
        await registry.register(tpl_gen)

        output = await registry.generate("greeting_gen", {"name": "World"})

        assert output.content["greeting"] == "Hello World!"


# --- Artifact Integration Tests ---


class TestGeneratorRegistryArtifactIntegration:
    async def test_generate_creates_artifact(self) -> None:
        gen_db = AsyncMemoryDatabase()
        art_db = AsyncMemoryDatabase()
        art_registry = ArtifactRegistry(art_db)

        registry = GeneratorRegistry(gen_db, artifact_registry=art_registry)
        gen = CountGenerator()
        await registry.register(gen)

        output = await registry.generate("count_gen", {"count": 2})

        assert "artifact_id" in output.metadata
        artifact_id = output.metadata["artifact_id"]

        artifact = await art_registry.get(artifact_id)
        assert artifact is not None
        assert artifact.content == {"items": ["item_0", "item_1"]}
        assert artifact.type == "generated_content"
        assert artifact.provenance.created_by == "system:generator:count_gen"

    async def test_generate_without_artifact_registry(self) -> None:
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db, artifact_registry=None)
        gen = CountGenerator()
        await registry.register(gen)

        output = await registry.generate("count_gen", {"count": 1})

        assert "artifact_id" not in output.metadata
        assert output.content["items"] == ["item_0"]

    async def test_generate_with_validation_errors_skips_artifact(self) -> None:
        gen_db = AsyncMemoryDatabase()
        art_db = AsyncMemoryDatabase()
        art_registry = ArtifactRegistry(art_db)

        # Template that produces output missing a required field
        tpl_gen = TemplateGenerator(
            generator_id="bad_gen",
            version="1.0.0",
            template="partial: Hello {{ name }}!",
            parameter_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "required": ["partial", "missing_field"],
                "properties": {
                    "partial": {"type": "string"},
                    "missing_field": {"type": "string"},
                },
            },
        )

        registry = GeneratorRegistry(gen_db, artifact_registry=art_registry)
        await registry.register(tpl_gen)

        output = await registry.generate("bad_gen", {"name": "World"})

        # Output has validation errors, so no artifact should be created
        assert len(output.validation_errors) > 0
        assert "artifact_id" not in output.metadata


# --- From Config Tests ---


class TestGeneratorRegistryFromConfig:
    async def test_from_config_loads_template_generators(self) -> None:
        db = AsyncMemoryDatabase()
        config = {
            "generators": [
                {
                    "type": "template",
                    "id": "greeting_gen",
                    "version": "1.0.0",
                    "template": "greeting: Hello {{ name }}!",
                    "parameter_schema": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {"name": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"greeting": {"type": "string"}},
                    },
                },
            ],
        }

        registry = await GeneratorRegistry.from_config(config, db)

        gen = await registry.get("greeting_gen")
        assert gen is not None
        assert gen.id == "greeting_gen"

        ids = await registry.list_all()
        assert "greeting_gen" in ids

    async def test_from_config_multiple_generators(self) -> None:
        db = AsyncMemoryDatabase()
        config = {
            "generators": [
                {
                    "type": "template",
                    "id": "gen_a",
                    "version": "1.0.0",
                    "template": "result: {{ val }}",
                    "parameter_schema": {
                        "type": "object",
                        "required": ["val"],
                        "properties": {"val": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                },
                {
                    "type": "template",
                    "id": "gen_b",
                    "version": "2.0.0",
                    "template": "output: {{ data }}",
                    "parameter_schema": {
                        "type": "object",
                        "required": ["data"],
                        "properties": {"data": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"output": {"type": "string"}},
                    },
                },
            ],
        }

        registry = await GeneratorRegistry.from_config(config, db)

        ids = await registry.list_all()
        assert sorted(ids) == ["gen_a", "gen_b"]

    async def test_from_config_empty(self) -> None:
        db = AsyncMemoryDatabase()
        registry = await GeneratorRegistry.from_config({}, db)

        ids = await registry.list_all()
        assert ids == []

    async def test_from_config_with_artifact_registry(self) -> None:
        gen_db = AsyncMemoryDatabase()
        art_db = AsyncMemoryDatabase()
        art_registry = ArtifactRegistry(art_db)

        config = {
            "generators": [
                {
                    "type": "template",
                    "id": "art_gen",
                    "version": "1.0.0",
                    "template": "content: {{ text }}",
                    "parameter_schema": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {"text": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                },
            ],
        }

        registry = await GeneratorRegistry.from_config(
            config, gen_db, artifact_registry=art_registry
        )
        output = await registry.generate("art_gen", {"text": "hello"})

        assert "artifact_id" in output.metadata

    async def test_from_config_skips_unknown_type(self) -> None:
        db = AsyncMemoryDatabase()
        config = {
            "generators": [
                {
                    "type": "unknown_custom_type",
                    "id": "custom_gen",
                },
            ],
        }

        registry = await GeneratorRegistry.from_config(config, db)

        ids = await registry.list_all()
        assert ids == []
