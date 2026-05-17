"""Tests for GeneratorRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.artifacts.provenance import create_provenance
from dataknobs_bots.artifacts.registry import ArtifactRegistry
from dataknobs_bots.generators.base import (
    Generator,
    GeneratorContext,
    GeneratorDefinition,
    GeneratorOutput,
)
from dataknobs_bots.generators.registry import GeneratorRegistry
from dataknobs_bots.generators.template_generator import TemplateGenerator
from dataknobs_data import SortOrder, SortSpec
from dataknobs_data.backends.memory import AsyncMemoryDatabase

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


# --- Metadata channel routing ---


class TestGeneratorRegistryMetadata:
    """Metadata channel routes through ``record.metadata``.

    The old ``register()`` had a shadow bug where a local variable named
    ``metadata`` was passed positionally as ``Record(metadata)``, landing
    in the ``data`` column.  Structural fields stayed in ``data``
    (incidentally correct), but the variable name was a footgun and
    there was no real metadata channel for cross-cutting filters.
    """

    async def test_structural_fields_route_to_data_column(self) -> None:
        """``generator_id`` / ``version`` / schemas live in the data column."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(CountGenerator())

        raw = await db.read("gen:count_gen")
        assert raw is not None
        assert raw.data["generator_id"] == "count_gen"
        assert raw.data["version"] == "1.0.0"
        assert "parameter_schema" in raw.data
        assert "output_schema" in raw.data

    async def test_metadata_defaults_empty_metadata_column(self) -> None:
        """No ``metadata=`` kwarg ⇒ empty metadata column, no leakage."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(CountGenerator())

        raw = await db.read("gen:count_gen")
        assert raw is not None
        assert raw.metadata == {}
        # And nothing named "metadata" should have leaked into data.
        assert "metadata" not in raw.data

    async def test_metadata_kwarg_routes_to_metadata_column(self) -> None:
        """``register(..., metadata=...)`` lands in the metadata column."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(
            CountGenerator(),
            metadata={"tenant_id": "acme", "author": "alice"},
        )

        raw = await db.read("gen:count_gen")
        assert raw is not None
        assert raw.metadata == {"tenant_id": "acme", "author": "alice"}
        # Structural fields untouched.
        assert raw.data["generator_id"] == "count_gen"
        # Metadata is NOT duplicated into the data column.
        assert "tenant_id" not in raw.data
        assert "author" not in raw.data

    async def test_get_definition_round_trip(self) -> None:
        """``get_definition`` reconstructs the persisted snapshot."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(
            CountGenerator(),
            metadata={"tenant_id": "acme"},
        )

        defn = await registry.get_definition("count_gen")
        assert defn is not None
        assert isinstance(defn, GeneratorDefinition)
        assert defn.generator_id == "count_gen"
        assert defn.version == "1.0.0"
        assert defn.metadata == {"tenant_id": "acme"}

    async def test_get_definition_missing(self) -> None:
        """``get_definition`` returns ``None`` for unregistered ids."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        assert await registry.get_definition("nonexistent") is None

    async def test_list_definitions_filter_metadata(self) -> None:
        """``filter_metadata`` selects only matching definitions."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        # Two generators that share an id intentionally would collide; use
        # template generators with distinct ids so we can vary metadata.
        gen_a = TemplateGenerator(
            generator_id="gen_a",
            version="1.0.0",
            template="r: {{ v }}",
            parameter_schema={
                "type": "object",
                "required": ["v"],
                "properties": {"v": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {"r": {"type": "string"}},
            },
        )
        gen_b = TemplateGenerator(
            generator_id="gen_b",
            version="1.0.0",
            template="r: {{ v }}",
            parameter_schema={
                "type": "object",
                "required": ["v"],
                "properties": {"v": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {"r": {"type": "string"}},
            },
        )

        await registry.register(gen_a, metadata={"tenant_id": "acme"})
        await registry.register(gen_b, metadata={"tenant_id": "globex"})

        acme = await registry.list_definitions(filter_metadata={"tenant_id": "acme"})
        assert {d.generator_id for d in acme} == {"gen_a"}

        globex = await registry.list_definitions(filter_metadata={"tenant_id": "globex"})
        assert {d.generator_id for d in globex} == {"gen_b"}

    async def test_list_definitions_empty_filter_metadata_is_no_filter(
        self,
    ) -> None:
        """``filter_metadata={}`` ≡ ``filter_metadata=None``."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(CountGenerator(), metadata={"tenant_id": "acme"})

        with_empty = await registry.list_definitions(filter_metadata={})
        with_none = await registry.list_definitions()
        assert {d.generator_id for d in with_empty} == {d.generator_id for d in with_none}


# --- Pagination + count ---


def _make_template_gen(
    generator_id: str, version: str = "1.0.0"
) -> TemplateGenerator:
    """Build a minimal :class:`TemplateGenerator` with a unique id.

    The schemas are kept trivial because pagination tests only care
    about ``generator_id`` ordering and counts, not generation behavior.
    """
    return TemplateGenerator(
        generator_id=generator_id,
        version=version,
        template="r: {{ v }}",
        parameter_schema={
            "type": "object",
            "required": ["v"],
            "properties": {"v": {"type": "string"}},
        },
        output_schema={
            "type": "object",
            "properties": {"r": {"type": "string"}},
        },
    )


class TestGeneratorRegistryPagination:
    """Tests for ``list_definitions()`` sort/limit/offset.

    Unlike the dual-write registries (artifacts, rubrics) where
    pagination must be applied post-dedup, ``GeneratorRegistry`` writes
    a single row per generator id, so sort/limit/offset can be pushed
    all the way to the database.  These tests pin that pushdown
    behavior end-to-end against the memory backend.
    """

    async def test_list_definitions_sort_by_generator_id_asc(self) -> None:
        """Sort by generator_id ascending returns definitions in id order."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(_make_template_gen("gen_c"))
        await registry.register(_make_template_gen("gen_a"))
        await registry.register(_make_template_gen("gen_b"))

        results = await registry.list_definitions(
            sort=[SortSpec(field="generator_id", order=SortOrder.ASC)],
        )
        assert [d.generator_id for d in results] == ["gen_a", "gen_b", "gen_c"]

    async def test_list_definitions_sort_by_generator_id_desc(self) -> None:
        """Sort by generator_id descending reverses the order."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(_make_template_gen("gen_c"))
        await registry.register(_make_template_gen("gen_a"))
        await registry.register(_make_template_gen("gen_b"))

        results = await registry.list_definitions(
            sort=[SortSpec(field="generator_id", order=SortOrder.DESC)],
        )
        assert [d.generator_id for d in results] == ["gen_c", "gen_b", "gen_a"]

    async def test_list_definitions_limit(self) -> None:
        """``limit`` caps the number of definitions returned."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        for gid in ["gen_a", "gen_b", "gen_c"]:
            await registry.register(_make_template_gen(gid))

        results = await registry.list_definitions(
            sort=[SortSpec(field="generator_id", order=SortOrder.ASC)],
            limit=2,
        )
        assert len(results) == 2
        assert [d.generator_id for d in results] == ["gen_a", "gen_b"]

    async def test_list_definitions_offset(self) -> None:
        """``offset`` skips the first N definitions."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        for gid in ["gen_a", "gen_b", "gen_c"]:
            await registry.register(_make_template_gen(gid))

        results = await registry.list_definitions(
            sort=[SortSpec(field="generator_id", order=SortOrder.ASC)],
            offset=1,
        )
        assert [d.generator_id for d in results] == ["gen_b", "gen_c"]

    async def test_list_definitions_limit_and_offset_combine(self) -> None:
        """``offset`` first, then ``limit`` — standard pagination semantics."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        for gid in ["gen_a", "gen_b", "gen_c", "gen_d", "gen_e"]:
            await registry.register(_make_template_gen(gid))

        page = await registry.list_definitions(
            sort=[SortSpec(field="generator_id", order=SortOrder.ASC)],
            offset=1,
            limit=2,
        )
        assert [d.generator_id for d in page] == ["gen_b", "gen_c"]

    async def test_list_definitions_sort_combines_with_filter_metadata(
        self,
    ) -> None:
        """Sort and ``filter_metadata`` compose: filter first, then sort."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(
            _make_template_gen("gen_c"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_a"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_b"), metadata={"tenant_id": "globex"}
        )

        results = await registry.list_definitions(
            filter_metadata={"tenant_id": "acme"},
            sort=[SortSpec(field="generator_id", order=SortOrder.ASC)],
        )
        assert [d.generator_id for d in results] == ["gen_a", "gen_c"]

    async def test_list_definitions_limit_zero_returns_empty(self) -> None:
        """``limit=0`` returns an empty list, not the full result set."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(_make_template_gen("gen_a"))
        await registry.register(_make_template_gen("gen_b"))

        results = await registry.list_definitions(limit=0)
        assert results == []

    async def test_list_definitions_offset_beyond_count_returns_empty(
        self,
    ) -> None:
        """``offset`` past the row count returns an empty list."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(_make_template_gen("gen_a"))
        await registry.register(_make_template_gen("gen_b"))

        results = await registry.list_definitions(offset=100)
        assert results == []


class TestGeneratorRegistryCount:
    """Tests for ``count_definitions()``.

    Single-write keying means the database's row count IS the
    definition count, so pushdown counts are safe.  These tests pin
    the contract: ``count_definitions`` returns
    ``len(list_definitions(...))`` for the same filter shape.
    """

    async def test_count_definitions_empty_returns_zero(self) -> None:
        """An empty registry has zero definitions."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        assert await registry.count_definitions() == 0

    async def test_count_definitions_returns_total(self) -> None:
        """``count_definitions`` returns the number of registered defs."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        for gid in ["gen_a", "gen_b", "gen_c"]:
            await registry.register(_make_template_gen(gid))

        assert await registry.count_definitions() == 3

    async def test_count_definitions_with_filter_metadata(self) -> None:
        """``filter_metadata`` narrows the count to matching rows."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(
            _make_template_gen("gen_a"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_b"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_c"), metadata={"tenant_id": "globex"}
        )

        assert (
            await registry.count_definitions(
                filter_metadata={"tenant_id": "acme"}
            )
            == 2
        )
        assert (
            await registry.count_definitions(
                filter_metadata={"tenant_id": "globex"}
            )
            == 1
        )

    async def test_count_definitions_matches_list_definitions_length(
        self,
    ) -> None:
        """``count_definitions`` equals ``len(list_definitions(...))``."""
        db = AsyncMemoryDatabase()
        registry = GeneratorRegistry(db)

        await registry.register(
            _make_template_gen("gen_a"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_b"), metadata={"tenant_id": "acme"}
        )
        await registry.register(
            _make_template_gen("gen_c"), metadata={"tenant_id": "globex"}
        )

        # Unfiltered parity.
        assert await registry.count_definitions() == len(
            await registry.list_definitions()
        )

        # Filtered parity.
        assert await registry.count_definitions(
            filter_metadata={"tenant_id": "acme"}
        ) == len(
            await registry.list_definitions(
                filter_metadata={"tenant_id": "acme"}
            )
        )
