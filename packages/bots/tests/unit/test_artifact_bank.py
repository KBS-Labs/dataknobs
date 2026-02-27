"""Tests for memory/artifact_bank.py."""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data import SyncDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase

from dataknobs_bots.memory.bank import EmptyBankProxy, MemoryBank
from dataknobs_bots.memory.artifact_bank import ArtifactBank


def _make_bank(
    name: str,
    required: list[str] | None = None,
    match_fields: list[str] | None = None,
) -> MemoryBank:
    """Create a MemoryBank backed by SyncMemoryDatabase."""
    schema: dict[str, Any] = {}
    if required:
        schema["required"] = required
    return MemoryBank(
        name=name,
        schema=schema,
        db=SyncMemoryDatabase(),
        match_fields=match_fields,
    )


def _make_recipe_artifact(
    with_data: bool = False,
) -> ArtifactBank:
    """Create a recipe artifact with ingredients and instructions sections."""
    ingredients = _make_bank("ingredients", required=["name"], match_fields=["name"])
    instructions = _make_bank("instructions", required=["instruction"])
    artifact = ArtifactBank(
        name="recipe",
        field_defs={"recipe_name": {"required": True}},
        sections={"ingredients": ingredients, "instructions": instructions},
    )
    if with_data:
        artifact.set_field("recipe_name", "Chocolate Chip Cookies")
        ingredients.add({"name": "flour", "amount": "2 cups"})
        ingredients.add({"name": "sugar", "amount": "1 cup"})
        instructions.add({"instruction": "Preheat oven to 375F"})
        instructions.add({"instruction": "Mix dry ingredients"})
    return artifact


class TestArtifactBankInit:
    """Tests for ArtifactBank construction."""

    def test_basic_construction(self) -> None:
        ingredients = _make_bank("ingredients", required=["name"])
        artifact = ArtifactBank(
            name="recipe",
            field_defs={"recipe_name": {"required": True}},
            sections={"ingredients": ingredients},
        )
        assert artifact.name == "recipe"
        assert "recipe_name" in artifact.field_defs
        assert "ingredients" in artifact.sections

    def test_field_values_start_as_none(self) -> None:
        artifact = _make_recipe_artifact()
        assert artifact.field("recipe_name") is None
        assert artifact.fields == {"recipe_name": None}

    def test_not_finalized_by_default(self) -> None:
        artifact = _make_recipe_artifact()
        assert artifact.is_finalized is False

    def test_section_configs_preserved(self) -> None:
        ingredients = _make_bank("ingredients")
        configs = {"ingredients": {"schema": {"required": ["name"]}}}
        artifact = ArtifactBank(
            name="test",
            field_defs={},
            sections={"ingredients": ingredients},
            section_configs=configs,
        )
        assert artifact.to_dict()["section_configs"] == configs


class TestArtifactBankFields:
    """Tests for field management."""

    def test_set_and_get_field(self) -> None:
        artifact = _make_recipe_artifact()
        artifact.set_field("recipe_name", "Test Recipe")
        assert artifact.field("recipe_name") == "Test Recipe"

    def test_fields_snapshot_is_copy(self) -> None:
        artifact = _make_recipe_artifact()
        artifact.set_field("recipe_name", "Test")
        snapshot = artifact.fields
        snapshot["recipe_name"] = "Modified"
        assert artifact.field("recipe_name") == "Test"

    def test_set_unknown_field_raises(self) -> None:
        artifact = _make_recipe_artifact()
        with pytest.raises(ValueError, match="Unknown field 'unknown'"):
            artifact.set_field("unknown", "value")

    def test_clear_fields(self) -> None:
        artifact = _make_recipe_artifact()
        artifact.set_field("recipe_name", "Test Recipe")
        artifact.clear_fields()
        assert artifact.field("recipe_name") is None

    def test_set_field_when_finalized_raises(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        artifact.finalize()
        with pytest.raises(ValueError, match="finalized"):
            artifact.set_field("recipe_name", "New Name")

    def test_field_returns_none_for_undefined(self) -> None:
        artifact = _make_recipe_artifact()
        assert artifact.field("nonexistent") is None


class TestArtifactBankSections:
    """Tests for section access."""

    def test_section_returns_memory_bank(self) -> None:
        artifact = _make_recipe_artifact()
        section = artifact.section("ingredients")
        assert isinstance(section, MemoryBank)
        assert section.name == "ingredients"

    def test_section_missing_returns_empty_proxy(self) -> None:
        artifact = _make_recipe_artifact()
        section = artifact.section("nonexistent")
        assert isinstance(section, EmptyBankProxy)
        assert section.name == "nonexistent"

    def test_sections_property_returns_copy(self) -> None:
        artifact = _make_recipe_artifact()
        sections = artifact.sections
        sections["extra"] = _make_bank("extra")
        assert "extra" not in artifact.sections

    def test_section_operations_work(self) -> None:
        artifact = _make_recipe_artifact()
        bank = artifact.section("ingredients")
        assert isinstance(bank, MemoryBank)
        record_id = bank.add({"name": "flour"})
        assert bank.count() == 1
        assert bank.get(record_id) is not None


class TestArtifactBankCompile:
    """Tests for compile()."""

    def test_compile_empty_artifact(self) -> None:
        artifact = _make_recipe_artifact()
        compiled = artifact.compile()
        assert compiled["_artifact_name"] == "recipe"
        assert "_compiled_at" in compiled
        assert compiled["recipe_name"] is None
        assert compiled["ingredients"] == []
        assert compiled["instructions"] == []

    def test_compile_with_data(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        compiled = artifact.compile()
        assert compiled["recipe_name"] == "Chocolate Chip Cookies"
        assert len(compiled["ingredients"]) == 2
        assert len(compiled["instructions"]) == 2
        # Verify record data is present
        ing_names = [r["name"] for r in compiled["ingredients"]]
        assert "flour" in ing_names
        assert "sugar" in ing_names
        inst_texts = [r["instruction"] for r in compiled["instructions"]]
        assert "Preheat oven to 375F" in inst_texts

    def test_compile_returns_dicts_not_bank_records(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        compiled = artifact.compile()
        for record in compiled["ingredients"]:
            assert isinstance(record, dict)
            assert "record_id" not in record


class TestArtifactBankValidation:
    """Tests for validate()."""

    def test_valid_artifact(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        errors = artifact.validate()
        assert errors == []

    def test_missing_required_field(self) -> None:
        artifact = _make_recipe_artifact()
        # Don't set recipe_name, but add records
        artifact.section("ingredients").add({"name": "flour"})
        artifact.section("instructions").add({"instruction": "Step 1"})
        errors = artifact.validate()
        assert len(errors) == 1
        assert "recipe_name" in errors[0]

    def test_empty_section(self) -> None:
        artifact = _make_recipe_artifact()
        artifact.set_field("recipe_name", "Test Recipe")
        # Don't add any records
        errors = artifact.validate()
        assert len(errors) == 2  # Both sections empty

    def test_multiple_errors(self) -> None:
        artifact = _make_recipe_artifact()
        # No field set, no records
        errors = artifact.validate()
        assert len(errors) == 3  # 1 field + 2 sections

    def test_optional_field_not_required(self) -> None:
        ingredients = _make_bank("ingredients", required=["name"])
        artifact = ArtifactBank(
            name="recipe",
            field_defs={
                "recipe_name": {"required": True},
                "description": {"required": False},
            },
            sections={"ingredients": ingredients},
        )
        artifact.set_field("recipe_name", "Test")
        ingredients.add({"name": "flour"})
        errors = artifact.validate()
        assert errors == []


class TestArtifactBankFinalization:
    """Tests for finalize() / unfinalize()."""

    def test_finalize_returns_compiled(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        result = artifact.finalize()
        assert result["recipe_name"] == "Chocolate Chip Cookies"
        assert len(result["ingredients"]) == 2
        assert artifact.is_finalized is True

    def test_finalize_with_errors_raises(self) -> None:
        artifact = _make_recipe_artifact()
        with pytest.raises(ValueError, match="Cannot finalize"):
            artifact.finalize()
        assert artifact.is_finalized is False

    def test_set_field_blocked_when_finalized(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        artifact.finalize()
        with pytest.raises(ValueError, match="finalized"):
            artifact.set_field("recipe_name", "New Name")

    def test_unfinalize_allows_edits(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        artifact.finalize()
        artifact.unfinalize()
        assert artifact.is_finalized is False
        artifact.set_field("recipe_name", "Updated Name")
        assert artifact.field("recipe_name") == "Updated Name"

    def test_finalize_validate_compile_chain(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        # First validate, then finalize
        errors = artifact.validate()
        assert errors == []
        compiled = artifact.finalize()
        assert compiled["_artifact_name"] == "recipe"
        assert artifact.is_finalized is True


class TestArtifactBankSerialization:
    """Tests for to_dict() / from_dict() roundtrip."""

    def test_roundtrip_empty(self) -> None:
        artifact = _make_recipe_artifact()
        data = artifact.to_dict()
        restored = ArtifactBank.from_dict(data)
        assert restored.name == "recipe"
        assert restored.field("recipe_name") is None
        assert restored.is_finalized is False
        assert "ingredients" in restored.sections
        assert "instructions" in restored.sections

    def test_roundtrip_with_data(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        data = artifact.to_dict()
        restored = ArtifactBank.from_dict(data)
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        assert restored.section("ingredients").count() == 2
        assert restored.section("instructions").count() == 2

    def test_roundtrip_finalized(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        artifact.finalize()
        data = artifact.to_dict()
        restored = ArtifactBank.from_dict(data)
        assert restored.is_finalized is True

    def test_roundtrip_preserves_section_configs(self) -> None:
        configs = {
            "ingredients": {"schema": {"required": ["name"]}, "max_records": 30},
        }
        ingredients = _make_bank("ingredients", required=["name"])
        artifact = ArtifactBank(
            name="recipe",
            field_defs={},
            sections={"ingredients": ingredients},
            section_configs=configs,
        )
        data = artifact.to_dict()
        assert data["section_configs"] == configs

    def test_from_dict_with_db_factory(self) -> None:
        """db_factory is called for non-memory backends during restore."""
        factory_calls: list[tuple[str, dict[str, Any]]] = []

        def mock_factory(
            name: str, cfg: dict[str, Any]
        ) -> tuple[SyncDatabase, str]:
            factory_calls.append((name, cfg))
            return SyncMemoryDatabase(), "external"

        artifact = _make_recipe_artifact(with_data=True)
        data = artifact.to_dict()
        # Simulate a non-memory backend in section_configs
        data["section_configs"] = {
            "ingredients": {"backend": "sqlite"},
            "instructions": {},  # memory (default)
        }
        restored = ArtifactBank.from_dict(data, db_factory=mock_factory)
        # Factory called for sqlite backend only
        assert len(factory_calls) == 1
        assert factory_calls[0][0] == "ingredients"
        # Both sections still restored
        assert "ingredients" in restored.sections
        assert "instructions" in restored.sections

    def test_serialized_format(self) -> None:
        artifact = _make_recipe_artifact(with_data=True)
        data = artifact.to_dict()
        assert data["name"] == "recipe"
        assert "field_defs" in data
        assert "field_values" in data
        assert "finalized" in data
        assert "section_configs" in data
        assert "sections" in data
        assert data["field_values"]["recipe_name"] == "Chocolate Chip Cookies"


class TestArtifactBankFromConfig:
    """Tests for from_config() factory."""

    def test_basic_config(self) -> None:
        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "max_records": 30,
                    "duplicate_strategy": "reject",
                    "match_fields": ["name"],
                },
                "instructions": {
                    "schema": {"required": ["instruction"]},
                    "max_records": 50,
                },
            },
        }
        artifact = ArtifactBank.from_config(config)
        assert artifact.name == "recipe"
        assert "recipe_name" in artifact.field_defs
        assert "ingredients" in artifact.sections
        assert "instructions" in artifact.sections

    def test_config_with_db_factory(self) -> None:
        factory_calls: list[tuple[str, dict[str, Any]]] = []

        def test_factory(
            name: str, cfg: dict[str, Any]
        ) -> tuple[SyncDatabase, str]:
            factory_calls.append((name, cfg))
            return SyncMemoryDatabase(), "inline"

        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {},
            "sections": {
                "ingredients": {"schema": {"required": ["name"]}},
            },
        }
        artifact = ArtifactBank.from_config(config, db_factory=test_factory)
        assert len(factory_calls) == 1
        assert factory_calls[0][0] == "ingredients"
        assert "ingredients" in artifact.sections

    def test_config_defaults(self) -> None:
        config: dict[str, Any] = {}
        artifact = ArtifactBank.from_config(config)
        assert artifact.name == "artifact"  # default name
        assert artifact.field_defs == {}
        assert artifact.sections == {}

    def test_config_section_properties(self) -> None:
        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {},
            "sections": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "match_fields": ["name"],
                    "duplicate_strategy": "reject",
                    "max_records": 10,
                },
            },
        }
        artifact = ArtifactBank.from_config(config)
        bank = artifact.section("ingredients")
        assert isinstance(bank, MemoryBank)
        assert bank.match_fields == ["name"]
        # max_records is enforced
        assert bank._max_records == 10

    def test_config_with_nested_duplicate_detection(self) -> None:
        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {},
            "sections": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "duplicate_detection": {
                        "strategy": "reject",
                        "match_fields": ["name"],
                    },
                },
            },
        }
        artifact = ArtifactBank.from_config(config)
        bank = artifact.section("ingredients")
        assert isinstance(bank, MemoryBank)
        assert bank.match_fields == ["name"]

    def test_section_configs_preserved_in_artifact(self) -> None:
        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {},
            "sections": {
                "ingredients": {"schema": {"required": ["name"]}},
            },
        }
        artifact = ArtifactBank.from_config(config)
        serialized = artifact.to_dict()
        assert "ingredients" in serialized["section_configs"]
