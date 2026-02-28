"""Tests for memory/artifact_io.py — ArtifactBank file import/export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.artifact_io import (
    append_to_book,
    list_book,
    load_artifact,
    load_from_book,
    save_artifact,
    save_book,
)
from dataknobs_bots.memory.bank import MemoryBank

# ---------------------------------------------------------------------------
# Shared config and helpers
# ---------------------------------------------------------------------------

RECIPE_CONFIG: dict[str, Any] = {
    "name": "recipe",
    "fields": {"recipe_name": {"required": True}},
    "sections": {
        "ingredients": {"schema": {"required": ["name"]}, "match_fields": ["name"]},
        "instructions": {"schema": {"required": ["instruction"]}},
    },
}


def _make_bank(
    name: str,
    required: list[str] | None = None,
    match_fields: list[str] | None = None,
) -> MemoryBank:
    schema: dict[str, Any] = {}
    if required:
        schema["required"] = required
    return MemoryBank(
        name=name,
        schema=schema,
        db=SyncMemoryDatabase(),
        match_fields=match_fields,
    )


def _make_recipe(
    *,
    recipe_name: str = "Chocolate Chip Cookies",
    servings: int | None = None,
) -> ArtifactBank:
    """Create a populated recipe artifact."""
    field_defs: dict[str, dict[str, Any]] = {"recipe_name": {"required": True}}
    if servings is not None:
        field_defs["servings"] = {"required": False}

    ingredients = _make_bank("ingredients", required=["name"], match_fields=["name"])
    instructions = _make_bank("instructions", required=["instruction"])
    artifact = ArtifactBank(
        name="recipe",
        field_defs=field_defs,
        sections={"ingredients": ingredients, "instructions": instructions},
    )
    artifact.set_field("recipe_name", recipe_name)
    if servings is not None:
        artifact.set_field("servings", servings)
    ingredients.add({"name": "flour", "amount": "2 cups"})
    ingredients.add({"name": "sugar", "amount": "1 cup"})
    instructions.add({"instruction": "Preheat oven to 375F"})
    instructions.add({"instruction": "Mix dry ingredients"})
    return artifact


# ---------------------------------------------------------------------------
# TestSaveAndLoadArtifact
# ---------------------------------------------------------------------------


class TestSaveAndLoadArtifact:
    """Round-trip tests for single-artifact JSON files."""

    def test_compiled_export_import_with_config(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        filepath = tmp_path / "recipe.json"

        save_artifact(artifact, filepath, compiled=True)
        restored = load_artifact(filepath, artifact_config=RECIPE_CONFIG)

        assert restored.name == "recipe"
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        assert restored.section("ingredients").count() == 2
        assert restored.section("instructions").count() == 2

        # Imported records have source_stage="import"
        for rec in restored.section("ingredients").all():
            assert rec.source_stage == "import"

    def test_full_state_export_import(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        artifact.finalize()
        filepath = tmp_path / "recipe.json"

        save_artifact(artifact, filepath, compiled=False)
        restored = load_artifact(filepath)

        assert restored.name == "recipe"
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        assert restored.is_finalized is True
        assert restored.section("ingredients").count() == 2

        # Provenance preserved via to_dict/from_dict round-trip
        orig_records = artifact.section("ingredients").all()
        rest_records = restored.section("ingredients").all()
        assert orig_records[0].source_stage == rest_records[0].source_stage

    def test_format_auto_detection_compiled(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        filepath = tmp_path / "recipe.json"
        save_artifact(artifact, filepath, compiled=True)

        data = json.loads(filepath.read_text())
        assert "_artifact_name" in data
        # load_artifact auto-detects compiled format
        restored = load_artifact(filepath, artifact_config=RECIPE_CONFIG)
        assert restored.name == "recipe"

    def test_format_auto_detection_full_state(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        filepath = tmp_path / "recipe.json"
        save_artifact(artifact, filepath, compiled=False)

        data = json.loads(filepath.read_text())
        assert "name" in data and "sections" in data
        # load_artifact auto-detects full-state format
        restored = load_artifact(filepath)
        assert restored.name == "recipe"

    def test_compiled_without_config_inference(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        filepath = tmp_path / "recipe.json"
        save_artifact(artifact, filepath, compiled=True)

        # Load without config — inference mode
        restored = load_artifact(filepath)

        assert restored.name == "recipe"
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        # Sections reconstructed (inferred as list[dict])
        assert restored.section("ingredients").count() == 2
        assert restored.section("instructions").count() == 2

    def test_compiled_non_string_field_values(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {
            "name": "recipe",
            "fields": {
                "recipe_name": {"required": True},
                "servings": {"required": False},
            },
            "sections": {
                "ingredients": {
                    "schema": {"required": ["name"]},
                    "match_fields": ["name"],
                },
                "instructions": {"schema": {"required": ["instruction"]}},
            },
        }
        artifact = _make_recipe(servings=4)
        filepath = tmp_path / "recipe.json"
        save_artifact(artifact, filepath, compiled=True)

        restored = load_artifact(filepath, artifact_config=config)
        assert restored.field("servings") == 4

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        filepath = tmp_path / "bad.json"
        filepath.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_artifact(filepath)


# ---------------------------------------------------------------------------
# TestBookOperations
# ---------------------------------------------------------------------------


class TestBookOperations:
    """Tests for JSONL book files (append, save, load, list)."""

    def test_append_and_list(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"

        for name in ["Cookies", "Brownies", "Cake"]:
            artifact = _make_recipe(recipe_name=name)
            append_to_book(artifact, filepath, compiled=True)

        entries = list_book(filepath)
        assert len(entries) == 3
        assert entries[0]["name"] == "recipe"
        assert entries[0]["index"] == 0
        assert entries[0]["format"] == "compiled"
        assert entries[2]["index"] == 2

    def test_append_creates_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "subdir" / "recipes.jsonl"
        assert not filepath.exists()

        artifact = _make_recipe()
        append_to_book(artifact, filepath)
        assert filepath.exists()
        entries = list_book(filepath)
        assert len(entries) == 1

    def test_load_by_name(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        for name in ["Cookies", "Brownies", "Cake"]:
            artifact = _make_recipe(recipe_name=name)
            append_to_book(artifact, filepath)

        # All have artifact name "recipe", but we can load by that
        restored = load_from_book(
            filepath, name="recipe", artifact_config=RECIPE_CONFIG
        )
        assert restored.name == "recipe"
        assert restored.field("recipe_name") == "Cookies"  # first match

    def test_load_by_index(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        for name in ["Cookies", "Brownies", "Cake"]:
            artifact = _make_recipe(recipe_name=name)
            append_to_book(artifact, filepath)

        restored = load_from_book(
            filepath, index=1, artifact_config=RECIPE_CONFIG
        )
        assert restored.field("recipe_name") == "Brownies"

    def test_load_name_not_found(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        artifact = _make_recipe()
        append_to_book(artifact, filepath)

        with pytest.raises(ValueError, match="No artifact named 'missing'"):
            load_from_book(filepath, name="missing")

    def test_load_index_out_of_range(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        artifact = _make_recipe()
        append_to_book(artifact, filepath)

        with pytest.raises(ValueError, match="out of range"):
            load_from_book(filepath, index=5)

    def test_load_neither_name_nor_index(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        artifact = _make_recipe()
        append_to_book(artifact, filepath)

        with pytest.raises(ValueError, match="Provide either"):
            load_from_book(filepath)

    def test_load_both_name_and_index(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        artifact = _make_recipe()
        append_to_book(artifact, filepath)

        with pytest.raises(ValueError, match="not both"):
            load_from_book(filepath, name="recipe", index=0)

    def test_save_book_overwrites(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"

        # Write 3 artifacts
        artifacts_3 = [_make_recipe(recipe_name=n) for n in ["A", "B", "C"]]
        save_book(artifacts_3, filepath)
        assert len(list_book(filepath)) == 3

        # Overwrite with 2
        artifacts_2 = [_make_recipe(recipe_name=n) for n in ["X", "Y"]]
        save_book(artifacts_2, filepath)
        entries = list_book(filepath)
        assert len(entries) == 2

    def test_save_book_full_state_roundtrip(self, tmp_path: Path) -> None:
        filepath = tmp_path / "recipes.jsonl"
        original = _make_recipe()
        original.finalize()

        save_book([original], filepath, compiled=False)
        restored = load_from_book(filepath, index=0)

        assert restored.name == "recipe"
        assert restored.is_finalized is True
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        assert restored.section("ingredients").count() == 2


# ---------------------------------------------------------------------------
# TestAtomicWrite
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Verify atomic write produces valid output with no temp file leftovers."""

    def test_save_artifact_valid_json_no_tmp(self, tmp_path: Path) -> None:
        artifact = _make_recipe()
        filepath = tmp_path / "recipe.json"
        save_artifact(artifact, filepath)

        # Valid JSON
        data = json.loads(filepath.read_text())
        assert data["_artifact_name"] == "recipe"

        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_save_book_valid_jsonl_no_tmp(self, tmp_path: Path) -> None:
        artifacts = [_make_recipe(recipe_name=n) for n in ["A", "B"]]
        filepath = tmp_path / "recipes.jsonl"
        save_book(artifacts, filepath)

        # Each line is valid JSON
        lines = [
            line for line in filepath.read_text().strip().split("\n") if line.strip()
        ]
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "_artifact_name" in data

        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# TestListBook
# ---------------------------------------------------------------------------


class TestListBook:
    """Tests for list_book."""

    def test_empty_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")
        assert list_book(filepath) == []

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_book(tmp_path / "nonexistent.jsonl")

    def test_mixed_formats(self, tmp_path: Path) -> None:
        filepath = tmp_path / "mixed.jsonl"

        # Write one compiled and one full-state
        artifact = _make_recipe()
        append_to_book(artifact, filepath, compiled=True)
        append_to_book(artifact, filepath, compiled=False)

        entries = list_book(filepath)
        assert len(entries) == 2
        assert entries[0]["format"] == "compiled"
        assert entries[1]["format"] == "full_state"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case round-trips."""

    def test_fields_only_no_sections(self, tmp_path: Path) -> None:
        artifact = ArtifactBank(
            name="simple",
            field_defs={"title": {"required": True}},
            sections={},
        )
        artifact.set_field("title", "Hello")

        filepath = tmp_path / "simple.json"
        # Compiled round-trip
        save_artifact(artifact, filepath, compiled=True)
        config: dict[str, Any] = {
            "name": "simple",
            "fields": {"title": {"required": True}},
            "sections": {},
        }
        restored = load_artifact(filepath, artifact_config=config)
        assert restored.field("title") == "Hello"
        assert restored.sections == {}

    def test_sections_only_no_fields(self, tmp_path: Path) -> None:
        ingredients = _make_bank("ingredients", required=["name"])
        artifact = ArtifactBank(
            name="list",
            field_defs={},
            sections={"ingredients": ingredients},
        )
        ingredients.add({"name": "flour"})

        filepath = tmp_path / "list.json"
        save_artifact(artifact, filepath, compiled=True)
        config: dict[str, Any] = {
            "name": "list",
            "fields": {},
            "sections": {
                "ingredients": {"schema": {"required": ["name"]}},
            },
        }
        restored = load_artifact(filepath, artifact_config=config)
        assert restored.field_defs == {}
        assert restored.section("ingredients").count() == 1

    def test_empty_artifact(self, tmp_path: Path) -> None:
        artifact = ArtifactBank(
            name="empty",
            field_defs={},
            sections={},
        )

        filepath = tmp_path / "empty.json"
        # Full-state round-trip
        save_artifact(artifact, filepath, compiled=False)
        restored = load_artifact(filepath)
        assert restored.name == "empty"
        assert restored.fields == {}
        assert restored.sections == {}

    def test_full_state_fields_only_roundtrip(self, tmp_path: Path) -> None:
        artifact = ArtifactBank(
            name="simple",
            field_defs={"title": {"required": True}},
            sections={},
        )
        artifact.set_field("title", "World")

        filepath = tmp_path / "simple_full.json"
        save_artifact(artifact, filepath, compiled=False)
        restored = load_artifact(filepath)
        assert restored.field("title") == "World"

    def test_full_state_sections_only_roundtrip(self, tmp_path: Path) -> None:
        ingredients = _make_bank("ingredients", required=["name"])
        artifact = ArtifactBank(
            name="list",
            field_defs={},
            sections={"ingredients": ingredients},
        )
        ingredients.add({"name": "salt"})

        filepath = tmp_path / "list_full.json"
        save_artifact(artifact, filepath, compiled=False)
        restored = load_artifact(filepath)
        assert restored.section("ingredients").count() == 1
        assert restored.section("ingredients").all()[0].data["name"] == "salt"
