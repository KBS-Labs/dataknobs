"""Tests for memory/catalog.py — ArtifactBankCatalog."""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.memory.catalog import ArtifactBankCatalog


def _make_artifact(
    name: str = "recipe",
    recipe_name: str | None = None,
    ingredients: list[dict[str, Any]] | None = None,
) -> ArtifactBank:
    """Create a minimal ArtifactBank with one field and one section."""
    section_db = SyncMemoryDatabase()
    sections = {
        "ingredients": MemoryBank(
            name="ingredients",
            schema={"required": ["name"]},
            db=section_db,
        ),
    }
    artifact = ArtifactBank(
        name=name,
        field_defs={"recipe_name": {"required": True}},
        sections=sections,
    )
    if recipe_name:
        artifact.set_field("recipe_name", recipe_name)
    if ingredients:
        bank = artifact.sections["ingredients"]
        for ing in ingredients:
            bank.add(ing, source_stage="test")
    return artifact


class TestArtifactBankCatalog:
    """Tests for ArtifactBankCatalog CRUD operations."""

    def test_empty_catalog(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        assert catalog.count() == 0
        assert catalog.list() == []

    def test_save_and_get(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour", "amount": "2 cups"}],
        )
        catalog.save(artifact)

        result = catalog.get("recipe")
        assert result is not None
        assert result["_artifact_name"] == "recipe"
        assert result["recipe_name"] == "Cookies"
        assert len(result["ingredients"]) == 1

    def test_save_validates(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact()  # No recipe_name set, no ingredients
        with pytest.raises(ValueError, match="Cannot save"):
            catalog.save(artifact)

    def test_upsert_overwrites(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact1 = _make_artifact(
            recipe_name="Cookies v1",
            ingredients=[{"name": "flour"}],
        )
        catalog.save(artifact1)

        artifact2 = _make_artifact(
            recipe_name="Cookies v2",
            ingredients=[{"name": "sugar"}],
        )
        catalog.save(artifact2)

        assert catalog.count() == 1
        result = catalog.get("recipe")
        assert result is not None
        assert result["recipe_name"] == "Cookies v2"

    def test_list_entries(self) -> None:
        db = SyncMemoryDatabase()
        catalog = ArtifactBankCatalog(db)

        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}, {"name": "sugar"}],
        )
        catalog.save(artifact)

        entries = catalog.list()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["name"] == "recipe"
        assert "ingredients" in entry["sections"]
        assert entry["field_count"] == 1  # recipe_name

    def test_delete_existing(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        catalog.save(artifact)
        assert catalog.count() == 1

        deleted = catalog.delete("recipe")
        assert deleted is True
        assert catalog.count() == 0

    def test_delete_nonexistent(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        assert catalog.delete("nonexistent") is False

    def test_get_nonexistent(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        assert catalog.get("nonexistent") is None

    def test_load_into(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())

        # Save an artifact
        source = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}, {"name": "sugar"}],
        )
        catalog.save(source)

        # Create a fresh target artifact
        target = _make_artifact()
        assert target.field("recipe_name") is None
        assert target.sections["ingredients"].count() == 0

        # Load from catalog
        found = catalog.load_into("recipe", target)
        assert found is True
        assert target.field("recipe_name") == "Cookies"
        assert target.sections["ingredients"].count() == 2

    def test_load_into_replaces_existing(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())

        # Save version 1
        v1 = _make_artifact(
            recipe_name="Cookies v1",
            ingredients=[{"name": "flour"}],
        )
        catalog.save(v1)

        # Create target with different data
        target = _make_artifact(
            recipe_name="Brownies",
            ingredients=[{"name": "cocoa"}, {"name": "butter"}],
        )
        assert target.sections["ingredients"].count() == 2

        # Load replaces everything
        catalog.load_into("recipe", target)
        assert target.field("recipe_name") == "Cookies v1"
        assert target.sections["ingredients"].count() == 1

    def test_load_into_nonexistent(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        target = _make_artifact()
        found = catalog.load_into("nonexistent", target)
        assert found is False

    def test_from_config_default_memory(self) -> None:
        catalog = ArtifactBankCatalog.from_config({})
        assert catalog.count() == 0
        # Verify it works by saving something
        artifact = _make_artifact(
            recipe_name="Test",
            ingredients=[{"name": "salt"}],
        )
        catalog.save(artifact)
        assert catalog.count() == 1

    def test_from_config_explicit_memory(self) -> None:
        catalog = ArtifactBankCatalog.from_config({
            "backend": "memory",
            "backend_config": {},
        })
        assert catalog.count() == 0


class TestArtifactBankPopulateReplace:
    """Tests for ArtifactBank.populate_from_compiled and replace_from_compiled."""

    def test_populate_from_compiled(self) -> None:
        artifact = _make_artifact()
        compiled = {
            "_artifact_name": "recipe",
            "recipe_name": "Cookies",
            "ingredients": [
                {"name": "flour", "amount": "2 cups"},
                {"name": "sugar", "amount": "1 cup"},
            ],
        }
        artifact.populate_from_compiled(compiled)
        assert artifact.field("recipe_name") == "Cookies"
        assert artifact.sections["ingredients"].count() == 2

    def test_populate_is_additive(self) -> None:
        artifact = _make_artifact(
            recipe_name="Base",
            ingredients=[{"name": "existing"}],
        )
        compiled = {
            "recipe_name": "Updated",
            "ingredients": [{"name": "new"}],
        }
        artifact.populate_from_compiled(compiled)
        assert artifact.field("recipe_name") == "Updated"
        # Additive: existing + new
        assert artifact.sections["ingredients"].count() == 2

    def test_replace_from_compiled(self) -> None:
        artifact = _make_artifact(
            recipe_name="Old Recipe",
            ingredients=[{"name": "old_ingredient"}],
        )
        assert artifact.sections["ingredients"].count() == 1

        compiled = {
            "recipe_name": "New Recipe",
            "ingredients": [
                {"name": "new_1"},
                {"name": "new_2"},
            ],
        }
        artifact.replace_from_compiled(compiled)

        assert artifact.field("recipe_name") == "New Recipe"
        assert artifact.sections["ingredients"].count() == 2

    def test_replace_unfinalizes(self) -> None:
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        artifact.finalize()
        assert artifact.is_finalized is True

        compiled = {
            "recipe_name": "Updated",
            "ingredients": [{"name": "sugar"}],
        }
        artifact.replace_from_compiled(compiled)

        assert artifact.is_finalized is False
        assert artifact.field("recipe_name") == "Updated"
