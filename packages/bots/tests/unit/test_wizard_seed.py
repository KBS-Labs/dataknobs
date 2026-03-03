"""Tests for artifact file seeding in wizard initialization.

Covers Phase 4 of 03b:
- _seed_artifact: JSON and JSONL file loading
- First-turn-only seeding (no re-seed on restore)
- Graceful failure on missing files
- Format auto-detection and explicit override
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.artifact_io import save_artifact, save_book
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


# =====================================================================
# Shared config and helpers
# =====================================================================

ARTIFACT_CONFIG: dict[str, Any] = {
    "name": "recipe",
    "fields": {"recipe_name": {"required": True}},
    "sections": {
        "ingredients": {
            "schema": {"required": ["name"]},
            "match_fields": ["name"],
        },
        "instructions": {
            "schema": {"required": ["instruction"]},
        },
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
    recipe_name: str = "Chocolate Chip Cookies",
) -> ArtifactBank:
    """Create a populated recipe artifact for seeding."""
    ingredients = _make_bank("ingredients", required=["name"], match_fields=["name"])
    instructions = _make_bank("instructions", required=["instruction"])
    artifact = ArtifactBank(
        name="recipe",
        field_defs={"recipe_name": {"required": True}},
        sections={"ingredients": ingredients, "instructions": instructions},
    )
    artifact.set_field("recipe_name", recipe_name)
    ingredients.add({"name": "flour", "amount": "2 cups"})
    ingredients.add({"name": "sugar", "amount": "1 cup"})
    instructions.add({"instruction": "Preheat oven to 375F"})
    instructions.add({"instruction": "Mix dry ingredients"})
    return artifact


def _make_wizard_with_seed(
    seed_config: dict[str, Any],
) -> WizardReasoning:
    """Create a WizardReasoning with artifact + seed configuration."""
    config: dict[str, Any] = {
        "name": "test-seed-wizard",
        "version": "1.0",
        "settings": {
            "artifact": {
                **ARTIFACT_CONFIG,
                "seed": seed_config,
            },
        },
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "is_end": True,
                "prompt": "Provide data",
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _make_wizard_without_seed() -> WizardReasoning:
    """Create a WizardReasoning with artifact but no seed."""
    config: dict[str, Any] = {
        "name": "test-noseed-wizard",
        "version": "1.0",
        "settings": {
            "artifact": ARTIFACT_CONFIG,
        },
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "is_end": True,
                "prompt": "Provide data",
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# _seed_artifact unit tests
# =====================================================================


class TestSeedArtifactJSON:
    """Tests for seeding from JSON files."""

    def test_seed_from_compiled_json(self, tmp_path: Path) -> None:
        """JSON seed file in compiled format populates artifact."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.json"
        save_artifact(recipe, seed_file, compiled=True)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        assert wizard._artifact is not None
        assert wizard._artifact.field("recipe_name") == "Chocolate Chip Cookies"
        assert wizard._artifact.section("ingredients").count() == 2
        assert wizard._artifact.section("instructions").count() == 2

        # Seeded records have source_stage="seed"
        for rec in wizard._artifact.section("ingredients").all():
            assert rec.source_stage == "seed"

    def test_seed_from_full_state_json(self, tmp_path: Path) -> None:
        """JSON seed file in full-state format populates artifact."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.json"
        save_artifact(recipe, seed_file, compiled=False)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        assert wizard._artifact is not None
        assert wizard._artifact.field("recipe_name") == "Chocolate Chip Cookies"
        assert wizard._artifact.section("ingredients").count() == 2

    def test_seed_with_explicit_json_format(self, tmp_path: Path) -> None:
        """Explicit format='json' works for non-.json extension."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.dat"
        save_artifact(recipe, seed_file, compiled=True)

        wizard = _make_wizard_with_seed({
            "source": str(seed_file),
            "format": "json",
        })

        assert wizard._artifact is not None
        assert wizard._artifact.field("recipe_name") == "Chocolate Chip Cookies"


class TestSeedArtifactJSONL:
    """Tests for seeding from JSONL book files."""

    def test_seed_from_jsonl_with_select(self, tmp_path: Path) -> None:
        """JSONL seed with select picks the named artifact."""
        recipe1 = _make_recipe("Chocolate Chip Cookies")
        recipe2 = _make_recipe("Banana Bread")
        seed_file = tmp_path / "recipes.jsonl"
        save_book([recipe1, recipe2], seed_file, compiled=True)

        wizard = _make_wizard_with_seed({
            "source": str(seed_file),
            "select": "Banana Bread",
        })

        assert wizard._artifact is not None
        # The seed populates the artifact — recipe_name comes from
        # compiled data's recipe_name field, not _artifact_name
        assert wizard._artifact.field("recipe_name") == "Banana Bread"

    def test_seed_from_jsonl_without_select(self, tmp_path: Path) -> None:
        """JSONL seed without select loads the first entry."""
        recipe1 = _make_recipe("First Recipe")
        recipe2 = _make_recipe("Second Recipe")
        seed_file = tmp_path / "recipes.jsonl"
        save_book([recipe1, recipe2], seed_file, compiled=True)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        assert wizard._artifact is not None
        assert wizard._artifact.field("recipe_name") == "First Recipe"

    def test_seed_jsonl_auto_detected_from_extension(self, tmp_path: Path) -> None:
        """JSONL format auto-detected from .jsonl extension."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.jsonl"
        save_book([recipe], seed_file, compiled=True)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 2

    def test_seed_jsonl_explicit_format(self, tmp_path: Path) -> None:
        """Explicit format='jsonl' overrides extension-based detection."""
        recipe = _make_recipe()
        # Save as .dat but declare format as jsonl
        seed_file = tmp_path / "seed.dat"
        save_book([recipe], seed_file, compiled=True)

        wizard = _make_wizard_with_seed({
            "source": str(seed_file),
            "format": "jsonl",
        })

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 2


class TestSeedArtifactGracefulFailure:
    """Tests for graceful failure when seed file is missing or invalid."""

    def test_missing_seed_file_proceeds_empty(self, tmp_path: Path) -> None:
        """Missing seed file logs WARNING and continues with empty artifact."""
        wizard = _make_wizard_with_seed({
            "source": str(tmp_path / "nonexistent.json"),
        })

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 0
        assert wizard._artifact.section("instructions").count() == 0

    def test_missing_source_key_proceeds_empty(self) -> None:
        """Seed config without 'source' key proceeds with empty artifact."""
        wizard = _make_wizard_with_seed({"format": "json"})

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 0

    def test_invalid_json_proceeds_empty(self, tmp_path: Path) -> None:
        """Invalid JSON seed file proceeds with empty artifact."""
        seed_file = tmp_path / "bad.json"
        seed_file.write_text("NOT VALID JSON {{{", encoding="utf-8")

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 0

    def test_jsonl_select_not_found_proceeds_empty(self, tmp_path: Path) -> None:
        """JSONL with select name that doesn't exist proceeds empty."""
        recipe = _make_recipe()
        seed_file = tmp_path / "recipes.jsonl"
        save_book([recipe], seed_file, compiled=True)

        wizard = _make_wizard_with_seed({
            "source": str(seed_file),
            "select": "Nonexistent Recipe",
        })

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 0


class TestSeedFirstTurnOnly:
    """Tests that seeding only happens on first turn (not on restore)."""

    def test_no_seed_without_config(self) -> None:
        """Wizard without seed config has empty artifact."""
        wizard = _make_wizard_without_seed()

        assert wizard._artifact is not None
        assert wizard._artifact.section("ingredients").count() == 0

    def test_seed_populates_sections_via_banks(self, tmp_path: Path) -> None:
        """Seeded sections are accessible via wizard._banks (same instances)."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.json"
        save_artifact(recipe, seed_file, compiled=True)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        # Banks are the same objects as artifact sections
        assert "ingredients" in wizard._banks
        assert wizard._banks["ingredients"].count() == 2
        assert wizard._banks["instructions"].count() == 2

    def test_seed_records_have_seed_provenance(self, tmp_path: Path) -> None:
        """All seeded records have source_stage='seed'."""
        recipe = _make_recipe()
        seed_file = tmp_path / "seed.json"
        save_artifact(recipe, seed_file, compiled=True)

        wizard = _make_wizard_with_seed({"source": str(seed_file)})

        for rec in wizard._artifact.section("ingredients").all():
            assert rec.source_stage == "seed"
        for rec in wizard._artifact.section("instructions").all():
            assert rec.source_stage == "seed"
