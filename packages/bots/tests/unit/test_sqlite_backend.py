"""Tests for config-driven SQLite backend support.

Verifies that the MemoryBank/ArtifactBank/Catalog infrastructure works
end-to-end with SyncSQLiteDatabase via the config-driven paths
(_create_bank_db, ArtifactBank.from_config with db_factory, and
ArtifactBankCatalog.from_config).

Existing tests in test_memory_bank.py cover SQLite with manually
connected databases.  These tests cover the config-driven paths
where databases are created via database_factory and must be
connected before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.memory.catalog import ArtifactBankCatalog
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data import SyncDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase


# =====================================================================
# Helpers
# =====================================================================

def _make_wizard() -> WizardReasoning:
    """Create a minimal WizardReasoning for testing _create_bank_db."""
    config: dict[str, Any] = {
        "name": "test-wizard",
        "version": "1.0",
        "settings": {},
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "test",
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


def _make_sqlite_db_factory(
    tmp_path: Path,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    "callable[[str, dict[str, Any]], tuple[SyncDatabase, str]]",
]:
    """Create a db_factory that produces SQLite backends in tmp_path.

    Returns (call_log, factory_fn).  call_log records all calls for
    assertion.
    """
    call_log: list[tuple[str, dict[str, Any]]] = []

    def factory(
        name: str, cfg: dict[str, Any]
    ) -> tuple[SyncDatabase, str]:
        call_log.append((name, cfg))
        db_path = str(tmp_path / f"{name}.db")
        db = SyncSQLiteDatabase({"path": db_path, "table": name})
        db.connect()
        return db, "external"

    return call_log, factory


def _make_artifact_with_data(
    db_factory: Any = None,
) -> ArtifactBank:
    """Create an ArtifactBank with one section, optionally via db_factory."""
    config: dict[str, Any] = {
        "name": "recipe",
        "fields": {"recipe_name": {"required": True}},
        "sections": {
            "ingredients": {
                "schema": {"required": ["name"]},
                "max_records": 30,
            },
        },
    }
    artifact = ArtifactBank.from_config(config, db_factory=db_factory)
    artifact.set_field("recipe_name", "Chocolate Chip Cookies")
    artifact.section("ingredients").add(
        {"name": "flour", "amount": "2 cups"}, source_stage="test"
    )
    artifact.section("ingredients").add(
        {"name": "sugar", "amount": "1 cup"}, source_stage="test"
    )
    return artifact


# =====================================================================
# _create_bank_db tests
# =====================================================================

class TestCreateBankDb:
    """Tests for WizardReasoning._create_bank_db()."""

    def test_memory_backend_returns_inline(self) -> None:
        wizard = _make_wizard()
        db, mode = wizard._create_bank_db("items", {})
        assert isinstance(db, SyncMemoryDatabase)
        assert mode == "inline"

    def test_memory_backend_explicit(self) -> None:
        wizard = _make_wizard()
        db, mode = wizard._create_bank_db("items", {"backend": "memory"})
        assert isinstance(db, SyncMemoryDatabase)
        assert mode == "inline"

    def test_sqlite_backend_returns_connected_db(self, tmp_path: Path) -> None:
        wizard = _make_wizard()
        db_path = str(tmp_path / "test.db")
        db, mode = wizard._create_bank_db("items", {
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        assert isinstance(db, SyncSQLiteDatabase)
        assert mode == "external"
        # Verify the database is connected and usable
        assert db._connected
        db.close()

    def test_sqlite_backend_table_defaults_to_bank_name(
        self, tmp_path: Path
    ) -> None:
        wizard = _make_wizard()
        db_path = str(tmp_path / "test.db")
        db, mode = wizard._create_bank_db("ingredients", {
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        assert db.table_name == "ingredients"
        db.close()

    def test_sqlite_backend_respects_table_config(
        self, tmp_path: Path
    ) -> None:
        wizard = _make_wizard()
        db_path = str(tmp_path / "test.db")
        db, mode = wizard._create_bank_db("ingredients", {
            "backend": "sqlite",
            "backend_config": {"path": db_path, "table": "custom_table"},
        })
        assert db.table_name == "custom_table"
        db.close()

    def test_sqlite_backend_usable_for_crud(self, tmp_path: Path) -> None:
        """A bank created via _create_bank_db with sqlite can do CRUD."""
        wizard = _make_wizard()
        db_path = str(tmp_path / "test.db")
        db, mode = wizard._create_bank_db("ingredients", {
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        bank = MemoryBank(
            name="ingredients",
            schema={"required": ["name"]},
            db=db,
            storage_mode=mode,
        )
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        assert bank.count() == 2
        records = bank.all()
        names = {r.data["name"] for r in records}
        assert names == {"flour", "sugar"}
        bank.close()


# =====================================================================
# File-based persistence tests
# =====================================================================

class TestSQLitePersistence:
    """Tests for cross-session persistence via file-based SQLite."""

    def test_records_survive_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "items.db")

        # Session 1: create bank, add records, close
        db1 = SyncSQLiteDatabase({"path": db_path, "table": "items"})
        db1.connect()
        bank1 = MemoryBank(
            name="items",
            schema={"required": ["name"]},
            db=db1,
            storage_mode="external",
        )
        bank1.add({"name": "flour", "amount": "2 cups"})
        bank1.add({"name": "sugar", "amount": "1 cup"})
        assert bank1.count() == 2
        bank1.close()

        # Session 2: reopen same file, records are present
        db2 = SyncSQLiteDatabase({"path": db_path, "table": "items"})
        db2.connect()
        bank2 = MemoryBank(
            name="items",
            schema={"required": ["name"]},
            db=db2,
            storage_mode="external",
        )
        assert bank2.count() == 2
        names = {r.data["name"] for r in bank2.all()}
        assert names == {"flour", "sugar"}
        bank2.close()

    def test_external_mode_serialization_roundtrip(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "items.db")

        # Create bank, add records
        db = SyncSQLiteDatabase({"path": db_path, "table": "items"})
        db.connect()
        bank = MemoryBank(
            name="items",
            schema={"required": ["name"]},
            db=db,
            storage_mode="external",
        )
        bank.add({"name": "flour"})
        bank.add({"name": "sugar"})

        # Serialize — external mode omits records
        d = bank.to_dict()
        assert d["storage_mode"] == "external"
        assert "records" not in d
        bank.close()

        # Restore from dict with a reconnected db
        db2 = SyncSQLiteDatabase({"path": db_path, "table": "items"})
        db2.connect()
        restored = MemoryBank.from_dict(d, db=db2)
        assert restored.count() == 2
        names = {r.data["name"] for r in restored.all()}
        assert names == {"flour", "sugar"}
        db2.close()

    def test_artifact_with_sqlite_section(self, tmp_path: Path) -> None:
        """ArtifactBank with SQLite-backed section via db_factory."""
        _call_log, factory = _make_sqlite_db_factory(tmp_path)

        artifact = _make_artifact_with_data(db_factory=factory)
        assert artifact.section("ingredients").count() == 2
        assert artifact.field("recipe_name") == "Chocolate Chip Cookies"

        # Verify factory was called for the section
        assert len(_call_log) == 1
        assert _call_log[0][0] == "ingredients"

        # Serialize — section should be external mode (no inline records)
        data = artifact.to_dict()
        section_dict = data["sections"]["ingredients"]
        assert section_dict["storage_mode"] == "external"
        assert "records" not in section_dict

        # Close section dbs
        artifact.section("ingredients").close()

    def test_artifact_restore_with_sqlite(self, tmp_path: Path) -> None:
        """ArtifactBank roundtrip: serialize, close, restore from same db."""
        db_path = str(tmp_path / "ingredients.db")

        # Create with factory
        def factory1(
            name: str, cfg: dict[str, Any]
        ) -> tuple[SyncDatabase, str]:
            db = SyncSQLiteDatabase({"path": db_path, "table": name})
            db.connect()
            return db, "external"

        artifact = _make_artifact_with_data(db_factory=factory1)
        data = artifact.to_dict()

        # Add section_configs so from_dict knows it's sqlite
        data["section_configs"] = {
            "ingredients": {
                "backend": "sqlite",
                "backend_config": {"path": db_path},
            },
        }

        artifact.section("ingredients").close()

        # Restore with a new factory that reconnects
        def factory2(
            name: str, cfg: dict[str, Any]
        ) -> tuple[SyncDatabase, str]:
            path = cfg.get("backend_config", {}).get("path", db_path)
            db = SyncSQLiteDatabase({"path": path, "table": name})
            db.connect()
            return db, "external"

        restored = ArtifactBank.from_dict(data, db_factory=factory2)
        assert restored.field("recipe_name") == "Chocolate Chip Cookies"
        assert restored.section("ingredients").count() == 2
        names = {
            r.data["name"]
            for r in restored.section("ingredients").all()
        }
        assert names == {"flour", "sugar"}
        restored.section("ingredients").close()


# =====================================================================
# Catalog SQLite backend tests
# =====================================================================

class TestCatalogSQLiteBackend:
    """Tests for ArtifactBankCatalog with SQLite backend."""

    def test_catalog_from_config_sqlite(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "catalog.db")
        catalog = ArtifactBankCatalog.from_config({
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })

        # Create and save an artifact
        artifact = _make_artifact_with_data()
        catalog.save(artifact)
        assert catalog.count() == 1

        entries = catalog.list()
        assert len(entries) == 1
        assert entries[0]["name"] == "recipe"

        result = catalog.get("recipe")
        assert result is not None
        assert result["recipe_name"] == "Chocolate Chip Cookies"

    def test_catalog_persistence(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "catalog.db")

        # Session 1: create catalog, save artifact
        catalog1 = ArtifactBankCatalog.from_config({
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        artifact = _make_artifact_with_data()
        catalog1.save(artifact)
        assert catalog1.count() == 1

        # Session 2: reopen catalog, artifact is still there
        catalog2 = ArtifactBankCatalog.from_config({
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        assert catalog2.count() == 1
        result = catalog2.get("recipe")
        assert result is not None
        assert result["recipe_name"] == "Chocolate Chip Cookies"

    def test_catalog_load_into_artifact(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "catalog.db")

        # Create catalog with saved artifact
        catalog = ArtifactBankCatalog.from_config({
            "backend": "sqlite",
            "backend_config": {"path": db_path},
        })
        artifact = _make_artifact_with_data()
        catalog.save(artifact)

        # Load into a fresh artifact
        fresh = ArtifactBank.from_config({
            "name": "recipe",
            "fields": {"recipe_name": {"required": True}},
            "sections": {
                "ingredients": {"schema": {"required": ["name"]}},
            },
        })
        assert fresh.section("ingredients").count() == 0

        loaded = catalog.load_into("recipe", fresh)
        assert loaded is True
        assert fresh.field("recipe_name") == "Chocolate Chip Cookies"
        assert fresh.section("ingredients").count() == 2
