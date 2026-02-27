"""Tests for MemoryBank — typed collection of structured records.

Covers Phase 1 (core CRUD, serialization, EmptyBankProxy) and
Phase 3 (duplicate detection, find) of the MemoryBank abstraction.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_bots.memory.bank import BankRecord, EmptyBankProxy, MemoryBank
from dataknobs_bots.reasoning.wizard import WizardReasoning
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_llm.conversations import ConversationManager


# =====================================================================
# Helpers
# =====================================================================

def _make_bank(
    name: str = "items",
    schema: dict[str, Any] | None = None,
    max_records: int | None = None,
    duplicate_strategy: str = "allow",
    match_fields: list[str] | None = None,
) -> MemoryBank:
    """Create a MemoryBank backed by SyncMemoryDatabase."""
    if schema is None:
        schema = {"required": ["name"]}
    return MemoryBank(
        name=name,
        schema=schema,
        db=SyncMemoryDatabase(),
        max_records=max_records,
        duplicate_strategy=duplicate_strategy,
        match_fields=match_fields,
    )


def _make_wizard_with_banks(
    banks_config: dict[str, Any] | None = None,
) -> WizardReasoning:
    """Create a WizardReasoning with bank config."""
    config: dict[str, Any] = {
        "name": "bank-wizard",
        "version": "1.0",
        "settings": {
            "banks": banks_config or {
                "ingredients": {
                    "schema": {
                        "required": ["name"],
                    },
                    "max_records": 50,
                },
            },
        },
        "stages": [
            {
                "name": "collect",
                "is_start": True,
                "prompt": "Add an ingredient",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount": {"type": "string"},
                    },
                    "required": ["name"],
                },
                "transitions": [
                    {
                        "target": "review",
                        "condition": (
                            "bank('ingredients').count() > 0"
                        ),
                    },
                ],
            },
            {
                "name": "review",
                "is_end": True,
                "prompt": "Here are your ingredients",
                "response_template": (
                    "You added {{ bank('ingredients').count() }} items."
                ),
            },
        ],
    }
    loader = WizardConfigLoader()
    fsm = loader.load_from_dict(config)
    return WizardReasoning(wizard_fsm=fsm, strict_validation=False)


# =====================================================================
# BankRecord tests
# =====================================================================

class TestBankRecord:

    def test_to_dict_roundtrip(self) -> None:
        record = BankRecord(
            record_id="abc123",
            data={"name": "flour", "amount": "2 cups"},
            source_stage="collect",
            created_at=1000.0,
            updated_at=1001.0,
        )
        d = record.to_dict()
        restored = BankRecord.from_dict(d)
        assert restored.record_id == "abc123"
        assert restored.data == {"name": "flour", "amount": "2 cups"}
        assert restored.source_stage == "collect"
        assert restored.created_at == 1000.0
        assert restored.updated_at == 1001.0

    def test_to_dict_is_json_safe(self) -> None:
        record = BankRecord(
            record_id="x",
            data={"nested": {"a": [1, 2]}},
        )
        json_str = json.dumps(record.to_dict())
        assert json_str  # No serialization error


# =====================================================================
# MemoryBank CRUD tests
# =====================================================================

class TestMemoryBankCRUD:

    def test_add_and_get(self) -> None:
        bank = _make_bank()
        rid = bank.add({"name": "flour", "amount": "2 cups"}, source_stage="s1")
        assert rid
        record = bank.get(rid)
        assert record is not None
        assert record.data["name"] == "flour"
        assert record.source_stage == "s1"

    def test_count(self) -> None:
        bank = _make_bank()
        assert bank.count() == 0
        bank.add({"name": "flour"})
        assert bank.count() == 1
        bank.add({"name": "sugar"})
        assert bank.count() == 2

    def test_all_returns_sorted_by_creation(self) -> None:
        bank = _make_bank()
        bank.add({"name": "b"})
        bank.add({"name": "a"})
        records = bank.all()
        assert len(records) == 2
        assert records[0].data["name"] == "b"
        assert records[1].data["name"] == "a"

    def test_update(self) -> None:
        bank = _make_bank()
        rid = bank.add({"name": "flour"})
        result = bank.update(rid, {"name": "whole wheat flour"})
        assert result is True
        updated = bank.get(rid)
        assert updated is not None
        assert updated.data["name"] == "whole wheat flour"

    def test_update_nonexistent(self) -> None:
        bank = _make_bank()
        assert bank.update("nonexistent", {"name": "x"}) is False

    def test_remove(self) -> None:
        bank = _make_bank()
        rid = bank.add({"name": "flour"})
        assert bank.count() == 1
        result = bank.remove(rid)
        assert result is True
        assert bank.count() == 0
        assert bank.get(rid) is None

    def test_remove_nonexistent(self) -> None:
        bank = _make_bank()
        assert bank.remove("nonexistent") is False

    def test_clear(self) -> None:
        bank = _make_bank()
        bank.add({"name": "a"})
        bank.add({"name": "b"})
        assert bank.count() == 2
        bank.clear()
        assert bank.count() == 0


# =====================================================================
# Validation tests
# =====================================================================

class TestMemoryBankValidation:

    def test_missing_required_field_rejected(self) -> None:
        bank = _make_bank(schema={"required": ["name", "amount"]})
        with pytest.raises(ValueError, match="missing required fields"):
            bank.add({"name": "flour"})  # Missing "amount"

    def test_none_required_field_rejected(self) -> None:
        bank = _make_bank(schema={"required": ["name"]})
        with pytest.raises(ValueError, match="missing required fields"):
            bank.add({"name": None})

    def test_no_required_fields_accepts_anything(self) -> None:
        bank = _make_bank(schema={})
        rid = bank.add({"arbitrary": "value"})
        assert bank.get(rid) is not None


# =====================================================================
# Max records tests
# =====================================================================

class TestMemoryBankMaxRecords:

    def test_max_records_enforced(self) -> None:
        bank = _make_bank(max_records=2)
        bank.add({"name": "a"})
        bank.add({"name": "b"})
        with pytest.raises(ValueError, match="full"):
            bank.add({"name": "c"})

    def test_no_max_records_unlimited(self) -> None:
        bank = _make_bank(max_records=None)
        for i in range(100):
            bank.add({"name": f"item-{i}"})
        assert bank.count() == 100


# =====================================================================
# Duplicate detection tests (Phase 3)
# =====================================================================

class TestMemoryBankDuplicates:

    def test_allow_strategy_permits_duplicates(self) -> None:
        bank = _make_bank(duplicate_strategy="allow")
        bank.add({"name": "flour"})
        bank.add({"name": "flour"})
        assert bank.count() == 2

    def test_reject_strategy_prevents_duplicates(self) -> None:
        bank = _make_bank(duplicate_strategy="reject")
        rid1 = bank.add({"name": "flour"})
        rid2 = bank.add({"name": "flour"})
        assert rid1 == rid2  # Returns existing ID
        assert bank.count() == 1

    def test_merge_strategy_updates_existing(self) -> None:
        bank = _make_bank(
            schema={"required": ["name"]},
            duplicate_strategy="merge",
            match_fields=["name"],
        )
        rid1 = bank.add({"name": "flour", "amount": "1 cup"})
        rid2 = bank.add({"name": "flour", "amount": "2 cups"})
        assert rid1 == rid2
        assert bank.count() == 1
        record = bank.get(rid1)
        assert record is not None
        assert record.data["amount"] == "2 cups"  # Merged

    def test_match_fields_scopes_comparison(self) -> None:
        bank = _make_bank(
            duplicate_strategy="reject",
            match_fields=["name"],
        )
        bank.add({"name": "flour", "amount": "1 cup"})
        rid2 = bank.add({"name": "flour", "amount": "2 cups"})
        assert bank.count() == 1  # Same name → duplicate

    def test_different_match_fields_not_duplicate(self) -> None:
        bank = _make_bank(
            duplicate_strategy="reject",
            match_fields=["name"],
        )
        bank.add({"name": "flour"})
        bank.add({"name": "sugar"})
        assert bank.count() == 2


# =====================================================================
# Find tests
# =====================================================================

class TestMemoryBankFind:

    def test_find_by_field(self) -> None:
        bank = _make_bank()
        bank.add({"name": "flour", "category": "dry"})
        bank.add({"name": "milk", "category": "wet"})
        bank.add({"name": "sugar", "category": "dry"})
        results = bank.find(category="dry")
        assert len(results) == 2
        names = {r.data["name"] for r in results}
        assert names == {"flour", "sugar"}

    def test_find_no_matches(self) -> None:
        bank = _make_bank()
        bank.add({"name": "flour"})
        assert bank.find(name="sugar") == []


# =====================================================================
# Serialization tests
# =====================================================================

class TestMemoryBankSerialization:

    def test_to_dict_roundtrip(self) -> None:
        bank = _make_bank(max_records=10)
        bank.add({"name": "flour", "amount": "2 cups"}, source_stage="s1")
        bank.add({"name": "sugar", "amount": "1 cup"}, source_stage="s1")

        d = bank.to_dict()
        restored = MemoryBank.from_dict(d)

        assert restored.name == "items"
        assert restored.count() == 2
        records = restored.all()
        assert records[0].data["name"] == "flour"
        assert records[1].data["name"] == "sugar"

    def test_to_dict_is_json_safe(self) -> None:
        bank = _make_bank()
        bank.add({"name": "flour"})
        json_str = json.dumps(bank.to_dict())
        assert json_str  # No serialization error

    def test_empty_bank_serialization(self) -> None:
        bank = _make_bank()
        d = bank.to_dict()
        restored = MemoryBank.from_dict(d)
        assert restored.count() == 0
        assert restored.name == "items"

    def test_duplicate_config_survives_roundtrip(self) -> None:
        bank = _make_bank(
            duplicate_strategy="reject",
            match_fields=["name"],
        )
        bank.add({"name": "flour"})
        d = bank.to_dict()
        restored = MemoryBank.from_dict(d)
        assert restored.count() == 1
        assert d["duplicate_strategy"] == "reject"
        assert d["match_fields"] == ["name"]


# =====================================================================
# EmptyBankProxy tests
# =====================================================================

class TestEmptyBankProxy:

    def test_count_is_zero(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.count() == 0

    def test_all_is_empty(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.all() == []

    def test_get_is_none(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.get("any-id") is None

    def test_add_returns_empty_string(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.add({"name": "x"}) == ""

    def test_remove_returns_false(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.remove("x") is False

    def test_update_returns_false(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.update("x", {}) is False

    def test_clear_noop(self) -> None:
        proxy = EmptyBankProxy("missing")
        proxy.clear()  # No error

    def test_find_returns_empty(self) -> None:
        proxy = EmptyBankProxy("missing")
        assert proxy.find(name="x") == []

    def test_name_property(self) -> None:
        proxy = EmptyBankProxy("test-bank")
        assert proxy.name == "test-bank"


# =====================================================================
# WizardReasoning integration tests
# =====================================================================

class TestWizardBankIntegration:

    def test_banks_initialised_from_config(self) -> None:
        reasoning = _make_wizard_with_banks()
        assert "ingredients" in reasoning._banks
        bank = reasoning._banks["ingredients"]
        assert bank.name == "ingredients"
        assert bank.count() == 0

    def test_bank_accessor_returns_bank(self) -> None:
        reasoning = _make_wizard_with_banks()
        accessor = reasoning._make_bank_accessor()
        bank = accessor("ingredients")
        assert bank.name == "ingredients"

    def test_bank_accessor_returns_proxy_for_unknown(self) -> None:
        reasoning = _make_wizard_with_banks()
        accessor = reasoning._make_bank_accessor()
        proxy = accessor("nonexistent")
        assert isinstance(proxy, EmptyBankProxy)
        assert proxy.count() == 0

    def test_no_banks_config_creates_empty_dict(self) -> None:
        config: dict[str, Any] = {
            "name": "no-banks",
            "version": "1.0",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        loader = WizardConfigLoader()
        fsm = loader.load_from_dict(config)
        reasoning = WizardReasoning(wizard_fsm=fsm, strict_validation=False)
        assert reasoning._banks == {}

    def test_bank_condition_evaluation(self) -> None:
        reasoning = _make_wizard_with_banks()
        # Initially empty → condition should be False
        result = reasoning._evaluate_condition(
            "bank('ingredients').count() > 0",
            {},
        )
        assert result is False

        # Add a record and re-evaluate
        reasoning._banks["ingredients"].add({"name": "flour"})
        result = reasoning._evaluate_condition(
            "bank('ingredients').count() > 0",
            {},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_banks_persist_across_save_restore(
        self, conversation_manager: ConversationManager
    ) -> None:
        reasoning = _make_wizard_with_banks()
        state = reasoning._get_wizard_state(conversation_manager)

        # Add records to the bank
        reasoning._banks["ingredients"].add(
            {"name": "flour"}, source_stage="collect"
        )
        reasoning._banks["ingredients"].add(
            {"name": "sugar"}, source_stage="collect"
        )
        assert reasoning._banks["ingredients"].count() == 2

        # Save state
        await reasoning._save_wizard_state(conversation_manager, state)

        # Verify banks are in metadata
        wizard_meta = conversation_manager.metadata.get("wizard", {})
        assert "banks" in wizard_meta
        assert "ingredients" in wizard_meta["banks"]
        banks_data = wizard_meta["banks"]["ingredients"]
        assert len(banks_data["records"]) == 2

        # Create fresh reasoning and restore
        reasoning2 = _make_wizard_with_banks()
        state2 = reasoning2._get_wizard_state(conversation_manager)
        assert state2.current_stage == state.current_stage
        assert reasoning2._banks["ingredients"].count() == 2
        records = reasoning2._banks["ingredients"].all()
        assert records[0].data["name"] == "flour"
        assert records[1].data["name"] == "sugar"

    @pytest.mark.asyncio
    async def test_restart_clears_banks(
        self, conversation_manager: ConversationManager
    ) -> None:
        reasoning = _make_wizard_with_banks()
        state = reasoning._get_wizard_state(conversation_manager)
        reasoning._banks["ingredients"].add({"name": "flour"})
        assert reasoning._banks["ingredients"].count() == 1

        # Execute restart
        from dataknobs_llm.llm.providers.echo import EchoProvider
        from dataknobs_llm.llm import LLMConfig

        provider = EchoProvider(LLMConfig(provider="echo", model="test"))
        await reasoning._execute_restart(
            "restart", state, conversation_manager, provider
        )
        assert reasoning._banks["ingredients"].count() == 0

    def test_bank_in_template_context(self) -> None:
        reasoning = _make_wizard_with_banks()
        reasoning._banks["ingredients"].add({"name": "flour"})

        state = reasoning._get_wizard_state.__func__  # noqa: not calling
        # Test template rendering directly
        from dataknobs_bots.reasoning.wizard import WizardState

        ws = WizardState(current_stage="review", data={})
        stage = {"name": "review", "response_template": "test"}
        # _render_response_template is async, but we can test the
        # context building via the public API path indirectly
        # by verifying the bank accessor works in templates
        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        template = env.from_string(
            "Count: {{ bank('ingredients').count() }}"
        )
        accessor = reasoning._make_bank_accessor()
        result = template.render(bank=accessor)
        assert result == "Count: 1"

    @pytest.mark.asyncio
    async def test_banks_metadata_is_json_safe(
        self, conversation_manager: ConversationManager
    ) -> None:
        reasoning = _make_wizard_with_banks()
        state = reasoning._get_wizard_state(conversation_manager)
        reasoning._banks["ingredients"].add({"name": "flour"})
        await reasoning._save_wizard_state(conversation_manager, state)

        # Must be JSON serializable
        import json

        json_str = json.dumps(conversation_manager.metadata)
        parsed = json.loads(json_str)
        assert "banks" in parsed["wizard"]


# =====================================================================
# Storage mode tests
# =====================================================================

class TestMemoryBankStorageMode:

    def test_storage_mode_defaults_to_inline(self) -> None:
        bank = _make_bank()
        d = bank.to_dict()
        assert d["storage_mode"] == "inline"
        assert "records" in d

    def test_to_dict_external_omits_records(self) -> None:
        bank = MemoryBank(
            name="items",
            schema={"required": ["name"]},
            db=SyncMemoryDatabase(),
            storage_mode="external",
        )
        bank.add({"name": "flour"})
        d = bank.to_dict()
        assert d["storage_mode"] == "external"
        assert "records" not in d

    def test_from_dict_with_injected_db(self) -> None:
        """Provided db is used instead of creating a fresh SyncMemoryDatabase."""
        injected_db = SyncMemoryDatabase()
        bank_dict = {
            "name": "items",
            "schema": {"required": ["name"]},
            "storage_mode": "external",
        }
        bank = MemoryBank.from_dict(bank_dict, db=injected_db)
        # Add a record through the bank and verify it lands in the injected db
        bank.add({"name": "flour"})
        from dataknobs_data import Query

        raw_records = list(injected_db.search(Query()))
        assert len(raw_records) == 1
        assert raw_records[0].data["name"] == "flour"

    def test_sqlite_backend_round_trip(self) -> None:
        """Records persist in SQLite across MemoryBank instances."""
        from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

        db = SyncSQLiteDatabase({"path": ":memory:", "table": "items"})
        db.connect()

        bank = MemoryBank(
            name="items",
            schema={"required": ["name"]},
            db=db,
            storage_mode="external",
        )
        bank.add({"name": "flour"})
        bank.add({"name": "sugar"})
        assert bank.count() == 2

        # to_dict in external mode has no records
        d = bank.to_dict()
        assert "records" not in d

        # Reconstruct from same db — records are still there
        bank2 = MemoryBank.from_dict(d, db=db)
        assert bank2.count() == 2
        names = {r.data["name"] for r in bank2.all()}
        assert names == {"flour", "sugar"}

        db.close()


# =====================================================================
# Close / cleanup tests
# =====================================================================

class TestMemoryBankClose:

    def test_close_calls_db_close(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank(
            name="items",
            schema={},
            db=db,
        )
        # SyncMemoryDatabase.close() is a no-op — just verify no error
        bank.close()

    def test_close_on_sqlite_closes_connection(self) -> None:
        from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

        db = SyncSQLiteDatabase({"path": ":memory:", "table": "items"})
        db.connect()
        bank = MemoryBank(name="items", schema={}, db=db)
        bank.add({"name": "flour"})

        bank.close()
        assert not db._connected

    @pytest.mark.asyncio
    async def test_wizard_close_closes_bank_dbs(self) -> None:
        from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

        reasoning = _make_wizard_with_banks(banks_config={
            "ingredients": {
                "schema": {"required": ["name"]},
                "max_records": 50,
            },
        })
        # Replace the memory db with an SQLite db so we can observe close
        sqlite_db = SyncSQLiteDatabase(
            {"path": ":memory:", "table": "ingredients"}
        )
        sqlite_db.connect()
        reasoning._banks["ingredients"] = MemoryBank(
            name="ingredients",
            schema={"required": ["name"]},
            db=sqlite_db,
            storage_mode="external",
        )
        reasoning._banks["ingredients"].add({"name": "flour"})
        assert sqlite_db._connected

        await reasoning.close()
        assert not sqlite_db._connected
