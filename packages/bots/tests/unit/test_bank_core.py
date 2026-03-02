"""Tests for _BankCore shared logic, Protocol conformance, and storage layer.

Covers Phases 2–3 of 03a: _BankCore logic, Protocol conformance (Phase 2),
storage_id as record key, direct lookups, Query/Filter find (Phase 3).
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.memory.bank import (
    AsyncBankProtocol,
    AsyncMemoryBank,
    BankRecord,
    EmptyBankProxy,
    MemoryBank,
    SyncBankProtocol,
    _BankCore,
)
from dataknobs_data import Record
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_core(
    name: str = "items",
    schema: dict[str, Any] | None = None,
    max_records: int | None = None,
    duplicate_strategy: str = "allow",
    match_fields: list[str] | None = None,
    storage_mode: str = "inline",
) -> _BankCore:
    if schema is None:
        schema = {"required": ["name"]}
    return _BankCore(
        name=name,
        schema=schema,
        max_records=max_records,
        duplicate_strategy=duplicate_strategy,
        match_fields=match_fields,
        storage_mode=storage_mode,
    )


# =====================================================================
# Protocol conformance
# =====================================================================


class TestSyncBankProtocolConformance:
    """Verify MemoryBank and EmptyBankProxy satisfy SyncBankProtocol."""

    def test_memory_bank_is_sync_protocol(self) -> None:
        bank = MemoryBank("test", {}, SyncMemoryDatabase())
        assert isinstance(bank, SyncBankProtocol)

    def test_empty_bank_proxy_is_sync_protocol(self) -> None:
        proxy = EmptyBankProxy("test")
        assert isinstance(proxy, SyncBankProtocol)


class TestAsyncBankProtocolConformance:
    """Verify AsyncMemoryBank satisfies AsyncBankProtocol."""

    def test_async_memory_bank_is_async_protocol(self) -> None:
        bank = AsyncMemoryBank("test", {}, AsyncMemoryDatabase())
        assert isinstance(bank, AsyncBankProtocol)


# =====================================================================
# _BankCore properties
# =====================================================================


class TestBankCoreProperties:
    """Test _BankCore configuration properties."""

    def test_name(self) -> None:
        core = _make_core(name="ingredients")
        assert core.name == "ingredients"

    def test_schema(self) -> None:
        core = _make_core(schema={"required": ["name", "amount"]})
        assert core.schema == {"required": ["name", "amount"]}

    def test_match_fields(self) -> None:
        core = _make_core(match_fields=["name"])
        assert core.match_fields == ["name"]

    def test_match_fields_default_none(self) -> None:
        core = _make_core()
        assert core.match_fields is None

    def test_max_records(self) -> None:
        core = _make_core(max_records=10)
        assert core.max_records == 10

    def test_duplicate_strategy(self) -> None:
        core = _make_core(duplicate_strategy="reject")
        assert core.duplicate_strategy == "reject"

    def test_storage_mode(self) -> None:
        core = _make_core(storage_mode="external")
        assert core.storage_mode == "external"


# =====================================================================
# _BankCore.validate()
# =====================================================================


class TestBankCoreValidate:
    """Test _BankCore.validate() — required field checking."""

    def test_raises_on_missing_required(self) -> None:
        core = _make_core(schema={"required": ["name"]})
        with pytest.raises(ValueError, match="missing required fields"):
            core.validate({"amount": "1 cup"})

    def test_raises_on_none_required(self) -> None:
        core = _make_core(schema={"required": ["name"]})
        with pytest.raises(ValueError, match="missing required fields"):
            core.validate({"name": None})

    def test_passes_with_all_required(self) -> None:
        core = _make_core(schema={"required": ["name"]})
        core.validate({"name": "flour"})  # No exception

    def test_passes_with_no_required_fields(self) -> None:
        core = _make_core(schema={})
        core.validate({"arbitrary": "value"})  # No exception


# =====================================================================
# _BankCore.check_capacity()
# =====================================================================


class TestBankCoreCheckCapacity:
    """Test _BankCore.check_capacity() — max_records enforcement."""

    def test_under_limit(self) -> None:
        core = _make_core(max_records=5)
        core.check_capacity(3)  # No exception

    def test_at_limit_raises(self) -> None:
        core = _make_core(max_records=5)
        with pytest.raises(ValueError, match="full"):
            core.check_capacity(5)

    def test_no_limit(self) -> None:
        core = _make_core(max_records=None)
        core.check_capacity(10000)  # No exception


# =====================================================================
# _BankCore.check_duplicate()
# =====================================================================


class TestBankCoreCheckDuplicate:
    """Test _BankCore.check_duplicate() — duplicate detection logic."""

    def test_finds_exact_match(self) -> None:
        core = _make_core()
        existing = [
            BankRecord(record_id="abc", data={"name": "flour"}),
        ]
        result = core.check_duplicate({"name": "flour"}, existing)
        assert result is not None
        assert result.record_id == "abc"

    def test_no_match(self) -> None:
        core = _make_core()
        existing = [
            BankRecord(record_id="abc", data={"name": "flour"}),
        ]
        result = core.check_duplicate({"name": "sugar"}, existing)
        assert result is None

    def test_respects_match_fields(self) -> None:
        core = _make_core(match_fields=["name"])
        existing = [
            BankRecord(
                record_id="abc",
                data={"name": "flour", "amount": "1 cup"},
            ),
        ]
        # Same name, different amount — still a match on match_fields
        result = core.check_duplicate(
            {"name": "flour", "amount": "2 cups"}, existing
        )
        assert result is not None

    def test_union_of_keys_prevents_false_duplicate(self) -> None:
        """Extra fields on existing record prevent false duplicate."""
        core = _make_core()
        existing = [
            BankRecord(
                record_id="abc",
                data={"name": "flour", "amount": "2 cups"},
            ),
        ]
        # New record has only "name" — existing has "amount" too
        result = core.check_duplicate({"name": "flour"}, existing)
        assert result is None

    def test_empty_existing_records(self) -> None:
        core = _make_core()
        result = core.check_duplicate({"name": "flour"}, [])
        assert result is None


# =====================================================================
# _BankCore.create_bank_record()
# =====================================================================


class TestBankCoreCreateBankRecord:
    """Test _BankCore.create_bank_record() — record creation."""

    def test_creates_bank_record_and_db_record(self) -> None:
        core = _make_core()
        bank_record, db_record = core.create_bank_record(
            {"name": "flour"}, source_stage="collect"
        )

        assert len(bank_record.record_id) == 12
        assert bank_record.data == {"name": "flour"}
        assert bank_record.source_stage == "collect"
        assert bank_record.created_at > 0
        assert bank_record.updated_at == bank_record.created_at
        assert bank_record.modified_in_stage is None

        assert db_record.data == {"name": "flour"}
        meta = db_record.metadata
        assert meta["record_id"] == bank_record.record_id
        assert meta["source_stage"] == "collect"
        assert meta["created_at"] == bank_record.created_at
        assert meta["updated_at"] == bank_record.updated_at

    def test_generates_unique_ids(self) -> None:
        core = _make_core()
        r1, _ = core.create_bank_record({"name": "a"})
        r2, _ = core.create_bank_record({"name": "b"})
        assert r1.record_id != r2.record_id


# =====================================================================
# _BankCore.create_updated_record()
# =====================================================================


class TestBankCoreCreateUpdatedRecord:
    """Test _BankCore.create_updated_record() — update record creation."""

    def test_preserves_existing_metadata(self) -> None:
        core = _make_core()
        existing_meta = {
            "record_id": "abc123",
            "source_stage": "collect",
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }
        updated = core.create_updated_record(
            {"name": "flour updated"}, existing_meta
        )
        assert updated.data == {"name": "flour updated"}
        meta = updated.metadata
        assert meta["record_id"] == "abc123"
        assert meta["source_stage"] == "collect"
        assert meta["created_at"] == 1000.0
        assert meta["updated_at"] > 1000.0

    def test_sets_modified_in_stage(self) -> None:
        core = _make_core()
        existing_meta = {"record_id": "abc123", "created_at": 1000.0}
        updated = core.create_updated_record(
            {"name": "flour"}, existing_meta, modified_in_stage="review"
        )
        assert updated.metadata["modified_in_stage"] == "review"

    def test_no_modified_in_stage_when_empty(self) -> None:
        core = _make_core()
        existing_meta = {"record_id": "abc123"}
        updated = core.create_updated_record(
            {"name": "flour"}, existing_meta
        )
        assert "modified_in_stage" not in updated.metadata


# =====================================================================
# _BankCore.to_bank_record()
# =====================================================================


class TestBankCoreToBankRecord:
    """Test _BankCore.to_bank_record() — Record → BankRecord conversion."""

    def test_converts_correctly(self) -> None:
        db_record = Record(
            data={"name": "flour"},
            metadata={
                "record_id": "abc123",
                "source_stage": "collect",
                "created_at": 1000.0,
                "updated_at": 2000.0,
                "modified_in_stage": "review",
            },
        )
        bank_record = _BankCore.to_bank_record(db_record)
        assert bank_record.record_id == "abc123"
        assert bank_record.data == {"name": "flour"}
        assert bank_record.source_stage == "collect"
        assert bank_record.created_at == 1000.0
        assert bank_record.updated_at == 2000.0
        assert bank_record.modified_in_stage == "review"

    def test_modified_in_stage_defaults_to_none(self) -> None:
        """Bug fix: modified_in_stage should default to None, not empty string."""
        db_record = Record(
            data={"name": "flour"},
            metadata={
                "record_id": "abc123",
                "source_stage": "collect",
                "created_at": 1000.0,
                "updated_at": 1000.0,
            },
        )
        bank_record = _BankCore.to_bank_record(db_record)
        assert bank_record.modified_in_stage is None

    def test_handles_empty_metadata(self) -> None:
        db_record = Record(data={"name": "flour"}, metadata={})
        bank_record = _BankCore.to_bank_record(db_record)
        assert bank_record.record_id == ""
        assert bank_record.data == {"name": "flour"}
        assert bank_record.modified_in_stage is None


# =====================================================================
# _BankCore.bank_record_to_db_record()
# =====================================================================


class TestBankCoreRecordToDbRecord:
    """Test _BankCore.bank_record_to_db_record() — BankRecord → Record."""

    def test_round_trips_correctly(self) -> None:
        bank_record = BankRecord(
            record_id="abc123",
            data={"name": "flour"},
            source_stage="collect",
            created_at=1000.0,
            updated_at=2000.0,
            modified_in_stage="review",
        )
        db_record = _BankCore.bank_record_to_db_record(bank_record)
        restored = _BankCore.to_bank_record(db_record)

        assert restored.record_id == bank_record.record_id
        assert restored.data == bank_record.data
        assert restored.source_stage == bank_record.source_stage
        assert restored.created_at == bank_record.created_at
        assert restored.updated_at == bank_record.updated_at
        assert restored.modified_in_stage == bank_record.modified_in_stage


# =====================================================================
# _BankCore serialization helpers
# =====================================================================


class TestBankCoreSerializationHelpers:
    """Test _BankCore.to_config_dict() and extract_config()."""

    def test_to_config_dict(self) -> None:
        core = _make_core(
            name="items",
            max_records=10,
            duplicate_strategy="reject",
            match_fields=["name"],
            storage_mode="external",
        )
        d = core.to_config_dict()
        assert d == {
            "name": "items",
            "schema": {"required": ["name"]},
            "max_records": 10,
            "duplicate_strategy": "reject",
            "match_fields": ["name"],
            "storage_mode": "external",
        }

    def test_extract_config(self) -> None:
        d = {
            "name": "items",
            "schema": {"required": ["name"]},
            "max_records": 5,
            "duplicate_strategy": "merge",
            "match_fields": ["name"],
            "storage_mode": "inline",
        }
        config = _BankCore.extract_config(d)
        assert config["name"] == "items"
        assert config["schema"] == {"required": ["name"]}
        assert config["max_records"] == 5
        assert config["duplicate_strategy"] == "merge"
        assert config["match_fields"] == ["name"]
        assert config["storage_mode"] == "inline"

    def test_extract_config_defaults(self) -> None:
        d = {"name": "items"}
        config = _BankCore.extract_config(d)
        assert config["schema"] == {}
        assert config["max_records"] is None
        assert config["duplicate_strategy"] == "allow"
        assert config["match_fields"] is None
        assert config["storage_mode"] == "inline"


# =====================================================================
# EmptyBankProxy new methods
# =====================================================================


class TestEmptyBankProxyNewMethods:
    """Test methods added to EmptyBankProxy for SyncBankProtocol conformance."""

    def test_to_dict(self) -> None:
        proxy = EmptyBankProxy("missing")
        d = proxy.to_dict()
        assert d["name"] == "missing"
        assert d["records"] == []
        assert d["schema"] == {}
        assert d["storage_mode"] == "inline"

    def test_close_noop(self) -> None:
        proxy = EmptyBankProxy("missing")
        proxy.close()  # No error


# =====================================================================
# Phase 3: Storage ID as record key
# =====================================================================


class TestBankCoreStorageId:
    """Verify storage_id is set on all created Records (Phase 3, AD-3)."""

    def test_create_bank_record_sets_storage_id(self) -> None:
        core = _make_core()
        bank_record, db_record = core.create_bank_record({"name": "flour"})
        assert db_record.storage_id == bank_record.record_id

    def test_bank_record_to_db_record_sets_storage_id(self) -> None:
        bank_record = BankRecord(
            record_id="abc123",
            data={"name": "flour"},
            created_at=1000.0,
            updated_at=1000.0,
        )
        db_record = _BankCore.bank_record_to_db_record(bank_record)
        assert db_record.storage_id == "abc123"

    def test_create_updated_record_sets_storage_id(self) -> None:
        core = _make_core()
        existing_meta = {
            "record_id": "abc123",
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }
        updated = core.create_updated_record(
            {"name": "flour updated"}, existing_meta
        )
        assert updated.storage_id == "abc123"

    def test_create_updated_record_no_record_id_no_storage_id(self) -> None:
        core = _make_core()
        existing_meta = {"created_at": 1000.0, "updated_at": 1000.0}
        updated = core.create_updated_record(
            {"name": "flour"}, existing_meta
        )
        # No record_id in metadata → storage_id not set
        assert updated.storage_id is None


class TestDirectLookupSync:
    """Verify MemoryBank uses direct DB lookups (Phase 3, AD-10)."""

    def test_get_uses_direct_lookup(self) -> None:
        """Record added via add() is retrievable by db.read(record_id)."""
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        record_id = bank.add({"name": "flour"})

        # Direct DB lookup should work (proves storage_id alignment)
        db_record = db.read(record_id)
        assert db_record is not None
        assert db_record.data == {"name": "flour"}

    def test_update_via_direct_lookup(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        record_id = bank.add({"name": "flour"})
        bank.update(record_id, {"name": "sugar"})

        # Verify via direct DB read
        db_record = db.read(record_id)
        assert db_record is not None
        assert db_record.data == {"name": "sugar"}

    def test_remove_via_direct_lookup(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        record_id = bank.add({"name": "flour"})
        assert bank.remove(record_id) is True
        assert db.read(record_id) is None

    def test_get_nonexistent_returns_none(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {}, db)
        assert bank.get("nonexistent") is None

    def test_remove_nonexistent_returns_false(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {}, db)
        assert bank.remove("nonexistent") is False


class TestDirectLookupAsync:
    """Verify AsyncMemoryBank uses direct DB lookups (Phase 3, AD-10)."""

    @pytest.mark.asyncio
    async def test_get_uses_direct_lookup(self) -> None:
        db = AsyncMemoryDatabase()
        bank = AsyncMemoryBank("items", {"required": ["name"]}, db)
        record_id = await bank.add({"name": "flour"})

        db_record = await db.read(record_id)
        assert db_record is not None
        assert db_record.data == {"name": "flour"}

    @pytest.mark.asyncio
    async def test_update_via_direct_lookup(self) -> None:
        db = AsyncMemoryDatabase()
        bank = AsyncMemoryBank("items", {"required": ["name"]}, db)
        record_id = await bank.add({"name": "flour"})
        await bank.update(record_id, {"name": "sugar"})

        db_record = await db.read(record_id)
        assert db_record is not None
        assert db_record.data == {"name": "sugar"}

    @pytest.mark.asyncio
    async def test_remove_via_direct_lookup(self) -> None:
        db = AsyncMemoryDatabase()
        bank = AsyncMemoryBank("items", {"required": ["name"]}, db)
        record_id = await bank.add({"name": "flour"})
        assert await bank.remove(record_id) is True
        assert await db.read(record_id) is None


class TestQueryFilterFind:
    """Verify find() uses Query/Filter for DB-level filtering (Phase 3)."""

    def test_find_with_field_values(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        bank.add({"name": "flour", "type": "dry"})
        bank.add({"name": "sugar", "type": "dry"})
        bank.add({"name": "milk", "type": "wet"})

        results = bank.find(type="dry")
        assert len(results) == 2
        names = {r.data["name"] for r in results}
        assert names == {"flour", "sugar"}

    def test_find_no_matches(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        bank.add({"name": "flour"})
        assert bank.find(name="nonexistent") == []

    def test_find_no_field_values_returns_all(self) -> None:
        db = SyncMemoryDatabase()
        bank = MemoryBank("items", {"required": ["name"]}, db)
        bank.add({"name": "flour"})
        bank.add({"name": "sugar"})
        results = bank.find()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_async_find_with_field_values(self) -> None:
        db = AsyncMemoryDatabase()
        bank = AsyncMemoryBank("items", {"required": ["name"]}, db)
        await bank.add({"name": "flour", "type": "dry"})
        await bank.add({"name": "sugar", "type": "dry"})
        await bank.add({"name": "milk", "type": "wet"})

        results = await bank.find(type="dry")
        assert len(results) == 2
        names = {r.data["name"] for r in results}
        assert names == {"flour", "sugar"}


class TestFromDictBackwardCompat:
    """Verify from_dict() restores old-format records with correct storage_id."""

    def test_sync_from_dict_restores_with_storage_id(self) -> None:
        """Records from from_dict() are accessible via direct lookup."""
        bank = MemoryBank("items", {"required": ["name"]}, SyncMemoryDatabase())
        rid = bank.add({"name": "flour"})
        serialized = bank.to_dict()

        # Restore
        restored = MemoryBank.from_dict(serialized)
        record = restored.get(rid)
        assert record is not None
        assert record.data == {"name": "flour"}

    def test_sync_from_dict_update_and_remove_work(self) -> None:
        bank = MemoryBank("items", {"required": ["name"]}, SyncMemoryDatabase())
        rid = bank.add({"name": "flour"})
        serialized = bank.to_dict()

        restored = MemoryBank.from_dict(serialized)
        assert restored.update(rid, {"name": "sugar"}) is True
        assert restored.get(rid).data == {"name": "sugar"}
        assert restored.remove(rid) is True
        assert restored.get(rid) is None

    @pytest.mark.asyncio
    async def test_async_from_dict_restores_with_storage_id(self) -> None:
        db = AsyncMemoryDatabase()
        bank = AsyncMemoryBank("items", {"required": ["name"]}, db)
        rid = await bank.add({"name": "flour"})
        serialized = await bank.to_dict()

        restored = await AsyncMemoryBank.from_dict(serialized)
        record = await restored.get(rid)
        assert record is not None
        assert record.data == {"name": "flour"}
