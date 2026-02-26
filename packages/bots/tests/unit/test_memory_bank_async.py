"""Tests for AsyncMemoryBank — async variant backed by AsyncDatabase.

Covers Phase 4: async CRUD, duplicate detection, serialization modes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_bots.memory.bank import AsyncMemoryBank, BankRecord
from dataknobs_data.backends.memory import AsyncMemoryDatabase


# =====================================================================
# Helpers
# =====================================================================

async def _make_async_bank(
    name: str = "items",
    schema: dict[str, Any] | None = None,
    max_records: int | None = None,
    duplicate_strategy: str = "allow",
    match_fields: list[str] | None = None,
    storage_mode: str = "inline",
) -> AsyncMemoryBank:
    if schema is None:
        schema = {"required": ["name"]}
    return AsyncMemoryBank(
        name=name,
        schema=schema,
        db=AsyncMemoryDatabase(),
        max_records=max_records,
        duplicate_strategy=duplicate_strategy,
        match_fields=match_fields,
        storage_mode=storage_mode,
    )


# =====================================================================
# Async CRUD tests
# =====================================================================

class TestAsyncMemoryBankCRUD:

    @pytest.mark.asyncio
    async def test_add_and_get(self) -> None:
        bank = await _make_async_bank()
        rid = await bank.add({"name": "flour"}, source_stage="s1")
        assert rid
        record = await bank.get(rid)
        assert record is not None
        assert record.data["name"] == "flour"
        assert record.source_stage == "s1"

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        bank = await _make_async_bank()
        assert await bank.count() == 0
        await bank.add({"name": "flour"})
        assert await bank.count() == 1
        await bank.add({"name": "sugar"})
        assert await bank.count() == 2

    @pytest.mark.asyncio
    async def test_all_sorted(self) -> None:
        bank = await _make_async_bank()
        await bank.add({"name": "b"})
        await bank.add({"name": "a"})
        records = await bank.all()
        assert len(records) == 2
        assert records[0].data["name"] == "b"
        assert records[1].data["name"] == "a"

    @pytest.mark.asyncio
    async def test_update(self) -> None:
        bank = await _make_async_bank()
        rid = await bank.add({"name": "flour"})
        result = await bank.update(rid, {"name": "whole wheat"})
        assert result is True
        updated = await bank.get(rid)
        assert updated is not None
        assert updated.data["name"] == "whole wheat"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self) -> None:
        bank = await _make_async_bank()
        assert await bank.update("nope", {"name": "x"}) is False

    @pytest.mark.asyncio
    async def test_remove(self) -> None:
        bank = await _make_async_bank()
        rid = await bank.add({"name": "flour"})
        assert await bank.count() == 1
        assert await bank.remove(rid) is True
        assert await bank.count() == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self) -> None:
        bank = await _make_async_bank()
        assert await bank.remove("nope") is False

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        bank = await _make_async_bank()
        await bank.add({"name": "a"})
        await bank.add({"name": "b"})
        assert await bank.count() == 2
        await bank.clear()
        assert await bank.count() == 0


# =====================================================================
# Validation
# =====================================================================

class TestAsyncMemoryBankValidation:

    @pytest.mark.asyncio
    async def test_missing_required(self) -> None:
        bank = await _make_async_bank(schema={"required": ["name"]})
        with pytest.raises(ValueError, match="missing required fields"):
            await bank.add({"amount": "1 cup"})

    @pytest.mark.asyncio
    async def test_max_records(self) -> None:
        bank = await _make_async_bank(max_records=2)
        await bank.add({"name": "a"})
        await bank.add({"name": "b"})
        with pytest.raises(ValueError, match="full"):
            await bank.add({"name": "c"})


# =====================================================================
# Duplicate detection
# =====================================================================

class TestAsyncMemoryBankDuplicates:

    @pytest.mark.asyncio
    async def test_reject(self) -> None:
        bank = await _make_async_bank(duplicate_strategy="reject")
        rid1 = await bank.add({"name": "flour"})
        rid2 = await bank.add({"name": "flour"})
        assert rid1 == rid2
        assert await bank.count() == 1

    @pytest.mark.asyncio
    async def test_merge(self) -> None:
        bank = await _make_async_bank(
            duplicate_strategy="merge",
            match_fields=["name"],
        )
        rid1 = await bank.add({"name": "flour", "amount": "1 cup"})
        rid2 = await bank.add({"name": "flour", "amount": "2 cups"})
        assert rid1 == rid2
        assert await bank.count() == 1
        rec = await bank.get(rid1)
        assert rec is not None
        assert rec.data["amount"] == "2 cups"


# =====================================================================
# Find
# =====================================================================

class TestAsyncMemoryBankFind:

    @pytest.mark.asyncio
    async def test_find(self) -> None:
        bank = await _make_async_bank()
        await bank.add({"name": "flour", "cat": "dry"})
        await bank.add({"name": "milk", "cat": "wet"})
        results = await bank.find(cat="dry")
        assert len(results) == 1
        assert results[0].data["name"] == "flour"


# =====================================================================
# Serialization
# =====================================================================

class TestAsyncMemoryBankSerialization:

    @pytest.mark.asyncio
    async def test_inline_roundtrip(self) -> None:
        bank = await _make_async_bank(storage_mode="inline")
        await bank.add({"name": "flour"})
        await bank.add({"name": "sugar"})

        d = await bank.to_dict()
        assert len(d["records"]) == 2
        assert d["storage_mode"] == "inline"

        restored = await AsyncMemoryBank.from_dict(d)
        assert await restored.count() == 2
        records = await restored.all()
        assert records[0].data["name"] == "flour"

    @pytest.mark.asyncio
    async def test_external_mode_no_records(self) -> None:
        bank = await _make_async_bank(storage_mode="external")
        await bank.add({"name": "flour"})

        d = await bank.to_dict()
        # External mode should NOT include records
        assert "records" not in d
        assert d["storage_mode"] == "external"

    @pytest.mark.asyncio
    async def test_to_dict_is_json_safe(self) -> None:
        bank = await _make_async_bank()
        await bank.add({"name": "flour"})
        d = await bank.to_dict()
        json_str = json.dumps(d)
        assert json_str

    @pytest.mark.asyncio
    async def test_empty_bank_roundtrip(self) -> None:
        bank = await _make_async_bank()
        d = await bank.to_dict()
        restored = await AsyncMemoryBank.from_dict(d)
        assert await restored.count() == 0
        assert restored.name == "items"
