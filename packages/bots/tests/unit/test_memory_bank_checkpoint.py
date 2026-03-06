"""Tests for MemoryBank checkpoint/undo and source_node_id provenance.

Phase 2a of the conversation undo plan: bank provenance and checkpointing.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.memory.bank import (
    AsyncMemoryBank,
    BankRecord,
    MemoryBank,
    _BankCore,
)
from dataknobs_data.backends.memory import (
    AsyncMemoryDatabase,
    SyncMemoryDatabase,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_bank(name: str = "items") -> MemoryBank:
    return MemoryBank(
        name=name,
        schema={"required": ["name"]},
        db=SyncMemoryDatabase(),
    )


async def _make_async_bank(name: str = "items") -> AsyncMemoryBank:
    return AsyncMemoryBank(
        name=name,
        schema={"required": ["name"]},
        db=AsyncMemoryDatabase(),
    )


# =====================================================================
# BankRecord source_node_id
# =====================================================================


class TestBankRecordSourceNodeId:
    """BankRecord includes source_node_id in serialization."""

    def test_default_is_empty(self):
        record = BankRecord(record_id="abc", data={"name": "flour"})
        assert record.source_node_id == ""

    def test_to_dict_includes_source_node_id(self):
        record = BankRecord(
            record_id="abc",
            data={"name": "flour"},
            source_node_id="0.0.0",
        )
        d = record.to_dict()
        assert d["source_node_id"] == "0.0.0"

    def test_from_dict_roundtrip(self):
        original = BankRecord(
            record_id="abc",
            data={"name": "flour"},
            source_stage="collect",
            source_node_id="0.0.0.0",
            created_at=1000.0,
            updated_at=1000.0,
        )
        restored = BankRecord.from_dict(original.to_dict())
        assert restored.source_node_id == "0.0.0.0"
        assert restored.source_stage == "collect"
        assert restored.record_id == "abc"

    def test_from_dict_missing_source_node_id_defaults_empty(self):
        d = {
            "record_id": "abc",
            "data": {"name": "flour"},
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }
        record = BankRecord.from_dict(d)
        assert record.source_node_id == ""


# =====================================================================
# Node ancestry helper
# =====================================================================


class TestIsAncestorOrEqual:
    """_BankCore.is_ancestor_or_equal() for checkpoint logic."""

    def test_equal_nodes(self):
        assert _BankCore.is_ancestor_or_equal("0.0.0", "0.0.0") is True

    def test_root_is_ancestor_of_everything(self):
        assert _BankCore.is_ancestor_or_equal("", "0.0.0") is True

    def test_parent_is_ancestor(self):
        assert _BankCore.is_ancestor_or_equal("0.0", "0.0.0") is True

    def test_grandparent_is_ancestor(self):
        assert _BankCore.is_ancestor_or_equal("0", "0.0.0") is True

    def test_child_is_not_ancestor(self):
        assert _BankCore.is_ancestor_or_equal("0.0.0", "0.0") is False

    def test_sibling_is_not_ancestor(self):
        assert _BankCore.is_ancestor_or_equal("0.1", "0.0.0") is False

    def test_different_branch(self):
        assert _BankCore.is_ancestor_or_equal("0.0.1", "0.0.0") is False

    def test_prefix_not_at_boundary(self):
        """Node "0.1" should NOT be an ancestor of "0.10"."""
        assert _BankCore.is_ancestor_or_equal("0.1", "0.10") is False

    def test_both_empty(self):
        assert _BankCore.is_ancestor_or_equal("", "") is True


# =====================================================================
# MemoryBank.add() with source_node_id
# =====================================================================


class TestMemoryBankAddSourceNodeId:
    """MemoryBank.add() stores source_node_id on the created record."""

    def test_add_stores_source_node_id(self):
        bank = _make_bank()
        rid = bank.add(
            {"name": "flour"}, source_stage="collect", source_node_id="0.0.0"
        )
        record = bank.get(rid)
        assert record is not None
        assert record.source_node_id == "0.0.0"

    def test_add_without_source_node_id(self):
        bank = _make_bank()
        rid = bank.add({"name": "flour"})
        record = bank.get(rid)
        assert record is not None
        assert record.source_node_id == ""

    def test_serialization_roundtrip_preserves_source_node_id(self):
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0.0")
        bank.add({"name": "sugar"}, source_node_id="0.0.0.0")

        d = bank.to_dict()
        restored = MemoryBank.from_dict(d)

        records = restored.all()
        assert len(records) == 2
        node_ids = {r.source_node_id for r in records}
        assert node_ids == {"0.0.0", "0.0.0.0"}


# =====================================================================
# MemoryBank.undo_to_checkpoint()
# =====================================================================


class TestMemoryBankUndoToCheckpoint:
    """MemoryBank.undo_to_checkpoint() removes records added after checkpoint."""

    def test_removes_records_after_checkpoint(self):
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0.0")
        bank.add({"name": "sugar"}, source_node_id="0.0.0.0")
        bank.add({"name": "butter"}, source_node_id="0.0.0.0.0")

        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 2
        assert bank.count() == 1
        assert bank.all()[0].data["name"] == "flour"

    def test_preserves_records_at_checkpoint(self):
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0.0")
        bank.add({"name": "sugar"}, source_node_id="0.0.0")

        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 0
        assert bank.count() == 2

    def test_preserves_ancestor_records(self):
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0")
        bank.add({"name": "sugar"}, source_node_id="0.0.0")
        bank.add({"name": "butter"}, source_node_id="0.0.0.0")

        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 1  # only butter removed
        assert bank.count() == 2
        names = {r.data["name"] for r in bank.all()}
        assert names == {"flour", "sugar"}

    def test_cross_branch_removes_other_branch(self):
        """Records on a different branch are removed."""
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0")  # common ancestor
        bank.add({"name": "sugar"}, source_node_id="0.0.0")  # branch A
        bank.add({"name": "butter"}, source_node_id="0.0.1")  # branch B

        # Checkpoint is on branch A — branch B records should be removed
        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 1
        names = {r.data["name"] for r in bank.all()}
        assert names == {"flour", "sugar"}

    def test_skips_records_without_source_node_id(self):
        """Records with no source_node_id are never removed."""
        bank = _make_bank()
        bank.add({"name": "flour"})  # no source_node_id
        bank.add({"name": "sugar"}, source_node_id="0.0.0.0")

        removed = bank.undo_to_checkpoint("0.0")
        assert removed == 1  # only sugar removed (not ancestor of 0.0)
        assert bank.count() == 1
        assert bank.all()[0].data["name"] == "flour"

    def test_empty_bank(self):
        bank = _make_bank()
        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 0

    def test_no_records_after_checkpoint(self):
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0")

        removed = bank.undo_to_checkpoint("0.0.0")
        assert removed == 0
        assert bank.count() == 1

    def test_does_not_revert_modifications(self):
        """v1 append-only: modified records are NOT reverted."""
        bank = _make_bank()
        rid = bank.add({"name": "flour"}, source_node_id="0.0")

        # Modify the record at a later node
        bank.update(rid, {"name": "whole wheat flour"}, modified_in_stage="refine")

        # Undo to checkpoint before modification — record survives with modifications
        removed = bank.undo_to_checkpoint("0.0")
        assert removed == 0
        record = bank.get(rid)
        assert record is not None
        assert record.data["name"] == "whole wheat flour"  # modification persists

    def test_multiple_undos(self):
        """Successive undo_to_checkpoint calls work correctly."""
        bank = _make_bank()
        bank.add({"name": "flour"}, source_node_id="0.0")
        bank.add({"name": "sugar"}, source_node_id="0.0.0")
        bank.add({"name": "butter"}, source_node_id="0.0.0.0")

        # First undo: remove butter
        bank.undo_to_checkpoint("0.0.0")
        assert bank.count() == 2

        # Second undo: remove sugar
        bank.undo_to_checkpoint("0.0")
        assert bank.count() == 1
        assert bank.all()[0].data["name"] == "flour"


# =====================================================================
# AsyncMemoryBank.add() with source_node_id
# =====================================================================


class TestAsyncMemoryBankAddSourceNodeId:
    """AsyncMemoryBank.add() stores source_node_id."""

    @pytest.mark.asyncio
    async def test_add_stores_source_node_id(self):
        bank = await _make_async_bank()
        rid = await bank.add(
            {"name": "flour"}, source_stage="collect", source_node_id="0.0.0"
        )
        record = await bank.get(rid)
        assert record is not None
        assert record.source_node_id == "0.0.0"


# =====================================================================
# AsyncMemoryBank.undo_to_checkpoint()
# =====================================================================


class TestAsyncMemoryBankUndoToCheckpoint:
    """AsyncMemoryBank.undo_to_checkpoint() mirrors sync behavior."""

    @pytest.mark.asyncio
    async def test_removes_records_after_checkpoint(self):
        bank = await _make_async_bank()
        await bank.add({"name": "flour"}, source_node_id="0.0.0")
        await bank.add({"name": "sugar"}, source_node_id="0.0.0.0")
        await bank.add({"name": "butter"}, source_node_id="0.0.0.0.0")

        removed = await bank.undo_to_checkpoint("0.0.0")
        assert removed == 2
        assert await bank.count() == 1

    @pytest.mark.asyncio
    async def test_preserves_records_at_checkpoint(self):
        bank = await _make_async_bank()
        await bank.add({"name": "flour"}, source_node_id="0.0.0")

        removed = await bank.undo_to_checkpoint("0.0.0")
        assert removed == 0
        assert await bank.count() == 1

    @pytest.mark.asyncio
    async def test_cross_branch(self):
        bank = await _make_async_bank()
        await bank.add({"name": "flour"}, source_node_id="0.0")
        await bank.add({"name": "sugar"}, source_node_id="0.0.0")
        await bank.add({"name": "butter"}, source_node_id="0.0.1")

        removed = await bank.undo_to_checkpoint("0.0.0")
        assert removed == 1
        records = await bank.all()
        names = {r.data["name"] for r in records}
        assert names == {"flour", "sugar"}

    @pytest.mark.asyncio
    async def test_does_not_revert_modifications(self):
        """v1 append-only: modified records are NOT reverted."""
        bank = await _make_async_bank()
        rid = await bank.add({"name": "flour"}, source_node_id="0.0")
        await bank.update(rid, {"name": "whole wheat flour"})

        removed = await bank.undo_to_checkpoint("0.0")
        assert removed == 0
        record = await bank.get(rid)
        assert record is not None
        assert record.data["name"] == "whole wheat flour"
