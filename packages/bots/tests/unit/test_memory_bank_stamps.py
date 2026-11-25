"""A bank record's stamps are Unix times, whatever its database stored.

A backend may stamp a record itself: the S3 ones write ISO 8601 text, over the
bank's float for ``updated_at`` and, in records written before they kept a
caller's value, for ``created_at`` too. One bank can hold both forms.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from dataknobs_bots.memory.bank import AsyncMemoryBank, BankRecord, MemoryBank, _BankCore
from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase

TEXT = "2026-01-01T00:00:00+00:00"
EPOCH = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
SCHEMA = {"required": ["n"]}


def _record(record_id: str, created_at: Any, updated_at: Any = TEXT) -> Record:
    return Record(
        {"n": record_id},
        storage_id=record_id,
        metadata={
            "record_id": record_id,
            "source_node_id": "node",
            "created_at": created_at,
            "updated_at": updated_at,
        },
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (EPOCH, EPOCH),
        (int(EPOCH), EPOCH),
        (TEXT, EPOCH),
        ("2026-01-01T00:00:00Z", EPOCH),
        ("2026-01-01T00:00:00", EPOCH),
        ("2026-01-01T02:00:00+02:00", EPOCH),
        (datetime(2026, 1, 1, tzinfo=UTC), EPOCH),
        (None, 0.0),
    ],
)
def test_either_stamp_reads_as_a_unix_time(stored: Any, expected: float) -> None:
    record = _BankCore.to_bank_record(_record("a", stored, stored))
    assert (record.created_at, record.updated_at) == (expected, expected)
    assert isinstance(record.created_at, float)
    assert isinstance(record.updated_at, float)


def test_a_record_with_no_stamps_reads_as_zero() -> None:
    record = _BankCore.to_bank_record(Record({"n": 1}))
    assert (record.created_at, record.updated_at) == (0.0, 0.0)


def test_a_serialized_record_with_text_stamps_reads_as_unix_times() -> None:
    """A bank over S3 serialized them as it read them."""
    raw = {"record_id": "a", "created_at": TEXT, "updated_at": TEXT}
    record = BankRecord.from_dict(raw)
    assert (record.created_at, record.updated_at) == (EPOCH, EPOCH)


def test_a_bank_holding_both_forms_orders_by_time() -> None:
    db = SyncMemoryDatabase()
    for record in (_record("text", TEXT), _record("later", EPOCH + 1), _record("old", EPOCH - 1)):
        db.create(record)
    bank = MemoryBank("b", schema=SCHEMA, db=db)
    assert [r.record_id for r in bank.all()] == ["old", "text", "later"]
    assert [r.record_id for r in bank.find(n="text")] == ["text"]
    assert len(bank.find()) == 3
    assert bank.undo_to_checkpoint(None) == 3


async def test_an_async_bank_holding_both_forms_orders_by_time() -> None:
    db = AsyncMemoryDatabase()
    for record in (_record("text", TEXT), _record("later", EPOCH + 1), _record("old", EPOCH - 1)):
        await db.create(record)
    bank = AsyncMemoryBank("b", schema=SCHEMA, db=db)
    assert [r.record_id for r in await bank.all()] == ["old", "text", "later"]
    assert len(await bank.find()) == 3
    assert await bank.undo_to_checkpoint(None) == 3


def _check_hooked(seen: list[BankRecord]) -> None:
    (record,) = seen
    assert record.created_at == EPOCH
    assert isinstance(record.updated_at, float)
    assert record.source_node_id == "node"


def test_an_update_hook_receives_unix_times_and_the_source_node() -> None:
    db = SyncMemoryDatabase()
    db.create(_record("a", TEXT))
    bank = MemoryBank("b", schema=SCHEMA, db=db)
    seen: list[BankRecord] = []
    bank.on_update(seen.append)
    assert bank.update("a", {"n": "new"})
    _check_hooked(seen)


async def test_an_async_update_hook_receives_unix_times_and_the_source_node() -> None:
    db = AsyncMemoryDatabase()
    await db.create(_record("a", TEXT))
    bank = AsyncMemoryBank("b", schema=SCHEMA, db=db)
    seen: list[BankRecord] = []

    async def hook(record: BankRecord) -> None:
        seen.append(record)

    bank.on_update(hook)
    assert await bank.update("a", {"n": "new"})
    _check_hooked(seen)
