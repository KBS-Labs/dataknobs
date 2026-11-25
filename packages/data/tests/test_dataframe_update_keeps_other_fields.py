# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A DataFrame update writes the frame's columns and keeps the rest of the record.

``update`` replaces a record, and the frame's columns were the whole of the
record sent, so every other field and all metadata was erased.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.pandas import BatchConfig, BatchOperations
from dataknobs_data.pandas.converter import ConversionOptions

STORED = {"a": 1, "b": {"x": [1]}}


class RefusingBatch(SyncMemoryDatabase):
    """Refuses a batch, so every row takes the per-record path."""

    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        raise ValueError("batch refused")


class RefusingOneRecord(RefusingBatch):
    """Refuses one record's write as well."""

    def update(self, id: str, record: Record, *, expected_version: str | None = None) -> bool:
        if id == "bad":
            raise ValueError("record refused")
        return super().update(id, record, expected_version=expected_version)


class AsyncAdapter:
    """Reads and writes an async database the way a test of the sync one does."""

    def __init__(self) -> None:
        self.db = AsyncMemoryDatabase()

    def upsert(self, id: str, record: Record) -> Any:
        return asyncio.run(self.db.upsert(id, record))

    def read(self, id: str) -> Any:
        return asyncio.run(self.db.read(id))


def seeded(kind: Any) -> tuple[Any, BatchOperations]:
    store = kind()
    for record_id in ("r1", "r2", "bad"):
        store.upsert(record_id, Record(STORED, metadata={"owner": "o", "n": 1}))
    return store, BatchOperations(getattr(store, "db", store))


def contents(store: Any, record_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = store.read(record_id)
    return {name: field.value for name, field in record.fields.items()}, record.metadata


KINDS = [SyncMemoryDatabase, AsyncAdapter, RefusingBatch]


@pytest.mark.parametrize("kind", KINDS)
def test_fields_and_metadata_the_frame_does_not_name_are_kept(kind: Any) -> None:
    store, ops = seeded(kind)

    stats = ops.update_from_dataframe(
        pd.DataFrame({"a": [9, 8]}, index=["r1", "r2"]), None, BatchConfig(error_handling="log")
    )

    assert (stats["updated"], stats["not_found"], stats["failed"]) == (2, 0, 0)
    assert contents(store, "r1") == ({**STORED, "a": 9}, {"owner": "o", "n": 1})
    assert contents(store, "r2") == ({**STORED, "a": 8}, {"owner": "o", "n": 1})
    assert contents(store, "bad") == (STORED, {"owner": "o", "n": 1})


def test_a_column_replaces_the_whole_field() -> None:
    store, ops = seeded(SyncMemoryDatabase)

    ops.update_from_dataframe(pd.DataFrame({"b": [{"y": 2}], "c": ["new"]}, index=["r1"]), None)

    assert contents(store, "r1")[0] == {"a": 1, "b": {"y": 2}, "c": "new"}


def test_a_metadata_column_changes_its_key_and_keeps_the_others() -> None:
    store, ops = seeded(SyncMemoryDatabase)

    ops.update_from_dataframe(
        pd.DataFrame({"n": [5]}, index=["r1"]),
        None,
        conversion_options=ConversionOptions(metadata_columns=["n"]),
    )

    assert contents(store, "r1") == (STORED, {"owner": "o", "n": 5})


@pytest.mark.parametrize("kind", KINDS)
def test_an_id_not_stored_is_counted_and_not_created(kind: Any) -> None:
    store, ops = seeded(kind)

    stats = ops.update_from_dataframe(
        pd.DataFrame({"a": [9, 8]}, index=["gone", "r2"]), None, BatchConfig(error_handling="log")
    )

    assert (stats["updated"], stats["not_found"], stats["failed"]) == (1, 1, 0)
    assert store.read("gone") is None
    assert contents(store, "r2")[0] == {**STORED, "a": 8}


def test_the_last_row_for_a_repeated_id_wins() -> None:
    store, ops = seeded(SyncMemoryDatabase)

    stats = ops.update_from_dataframe(pd.DataFrame({"a": [9, 8]}, index=["r1", "r1"]), None)

    assert stats["updated"] == 2
    assert contents(store, "r1")[0] == {**STORED, "a": 8}


def test_a_refused_row_leaves_the_others_updated() -> None:
    store, ops = seeded(RefusingOneRecord)

    stats = ops.update_from_dataframe(
        pd.DataFrame({"a": [9, 8]}, index=["bad", "r2"]), None, BatchConfig(error_handling="log")
    )

    assert (stats["updated"], stats["failed"]) == (1, 1)
    assert stats["errors"] == ["record refused"]
    assert contents(store, "bad")[0] == STORED
    assert contents(store, "r2")[0] == {**STORED, "a": 8}


def test_a_refused_batch_raises_when_asked_to() -> None:
    _, ops = seeded(RefusingBatch)

    with pytest.raises(ValueError, match="batch refused"):
        ops.update_from_dataframe(
            pd.DataFrame({"a": [9]}, index=["r1"]), None, BatchConfig(error_handling="raise")
        )
