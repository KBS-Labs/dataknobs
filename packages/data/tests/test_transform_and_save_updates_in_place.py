# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``transform_and_save`` writes the cells a transformer changed, in place.

It saved by index name, which the frame had only for some ids, and inserted a
copy of each record otherwise. Where it did update, it wrote the whole frame:
``NaN`` into fields a record never had, and ints as floats.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dataknobs_data import Filter, Operator, Query, Record, database_factory
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.pandas import BatchConfig, BatchOperations

UUID = "0b4f5a0e-3c1c-4c55-9b0a-5c6f1a2b3c4d"
META = {"owner": "o"}


class Store:
    """One database of either flavour, read and seeded alike."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.ops = BatchOperations(db)

    def _do(self, result: Any) -> Any:
        return asyncio.run(result) if asyncio.iscoroutine(result) else result

    def seed(self, rows: dict[str, dict[str, Any]]) -> None:
        for record_id, data in rows.items():
            self._do(self.db.upsert(record_id, Record(data, metadata=dict(META))))

    def contents(self) -> dict[str, dict[str, Any]]:
        records = self._do(self.db.search(Query()))
        assert all(r.metadata["owner"] == "o" for r in records), "metadata was lost"
        return {
            r.storage_id: {name: field.value for name, field in r.fields.items()} for r in records
        }


@pytest.fixture(params=["memory", "async-memory", "sqlite", "duckdb", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        yield Store(SyncMemoryDatabase())
        return
    if request.param == "async-memory":
        yield Store(AsyncMemoryDatabase())
        return
    suffix = "json" if request.param == "file" else "db"
    db = database_factory.create(backend=request.param, path=str(tmp_path / f"s.{suffix}"))
    db.connect()
    yield Store(db)
    db.close()


@pytest.fixture
def memory() -> Store:
    return Store(SyncMemoryDatabase())


def set_a(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(a=5)


IDENTITIES: dict[str, dict[str, dict[str, Any]]] = {
    "generated": {UUID: {"a": 1, "b": 2}},
    "named": {"r1": {"a": 1, "b": 2}},
    "both": {"r1": {"a": 1}, UUID: {"a": 2}},
    "an-id-field": {"r1": {"a": 1, "id": "x"}},
    "a-record_id-field": {"r1": {"a": 1, "record_id": "x"}},
    "no-fields": {"r1": {}},
}


@pytest.mark.parametrize("name", IDENTITIES)
def test_each_record_is_updated_whatever_its_id(store: Store, name: str) -> None:
    rows = IDENTITIES[name]
    store.seed(rows)

    stats = store.ops.transform_and_save(Query(), set_a)

    assert store.contents() == {k: {**v, "a": 5} for k, v in rows.items()}
    assert stats == {
        "total_rows": len(rows),
        "updated": len(rows),
        "unchanged": 0,
        "not_found": 0,
        "failed": 0,
        "errors": [],
    }


def test_generated_ids_are_updated_by_create(memory: Store) -> None:
    record_id = memory.db.create(Record({"a": 1}, metadata=dict(META)))
    memory.ops.transform_and_save(Query(), set_a)
    assert memory.contents() == {record_id: {"a": 5}}


def test_only_changed_cells_are_written(store: Store) -> None:
    rows = {"r1": {"a": 1, "n": None, "s": "t"}, "r2": {"a": 2, "k": 7, "n": 3}}
    store.seed(rows)

    store.ops.transform_and_save(Query(), set_a)

    after = store.contents()
    assert after == {k: {**v, "a": 5} for k, v in rows.items()}
    assert type(after["r2"]["k"]) is int
    assert type(after["r2"]["n"]) is int


def test_a_record_that_does_not_change_is_not_written(memory: Store) -> None:
    written: list[str] = []

    class Witness(SyncMemoryDatabase):
        def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
            written.extend(record_id for record_id, _ in updates)
            return super().update_batch(updates)

    store = Store(Witness())
    store.seed({"r1": {"a": 5}, "r2": {"a": 1}})

    stats = store.ops.transform_and_save(Query(), set_a)

    assert written == ["r2"]
    assert (stats["total_rows"], stats["updated"], stats["unchanged"]) == (2, 1, 1)


def test_a_transformer_may_change_its_frame_in_place(store: Store) -> None:
    store.seed({"r1": {"a": 1, "l": [1], "d": {"x": [1]}}})

    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df["a"] = 5
        df["l"].iloc[0].append(2)
        df["d"].iloc[0]["x"].append(2)
        return df

    store.ops.transform_and_save(Query(), mutate)
    assert store.contents() == {"r1": {"a": 5, "l": [1, 2], "d": {"x": [1, 2]}}}


def test_a_cell_made_missing_is_written_as_none(store: Store) -> None:
    store.seed({"r1": {"a": 1}, "r2": {"a": 2}})
    store.ops.transform_and_save(Query(), lambda df: df.assign(a=df["a"].where(df["a"] > 1)))
    assert store.contents() == {"r1": {"a": None}, "r2": {"a": 2}}


@pytest.mark.parametrize(
    ("before", "after"), [(1, True), (True, 1), (1, "1"), ([1], [1, 2]), (None, 0), (1, 1.5)]
)
def test_a_change_of_value_is_seen(memory: Store, before: Any, after: Any) -> None:
    memory.seed({"r1": {"a": before}})
    memory.ops.transform_and_save(
        Query(), lambda df: df.assign(a=pd.Series([after], index=df.index, dtype=object))
    )
    value = memory.contents()["r1"]["a"]
    assert value == after
    assert type(value) is type(after)


@pytest.mark.parametrize("value", [[None], [float("nan")], [], {}, "", [[1], {"k": None}]])
def test_an_unchanged_value_is_no_change(memory: Store, value: Any) -> None:
    memory.seed({"r1": {"a": value}})
    stats = memory.ops.transform_and_save(Query(), lambda df: df)
    assert stats["unchanged"] == 1


def _object_array(*items: Any) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        array[i] = item
    return array


# Values that do not answer ``==`` with one bool, or are not equal to themselves.
UNCOMPARABLE: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    "NA in a list": (lambda: [pd.NA, 1], lambda: [1, 1]),
    "NA put in a list": (lambda: [2, 1], lambda: [pd.NA, 1]),
    "NA in a dict": (lambda: {"k": pd.NA}, lambda: {"k": 1}),
    "NaT in a list": (lambda: [pd.NaT], lambda: [pd.Timestamp(0)]),
    "array": (lambda: np.array([1, 2]), lambda: np.array([1, 3])),
    "array of NaN": (lambda: np.array([1.0, np.nan]), lambda: np.array([np.nan, 1.0])),
    "array of NaT": (
        lambda: np.array(["NaT"], dtype="datetime64[ns]"),
        lambda: np.array(["2024-01-01"], dtype="datetime64[ns]"),
    ),
    "array of str": (lambda: np.array(["a"]), lambda: np.array(["b"])),
    "array of another dtype": (
        lambda: np.array([1.0], dtype=np.float32),
        lambda: np.array([1.0], dtype=np.float64),
    ),
    "array of another shape": (lambda: np.array([1, 1]), lambda: np.array([[1, 1]])),
    "empty array of another shape": (
        lambda: np.empty((0, 2), dtype=object),
        lambda: np.empty((0, 3), dtype=object),
    ),
    "array of arrays": (
        lambda: _object_array(np.array([1, 2]), np.array([1])),
        lambda: _object_array(np.array([1, 2]), np.array([2])),
    ),
    "array of NA": (lambda: _object_array(pd.NA), lambda: _object_array(1)),
    "Series": (lambda: pd.Series([1, np.nan]), lambda: pd.Series([1, 2.0])),
    "Series made a list": (lambda: pd.Series([1, 2]), lambda: [1, 2]),
    "number made a Series": (lambda: 1, lambda: pd.Series([1, 1])),
    "Series of another index": (lambda: pd.Series([1]), lambda: pd.Series([1], index=["a"])),
    "frame": (lambda: pd.DataFrame({"a": [1]}), lambda: pd.DataFrame({"a": [2]})),
    "Index": (lambda: pd.Index([1, 2]), lambda: pd.Index([1, 3])),
}


def _put(value: Callable[[], Any]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df.assign(v=pd.Series(_object_array(value()), index=df.index))


@pytest.mark.parametrize("name", UNCOMPARABLE)
def test_a_value_without_plain_equality_is_compared(memory: Store, name: str) -> None:
    before, after = UNCOMPARABLE[name]
    memory.seed({"a": {"v": before()}})

    assert memory.ops.transform_and_save(Query(), _put(before))["unchanged"] == 1
    assert memory.ops.transform_and_save(Query(), _put(after))["updated"] == 1
    # What was written is what the next call finds unchanged.
    assert memory.ops.transform_and_save(Query(), _put(after))["unchanged"] == 1


def test_rows_may_be_filtered_and_reordered(store: Store) -> None:
    store.seed({"r1": {"a": 1}, "r2": {"a": 2}, "r3": {"a": 3}})

    def top(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["a"] > 1].sort_values("a", ascending=False).assign(rank=[1, 2])

    stats = store.ops.transform_and_save(Query(), top)
    assert store.contents() == {
        "r1": {"a": 1},
        "r2": {"a": 2, "rank": 2},
        "r3": {"a": 3, "rank": 1},
    }
    assert stats["total_rows"] == 2


def test_pieces_of_the_frame_may_be_joined_again(memory: Store) -> None:
    memory.seed({"r1": {"a": 1}, "r2": {"a": 2}})
    memory.ops.transform_and_save(
        Query(), lambda df: pd.concat([df.iloc[1:], df.iloc[:1]]).assign(a=5)
    )
    assert memory.contents() == {"r1": {"a": 5}, "r2": {"a": 5}}


@pytest.mark.parametrize("name", [None, "record_id"])
def test_the_index_is_named_unless_a_field_is(memory: Store, name: str | None) -> None:
    memory.seed({"r1": {"a": 1, "z": 2} if name else {"a": 1, "record_id": 2}})
    names = []

    def sort(df: pd.DataFrame) -> pd.DataFrame:
        names.append(df.index.name)
        return df.sort_values("record_id").assign(a=5)

    memory.ops.transform_and_save(Query(), sort)
    assert names == [name]
    assert memory.contents()["r1"]["a"] == 5


def test_a_dropped_column_is_left_as_stored(memory: Store) -> None:
    memory.seed({"r1": {"a": 1, "b": 2}})
    stats = memory.ops.transform_and_save(Query(), lambda df: df.drop(columns="b"))
    assert memory.contents() == {"r1": {"a": 1, "b": 2}}
    assert stats["unchanged"] == 1


REFUSED: dict[str, Callable[[pd.DataFrame], Any]] = {
    "reset_index": lambda df: df.reset_index(drop=True),
    "a-new-row": lambda df: pd.concat([df, pd.DataFrame({"a": [9]}, index=["new"])]),
    "aggregate": lambda df: df.groupby("g").sum(),
    "aggregate-without-index": lambda df: df.groupby("g", as_index=False).sum(),
    "merge": lambda df: df.merge(pd.DataFrame({"g": ["x"], "z": [1]}), on="g"),
    "a-repeated-row": lambda df: pd.concat([df, df]),
    "a-repeated-column": lambda df: pd.concat([df, df], axis=1),
}


@pytest.mark.parametrize("name", REFUSED)
def test_a_row_that_names_no_record_is_refused_before_any_write(store: Store, name: str) -> None:
    rows = {"0": {"g": "x", "a": 1}, "1": {"g": "x", "a": 2}}
    store.seed(rows)

    with pytest.raises(ValueError, match="transformer"):
        store.ops.transform_and_save(Query(), lambda df: REFUSED[name](set_a(df)))

    assert store.contents() == rows


def test_a_transformer_that_returns_no_frame_is_refused(memory: Store) -> None:
    memory.seed({"r1": {"a": 1}})
    with pytest.raises(TypeError, match="transformer must return a DataFrame"):
        memory.ops.transform_and_save(Query(), lambda df: None)  # type: ignore[arg-type, return-value]


def test_nothing_found_answers_the_same_statistics(store: Store) -> None:
    stats = store.ops.transform_and_save(Query(), set_a)
    assert stats == {
        "total_rows": 0,
        "updated": 0,
        "unchanged": 0,
        "not_found": 0,
        "failed": 0,
        "errors": [],
    }


def test_a_record_deleted_meanwhile_is_not_found(memory: Store) -> None:
    memory.seed({"r1": {"a": 1}, "r2": {"a": 2}})

    def delete_one(df: pd.DataFrame) -> pd.DataFrame:
        memory.db.delete("r1")
        return set_a(df)

    stats = memory.ops.transform_and_save(Query(), delete_one)
    assert memory.contents() == {"r2": {"a": 5}}
    assert (stats["updated"], stats["not_found"]) == (1, 1)


def test_only_the_queried_records_are_written(store: Store) -> None:
    store.seed({"r1": {"a": 1}, "r2": {"a": 2}})
    store.ops.transform_and_save(Query(filters=[Filter("a", Operator.GT, 1)]), set_a)
    assert store.contents() == {"r1": {"a": 1}, "r2": {"a": 5}}


def test_chunks_and_progress(memory: Store) -> None:
    memory.seed({f"r{i}": {"a": i + 10} for i in range(5)})
    seen: list[tuple[int, int]] = []
    config = BatchConfig(
        chunk_size=2, progress_callback=lambda done, total: seen.append((done, total))
    )
    memory.ops.transform_and_save(Query(), set_a, config)
    assert seen == [(2, 5), (4, 5), (5, 5)]
