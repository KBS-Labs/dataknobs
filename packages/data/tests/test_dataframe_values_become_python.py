# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A record made from a DataFrame holds Python values.

The frame's own were passed on: numpy scalars and pandas timestamps, which a
JSON column refuses, ``NaN`` for a missing value, and an int beside a float
column as a float.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dataknobs_data import FieldType, Query, database_factory
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.pandas import BatchOperations, ConversionOptions, DataFrameConverter
from dataknobs_data.pandas.metadata import MetadataStrategy
from dataknobs_data.pandas.type_mapper import TypeMapper
from dataknobs_data.records import Record

BIG = 2**53 + 1
WHEN = datetime(2024, 1, 2, 3, 4, 5)


CELLS: list[tuple[str, pd.Series, Any, FieldType]] = [
    ("int", pd.Series([1]), 1, FieldType.INTEGER),
    ("int8", pd.Series([1], dtype="int8"), 1, FieldType.INTEGER),
    ("nullable-int", pd.Series([BIG], dtype="Int64"), BIG, FieldType.INTEGER),
    ("float", pd.Series([1.5]), 1.5, FieldType.FLOAT),
    ("float32", pd.Series([1.5], dtype="float32"), 1.5, FieldType.FLOAT),
    ("bool", pd.Series([True]), True, FieldType.BOOLEAN),
    ("nullable-bool", pd.Series([True], dtype="boolean"), True, FieldType.BOOLEAN),
    ("numpy-object", pd.Series([np.int64(3)], dtype=object), 3, FieldType.INTEGER),
    ("string", pd.Series(["t"], dtype="string"), "t", FieldType.STRING),
    ("json-text", pd.Series(["[1]"]), "[1]", FieldType.STRING),
    ("category", pd.Series(["lo"], dtype="category"), "lo", FieldType.STRING),
    ("timestamp", pd.Series([pd.Timestamp(WHEN)]), WHEN, FieldType.DATETIME),
    (
        "aware-timestamp",
        pd.Series([pd.Timestamp(WHEN, tz="UTC")]),
        WHEN.replace(tzinfo=UTC),
        FieldType.DATETIME,
    ),
    ("list", pd.Series([[1, 2]]), [1, 2], FieldType.JSON),
    ("list-of-none", pd.Series([[None]]), [None], FieldType.JSON),
    ("empty-list", pd.Series([[]]), [], FieldType.JSON),
    ("mapping", pd.Series([{"k": 1}]), {"k": 1}, FieldType.JSON),
]


@pytest.mark.parametrize(
    ("column", "value", "field_type"), [c[1:] for c in CELLS], ids=[c[0] for c in CELLS]
)
def test_a_cell_becomes_a_python_value(
    column: pd.Series, value: Any, field_type: FieldType
) -> None:
    (record,) = DataFrameConverter().dataframe_to_records(pd.DataFrame({"c": column}))
    field = record.fields["c"]
    assert field.value == value
    assert type(field.value) is type(value)
    assert field.type is field_type


@pytest.mark.parametrize(
    "missing",
    [
        pd.Series([np.nan]),
        pd.Series([pd.NaT]),
        pd.Series([pd.NA], dtype="Int64"),
        pd.Series([pd.NA], dtype="string"),
        pd.Series([None], dtype=object),
    ],
    ids=["nan", "nat", "na-int", "na-string", "none"],
)
def test_a_missing_cell_becomes_none(missing: pd.Series) -> None:
    (record,) = DataFrameConverter().dataframe_to_records(pd.DataFrame({"c": missing}))
    assert record.fields["c"].value is None


def test_a_column_keeps_its_type_beside_another() -> None:
    frame = pd.DataFrame({"i": [BIG], "f": [0.5], "b": [True]})
    (record,) = DataFrameConverter().dataframe_to_records(frame)
    assert {n: (f.value, type(f.value)) for n, f in record.fields.items()} == {
        "i": (BIG, int),
        "f": (0.5, float),
        "b": (True, bool),
    }


def test_an_int_beside_only_floats_stays_an_int() -> None:
    """With no other kind of column, pandas makes the row one of floats."""
    (record,) = DataFrameConverter().dataframe_to_records(pd.DataFrame({"i": [BIG], "f": [0.5]}))
    value = record.get_value("i")
    assert value == BIG
    assert type(value) is int


def test_metadata_cells_become_python_values() -> None:
    frame = pd.DataFrame({"a": [1], "_meta_n": [2], "m": [3]})
    options = ConversionOptions(metadata_strategy=MetadataStrategy.COLUMNS, metadata_columns=["m"])
    (record,) = DataFrameConverter().dataframe_to_records(frame, options)
    assert {k: type(v) for k, v in record.metadata.items()} == {"n": int, "m": int}


def test_a_list_of_one_missing_value_is_no_missing_value() -> None:
    mapper = TypeMapper()
    assert mapper.convert_value_from_pandas([None], FieldType.JSON) == [None]
    assert mapper.infer_field_type_from_value([None]) is FieldType.JSON
    assert mapper.convert_value([None, 1], "json") == [None, 1]


def test_an_id_column_names_its_records_beside_a_float_column() -> None:
    db = SyncMemoryDatabase()
    db.upsert("1", Record({"x": 0.0}))
    stats = BatchOperations(db).update_from_dataframe(
        pd.DataFrame({"id": [1], "x": [1.5]}), id_column="id"
    )
    assert stats["updated"] == 1
    assert db.read("1").get_value("x") == 1.5


@pytest.mark.parametrize("backend", ["memory", "sqlite", "duckdb", "file"])
def test_numbers_and_missing_values_are_stored(backend: str, tmp_path: Path) -> None:
    frame = pd.DataFrame({"i": [1, 2], "f": [0.5, np.nan]})
    config = {} if backend == "memory" else {"path": str(tmp_path / f"s.{backend}.json")}
    db = database_factory.create(backend=backend, **config)
    db.connect()
    try:
        stats = BatchOperations(db).bulk_insert_dataframe(frame)
        assert stats["inserted"] == 2
        rows = {r.get_value("i"): r for r in db.search(Query())}
        assert rows[1].get_value("f") == 0.5
        assert rows[2].get_value("f") is None
    finally:
        db.close()


def test_nanoseconds_are_dropped_with_a_warning() -> None:
    frame = pd.DataFrame({"t": [pd.Timestamp("2024-01-01 00:00:00.000001234")]})
    with pytest.warns(UserWarning, match="nanoseconds"):
        (record,) = DataFrameConverter().dataframe_to_records(frame)
    assert record.fields["t"].value == datetime(2024, 1, 1, 0, 0, 0, 1)
