"""Tests for dataknobs_xization.lexicon.

Currently scoped to ``MultiAuthorityData.get_unique_vals_df``. The rest of the
module is uncovered; extending this file is welcome.
"""

import pandas as pd
import pytest

import dataknobs_xization.lexicon as dk_lex


def _unique_vals(values, dtype):
    return dk_lex.MultiAuthorityData.get_unique_vals_df(pd.Series(values, dtype=dtype), "col")


# --- integer columns: IDs are the integers themselves ----------------------


@pytest.mark.parametrize("dtype", ["int64", "int32", "uint8", "uint64", "Int64"])
def test_integer_column_indexes_by_value(dtype):
    """Integer columns — including unsigned and the nullable extension dtype — index by value."""
    df = _unique_vals([3, 1, 2, 1], dtype)

    assert df["col"].tolist() == [1, 2, 3]
    assert df.index.tolist() == [1, 2, 3]


def test_integer_column_drops_missing_values():
    df = _unique_vals([3, None, 1, None], "Int64")

    assert df["col"].tolist() == [1, 3]
    assert df.index.tolist() == [1, 3]


# --- non-integer columns: IDs are auto-generated 0..n-1 --------------------


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([2.5, 1.5, 2.5], "float64", [1.5, 2.5]),
        ([2.5, 1.5, 2.5], "Float64", [1.5, 2.5]),
        ([True, False, True], "bool", [False, True]),
        ([True, False, True], "boolean", [False, True]),
        (["b", "a", "b"], "category", ["a", "b"]),
        (["b", "a", "b"], "string", ["a", "b"]),
        (["b", "a", "b"], "object", ["a", "b"]),
    ],
)
def test_non_integer_column_uses_positional_ids(values, dtype, expected):
    """Non-integer columns — including extension dtypes — get a 0..n-1 index."""
    df = _unique_vals(values, dtype)

    assert df["col"].tolist() == expected
    assert df.index.tolist() == list(range(len(expected)))


# --- temporal columns ------------------------------------------------------


def test_datetime_column_uses_positional_ids():
    """``datetime64`` is not an integer dtype under either predicate."""
    df = _unique_vals(pd.to_datetime(["2020-01-02", "2020-01-01"]), "datetime64[ns]")

    assert df["col"].tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert df.index.tolist() == [0, 1]


def test_timedelta_column_uses_positional_ids():
    """``timedelta64`` takes the positional branch — the one silent branch change.

    ``np.timedelta64`` subclasses ``np.signedinteger``, so the previous
    ``np.issubdtype(col.dtype, np.integer)`` test returned ``True`` for a
    timedelta column and used the raw timedelta values as row IDs.
    ``pd.api.types.is_integer_dtype`` returns ``False``, so such a column now
    gets a 0..n-1 index like every other non-integer column.

    This is the *only* dtype whose branch changed without previously raising —
    every other behavioural difference replaced a ``TypeError`` with a result.
    It is pinned here so the switch stays deliberate.
    """
    df = _unique_vals(pd.to_timedelta([2, 1], unit="D"), "timedelta64[ns]")

    assert df["col"].tolist() == pd.to_timedelta([1, 2], unit="D").tolist()
    assert df.index.tolist() == [0, 1]


def test_non_integer_column_drops_missing_values():
    df = _unique_vals(["b", None, "a"], "string")

    assert df["col"].tolist() == ["a", "b"]
    assert df.index.tolist() == [0, 1]


def test_column_name_is_applied():
    df = dk_lex.MultiAuthorityData.get_unique_vals_df(
        pd.Series([1, 2], dtype="int64"), "authority_id"
    )

    assert list(df.columns) == ["authority_id"]
