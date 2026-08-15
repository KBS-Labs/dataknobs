"""Tests for dataknobs_xization.lexicon.

Currently scoped to ``MultiAuthorityData.get_unique_vals_df``. The rest of the
module is uncovered; extending this file is welcome.
"""

import pandas as pd
import pytest

import dataknobs_xization.authorities as dk_auth
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


# --- peeking at built sub-authority data -----------------------------------


class _StubAuthority(dk_auth.AuthorityData):
    """Minimal real AuthorityData, so the container is exercised with the type it holds."""

    def __init__(self, name):
        super().__init__(pd.DataFrame({"value": [1]}), name)


class _CountingMultiAuthority(dk_lex.MultiAuthorityData):
    """MultiAuthorityData with the one abstract method filled in, counting builds."""

    def __init__(self, df, name):
        super().__init__(df, name)
        self.builds = []

    def build_authority_data(self, name):
        self.builds.append(name)
        return _StubAuthority(name)


def _multi():
    return _CountingMultiAuthority(pd.DataFrame({"a": [1, 2]}), "top")


def test_peek_returns_none_before_the_sub_authority_is_built():
    """Bug: this accessor was a @property with a required parameter.

    A property getter is invoked by attribute access with no arguments, so
    ``authority_data`` raised TypeError on every access and the "retrieve
    without building" capability did not exist at all -- while its docstring
    said it returned None when absent and its annotation said it could not.
    """
    multi = _multi()

    assert multi.peek_authority_data("missing") is None
    assert multi.builds == [], "peeking must not build"


def test_peek_returns_the_built_object_without_rebuilding():
    multi = _multi()
    built = multi.get_authority_data("colour")

    assert multi.peek_authority_data("colour") is built
    assert multi.builds == ["colour"], "peeking after a build must not build again"


# --- masking against a sub-authority that has not been built ---------------


def test_lookup_subauth_values_is_none_when_the_sub_authority_is_unbuilt():
    """``lookup_subauth_values`` peeks rather than builds, so None is its normal answer.

    Its annotation said ``-> pd.DataFrame`` while its body initialized the
    result to None and returned it untouched whenever the peek missed --
    which, because the peek deliberately does not build, is the default state
    rather than an edge case.
    """
    multi = _multi()

    assert multi.lookup_subauth_values("a", 1, is_id=True) is None
    assert multi.builds == [], "looking up sub-values must not build"


def test_auth_values_mask_is_all_false_when_the_sub_authority_is_unbuilt():
    """Bug: this raised TypeError on the default path.

    ``auth_values_mask`` subscripted ``lookup_subauth_values``'s result
    without checking it, so an unbuilt sub-authority produced
    ``TypeError: 'NoneType' object is not subscriptable``. No sub-authority
    values means no record carries one, which is an all-False mask -- and
    ``auth_records_mask`` conjoins these, where all-False correctly excludes
    every record rather than exploding.
    """
    multi = _multi()

    mask = multi.auth_values_mask("a", 1)

    assert not mask.any(), "no sub-authority values means no record can match"
    assert mask.index.equals(multi.df.index), "the mask must align with the authority rows"


def test_auth_records_mask_conjoins_an_unbuilt_field_without_raising():
    """The consumer-visible half: the only caller of auth_values_mask."""
    multi = _multi()

    mask = multi.auth_records_mask({"a": 1})

    assert not mask.any()
