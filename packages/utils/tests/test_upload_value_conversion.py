"""``upload``'s value conversion, one case per branch — without a server.

The round-trip half of this behaviour lives in
``tests/integration/test_upload_value_rendering.py`` and needs PostgreSQL. The
conversion itself does not: :meth:`PostgresDB._column_values_for_insert` takes a
Series and returns the list handed to psycopg2, which is checkable in-process.

Keeping it here rather than beside the round-trip tests is deliberate. Under a
module-level ``requires_postgres`` these cases skip wherever no server is
running, which is every CI job — so a branch they cover can break with nothing
reporting it. The conversion is also where the subtle failures are: what reaches
psycopg2 as ``None``, what stays typed, and what must not be touched at all.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from dataknobs_utils.sql_utils import PostgresDB


class TestScalarConversion:
    """One case per branch of the conversion ladder."""

    def test_text_column_is_rendered_with_str(self) -> None:
        got = PostgresDB._column_values_for_insert(pd.Series([1, "a"], dtype=object))

        assert got == ["1", "a"]

    def test_typed_column_is_not_rendered(self) -> None:
        got = PostgresDB._column_values_for_insert(pd.Series([1, 2], dtype="int64"))

        assert got == [1, 2]
        assert all(type(v) is int for v in got), "numpy scalars reached psycopg2"

    def test_nulls_become_none_in_both_kinds(self) -> None:
        text = PostgresDB._column_values_for_insert(pd.Series(["a", None], dtype=object))
        typed = PostgresDB._column_values_for_insert(pd.array([1, None], dtype="Int64"))

        assert text == ["a", None]
        assert list(typed) == [1, None]

    def test_timedelta_becomes_a_python_timedelta(self) -> None:
        got = PostgresDB._column_values_for_insert(pd.to_timedelta(["1 days"]).to_series())

        assert got == [dt.timedelta(days=1)]
        assert type(got[0]) is dt.timedelta

    def test_timestamp_becomes_a_python_datetime(self) -> None:
        got = PostgresDB._column_values_for_insert(pd.to_datetime(["2024-01-01"]).to_series())

        assert got == [dt.datetime(2024, 1, 1)]

    def test_text_predicate_matches_the_schema_ladder(self) -> None:
        """The two sides have to agree, so the agreement is asserted rather
        than left to two copies of one predicate staying in step.
        """
        cases = {
            "b": pd.Series([True]),
            "i": pd.array([1], dtype="int64"),
            "f": pd.Series([1.0]),
            "ts": pd.to_datetime(["2024-01-01"]),
            "td": pd.to_timedelta(["1 days"]),
            "s": pd.Series(["x"]),
        }
        for name, values in cases.items():
            frame = pd.DataFrame({name: values})
            declared_varchar = "varchar" in PostgresDB._psql_schema_line(frame, name)

            assert PostgresDB._column_is_text(frame[name].dtype) == declared_varchar, name


class TestNonScalarCellsAreRenderedNotInspected:
    """A cell holding a container is text, and the null check must not touch it.

    ``pd.isna`` is elementwise: handed a ``list`` or an ``ndarray`` it returns an
    *array* of answers rather than one, and the surrounding ``or`` then asks that
    array for its truth value. The result is ``ValueError: The truth value of an
    array with more than one element is ambiguous`` — raised while deciding
    whether a value is null, on a value that plainly is not.

    These frames uploaded cleanly before values were sent typed, because ``str``
    never asks a question about its argument. The schema half still handles them
    — ``_psql_varchar_width`` measures with ``Series.dropna()``, which is
    elementwise-safe — so the defect splits the two halves of one feature apart,
    which is the failure this whole area exists to prevent.
    """

    @pytest.mark.parametrize(
        ("label", "cell"),
        [
            ("list", [1, 2]),
            ("empty list", []),
            ("ndarray", np.array([1, 2])),
            ("nested list", [[1], [2]]),
            ("tuple", (1, 2)),
            ("dict", {"a": 1}),
            ("set", {1, 2}),
        ],
    )
    def test_a_container_cell_is_rendered_as_text(self, label: str, cell: object) -> None:
        """Bug: the null check raised ``ValueError`` on ``list``/``ndarray``
        cells. ``tuple``/``dict``/``set`` were unaffected — ``pd.isna`` answers
        scalar-wise for those — which is what made the gap easy to miss.
        """
        got = PostgresDB._column_values_for_insert(pd.Series([cell], dtype=object))

        assert got == [str(cell)], label

    def test_a_container_column_still_declares_a_width_that_fits(self) -> None:
        """The two halves have to agree on the same rendering.

        The declared width comes from ``str``; so must the written value, or a
        frame that passes ``CREATE TABLE`` fails its own ``INSERT``.
        """
        frame = pd.DataFrame({"c": pd.Series([[1, 2], ["a"]], dtype=object)})

        line = PostgresDB._psql_schema_line(frame, "c")
        values = PostgresDB._column_values_for_insert(frame["c"])

        assert "varchar" in line
        declared = int(line.split("varchar(")[1].rstrip(")"))
        assert max(len(v) for v in values) <= declared

    def test_nulls_beside_containers_still_become_none(self) -> None:
        """The scalar guard must not cost the null handling it sits in front of."""
        got = PostgresDB._column_values_for_insert(
            pd.Series([[1, 2], None, np.nan, pd.NA], dtype=object)
        )

        assert got == ["[1, 2]", None, None, None]
