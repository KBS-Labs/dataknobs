"""``upload`` must send values, not their ``str``.

Every cell used to be rendered with ``str(row[col])`` over ``df.to_records()``,
so nothing reached psycopg2 as a typed parameter. That single decision produced
five distinct failures, and the reason they went unnoticed for so long is that
they need a *typed* column to show up — which is the other half of this work.
Each class below is one of them.

The schema ladder guesses a column's type; this is the write side honouring the
guess. The two are the same feature seen from either end, and the previous
arrangement had them disagreeing: the ladder declared ``integer`` and the INSERT
sent ``'1.0'``.

Requires a reachable PostgreSQL instance (``bin/dk up``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_utils.sql_utils import PostgresDB

pytestmark = [requires_postgres, pytest.mark.postgres, pytest.mark.integration]


@pytest.fixture
def render_db(make_postgres_test_db: Any) -> Iterator[dict[str, Any]]:
    yield from make_postgres_test_db("test_upload_render_")


@pytest.fixture
def db(render_db: dict[str, Any]) -> Iterator[PostgresDB]:
    database = PostgresDB(
        host=render_db["host"],
        db=render_db["database"],
        user=render_db["user"],
        pwd=render_db["password"],
        port=render_db["port"],
    )
    yield database
    database.close()


def _round_trip(db: PostgresDB, table: str, df: pd.DataFrame) -> pd.DataFrame:
    db.upload(table, df)
    return db.query(f'SELECT * FROM "{table}"')


class TestNullsBecomeSqlNull:
    """A null rendered as text is not a null, and no typed column takes it."""

    def test_null_in_a_typed_column(self, db: PostgresDB, render_db: dict[str, Any]) -> None:
        """Bug: the INSERT carried the string ``'nan'`` into an integer column."""
        df = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")})

        rows = _round_trip(db, render_db["table"], df).sort_values("n", na_position="last")

        assert rows["n"].iloc[0] == 1
        assert pd.isna(rows["n"].iloc[1]), "the null did not arrive as SQL NULL"

    def test_null_in_a_text_column_does_not_become_the_word_nan(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        """The varchar case, which failed as a *width* error rather than a type
        error: ``_psql_varchar_width`` measures the non-null values, so
        ``['a', None]`` declared ``varchar(1)`` and then sent 3 characters.
        """
        df = pd.DataFrame({"t": ["a", None]})

        rows = _round_trip(db, render_db["table"], df)

        values = list(rows["t"])
        assert "nan" not in values, "a null was written as the text 'nan'"
        assert None in values or any(pd.isna(v) for v in values)

    def test_pandas_na_in_a_text_column(self, db: PostgresDB, render_db: dict[str, Any]) -> None:
        """``pd.NA`` renders as the 4-character ``'<NA>'``, so it overflowed a
        width measured at 1 rather than merely being wrong.
        """
        df = pd.DataFrame({"t": pd.array(["a", pd.NA], dtype="string")})

        rows = _round_trip(db, render_db["table"], df)

        assert "<NA>" not in list(rows["t"])


class TestNullableDtypesAreNotUpcast:
    def test_nullable_int_stays_integral(self, db: PostgresDB, render_db: dict[str, Any]) -> None:
        """Bug: ``df.to_records()`` upcast ``Int64`` to float64, so the INSERT
        sent ``'1.0'`` into the integer column the ladder had just created.
        """
        df = pd.DataFrame({"n": pd.array([1, 2], dtype="Int64")})

        rows = _round_trip(db, render_db["table"], df)

        assert sorted(rows["n"]) == [1, 2]

    def test_mixed_dtypes_in_one_row_do_not_upcast_each_other(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        """The mechanism behind the upcast: a records array has ONE dtype per
        row, so a row mixing an int column with a float one promoted the int.
        Building per column is what avoids it.
        """
        df = pd.DataFrame(
            {
                "i": pd.array([7], dtype="int64"),
                "f": [1.5],
                "s": ["x"],
            }
        )

        rows = _round_trip(db, render_db["table"], df)

        assert rows["i"].iloc[0] == 7
        assert rows["f"].iloc[0] == 1.5
        assert rows["s"].iloc[0] == "x"


class TestTemporalValuesKeepTheirType:
    def test_nanosecond_timedelta_is_accepted(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        """Bug: ``str`` follows the column's resolution, so a ``timedelta64[ns]``
        column rendered ``'86400000000000 nanoseconds'`` — and ``interval`` has
        no unit finer than a microsecond, so PostgreSQL rejected it outright.
        """
        df = pd.DataFrame({"d": pd.to_timedelta(["1 days"]).as_unit("ns")})

        rows = _round_trip(db, render_db["table"], df)

        assert rows["d"].iloc[0] == pd.Timedelta("1 days")

    def test_timestamp_arrives_as_a_timestamp(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-03-01 08:15:00"])})

        rows = _round_trip(db, render_db["table"], df)

        assert rows["ts"].iloc[0] == pd.Timestamp("2024-03-01 08:15:00")

    def test_null_timestamp(self, db: PostgresDB, render_db: dict[str, Any]) -> None:
        """``NaT`` is the temporal null and took the same text path."""
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-03-01", None])})

        rows = _round_trip(db, render_db["table"], df)

        assert rows["ts"].isna().sum() == 1


class TestValuesGoThroughPsycopg2Adaptation:
    """Not a separate failure so much as the cause of the others."""

    def test_a_quote_in_a_text_value_is_escaped_by_the_driver(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        """Text still goes through ``str``, but through ``mogrify`` as a
        parameter — so a value containing a quote is the driver's problem and
        not a syntax error.
        """
        df = pd.DataFrame({"t": ["O'Brien", 'say "hi"']})

        rows = _round_trip(db, render_db["table"], df)

        assert sorted(rows["t"]) == ["O'Brien", 'say "hi"']

    def test_booleans_arrive_as_booleans(self, db: PostgresDB, render_db: dict[str, Any]) -> None:
        df = pd.DataFrame({"b": [True, False]})

        rows = _round_trip(db, render_db["table"], df)

        assert sorted(rows["b"]) == [False, True]


class TestContainerCellsRoundTrip:
    """The server-side half of the non-scalar-cell fix.

    The conversion itself is pinned without a server in
    ``tests/test_upload_value_conversion.py``; this is the end-to-end proof that
    a frame carrying containers still creates a table and fills it.
    """

    def test_a_column_of_lists_round_trips_as_text(
        self, db: PostgresDB, render_db: dict[str, Any]
    ) -> None:
        df = pd.DataFrame({"c": pd.Series([[1, 2], ["a"]], dtype=object)})

        rows = _round_trip(db, render_db["table"], df)

        assert sorted(rows["c"]) == ["['a']", "[1, 2]"]
