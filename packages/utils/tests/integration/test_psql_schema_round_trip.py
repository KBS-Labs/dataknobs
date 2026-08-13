"""End-to-end check that the dtype ladder emits column types PostgreSQL takes.

The unit tests in ``test_psql_schema_dtypes.py`` assert the strings
``_psql_schema_line`` returns. They cannot answer the question that actually
matters for a CREATE TABLE generator: does the emitted type accept what
``upload`` then sends into it? A ladder that maps ``timedelta64`` to ``interval``
has only replaced one wrong type with another if PostgreSQL rejects the literal.

``upload`` renders every cell with ``str(value)`` over ``df.to_records()``, so
this exercises the rendered forms — ``'True'`` for a bool, ISO-8601 with a ``T``
for a timestamp, ``'86400000000 microseconds'`` for a timedelta — against real
type input functions.

**Nulls are deliberately absent from the frames below.** ``str`` renders them as
the text ``'nan'``/``'<NA>'``, which no typed column accepts; that is a defect in
``upload``'s value rendering rather than in the schema ladder, and it is
recorded separately. Including a null here would fail this test for a reason it
is not testing.

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
def schema_test_db(make_postgres_test_db: Any) -> Iterator[dict[str, Any]]:
    """A uniquely-named scratch table, dropped on teardown.

    ``make_postgres_test_db`` comes from ``dataknobs_common.testing``'s pytest11
    plugin, so it supplies the same credentials every other integration suite
    uses and the table name is unique per test rather than shared.
    """
    yield from make_postgres_test_db("test_psql_schema_")


@pytest.fixture
def db(schema_test_db: dict[str, Any]) -> PostgresDB:
    """A ``PostgresDB`` pointed at the shared integration-test database."""
    return PostgresDB(
        host=schema_test_db["host"],
        db=schema_test_db["database"],
        user=schema_test_db["user"],
        pwd=schema_test_db["password"],
        port=schema_test_db["port"],
    )


def _column_types(database: PostgresDB, table: str) -> dict[str, str]:
    cols = database.get_columns(table)
    return dict(zip(cols["column_name"], cols["data_type"], strict=True))


class TestDtypeRoundTrip:
    """Each dtype family must create a column and accept its own values."""

    def test_every_dtype_family_creates_and_uploads(
        self, db: PostgresDB, schema_test_db: dict[str, Any]
    ) -> None:
        """Bug: bool and datetime columns crashed before any SQL was emitted, and
        a timedelta column was created as `integer`, which its own values do not
        fit. All five families must now survive CREATE TABLE + INSERT.
        """
        df = pd.DataFrame(
            {
                "flag": [True, False],
                "count": pd.array([1, 2], dtype="int64"),
                "score": [1.5, 2.5],
                "ts": pd.to_datetime(["2024-01-01 10:30:00", "2024-06-01 00:00:00"]),
                "dur": pd.to_timedelta(["1 days", "2 days"]),
                "label": ["alpha", "bb"],
            }
        )

        table = schema_test_db["table"]
        db.upload(table, df)

        types = _column_types(db, table)
        assert types["flag"] == "boolean"
        assert types["count"] == "integer"
        assert types["score"] == "real"
        assert types["ts"] == "timestamp without time zone"
        assert types["dur"] == "interval"
        assert types["label"] == "character varying"

        rows = db.query(f'SELECT * FROM "{table}" ORDER BY "count"')
        assert len(rows) == 2
        assert list(rows["flag"]) == [True, False]
        assert list(rows["dur"]) == [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
        assert list(rows["ts"]) == [
            pd.Timestamp("2024-01-01 10:30:00"),
            pd.Timestamp("2024-06-01 00:00:00"),
        ]

    def test_tz_aware_column_keeps_its_offset(
        self, db: PostgresDB, schema_test_db: dict[str, Any]
    ) -> None:
        """Emitting a bare `timestamp` for a tz-aware column would drop the
        offset silently; `timestamptz` is what preserves the instant.
        """
        df = pd.DataFrame(
            {
                "count": pd.array([1], dtype="int64"),
                "ts": pd.to_datetime(["2024-01-01 12:00:00+05:00"], utc=True),
            }
        )

        table = schema_test_db["table"]
        db.upload(table, df)

        types = _column_types(db, table)
        assert types["ts"] == "timestamp with time zone"

        rows = db.query(f'SELECT * FROM "{table}"')
        assert rows["ts"].iloc[0] == pd.Timestamp("2024-01-01 07:00:00+00:00")

    def test_varchar_width_fits_the_rendered_value(
        self, db: PostgresDB, schema_test_db: dict[str, Any]
    ) -> None:
        """The width the fallback measures must be the width upload() sends, or
        the INSERT overflows the column the CREATE TABLE just made.
        """
        df = pd.DataFrame({"c": pd.array([1, 222], dtype=object)})

        table = schema_test_db["table"]
        db.upload(table, df)

        rows = db.query(f'SELECT * FROM "{table}" ORDER BY "c"')
        assert sorted(rows["c"]) == ["1", "222"]
