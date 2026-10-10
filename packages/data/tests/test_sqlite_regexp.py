"""``Operator.REGEX`` answers on SQLite as ``Filter.matches`` answers.

SQLite parses ``x REGEXP y`` but ships no function behind it: a connection
that registers none raises ``no such function: REGEXP``. Both SQLite backends
register one on every connection they open, under either layout: an unanchored
``re.search`` over a string, and no match for a value that is not one.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.backends.sqlite_mixins import register_regexp, sqlite_regexp
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.records import Record

ROWS = {"beagle": "Beagle", "poodle": "Poodle", "five": 5, "none": None}
QUERY = Query(filters=[Filter("t", Operator.REGEX, "eag")])


def _search(cls: type, config: dict[str, Any], rows: list[Record] | None = None) -> set[str]:
    db = cls(config)
    if cls is AsyncSQLiteDatabase:

        async def run() -> list[Record]:
            await db.connect()
            try:
                if rows:
                    await db.create_batch(rows)
                return await db.search(QUERY)
            finally:
                await db.close()

        found = asyncio.run(run())
    else:
        db.connect()
        try:
            if rows:
                db.create_batch(rows)
            found = db.search(QUERY)
        finally:
            db.close()
    return {str(r.storage_id) for r in found}


@pytest.mark.parametrize("cls", [SyncSQLiteDatabase, AsyncSQLiteDatabase], ids=lambda c: c.__name__)
@pytest.mark.parametrize("path", [":memory:", "file"])
def test_regex_under_the_json_layout(cls: type, path: str, tmp_path: Path) -> None:
    """Bug: every SQLite backend raised ``no such function: REGEXP``."""
    rows = [Record({"t": value}, storage_id=key) for key, value in ROWS.items()]
    location = path if path == ":memory:" else str(tmp_path / "regexp.db")
    assert _search(cls, {"path": location}, rows) == {"beagle"}


@pytest.mark.parametrize("cls", [SyncSQLiteDatabase, AsyncSQLiteDatabase], ids=lambda c: c.__name__)
def test_regex_under_the_native_layout(cls: type, tmp_path: Path) -> None:
    path = tmp_path / "dogs.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE dogs (k TEXT PRIMARY KEY, t TEXT)")
    conn.executemany("INSERT INTO dogs VALUES (?, ?)", [("beagle", "Beagle"), ("poodle", "Poodle")])
    conn.commit()
    conn.close()
    config = {
        "path": str(path),
        "table": "dogs",
        "layout": "native",
        "id_column": "k",
        "schema": {"fields": {"k": "string", "t": "string"}},
    }
    assert _search(cls, config) == {"beagle"}


@pytest.mark.parametrize(
    ("pattern", "value", "expected"),
    [
        ("eag", "Beagle", True),
        ("^eag", "Beagle", False),
        ("eag", "beagle", True),
        ("EAG", "Beagle", False),
        ("5", 5, False),
        ("x", None, False),
        ("x", b"x", False),
    ],
)
def test_the_function_answers_as_filter_matches(pattern: str, value: Any, expected: bool) -> None:
    assert sqlite_regexp(pattern, value) is expected
    assert Filter("t", Operator.REGEX, pattern).matches(value) is expected


def test_a_registered_connection_answers_in_sql() -> None:
    conn = sqlite3.connect(":memory:")
    register_regexp(conn)
    assert conn.execute(
        "SELECT 'Beagle' REGEXP 'eag', 5 REGEXP '5', NULL REGEXP 'x'"
    ).fetchone() == (
        1,
        0,
        0,
    )
    conn.close()
