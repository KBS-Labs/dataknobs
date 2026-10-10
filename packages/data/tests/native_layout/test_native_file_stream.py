"""A stream over somebody else's SQLite or DuckDB table pages by its sort keys.

The store reads the file read-only, and its owner may write it meanwhile. Each
page after the first is found by the last row's keys, which the engine reads
back alongside the row, so a write ahead of the stream's position moves
nothing, and for a table nobody writes the stream is exactly the search.

Only SQLite is written during a stream here: DuckDB refuses, within one
process, a writer on a file a read-only connection holds open.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from _helpdesk import (
    T1,
    T2,
    T3,
    T4,
    TICKET_FIELDS,
    TICKETS,
    TWINS,
    Helpdesk,
    opened,
    write_duckdb_file,
    write_sqlite_file,
)

from dataknobs_data.query import (
    RESERVED_KEY_FIELD,
    Query,
    SortOrder,
    SortSpec,
    is_storage_key_field,
)
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig

FILE_ENGINES = ["sqlite", "duckdb"]

#: Every ticket with one of them twice: a view whose key is not unique.
DOUBLED = (
    "CREATE VIEW doubled AS SELECT id, tenant_id, number, subject FROM tickets "
    "UNION ALL SELECT id, tenant_id, number, subject FROM tickets WHERE number = 102"
)
DOUBLED_FIELDS = {name: TICKET_FIELDS[name] for name in ("id", "tenant_id", "number", "subject")}


def _write(engine: str, directory: Path) -> Path:
    if engine == "sqlite":
        path = write_sqlite_file(directory / "owner.db", (TICKETS,))
        conn = sqlite3.connect(path)
        conn.execute(DOUBLED)
        conn.commit()
        conn.close()
        return path
    duckdb = pytest.importorskip("duckdb")
    path = write_duckdb_file(directory / "owner.duckdb", (TICKETS,))
    conn = duckdb.connect(str(path))
    conn.execute(DOUBLED)
    conn.close()
    return path


@pytest.fixture(params=FILE_ENGINES)
def owned(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    engine = str(request.param)
    yield engine, _write(engine, tmp_path)


@pytest.fixture(params=TWINS)
def twin(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _ids(records: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in records]


def _with_key(query: Query) -> Query:
    total = query.copy()
    if not any(is_storage_key_field(s.field) for s in total.sort_specs):
        total.sort_specs.append(SortSpec(RESERVED_KEY_FIELD))
    return total


# --- the owner writes between two pages ------------------------------------------------

BY_NUMBER = Query(sort_specs=[SortSpec("number")])
#: Acme's four tickets, two to a page: the write lands between the pages.
TWO = StreamConfig(batch_size=2)


def _owner_runs(path: Path, sql: str, *params: object) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Path:
    return _write("sqlite", tmp_path)


def test_the_owner_inserting_ahead_of_a_stream_moves_nothing(sqlite_file: Path, twin: str) -> None:
    """Bug: the second page's offset of two counted the inserted ticket, so
    the second ticket read was read again.
    """
    config = Helpdesk("sqlite", {"path": str(sqlite_file)}).config("tickets", TICKET_FIELDS)
    with opened(twin, config) as db:
        seen = db.stream(
            BY_NUMBER,
            TWO,
            write_after=2,
            write=lambda: _owner_runs(
                sqlite_file,
                "INSERT INTO tickets (id, tenant_id, number, subject, status, priority, "
                "opened_at, tags) VALUES (?, 'acme', 100, 'new', 'open', 1, "
                "'2026-03-01T00:00:00+00:00', '[]')",
                str(uuid.uuid4()),
            ),
        )
    assert _ids(seen) == [str(t) for t in (T1, T2, T3, T4)]


def test_the_owner_deleting_behind_a_stream_moves_nothing(sqlite_file: Path, twin: str) -> None:
    """Bug: with the first ticket gone, the second page began a row late."""
    config = Helpdesk("sqlite", {"path": str(sqlite_file)}).config("tickets", TICKET_FIELDS)
    with opened(twin, config) as db:
        seen = db.stream(
            BY_NUMBER,
            TWO,
            write_after=2,
            write=lambda: _owner_runs(sqlite_file, "DELETE FROM tickets WHERE number = 101"),
        )
    assert _ids(seen) == [str(t) for t in (T1, T2, T3, T4)]


# --- an unwritten table: the stream is the search --------------------------------------

QUERIES = {
    "a zoned time": Query(sort_specs=[SortSpec("opened_at")]),
    "a zoned time, descending": Query(sort_specs=[SortSpec("opened_at", SortOrder.DESC)]),
    "priority desc, number": Query(
        sort_specs=[SortSpec("priority", SortOrder.DESC), SortSpec("number")]
    ),
    "status, then time desc": Query(
        sort_specs=[SortSpec("status"), SortSpec("opened_at", SortOrder.DESC)]
    ),
    "a key some rows lack": Query(sort_specs=[SortSpec("category_id")]),
    "the key, descending": Query(sort_specs=[SortSpec(RESERVED_KEY_FIELD, SortOrder.DESC)]),
    "subject, offset and limit": Query(
        sort_specs=[SortSpec("subject")], offset_value=1, limit_value=3
    ),
}


@pytest.mark.parametrize("batch", [1, 2, 10])
@pytest.mark.parametrize("name", list(QUERIES))
def test_a_native_stream_reads_what_search_does(
    owned: tuple[str, Path], twin: str, name: str, batch: int
) -> None:
    """Every tenant's tickets, so ties and a missing category meet a page boundary."""
    engine, path = owned
    config = Helpdesk(engine, {"path": str(path)}).config("tickets", TICKET_FIELDS, scope=[])
    query = QUERIES[name]
    with opened(twin, config) as db:
        expected = db.search(_with_key(query))
        seen = db.stream(query, StreamConfig(batch_size=batch))
    assert _ids(seen) == _ids(expected)
    assert [r.to_dict() for r in seen] == [r.to_dict() for r in expected]


# --- a view whose key is not unique ----------------------------------------------------


def test_a_row_tying_a_page_boundary_on_every_key_is_read_once(
    owned: tuple[str, Path], twin: str
) -> None:
    """A stream needs a unique key. On a view without one, the second of two
    rows identical in every key, falling after a page boundary, is not read:
    the next page begins strictly after the first. Nothing is read twice.
    """
    engine, path = owned
    config = Helpdesk(engine, {"path": str(path)}).config("doubled", DOUBLED_FIELDS)
    with opened(twin, config) as db:
        assert _ids(db.search(_with_key(BY_NUMBER))) == [str(t) for t in (T1, T2, T2, T3, T4)]
        seen = db.stream(BY_NUMBER, TWO)
    assert _ids(seen) == [str(t) for t in (T1, T2, T3, T4)]
