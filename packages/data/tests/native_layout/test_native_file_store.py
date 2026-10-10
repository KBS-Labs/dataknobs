"""A SQLite or DuckDB store reads somebody else's file without changing it.

The file is opened read-only at the driver, so nothing a store does can write
to it: no table, no journal mode, no directory, and no file where there was
none. What the owner made is read in place -- a view as well as a table.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
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
    ids,
    named,
    opened,
    write_duckdb_file,
    write_sqlite_file,
)

from dataknobs_data.query import Query

FILE_ENGINES = ["sqlite", "duckdb"]

#: A view over the tickets that are open, made by the file's owner.
OPEN_TICKETS = (
    "CREATE VIEW open_tickets AS SELECT id, tenant_id, subject, status FROM tickets "
    "WHERE status = 'open'"
)
VIEW_FIELDS = {name: TICKET_FIELDS[name] for name in ("id", "tenant_id", "subject", "status")}


def _write(engine: str, directory: Path) -> Path:
    if engine == "sqlite":
        path = write_sqlite_file(directory / "owner.db", (TICKETS,))
        conn = sqlite3.connect(path)
        conn.execute(OPEN_TICKETS)
        conn.commit()
        conn.close()
        return path
    duckdb = pytest.importorskip("duckdb")
    path = write_duckdb_file(directory / "owner.duckdb", (TICKETS,))
    conn = duckdb.connect(str(path))
    conn.execute(OPEN_TICKETS)
    conn.close()
    return path


@pytest.fixture(params=FILE_ENGINES)
def owned(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A file somebody else wrote, holding ``tickets`` and a view over it."""
    engine = str(request.param)
    yield engine, _write(engine, tmp_path)


@pytest.fixture(params=TWINS)
def twin(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _desk(engine: str, path: Path) -> Helpdesk:
    return Helpdesk(engine, {"path": str(path)})


def _snapshot(directory: Path) -> dict[str, str]:
    """Every file in ``directory`` and a hash of its bytes."""
    return {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in sorted(directory.iterdir())
    }


def test_a_view_is_read_in_place(owned: tuple[str, Path], twin: str) -> None:
    """Bug: SQLite's existence check read ``sqlite_master`` for tables only,
    so a view -- a plausible thing to read in place -- was refused as missing.
    """
    engine, path = owned
    config = _desk(engine, path).config("open_tickets", VIEW_FIELDS)
    with opened(twin, config) as db:
        assert ids(db.search()) == named(T1, T3)
        assert db.count() == 2


def test_reading_leaves_the_file_as_it_was(owned: tuple[str, Path], twin: str) -> None:
    """Every byte of the owner's file, and nothing new beside it.

    The async SQLite backend sets a ``WAL`` journal on a file by default, which
    is written into the file and stays after the store closes.
    """
    engine, path = owned
    before = _snapshot(path.parent)
    with opened(twin, _desk(engine, path).config("tickets", TICKET_FIELDS)) as db:
        assert db.count() == 4
    assert _snapshot(path.parent) == before
    if engine == "sqlite":
        conn = sqlite3.connect(path)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        conn.close()


@pytest.fixture
def read_only_file(owned: tuple[str, Path]) -> Iterator[tuple[str, Path]]:
    """The owner's file with no write permission on it or on its directory."""
    engine, path = owned
    path.chmod(0o444)
    path.parent.chmod(0o555)
    try:
        yield engine, path
    finally:
        path.parent.chmod(0o755)
        path.chmod(0o644)


def test_a_file_this_process_cannot_write_is_read(
    read_only_file: tuple[str, Path], twin: str
) -> None:
    """Read-only at the driver, so a file shared without write permission reads.

    The JSON layout creates its table on connect, which a file it cannot write
    refuses: the positive control, showing the file really is unwritable.
    """
    if os.geteuid() == 0:
        pytest.skip("root writes to a file whatever its permissions")
    engine, path = read_only_file
    config = _desk(engine, path).config("tickets", TICKET_FIELDS)
    with opened(twin, config) as db:
        assert ids(db.search()) == named(T1, T2, T3, T4)
        assert db.count() == 4

    plain = {"backend": engine, "path": str(path), "table": "records"}
    with pytest.raises(Exception):  # noqa: B017 -- each driver names its own refusal
        with opened(twin, plain):
            pass


def test_a_missing_file_is_refused_and_nothing_is_created(tmp_path: Path, twin: str) -> None:
    """Bug: connecting made the directory and, through the driver, an empty
    file, then reported the table missing from a database it had just created.
    """
    for engine in FILE_ENGINES:
        if engine == "duckdb":
            pytest.importorskip("duckdb")
        path = tmp_path / "never" / f"made.{engine}"
        config = _desk(engine, path).config("tickets", TICKET_FIELDS)
        with pytest.raises(RuntimeError) as caught:
            with opened(twin, config):
                pass
        message = str(caught.value)
        assert str(path) in message and "layout: native" in message
        assert not path.parent.exists(), "neither the directory nor the file is made"


def test_two_twins_read_one_file_at_once(owned: tuple[str, Path]) -> None:
    """Every native reader opens the file the same way, so two can share it."""
    engine, path = owned
    config = _desk(engine, path).config("tickets", TICKET_FIELDS)
    with opened("sync", config) as sync_db, opened("async", config) as async_db:
        assert sync_db.count() == async_db.count() == 4
        assert ids(sync_db.search(Query())) == ids(async_db.search(Query()))


def test_a_refused_connect_leaves_nothing_open(owned: tuple[str, Path], twin: str) -> None:
    """Bug: a connect refused after the file was opened -- a table that is not
    there -- left the connection open. On async SQLite its worker thread kept
    the process from exiting; on DuckDB the file stayed held, so its owner
    could not open it for writing in the same process.
    """
    import threading

    engine, path = owned
    config = _desk(engine, path).config("no_such_table", {"id": TICKET_FIELDS["id"]}, scope=[])
    threads = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="no_such_table"):
        with opened(twin, config):
            pass
    assert set(threading.enumerate()) - threads == set(), (
        "nothing the refused connect started runs on"
    )
    if engine == "duckdb":
        duckdb = pytest.importorskip("duckdb")
        duckdb.connect(str(path)).close()
