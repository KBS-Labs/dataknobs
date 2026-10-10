"""Async DuckDB's ``connect`` and ``close`` close no connection in use and leave none open.

Every statement on the store's connection runs on its pool under the store's
lock. ``close`` closes the connection under that lock too, and refuses an
operation from the moment it starts. A ``connect`` cancelled while its pool
thread opens the file leaves nothing holding DuckDB's lock on the file.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record

duckdb = pytest.importorskip("duckdb")

pytestmark = pytest.mark.asyncio

#: Opens the file for writing in a process of its own, as its owner would:
#: DuckDB refuses that while any other process holds the file open.
OPENS_FOR_WRITING = "import duckdb, sys; duckdb.connect(sys.argv[1]).close()"


def _opened_for_writing_elsewhere(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", OPENS_FOR_WRITING, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


async def test_close_waits_for_a_statement_running_on_the_connection(tmp_path: Path) -> None:
    """Bug: ``close`` sent ``conn.close`` to the pool without the lock every
    statement holds, so it closed the connection under a statement still
    running on it, which then failed with DuckDB's ``Connection already
    closed``. And it cleared ``_connected`` only once that close returned, so
    a read started meanwhile was queued onto the closing connection.
    """
    db = AsyncDuckDBDatabase({"path": str(tmp_path / "records.duckdb"), "table": "records"})
    await db.connect()
    await db.create(Record({"k": 1}, storage_id="a"))
    conn = db.conn
    assert conn is not None
    # The lock held here stands for a statement running on the connection.
    with db._lock:
        closing = asyncio.create_task(db.close())
        await asyncio.sleep(0.2)
        assert conn.execute("SELECT 1").fetchone() == (1,), "still open under the statement"
        with pytest.raises(RuntimeError, match="not connected"):
            await asyncio.wait_for(db.read("a"), timeout=5)
    await closing
    with pytest.raises(duckdb.ConnectionException):
        conn.execute("SELECT 1")


async def test_a_cancelled_connect_leaves_the_file_free(tmp_path: Path) -> None:
    """A ``connect`` cancelled while its pool thread opens the file discards the
    connection that thread goes on to return. Nothing refers to it, so it is
    closed as it is freed, and the owner can open its file again.
    """
    path = tmp_path / "records.duckdb"
    started, release = threading.Event(), threading.Event()

    class HeldAtTheCheck(AsyncDuckDBDatabase):
        """Stops in the pool thread, the file open, until the test lets it go on."""

        def _check_relation(self, conn: Any) -> None:
            started.set()
            release.wait(30)
            super()._check_relation(conn)

    db = HeldAtTheCheck({"path": str(path), "table": "records"})
    connecting = asyncio.create_task(db.connect())
    assert await asyncio.to_thread(started.wait, 30)
    connecting.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    assert db.conn is None and not db._connected
    owner = _opened_for_writing_elsewhere(path)
    assert owner.returncode == 0, owner.stderr


async def test_native_reads_run_side_by_side(tmp_path: Path) -> None:
    """Each native read opens the file on a connection of its own, so reads
    take no lock between them and the pool runs them at once: each still
    answers as it would alone.
    """
    path = tmp_path / "owner.duckdb"
    owner = duckdb.connect(str(path))
    owner.execute("CREATE TABLE things (k VARCHAR PRIMARY KEY, n INTEGER)")
    owner.executemany("INSERT INTO things VALUES (?, ?)", [(f"k{i}", i) for i in range(50)])
    owner.close()
    db = AsyncDuckDBDatabase(
        {
            "path": str(path),
            "table": "things",
            "layout": "native",
            "id_column": "k",
            "schema": {"fields": {"k": "string", "n": "integer"}},
            "max_workers": 8,
        }
    )
    await db.connect()
    try:
        counts, records = await asyncio.gather(
            asyncio.gather(*(db.count() for _ in range(16))),
            asyncio.gather(*(db.read(f"k{i}") for i in range(16))),
        )
    finally:
        await db.close()
    assert counts == [50] * 16
    assert [record["n"] for record in records] == list(range(16))


def _store(path: Path) -> AsyncDuckDBDatabase:
    return AsyncDuckDBDatabase({"path": str(path), "table": "records", "max_workers": 4})


async def _seeded(path: Path) -> AsyncDuckDBDatabase:
    db = _store(path)
    await db.connect()
    await db.create(Record({"k": 1}, storage_id="a"))
    return db


#: Each write, given the store and the version token of record ``a``.
WRITES: dict[str, Any] = {
    "create": lambda db, v: db.create(Record({"k": 2}, storage_id="b")),
    "update": lambda db, v: db.update("a", Record({"k": 2})),
    "update-expected": lambda db, v: db.update("a", Record({"k": 2}), expected_version=v),
    "delete": lambda db, v: db.delete("a"),
    "delete-expected": lambda db, v: db.delete("a", expected_version=v),
    "create_batch": lambda db, v: db.create_batch([Record({"k": 2}, storage_id="b")]),
    "upsert_batch": lambda db, v: db.upsert_batch([Record({"k": 2}, storage_id="a")]),
    "update_batch": lambda db, v: db.update_batch([("a", Record({"k": 2}))]),
    "delete_batch": lambda db, v: db.delete_batch(["a"]),
}


@pytest.mark.parametrize("write", list(WRITES))
async def test_a_write_queued_before_close_is_refused_by_name(tmp_path: Path, write: str) -> None:
    """Bug: the write cores read ``self.conn`` under the lock without asking
    for it, and ``close`` clears it before taking the lock. A write already
    waiting for the lock when ``close`` started then failed with
    ``AttributeError: 'NoneType' object has no attribute ...``, where a read
    is refused with the store's own error.
    """
    db = await _seeded(tmp_path / "records.duckdb")
    version = await db.get_version("a")
    # The lock held here stands for a statement running on the connection.
    with db._lock:
        writing = asyncio.ensure_future(WRITES[write](db, version))
        await asyncio.sleep(0.2)
        closing = asyncio.create_task(db.close())
        await asyncio.sleep(0.2)
    with pytest.raises(RuntimeError, match="not connected"):
        await asyncio.wait_for(writing, timeout=10)
    await closing


class HeldWithTheConnection(AsyncDuckDBDatabase):
    """Stops a write in the pool thread, the connection in hand, until the test lets it go on."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.armed = False
        self.started, self.release = threading.Event(), threading.Event()

    def _require_conn(self) -> Any:
        conn = super()._require_conn()
        if self.armed:
            self.armed = False
            self.started.set()
            self.release.wait(30)
        return conn


#: A write, and what it leaves stored under ``a`` and ``b``.
HELD_WRITES: dict[str, tuple[Any, dict[str, Any]]] = {
    "delete_batch": (lambda db: db.delete_batch(["a"]), {"a": None, "b": None}),
    "update_batch": (lambda db: db.update_batch([("a", Record({"k": 2}))]), {"a": 2, "b": None}),
    "create_batch-in-a-transaction": (
        lambda db: db.create_batch([Record({"k": 2}, storage_id="b")], _tx=object()),
        {"a": 1, "b": 2},
    ),
}


@pytest.mark.parametrize("write", list(HELD_WRITES))
async def test_close_waits_for_a_write_holding_the_connection(tmp_path: Path, write: str) -> None:
    """Bug: a write core asked for the connection once and went on reading
    ``self.conn``, which ``close`` clears before it waits for the lock, so the
    write failed partway with ``AttributeError``. It now keeps the connection
    it asked for, and ``close`` closes it once the write is done.
    """
    path = tmp_path / "records.duckdb"
    db = HeldWithTheConnection({"path": str(path), "table": "records", "max_workers": 4})
    await db.connect()
    await db.create(Record({"k": 1}, storage_id="a"))
    run, stored = HELD_WRITES[write]
    db.armed = True
    writing = asyncio.ensure_future(run(db))
    assert await asyncio.to_thread(db.started.wait, 30)
    closing = asyncio.create_task(db.close())
    await asyncio.sleep(0.2)
    db.release.set()
    await asyncio.wait_for(writing, timeout=10)
    await closing
    reopened = _store(path)
    await reopened.connect()
    try:
        got = {}
        for key in stored:
            record = await reopened.read(key)
            got[key] = None if record is None else record["k"]
    finally:
        await reopened.close()
    assert got == stored


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


async def test_a_transaction_closed_in_its_body_is_refused_by_name_once(tmp_path: Path) -> None:
    """Bug: the commit read the cleared connection and raised
    ``AttributeError``, and so did the rollback that handled it. Closing
    discards the open transaction, so the rollback now has nothing to do, and
    the commit's refusal is the one error.
    """
    db = await _seeded(tmp_path / "records.duckdb")
    with pytest.raises(RuntimeError, match="not connected") as raised:
        async with db._transaction():
            await db.close()
    assert [type(e) for e in _chain(raised.value)] == [RuntimeError]


#: Every operation, given a store.
OPERATIONS: dict[str, Any] = {
    "create": lambda db: db.create(Record({"k": 1})),
    "read": lambda db: db.read("a"),
    "update": lambda db: db.update("a", Record({"k": 1})),
    "delete": lambda db: db.delete("a"),
    "exists": lambda db: db.exists("a"),
    "search": lambda db: db.search(Query()),
    "count": lambda db: db.count(),
    "create_batch": lambda db: db.create_batch([Record({"k": 1})]),
    "upsert_batch": lambda db: db.upsert_batch([Record({"k": 1}, storage_id="a")]),
    "update_batch": lambda db: db.update_batch([("a", Record({"k": 1}))]),
    "delete_batch": lambda db: db.delete_batch(["a"]),
}


@pytest.mark.parametrize("operation", list(OPERATIONS))
async def test_every_operation_on_a_store_never_connected_is_refused_by_name(
    tmp_path: Path, operation: str
) -> None:
    sync = SyncDuckDBDatabase({"path": str(tmp_path / "s.duckdb"), "table": "records"})
    with pytest.raises(RuntimeError, match="not connected"):
        OPERATIONS[operation](sync)
    with pytest.raises(RuntimeError, match="not connected"):
        await OPERATIONS[operation](_store(tmp_path / "a.duckdb"))
