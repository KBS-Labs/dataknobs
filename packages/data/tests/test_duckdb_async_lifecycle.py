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

from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase
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
