"""The sync Postgres backend reads on a connection of its own, never the caller's.

``PostgresDB`` keeps two connections per thread: the one ``get_conn()`` hands
a caller, whose transaction is the caller's alone, and the one its own
``query`` / ``execute`` share. The sync backend's reads went through the
caller's, each in a ``with conn`` that commits on the way out, and its
``stream_read`` held a named cursor in a transaction on that same connection.

Two failures followed, asserted here against a real server:

- **A read inside a stream ended the stream.** A named cursor lives only as
  long as its transaction, and the inner read's ``with conn`` committed it, so
  the next fetch raised ``named cursor isn't valid anymore``. Two streams
  consumed together on one instance broke the same way when the inner one
  finished.
- **A read committed the caller's transaction.** A consumer holding
  ``backend.db.get_conn()`` for work of its own had it committed by an
  ordinary ``backend.read()``.

The async twin needs neither test's fix: each of its streams acquires a pool
connection of its own.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import SyncPostgresDatabase
from dataknobs_data.query import Query, SortSpec
from dataknobs_data.streaming import StreamConfig

pytestmark = requires_postgres

ROWS = [Record({"name": f"n{i:02d}", "rank": i}, id=f"r{i:02d}") for i in range(12)]


@pytest.fixture
def sync_pg(make_postgres_test_db: Any) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_sync_stream_conn_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        for record in ROWS:
            db.create(record)
        try:
            yield db
        finally:
            db.close()


def _by_rank() -> Query:
    return Query(sort_specs=[SortSpec("rank")])


def test_a_read_inside_a_stream_does_not_end_it(sync_pg: SyncPostgresDatabase) -> None:
    """Every record streams, with a read of another record between each fetch."""
    seen = []
    for record in sync_pg.stream_read(_by_rank(), StreamConfig(batch_size=2)):
        assert sync_pg.read("r00") is not None
        assert sync_pg.exists("r11")
        assert sync_pg.count() == len(ROWS)
        seen.append(record.get_value("name"))
    assert seen == [r.get_value("name") for r in ROWS]


def test_two_streams_interleave_on_one_instance(sync_pg: SyncPostgresDatabase) -> None:
    """Zipping two streams yields both in full, whichever finishes first."""
    short = sync_pg.stream_read(_by_rank().limit(3), StreamConfig(batch_size=1))
    long = sync_pg.stream_read(_by_rank(), StreamConfig(batch_size=1))
    pairs = list(zip(short, long, strict=False))
    assert [a.get_value("rank") for a, _ in pairs] == [0, 1, 2]
    # The short stream is exhausted; the long one keeps fetching after it.
    rest = [record.get_value("rank") for record in long]
    assert rest == list(range(3, len(ROWS)))


def test_a_read_leaves_the_callers_transaction_open(sync_pg: SyncPostgresDatabase) -> None:
    """Work left open on ``db.get_conn()`` is still the caller's to commit or roll back."""
    conn = sync_pg.db.get_conn()
    table = sync_pg._q_qualified
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE id = %(id)s", {"id": "r05"})
    try:
        sync_pg.read("r00")
        list(sync_pg.stream_read(_by_rank()))
        sync_pg.search(_by_rank())
    finally:
        conn.rollback()
    assert sync_pg.read("r05") is not None, "a backend read committed the caller's DELETE"


def test_an_abandoned_stream_releases_its_connection(sync_pg: SyncPostgresDatabase) -> None:
    """Closing a stream part-way closes the connection it read on.

    The stream's server backend is the one holding a lock on this test's
    table: its cursor's transaction keeps one for as long as it is open, and
    the table's name is this test's alone, so a stream another test holds open
    on the same server is not counted. Once the stream is closed that backend
    is gone.

    With nothing else referencing the connection, reference counting would
    close it even without the stream's own ``close()``; what this catches is a
    connection something still holds -- a strong registry, a lingering frame --
    that only the explicit close releases.
    """
    stream = sync_pg.stream_read(_by_rank(), StreamConfig(batch_size=1))
    next(stream)
    holders = sync_pg.db.query_rows(
        "SELECT DISTINCT pid FROM pg_locks "
        "WHERE relation = to_regclass(%(table)s) AND pid <> pg_backend_pid()",
        {"table": sync_pg._q_qualified},
    )
    assert len(holders) == 1, holders
    pid = holders[0]["pid"]
    alive = "SELECT count(*) AS n FROM pg_stat_activity WHERE pid = %(pid)s"
    assert sync_pg.db.query_rows(alive, {"pid": pid})[0]["n"] == 1
    stream.close()
    deadline = time.monotonic() + 5.0
    while sync_pg.db.query_rows(alive, {"pid": pid})[0]["n"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sync_pg.db.query_rows(alive, {"pid": pid})[0]["n"] == 0
