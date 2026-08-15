"""``SyncPostgresDatabase.close()`` has to close something.

It used to close nothing. The body was a comment — *"PostgresDB manages its own
connections via context managers"* — and an assignment to ``_connected``. The
comment is false: psycopg2's ``with conn`` is a transaction scope, not a close,
and ``PostgresDB`` had no ``close`` at all to delegate to.

Nothing showed up as exhausted connections because CPython reclaimed each
connection when the frame exited, which is an interpreter detail rather than
anything this class arranged. What it did leave unmet was the contract: a
caller that closes a database expects the connection to go away, and a caller
that does *not* close one expects to keep paying for it — neither was true.

The second test here is the other half of the same fix. Every CRUD operation
went through ``PostgresDB.query``/``execute``, each of which opened its own
connection, so a read and a write never shared a backend. That is the cost the
refcounting mask hid.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import SyncPostgresDatabase

pytestmark = requires_postgres


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_close_ownership_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        try:
            yield db
        finally:
            db.close()


def _backend_pid(db: SyncPostgresDatabase) -> int:
    """The server-side PID serving this backend's calls."""
    return int(db.db.query("SELECT pg_backend_pid() AS pid")["pid"].iloc[0])


class TestCloseClosesTheConnection:
    def test_close_closes_the_psycopg2_connection(self, sync_pg: SyncPostgresDatabase) -> None:
        """Bug: close() set a flag and left the connection open."""
        sync_pg.create(Record({"probe": "a"}))
        conn = sync_pg.db.get_conn()
        assert conn.closed == 0, "precondition: the connection is open"

        sync_pg.close()

        assert conn.closed != 0, "close() left the psycopg2 connection open"

    def test_close_is_idempotent(self, sync_pg: SyncPostgresDatabase) -> None:
        """The fixture closes again on teardown, so this is not hypothetical."""
        sync_pg.close()
        sync_pg.close()

    def test_close_marks_disconnected(self, sync_pg: SyncPostgresDatabase) -> None:
        """The one thing the old body did get right must survive the fix."""
        sync_pg.close()

        assert sync_pg._connected is False


class TestOperationsShareOneConnection:
    def test_reads_and_writes_share_a_backend(self, sync_pg: SyncPostgresDatabase) -> None:
        """Bug: every CRUD operation opened its own connection, so a full
        TCP+auth handshake sat in front of each read and each write.
        """
        before = _backend_pid(sync_pg)
        record = sync_pg.create(Record({"probe": "b"}))
        sync_pg.read(record if isinstance(record, str) else record.id)
        after = _backend_pid(sync_pg)

        assert after == before, "a CRUD operation opened a second connection"
