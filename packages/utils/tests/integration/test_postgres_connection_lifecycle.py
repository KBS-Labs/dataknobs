"""Connection lifecycle for ``PostgresDB`` — reuse, and an actual close.

Reproduce-first for two defects that look like one and are not.

**Nothing in the class ever closed a connection.** ``with conn`` is psycopg2's
*transaction* scope, not a close: after the block ``conn.closed`` is still 0.
The three internal acquisitions got away with it because CPython reclaims the
local when the frame exits and psycopg2's dealloc closes the socket — so a
count of live backends never grows, and a test written to catch a leak that way
passes against the unfixed code. That mask is an implementation detail of the
interpreter, and it does not extend to a caller of the public ``get_conn()``,
which is the idiom this class models in three places.

**What the mask does not hide is the cost.** Every ``query`` / ``execute`` /
``upload`` paid a full TCP+auth handshake: measured at 20 connects for 20
``SELECT 1`` calls, 79% of wall time inside ``connect()``. The consumer that
felt it is ``dataknobs_data``'s ``SyncPostgresDatabase``, which is built on this
class and pays one handshake per CRUD operation.

Identity is asserted with ``pg_backend_pid()`` rather than by counting rows in
``pg_stat_activity``. A count answers "how many are open", which refcounting
already kept at zero; the backend PID answers "is this the *same* connection",
which is the property under test and the one that was false.

Requires a reachable PostgreSQL instance (``bin/dk up``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_utils.sql_utils import DotenvPostgresConnector, PostgresDB

pytestmark = [requires_postgres, pytest.mark.postgres, pytest.mark.integration]


@pytest.fixture
def lifecycle_db(make_postgres_test_db: Any) -> Iterator[dict[str, Any]]:
    """Credentials from the shared integration plugin, with a scratch table."""
    yield from make_postgres_test_db("test_conn_lifecycle_")


@pytest.fixture
def db(lifecycle_db: dict[str, Any]) -> Iterator[PostgresDB]:
    database = PostgresDB(
        host=lifecycle_db["host"],
        db=lifecycle_db["database"],
        user=lifecycle_db["user"],
        pwd=lifecycle_db["password"],
        port=lifecycle_db["port"],
    )
    yield database
    database.close()


def _backend_pid(database: PostgresDB) -> int:
    """The server-side PID serving this call — stable iff the connection is."""
    return int(database.query("SELECT pg_backend_pid() AS pid")["pid"].iloc[0])


class TestConnectionIsReused:
    """One connection per ``PostgresDB``, not one per call."""

    def test_repeated_queries_share_one_backend(self, db: PostgresDB) -> None:
        """Bug: each query opened a new connection, so every call paid a full
        handshake. Measured before the fix: 20 connects for 20 queries, 79% of
        wall time in ``connect()``.
        """
        pids = {_backend_pid(db) for _ in range(5)}

        assert len(pids) == 1, f"expected one reused backend, saw {len(pids)}: {pids}"

    def test_execute_shares_the_query_connection(self, db: PostgresDB) -> None:
        """``execute`` acquired separately from ``query``; both go through the
        same connector, so both must land on the same backend.
        """
        before = _backend_pid(db)
        db.execute("SELECT 1")

        assert _backend_pid(db) == before


class TestCloseActuallyCloses:
    """The class had no ``close`` at all — not a no-op one, none."""

    def test_close_closes_the_underlying_connection(self, db: PostgresDB) -> None:
        conn = db.get_conn()
        assert conn.closed == 0, "precondition: a fresh connection is open"

        db.close()

        assert conn.closed != 0, "close() left the psycopg2 connection open"

    def test_with_block_leaves_the_connection_open(self, db: PostgresDB) -> None:
        """The mechanism behind the whole finding, pinned so it cannot be
        misread again: psycopg2's ``with conn`` commits a transaction and does
        NOT close. Anything relying on it to close is relying on refcounting.
        """
        conn = db.get_conn()

        with conn:
            with conn.cursor() as curs:
                curs.execute("SELECT 1")

        assert conn.closed == 0, "psycopg2 changed: `with conn` now closes"

    def test_query_after_close_reopens(self, db: PostgresDB) -> None:
        """Closing is not poisoning. A reused connection has to survive being
        closed underneath it — a dropped server-side connection is the ordinary
        case, and the reopen path is the same one.
        """
        first = _backend_pid(db)
        db.close()
        second = _backend_pid(db)

        assert second != first, "expected a new backend after close()"

    def test_context_manager_closes_on_exit(self, lifecycle_db: dict[str, Any]) -> None:
        with PostgresDB(
            host=lifecycle_db["host"],
            db=lifecycle_db["database"],
            user=lifecycle_db["user"],
            pwd=lifecycle_db["password"],
            port=lifecycle_db["port"],
        ) as database:
            conn = database.get_conn()
            assert conn.closed == 0

        assert conn.closed != 0, "__exit__ did not close the connection"

    def test_close_is_idempotent(self, db: PostgresDB) -> None:
        db.close()
        db.close()


class TestReuseIsPerThread:
    """Reuse must not become sharing.

    A single connection shared across threads breaks in two ways, and the
    louder one is not the worse one. psycopg2's ``with conn`` transaction block
    is not re-entrant, so a second thread entering it raises
    ``ProgrammingError: the connection cannot be re-entered recursively`` — and
    beneath that, two threads on one connection are inside *one transaction*,
    so either one's commit would commit the other's uncommitted work.

    psycopg2 is threadsafety level 2, meaning connections *may* be shared
    between threads. That permits passing one around; it does not make a
    transaction block shareable, which is what this class pins.
    """

    def test_threads_do_not_share_a_backend(self, db: PostgresDB) -> None:
        """Each thread gets its own connection, so each gets its own backend."""
        seen: list[int] = []
        lock = threading.Lock()

        def probe() -> None:
            pid = _backend_pid(db)
            with lock:
                seen.append(pid)

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda _: probe(), range(5)))

        assert len(set(seen)) > 1, "all five threads shared one backend"

    def test_a_thread_still_reuses_its_own_connection(self, db: PostgresDB) -> None:
        """Per-thread must not mean per-call: the whole point is still reuse."""
        pids: list[int] = []

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: pids.extend(_backend_pid(db) for _ in range(4))).result()

        assert len(set(pids)) == 1, f"one thread opened {len(set(pids))} connections"

    def test_concurrent_writes_do_not_re_enter_one_connection(
        self, db: PostgresDB, lifecycle_db: dict[str, Any]
    ) -> None:
        """The exact shape that failed: concurrent writers through one object.

        Regression guard for `ProgrammingError: the connection cannot be
        re-entered recursively`, raised when two threads entered the same
        connection's transaction block.
        """
        table = lifecycle_db["table"]
        db.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (n integer)')

        def insert(n: int) -> None:
            db.execute(f'INSERT INTO "{table}" (n) VALUES (%(n)s)', {"n": n})

        with ThreadPoolExecutor(max_workers=5) as pool:
            for future in [pool.submit(insert, n) for n in range(10)]:
                future.result()

        rows = db.query(f'SELECT n FROM "{table}" ORDER BY n')
        assert sorted(rows["n"]) == list(range(10))

    def test_close_closes_connections_opened_by_other_threads(self, db: PostgresDB) -> None:
        """Per-thread caching must not become a leak the owner cannot reach.

        ``close()`` is called from one thread and has to reach every connection
        the object opened, or a thread pool leaves one behind per worker.
        """
        opened: list[Any] = []

        def open_one() -> None:
            _backend_pid(db)
            opened.append(db.get_conn())

        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in [pool.submit(open_one) for _ in range(3)]:
                future.result()
        assert len({id(c) for c in opened}) == 3, "expected three distinct connections"

        db.close()

        assert all(c.closed for c in opened), "close() missed another thread's connection"


class TestConnectorOwnsTheLifecycle:
    """The seam the fix lands on, exercised directly.

    ``PostgresDB`` delegates to the connector, and the connector is the
    injectable point — a consumer supplying its own is the documented way to
    change connection behaviour, so the reuse and the close both have to be
    observable there rather than only through the wrapper.
    """

    def test_connector_hands_back_the_same_connection(self, lifecycle_db: dict[str, Any]) -> None:
        connector = DotenvPostgresConnector(
            host=lifecycle_db["host"],
            db=lifecycle_db["database"],
            user=lifecycle_db["user"],
            pwd=lifecycle_db["password"],
            port=lifecycle_db["port"],
        )
        try:
            assert connector.get_conn() is connector.get_conn()
        finally:
            connector.close()

    def test_connector_reopens_a_connection_closed_underneath_it(
        self, lifecycle_db: dict[str, Any]
    ) -> None:
        """A cached connection that someone else closed must not be handed out
        again — the failure would be a ``InterfaceError: connection already
        closed`` on a call that did nothing wrong.
        """
        connector = DotenvPostgresConnector(
            host=lifecycle_db["host"],
            db=lifecycle_db["database"],
            user=lifecycle_db["user"],
            pwd=lifecycle_db["password"],
            port=lifecycle_db["port"],
        )
        try:
            first = connector.get_conn()
            first.close()

            second = connector.get_conn()

            assert second.closed == 0
            assert second is not first
        finally:
            connector.close()
