"""Tests for ``dataknobs_common.testing.postgres_fixtures``.

Covers:

* ``wait_for_postgres`` retry + timeout paths (psycopg2 stubbed to control
  failure / success ordering, ``time.sleep`` neutralized).
* ``postgres_connection_params`` Docker-detection branch (env-var driven).
* Factory-fixture cleanup runs even when the test body raises
  (gated by ``@requires_postgres`` — needs a real cluster).
"""

from __future__ import annotations

import os

import pytest

from dataknobs_common.testing import (
    postgres_fixtures,
    requires_postgres,
    safe_sql_ident,
    wait_for_postgres,
)

# -- wait_for_postgres ------------------------------------------------------


class _FakeOperationalError(Exception):
    """Stand-in for psycopg2.OperationalError so tests don't need psycopg2."""


class _FakePsycopg2:
    """Minimal psycopg2 stand-in for retry/timeout testing.

    Mocking here is justified: the goal is to verify the retry loop's
    control flow under a deterministic failure pattern, not to exercise
    real driver behavior. Driver behavior is covered by the
    ``@requires_postgres``-gated integration smoke test below.
    """

    def __init__(self, fail_count: int) -> None:
        self.OperationalError = _FakeOperationalError
        self._fail_count = fail_count
        self.connect_calls = 0

    def connect(self, **kwargs: object) -> object:
        self.connect_calls += 1
        if self.connect_calls <= self._fail_count:
            raise self.OperationalError("simulated connection failure")

        class _Conn:
            def close(self) -> None:
                pass

        return _Conn()


def test_wait_for_postgres_succeeds_after_transient_failures(monkeypatch):
    """Retry loop returns True after psycopg2 reports failures then succeeds."""
    fake = _FakePsycopg2(fail_count=3)
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)
    monkeypatch.setattr(
        "dataknobs_common.testing.postgres_fixtures.time.sleep",
        lambda _seconds: None,
    )

    assert wait_for_postgres("h", 5432, "u", "p", max_retries=10) is True
    assert fake.connect_calls == 4


def test_wait_for_postgres_raises_after_max_retries(monkeypatch):
    """Retry loop re-raises psycopg2.OperationalError after exhausting retries."""
    fake = _FakePsycopg2(fail_count=99)  # always fails
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)
    monkeypatch.setattr(
        "dataknobs_common.testing.postgres_fixtures.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(_FakeOperationalError):
        wait_for_postgres("h", 5432, "u", "p", max_retries=3)
    assert fake.connect_calls == 3


# -- postgres_connection_params Docker detection ---------------------------


def _call_params_fixture(env_overrides: dict[str, str | None]) -> dict[str, object]:
    """Invoke the underlying fixture function with a controlled environment."""
    fixture_fn = postgres_fixtures.postgres_connection_params.__wrapped__  # type: ignore[attr-defined]
    return fixture_fn()


def _clear_postgres_env(monkeypatch) -> None:
    for name in (
        "DOCKER_CONTAINER",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
    ):
        monkeypatch.delenv(name, raising=False)


def test_postgres_connection_params_localhost_default(monkeypatch):
    """Outside Docker, ``host`` defaults to ``localhost``."""
    _clear_postgres_env(monkeypatch)
    # Ensure /.dockerenv lookup returns False from this test's perspective —
    # patch os.path.exists for that specific path.
    real_exists = os.path.exists
    monkeypatch.setattr(
        "dataknobs_common.testing.postgres_fixtures.os.path.exists",
        lambda p: False if p == "/.dockerenv" else real_exists(p),
    )

    params = _call_params_fixture({})
    assert params["host"] == "localhost"
    assert params["port"] == 5432
    assert params["database"] == "dataknobs_test"


def test_postgres_connection_params_docker_default(monkeypatch):
    """Inside a Docker container (DOCKER_CONTAINER set), host defaults to ``postgres``."""
    _clear_postgres_env(monkeypatch)
    monkeypatch.setenv("DOCKER_CONTAINER", "1")
    monkeypatch.setattr(
        "dataknobs_common.testing.postgres_fixtures.os.path.exists",
        lambda _p: False,
    )

    params = _call_params_fixture({})
    assert params["host"] == "postgres"


def test_postgres_connection_params_explicit_env_overrides(monkeypatch):
    """Explicit ``POSTGRES_*`` env vars override Docker-detection defaults."""
    _clear_postgres_env(monkeypatch)
    monkeypatch.setenv("DOCKER_CONTAINER", "1")
    monkeypatch.setenv("POSTGRES_HOST", "custom-host")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_USER", "alice")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "mydb")

    params = _call_params_fixture({})
    assert params == {
        "host": "custom-host",
        "port": 6543,
        "user": "alice",
        "password": "secret",
        "database": "mydb",
    }


# -- safe_sql_ident is the same module's identifier guard ------------------


def test_safe_sql_ident_used_by_fixtures():
    """Sanity: fixtures import the same ``safe_sql_ident`` exported from the package."""
    assert safe_sql_ident("test_records_abcdef") == "test_records_abcdef"
    with pytest.raises(ValueError):
        safe_sql_ident("bad; DROP TABLE x; --")


# -- Integration smoke test (real Postgres) --------------------------------


class _FixedUUID:
    """Deterministic stand-in so the fixture's table name is predictable."""

    hex = "deadbeefcafef00d"


def _table_exists(params: dict, schema: str, table: str) -> bool:
    import psycopg2

    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        row = cursor.fetchone()
        cursor.close()
        return row is not None and row[0] is not None
    finally:
        conn.close()


@requires_postgres
def test_make_pgvector_test_table_pre_drops_leaked_table(
    make_pgvector_test_table, postgres_connection_params, monkeypatch
):
    """The **pre-drop** removes a leaked same-named table before yield.

    A killed prior session leaks ``public.test_*``; post-only teardown
    can never guarantee a clean slate under killed sessions, so the
    factory drops the (deterministically-named) table *before* yielding.
    This proves that mechanism using only psycopg2 — deliberately
    avoiding a ``dataknobs-data`` import so ``dataknobs-common``'s test
    suite keeps no cross-package coupling (the production init-time
    dimension guard, which relies on this pre-drop in production, has
    its own reproduce-first test in ``dataknobs-data``).
    """
    import psycopg2

    monkeypatch.setattr(
        "dataknobs_common.testing.postgres_fixtures.uuid.uuid4",
        lambda: _FixedUUID(),
    )
    prefix = "test_pgv_predrop_"
    table = f"{prefix}{_FixedUUID.hex[:8]}"
    params = postgres_connection_params

    # Simulate the leaked table from a killed session.
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DROP TABLE IF EXISTS public.{safe_sql_ident(table)} CASCADE"
        )
        cursor.execute(
            f"CREATE TABLE public.{safe_sql_ident(table)} (id int)"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    assert _table_exists(params, "public", table)

    gen = make_pgvector_test_table(prefix, dimensions=768)
    config = next(gen)

    # Pre-drop must have removed the leaked table by the time the
    # fixture yields its config.
    assert config["table_name"] == table
    assert config["dimensions"] == 768
    assert not _table_exists(params, "public", table)

    # Give teardown something to drop, then exhaust the generator.
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE TABLE public.{safe_sql_ident(table)} (id int)"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    with pytest.raises(StopIteration):
        next(gen)
    assert not _table_exists(params, "public", table)


# -- Change B: gated orphan sweep ------------------------------------------


def _run_sweep(params: dict) -> None:
    # ``@pytest.fixture`` sets ``__wrapped__`` to the undecorated
    # function; calling it directly exercises the sweep body without
    # pytest's fixture injection.
    postgres_fixtures._sweep_orphan_test_tables.__wrapped__(params)  # type: ignore[attr-defined]


def test_sweep_orphan_noop_when_flag_unset(monkeypatch):
    """Flag unset → returns before any connection (bogus host is safe)."""
    monkeypatch.delenv("DK_SWEEP_ORPHAN_TEST_TABLES", raising=False)
    # If it tried to connect to this host it would error; it must not.
    _run_sweep({"database": "test_x", "host": "127.0.0.1", "port": 1})


@pytest.mark.parametrize(
    ("db", "allowed"),
    [
        ("dataknobs_test", True),
        ("test_dataknobs", True),
        ("test_records_ab12", True),
        ("something_test", True),
        ("dataknobs", False),
        ("prod", False),
        ("", False),
    ],
)
def test_sweep_orphan_allowlist(monkeypatch, caplog, db, allowed):
    """Fail-closed: non-test DB names are refused (warn, no connection)."""
    monkeypatch.setenv("DK_SWEEP_ORPHAN_TEST_TABLES", "true")
    # Unreachable host: an *allowed* db proceeds to connect and is
    # swallowed as "unreachable"; a *refused* db returns before connect.
    with caplog.at_level("WARNING"):
        _run_sweep(
            {
                "database": db,
                "host": "127.0.0.1",
                "port": 1,
                "user": "u",
                "password": "p",
            }
        )
    refused_phrase = "not on the test-DB allowlist"
    if allowed:
        assert "unreachable" in caplog.text
        assert refused_phrase not in caplog.text
    else:
        assert refused_phrase in caplog.text
        assert "unreachable" not in caplog.text


@requires_postgres
def test_sweep_orphan_drops_only_test_tables(
    postgres_connection_params, monkeypatch
):
    """Real sweep drops ``public.test_*`` and preserves non-test tables."""
    import uuid as _uuid

    import psycopg2

    params = postgres_connection_params
    suffix = _uuid.uuid4().hex[:8]
    orphan = f"test_sweep_orphan_{suffix}"
    keep = f"zz_sweep_keep_{suffix}"

    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        user=params["user"],
        password=params["password"],
        database=params["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE TABLE public.{safe_sql_ident(orphan)} (id int)"
        )
        cursor.execute(
            f"CREATE TABLE public.{safe_sql_ident(keep)} (id int)"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    monkeypatch.setenv("DK_SWEEP_ORPHAN_TEST_TABLES", "true")
    try:
        _run_sweep(params)
        assert not _table_exists(params, "public", orphan)
        assert _table_exists(params, "public", keep)
    finally:
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            database=params["database"],
        )
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"DROP TABLE IF EXISTS public.{safe_sql_ident(keep)} CASCADE"
            )
            conn.commit()
            cursor.close()
        finally:
            conn.close()


@requires_postgres
def test_make_postgres_test_db_cleanup_on_exception(make_postgres_test_db):
    """Factory-fixture teardown runs even when the test body raises.

    Drives a manual generator so we can simulate a test failure inside the
    yield block, then verify cleanup completes (table is dropped) without
    leaving the connection or transaction dangling.
    """
    import psycopg2

    gen = make_postgres_test_db("test_smoke_cleanup_")
    config = next(gen)

    table = config["table"]
    schema = config["schema"]

    # Create the table so cleanup has something to drop.
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE TABLE {safe_sql_ident(schema)}."
            f"{safe_sql_ident(table)} (id INT)"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    # Simulate a failing test body — drive the generator with throw().
    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("simulated test failure"))

    # Verify the table is gone.
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT to_regclass(%s)",
            (f"{schema}.{table}",),
        )
        result = cursor.fetchone()
        assert result is not None and result[0] is None, (
            f"Cleanup did not drop {schema}.{table}"
        )
        cursor.close()
    finally:
        conn.close()
