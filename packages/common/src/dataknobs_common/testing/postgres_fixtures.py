"""Shared Postgres pytest fixtures for dataknobs integration tests.

This module is a pytest11 plugin (registered in ``packages/common/pyproject.toml``)
so any package depending on ``dataknobs-common`` automatically gets these
fixtures via pytest's plugin discovery — no explicit ``conftest.py`` imports
required.

Consumers wrap :func:`make_postgres_test_db` with a thin per-prefix fixture
to get a clean per-test table:

    @pytest.fixture
    def postgres_test_db(make_postgres_test_db):
        yield from make_postgres_test_db("test_conversations_")

The factory pattern keeps the table prefix consumer-controlled without
forcing each consumer to re-declare ``@pytest.fixture(params=[...])`` indirect
parameterization.

Environment variables, resolved by :func:`postgres_env_params` (which
the ``postgres_connection_params`` fixture wraps, and which non-fixture
callers should use directly rather than re-reading these by hand):

- ``POSTGRES_HOST`` (default: ``postgres`` in Docker, ``localhost`` otherwise)
- ``POSTGRES_PORT`` (default: ``5432``)
- ``POSTGRES_USER`` (default: ``postgres``)
- ``POSTGRES_PASSWORD`` (default: ``postgres``)
- ``POSTGRES_DB`` (default: ``dataknobs_test``)
- ``DOCKER_CONTAINER`` (``true``/``1``/``yes``/``on`` forces the
  ``postgres`` host default; anything else, including ``false``, does not)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from dataknobs_common.postgres_config import build_postgres_dsn
from dataknobs_common.testing._core import _docker_aware_default_host, safe_sql_ident

logger = logging.getLogger(__name__)


def postgres_env_params() -> dict[str, Any]:
    """Resolve Postgres connection parameters from the environment.

    The single definition of what "the test Postgres" means: which
    host, port, user, password and database an integration test reaches
    when the environment does not say otherwise. The
    :func:`postgres_connection_params` fixture is a thin wrapper over
    this, so a caller that cannot take a fixture — a module-level
    constant, a plain helper function — resolves identically instead of
    restating the defaults.

    Restating them is not hypothetical. Four sites that read these env
    vars by hand had drifted on three of the five fields: they defaulted
    ``POSTGRES_DB`` to a name no runner creates, left ``port`` a string,
    and omitted the container check below entirely — so under Docker
    they resolved to ``localhost`` while every fixture-based test
    resolved to the compose service.

    Detects whether the test process is running inside a Docker
    container (see :func:`_in_docker_container`) and defaults the host to
    ``postgres`` (the typical compose service name) in that case,
    ``localhost`` otherwise.

    Returns:
        A fresh dict with ``host``, ``port`` (``int``), ``user``,
        ``password`` and ``database``. Suitable for
        :func:`postgres_dsn`, or for splatting into a connect call.
    """
    default_host = _docker_aware_default_host("postgres")

    return {
        "host": os.environ.get("POSTGRES_HOST", default_host),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "database": os.environ.get("POSTGRES_DB", "dataknobs_test"),
    }


def postgres_dsn(params: Mapping[str, Any]) -> str:
    """Build a libpq URI from a ``postgres_connection_params`` mapping.

    The counterpart to the :func:`postgres_connection_params` fixture:
    tests that need a connection *string* rather than keyword arguments
    pass the fixture's dict straight through.

    Only the five connection keys are read, so a mapping that has been
    copied and extended — as the table fixtures do, adding ``table`` and
    ``schema`` — can be passed without stripping it first.

    Encoding and validation come from
    :func:`~dataknobs_common.postgres_config.build_postgres_dsn`, which
    is also what the production normalizer uses. That shared definition
    is the point: a hand-rolled f-string in a test silently accepts a
    password containing ``@`` and produces a URI aimed at another host,
    so a test written that way passes or fails for reasons unrelated to
    its subject.

    Args:
        params: Mapping carrying ``host``, ``port``, ``database``,
            ``user`` and ``password``. Extra keys are ignored.

    Returns:
        A ``postgresql://`` URI.

    Raises:
        KeyError: If any of the five connection keys is absent.
        ValueError: If ``host`` or ``database`` would produce a
            malformed or misrouted URI.
    """
    return build_postgres_dsn(
        host=params["host"],
        port=params["port"],
        database=params["database"],
        user=params["user"],
        password=params["password"],
    )


def wait_for_postgres(
    host: str,
    port: int,
    user: str,
    password: str,
    max_retries: int = 30,
) -> bool:
    """Wait for PostgreSQL to accept connections on the maintenance database.

    Args:
        host: PostgreSQL host.
        port: PostgreSQL port.
        user: PostgreSQL user.
        password: PostgreSQL password.
        max_retries: Maximum number of attempts (1-second sleep between).

    Returns:
        True once a connection succeeds.

    Raises:
        psycopg2.OperationalError: If all retries exhaust without a successful
            connection.
    """
    import psycopg2

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres",
            )
            conn.close()
            return True
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(1)
    return False


try:
    import pytest

    @pytest.fixture(scope="session")
    def postgres_connection_params() -> dict[str, Any]:
        """PostgreSQL connection parameters for integration tests.

        The fixture form of :func:`postgres_env_params`, which holds the
        resolution rules and their defaults.
        """
        return postgres_env_params()

    @pytest.fixture(scope="session")
    def ensure_postgres_ready(
        postgres_connection_params: dict[str, Any],
    ) -> None:
        """Ensure PostgreSQL is reachable and the test database exists.

        Waits for the server to accept connections on the maintenance
        ``postgres`` database, then creates the configured test database
        (``POSTGRES_DB``) if it does not already exist.
        """
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        wait_for_postgres(
            host=postgres_connection_params["host"],
            port=postgres_connection_params["port"],
            user=postgres_connection_params["user"],
            password=postgres_connection_params["password"],
        )

        conn = psycopg2.connect(
            host=postgres_connection_params["host"],
            port=postgres_connection_params["port"],
            user=postgres_connection_params["user"],
            password=postgres_connection_params["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (postgres_connection_params["database"],),
            )
            if not cursor.fetchone():
                cursor.execute(
                    f"CREATE DATABASE {safe_sql_ident(postgres_connection_params['database'])}"
                )
        except psycopg2.errors.DuplicateDatabase:
            pass
        finally:
            cursor.close()
            conn.close()

    @pytest.fixture
    def make_postgres_test_db(
        ensure_postgres_ready: None,
        postgres_connection_params: dict[str, Any],
    ) -> Callable[[str], Iterator[dict[str, Any]]]:
        """Factory fixture for per-test Postgres tables.

        Returns a callable ``factory(table_prefix)`` that yields a connection-
        config dict including a unique ``table`` name and drops that table on
        teardown. Consumer fixtures use ``yield from`` to thread the cleanup
        through:

            @pytest.fixture
            def postgres_test_db(make_postgres_test_db):
                yield from make_postgres_test_db("test_conversations_")

        The yielded config dict has the same shape as
        ``postgres_connection_params`` plus:

        - ``table``: ``f"{table_prefix}{uuid8}"``
        - ``schema``: ``"public"``

        Args:
            ensure_postgres_ready: Session fixture ensuring the test DB exists.
            postgres_connection_params: Session-scoped connection params.

        Returns:
            A callable that, given a ``table_prefix``, yields a config dict
            and tears down the table on completion.
        """

        def factory(table_prefix: str) -> Iterator[dict[str, Any]]:
            import psycopg2

            test_id = uuid.uuid4().hex[:8]
            config = postgres_connection_params.copy()
            config["table"] = f"{table_prefix}{test_id}"
            config["schema"] = "public"

            try:
                yield config
            finally:
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
                        f"DROP TABLE IF EXISTS "
                        f"{safe_sql_ident(config['schema'])}."
                        f"{safe_sql_ident(config['table'])} CASCADE"
                    )
                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()

        return factory

    @pytest.fixture
    def make_pgvector_test_table(
        ensure_postgres_ready: None,
        postgres_connection_params: dict[str, Any],
    ) -> Callable[..., Iterator[dict[str, Any]]]:
        """Factory: per-test pgvector table, drop-BEFORE-create + teardown.

        Mirrors :func:`make_postgres_test_db` (sync psycopg2 generator,
        consumed via ``yield from``) but yields a ``PgVectorStore`` config
        dict. The **pre-drop** is the load-bearing change: it defeats the
        ``CREATE TABLE IF NOT EXISTS`` dimension shadow that a killed
        prior session can leave behind (the same root cause the
        production ``PgVectorStore`` init-time dimension guard
        addresses), which a post-only-teardown DROP can never
        guarantee under killed sessions. The teardown DROP is retained
        best-effort, matching ``make_postgres_test_db``.

            @pytest.fixture
            def pgvector_config(make_pgvector_test_table):
                yield from make_pgvector_test_table(
                    "test_tombstone_", dimensions=768)

        Args:
            ensure_postgres_ready: Session fixture ensuring the test DB
                exists.
            postgres_connection_params: Session-scoped connection params.

        Returns:
            A callable ``factory(prefix, *, dimensions, schema="public")``
            that yields a ``PgVectorStore`` config dict and drops the
            table before yielding and again on teardown.
        """

        def factory(
            prefix: str, *, dimensions: int, schema: str = "public"
        ) -> Iterator[dict[str, Any]]:
            import psycopg2

            table = f"{prefix}{uuid.uuid4().hex[:8]}"
            params = postgres_connection_params

            def _drop() -> None:
                conn = psycopg2.connect(
                    host=params["host"],
                    port=params["port"],
                    user=params["user"],
                    password=params["password"],
                    database=params["database"],
                )
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            f"DROP TABLE IF EXISTS "
                            f"{safe_sql_ident(schema)}."
                            f"{safe_sql_ident(table)} CASCADE"
                        )
                    conn.commit()
                finally:
                    conn.close()

            def _ensure_vector_extension() -> None:
                # The pgvector extension is a pgvector-specific
                # prerequisite, so ensuring it is this fixture's
                # responsibility — consumers no longer need a local
                # ``_ensure_pgvector_extension`` helper.
                conn = psycopg2.connect(
                    host=params["host"],
                    port=params["port"],
                    user=params["user"],
                    password=params["password"],
                    database=params["database"],
                )
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                finally:
                    conn.close()

            _ensure_vector_extension()
            _drop()  # pre-drop: defeat the IF-NOT-EXISTS dimension shadow
            config = {
                "backend": "pgvector",
                "connection_string": postgres_dsn(params),
                "dimensions": dimensions,
                "schema": schema,
                "table_name": table,
                "auto_create_table": True,
                "id_type": "text",
            }
            try:
                yield config
            finally:
                _drop()  # best-effort teardown

        return factory

    @pytest.fixture(scope="session", autouse=True)
    def _sweep_orphan_test_tables(
        postgres_connection_params: dict[str, Any],
    ) -> None:
        """Belt-and-suspenders sweep of *other* sessions' leaked tables.

        Killed test sessions leak ``public.test_*`` tables (post-only
        teardown can't drop them). This one-time session sweep removes
        them, but is **fail-closed and opt-in**:

        - no-op unless ``DK_SWEEP_ORPHAN_TEST_TABLES=true``;
        - refuses (warns, drops nothing) unless the connected DB name is
          on a test-DB allowlist — a ``test_`` prefix, a ``_test``
          suffix, or one of the known test DB names. ``dataknobs_test``
          (this repo's verified ``POSTGRES_DB`` default) matches via the
          ``_test`` suffix; a production-looking name (``dataknobs``,
          ``prod``) does not and is refused;
        - drops only ``public.test_*``; never touches non-test tables;
        - no-op (swallowed) when Postgres is unreachable.

        Args:
            postgres_connection_params: Session-scoped connection params
                (read-only; this fixture deliberately does NOT depend on
                ``ensure_postgres_ready`` so it cannot force DB creation).
        """
        if os.environ.get("DK_SWEEP_ORPHAN_TEST_TABLES", "").lower() != "true":
            return

        db = str(postgres_connection_params.get("database", ""))
        allowed = (
            db in {"test_dataknobs", "dataknobs_test"}
            or db.startswith("test_")
            or db.endswith("_test")
        )
        if not allowed:
            logger.warning(
                "Orphan test-table sweep refused: database %r is not on "
                "the test-DB allowlist (test_ prefix / _test suffix). "
                "Dropping nothing.",
                db,
            )
            return

        import psycopg2

        try:
            conn = psycopg2.connect(
                host=postgres_connection_params["host"],
                port=postgres_connection_params["port"],
                user=postgres_connection_params["user"],
                password=postgres_connection_params["password"],
                database=db,
            )
        except (OSError, psycopg2.OperationalError) as exc:
            logger.warning(
                "Orphan test-table sweep skipped: Postgres unreachable (%s)",
                exc,
            )
            return

        # Autocommit + per-table DROP: a leaked backlog can be hundreds
        # of tables (the very condition this sweep exists to clear);
        # batching them into one transaction exhausts
        # max_locks_per_transaction ("out of shared memory"). Each DROP
        # commits and releases its locks immediately, and a single bad
        # table cannot abort the rest.
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    r"""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename LIKE 'test\_%'
                    """
                )
                orphans = [row[0] for row in cursor.fetchall()]
                dropped = 0
                for tbl in orphans:
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS public.{safe_sql_ident(tbl)} CASCADE")
                        dropped += 1
                    except psycopg2.Error as exc:
                        logger.warning(
                            "Orphan test-table sweep could not drop %r: %s",
                            tbl,
                            exc,
                        )
                if dropped:
                    logger.info(
                        "Orphan test-table sweep dropped %d/%d public.test_* table(s) in %r",
                        dropped,
                        len(orphans),
                        db,
                    )
        except psycopg2.Error as exc:
            logger.warning("Orphan test-table sweep error: %s", exc)
        finally:
            conn.close()

except ImportError:
    # pytest not installed — fixture decorators unavailable.
    # The wait_for_postgres helper above remains usable.
    postgres_connection_params = None  # type: ignore[assignment]
    ensure_postgres_ready = None  # type: ignore[assignment]
    make_postgres_test_db = None  # type: ignore[assignment]
    make_pgvector_test_table = None  # type: ignore[assignment]
    _sweep_orphan_test_tables = None  # type: ignore[assignment]
