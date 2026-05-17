"""``PgVectorStore`` init-time vector-dimension mismatch guard (Item 129 C).

``_create_table`` uses ``CREATE TABLE IF NOT EXISTS ... embedding
vector({dimensions})``. When a same-named table already exists at a
*different* dimensionality, creation is a silent no-op and the mismatch
only surfaces much later as an opaque ``asyncpg.DataError`` at the first
insert (``expected <stale> dimensions, not <configured>``) — a
production footgun on any embedding-model dimension swap.

Reproduce-first: a real prior store creates the table at ``vector(384)``;
a second store configured for ``vector(768)`` against the same table must
fail loudly at ``initialize()`` with a ``ConfigurationError`` naming both
dimensions, instead of deferring to the insert-time ``DataError``. The
guard is read-only (reads ``pg_attribute.atttypmod``; no DDL).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.testing import requires_postgres, safe_sql_ident

try:
    import asyncpg

    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


_pgvector_marks = [
    requires_postgres,
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not ASYNCPG_AVAILABLE,
        reason="pgvector tests require TEST_POSTGRES=true and asyncpg",
    ),
]


def _get_test_connection_string() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="session")
def _ensure_pgvector_extension() -> None:
    if not ASYNCPG_AVAILABLE:
        return

    async def _setup() -> None:
        conn = await asyncpg.connect(_get_test_connection_string())
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

    try:
        asyncio.run(_setup())
    except (OSError, asyncpg.PostgresError):
        pass


@pytest_asyncio.fixture
async def pg_table_name(
    _ensure_pgvector_extension: None,
) -> AsyncIterator[str]:
    """A unique table name, dropped before and after the test.

    This file deliberately exercises the *stale same-named table* path,
    so it cannot use the pre-dropping shared fixture for the stale-table
    arrangement — it creates the prior-dimension table on purpose.
    """
    table = f"test_pgv_dimguard_{uuid.uuid4().hex[:8]}"

    async def _drop() -> None:
        conn = await asyncpg.connect(_get_test_connection_string())
        try:
            await conn.execute(
                f"DROP TABLE IF EXISTS public.{safe_sql_ident(table)} CASCADE"
            )
        finally:
            await conn.close()

    await _drop()
    try:
        yield table
    finally:
        await _drop()


def _store_config(table: str, dimensions: int) -> dict[str, Any]:
    return {
        "connection_string": _get_test_connection_string(),
        "dimensions": dimensions,
        "metric": "cosine",
        "schema": "public",
        "table_name": table,
        "auto_create_table": True,
        "id_type": "text",
    }


async def _create_prior_table(table: str, dimensions: int) -> None:
    """Materialize a real prior ``PgVectorStore`` table at ``dimensions``."""
    store = PgVectorStore(_store_config(table, dimensions))
    await store.initialize()
    await store.close()


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_pgvector_dimension_mismatch_raises_configuration_error(
    _m: None, pg_table_name: str
) -> None:
    """Reproduce-first: stale vector(384), store wants vector(768).

    Pre-guard this does NOT raise at ``initialize()`` (the silent
    ``IF NOT EXISTS`` no-op defers the failure to an opaque
    ``asyncpg.DataError`` at first insert) — the test fails with
    ``DID NOT RAISE``. Post-guard ``initialize()`` raises a
    ``ConfigurationError`` naming both ``384`` and ``768``.
    """
    await _create_prior_table(pg_table_name, dimensions=384)

    store = PgVectorStore(_store_config(pg_table_name, dimensions=768))
    with pytest.raises(ConfigurationError) as excinfo:
        await store.initialize()

    msg = str(excinfo.value)
    assert "384" in msg, msg
    assert "768" in msg, msg


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_pgvector_matching_dimension_initializes_cleanly(
    _m: None, pg_table_name: str
) -> None:
    """Positive control: a matching dimension must NOT raise."""
    await _create_prior_table(pg_table_name, dimensions=384)

    store = PgVectorStore(_store_config(pg_table_name, dimensions=384))
    try:
        await store.initialize()  # must not raise
    finally:
        await store.close()


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_pgvector_atttypmod_decodes_to_vector_dimension(
    _m: None, pg_table_name: str
) -> None:
    """Pin the ``atttypmod`` -> dimension decode empirically.

    pgvector stores ``vector(N)`` dimensionality in ``atttypmod``
    directly (no VARCHAR-style ``-4`` adjustment). This pins that
    decode against the running pgvector so the guard's comparison is
    not asserted on faith.
    """
    conn = await asyncpg.connect(_get_test_connection_string())
    try:
        await conn.execute(
            f"CREATE TABLE public.{safe_sql_ident(pg_table_name)} "
            f"(id text PRIMARY KEY, embedding vector(384))"
        )
        atttypmod = await conn.fetchval(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 AND c.relname = $2
              AND a.attname = $3 AND NOT a.attisdropped
            """,
            "public",
            pg_table_name,
            "embedding",
        )
    finally:
        await conn.close()

    assert atttypmod == 384
