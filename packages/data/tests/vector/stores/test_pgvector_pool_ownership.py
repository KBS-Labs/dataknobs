"""``PgVectorStore`` connection-pool ownership across ``close()``.

A consumer that manages one asyncpg pool and shares it across several
stores builds each via ``PgVectorStore.from_components(pool=shared)``. The
store must treat that pool as caller-owned: ``initialize()`` runs the
schema/table setup against it but does not create a new pool, and
``close()`` leaves it open. The config / connection-string path is
unchanged — it builds its own pool and owns it.

Reproduce-first: inject an external pool, close the store, and assert the
pool is still open and directly usable. Before the ownership gate,
``close()`` tore down the injected pool (``is_closing()`` → ``True``) and a
sibling store sharing it lost its connection — so these assertions fail.
The self-owned regression test pins that a config-built store still closes
its own pool.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

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
    """A unique table name, dropped before and after the test."""
    table = f"test_pgv_poolown_{uuid.uuid4().hex[:8]}"

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


def _store_config(table: str, dimensions: int = 384) -> dict[str, Any]:
    return {
        "connection_string": _get_test_connection_string(),
        "dimensions": dimensions,
        "metric": "cosine",
        "schema": "public",
        "table_name": table,
        "auto_create_table": True,
        "id_type": "text",
    }


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_injected_pool_survives_store_close(
    _m: None, pg_table_name: str
) -> None:
    """An injected pool is left open by ``close()`` and stays usable.

    Reproduce-first: pre-gate, ``close()`` closes the injected pool, so
    ``external.is_closing()`` is ``True`` and the direct query fails.
    """
    external = await asyncpg.create_pool(
        _get_test_connection_string(), min_size=1, max_size=2
    )
    try:
        store = PgVectorStore.from_components(
            _store_config(pg_table_name), pool=external
        )
        await store.initialize()
        # Sanity: the store works against the injected pool.
        assert await store.count() == 0

        await store.close()

        # The store dropped its handle and reset init state...
        assert store._pool is None
        assert store._initialized is False
        # ...but the externally owned pool is untouched and usable.
        assert external.is_closing() is False
        async with external.acquire() as conn:
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await external.close()


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_self_owned_pool_is_closed(
    _m: None, pg_table_name: str
) -> None:
    """A config-built store owns its pool and closes it (regression)."""
    store = PgVectorStore(_store_config(pg_table_name))
    await store.initialize()
    assert store._owns_pool is True
    pool = store._pool
    assert pool is not None

    await store.close()

    assert store._pool is None
    assert store._initialized is False
    assert pool.is_closing() is True


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_pgvector_marks)])
@pytest.mark.asyncio
async def test_two_stores_share_one_pool(
    _m: None, pg_table_name: str
) -> None:
    """Closing one store over a shared pool leaves the second working.

    The canonical consumer pattern that triggered the gap: one pool, many
    stores. Closing the first store must not tear down the pool the second
    still depends on.
    """
    external = await asyncpg.create_pool(
        _get_test_connection_string(), min_size=1, max_size=3
    )
    try:
        store_a = PgVectorStore.from_components(
            _store_config(pg_table_name), pool=external
        )
        store_b = PgVectorStore.from_components(
            _store_config(pg_table_name), pool=external
        )
        await store_a.initialize()
        await store_b.initialize()

        await store_a.close()

        # store_b's shared pool is still live and queryable.
        assert external.is_closing() is False
        assert await store_b.count() == 0
    finally:
        await external.close()
