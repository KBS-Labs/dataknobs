"""``PgVectorStore`` connection-pool ownership across ``close()``.

A consumer that manages one asyncpg pool and shares it across several
stores builds each via ``PgVectorStore.from_components(pool=shared)``. The
store must treat that pool as caller-owned: ``initialize()`` runs the
schema/table setup against it but does not create a new pool, and
``close()`` leaves it open **and retains the reference** so the store can
be re-initialized (reopened) against the shared pool. The config /
connection-string path is unchanged — it builds its own pool, owns it, and
closes + drops it on ``close()``.

Reproduce-first: inject an external pool, close the store, and assert the
pool is still open and directly usable. Before the ownership gate,
``close()`` tore down the injected pool (``is_closing()`` → ``True``) and a
sibling store sharing it lost its connection — so these assertions fail.
The self-owned regression test pins that a config-built store still closes
its own pool; the retain/reopen tests pin that an injected store keeps its
pool across ``close()`` and reuses it on the next ``initialize()``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from dataknobs_common.testing import (
    requires_real_postgres,
    safe_sql_ident,
)

pytest.importorskip("asyncpg")

import asyncpg

from dataknobs_data.vector.stores.pgvector import PgVectorStore


def _get_test_connection_string() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="session")
def _ensure_pgvector_extension() -> None:
    # asyncpg availability is guaranteed by the module-level importorskip;
    # this fixture only runs for the @requires_real_postgres tests.
    async def _setup() -> None:
        conn = await asyncpg.connect(_get_test_connection_string())
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

    # Best-effort: this is a sync session-scoped fixture (no event loop is
    # running at session-setup time, so asyncio.run is safe here). A
    # connect/extension failure is swallowed deliberately — the dependent
    # tests are gated by @requires_real_postgres and a per-test
    # asyncpg.connect would surface a clearer error than an errored
    # session-scoped fixture, which would error (not skip) every test.
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


@requires_real_postgres
@pytest.mark.asyncio
async def test_injected_pool_survives_store_close(
    pg_table_name: str,
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

        # The store is logically closed (init state reset) but RETAINS its
        # injected pool reference so it can be reopened — the pool is
        # caller-owned and still live.
        assert store._pool is external
        assert store._initialized is False
        # ...and the externally owned pool is untouched and usable.
        assert external.is_closing() is False
        async with external.acquire() as conn:
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await external.close()


@requires_real_postgres
@pytest.mark.asyncio
async def test_self_owned_pool_is_closed(
    pg_table_name: str,
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


@requires_real_postgres
@pytest.mark.asyncio
async def test_two_stores_share_one_pool(
    pg_table_name: str,
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


@pytest.mark.asyncio
async def test_injected_pool_reference_retained_across_close() -> None:
    """``close()`` on an injected store retains the pool and never closes it.

    Pins the ownership contract at the unit level, no live server needed: a
    plain sentinel stands in for the injected pool. ``close()`` must not
    call ``.close()`` on it (the sentinel has no such method — an erroneous
    cascade would raise ``AttributeError``) and must keep the reference so
    the store can later reopen against the still-live, caller-owned pool.
    ``_owns_pool`` is immutable; only ``_initialized`` flips.
    """
    sentinel_pool = object()
    store = PgVectorStore.from_components(
        _store_config("noserver_retain", dimensions=8), pool=sentinel_pool
    )
    assert store._owns_pool is False
    assert store._pool is sentinel_pool

    await store.close()

    # The injected pool was neither closed nor dropped; only the logical
    # init state was reset. Ownership is unchanged.
    assert store._pool is sentinel_pool
    assert store._owns_pool is False
    assert store._initialized is False


@requires_real_postgres
@pytest.mark.asyncio
async def test_injected_store_reopens_after_close(
    pg_table_name: str,
) -> None:
    """An injected store can be re-initialized to reopen against its pool.

    Design contract: ``close()`` leaves an injected pool live and retains
    the reference, so ``initialize()`` after ``close()`` reuses the same
    pool (it never fabricates one) and the store is fully usable again.
    """
    external = await asyncpg.create_pool(
        _get_test_connection_string(), min_size=1, max_size=2
    )
    try:
        store = PgVectorStore.from_components(
            _store_config(pg_table_name), pool=external
        )
        await store.initialize()
        assert await store.count() == 0

        await store.close()
        # Reopen: same pool, no new pool created, store usable again.
        await store.initialize()
        assert store._pool is external
        assert store._owns_pool is False
        assert external.is_closing() is False
        assert await store.count() == 0
    finally:
        await external.close()
