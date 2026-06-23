"""Integration tests for shared connection-pool ownership (real Postgres).

Two ``AsyncPostgresDatabase`` instances on the same DSN share one pooled
asyncpg pool (``ConnectionPoolManager`` keys on host/port/database/user,
not table). These tests pin the ownership contract end-to-end: a sibling's
``close()`` is a *release*, not a teardown, so the shared pool survives
until the last holder releases.

On ``main`` (pre-fix) test 7 reproduces ``asyncpg.InterfaceError: pool is
closed`` — a sibling close hard-closed the pool out from under the other
holder. After the fix it passes.
"""

import os

import pytest

from dataknobs_data import AsyncDatabase, Record
from dataknobs_data.backends.postgres import _pool_manager

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance",
)


def _sibling_config(base_config: dict, table: str) -> dict:
    """Copy a postgres config dict, overriding only the table name.

    The two configs share host/port/database/user, so they collapse to
    one pool in the manager — exactly the production shape the contract
    must protect.
    """
    sibling = dict(base_config)
    sibling["table"] = table
    return sibling


class TestPostgresPoolSharing:
    """Shared-pool ownership across sibling AsyncPostgresDatabase holders."""

    @pytest.mark.asyncio
    async def test_sibling_close_does_not_kill_shared_pool(self, postgres_test_db):
        """A sibling's close() must not close the pool the other holder uses.

        Reproduces the incident: on ``main`` ``b.close()`` hard-closes the
        shared pool, so the next ``a.read(...)`` raises
        ``asyncpg.InterfaceError: pool is closed``.
        """
        config_a = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_a")
        config_b = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_b")

        # Counts are relative to a baseline: _pool_manager is a module
        # global shared across tests.
        baseline = _pool_manager.get_pool_count()

        a = await AsyncDatabase.from_backend("postgres", config_a)
        b = await AsyncDatabase.from_backend("postgres", config_b)
        try:
            # Same DSN -> one shared pool, two holders.
            assert a._pool is b._pool
            assert _pool_manager.get_pool_count() == baseline + 1

            rec_id = await a.create(Record({"v": "before"}))
            assert await a.read(rec_id) is not None

            # Sibling release must leave the shared pool alive.
            await b.close()

            # The exact operation that raised InterfaceError on main.
            assert await a.read(rec_id) is not None
            await a.delete(rec_id)
        finally:
            await a.close()
            await b.close()

    @pytest.mark.asyncio
    async def test_last_holder_close_tears_down_pool(self, postgres_test_db):
        """Closing every holder evicts the pool; a fresh connect rebuilds it."""
        config_a = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_a")
        config_b = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_b")

        baseline = _pool_manager.get_pool_count()

        a = await AsyncDatabase.from_backend("postgres", config_a)
        b = await AsyncDatabase.from_backend("postgres", config_b)
        shared_pool = a._pool
        assert _pool_manager.get_pool_count() == baseline + 1

        await a.close()
        # One holder remains -> pool still tracked and alive.
        assert _pool_manager.get_pool_count() == baseline + 1

        await b.close()
        # Last holder released -> pool evicted.
        assert _pool_manager.get_pool_count() == baseline

        # A fresh connect builds a *new* pool object, not the closed one.
        c = await AsyncDatabase.from_backend("postgres", config_a)
        try:
            assert c._pool is not shared_pool
            assert _pool_manager.get_pool_count() == baseline + 1
        finally:
            await c.close()

    @pytest.mark.asyncio
    async def test_connected_singleton_survives_sibling_lifecycle(self, postgres_test_db):
        """A long-lived holder keeps reading across a sibling's full lifecycle.

        Models the registry singleton: holder A connects once and stays
        connected while holder B runs a full connect -> read -> close
        cycle. A must read throughout — the production shape of the
        cascade the fix prevents.
        """
        config_a = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_a")
        config_b = _sibling_config(postgres_test_db, postgres_test_db["table"] + "_b")

        a = await AsyncDatabase.from_backend("postgres", config_a)
        try:
            rec_id = await a.create(Record({"v": "singleton"}))
            assert await a.read(rec_id) is not None

            # Full sibling lifecycle.
            b = await AsyncDatabase.from_backend("postgres", config_b)
            b_id = await b.create(Record({"v": "sibling"}))
            assert await b.read(b_id) is not None
            await b.delete(b_id)
            await b.close()

            # A still reads after the sibling tore down its claim.
            assert await a.read(rec_id) is not None
            await a.delete(rec_id)
        finally:
            await a.close()
