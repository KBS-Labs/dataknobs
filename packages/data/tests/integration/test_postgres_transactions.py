"""Integration tests for the ``AsyncDatabase`` buffered transaction on Postgres.

Postgres is a second real transactional backend (alongside the in-process
SQLite coverage in ``tests/test_transactions.py``): proves the buffered
transaction's commit flush actually persists through ``asyncpg`` and that
``supports_transactions()`` is truthful for a pooled, server-backed backend.

Requires a running PostgreSQL instance. Fixtures are provided by the
``dataknobs_common_postgres`` pytest11 plugin (``postgres_test_db`` from the
local conftest wraps ``make_postgres_test_db``).
"""

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

pytestmark = requires_postgres


@pytest.mark.asyncio
async def test_postgres_supports_transactions(postgres_test_db):
    db = AsyncPostgresDatabase(postgres_test_db)
    try:
        await db.connect()
        assert db.supports_transactions() is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_transaction_commit_persists(postgres_test_db):
    db = AsyncPostgresDatabase(postgres_test_db)
    try:
        await db.connect()
        async with db.transaction() as tx:  # default policy="strict"
            await tx.create(Record({"name": "a"}))
            await tx.create(Record({"name": "b"}))
        assert await db.count() == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_all_upsert_buffer_commits_atomically(postgres_test_db):
    """A single-kind all-upsert buffer coalesces into one atomic
    ``upsert_batch`` on postgres — ``is_atomic`` is True and a re-commit of the
    same ids overwrites idempotently rather than duplicating.
    """
    db = AsyncPostgresDatabase(postgres_test_db)
    try:
        await db.connect()
        async with db.transaction() as tx:  # strict; postgres is transactional
            await tx.upsert_batch(
                [Record({"id": "1", "v": "a"}), Record({"id": "2", "v": "b"})]
            )
            assert tx.is_atomic is True
        assert await db.count() == 2
        async with db.transaction() as tx:
            await tx.upsert_batch(
                [Record({"id": "1", "v": "A"}), Record({"id": "2", "v": "B"})]
            )
        assert await db.count() == 2
        rec = await db.read("1")
        assert rec is not None and rec.get_value("v") == "A"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_transaction_rollback_persists_nothing(postgres_test_db):
    db = AsyncPostgresDatabase(postgres_test_db)
    try:
        await db.connect()
        with pytest.raises(ValueError, match="nope"):
            async with db.transaction() as tx:
                await tx.create(Record({"name": "a"}))
                raise ValueError("nope")
        assert await db.count() == 0
    finally:
        await db.close()


# ---- cross-kind atomic commit on the pooled connection model ------------
#
# Postgres is the pooled backend: a multi-kind flush must pin ONE connection
# and route every batch through it inside one native transaction. These run on
# a size-1 pool so a stray second acquire would deadlock — the load-bearing
# proof that the batch methods use the threaded handle rather than re-acquiring.


class _PgMidFlushDeleteFailure(AsyncPostgresDatabase):
    """Real postgres backend whose ``delete_batch`` raises mid-flush.

    Proves a multi-kind commit rolls the whole flush back through the pinned
    pooled connection — no partial persistence, no size-1-pool deadlock.
    """

    async def delete_batch(self, ids, *, _tx=None):  # type: ignore[override]
        raise RuntimeError("delete batch failed mid-flush")


@pytest.mark.asyncio
async def test_postgres_multi_kind_commits_atomically_on_size1_pool(postgres_test_db):
    """A multi-kind buffer commits all-or-nothing inside one native transaction.

    On a size-1 pool: the pinned connection is reused for every coalesced batch
    (no second ``pool.acquire()`` that would deadlock), so the whole commit is
    atomic and ``is_atomic`` is True.
    """
    db = AsyncPostgresDatabase(
        {**postgres_test_db, "min_pool_size": 1, "max_pool_size": 1}
    )
    try:
        await db.connect()
        seed_id = await db.create(Record({"v": "s"}))
        async with db.transaction() as tx:  # strict; postgres transactional
            await tx.create(Record({"id": "c1", "v": "a"}))
            await tx.delete(seed_id)
            await tx.upsert("u1", Record({"v": "c"}))
            assert tx.is_atomic is True  # multi-kind, one spanning native txn
        assert await db.count() == 2  # c1 + u1; seed removed
        assert await db.read(seed_id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_postgres_multi_kind_rolls_back_whole_flush_on_size1_pool(postgres_test_db):
    db = _PgMidFlushDeleteFailure(
        {**postgres_test_db, "min_pool_size": 1, "max_pool_size": 1}
    )
    try:
        await db.connect()
        tx = await db.begin_transaction()  # strict; postgres transactional
        await tx.create(Record({"id": "early", "v": "a"}))
        await tx.delete("whatever")
        assert tx.is_atomic is True
        with pytest.raises(RuntimeError, match="mid-flush"):
            await tx.commit()
        # The earlier create, run inside the pinned transaction, is rolled back.
        assert await db.count() == 0
    finally:
        await db.close()
