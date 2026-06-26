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
