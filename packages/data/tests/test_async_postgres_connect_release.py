"""Unit test for AsyncPostgresDatabase connect() holder-release on failure.

Does NOT require a running PostgreSQL instance: the asyncpg pool creation
is stubbed (the external dependency we cannot run without a server) so the
real ``_pool_manager`` refcount path runs, and table setup is forced to
fail to exercise the partial-connect cleanup.
"""

from unittest.mock import AsyncMock, patch

import pytest

from dataknobs_data.backends.postgres import AsyncPostgresDatabase, _pool_manager
from dataknobs_data.query import Filter, Operator, Query


@pytest.mark.asyncio
async def test_connect_releases_holder_when_table_setup_fails():
    """A connect() that fails after acquiring the shared pool must release it.

    Reproduce-first for the partial-connect refcount leak (same defect
    class as the Elasticsearch path): connect() increments the manager
    holder count via get_pool, then runs _ensure_table(). If table setup
    raises, _connected is never set; connect() must release the holder
    before propagating so the manager's count returns to baseline rather
    than pinning the shared pool for the life of the process.
    """
    await _pool_manager.close_all()
    baseline = _pool_manager.get_pool_count()

    db = AsyncPostgresDatabase(
        {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "table": "test_table",
        }
    )

    fake_pool = AsyncMock()  # stands in for the asyncpg pool (no server)

    async def fake_create_pool(cfg):
        return fake_pool

    with (
        patch(
            "dataknobs_data.backends.postgres.create_asyncpg_pool",
            new=fake_create_pool,
        ),
        patch.object(db, "_ensure_table", side_effect=RuntimeError("table setup failed")),
    ):
        with pytest.raises(RuntimeError, match="table setup failed"):
            await db.connect()

    # The failed connect must not leak a holder slot.
    assert db._connected is False
    assert db._pool is None
    assert _pool_manager.get_pool_count() == baseline


@pytest.mark.asyncio
async def test_no_reader_reaches_the_query_builder_before_connect_binds_it():
    """What replaced the guard `search` used to carry, and `stream_read` never did.

    The asymmetry was the finding: one reader re-created the builder under a
    `hasattr` check and the other reached for it directly, so the pair read as
    though `stream_read` had lost a guard it needed. It never needed one.
    `connect()` binds the builder before it sets `_connected`, and every reader
    calls `_check_connection()` first, so the unbound state is unreachable from
    the outside -- which is what makes the guard removable rather than merely
    unused.

    Pinned from the outside, through the two doors, because the claim is about
    reachability rather than about either statement: an ordering change inside
    `connect()` that set `_connected` first would make this fail without
    touching either reader.
    """
    db = AsyncPostgresDatabase(
        host="127.0.0.1", port=1, database="x", user="u", password="p", table="t"
    )
    query = Query(filters=[Filter("a", Operator.EQ, 1)])

    assert db.query_builder is None, "declared, unbound, and the sync twin's shape"

    with pytest.raises(RuntimeError, match="not connected"):
        await db.search(query)
    with pytest.raises(RuntimeError, match="not connected"):
        await db.stream_read(query).__anext__()
