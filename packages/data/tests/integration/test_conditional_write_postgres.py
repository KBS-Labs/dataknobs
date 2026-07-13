"""Optimistic-concurrency (conditional write) on the Postgres backends.

``get_version(id)`` returns the row's ``xmin`` transaction id, and passing it
back as ``expected_version`` turns ``update`` / ``upsert`` into a compare-and-set
enforced server-side via ``WHERE ... AND xmin = …``. A stale token raises
``ConcurrencyError`` instead of last-writer-wins; the concurrent test proves
exactly one of several racers wins because the CAS is atomic in the database.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = requires_postgres


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_cond_write_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_cond_write_async_"):
        db = AsyncPostgresDatabase(pg)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


def test_sync_get_version_absent_and_present(sync_pg: SyncPostgresDatabase) -> None:
    assert sync_pg.get_version("missing") is None
    sync_pg.create(Record({"v": 0}, id="k"))
    assert sync_pg.get_version("k") is not None


def test_sync_conditional_update_fresh_token_succeeds(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": 0}, id="k"))
    token = sync_pg.get_version("k")
    assert sync_pg.update("k", Record({"v": 1}, id="k"), expected_version=token) is True
    assert sync_pg.read("k").get_value("v") == 1
    # xmin advances on the update, so the token changes.
    assert sync_pg.get_version("k") != token


def test_sync_conditional_update_stale_token_raises(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": 0}, id="k"))
    stale = sync_pg.get_version("k")
    sync_pg.update("k", Record({"v": "A"}, id="k"), expected_version=stale)
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_pg.update("k", Record({"v": "B"}, id="k"), expected_version=stale)
    assert excinfo.value.context["id"] == "k"
    assert excinfo.value.context["expected_version"] == stale
    assert sync_pg.read("k").get_value("v") == "A"


def test_sync_unconditional_update_still_clobbers(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": 0}, id="k"))
    assert sync_pg.update("k", Record({"v": 1}, id="k")) is True
    assert sync_pg.update("k", Record({"v": 2}, id="k")) is True
    assert sync_pg.read("k").get_value("v") == 2


def test_sync_conditional_upsert_absent_raises(sync_pg: SyncPostgresDatabase) -> None:
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_pg.upsert("ghost", Record({"v": 1}, id="ghost"), expected_version="1")
    assert excinfo.value.context["actual_version"] is None


def test_sync_conditional_upsert_stale_raises(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": 0}, id="k"))
    stale = sync_pg.get_version("k")
    sync_pg.update("k", Record({"v": "A"}, id="k"), expected_version=stale)
    with pytest.raises(ConcurrencyError):
        sync_pg.upsert("k", Record({"v": "B"}, id="k"), expected_version=stale)
    assert sync_pg.read("k").get_value("v") == "A"


async def test_async_conditional_update_stale_token_raises(
    async_pg: AsyncPostgresDatabase,
) -> None:
    await async_pg.create(Record({"v": 0}, id="k"))
    stale = await async_pg.get_version("k")
    await async_pg.update("k", Record({"v": "A"}, id="k"), expected_version=stale)
    with pytest.raises(ConcurrencyError):
        await async_pg.update("k", Record({"v": "B"}, id="k"), expected_version=stale)
    got = await async_pg.read("k")
    assert got.get_value("v") == "A"


async def test_async_concurrent_conditional_update_one_wins(
    async_pg: AsyncPostgresDatabase,
) -> None:
    await async_pg.create(Record({"v": 0}, id="race"))
    token = await async_pg.get_version("race")
    results = await asyncio.gather(
        async_pg.update("race", Record({"who": "a"}, id="race"), expected_version=token),
        async_pg.update("race", Record({"who": "b"}, id="race"), expected_version=token),
        async_pg.update("race", Record({"who": "c"}, id="race"), expected_version=token),
        return_exceptions=True,
    )
    successes = [r for r in results if r is True]
    conflicts = [r for r in results if isinstance(r, ConcurrencyError)]
    assert len(successes) == 1
    assert len(conflicts) == 2
