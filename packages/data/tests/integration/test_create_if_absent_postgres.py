"""Atomic create-if-absent on the Postgres backends (real service).

Before the unified contract, a colliding ``create()`` surfaced the raw
driver ``UniqueViolation`` / ``UniqueViolationError``. These tests pin that a
duplicate id now raises the typed ``DuplicateRecordError`` (which subclasses
``ValueError``) on both the sync and async Postgres backends, and that the
original record is not overwritten.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = requires_postgres


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_create_absent_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_create_absent_async_"):
        db = AsyncPostgresDatabase(pg)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


def test_sync_duplicate_create_raises_typed(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_pg.create(Record({"v": "loser"}, id="dup"))
    assert excinfo.value.id == "dup"
    # Subclasses ValueError, and the original record survives.
    assert isinstance(excinfo.value, ValueError)
    assert sync_pg.read("dup").get_value("v") == "winner"


async def test_async_duplicate_create_raises_typed(async_pg: AsyncPostgresDatabase) -> None:
    await async_pg.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_pg.create(Record({"v": "loser"}, id="dup"))
    assert excinfo.value.id == "dup"
    assert isinstance(excinfo.value, ValueError)
    got = await async_pg.read("dup")
    assert got.get_value("v") == "winner"


async def test_async_concurrent_create_exactly_one_wins(async_pg: AsyncPostgresDatabase) -> None:
    results = await asyncio.gather(
        async_pg.create(Record({"who": "a"}, id="race")),
        async_pg.create(Record({"who": "b"}, id="race")),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    duplicates = [r for r in results if isinstance(r, DuplicateRecordError)]
    assert len(successes) == 1
    assert len(duplicates) == 1
