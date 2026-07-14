"""Fail-closed ``create_batch`` and streaming INSERT on Postgres (real service).

``create_batch()`` now honors the same atomic-insert contract as ``create()``: a
colliding id raises ``DuplicateRecordError`` and — because the multi-row INSERT
runs in one transaction — the whole batch is rolled back (nothing written). A
caller-supplied ``record.id`` is honored (no minting).

The streaming INSERT path is covered too: it routes through the atomic
``_write_batch`` fast-path with a per-record ``create()`` fallback, so a colliding
source id in a re-run into a populated target is recorded as a failure with the
source id preserved (not a fresh-id row).

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.streaming import StreamConfig

pytestmark = requires_postgres


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_create_batch_fc_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_create_batch_fc_async_"):
        db = AsyncPostgresDatabase(pg)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


def test_sync_create_batch_fails_closed_and_is_atomic(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": "old"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        sync_pg.create_batch(
            [
                Record({"v": 1}, id="new1"),
                Record({"v": 2}, id="dup"),
                Record({"v": 3}, id="new2"),
            ]
        )
    # Atomic: the single INSERT rolled back, so nothing from the batch persisted.
    assert sync_pg.read("new1") is None
    assert sync_pg.read("new2") is None
    assert sync_pg.read("dup").get_value("v") == "old"


def test_sync_create_batch_preserves_ids(sync_pg: SyncPostgresDatabase) -> None:
    ids = sync_pg.create_batch([Record({"v": 1}, id="x"), Record({"v": 2}, id="y")])
    assert ids == ["x", "y"]
    assert sync_pg.read("x").get_value("v") == 1


def test_sync_streaming_insert_fails_closed(sync_pg: SyncPostgresDatabase) -> None:
    sync_pg.create(Record({"v": "old"}, id="2"))
    records = [Record({"v": "src"}, id=str(i)) for i in (1, 2, 3)]
    result = sync_pg.stream_write(
        iter(records), StreamConfig(on_error=lambda e, r: True)
    )
    assert result.failed == 1
    assert result.successful == 2
    assert sync_pg.read("2").get_value("v") == "old"
    assert sync_pg.read("1").get_value("v") == "src"


async def test_async_create_batch_fails_closed_and_is_atomic(
    async_pg: AsyncPostgresDatabase,
) -> None:
    await async_pg.create(Record({"v": "old"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        await async_pg.create_batch(
            [
                Record({"v": 1}, id="new1"),
                Record({"v": 2}, id="dup"),
                Record({"v": 3}, id="new2"),
            ]
        )
    assert await async_pg.read("new1") is None
    assert await async_pg.read("new2") is None
    got = await async_pg.read("dup")
    assert got.get_value("v") == "old"


async def test_async_create_batch_preserves_ids(async_pg: AsyncPostgresDatabase) -> None:
    ids = await async_pg.create_batch(
        [Record({"v": 1}, id="x"), Record({"v": 2}, id="y")]
    )
    assert ids == ["x", "y"]
    assert (await async_pg.read("x")).get_value("v") == 1
