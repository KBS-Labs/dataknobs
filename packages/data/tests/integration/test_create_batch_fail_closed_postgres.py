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
    result = sync_pg.stream_write(iter(records), StreamConfig(on_error=lambda e, r: True))
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
    ids = await async_pg.create_batch([Record({"v": 1}, id="x"), Record({"v": 2}, id="y")])
    assert ids == ["x", "y"]
    assert (await async_pg.read("x")).get_value("v") == 1


async def test_async_streaming_insert_fails_closed(
    async_pg: AsyncPostgresDatabase,
) -> None:
    # The async INSERT fast-path is COPY (a single atomic command), so a
    # colliding id rolls the batch back and the per-record create() fallback
    # replays it cleanly — no double-write, the collision counted as one failure.
    await async_pg.create(Record({"v": "old"}, id="2"))
    records = [Record({"v": "src"}, id=str(i)) for i in (1, 2, 3)]

    async def _aiter() -> AsyncGenerator[Record, None]:
        for r in records:
            yield r

    result = await async_pg.stream_write(_aiter(), StreamConfig(on_error=lambda e, r: True))
    assert result.failed == 1
    assert result.successful == 2
    assert (await async_pg.read("2")).get_value("v") == "old"
    assert (await async_pg.read("1")).get_value("v") == "src"
    assert (await async_pg.read("3")).get_value("v") == "src"


# Naming the colliding id is a lookup over the whole batch. Asking once per
# record made a failed batch cost one round trip per record before the one that
# collided, so these put it last and count what was asked.

_PRECEDING = 20


class _CountingSyncPostgres(SyncPostgresDatabase):
    """Counts the statements its reads run; every one still runs."""

    reads = 0

    def _rows(self, sql, params):
        self.reads += 1
        return super()._rows(sql, params)


def _colliding_last() -> list[Record]:
    """Records nothing has stored, one that mints its id, then the collision."""
    fresh = [Record({"v": i}, id=f"new{i}") for i in range(_PRECEDING)]
    return [Record({"v": "minted"}), *fresh, Record({"v": "src"}, id="dup")]


@pytest.mark.parametrize("verb", ["create_batch", "_write_batch"])
def test_sync_names_the_collision_in_one_lookup(make_postgres_test_db, verb: str) -> None:
    for pg in make_postgres_test_db("test_create_batch_fc_count_"):
        db = _CountingSyncPostgres(pg)
        db.connect()
        try:
            db.create(Record({"v": "old"}, id="dup"))
            db.reads = 0
            with pytest.raises(DuplicateRecordError) as raised:
                getattr(db, verb)(_colliding_last())
            assert raised.value.id == "dup"
            assert db.reads == 1
        finally:
            db.close()


@pytest.mark.parametrize("verb", ["create_batch", "_write_batch"])
async def test_async_names_the_collision_in_one_lookup(
    async_pg: AsyncPostgresDatabase, verb: str
) -> None:
    await async_pg.create(Record({"v": "old"}, id="dup"))
    batch = _colliding_last()

    # asyncpg reports each statement a connection runs, so this counts what
    # reached the server on whichever pooled connection ran it.
    statements: list[str] = []

    def _log(query) -> None:
        statements.append(query.query)

    pool = async_pg._require_pool()
    held = [await pool.acquire() for _ in range(pool.get_max_size())]
    for conn in held:
        conn.add_query_logger(_log)
    for conn in held:
        await pool.release(conn)

    with pytest.raises(DuplicateRecordError) as raised:
        await getattr(async_pg, verb)(batch)
    assert raised.value.id == "dup"
    lookups = [s for s in statements if s.lstrip().upper().startswith("SELECT") and "pg_" not in s]
    assert len(lookups) == 1, lookups
