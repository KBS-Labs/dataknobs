"""Fail-closed ``create_batch`` and streaming INSERT on Elasticsearch (real service).

``create_batch()`` now uses the bulk ``create`` op keyed on ``record.id`` (no more
server-assigned ids or silent overwrite): a colliding id fails closed with a 409,
surfaced as ``DuplicateRecordError``, and a caller-supplied ``record.id`` is
honored. The Elasticsearch bulk API is per-item (non-atomic), so — exactly like a
``create()`` loop — the non-colliding records may already be indexed when the
conflict is raised; these tests therefore assert the fail-closed signal and id
honoring, not whole-batch atomicity.

The streaming INSERT path — and the batched ``Migrator.migrate()`` INSERT path —
both route through per-record ``create()`` (the non-atomic bulk would double-write
under the per-record fallback), so a colliding source id in a re-run into a
populated target is recorded as a failure with the source id preserved.

Enable with ``TEST_ELASTICSEARCH=true`` and a running Elasticsearch; the module
skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.streaming import StreamConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled",
)


def test_sync_create_batch_fails_closed(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        db.create(Record({"v": "old"}, id="dup"))
        with pytest.raises(DuplicateRecordError):
            db.create_batch(
                [Record({"v": 1}, id="fresh"), Record({"v": 2}, id="dup")]
            )
        # The pre-existing document is not overwritten by the failed batch.
        assert db.read("dup").get_value("v") == "old"
    finally:
        db.close()


def test_sync_create_batch_preserves_ids(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        ids = db.create_batch([Record({"v": 1}, id="x"), Record({"v": 2}, id="y")])
        assert sorted(ids) == ["x", "y"]
        assert db.read("x").get_value("v") == 1
    finally:
        db.close()


def test_sync_streaming_insert_fails_closed(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        db.create(Record({"v": "old"}, id="2"))
        records = [Record({"v": "src"}, id=str(i)) for i in (1, 2, 3)]
        result = db.stream_write(
            iter(records), StreamConfig(on_error=lambda e, r: True)
        )
        assert result.failed == 1
        assert result.successful == 2
        assert db.read("2").get_value("v") == "old"
        assert db.read("1").get_value("v") == "src"
    finally:
        db.close()


def test_sync_migrate_insert_fails_closed_into_populated_index(
    elasticsearch_test_index,
) -> None:
    # Regression: Migrator.migrate() INSERT into a populated ES index must route
    # per-record, exactly as ES's own stream_write does. ES create_batch is
    # non-atomic on raise (its per-item bulk indexes the non-colliding rows
    # before the 409), so riding the bulk fast-path here would let the
    # per-record fallback re-write those already-indexed rows and mis-count them
    # as duplicate failures. ES inherits _insert_batch_atomic()=False, so the
    # migrator skips the bulk verb and attributes each id correctly.
    from dataknobs_data.backends.memory import SyncMemoryDatabase
    from dataknobs_data.migration import Migrator

    src = SyncMemoryDatabase()
    for i in (1, 2, 3):
        src.create(Record({"v": "src"}, id=str(i)))

    tgt = SyncElasticsearchDatabase(elasticsearch_test_index)
    tgt.connect()
    try:
        tgt.create(Record({"v": "old"}, id="2"))  # collides with a source id
        progress = Migrator().migrate(src, tgt, on_error=lambda e, r: True)
        assert progress.succeeded == 2  # ids 1 and 3 written
        assert progress.failed == 1     # id 2 collided
        assert tgt.read("2").get_value("v") == "old"  # collider untouched
        assert tgt.read("1").get_value("v") == "src"
        assert tgt.read("3").get_value("v") == "src"
    finally:
        tgt.close()


async def test_async_create_batch_fails_closed(elasticsearch_test_index) -> None:
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await db.create(Record({"v": "old"}, id="dup"))
        with pytest.raises(DuplicateRecordError):
            await db.create_batch(
                [Record({"v": 1}, id="fresh"), Record({"v": 2}, id="dup")]
            )
        got = await db.read("dup")
        assert got.get_value("v") == "old"
    finally:
        await db.close()


async def test_async_create_batch_preserves_ids(elasticsearch_test_index) -> None:
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        ids = await db.create_batch(
            [Record({"v": 1}, id="x"), Record({"v": 2}, id="y")]
        )
        assert sorted(ids) == ["x", "y"]
        assert (await db.read("x")).get_value("v") == 1
    finally:
        await db.close()


async def test_async_streaming_insert_fails_closed(elasticsearch_test_index) -> None:
    # Async ES routes streaming INSERT per-record via create() (its non-atomic
    # bulk would double-write under the fallback), so a colliding source id in a
    # re-run into a populated target is recorded as a failure with the id
    # preserved — never overwritten.
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await db.create(Record({"v": "old"}, id="2"))
        records = [Record({"v": "src"}, id=str(i)) for i in (1, 2, 3)]

        async def _aiter():
            for r in records:
                yield r

        result = await db.stream_write(
            _aiter(), StreamConfig(on_error=lambda e, r: True)
        )
        assert result.failed == 1
        assert result.successful == 2
        assert (await db.read("2")).get_value("v") == "old"
        assert (await db.read("1")).get_value("v") == "src"
    finally:
        await db.close()
