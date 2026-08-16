"""Service-backed backends route their ``create()`` / ``upsert()`` mint through ``_generate_id()``.

The in-process matrix (``tests/test_create_mint_id_hook.py``) proves the base
write-keying helper, the shared ``_resolve_upsert_id`` preamble, and the SQL
query builders route the mint fallback through the overridable
``_generate_id()`` hook. This module covers the backends whose ``create()`` /
``create_batch()`` / ``upsert()`` / ``upsert_batch()`` mint the id at an
*inline* site rather than through those shared chokepoints — Postgres and
Elasticsearch — so a consumer overriding the hook governs their create *and*
upsert paths too. (S3's ``create()`` / single ``upsert()`` route through the
base helper, so they are already covered by the in-process file/memory tests.)

These backends also cover the copy-first invariant on real services: neither
``upsert(record)`` nor ``upsert(id, record)`` mutates the caller's record.

Real services, no mocks: per-backend sentinel subclasses override
``_generate_id`` to a recognizable prefix. Postgres requires a running server
(``@requires_postgres``); Elasticsearch requires a reachable cluster,
``TEST_ELASTICSEARCH=true`` and the driver installed
(``@requires_real_elasticsearch``). Each skips when its service is unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Generator

import pytest
from dataknobs_common.testing import requires_postgres, requires_real_elasticsearch

from dataknobs_data import Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase

pytestmark = pytest.mark.integration

_PREFIX = "MINT-SENTINEL-"

_requires_es = requires_real_elasticsearch


class _SentinelMixin:
    """Overrides the single mint hook a consumer would override; the sentinel
    prefix makes a minted id recognizable in assertions.
    """

    def _generate_id(self) -> str:
        return f"{_PREFIX}{uuid.uuid4().hex}"


class _SentinelSyncPostgres(_SentinelMixin, SyncPostgresDatabase):
    pass


class _SentinelAsyncPostgres(_SentinelMixin, AsyncPostgresDatabase):
    pass


class _SentinelSyncElasticsearch(_SentinelMixin, SyncElasticsearchDatabase):
    pass


class _SentinelAsyncElasticsearch(_SentinelMixin, AsyncElasticsearchDatabase):
    pass


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[_SentinelSyncPostgres, None, None]:
    for pg in make_postgres_test_db("test_mint_hook_"):
        db = _SentinelSyncPostgres(pg)
        db.connect()
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[_SentinelAsyncPostgres, None]:
    for pg in make_postgres_test_db("test_mint_hook_async_"):
        db = _SentinelAsyncPostgres(pg)
        await db.connect()
        try:
            yield db
        finally:
            await db.close()


@requires_postgres
def test_sync_pg_create_mints_via_hook(sync_pg: _SentinelSyncPostgres) -> None:
    new_id = sync_pg.create(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    assert sync_pg.read(new_id).get_value("v") == 1


@requires_postgres
def test_sync_pg_create_honors_caller_id(sync_pg: _SentinelSyncPostgres) -> None:
    new_id = sync_pg.create(Record({"v": 1}, id="explicit"))
    assert new_id == "explicit"
    assert not new_id.startswith(_PREFIX)


@requires_postgres
def test_sync_pg_create_batch_mints_via_hook(sync_pg: _SentinelSyncPostgres) -> None:
    ids = sync_pg.create_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


@requires_postgres
async def test_async_pg_create_mints_via_hook(async_pg: _SentinelAsyncPostgres) -> None:
    new_id = await async_pg.create(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    got = await async_pg.read(new_id)
    assert got.get_value("v") == 1


@requires_postgres
async def test_async_pg_create_batch_mints_via_hook(async_pg: _SentinelAsyncPostgres) -> None:
    ids = await async_pg.create_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


@requires_postgres
def test_sync_pg_upsert_mints_via_hook(sync_pg: _SentinelSyncPostgres) -> None:
    new_id = sync_pg.upsert(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    assert sync_pg.read(new_id).get_value("v") == 1


@requires_postgres
def test_sync_pg_upsert_batch_mints_via_hook(sync_pg: _SentinelSyncPostgres) -> None:
    ids = sync_pg.upsert_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


@requires_postgres
async def test_async_pg_upsert_mints_via_hook(async_pg: _SentinelAsyncPostgres) -> None:
    new_id = await async_pg.upsert(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    got = await async_pg.read(new_id)
    assert got.get_value("v") == 1


@requires_postgres
async def test_async_pg_upsert_batch_mints_via_hook(
    async_pg: _SentinelAsyncPostgres,
) -> None:
    ids = await async_pg.upsert_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


@requires_postgres
def test_sync_pg_upsert_does_not_mutate_caller(sync_pg: _SentinelSyncPostgres) -> None:
    """Neither ``upsert`` call form stamps the caller's record (copy-first)."""
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = sync_pg.upsert(rec)
    assert new_id and rec.storage_id == before
    assert sync_pg.read(new_id).get_value("v") == 1

    rec2 = Record({"v": 2})  # no id
    before2 = rec2.storage_id
    assert sync_pg.upsert("explicit", rec2) == "explicit"
    assert rec2.storage_id == before2  # caller record NOT stamped to "explicit"
    assert sync_pg.read("explicit").get_value("v") == 2


@requires_postgres
async def test_async_pg_upsert_does_not_mutate_caller(
    async_pg: _SentinelAsyncPostgres,
) -> None:
    """Neither ``upsert`` call form stamps the caller's record (copy-first)."""
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = await async_pg.upsert(rec)
    assert new_id and rec.storage_id == before
    got = await async_pg.read(new_id)
    assert got.get_value("v") == 1

    rec2 = Record({"v": 2})  # no id
    before2 = rec2.storage_id
    assert await async_pg.upsert("explicit", rec2) == "explicit"
    assert rec2.storage_id == before2  # caller record NOT stamped to "explicit"
    got2 = await async_pg.read("explicit")
    assert got2.get_value("v") == 2


@requires_postgres
def test_sync_pg_stream_write_insert_mints_via_hook(sync_pg: _SentinelSyncPostgres) -> None:
    """The streaming INSERT fast-path (``_write_batch``) mints via the hook too."""
    result = sync_pg.stream_write(iter([Record({"v": i}) for i in range(3)]))
    assert result.successful == 3
    ids = [r.id for r in sync_pg.all()]
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)


@requires_postgres
async def test_async_pg_stream_write_insert_mints_via_hook(
    async_pg: _SentinelAsyncPostgres,
) -> None:
    """The async streaming COPY INSERT fast-path mints via the hook too."""

    async def _records() -> AsyncIterator[Record]:
        for i in range(3):
            yield Record({"v": i})

    result = await async_pg.stream_write(_records())
    assert result.successful == 3
    ids = [r.id for r in await async_pg.all()]
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------
@_requires_es
def test_sync_es_create_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        new_id = db.create(Record({"v": 1}))
        assert new_id.startswith(_PREFIX)
        assert db.read(new_id).get_value("v") == 1
    finally:
        db.close()


@_requires_es
def test_sync_es_create_honors_caller_id(elasticsearch_test_index) -> None:
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        new_id = db.create(Record({"v": 1}, id="explicit"))
        assert new_id == "explicit"
        assert not new_id.startswith(_PREFIX)
    finally:
        db.close()


@_requires_es
def test_sync_es_create_batch_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        ids = db.create_batch([Record({"v": i}) for i in range(3)])
        assert len(ids) == 3
        assert all(rid.startswith(_PREFIX) for rid in ids)
        assert len(set(ids)) == 3
    finally:
        db.close()


@_requires_es
async def test_async_es_create_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelAsyncElasticsearch(elasticsearch_test_index)
    await db.connect()
    try:
        new_id = await db.create(Record({"v": 1}))
        assert new_id.startswith(_PREFIX)
        got = await db.read(new_id)
        assert got.get_value("v") == 1
    finally:
        await db.close()


@_requires_es
async def test_async_es_create_batch_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelAsyncElasticsearch(elasticsearch_test_index)
    await db.connect()
    try:
        ids = await db.create_batch([Record({"v": i}) for i in range(3)])
        assert len(ids) == 3
        assert all(rid.startswith(_PREFIX) for rid in ids)
        assert len(set(ids)) == 3
    finally:
        await db.close()


@_requires_es
def test_sync_es_upsert_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        new_id = db.upsert(Record({"v": 1}))
        assert new_id.startswith(_PREFIX)
        assert db.read(new_id).get_value("v") == 1
    finally:
        db.close()


@_requires_es
def test_sync_es_upsert_batch_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        ids = db.upsert_batch([Record({"v": i}) for i in range(3)])
        assert len(ids) == 3
        assert all(rid.startswith(_PREFIX) for rid in ids)
        assert len(set(ids)) == 3
    finally:
        db.close()


@_requires_es
async def test_async_es_upsert_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelAsyncElasticsearch(elasticsearch_test_index)
    await db.connect()
    try:
        new_id = await db.upsert(Record({"v": 1}))
        assert new_id.startswith(_PREFIX)
        got = await db.read(new_id)
        assert got.get_value("v") == 1
    finally:
        await db.close()


@_requires_es
async def test_async_es_upsert_batch_mints_via_hook(elasticsearch_test_index) -> None:
    db = _SentinelAsyncElasticsearch(elasticsearch_test_index)
    await db.connect()
    try:
        ids = await db.upsert_batch([Record({"v": i}) for i in range(3)])
        assert len(ids) == 3
        assert all(rid.startswith(_PREFIX) for rid in ids)
        assert len(set(ids)) == 3
    finally:
        await db.close()


@_requires_es
def test_sync_es_upsert_does_not_mutate_caller(elasticsearch_test_index) -> None:
    """Neither ``upsert`` call form stamps the caller's record (copy-first)."""
    db = _SentinelSyncElasticsearch(elasticsearch_test_index)
    db.connect()
    try:
        rec = Record({"v": 1})  # no id
        before = rec.storage_id
        new_id = db.upsert(rec)
        assert new_id and rec.storage_id == before
        assert db.read(new_id).get_value("v") == 1

        rec2 = Record({"v": 2})  # no id
        before2 = rec2.storage_id
        assert db.upsert("explicit", rec2) == "explicit"
        assert rec2.storage_id == before2  # caller record NOT stamped
        assert db.read("explicit").get_value("v") == 2
    finally:
        db.close()


@_requires_es
async def test_async_es_upsert_does_not_mutate_caller(elasticsearch_test_index) -> None:
    """Neither ``upsert`` call form stamps the caller's record (copy-first)."""
    db = _SentinelAsyncElasticsearch(elasticsearch_test_index)
    await db.connect()
    try:
        rec = Record({"v": 1})  # no id
        before = rec.storage_id
        new_id = await db.upsert(rec)
        assert new_id and rec.storage_id == before
        got = await db.read(new_id)
        assert got.get_value("v") == 1

        rec2 = Record({"v": 2})  # no id
        before2 = rec2.storage_id
        assert await db.upsert("explicit", rec2) == "explicit"
        assert rec2.storage_id == before2  # caller record NOT stamped
        got2 = await db.read("explicit")
        assert got2.get_value("v") == 2
    finally:
        await db.close()
