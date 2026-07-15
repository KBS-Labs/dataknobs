"""Queryable record identifiers on the Elasticsearch backends.

``Filter("id", ...)`` resolves to the document's storage key, and
``Operator.STARTS_WITH`` compiles to a case-sensitive ``prefix`` query on the
``id`` keyword field. This module pins the async backend at parity with the sync
one — reproduce-first, since the async ``search`` / ``count`` loops previously
built ``data.id`` unconditionally, so an id filter never resolved to the storage
key.

Enable with ``TEST_ELASTICSEARCH=true`` and a running Elasticsearch; the module
skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled",
)

_KEYS = [
    "artifacts/alice/report/final",
    "artifacts/alice/report/draft",
    "artifacts/bob/report/final",
    "orders/1",
    "orders/2",
]


def _ids(records: list[Record]) -> set[str]:
    return {r.id for r in records}


def _seed_sync(db: SyncElasticsearchDatabase) -> None:
    for k in _KEYS:
        db.create(Record({"payload": k}, id=k))


async def _seed_async(db: AsyncElasticsearchDatabase) -> None:
    for k in _KEYS:
        await db.create(Record({"payload": k}, id=k))


def test_sync_id_prefix_and_operators(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        _seed_sync(db)
        assert _ids(
            db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/")]))
        ) == {"artifacts/alice/report/final", "artifacts/alice/report/draft"}
        assert _ids(db.search(Query(filters=[Filter("id", Operator.EQ, "orders/1")]))) == {
            "orders/1"
        }
        assert _ids(
            db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "orders/")]))
        ) == {"orders/1", "orders/2"}
    finally:
        db.close()


async def test_async_id_resolves_to_storage_key(elasticsearch_test_index) -> None:
    """Reproduce-first: async id filters now target the storage key, not data.id."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await _seed_async(db)
        eq = await db.search(Query(filters=[Filter("id", Operator.EQ, "orders/1")]))
        prefix = await db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/")])
        )
        in_ = await db.search(
            Query(filters=[Filter("id", Operator.IN, ["orders/1", "orders/2"])])
        )
        assert _ids(eq) == {"orders/1"}
        assert _ids(prefix) == {
            "artifacts/alice/report/final",
            "artifacts/alice/report/draft",
        }
        assert _ids(in_) == {"orders/1", "orders/2"}
    finally:
        await db.close()


async def test_async_count_id_prefix(elasticsearch_test_index) -> None:
    """The async ``count`` loop honors the same id resolution + prefix."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await _seed_async(db)
        n = await db.count(Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")]))
        assert n == 3
        n_orders = await db.count(Query(filters=[Filter("id", Operator.EQ, "orders/2")]))
        assert n_orders == 1
    finally:
        await db.close()


async def test_async_matches_sync(elasticsearch_test_index) -> None:
    """Async and sync ES return the same rows for the same id prefix query."""
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_sync(sync_db)
        q = Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")])
        assert _ids(await async_db.search(q)) == _ids(sync_db.search(q))
    finally:
        sync_db.close()
        await async_db.close()


def test_sync_count_between_is_filtered(elasticsearch_test_index) -> None:
    """Reproduce-first: a BETWEEN-only ``count()`` returns the filtered count.

    Before the sync search/count paths were routed through the single shared
    per-filter translator, ``count()`` had its own inline loop with no
    ``BETWEEN`` branch, so the filter was dropped and the query fell back to
    ``match_all`` — returning the total (5) instead of the filtered count (3).
    """
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        for i in range(1, 6):
            db.create(Record({"n": i}, id=f"r{i}"))
        assert db.count() == 5
        assert db.count(Query(filters=[Filter("n", Operator.BETWEEN, [2, 4])])) == 3
    finally:
        db.close()
