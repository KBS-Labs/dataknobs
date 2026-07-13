"""Atomic create-if-absent on the Elasticsearch backends (real service).

``create()`` indexes with ``op_type=create`` so a colliding id yields a 409
version conflict, surfaced as ``DuplicateRecordError`` instead of silently
overwriting the existing document. Pinned on both the sync backend (via the
``SimplifiedElasticsearchIndex`` REST wrapper) and the async backend (native
``elasticsearch-py`` client).

Enable with ``TEST_ELASTICSEARCH=true`` and a running Elasticsearch; the
module skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled",
)


def test_sync_duplicate_create_raises(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        db.create(Record({"v": "winner"}, id="dup"))
        with pytest.raises(DuplicateRecordError) as excinfo:
            db.create(Record({"v": "loser"}, id="dup"))
        assert excinfo.value.id == "dup"
        assert db.read("dup").get_value("v") == "winner"
    finally:
        db.close()


async def test_async_duplicate_create_raises(elasticsearch_test_index) -> None:
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await db.create(Record({"v": "winner"}, id="dup"))
        with pytest.raises(DuplicateRecordError) as excinfo:
            await db.create(Record({"v": "loser"}, id="dup"))
        assert excinfo.value.id == "dup"
        got = await db.read("dup")
        assert got.get_value("v") == "winner"
    finally:
        await db.close()
