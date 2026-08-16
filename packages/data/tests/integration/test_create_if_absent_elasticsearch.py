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


import pytest

from dataknobs_common.testing import requires_real_elasticsearch
from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = requires_real_elasticsearch


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


def test_sync_create_honors_payload_id_field(elasticsearch_test_index) -> None:
    """A payload ``id`` data field is the storage key; a collision fails closed."""
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        assert db.create(Record({"id": "x", "v": 1})) == "x"
        assert db.read("x").get_value("v") == 1
        with pytest.raises(DuplicateRecordError) as excinfo:
            db.create(Record({"id": "x", "v": 2}))
        assert excinfo.value.id == "x"
    finally:
        db.close()


async def test_async_create_honors_payload_id_field(elasticsearch_test_index) -> None:
    """A payload ``id`` data field is the storage key; a collision fails closed."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        assert await db.create(Record({"id": "x", "v": 1})) == "x"
        assert (await db.read("x")).get_value("v") == 1
        with pytest.raises(DuplicateRecordError) as excinfo:
            await db.create(Record({"id": "x", "v": 2}))
        assert excinfo.value.id == "x"
    finally:
        await db.close()
