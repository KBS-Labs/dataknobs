"""Optimistic-concurrency (conditional write) on the Elasticsearch backends.

``get_version(id)`` returns the document's native ``_seq_no``/``_primary_term``
pair as one opaque token; passing it back as ``expected_version`` carries ES's
``if_seq_no``/``if_primary_term`` guards so the compare-and-set is enforced
server-side. A stale token raises ``ConcurrencyError`` instead of last-writer-
wins. Pinned on both the sync backend (``SimplifiedElasticsearchIndex`` REST
wrapper) and the async backend (native ``elasticsearch-py`` client).

Enable with ``TEST_ELASTICSEARCH=true`` and a running Elasticsearch; the
module skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled",
)


def test_sync_conditional_write(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        assert db.get_version("missing") is None
        db.create(Record({"v": 0}, id="k"))
        token = db.get_version("k")
        assert token is not None

        # Fresh token succeeds and advances the version.
        assert db.update("k", Record({"v": 1}, id="k"), expected_version=token) is True
        assert db.read("k").get_value("v") == 1
        assert db.get_version("k") != token

        # A stale token loses with ConcurrencyError.
        stale = token
        with pytest.raises(ConcurrencyError) as excinfo:
            db.update("k", Record({"v": 2}, id="k"), expected_version=stale)
        assert excinfo.value.context["id"] == "k"
        assert db.read("k").get_value("v") == 1

        # Conditional upsert against a missing id never inserts.
        with pytest.raises(ConcurrencyError) as excinfo2:
            db.upsert("ghost", Record({"v": 9}, id="ghost"), expected_version="1:1")
        assert excinfo2.value.context["actual_version"] is None
    finally:
        db.close()


async def test_async_conditional_write(elasticsearch_test_index) -> None:
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        assert await db.get_version("missing") is None
        await db.create(Record({"v": 0}, id="k"))
        token = await db.get_version("k")
        assert token is not None

        assert (
            await db.update("k", Record({"v": 1}, id="k"), expected_version=token)
        ) is True
        got = await db.read("k")
        assert got.get_value("v") == 1

        with pytest.raises(ConcurrencyError):
            await db.update("k", Record({"v": 2}, id="k"), expected_version=token)
        got = await db.read("k")
        assert got.get_value("v") == 1
    finally:
        await db.close()
