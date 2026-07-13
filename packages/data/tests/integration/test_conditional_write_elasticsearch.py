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


def test_sync_conditional_update_missing_returns_false(elasticsearch_test_index) -> None:
    """A conditional update of an absent id returns False (never inserts)."""
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        result = db.update(
            "ghost", Record({"v": 1}, id="ghost"), expected_version="1:1"
        )
        assert result is False
        assert db.read("ghost") is None
    finally:
        db.close()


async def test_async_conditional_update_missing_returns_false(
    elasticsearch_test_index,
) -> None:
    """Reproduce-first: a conditional update of an absent id must return False.

    The async backend caught only ``ConflictError``, so ES's 404
    ``document_missing_exception`` (raised by ``elasticsearch-py`` as
    ``NotFoundError``) propagated uncaught instead of the documented ``False``
    — diverging from the sync backend and every other backend. This exercises
    the missing-record path through ``update()`` directly; the pre-existing
    conditional tests only reach the missing path via ``upsert()``.
    """
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        result = await db.update(
            "ghost", Record({"v": 1}, id="ghost"), expected_version="1:1"
        )
        assert result is False
        assert await db.read("ghost") is None
    finally:
        await db.close()


def test_sync_conditional_delete(elasticsearch_test_index) -> None:
    """Conditional delete: stale raises, fresh removes, missing returns False."""
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        db.create(Record({"v": 0}, id="k"))
        stale = db.get_version("k")
        # An interleaved writer advances the token.
        db.update("k", Record({"v": 1}, id="k"), expected_version=stale)
        with pytest.raises(ConcurrencyError):
            db.delete("k", expected_version=stale)
        assert db.read("k") is not None
        # Fresh token deletes.
        fresh = db.get_version("k")
        assert db.delete("k", expected_version=fresh) is True
        assert db.read("k") is None
        # Conditional delete of an absent id returns False.
        assert db.delete("k", expected_version="1:1") is False
    finally:
        db.close()


async def test_async_conditional_delete(elasticsearch_test_index) -> None:
    """Conditional delete: stale raises, fresh removes, missing returns False."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await db.create(Record({"v": 0}, id="k"))
        stale = await db.get_version("k")
        await db.update("k", Record({"v": 1}, id="k"), expected_version=stale)
        with pytest.raises(ConcurrencyError):
            await db.delete("k", expected_version=stale)
        assert await db.read("k") is not None
        fresh = await db.get_version("k")
        assert await db.delete("k", expected_version=fresh) is True
        assert await db.read("k") is None
        assert await db.delete("k", expected_version="1:1") is False
    finally:
        await db.close()
