"""``ChromaVectorStore.add_documents`` write-path parity with ``add_vectors``.

Both methods write rows into the same collection and every read path
treats those rows identically, so what one write path stamps onto a row
the other has to stamp too. ``add_documents`` did not: rows it wrote
carried no configured ``domain_id``, which made them invisible to the
scoped reads that every other method applies — the store wrote a
document and then could not find it.

The embedding here comes from chromadb's own default function, so these
stores are configured at its dimensionality rather than the small
vectors the sibling suites use.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from dataknobs_common.testing import is_chromadb_available

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

requires_chromadb = pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed")

pytestmark = [pytest.mark.asyncio, requires_chromadb]

# chromadb's default embedding function decides this, not the caller.
DEFAULT_EF_DIMENSIONS = 384


async def _store(**config: Any) -> Any:
    store = ChromaVectorStore(
        {
            "dimensions": DEFAULT_EF_DIMENSIONS,
            "collection_name": f"test_add_docs_{uuid.uuid4().hex[:8]}",
            **config,
        }
    )
    await store.initialize()
    return store


async def _drop(store: Any) -> None:
    try:
        store.client.delete_collection(name=store.collection_name)
    finally:
        await store.close()


async def test_add_documents_applies_the_configured_domain():  # type: ignore[no-untyped-def]
    """A scoped store can read back a document it just wrote.

    ``add_vectors`` defaults the configured ``domain_id`` into every
    row it writes; ``add_documents`` reached the collection without it.
    Every scoped read then filtered the row out — ``count()`` did not
    see it, and a document search for the row's own text returned some
    other row instead.
    """
    store = await _store(domain_id="tenant-a")
    try:
        await store.add_documents(["a written document"], ids=["d1"], metadata=[{"kind": "doc"}])

        rows = await store.get_vectors(["d1"])
        meta = rows[0][1]
        assert meta is not None
        assert meta.get("domain_id") == "tenant-a", f"row written unscoped: {meta}"
        assert meta.get("kind") == "doc"

        # The scoped reads agree that the row is there.
        assert await store.count() == 1
        hits = await store.search_documents("a written document", k=5)
        assert [h[0] for h in hits] == ["d1"]
    finally:
        await _drop(store)


async def test_add_documents_does_not_cross_domains():  # type: ignore[no-untyped-def]
    """The default is a default, not an override.

    A row carrying its own ``domain_id`` keeps it, which is what makes
    the stamp safe to apply unconditionally.
    """
    store = await _store(domain_id="tenant-a")
    try:
        await store.add_documents(
            ["mine", "theirs"],
            ids=["d1", "d2"],
            metadata=[{}, {"domain_id": "tenant-b"}],
        )
        assert await store.count() == 1
        rows = await store.get_vectors(["d1", "d2"])
        assert (rows[0][1] or {}).get("domain_id") == "tenant-a"
        assert (rows[1][1] or {}).get("domain_id") == "tenant-b"
    finally:
        await _drop(store)


async def test_add_documents_tracks_timestamps():  # type: ignore[no-untyped-def]
    """Rows arriving this way are timestamped like any other.

    ``search`` does not care which write path produced a row, so a
    document written without timestamps would report ``None`` for them
    permanently.
    """
    store = await _store()
    try:
        await store.add_documents(["hello"], ids=["d1"], metadata=[{"kind": "doc"}])
        rows = await store.get_vectors(["d1"], include_timestamps=True)
        meta = rows[0][1]
        assert meta is not None
        assert meta["_created_at"] is not None
        assert meta["_updated_at"] is not None
        # And the reserved storage keys stay out of the consumer's view.
        assert all(not k.startswith("\x00dk\x00") for k in meta), meta
    finally:
        await _drop(store)


async def test_readd_document_id_is_an_upsert():  # type: ignore[no-untyped-def]
    """Re-writing a document id replaces it, matching ``add_vectors``."""
    store = await _store()
    try:
        await store.add_documents(["first text"], ids=["d1"], metadata=[{"v": 1}])
        await store.add_documents(["second text"], ids=["d1"], metadata=[{"v": 2}])

        assert await store.count() == 1
        rows = await store.get_vectors(["d1"])
        assert (rows[0][1] or {}).get("v") == 2
    finally:
        await _drop(store)
