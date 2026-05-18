"""Non-scalar metadata handling for ``ChromaVectorStore``.

chromadb's metadata contract is scalar-only (str/int/float/bool). It
rejects an empty/``None`` metadata dict outright and — the dangerous
case — *silently accepts* a list-valued metadata value, then corrupts
it: the list bleeds positionally across unrelated collections that
share chromadb's process-wide in-memory ``System``. That surfaced as
the originally-reported ``pytest-randomly`` flake where a ``tags`` key
from one test contaminated another test's ``metadata_fields()``, and is
a real production defect for any process running more than one
in-memory Chroma store.

``ChromaVectorStore`` encodes every non-scalar metadata value behind a
reversible JSON sentinel so chromadb only ever stores scalars, while
the cross-backend round-trip contract (Memory/FAISS preserve real
list/dict values) still holds. These tests exercise the real
``ChromaVectorStore`` (no mocks) and fail without that encoding.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataknobs_common.testing import is_chromadb_available, requires_chromadb

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore


@requires_chromadb
@pytest.mark.asyncio
async def test_list_metadata_does_not_bleed_from_deleted_collection() -> None:
    """A deleted collection's list metadata must not leak into a later store.

    Mirrors the originally-reported flake: store A (a ``tags`` list) is
    fully torn down, then an independent store B reports
    ``metadata_fields()`` — ``tags`` must not appear.
    """
    a = ChromaVectorStore({"dimensions": 4, "collection_name": "bleed_src"})
    await a.initialize()
    await a.add_vectors(
        np.random.rand(3, 4).astype(np.float32),
        ids=["a", "b", "c"],
        metadata=[
            {"domain_id": "x", "tags": ["red"]},
            {"domain_id": "x", "tags": ["blue"]},
            {"domain_id": "y", "tags": ["red"]},
        ],
    )
    # Explicit teardown — the collection is gone before B exists.
    a.client.delete_collection(name=a.collection_name)
    await a.close()

    b = ChromaVectorStore(
        {"dimensions": 384, "metric": "cosine", "collection_name": "bleed_dst"}
    )
    await b.initialize()
    try:
        await b.add_vectors(
            np.random.rand(3, 384).astype(np.float32),
            ids=["1", "2", "3"],
            metadata=[
                {"headings": "A", "source": "doc.md"},
                {"headings": "B", "category": "test"},
                {"author": "alice"},
            ],
        )
        assert await b.metadata_fields() == {
            "headings",
            "source",
            "category",
            "author",
        }
    finally:
        await b.close()


@requires_chromadb
@pytest.mark.asyncio
async def test_list_metadata_does_not_bleed_across_concurrent_stores() -> None:
    """Two live in-memory stores must not see each other's list metadata."""
    a = ChromaVectorStore({"dimensions": 4, "collection_name": "iso_a"})
    b = ChromaVectorStore({"dimensions": 4, "collection_name": "iso_b"})
    await a.initialize()
    await b.initialize()
    try:
        await a.add_vectors(
            np.random.rand(2, 4).astype(np.float32),
            ids=["a1", "a2"],
            metadata=[{"tags": ["a"]}, {"tags": ["a"]}],
        )
        await b.add_vectors(
            np.random.rand(2, 4).astype(np.float32),
            ids=["b1", "b2"],
            metadata=[{"labels": ["b"]}, {"labels": ["b"]}],
        )
        assert await a.metadata_fields() == {"tags"}
        assert await b.metadata_fields() == {"labels"}
    finally:
        await a.close()
        await b.close()


@requires_chromadb
@pytest.mark.asyncio
async def test_nonscalar_metadata_round_trips() -> None:
    """Non-empty lists, empty lists, and dicts survive write -> read.

    The cross-backend contract: Memory/FAISS preserve real list/dict
    values, so Chroma must too despite chromadb's scalar-only store.
    """
    store = ChromaVectorStore(
        {"dimensions": 4, "collection_name": "roundtrip"}
    )
    await store.initialize()
    try:
        meta = {
            "tags": ["red", "blue"],
            "empty": [],
            "nested": {"k": [1, 2]},
            "scalar": "plain",
            "n": 7,
        }
        await store.add_vectors(
            np.random.rand(1, 4).astype(np.float32),
            ids=["r1"],
            metadata=[meta],
        )
        (_, got), = await store.get_vectors(["r1"])
        assert got == meta
    finally:
        await store.close()
