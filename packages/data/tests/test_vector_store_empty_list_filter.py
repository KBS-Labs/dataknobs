"""Cross-backend conformance test for the empty-list filter contract.

An empty-list filter value is an unsatisfiable predicate:
``{key: []}`` matches no record on any vector-store backend (see the
``_match_metadata_filter`` docstring in ``common.py``). This invariant is
enforced by four independent code paths — the shared
``_match_metadata_filter`` (memory, faiss), chroma's empty-list
short-circuit, and pgvector's literal ``FALSE`` predicate — with no
shared test guarding it across backends until now.

It is load-bearing: a consumer (e.g. ``VectorMemory.clear()``) AND-merges
a deliberate ``{key: []}`` contradiction into a caller filter so a
cross-tenant clear becomes a no-op rather than wiping another tenant's
rows. A regression in any one backend's filter translation — or an
omission in a future backend — would silently break that tenant
isolation, so this parametrized test runs against every available
backend.

Construction-only backends (memory) always run; optional-dependency
backends are gated by markers. pgvector requires a running PostgreSQL
with the pgvector extension (``TEST_POSTGRES=true``).
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest
from dataknobs_common.testing import requires_chromadb, requires_faiss

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dataknobs_data.vector.stores.base import VectorStore

DIMS = 8

_RUN_PG = os.environ.get("TEST_POSTGRES", "").lower() == "true"


def _pg_connection_string() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _seed_vectors() -> tuple[np.ndarray, list[str], list[dict]]:
    """Three rows: one scalar-valued ``tag``, two list-valued ``tags``.

    Covers both the scalar-metadata and list-metadata sides of the
    four-quadrant filter so the empty-list assertion is exercised against
    each.
    """
    vectors = np.eye(3, DIMS, dtype=np.float32)
    ids = [f"row-{i}" for i in range(3)]
    metadata = [
        {"tag": "alpha", "tags": ["x", "y"]},
        {"tag": "beta", "tags": ["y", "z"]},
        {"tag": "alpha", "tags": ["z"]},
    ]
    return vectors, ids, metadata


async def _make_memory() -> VectorStore:
    return MemoryVectorStore({"dimensions": DIMS})


async def _make_faiss() -> VectorStore:
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

    return FaissVectorStore({"dimensions": DIMS, "index_type": "flat"})


async def _make_chroma() -> VectorStore:
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

    return ChromaVectorStore(
        {
            "dimensions": DIMS,
            "collection_name": f"empty_list_{uuid.uuid4().hex[:8]}",
        }
    )


async def _make_pgvector() -> VectorStore:
    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    return PgVectorStore(
        {
            "connection_string": _pg_connection_string(),
            "dimensions": DIMS,
            "table_name": f"empty_list_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
        }
    )


_BACKENDS = [
    pytest.param(_make_memory, id="memory"),
    pytest.param(_make_faiss, id="faiss", marks=requires_faiss),
    pytest.param(_make_chroma, id="chroma", marks=requires_chromadb),
    pytest.param(
        _make_pgvector,
        id="pgvector",
        marks=pytest.mark.skipif(
            not _RUN_PG,
            reason="pgvector requires TEST_POSTGRES=true and a running instance",
        ),
    ),
]


@pytest.fixture(params=_BACKENDS)
async def seeded_store(request) -> AsyncIterator[VectorStore]:
    """Yield an initialized store seeded with the conformance corpus."""
    store = await request.param()
    await store.initialize()
    vectors, ids, metadata = _seed_vectors()
    await store.add_vectors(vectors, ids=ids, metadata=metadata)
    try:
        yield store
    finally:
        # Best-effort cleanup: drop all rows then close (drops the
        # pgvector temp table's data; the unique table name avoids
        # cross-run collision).
        try:
            await store.clear()
        finally:
            await store.close()


class TestEmptyListFilterMatchesNothing:
    """``{key: []}`` is unsatisfiable on every backend."""

    @pytest.mark.parametrize("key", ["tag", "tags"])
    async def test_count_empty_list_is_zero(
        self, seeded_store: VectorStore, key: str
    ) -> None:
        # Scalar-valued (``tag``) and list-valued (``tags``) metadata both
        # yield zero under an empty-list filter.
        assert await seeded_store.count({key: []}) == 0

    @pytest.mark.parametrize("key", ["tag", "tags"])
    async def test_search_empty_list_is_zero(
        self, seeded_store: VectorStore, key: str
    ) -> None:
        query = np.ones(DIMS, dtype=np.float32)
        results = await seeded_store.search(query, k=10, filter={key: []})
        assert results == []

    async def test_clear_empty_list_removes_nothing(
        self, seeded_store: VectorStore
    ) -> None:
        before = await seeded_store.count()
        assert before == 3  # sanity: corpus is present
        await seeded_store.clear(filter={"tag": []})
        assert await seeded_store.count() == before

    async def test_clear_real_value_removes_rows_counterexample(
        self, seeded_store: VectorStore
    ) -> None:
        # The empty-list no-op assertion is not vacuously passing: a real
        # filter value DOES remove the matching rows.
        await seeded_store.clear(filter={"tag": "alpha"})
        remaining = await seeded_store.count()
        assert remaining == 1  # only the "beta" row survives
