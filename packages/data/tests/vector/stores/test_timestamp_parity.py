"""Cross-backend parity tests for timestamp exposure.

The shared timestamp abstraction exists so consumers can runtime-swap between
vector store backends without behavioral surprises. These tests
parameterize the same body over every shipping backend and assert
identical timestamp semantics:

- ``_created_at`` / ``_updated_at`` are present when
  ``include_timestamps=True`` and absent by default.
- Upsert preserves ``_created_at`` and advances ``_updated_at``.

Every in-tree backend runs: memory, faiss and chroma need no
services, and pgvector joins them when one is reachable. That is the
point of the suite — a backend that implements the feature without
appearing here is a divergence nothing would catch.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    is_package_available,
    requires_real_postgres,
)

from dataknobs_data.testing import vector as _vector
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if is_package_available("asyncpg"):
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


# requires_real_postgres is exactly the three terms this list assembled by
# hand: a reachable server, TEST_POSTGRES=true, and asyncpg installed.
_pgvector_marks = [requires_real_postgres]


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common``
    fixture (pre-drop + teardown drop + pgvector-extension ensure live
    there now). ``metric`` is preserved at ``cosine`` to keep behavior
    byte-identical to the prior hand-rolled config.
    """
    gen = make_pgvector_test_table("test_parity_", dimensions=4)
    cfg = next(gen)
    cfg["metric"] = "cosine"
    try:
        yield cfg
    finally:
        gen.close()


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed"),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(not is_chromadb_available(), reason="chromadb not installed"),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def any_vector_store(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Any]:
    """Yield a freshly-initialized VectorStore for each backend param."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": f"test_parity_{uuid.uuid4().hex[:8]}",
            }
        )
        await store.initialize()
        try:
            yield store
        finally:
            try:
                store.client.delete_collection(name=store.collection_name)
            except Exception as exc:  # pragma: no cover - teardown best effort
                logging.getLogger(__name__).warning(
                    "Chroma teardown failed for collection %r: %s",
                    store.collection_name,
                    exc,
                )
            await store.close()
    elif backend == "pgvector":
        store = PgVectorStore(request.getfixturevalue("pgvector_config"))
        await store.initialize()
        try:
            yield store
        finally:
            # The shared make_pgvector_test_table fixture owns the
            # table drop (it tears down after this store is closed,
            # since any_vector_store depends on pgvector_config).
            await store.close()
    else:
        pytest.fail(f"Unknown backend param: {backend}")


@pytest.mark.asyncio
async def test_timestamps_present_and_ordered(any_vector_store: Any) -> None:
    """include_timestamps=True exposes _created_at / _updated_at on every backend."""
    vec = _vector(4)
    await any_vector_store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

    results = await any_vector_store.get_vectors(["t1"], include_timestamps=True)
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" in meta
    assert "_updated_at" in meta
    assert meta["_created_at"] is not None
    assert meta["_updated_at"] is not None
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_timestamps_absent_by_default(any_vector_store: Any) -> None:
    """Default get_vectors() omits timestamp keys on every backend."""
    vec = _vector(4)
    await any_vector_store.add_vectors([vec], ids=["t1"], metadata=[{"k": "v"}])

    results = await any_vector_store.get_vectors(["t1"])
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" not in meta
    assert "_updated_at" not in meta
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_upsert_refreshes_updated_consistently(
    any_vector_store: Any,
) -> None:
    """Second add_vectors with same id: created preserved, updated advances."""
    vec1 = _vector(4)
    vec2 = _vector(4, seed=1)

    await any_vector_store.add_vectors([vec1], ids=["t1"])
    first_results = await any_vector_store.get_vectors(["t1"], include_timestamps=True)
    first = first_results[0][1]
    assert first is not None

    # Sleep longer than the backend clock resolution so updated_at
    # must strictly advance under ISO-string lexicographic comparison.
    await asyncio.sleep(0.05)

    await any_vector_store.add_vectors([vec2], ids=["t1"])
    second_results = await any_vector_store.get_vectors(["t1"], include_timestamps=True)
    second = second_results[0][1]
    assert second is not None

    assert second["_created_at"] == first["_created_at"], (
        "created_at must not change on upsert (backend-dependent parity violation)"
    )
    assert second["_updated_at"] > first["_updated_at"], "updated_at must advance on upsert"
