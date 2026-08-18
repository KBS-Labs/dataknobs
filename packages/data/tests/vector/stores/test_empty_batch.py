"""The empty-batch contract for ``add_vectors``, across every backend.

An empty batch is something a caller *produces*, not something they set
out to write: a list comprehension that filtered everything out, a
chunker handed a blank document, a retry loop whose work is already
done. Nothing in the ABC said what happens then, and each backend
answered differently — Memory minted an id for a zero-dimension vector
and grew by one row, FAISS raised a bare ``AssertionError`` from inside
the C++ layer with no message, and Chroma raised either ``ValueError``
or ``IndexError`` depending on whether the caller passed ``[]`` or an
empty ndarray.

So the same guard-free consumer code corrupted one store, crashed on
two others with errors that name nothing actionable, and the only way
to write portable code was an ``if items:`` at every call site.

The contract pinned here is the one every collection-shaped API uses:
**empty in, empty out, no-op**. Both spellings of empty are accepted,
because ``np.array([])`` and ``[]`` are the same intent and a caller
building a batch numerically will produce the former.

The store fixture here is deliberately *unseeded* — the assertion is
about a store's row count not moving, which is clearest when it starts
at zero.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    is_package_available,
    requires_real_postgres,
)

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if is_package_available("asyncpg"):
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


_pgvector_marks = [requires_real_postgres]


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common`` fixture."""
    gen = make_pgvector_test_table("test_empty_batch_", dimensions=4)
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
async def empty_store(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Any]:
    """An initialized, empty store for each backend."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {"dimensions": 4, "collection_name": f"test_empty_batch_{uuid.uuid4().hex[:8]}"}
        )
    elif backend == "pgvector":
        store = PgVectorStore(request.getfixturevalue("pgvector_config"))
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        yield store
    finally:
        if backend == "chroma":
            store.client.delete_collection(name=store.collection_name)
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["list", "ndarray"])
async def test_adding_an_empty_batch_writes_nothing(empty_store: Any, spelling: str) -> None:
    """No ids minted, no rows added, no exception."""
    batch: Any = [] if spelling == "list" else np.array([], dtype=np.float32)

    assert await empty_store.add_vectors(batch) == []
    assert await empty_store.count() == 0


@pytest.mark.asyncio
async def test_an_empty_batch_does_not_disturb_existing_rows(empty_store: Any) -> None:
    """The no-op is a no-op on a store that already holds rows.

    Distinct from the case above: Memory's failure was to *append* a
    fabricated row, which an empty store shows as ``count() == 1`` but
    which matters because it corrupts a store that was otherwise fine.
    """
    await empty_store.add_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), ids=["r1"])

    assert await empty_store.add_vectors([]) == []
    assert await empty_store.count() == 1
    assert (await empty_store.get_vectors(["r1"]))[0][0] is not None


@pytest.mark.asyncio
async def test_the_id_keyed_verbs_accept_an_empty_id_list(empty_store: Any) -> None:
    """An empty id list is the empty answer, not an error.

    The same "a comprehension filtered everything out" case
    ``add_vectors`` handles, reached through the id-keyed verbs instead:
    a caller that assembled a list of ids to fetch or delete and found
    none is asking a well-formed question with an empty answer.

    Chroma alone raised here, because chromadb's ``validate_ids``
    rejects an empty list before the query runs — so a consumer whose
    code was correct on three backends crashed on the fourth after a
    config change.
    """
    await empty_store.add_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), ids=["r1"])

    assert await empty_store.get_vectors([]) == []
    assert await empty_store.delete_vectors([]) == 0
    assert await empty_store.update_metadata([], []) == 0

    # The store is untouched by any of them.
    assert await empty_store.count() == 1


@pytest.mark.asyncio
async def test_adding_no_documents_writes_nothing(empty_store: Any) -> None:
    """``add_documents`` is the sibling write path and answers the same.

    Chroma-only surface, so the other backends have nothing to compare
    against — but the contract is the store's, not the backend's, and a
    chunker handed a blank document produces this call on whichever
    backend is configured.
    """
    if not hasattr(empty_store, "add_documents"):
        pytest.skip("add_documents is a Chroma-only surface")

    assert await empty_store.add_documents([]) == []
    assert await empty_store.count() == 0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.array(5.0), id="zero-d-array"),
        pytest.param(np.float32(1.0), id="numpy-scalar"),
    ],
)
def test_a_zero_d_input_is_not_an_empty_batch(value: Any) -> None:
    """The emptiness predicate is total over what a caller can pass.

    A 0-d array has no length, so the check raised ``TypeError: len() of
    unsized object`` from inside itself — an error about lengths,
    surfacing from a question about emptiness, before any backend got
    the chance to say what shape it wanted. Answering ``False`` hands
    the input on to the dimension validation that can describe it.
    """
    assert MemoryVectorStore({"dimensions": 4})._is_empty_batch(value) is False
