"""Async-correctness tests for ``ChromaVectorStore``'s ``to_thread`` offload.

Every ``ChromaVectorStore`` async method drives the synchronous chromadb
client/collection. Run on the event loop, those calls stall it. The fix
offloads each chromadb call via :func:`asyncio.to_thread`.

**Detector blind spot (why only ``initialize`` is reproduce-first).**
chromadb 1.x performs its per-operation persistence in a native (Rust)
core; ``blockbuster`` patches Python-level syscalls only, so an
``add`` / ``query`` / ``get`` against a PersistentClient does *not* trip
the detector even though the native call blocks the loop thread (verified:
a 500-vector persistent add inside ``assert_no_blocking`` passes pre-fix).
The one operation ``blockbuster`` *can* observe is ``initialize`` — it does
a Python-level ``os.stat`` on the persist directory — so
``test_initialize_does_not_block`` is the genuine reproduce-first
red→green test. The remaining ops are still offloaded (native synchronous
I/O blocks the loop regardless of detector visibility) and are guarded
here by functional round-trip tests proving the offload preserves
outcomes.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest
from dataknobs_common.testing import (
    assert_no_blocking,
    is_chromadb_available,
    requires_blockbuster,
)

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

requires_chromadb = pytest.mark.skipif(
    not is_chromadb_available(), reason="chromadb not installed"
)

pytestmark = [pytest.mark.asyncio, requires_chromadb]

_DIM = 8


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(_DIM, dtype=np.float32)


@pytest.fixture
def make_store() -> Iterator[Callable[..., ChromaVectorStore]]:
    """Build isolated ChromaVectorStores with guaranteed teardown.

    Unique per-test collection name (chromadb's in-process client leaks
    a fixed collection name across tests) + teardown that drops every
    created collection.
    """
    created: list[ChromaVectorStore] = []

    def _factory(*, persist_path: str | None = None) -> ChromaVectorStore:
        config: dict[str, object] = {
            "dimensions": _DIM,
            "metric": "cosine",
            "collection_name": f"offload_{uuid4().hex[:8]}",
        }
        if persist_path is not None:
            config["persist_path"] = persist_path
        store = ChromaVectorStore(config)
        created.append(store)
        return store

    yield _factory

    for store in created:
        if store.client is not None:
            with contextlib.suppress(Exception):
                store.client.delete_collection(name=store.collection_name)


# ---------------------------------------------------------------------------
# Reproduce-first — the one op blockbuster can observe (Python-level os.stat)
# ---------------------------------------------------------------------------


@requires_blockbuster
async def test_initialize_does_not_block(make_store, tmp_path) -> None:
    """PersistentClient init must not block the loop.

    FAILS pre-offload (the on-disk sqlite load runs ``os.stat`` on the
    loop) and PASSES once construction is offloaded via ``to_thread``.
    """
    store = make_store(persist_path=str(tmp_path / "chroma"))
    with assert_no_blocking():
        await store.initialize()


# ---------------------------------------------------------------------------
# Functional regression — the offload must preserve outcomes for every op
# (native per-op I/O is undetectable by blockbuster; see module docstring)
# ---------------------------------------------------------------------------


async def test_add_and_search_round_trip(make_store, tmp_path) -> None:
    store = make_store(persist_path=str(tmp_path / "rt"))
    await store.initialize()
    await store.add_vectors(
        [_vec(1), _vec(2)], ids=["a", "b"], metadata=[{"k": "v"}, {"k": "w"}]
    )
    results = await store.search(_vec(1), k=2)
    assert {r[0] for r in results} == {"a", "b"}


async def test_get_vectors_returns_metadata(make_store, tmp_path) -> None:
    store = make_store(persist_path=str(tmp_path / "get"))
    await store.initialize()
    await store.add_vectors([_vec(1)], ids=["a"], metadata=[{"k": "v"}])
    (vec, meta), = await store.get_vectors(["a"])
    assert vec is not None
    assert meta == {"k": "v"}


async def test_count_reflects_adds(make_store, tmp_path) -> None:
    store = make_store(persist_path=str(tmp_path / "count"))
    await store.initialize()
    await store.add_vectors([_vec(1), _vec(2)], ids=["a", "b"])
    assert await store.count() == 2


async def test_delete_vectors_removes(make_store, tmp_path) -> None:
    store = make_store(persist_path=str(tmp_path / "del"))
    await store.initialize()
    await store.add_vectors([_vec(1), _vec(2)], ids=["a", "b"])
    removed = await store.delete_vectors(["a"])
    assert removed == 1
    assert await store.count() == 1


async def test_clear_empties_collection(make_store, tmp_path) -> None:
    store = make_store(persist_path=str(tmp_path / "clear"))
    await store.initialize()
    await store.add_vectors([_vec(1), _vec(2)], ids=["a", "b"])
    await store.clear()
    assert await store.count() == 0
