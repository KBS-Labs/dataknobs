"""Reproduce-first async-correctness tests for ``FaissVectorStore`` I/O.

``FaissVectorStore.save`` / ``load`` perform blocking disk I/O on the
event loop: ``faiss.write_index`` / ``read_index``, ``open`` +
``pickle.dump`` / ``pickle.load`` for the metadata side-car, and
``os.makedirs``. The fix offloads each method's disk body via
:func:`asyncio.to_thread`.

In-memory ``add`` / ``search`` are CPU-bound C++ (the GIL is released
inside FAISS) and are explicitly OUT of scope — offloading pure compute
to a thread buys nothing. Only the disk I/O is offloaded.

Each test wraps a single awaited ``save`` / ``load`` in
:func:`assert_no_blocking`. Against the pre-offload code these FAIL with
``blockbuster.BlockingError`` (the disk syscall runs on the loop); after
the offload they PASS. ``load`` also asserts the round-trip is intact.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataknobs_common.testing import (
    assert_no_blocking,
    is_faiss_available,
    requires_blockbuster,
)

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

requires_faiss = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss not installed"
)

pytestmark = [pytest.mark.asyncio, requires_faiss, requires_blockbuster]

_DIM = 16


def _store(persist_path: str) -> FaissVectorStore:
    return FaissVectorStore(
        {
            "dimensions": _DIM,
            "metric": "euclidean",
            "index_type": "flat",
            "persist_path": persist_path,
        }
    )


def _vecs(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, _DIM), dtype=np.float32)


async def test_save_does_not_block(tmp_path) -> None:
    store = _store(str(tmp_path / "faiss.index"))
    await store.initialize()
    await store.add_vectors(
        _vecs(5),
        ids=[str(i) for i in range(5)],
        metadata=[{"i": i} for i in range(5)],
    )
    with assert_no_blocking():
        await store.save()


async def test_load_does_not_block_and_round_trips(tmp_path) -> None:
    path = str(tmp_path / "faiss.index")
    src = _store(path)
    await src.initialize()
    await src.add_vectors(
        _vecs(5),
        ids=[str(i) for i in range(5)],
        metadata=[{"i": i} for i in range(5)],
    )
    await src.save()

    # Fresh store on the same path: load() runs during initialize().
    dst = _store(path)
    with assert_no_blocking():
        await dst.load()

    assert dst.index.ntotal == 5
    assert len(dst.id_map) == 5
    (vec, meta), = await dst.get_vectors(["3"])
    assert vec is not None
    assert meta == {"i": 3}
