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

import asyncio
import os

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

# ``requires_blockbuster`` is scoped to the two assert_no_blocking tests
# below — the concurrency guard needs faiss + asyncio, not blockbuster.
pytestmark = [pytest.mark.asyncio, requires_faiss]

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


@requires_blockbuster
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


@requires_blockbuster
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


async def test_save_before_initialize_is_a_noop(tmp_path) -> None:
    """save() on a never-initialized store must not crash.

    The FAISS index is created in initialize(); before that ``self.index``
    is None. FAILS pre-fix (``faiss.write_index(None, ...)`` raises a hard
    C++ error); PASSES once save() short-circuits on a None index. Nothing
    is persisted because the store holds no data yet.
    """
    path = str(tmp_path / "faiss.index")
    store = _store(path)  # no initialize(), no add_vectors

    await store.save()  # pre-fix: faiss.write_index(None) crashes

    assert not os.path.exists(path)


async def test_save_is_race_free_under_concurrent_mutation(tmp_path) -> None:
    """A concurrent add_vectors must not corrupt an in-flight save().

    Targets the offload race: pre-fix ``_save_to_disk`` serialized the
    *live* ``self.index`` (``faiss.write_index``) and pickled the live
    ``id_map`` / ``metadata_store`` / ``vectors`` dicts on the worker
    thread, so a concurrent ``add_with_ids`` + dict mutation on the loop
    could race both (``RuntimeError: dictionary changed size during
    iteration`` on the meta side-car, plus an inconsistent index write).
    As with the memory store, reproducing that crash is probabilistic
    (it depends on the interleave), so a large corpus + many concurrent
    adds widens the window.

    The deterministic guarantee this test pins is the snapshot semantics:
    after the fix, save() snapshots the dicts and deep-clones the index on
    the loop before offloading, so the persisted index + ``.meta`` reflect
    the save()-time state, mutually consistent and excluding the rows added
    afterward.
    """
    path = str(tmp_path / "faiss.index")
    store = _store(path)
    await store.initialize()

    initial = 400
    await store.add_vectors(
        _vecs(initial),
        ids=[str(i) for i in range(initial)],
        metadata=[{"i": i} for i in range(initial)],
    )

    # Start the offloaded save, let it reach the to_thread dispatch, then
    # mutate the same state on the loop while the worker serializes it.
    save_task = asyncio.create_task(store.save())
    await asyncio.sleep(0)
    for i in range(400):
        await store.add_vectors(
            _vecs(1), ids=[f"x{i}"], metadata=[{"i": -1}]
        )

    await save_task  # pre-fix: races (RuntimeError / inconsistent index)

    # The persisted index and metadata are the save()-time snapshot, and
    # mutually consistent — the ``x`` rows added afterward are absent.
    dst = _store(path)
    await dst.initialize()
    assert dst.index.ntotal == initial
    assert len(dst.id_map) == initial
    assert "0" in dst.id_map
    assert "x0" not in dst.id_map
