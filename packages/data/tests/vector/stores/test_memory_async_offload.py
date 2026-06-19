"""Async-correctness tests for ``MemoryVectorStore``'s persist-path offload.

``MemoryVectorStore.save`` / ``load`` persist the in-memory vectors to a
pickle file. Run on the event loop, the ``open()`` + ``pickle`` disk I/O
(and the ``os.path.exists`` guard ``initialize`` used to do before
``load``) stalls every other task for the duration of the write/read.

Each test wraps a single awaited persist call in :func:`assert_no_blocking`.
Against the pre-offload code these FAIL with ``blockbuster.BlockingError``
(the disk syscall runs on the loop); after the ``asyncio.to_thread``
offload they PASS. The round-trip test guards the real regression risk —
that offloading must still persist and reload the vectors, metadata, and
timestamps. No service needed; this is local disk I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

_DIM = 8


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(_DIM, dtype=np.float32)


def _store(persist_path: str) -> MemoryVectorStore:
    return MemoryVectorStore(
        {"dimensions": _DIM, "metric": "cosine", "persist_path": persist_path}
    )


@requires_blockbuster
async def test_save_does_not_block(tmp_path: Path) -> None:
    """save() must not run its open()+pickle.dump on the event loop.

    FAILS pre-offload (blockbuster trips on the ``open()``); PASSES once
    the write is offloaded via ``to_thread``.
    """
    store = _store(str(tmp_path / "store.pkl"))
    await store.initialize()
    await store.add_vectors([_vec(1), _vec(2)], ids=["a", "b"])
    with assert_no_blocking():
        await store.save()


@requires_blockbuster
async def test_load_does_not_block(tmp_path: Path) -> None:
    """load() must not run its open()+pickle.load on the event loop."""
    path = str(tmp_path / "store.pkl")
    writer = _store(path)
    await writer.initialize()
    await writer.add_vectors([_vec(1)], ids=["a"])
    await writer.save()

    reader = _store(path)
    with assert_no_blocking():
        await reader.load()


@requires_blockbuster
async def test_initialize_does_not_block(tmp_path: Path) -> None:
    """initialize() must not stat the persist path on the loop.

    The on-loop ``os.path.exists`` guard was removed; ``load`` self-guards
    on the file's existence inside the ``to_thread`` worker.
    """
    path = str(tmp_path / "store.pkl")
    writer = _store(path)
    await writer.initialize()
    await writer.add_vectors([_vec(1)], ids=["a"])
    await writer.save()

    reader = _store(path)
    with assert_no_blocking():
        await reader.initialize()


async def test_persist_round_trip(tmp_path: Path) -> None:
    """Saved vectors + metadata + timestamps survive a save/load cycle."""
    path = str(tmp_path / "store.pkl")
    writer = _store(path)
    await writer.initialize()
    await writer.add_vectors(
        [_vec(1), _vec(2)], ids=["a", "b"], metadata=[{"k": "v"}, {"k": "w"}]
    )
    await writer.save()

    reader = _store(path)
    await reader.initialize()  # loads from disk via the offloaded load()

    (vec_a, meta_a), = await reader.get_vectors(["a"], include_timestamps=True)
    assert vec_a is not None
    np.testing.assert_array_almost_equal(vec_a, writer.vectors["a"])
    assert meta_a is not None and meta_a["k"] == "v"
    # The persisted timestamps survive the round-trip.
    assert "a" in reader.timestamps and "b" in reader.timestamps
