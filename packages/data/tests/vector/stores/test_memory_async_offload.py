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

import asyncio
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


async def test_save_to_dirless_persist_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save() must tolerate a persist_path with no directory component.

    A bare filename (``"store.pkl"``) has no parent dir, so
    ``os.path.dirname`` returns ``""`` and ``os.makedirs("", exist_ok=True)``
    raises ``FileNotFoundError``. FAILS pre-fix (the unguarded makedirs);
    PASSES once ``_save_to_disk`` skips makedirs when there is no parent
    dir (parity with FaissVectorStore).
    """
    monkeypatch.chdir(tmp_path)
    store = _store("store.pkl")  # relative, dir-less
    await store.initialize()
    await store.add_vectors([_vec(1)], ids=["a"])

    await store.save()  # pre-fix: FileNotFoundError from makedirs("")

    assert (tmp_path / "store.pkl").exists()


async def test_save_is_race_free_under_concurrent_mutation(
    tmp_path: Path,
) -> None:
    """A concurrent add_vectors must not corrupt an in-flight save().

    Targets the offload race: pre-fix ``_save_to_disk`` ran
    ``{k: v.tolist() for k, v in self.vectors.items()}`` on the worker
    thread while iterating the *live* dict, so a concurrent ``add_vectors``
    on the loop during that iteration could raise ``RuntimeError:
    dictionary changed size during iteration`` in the worker (propagating
    to the awaited save()). Reproducing that crash is inherently
    probabilistic — it depends on the worker iterating while the loop
    mutates — so a large corpus + many concurrent adds widens the window;
    it failed reliably against the pre-fix code in practice but the
    structure does not *guarantee* the interleave.

    The deterministic guarantee this test pins is the snapshot semantics:
    after the fix, the persisted file reflects the state captured when
    save() was called, never a partially-mutated mix, so the rows added
    after the snapshot are absent.
    """
    path = str(tmp_path / "store.pkl")
    store = _store(path)
    await store.initialize()

    # Large initial corpus so the worker's dict comprehension spans many
    # iterations, widening the window for a concurrent mutation to land.
    initial = 4000
    await store.add_vectors(
        [_vec(i) for i in range(initial)],
        ids=[f"v{i}" for i in range(initial)],
    )

    # Start the offloaded save, then let it reach the to_thread dispatch
    # (the worker is now serializing the snapshot/live dict). While it runs,
    # hammer the same dict with concurrent adds on the loop thread.
    save_task = asyncio.create_task(store.save())
    await asyncio.sleep(0)
    for i in range(2000):
        await store.add_vectors([_vec(initial + i)], ids=[f"x{i}"])

    await save_task  # pre-fix: RuntimeError(dictionary changed size...) here

    # The persisted state is the save()-time snapshot — the ``x`` rows added
    # after the snapshot are absent.
    reader = _store(path)
    await reader.initialize()
    assert len(reader.vectors) == initial
    assert "v0" in reader.vectors
    assert "x0" not in reader.vectors
