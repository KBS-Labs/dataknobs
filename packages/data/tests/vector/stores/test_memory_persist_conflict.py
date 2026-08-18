"""Two ``MemoryVectorStore`` instances over one ``persist_path``.

The same defect the FAISS suite next door covers, in the other store
that persists by serializing its whole in-memory state over one file:
``save()`` writes everything this instance holds, so two instances whose
lifetimes overlap ended with the last writer's snapshot — which never
saw the other's rows — replacing the file outright. The earlier writer's
rows were gone from disk entirely, with no error and nothing in the log.

Discovering it in FAISS was incidental. Nothing about the hazard is
FAISS-shaped: it belongs to the whole-state rewrite, not to the format
written, which is why the guard lives on ``VectorStoreBase`` and both
stores use it.
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataknobs_common.exceptions import ConcurrencyError

from dataknobs_data.vector.stores.memory import MemoryVectorStore

pytestmark = pytest.mark.asyncio

DIMENSIONS = 4


def _vectors(count: int, seed: int) -> np.ndarray:
    """``count`` distinct rows, deterministic per ``seed``."""
    rows = np.zeros((count, DIMENSIONS), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 1] = [0.01 * (seed * 100 + i) for i in range(count)]
    return rows


async def _open(persist: Path) -> MemoryVectorStore:
    """An initialized store bound to ``persist`` (loading it if present)."""
    store = MemoryVectorStore(
        {"dimensions": DIMENSIONS, "metric": "cosine", "persist_path": str(persist)}
    )
    await store.initialize()
    return store


async def _ingest(store: MemoryVectorStore, prefix: str, count: int, seed: int) -> None:
    await store.add_vectors(
        _vectors(count, seed),
        ids=[f"{prefix}{i}" for i in range(count)],
        metadata=[{"owner": prefix} for _ in range(count)],
    )


async def _count_on_disk(persist: Path) -> int:
    """Rows a fresh reader finds in the persisted store."""
    reader = await _open(persist)
    try:
        return int(await reader.count())
    finally:
        await reader.close()


async def test_overlapping_save_raises_instead_of_clobbering(tmp_path: Path) -> None:
    """The second writer of a shared file raises rather than overwriting.

    Pre-fix this passed silently and the assertion below found five rows
    where nine had been added — the four the first writer persisted were
    simply gone.
    """
    persist = tmp_path / "shared.pkl"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.save()
    with pytest.raises(ConcurrencyError):
        await second.save()

    assert await _count_on_disk(persist) == 5


async def test_reader_does_not_clobber_the_writer(tmp_path: Path) -> None:
    """An instance that only read must not cost the writer its rows.

    ``close()`` persists, so a store opened purely to read is a writer at
    teardown unless something says otherwise — and once the guard exists,
    that write is what makes the real writer's save raise.
    """
    persist = tmp_path / "shared.pkl"
    writer = await _open(persist)
    await _ingest(writer, "writer", 5, seed=1)

    reader = await _open(persist)
    assert await reader.count() == 0
    await reader.close()

    await writer.close()

    assert await _count_on_disk(persist) == 5


async def test_close_releases_the_store_even_when_the_save_is_refused(tmp_path: Path) -> None:
    """A refused save must not leave the store stuck open."""
    persist = tmp_path / "shared.pkl"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.close()
    with pytest.raises(ConcurrencyError):
        await second.close()

    assert second._initialized is False
    await second.close()


async def test_force_is_the_way_out_of_a_refusal(tmp_path: Path) -> None:
    """``save(force=True)`` overwrites deliberately, accepting the loss."""
    persist = tmp_path / "shared.pkl"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.save()
    with pytest.raises(ConcurrencyError):
        await second.save()

    await second.save(force=True)

    assert await _count_on_disk(persist) == 4
    await second.close()
    await first.close()


async def test_sequential_lifetimes_still_append(tmp_path: Path) -> None:
    """Non-overlapping instances keep building on each other.

    The shape almost every consumer actually has, and it must stay
    unaffected by the guard.
    """
    persist = tmp_path / "sequential.pkl"

    first = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await first.close()

    second = await _open(persist)
    assert await second.count() == 5
    await _ingest(second, "second", 4, seed=2)
    await second.close()

    assert await _count_on_disk(persist) == 9


async def test_single_writer_repeated_saves_ok(tmp_path: Path) -> None:
    """One instance saving repeatedly does not trip its own check."""
    persist = tmp_path / "single.pkl"
    store = await _open(persist)
    try:
        await _ingest(store, "first", 3, seed=1)
        await store.save()
        await _ingest(store, "second", 2, seed=2)
        await store.save()
        await store.save()
    finally:
        await store.close()

    assert await _count_on_disk(persist) == 5


async def test_overlapping_saves_on_one_instance_do_not_conflict(tmp_path: Path) -> None:
    """A store must not raise a concurrency error against itself."""
    persist = tmp_path / "concurrent.pkl"
    store = await _open(persist)
    try:
        for round_ in range(5):
            await _ingest(store, f"r{round_}", 20, seed=round_)
            await asyncio.gather(store.save(), store.save(), store.save())
    finally:
        await store.close()

    assert await _count_on_disk(persist) == 100


async def test_failed_write_leaves_the_previous_state_loadable(tmp_path: Path) -> None:
    """A pickle that fails midway must not truncate the persisted file.

    Written directly over the target, a value ``pickle`` cannot
    serialize leaves a partial file behind — and the next reader gets
    ``EOFError`` rather than the rows the last good save had put there.
    """
    persist = tmp_path / "partial.pkl"
    store = await _open(persist)
    await _ingest(store, "good", 2, seed=1)
    await store.save()

    await store.add_vectors(
        _vectors(1, seed=2),
        ids=["unpicklable"],
        metadata=[{"callback": lambda: None}],
    )
    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        await store.save()

    assert await _count_on_disk(persist) == 2

    # And this instance can still save once the offending row is gone.
    await store.delete_vectors(["unpicklable"])
    await store.save()
    assert await _count_on_disk(persist) == 2
    await store.close()


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda s: s.add_vectors(_vectors(2, seed=7), ids=["x0", "x1"]), 5, id="add_vectors"
        ),
        pytest.param(lambda s: s.delete_vectors(["first0", "first1"]), 1, id="delete_vectors"),
        pytest.param(
            lambda s: s.update_metadata(["first0"], [{"owner": "changed"}]),
            3,
            id="update_metadata",
        ),
        pytest.param(
            lambda s: s.update_metadata_where({"owner": "first"}, {"owner": "changed"}),
            3,
            id="update_metadata_where",
        ),
        pytest.param(lambda s: s.clear(filter={"owner": "first"}), 0, id="clear_filtered"),
        pytest.param(lambda s: s.clear(), 0, id="clear_all"),
    ],
)
async def test_every_mutator_survives_close(tmp_path: Path, mutate: Any, expected: int) -> None:
    """Each mutator's change reaches disk through ``close()`` alone.

    ``close()`` persists only a store that was mutated, which is what
    keeps a reader from clobbering a writer. The cost of that is a list
    of mutators, and a mutator missing from it loses its change silently
    on teardown. This is that list, asserted through the file rather
    than through the flag.
    """
    persist = tmp_path / "mutator.pkl"
    store = await _open(persist)
    await _ingest(store, "first", 3, seed=1)
    await store.save()

    await mutate(store)
    await store.close()

    reader = await _open(persist)
    try:
        assert await reader.count() == expected
        if expected == 3:
            assert await reader.count(filter={"owner": "changed"}) >= 1
    finally:
        await reader.close()
