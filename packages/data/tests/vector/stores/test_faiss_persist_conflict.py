"""Two ``FaissVectorStore`` instances over one ``persist_path``.

``save()`` serializes the whole in-memory index over the file, so two
instances whose lifetimes overlap used to end with the last writer's
snapshot — which never saw the other's rows — replacing the file
outright. The earlier writer's rows were gone from disk entirely, with
no error and nothing in the log.

Nothing in the config or the API marks ``persist_path`` as an exclusive
resource, and the same two-instances-one-store shape is correct on
``PgVectorStore``, so this is not misuse to be documented away. A store
now refuses to overwrite a file that changed underneath it. Sequential
lifetimes are unaffected and keep appending, which is the shape almost
every consumer actually has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataknobs_common.exceptions import ConcurrencyError
from dataknobs_common.testing import is_faiss_available

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

requires_faiss = pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed")

pytestmark = [pytest.mark.asyncio, requires_faiss]

DIMENSIONS = 4


def _vectors(count: int, seed: int) -> np.ndarray:
    """``count`` distinct unit-ish rows, deterministic per ``seed``."""
    rows = np.zeros((count, DIMENSIONS), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 1] = [0.01 * (seed * 100 + i) for i in range(count)]
    return rows


async def _open(persist: Path) -> Any:
    """An initialized store bound to ``persist`` (loading it if present)."""
    store = FaissVectorStore(
        {"dimensions": DIMENSIONS, "metric": "cosine", "persist_path": str(persist)}
    )
    await store.initialize()
    return store


async def _ingest(store: Any, prefix: str, count: int, seed: int) -> None:
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


@pytest.mark.parametrize("pre_existing", [False, True], ids=["fresh", "loaded"])
async def test_overlapping_save_raises_instead_of_clobbering(
    tmp_path: Path, pre_existing: bool
) -> None:
    """The second writer of a shared file raises rather than overwriting.

    Run twice: on a path with nothing at it yet, and on one both
    instances loaded from. Both reach the same place — one instance
    holding a snapshot that predates another instance's write.
    """
    persist = tmp_path / "shared.index"
    if pre_existing:
        seeded = await _open(persist)
        await _ingest(seeded, "seed", 2, seed=9)
        await seeded.close()

    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.save()
    with pytest.raises(ConcurrencyError):
        await second.save()

    # The refusal is worth having only if it protects the rows it
    # declined to overwrite.
    expected = 5 + (2 if pre_existing else 0)
    assert await _count_on_disk(persist) == expected


async def test_overlapping_close_raises_instead_of_clobbering(tmp_path: Path) -> None:
    """``close()`` is the path a consumer reaches without calling ``save``.

    Ordinary teardown persists, so a consumer who never writes a
    ``save()`` of their own still lost the other instance's rows.
    """
    persist = tmp_path / "shared.index"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.close()
    with pytest.raises(ConcurrencyError):
        await second.close()

    assert await _count_on_disk(persist) == 5


async def test_sequential_lifetimes_still_append(tmp_path: Path) -> None:
    """Non-overlapping instances keep building on each other.

    ``initialize()`` loads whatever is on disk first, so a later
    instance starts from the earlier one's rows. This is the shape
    almost every consumer has and it must stay unaffected.
    """
    persist = tmp_path / "sequential.index"

    first = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await first.close()

    second = await _open(persist)
    assert await second.count() == 5
    await _ingest(second, "second", 4, seed=2)
    await second.close()

    assert await _count_on_disk(persist) == 9


async def test_single_writer_repeated_saves_ok(tmp_path: Path) -> None:
    """One instance saving repeatedly does not trip its own check.

    A staleness check that compares against the file as it was when
    loaded, rather than as this instance last left it, fails here on the
    second save.
    """
    persist = tmp_path / "single.index"
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
