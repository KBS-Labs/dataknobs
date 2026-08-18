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

import asyncio
import pickle
import shutil
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


@pytest.mark.parametrize("pre_existing", [False, True], ids=["fresh", "loaded"])
async def test_reader_does_not_clobber_the_writer(tmp_path: Path, pre_existing: bool) -> None:
    """An instance that only read must not cost the writer its rows.

    ``close()`` persists, and a store opened purely to read is a writer
    at teardown unless something says otherwise. Its write moves the
    file's identity, so the instance actually holding new rows then finds
    the file changed underneath it, refuses to save, and its rows are
    lost — while the file keeps the reader's older snapshot.

    Both orderings of the same shape: the reader may hold nothing at all
    (``fresh``) or the seeded rows (``loaded``). Neither reader mutated
    anything, so neither has anything to persist.
    """
    persist = tmp_path / "shared.index"
    if pre_existing:
        seeded = await _open(persist)
        await _ingest(seeded, "seed", 2, seed=9)
        await seeded.close()

    writer = await _open(persist)
    await _ingest(writer, "writer", 5, seed=1)

    reader = await _open(persist)
    assert await reader.count() == (2 if pre_existing else 0)
    await reader.close()

    await writer.close()  # pre-fix: ConcurrencyError, and the rows go

    assert await _count_on_disk(persist) == 5 + (2 if pre_existing else 0)


async def test_close_releases_the_store_even_when_the_save_is_refused(tmp_path: Path) -> None:
    """A refused save must not leave the store stuck open.

    Persisting and releasing are separate obligations. With the save
    outside a ``finally`` the failure skips the release, so the store
    still reports itself initialized and every retry re-enters the same
    save and raises again — there is no way to close it.
    """
    persist = tmp_path / "shared.index"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.close()
    with pytest.raises(ConcurrencyError):
        await second.close()

    assert second._initialized is False
    # And having been released, closing again is a no-op rather than a
    # second raise from a store the caller cannot get rid of.
    await second.close()


async def test_force_is_the_way_out_of_a_refusal(tmp_path: Path) -> None:
    """``save(force=True)`` overwrites deliberately.

    Without it a refusal is terminal: the stamp only advances on a
    *successful* write, so every later save compares against the same
    stale value and raises too. The store holds rows it has no way to
    persist, and ``load()`` is not a recovery — it would replace the very
    rows the caller is trying to keep.
    """
    persist = tmp_path / "shared.index"
    first = await _open(persist)
    second = await _open(persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.save()
    with pytest.raises(ConcurrencyError):
        await second.save()
    # Still refusing, on a store whose in-memory rows are intact.
    with pytest.raises(ConcurrencyError):
        await second.save()
    assert await second.count() == 4

    await second.save(force=True)

    # The other writer's rows are gone — that is what force means, and
    # the docstring says so.
    assert await _count_on_disk(persist) == 4
    # And the instance is in step with the file again, so ordinary saves
    # work from here without force.
    await _ingest(second, "third", 1, seed=3)
    await second.save()
    assert await _count_on_disk(persist) == 5
    await second.close()
    await first.close()


async def test_metadata_write_failure_leaves_the_index_and_the_stamp_intact(
    tmp_path: Path,
) -> None:
    """A ``.meta`` that will not pickle must not consume the index write.

    Written directly, the index file is replaced first and the side-car
    write then fails, leaving a *new* index beside a *stale* ``.meta``
    describing a different corpus — and the identity stamp still naming
    the file the write had already replaced, so every later save of this
    instance raises ``ConcurrencyError`` about a conflict that never
    happened.

    The unpicklable value is a real one, not an injected failure: a
    consumer metadata dict holding something ``pickle`` cannot serialize.
    """
    persist = tmp_path / "meta_fail.index"
    store = await _open(persist)
    await _ingest(store, "good", 1, seed=1)
    await store.save()

    await store.add_vectors(
        _vectors(1, seed=2),
        ids=["unpicklable"],
        metadata=[{"callback": lambda: None}],
    )
    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        await store.save()

    # The file still holds only what the successful save put there.
    assert await _count_on_disk(persist) == 1

    # And this instance can still save once the offending row is gone —
    # pre-fix it could not, because its stamp had been stranded.
    await store.delete_vectors(["unpicklable"])
    await store.save()
    assert await _count_on_disk(persist) == 1
    await store.close()


async def test_a_failed_rename_does_not_lock_the_store_out_of_its_own_file(
    tmp_path: Path,
) -> None:
    """The *publish* phase can fail partway too, not only the write phase.

    Staging both files before renaming either one closes the failure the
    test above covers: a write that fails now leaves both targets alone.
    It does not close the narrower one, because ``os.replace`` is atomic
    per file and there are two of them — the index can be renamed into
    place and the side-car's rename then fail. The identity stamp is
    taken only after *both* renames succeed, so it went on naming the
    file this instance had itself just replaced, and every later save
    raised ``ConcurrencyError`` against a writer that did not exist.

    That left the store holding rows with no way to persist them short of
    ``save(force=True)`` — the one call that exists to discard somebody
    else's. Recovering from a self-inflicted failure by invoking the
    lose-data escape hatch is not a recovery.

    The failure is a real filesystem refusal rather than an injected one:
    ``os.replace`` will not put a file over a non-empty directory.
    """
    persist = tmp_path / "rename_fail.index"
    store = await _open(persist)
    await _ingest(store, "first", 2, seed=1)
    await store.save()

    side_car = Path(f"{persist}.meta")

    def _obstruct() -> None:
        """Replace the side-car with a directory ``os.replace`` cannot clobber."""
        side_car.unlink()
        side_car.mkdir()
        (side_car / "occupant").write_text("a non-empty directory cannot be replaced")

    # Obstruct only the side-car, so the index rename ahead of it lands.
    # Genuinely offloaded, not relocated into a sync helper to quiet
    # ASYNC240: `to_thread` keeps it off the loop, where a plain call to
    # `_obstruct()` would block exactly as the inline version did. A
    # per-file waiver is the wrong trade here — this file's subject is
    # the persisted file, so it would unflag the calls most likely to be
    # added to it later.
    await asyncio.to_thread(_obstruct)

    await _ingest(store, "second", 2, seed=2)
    with pytest.raises(OSError):
        await store.save()

    # Nothing else has touched this path, so clearing the obstruction has
    # to be enough. Pre-fix this raised ConcurrencyError.
    await asyncio.to_thread(shutil.rmtree, side_car)

    # ``close()`` rather than ``save()`` on purpose: it persists only a
    # *dirty* store, so reaching the file at all also proves the failed
    # attempt left this store still knowing it had rows to write. The
    # obvious way to stop the phantom conflict — stamping the file as
    # saved on the way out of the failure — passes a ``save()`` here and
    # fails this, by turning the failed save into a silent one.
    await store.close()

    assert await _count_on_disk(persist) == 4


async def test_overlapping_saves_on_one_instance_do_not_conflict(tmp_path: Path) -> None:
    """A store must not raise a concurrency error against itself.

    The staleness check and the write it guards are two operations on a
    worker thread. Unserialized, two saves of the *same* instance — an
    autosave overlapping ``close()``, or a bare ``gather`` — either both
    pass the check and race on the file, or one stats the other's
    half-written file and raises. Neither involves a second instance,
    which is the only thing the check exists to detect.
    """
    persist = tmp_path / "concurrent.index"
    store = await _open(persist)
    try:
        for round_ in range(5):
            await _ingest(store, f"r{round_}", 20, seed=round_)
            await asyncio.gather(store.save(), store.save(), store.save())
    finally:
        await store.close()

    assert await _count_on_disk(persist) == 100


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(
            lambda s: s.add_vectors(_vectors(2, seed=7), ids=["x0", "x1"]),
            5,
            id="add_vectors",
        ),
        pytest.param(lambda s: s.delete_vectors(["first0", "first1"]), 1, id="delete_vectors"),
        pytest.param(
            lambda s: s.update_metadata(["first0"], [{"owner": "changed"}]), 3, id="update_metadata"
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
    than through the flag, so it pins the behaviour and not the
    mechanism.

    ``expected`` is the row count on disk afterwards; the metadata cases
    keep all three rows and are separately checked below.
    """
    persist = tmp_path / "mutator.index"
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
