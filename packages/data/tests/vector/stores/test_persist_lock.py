"""Overlapping saves on one ``persist_path`` are serialized, not raced.

The staleness check that makes a single-file store refuse to clobber a
sibling runs on a worker thread with the write after it, so the check
and the write it guards are two operations with a scheduling point
between them. Two instances that both pass the check before either
writes both proceed — the second replaces the first's file with a
snapshot that never saw its rows, and **neither raises**. The refusal
that exists to prevent exactly this never fires, because by the time it
would have run, both had already been waved through.

That is a time-of-check/time-of-use hole, and closing it needs a lock
the check and the write are both inside. ``_save_lock`` is per instance
and cannot see a second one; ``FileLock`` — the cross-process file lock
``AsyncFileDatabase`` already takes for the same hazard — can.

Covered here, all at the ``VectorStoreBase`` layer both stores share:

* two instances saving at once (the TOCTOU above),
* a load overlapping this instance's own save, which used to run
  concurrently and stamp fields the save path owns,
* the scratch file two stagers used to collide on,
* the WARNING a destructive ``force=True`` bypass has to leave behind.

Reviewing those fixes found more, covered below:

* a ``write`` that raises leaked the scratch file it had already
  created, which unique names turned from self-healing into unbounded,
* a scratch file left by a killed process was never swept,
* a published file was not ``fsync``ed before the rename that publishes
  it, so staging survived a crashed process but not a power cut,
* the ``makedirs`` the lock depends on stayed duplicated in each store
  instead of moving into the bracket with it,
* taking the lock made a *read* require write access to the directory,
  which refuses an index on a read-only mount,
* ``force=True`` reported a discarded write for a file that was merely
  gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataknobs_common.exceptions import ConcurrencyError
from dataknobs_common.testing import is_faiss_available

from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

requires_faiss = pytest.mark.skipif(not is_faiss_available(), reason="faiss not installed")

pytestmark = pytest.mark.asyncio

DIMENSIONS = 4

BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("faiss", id="faiss", marks=requires_faiss),
]


def _vectors(count: int, seed: int) -> np.ndarray:
    """``count`` distinct rows, deterministic per ``seed``."""
    rows = np.zeros((count, DIMENSIONS), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 1] = [0.01 * (seed * 100 + i) for i in range(count)]
    return rows


def _base(backend: str) -> type:
    return MemoryVectorStore if backend == "memory" else FaissVectorStore


def _instrumented(backend: str, *, before_save: Any = None, log: list[str] | None = None) -> type:
    """The real store with observation points, not a substitute for it.

    ``before_save`` runs on the worker thread at the top of
    ``_save_to_disk`` — *before* the lock, so it can hold every writer
    at one line and let them race from there rather than reproducing a
    race by luck. ``log`` records entry and exit of both disk bodies, so
    a test can assert they did not interleave.
    """
    base = _base(backend)

    class _Instrumented(base):  # type: ignore[valid-type,misc]
        def _save_to_disk(self, *args: Any, **kwargs: Any) -> None:
            if log is not None:
                log.append("save-enter")
            if before_save is not None:
                before_save()
            try:
                super()._save_to_disk(*args, **kwargs)
            finally:
                if log is not None:
                    log.append("save-exit")

        def _load_from_disk(self) -> None:
            if log is not None:
                log.append("load-enter")
            try:
                super()._load_from_disk()
            finally:
                if log is not None:
                    log.append("load-exit")

    return _Instrumented


async def _open(cls: type, persist: Path) -> Any:
    store = cls({"dimensions": DIMENSIONS, "metric": "cosine", "persist_path": str(persist)})
    await store.initialize()
    return store


async def _ingest(store: Any, prefix: str, count: int, seed: int) -> None:
    await store.add_vectors(
        _vectors(count, seed),
        ids=[f"{prefix}{i}" for i in range(count)],
        metadata=[{"owner": prefix} for _ in range(count)],
    )


async def _shutdown(*stores: Any) -> None:
    """Close each store, tolerating the refusal a test may have provoked.

    ``close()`` persists a dirty store, so a test that deliberately left
    two instances contending would get the refusal again on teardown.
    """
    for store in stores:
        with contextlib.suppress(ConcurrencyError):
            await store.close()


@pytest.mark.parametrize("backend", BACKENDS)
async def test_two_instances_saving_at_once_do_not_both_pass_the_guard(
    backend: str, tmp_path: Path
) -> None:
    """Exactly one of two simultaneous savers writes; the other refuses.

    Pre-fix both saves returned cleanly and the file held whichever
    landed second — the other instance's rows gone from disk, with the
    guard that exists to stop that never having fired.
    """
    persist = tmp_path / "shared.idx"
    gate = threading.Barrier(2, timeout=10)
    opened = threading.Event()

    def hold_both_at_the_line() -> None:
        """Release only once both writers are in, then stand down.

        One-shot: teardown closes a store the refusal left dirty, which
        saves again, and a third arrival at a two-party barrier would
        wait out the timeout.
        """
        if opened.is_set():
            return
        try:
            gate.wait()
        finally:
            opened.set()

    cls = _instrumented(backend, before_save=hold_both_at_the_line)

    first = await _open(cls, persist)
    second = await _open(cls, persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    outcomes = await asyncio.gather(first.save(), second.save(), return_exceptions=True)
    refused = [o for o in outcomes if isinstance(o, ConcurrencyError)]
    other = [
        o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, ConcurrencyError)
    ]

    assert len(refused) == 1, (
        "both writers passed the staleness check and went on to write — "
        f"the second replaced the first's rows without raising. Got {outcomes}"
    )
    assert other == [], f"unexpected failure: {other}"

    await _shutdown(first, second)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_load_does_not_run_inside_this_instance_s_own_save(
    backend: str, tmp_path: Path
) -> None:
    """``load()`` and ``save()`` on one store do not interleave.

    ``_load_from_disk`` stamps ``_persisted_identity`` and clears
    ``_dirty`` — both owned by the save path — so a load running inside
    a save can mark the store in step with a file that save has not
    written yet, after which ``close()`` skips a mutation it never
    persisted. For FAISS it is also a torn read: the index and its
    ``.meta`` side-car are published by two renames, and a reader
    between them sees a new index beside a stale side-car.

    Pre-fix ``load()`` took no lock at all and the sequence below
    interleaved.
    """
    persist = tmp_path / "shared.idx"
    order: list[str] = []
    parked = threading.Event()
    release = threading.Event()

    def park() -> None:
        # Only the first save waits; the load is what has to be kept out.
        if not parked.is_set():
            parked.set()
            release.wait(timeout=10)

    cls = _instrumented(backend, before_save=park, log=order)
    store = await _open(cls, persist)
    await _ingest(store, "row", 3, seed=1)
    # ``initialize()`` loads, so the interesting window starts here.
    order.clear()

    saving = asyncio.create_task(store.save())
    await asyncio.to_thread(parked.wait, 10)
    loading = asyncio.create_task(store.load())
    await asyncio.sleep(0.1)
    release.set()
    await asyncio.gather(saving, loading)

    save_span = (order.index("save-enter"), order.index("save-exit"))
    load_span = (order.index("load-enter"), order.index("load-exit"))
    assert not (save_span[0] < load_span[0] < save_span[1]), (
        f"the load ran inside the save: {order}"
    )

    await _shutdown(store)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_two_stagers_do_not_collide_on_one_scratch_file(backend: str, tmp_path: Path) -> None:
    """Concurrent publishes stage to different files.

    A fixed ``<final>.tmp`` sibling is shared by every writer of that
    path: one stages over the other's bytes, and the loser's ``finally``
    unlink can delete a scratch file the winner is about to rename —
    turning a silent clobber into a spurious failure. The file lock
    above keeps two *saves* apart, so this pins the staging itself
    rather than relying on the caller that happens to hold a lock.
    """
    persist = tmp_path / "shared.idx"
    store = await _open(_base(backend), persist)
    final = str(tmp_path / "artifact.bin")
    staged: list[str] = []
    gate = threading.Barrier(2, timeout=10)
    record = threading.Lock()

    def writer(payload: bytes):
        def write(path: str) -> None:
            with record:
                staged.append(path)
            gate.wait()
            Path(path).write_bytes(payload)

        return write

    def publish(payload: bytes) -> None:
        store._write_then_publish([(final, writer(payload))])

    await asyncio.gather(
        asyncio.to_thread(publish, b"a" * 16),
        asyncio.to_thread(publish, b"b" * 16),
    )

    assert len(set(staged)) == 2, f"both writers staged to one path: {staged}"

    # Read back off the loop: this test schedules concurrency, so the
    # test-loop carve-out in ``rules/async-transport.md`` does not reach
    # it and a blocking stat here would be a real finding.
    published, leftovers = await asyncio.to_thread(
        lambda: (Path(final).read_bytes(), list(tmp_path.glob("*.tmp")))
    )
    assert published in (b"a" * 16, b"b" * 16)
    assert leftovers == [], "a scratch file was left behind"

    await _shutdown(store)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_forcing_past_the_guard_is_logged_at_warning(
    backend: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``force=True`` is a destructive bypass and has to say so.

    ``rules/security.md`` §8 requires a security bypass to log at
    WARNING when enabled, and this is also the line an operator wants
    when asking where the rows went.
    """
    persist = tmp_path / "shared.idx"
    cls = _base(backend)
    first = await _open(cls, persist)
    second = await _open(cls, persist)
    await _ingest(first, "first", 5, seed=1)
    await _ingest(second, "second", 4, seed=2)

    await first.save()
    with pytest.raises(ConcurrencyError):
        await second.save()

    with caplog.at_level(logging.WARNING, logger="dataknobs_data.vector.stores.common"):
        await second.save(force=True)

    assert any(
        record.levelno == logging.WARNING and "force" in record.getMessage().lower()
        for record in caplog.records
    ), f"no WARNING recorded for the bypass: {[r.getMessage() for r in caplog.records]}"

    await _shutdown(first, second)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_persist_path_under_a_missing_directory_still_opens(
    backend: str, tmp_path: Path
) -> None:
    """First run: the directory appears on save, not before.

    The lockfile is a sibling of the target, so a load that takes the
    lock before checking for the file would open it in a directory that
    does not exist yet and fail the load outright — where the old
    ``os.path.exists`` simply reported nothing to read.
    """
    persist = tmp_path / "not-created-yet" / "shared.idx"
    store = await _open(_base(backend), persist)
    await _ingest(store, "row", 3, seed=1)
    await store.save()

    assert await asyncio.to_thread(persist.exists)
    reopened = await _open(_base(backend), persist)
    assert await reopened.count() == 3
    await _shutdown(store, reopened)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_write_that_raises_leaves_no_scratch_file_behind(
    backend: str, tmp_path: Path
) -> None:
    """The likeliest failure is the one that used to leak.

    The scratch path was recorded only *after* its ``write`` returned, so
    the cleanup ran over a list that never contained the file the failed
    write had already created. Unique scratch names made that unbounded:
    the previous fixed ``<final>.tmp`` was overwritten by the next save,
    while a fresh name leaks once per failure — an autosave loop over one
    unpicklable value fills the disk with partial snapshots.
    """
    persist = tmp_path / "shared.idx"
    store = await _open(_base(backend), persist)
    final = str(tmp_path / "artifact.bin")

    def write_that_fails(path: str) -> None:
        # The file exists by now: ``mkstemp`` created it before the
        # writer was called, which is exactly why a failure here leaks.
        Path(path).write_bytes(b"partial")
        raise RuntimeError("cannot serialize")

    with pytest.raises(RuntimeError, match="cannot serialize"):
        await asyncio.to_thread(store._write_then_publish, [(final, write_that_fails)])

    leftovers = await asyncio.to_thread(lambda: list(tmp_path.glob("artifact.bin.*.tmp")))
    assert leftovers == [], f"a failed write left its scratch file behind: {leftovers}"

    await _shutdown(store)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_dead_writer_s_scratch_file_is_swept_on_the_next_save(
    backend: str, tmp_path: Path
) -> None:
    """A process killed mid-save costs one stray file, not one per kill.

    Nothing else removes them: ``load`` ignores them and the unique names
    that stopped two stagers colliding also stopped the next save from
    overwriting the last one's leftovers. The sweep runs under the file
    lock, which is what makes it safe — a live writer of this target
    holds that lock, so anything visible here is orphaned.
    """
    persist = tmp_path / "shared.idx"
    store = await _open(_base(backend), persist)
    await _ingest(store, "row", 3, seed=1)

    orphan = tmp_path / "shared.idx.deadbeef.tmp"
    await asyncio.to_thread(orphan.write_bytes, b"half a snapshot")

    await store.save()

    assert not await asyncio.to_thread(orphan.exists), "an orphaned scratch file survived a save"
    assert await asyncio.to_thread(persist.exists), "the save itself did not land"

    await _shutdown(store)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_publish_flushes_the_file_and_the_rename(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging survives a crash; without ``fsync`` it does not survive power loss.

    ``os.replace`` is atomic against readers, not against a power cut: on
    a journalled filesystem the rename metadata can land while the data
    it names has not, leaving a truncated file that has *already*
    replaced a known-good one — the one outcome staging exists to
    prevent.

    Durability is not observable from a test, so this asserts the
    syscall. ``os.fsync`` is replaced with a real recording function
    rather than a mock: the inodes it captures are checked against the
    published file and its directory, so the test pins *what* was
    flushed and not merely that something was.
    """
    persist = tmp_path / "shared.idx"
    store = await _open(_base(backend), persist)
    await _ingest(store, "row", 3, seed=1)

    real_fsync = os.fsync
    flushed: list[int] = []

    def recording_fsync(fd: int) -> None:
        flushed.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    await store.save()
    monkeypatch.undo()

    file_ino, dir_ino = await asyncio.to_thread(
        lambda: (persist.stat().st_ino, tmp_path.stat().st_ino)
    )
    assert file_ino in flushed, "the published file was never flushed to disk"
    assert dir_ino in flushed, "the rename that published it was never flushed"

    await _shutdown(store)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_the_shared_bracket_creates_the_directory_it_locks_in(
    backend: str, tmp_path: Path
) -> None:
    """The lock's precondition lives with the lock, not in each store.

    ``_persisted_save`` opens a lockfile beside the target, so the
    target's directory has to exist first. Both stores carried that
    ``makedirs`` verbatim, which left it out of the bracket a third store
    would inherit — such a store would get ``FileNotFoundError`` out of
    the lock on its first save. Asserted against the base method rather
    than through a store, because the base method is where it has to be.
    """
    store = await _open(_base(backend), tmp_path / "unused.idx")
    target = tmp_path / "made" / "on" / "demand" / "state.bin"

    def save_through_the_bracket() -> None:
        with store._persisted_save(str(target), force=False):
            Path(target).write_bytes(b"payload")

    await asyncio.to_thread(save_through_the_bracket)

    assert await asyncio.to_thread(target.exists), "the bracket did not create its directory"

    await _shutdown(store)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_read_only_directory_still_loads(backend: str, tmp_path: Path) -> None:
    """Taking the lock must not become a requirement to *read*.

    ``FileLock`` opens ``<path>.lock`` ``O_RDWR``, so a directory this
    process cannot write is a directory it cannot lock — an index baked
    into a read-only image layer, or served from a read-only mount. Both
    loaded before the lock existed and failed outright after it.

    Degrading is sound rather than lenient, and only here: publishing is
    an ``os.replace`` into this same directory, so a writer to exclude
    cannot exist. ``_persisted_save`` keeps the hard lock.
    """
    served = tmp_path / "served"
    await asyncio.to_thread(served.mkdir)
    persist = served / "shared.idx"

    writer = await _open(_base(backend), persist)
    await _ingest(writer, "row", 3, seed=1)
    await writer.save()
    await _shutdown(writer)

    def make_read_only() -> None:
        # Drop the lockfile first: an existing one is openable without
        # creating anything, so leaving it would hide the failure.
        for lock in served.glob("*.lock"):
            lock.unlink()
        served.chmod(0o555)

    await asyncio.to_thread(make_read_only)
    try:
        reopened = await _open(_base(backend), persist)
        assert await reopened.count() == 3, "a read-only directory refused a load"
        await _shutdown(reopened)
    finally:
        await asyncio.to_thread(served.chmod, 0o755)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_forcing_over_a_file_that_is_gone_does_not_claim_a_loss(
    backend: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing file and a changed file are not the same bypass.

    Both fail the identity comparison, so both took the branch announcing
    that "another writer's rows are being discarded" — but a file that is
    gone has no rows in it. The WARNING is the line an operator reads
    when asking where the rows went, so a loss it invents sends them
    looking for something that never happened.
    """
    persist = tmp_path / "shared.idx"
    store = await _open(_base(backend), persist)
    await _ingest(store, "row", 3, seed=1)
    await store.save()

    # Whatever the store publishes, remove it: the identity stamp now
    # points at a file that is not there.
    await asyncio.to_thread(persist.unlink)

    with caplog.at_level(logging.WARNING, logger="dataknobs_data.vector.stores.common"):
        await store.save(force=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any("no longer there" in message for message in messages), (
        f"the bypass did not report a deleted file as such: {messages}"
    )
    assert not any("being discarded" in message for message in messages), (
        f"the bypass claimed a loss that did not happen: {messages}"
    )

    await _shutdown(store)
