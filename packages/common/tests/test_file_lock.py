"""``FileLock`` must exclude every overlapping holder, not just every process.

The lock guards whole-file rewrites — a single-file store serializing
its entire state over one path. Two holders inside the section means one
snapshot replacing another's, silently and completely, which is the
failure the lock exists to prevent rather than a performance concern.

Two ways it did not:

* **POSIX record locks are owned by the process.** ``fcntl.lockf`` is
  ``fcntl(F_SETLKW)``, whose owner is the process, so a second thread of
  the *same* process acquires immediately — measured, not inferred. Two
  store instances in one interpreter therefore got no exclusion at all
  from a lock whose whole purpose was to provide it.
* **Release unlinked the lockfile.** Closing the handle hands the lock to
  a blocked waiter; unlinking the name then lets the *next* ``acquire``
  create a fresh inode and lock that instead. Two holders, no error.

Both are reproduce-first below: each fails against the pre-fix lock.

A third test guards the boundary the two fixes had to respect: an async
caller constructs the lock on its event loop and offloads only
``acquire`` to a worker thread, so construction has to stay off the
filesystem however the exclusion above is implemented.

Three more failures of the same kind — exclusion reported but not
delivered — were found reviewing the two fixes above, and are covered
below:

* **A symlink and its target took two locks.** The lockfile was
  ``realpath(filepath + ".lock")``, and only the target is a symlink, so
  the suffix stopped ``realpath`` from resolving anything.
* **A forked child inherited a locked mutex** held by a thread that does
  not exist in the child, and blocked on it forever.
* **Acquiring truncated the lockfile**, which stopped being harmless
  once the file became permanent.

Two cover ``timeout``, which is a new knob rather than a fixed defect:
every caller reaches ``acquire`` through ``asyncio.to_thread``, where an
unbounded wait parks a thread of the loop's shared executor.

The last two are one defect in two settings, found reviewing the knob
against the fixes. Closing *any* descriptor on a file releases every
record lock the process holds on it, so a per-instance handle meant a
thread that was correctly *refused* the lock released the holder's on
its way out, and a descriptor inherited across a fork released the
child's. Both are invisible from inside the process — the mutex refused
the second thread, the holder still believes it holds the lock — so
both read the verdict from an external observer.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dataknobs_common.exceptions import TimeoutError as DataknobsTimeoutError
from dataknobs_common.locks import FileLock
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

HOLD = 0.15
"""Seconds a holder stays in the section — long enough that an overlap is
observed rather than raced past, short enough to keep the suite quick."""


def test_two_threads_of_one_process_do_not_both_hold(tmp_path: Path) -> None:
    """Concurrent holders peak at one, even in a single interpreter.

    Pre-fix this recorded a peak of two: ``fcntl.lockf`` grants the same
    process a lock it already holds, so both threads walked straight in.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")

    inside: list[str] = []
    peak = 0
    bookkeeping = threading.Lock()

    def hold(name: str) -> None:
        nonlocal peak
        with FileLock(str(target)):
            with bookkeeping:
                inside.append(name)
                peak = max(peak, len(inside))
            time.sleep(HOLD)
            with bookkeeping:
                inside.remove(name)

    threads = [threading.Thread(target=hold, args=(n,)) for n in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(t.is_alive() for t in threads), "a holder never released"
    assert inside == []
    assert peak == 1, f"{peak} holders were inside the section at once"


def test_releasing_does_not_hand_the_lock_to_two_holders(tmp_path: Path) -> None:
    """A waiter that has been handed the lock keeps it across a re-acquire.

    Pre-fix, ``release`` unlinked the lockfile: the waiter took over the
    now-nameless inode while the releasing holder's next ``acquire``
    created a new one and locked *that*. Both were inside at once.

    The intra-process guard added alongside this would mask the race from
    a two-thread test, so this asserts the structural property that
    removes it — the lockfile outlives the holder that created it, so
    every acquirer contends for the same inode.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")
    lockfile = Path(str(target) + ".lock")

    lock = FileLock(str(target))
    lock.acquire()
    assert lockfile.exists()
    lock.release()

    assert lockfile.exists(), (
        "release unlinked the lockfile — the next acquire creates a new "
        "inode and locks it while a waiter still holds the old one"
    )

    before = lockfile.stat().st_ino
    with FileLock(str(target)):
        pass
    assert lockfile.stat().st_ino == before, "a re-acquire replaced the lock's inode"


_CHILD = """
import sys, time
from pathlib import Path
from dataknobs_common.locks import FileLock

target = Path(sys.argv[1])
with FileLock(str(target)):
    current = int(target.read_text() or "0")
    time.sleep(0.05)
    target.write_text(str(current + 1))
"""
"""A read-modify-write of one counter, held under the lock.

Run as a separate interpreter rather than a forked worker: the body has
to be importable by name from a fresh process, and a pytest test module
is not.
"""


def test_separate_processes_do_not_lose_an_update(tmp_path: Path) -> None:
    """Four interpreters each incrementing one counter all land.

    The half neither test above can reach — they run in one process, and
    the intra-process guard would satisfy them even if the file lock did
    nothing. A lost update here means two interpreters were inside at
    once.
    """
    target = tmp_path / "counter.txt"
    target.write_text("0")

    children = [subprocess.Popen([sys.executable, "-c", _CHILD, str(target)]) for _ in range(4)]
    for child in children:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            pytest.fail("a child never finished — the lock did not hand over")

    assert [c.returncode for c in children] == [0, 0, 0, 0]
    assert target.read_text() == "4", "an update was lost — no mutual exclusion"


def test_distinct_paths_do_not_serialize(tmp_path: Path) -> None:
    """The intra-process guard is per file, not one mutex for the process."""
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"")
    second.write_bytes(b"")

    peak = 0
    inside = 0
    bookkeeping = threading.Lock()

    def hold(path: Path) -> None:
        nonlocal peak, inside
        with FileLock(str(path)):
            with bookkeeping:
                inside += 1
                peak = max(peak, inside)
            time.sleep(HOLD)
            with bookkeeping:
                inside -= 1

    threads = [threading.Thread(target=hold, args=(p,)) for p in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak == 2, "two different files serialized against each other"


def test_the_same_file_spelled_two_ways_shares_one_lock(tmp_path: Path) -> None:
    """A relative and an absolute spelling of one path must not both hold."""
    target = tmp_path / "state.bin"
    target.write_bytes(b"")
    # Built as a string, not a ``Path``: ``Path`` would normalize the
    # "." away and the two spellings would be identical before the
    # lock ever saw them.
    spellings = [str(target), f"{tmp_path}/./state.bin"]

    peak = 0
    inside = 0
    bookkeeping = threading.Lock()

    def hold(path: str) -> None:
        nonlocal peak, inside
        with FileLock(path):
            with bookkeeping:
                inside += 1
                peak = max(peak, inside)
            time.sleep(HOLD)
            with bookkeeping:
                inside -= 1

    threads = [threading.Thread(target=hold, args=(s,)) for s in spellings]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak == 1, "two spellings of one path took two different locks"


@requires_blockbuster
async def test_constructing_a_lock_does_not_block_the_loop(tmp_path: Path) -> None:
    """Construction stays off the filesystem, because callers do it on a loop.

    ``acquire`` blocks without bound by design, and every async caller
    offloads it to a worker thread — but it constructs the lock first,
    on the event loop, where a stat of the path stalls every co-tenant
    before the offload begins. Canonicalizing the path is filesystem
    I/O (``realpath`` resolves each component), so it belongs with the
    blocking half rather than in ``__init__``.
    """
    with assert_no_blocking():
        FileLock(str(tmp_path / "state.bin"))


def test_a_symlink_and_its_target_take_one_lock(tmp_path: Path) -> None:
    """A stable name pointing at versioned storage is still one file.

    Pre-fix the lockfile was ``realpath(filepath + ".lock")``. Only the
    *target* is a symlink — ``current.bin.lock`` is not — so ``realpath``
    left the final component alone and the two spellings locked two
    different files, giving zero exclusion in the layout most likely to
    have a second writer: a rollover job holding the versioned path while
    a store holds the stable one.
    """
    versioned = tmp_path / "v2"
    versioned.mkdir()
    target = versioned / "index.bin"
    target.write_bytes(b"")
    stable = tmp_path / "current.bin"
    stable.symlink_to(target)

    peak = 0
    inside = 0
    bookkeeping = threading.Lock()

    def hold(path: str) -> None:
        nonlocal peak, inside
        with FileLock(path):
            with bookkeeping:
                inside += 1
                peak = max(peak, inside)
            time.sleep(HOLD)
            with bookkeeping:
                inside -= 1

    threads = [threading.Thread(target=hold, args=(s,)) for s in (str(stable), str(target))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak == 1, "a symlink and its target took two different locks"


def test_acquiring_does_not_truncate_the_lockfile(tmp_path: Path) -> None:
    """The lockfile survives an acquire with its contents intact.

    Pre-fix, ``open(lockfile, "wb")`` truncated on every acquire. That
    was harmless while the file was recreated each time, and stopped
    being harmless when it became permanent: a lockfile nothing can
    write to durably is one no future version can record an owner in.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")
    lockfile = Path(str(target) + ".lock")
    lockfile.write_bytes(b"owner=1234\n")

    with FileLock(str(target)):
        pass

    assert lockfile.read_bytes() == b"owner=1234\n", "acquire truncated the lockfile"


def test_a_bounded_acquire_gives_up_instead_of_parking_the_thread(tmp_path: Path) -> None:
    """``timeout`` returns ``False`` rather than waiting out the holder.

    Not a pre-existing bug but the absence of a knob: every caller
    reaches this through ``asyncio.to_thread``, so an unbounded wait
    parks a worker of the loop's *shared* executor for as long as the
    current holder runs, and every unrelated offload queues behind it.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")

    holder = FileLock(str(target))
    assert holder.acquire() is True
    try:
        waiter = FileLock(str(target), timeout=0.2)
        started = time.monotonic()
        assert waiter.acquire() is False, "a bounded acquire reported success"
        assert time.monotonic() - started < 5, "it waited well past its timeout"
    finally:
        holder.release()

    # The timeout did not leave the lock unusable for the next holder.
    with FileLock(str(target), timeout=5):
        pass


def test_entering_a_bounded_lock_raises_rather_than_running_unlocked(
    tmp_path: Path,
) -> None:
    """A ``with`` block whose lock timed out must not execute its body.

    The one outcome worse than waiting is proceeding: the body of every
    caller is a whole-state rewrite, so running it unlocked is the silent
    clobber this module exists to prevent.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")

    holder = FileLock(str(target))
    holder.acquire()
    try:
        entered = False
        with pytest.raises(DataknobsTimeoutError):
            with FileLock(str(target), timeout=0.1):
                entered = True
        assert not entered, "the body ran without the lock"
    finally:
        holder.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
# Forking a threaded parent is the hazard under test, not an accident, so
# the interpreter's warning about it is expected output here.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_is_not_wedged_by_an_inherited_mutex(tmp_path: Path) -> None:
    """A child forked while another thread held the lock can still acquire.

    Only the forking thread survives a fork, so a mutex held by any other
    thread is locked in the child with no owner left to release it. Pre-
    fix the child blocked on that inherited mutex forever — and
    ``acquire`` had no timeout to escape through, so the child was wedged
    rather than slow.

    ``fork`` from a threaded parent is exactly the hazard under test, so
    the child does the minimum and leaves via ``os._exit``: no cleanup
    handlers, no imports, nothing that could take a second inherited
    lock and confuse the result.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")

    holder_inside = threading.Event()
    may_release = threading.Event()

    def hold() -> None:
        with FileLock(str(target)):
            holder_inside.set()
            may_release.wait(timeout=10)

    thread = threading.Thread(target=hold)
    thread.start()
    assert holder_inside.wait(timeout=10), "the holder thread never got the lock"

    pid = os.fork()
    if pid == 0:  # pragma: no cover — the child never reports coverage
        # SIGALRM kills the child outright if the acquire never returns,
        # which the parent reads as "did not exit normally".
        signal.alarm(10)
        try:
            with FileLock(str(target)):
                pass
        except BaseException:
            os._exit(1)
        os._exit(0)

    # Let the child reach its acquire while the lock is genuinely held,
    # so it is the fork — not a free lock — that decides the outcome.
    time.sleep(0.2)
    may_release.set()
    thread.join(timeout=10)

    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), (
        "the forked child never returned from acquire — it inherited a mutex "
        "locked by a thread that does not exist in the child"
    )
    assert os.WEXITSTATUS(status) == 0, "the forked child failed to acquire"


_PROBE = """
import fcntl, os, sys

fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o666)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("BLOCKED")
else:
    print("GRANTED")
"""
"""Ask, from outside this process, whether the lock is currently free.

An external observer is the only one that can see the failure below: the
intra-process mutex still excludes the second thread, so every assertion
available inside this interpreter passes while the lock is gone.
"""


def _lock_is_held_externally(lockfile: str) -> bool:
    """Whether a separate process is refused the lock on ``lockfile``."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, lockfile],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "BLOCKED"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX record-lock semantics")
def test_a_refused_acquire_does_not_release_the_holder(tmp_path: Path) -> None:
    """A second thread giving up must not drop the lock the first holds.

    ``fcntl(2)``: closing *any* descriptor referring to a file releases
    every lock the process holds on it, whichever descriptor took them.
    So a second thread that opens the lockfile, fails to get the mutex,
    and closes its handle on the way out releases a lock it never took
    and does not know about — the first thread's.

    The give-up path is what a bounded ``acquire`` runs, and a bounded
    acquire is what the docs recommend to every ``asyncio.to_thread``
    caller, so the two features compose into the defect. Nothing inside
    this interpreter can see it: the mutex correctly refuses the second
    thread, and the first still believes it holds the lock. Only a
    separate process can observe that the lock is now free.
    """
    target = tmp_path / "index.pkl"
    target.write_bytes(b"")
    lockfile = str(target) + ".lock"

    holder = FileLock(str(target))
    assert holder.acquire() is True
    try:
        assert _lock_is_held_externally(lockfile), (
            "precondition: an outside process must be refused while the lock is held"
        )

        refused: list[bool] = []

        def give_up() -> None:
            refused.append(FileLock(str(target), timeout=0.05).acquire())

        thread = threading.Thread(target=give_up)
        thread.start()
        thread.join()
        assert refused == [False], "the second thread should have been refused"

        assert _lock_is_held_externally(lockfile), (
            "a refused acquire released the lock the first thread still "
            "holds — two writers can now be inside the section"
        )
    finally:
        holder.release()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork and record locks")
def test_a_forked_child_does_not_lose_its_lock_to_an_inherited_descriptor(
    tmp_path: Path,
) -> None:
    """A descriptor inherited across a fork must not release the child's lock.

    The child inherits every open descriptor, including the one this
    process holds the lockfile open on, but inherits none of the record
    locks. So the inherited descriptor is a live second reference to a
    file the child is about to lock through a *different* one — and by
    the same POSIX rule as above, closing it releases what the child
    holds.

    Fixed by closing the inherited descriptors in the fork handler
    rather than merely dropping them. That is the one moment at which
    closing is safe, precisely because the child holds no lock yet.

    Reads the outcome from an external process for the same reason as
    the refused-acquire test: inside the child, a released lock and a
    held one look identical.
    """
    target = tmp_path / "state.bin"
    target.write_bytes(b"")
    lockfile = str(target) + ".lock"

    # Hold the lock across the fork, so the child inherits a descriptor
    # on a lockfile this process has genuinely locked.
    holder = FileLock(str(target))
    assert holder.acquire() is True

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover — the child never reports coverage
        signal.alarm(10)
        os.close(read_fd)
        try:
            child_lock = FileLock(str(target))
            child_lock.acquire()  # Waits for the parent below to release.
            # Anything that would close an inherited descriptor: the
            # parent's lock object is still reachable here.
            holder.release()
            held = _lock_is_held_externally(lockfile)
            os.write(write_fd, b"held" if held else b"lost")
        except BaseException:
            os.write(write_fd, b"errd")
        os._exit(0)

    os.close(write_fd)
    # Let the child reach its acquire before the lock becomes free, so
    # it takes the lock through a descriptor of its own.
    time.sleep(0.2)
    holder.release()

    verdict = os.read(read_fd, 4)
    os.close(read_fd)
    os.waitpid(pid, 0)
    assert verdict == b"held", (
        "the child's lock was released by a descriptor it inherited across "
        "the fork — a second process can now enter the section with it"
    )
