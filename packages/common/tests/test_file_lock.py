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
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dataknobs_common.locks import FileLock

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
