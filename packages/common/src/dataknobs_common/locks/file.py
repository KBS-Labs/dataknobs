"""Advisory whole-file locking, held against every overlapping holder.

The hazard this addresses is a store that persists by serializing its
whole state over one path: two writers inside the section means one
snapshot replacing the other's outright, with no error and nothing in
the log. That is a correctness guarantee rather than a throughput one,
so a lock that excludes *some* overlapping holders does not partly
work — it reports success while the rows go missing.

Distinct from :class:`~dataknobs_common.locks.lock.DistributedLock` next
door, and deliberately so. That one is async, keyed by an opaque name,
and scoped to whoever shares its backing store; this is synchronous,
keyed by a filesystem path, and scoped to whoever can see that path.
The two do not substitute for each other: this one is taken *inside* a
worker thread, around blocking I/O, where there is no event loop to
await on.

## Two holders, two mechanisms

Exclusion needs both halves, because each covers what the other cannot:

* **Within one process** — a :class:`threading.Lock` per lockfile.
  POSIX record locks (``fcntl.lockf``, i.e. ``fcntl(F_SETLKW)``) are
  owned by the *process*, so a second thread of the same interpreter is
  granted a lock the first already holds. Measured, not inferred:
  without this half, two threads sit inside the section simultaneously
  and neither blocks.
* **Across processes** — ``fcntl.lockf`` on POSIX, ``msvcrt.locking`` on
  Windows, on a sibling ``<path>.lock`` file.

## One file, one lock, however it is spelled

Both halves key off the *file*, never off the string naming it, because
a lock that two spellings of one path can both hold excludes nothing:

* The lockfile is a sibling of the **resolved** target —
  ``realpath(filepath) + ".lock"``, not ``realpath(filepath + ".lock")``.
  The latter is what a symlinked target defeats: ``current.pkl`` is a
  symlink but ``current.pkl.lock`` is not, so ``realpath`` leaves the
  final component alone and the symlink and its target take two
  different lockfiles — the exact defect this module exists to close, in
  the layout (a stable name pointing at versioned storage) most likely
  to have two writers.
* The intra-process mutex is keyed by the lockfile's ``(st_dev,
  st_ino)`` rather than by its path. Inode identity is exact where path
  identity is a guess: it collapses hard links, and case-insensitive
  volumes where ``realpath`` preserves the case it was handed and
  ``/d/Index.pkl`` and ``/d/index.pkl`` are one file spelled two ways.

## Both the name and the descriptor outlive every holder

Releasing unlocks; it does **not** unlink ``<path>.lock`` and does not
close the descriptor. The two are one point, and it is the subtlest
thing here.

*The name*, because release hands the lock to a blocked waiter, which
now holds an inode with no name; unlinking at that moment would let the
next ``acquire`` create a *fresh* inode and lock that instead, putting
two holders inside the section. So a zero-byte ``.lock`` file is left
beside the target permanently — the conventional shape for exactly this
reason, and cheap next to what removing it costs.

*The descriptor*, because of a POSIX rule with no analogue elsewhere in
the standard: closing **any** descriptor referring to a file releases
every record lock the process holds on that file, whatever descriptor
took them. A per-instance handle therefore makes a second thread's
cleanup lethal — it opens the lockfile, finds the mutex taken, closes
its handle on the way out, and releases a lock it never held and does
not know about. Nothing inside the process can observe that: the mutex
correctly refused the second thread, and the holder still believes it
holds the lock. So there is exactly one descriptor per lockfile inode,
owned by the registry rather than by any instance, opened once and
closed never — see ``_LockEntry``. Release is ``LOCK_UN`` on it.

The lockfile is opened and **never truncated**, which matters only
because it is now permanent: the previous ``open(path, "wb")`` discarded
its contents on every acquire, so nothing could ever be recorded there.
The creation mode stays ``0o666`` before umask — what ``"wb"`` already
used — because a permanent lockfile created by one uid must stay
openable by every other uid that can write the directory. Note the
limit: umask still applies, so a lockfile a root process creates under
the usual ``022`` is ``0o644`` and locks out an unprivileged writer for
good. A deployment sharing a directory across uids needs a permissive
umask; there is no mode this module can pass that substitutes for one.

## Across a fork

The registry is reset in the child after ``os.fork``, both halves of it.
Only the forking thread survives, so a mutex another thread held at the
moment of the fork stays locked forever in the child, and the child's
first acquire on that path would block on a holder that does not exist.
The inherited descriptors are closed in the same handler, which is the
one instant at which that is safe *and* the only one at which it is
sufficient: record locks are not inherited across a fork, so the child
provably holds none to lose, and closing before it can acquire anything
is what stops an inherited descriptor from later releasing a lock the
child does hold. The entries are process-local bookkeeping either way,
so discarding them is the correct reconstruction rather than a
workaround.

## What it does not promise

* **Advisory, and local-filesystem only.** A writer that never takes the
  lock is not stopped by it, and ``fcntl`` semantics over NFS and other
  network mounts are unreliable. Same caveat the single-file stores'
  mtime/inode identity check already carries.
* **Acquiring needs to write.** The lockfile is opened ``O_RDWR``, so a
  holder needs create-or-write permission on the target's directory even
  to *read* under the lock. A reader that cannot is a reader with no
  writer to exclude — nothing can be published into a directory it
  cannot write — so a read path should degrade to an unlocked read
  rather than fail. A write path must not: there the write is the very
  thing that needs excluding.
* **``acquire`` blocks without bound by default.** Correct on a worker
  thread, fatal on an event loop — never call it from a coroutine.
  Constructing the lock there is fine: ``__init__`` touches no
  filesystem, so the usual shape — build the lock on the loop, offload
  ``acquire`` — holds. Pass ``timeout`` to bound the wait; the default
  parks a worker thread indefinitely, which on the shared
  ``asyncio.to_thread`` executor starves every unrelated offload behind
  it.
* **Not reentrant.** One thread acquiring twice deadlocks. This differs
  from the pre-``2.1`` lock, which appeared reentrant only because
  ``fcntl`` grants the owning process a lock it already holds — i.e. by
  way of the defect the intra-process mutex above exists to fix.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

from ..exceptions import TimeoutError as DataknobsTimeoutError

if TYPE_CHECKING:
    from types import TracebackType

_LockKey = tuple[int, int]
"""``(st_dev, st_ino)`` of a lockfile — see the module docstring."""


class _LockEntry:
    """The process-wide state for one lockfile inode.

    Pairs the intra-process mutex with the descriptor the OS lock is
    taken on, because the two have to share a lifetime: the descriptor
    must outlive every holder, and the mutex is what makes that safe.

    ``fd`` is the *only* descriptor this process ever opens on the
    inode. Opened once and closed never, because closing any descriptor
    on a file releases every record lock the process holds on it — so a
    descriptor per :class:`FileLock` would let one instance's cleanup
    release another's lock. ``-1`` once a fork has invalidated it, which
    is the one time the descriptor is closed; see ``_reset_after_fork``.
    """

    __slots__ = ("fd", "mutex")

    def __init__(self, mutex: threading.Lock, fd: int) -> None:
        self.mutex = mutex
        self.fd = fd


_intra_process_locks: dict[_LockKey, _LockEntry] = {}
"""One entry per lockfile inode, shared by every instance.

Never pruned while the process lives. An entry costs a mutex, a
descriptor and a dict slot, and the set is bounded by how many distinct
files the process locks, whereas dropping one while a holder still had
it would silently reopen the defect this exists to close — a later
acquirer would build a second mutex and walk in, and closing the
descriptor would release the holder's lock outright. Reset wholesale in
a forked child, where every entry is unreconstructable rather than
merely stale; see ``_reset_after_fork``.
"""

_registry_guard = threading.Lock()
"""Serializes construction of the entries above, not their acquisition."""

_POLL_INTERVAL = 0.01
"""Seconds between attempts when a bounded ``acquire`` has to retry.

Only reached on the timeout path: an unbounded acquire blocks in the
kernel rather than spinning.
"""

_BINARY = getattr(os, "O_BINARY", 0)
"""``os.O_BINARY`` where it exists, else a no-op.

Windows translates line endings on a text-mode descriptor. Nothing is
read or written through the lock's handle today, but the file is
deliberately never truncated so that it *can* carry contents, and a
descriptor opened text-mode would corrupt them the day something does.
POSIX has no such mode and no such constant.
"""

_IDENTITY_ATTEMPTS = 3
"""Times to re-look-up a lockfile that was replaced mid-registration.

Only a foreign actor unlinking the lockfile can cause one retry, and
that is the interference the module already declares unsupported; the
bound is here so a pathological loop terminates rather than spins.
"""


def _entry_for(lockfile: str) -> _LockEntry:
    """The process-wide entry for ``lockfile``, created on first request.

    The inode is read with a **path** stat rather than by opening the
    file, and that is load-bearing rather than incidental. A descriptor
    is what makes the identity exact, but opening one to ask the
    question is what this function exists to avoid: if the inode turns
    out to be registered already, the descriptor is a second one on a
    file this process may hold a lock on, and closing it would release
    that lock. So the descriptor is opened only on the branch that
    commits to keeping it forever.
    """
    for _ in range(_IDENTITY_ATTEMPTS):
        try:
            # A stat on the *path*, which is the whole point: it reads
            # the inode without opening a descriptor on it.
            stat = Path(lockfile).stat()
        except FileNotFoundError:
            _create_lockfile(lockfile)
            continue
        key = (stat.st_dev, stat.st_ino)
        with _registry_guard:
            entry = _intra_process_locks.get(key)
            if entry is not None:
                # Registered, so no descriptor is opened at all: this
                # process may hold a record lock on the inode, and a
                # second descriptor is exactly what must not exist.
                return entry
            # Unregistered, so this process holds no lock on the inode —
            # every lock it takes goes through an entry — which is what
            # makes opening, and closing again below, safe here.
            try:
                fd = os.open(lockfile, os.O_RDWR | _BINARY)
            except FileNotFoundError:
                # Unlinked between the stat and the open; recreate and
                # look it up again.
                continue
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) == key:
                entry = _LockEntry(threading.Lock(), fd)
                _intra_process_locks[key] = entry
                return entry
            # Replaced between the stat and the open, so the descriptor
            # is on some other inode. If that one is registered, this
            # descriptor is deliberately leaked rather than closed: the
            # entry may be locked through its own, and closing any
            # descriptor would release that lock. One leaked descriptor
            # per foreign unlink, against dropping a live lock.
            replaced = _intra_process_locks.get((opened.st_dev, opened.st_ino))
            if replaced is not None:
                return replaced
            os.close(fd)
    raise OSError(
        f"FileLock: {lockfile} was replaced on every attempt to identify "
        "it. Something outside this process is unlinking lockfiles, which "
        "defeats the lock — see the dataknobs_common.locks.file docstring."
    )


def _create_lockfile(lockfile: str) -> None:
    """Create ``lockfile`` if it is not there, leaving no lock behind.

    ``O_EXCL`` so the descriptor closed here is provably one nobody can
    hold a lock through: it is a file that did not exist a moment ago.
    A concurrent creator winning the race is the ordinary outcome, not
    an error — the caller re-stats either way.
    """
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR | _BINARY, 0o666)
    except FileExistsError:
        return
    os.close(fd)


def _reset_after_fork() -> None:
    """Discard inherited lock state in a forked child.

    Every entry is unrecoverable rather than merely stale, on both
    halves. Only the forking thread exists here, so a mutex any other
    thread held is locked with no owner left to release it, and
    ``_registry_guard`` itself is in that set — a fork landing inside
    ``_entry_for`` would otherwise wedge every path in the child, not
    just one file's.

    The inherited descriptors are closed rather than merely dropped,
    and this is the one moment at which that is safe: POSIX record
    locks are not inherited across a fork, so the child provably holds
    none, and closing before it can acquire anything is what stops an
    inherited descriptor from releasing a lock the child later takes.
    Leaving them open would leak one descriptor per lockfile per fork.
    """
    global _registry_guard  # noqa: PLW0603 — rebuilding process-global state

    for entry in _intra_process_locks.values():
        fd, entry.fd = entry.fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - already closed
                pass
    _intra_process_locks.clear()
    _registry_guard = threading.Lock()


if hasattr(os, "register_at_fork"):  # POSIX only; no fork on Windows.
    os.register_at_fork(after_in_child=_reset_after_fork)


class FileLock:
    """Cross-platform, cross-process advisory lock on a single file.

    Args:
        filepath: The file being guarded. The lock itself is taken on a
            sibling ``<filepath>.lock`` beside the *resolved* target, so
            the target need not exist but its directory must.
        timeout: Seconds to wait before giving up, or ``None`` (the
            default) to wait without bound. A bounded wait is what an
            ``asyncio.to_thread`` caller wants: the default parks a
            worker of the loop's shared executor for as long as the
            current holder runs.

    Example:
        ```python
        from dataknobs_common.locks import FileLock

        with FileLock("/var/lib/app/index.pkl"):
            ...  # read-modify-write, serialized against every holder
        ```

        Bounded, for a caller that must not park a pooled thread:

        ```python
        lock = FileLock("/var/lib/app/index.pkl", timeout=30)
        if lock.acquire():
            try:
                ...
            finally:
                lock.release()
        ```
    """

    def __init__(self, filepath: str, *, timeout: float | None = None):
        self.filepath = filepath
        self.timeout = timeout
        # Not known until ``acquire`` resolves the target: the lockfile
        # is a sibling of the *resolved* path, and resolving is
        # filesystem I/O that ``__init__`` must not do because an async
        # caller builds the lock on its event loop. ``None`` rather than
        # the unresolved sibling, which reads as an answer while naming
        # a file that is not the one locked whenever a symlink is in the
        # path. See the module docstring.
        self.lockfile: str | None = None
        # Set between a successful ``acquire`` and its ``release``, and
        # ``None`` otherwise — which is also how ``release`` knows there
        # is nothing to hand on. The entry is process-wide state this
        # instance borrows, never state it owns: the descriptor inside
        # outlives every holder.
        self._entry: _LockEntry | None = None

    def acquire(self) -> bool:
        """Block until this is the only holder, in this process and beyond.

        Returns ``True`` when the lock is held and ``False`` when
        ``timeout`` elapsed first — the same shape as
        :meth:`~dataknobs_common.locks.lock.DistributedLock.acquire`, so
        the two concurrency primitives read alike. Without a ``timeout``
        the return is always ``True``.

        Resolving the path and reaching the lockfile are filesystem I/O,
        so they happen here rather than in ``__init__``. An async caller
        builds the lock on its event loop and offloads only this call to
        a worker thread; a stat in the constructor would stall that loop
        just before the offload meant to keep it free.

        Ordering is load-bearing. The mutex is taken *before* the OS
        lock and released *after* it, because a POSIX record lock is
        owned by the process: a second thread of this interpreter would
        be granted ``fcntl`` immediately, so the mutex is the only thing
        standing between it and a lock this instance still holds. Note
        what does **not** happen on the way out of a refused acquire —
        no descriptor is closed, because there is no descriptor of this
        instance's own to close. See ``_LockEntry``.
        """
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        # Resolve the *target*, then append the suffix. Appending first
        # and resolving after leaves the ``.lock`` component unresolved,
        # which is how a symlink and its target end up with two lockfiles.
        self.lockfile = os.path.realpath(self.filepath) + ".lock"

        entry = _entry_for(self.lockfile)
        if not _acquire_mutex(entry.mutex, deadline):
            return False

        try:
            if not self._lock_file(entry.fd, deadline):
                entry.mutex.release()
                return False
        except BaseException:
            entry.mutex.release()
            raise

        self._entry = entry
        return True

    def _lock_file(self, fd: int, deadline: float | None) -> bool:
        """Take the OS-level lock. ``False`` if ``deadline`` passed first."""
        # ``sys.platform`` rather than ``platform.system()``: a type
        # checker narrows the former, so the branch it cannot verify on
        # this host is skipped instead of being reported as a module
        # without the attributes it only has on Windows.
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    if not _wait_to_retry(deadline):
                        return False
                    continue
                return True
        else:
            import fcntl

            if deadline is None:
                # Blocks in the kernel rather than spinning, which is
                # what an unbounded wait should do.
                fcntl.lockf(fd, fcntl.LOCK_EX)
                return True
            while True:
                try:
                    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    if not _wait_to_retry(deadline):
                        return False
                    continue
                return True

    def release(self) -> None:
        """Hand the lock to the next holder. A no-op if not held.

        Unlocks the descriptor rather than closing it, and leaves the
        lockfile in place. Both are the same point: a descriptor and a
        name that outlive their holder are what make a handover safe.
        Closing would release every lock this *process* holds on the
        file, not just this one's, and unlinking would let the next
        acquire lock a fresh inode. See the module docstring.
        """
        entry = self._entry
        if entry is None:
            return
        self._entry = None
        try:
            if entry.fd >= 0:
                self._unlock_file(entry.fd)
        finally:
            entry.mutex.release()

    @staticmethod
    def _unlock_file(fd: int) -> None:
        """Drop the OS-level lock, keeping the descriptor open."""
        if sys.platform == "win32":
            import msvcrt

            with contextlib.suppress(OSError):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.lockf(fd, fcntl.LOCK_UN)

    def __enter__(self) -> Self:
        if not self.acquire():
            raise DataknobsTimeoutError(
                f"FileLock: waited {self.timeout}s for {self.filepath} and "
                "the lock was still held. A single-file store serializes its "
                "whole state under this lock, so a save can hold it for as "
                "long as that takes; raise timeout, or give each writer its "
                "own path.",
                context={"filepath": self.filepath, "timeout": self.timeout},
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


def _acquire_mutex(mutex: threading.Lock, deadline: float | None) -> bool:
    """Take ``mutex``, bounded by ``deadline`` when there is one."""
    if deadline is None:
        mutex.acquire()
        return True
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        # ``Lock.acquire(timeout=0)`` blocks; a non-blocking try is the
        # only way to spend no time at all here.
        return mutex.acquire(blocking=False)
    return mutex.acquire(timeout=remaining)


def _wait_to_retry(deadline: float | None) -> bool:
    """Sleep before the next attempt. ``False`` once ``deadline`` passed."""
    remaining = float("inf") if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        return False
    time.sleep(min(_POLL_INTERVAL, remaining))
    return True
