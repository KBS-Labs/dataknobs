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

## The lockfile outlives its holder

Releasing does **not** unlink ``<path>.lock``, and that is the whole of
why a handover is safe. Closing the handle grants the lock to a blocked
waiter, which now holds an inode with no name; unlinking at that moment
would let the next ``acquire`` create a *fresh* inode and lock that
instead, putting two holders inside the section. So a zero-byte
``.lock`` file is left beside the target permanently — the conventional
shape for exactly this reason, and cheap next to what removing it costs.

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
* **:meth:`acquire` blocks without bound.** Correct on a worker thread,
  fatal on an event loop — never call it from a coroutine. Constructing
  the lock there is fine: ``__init__`` touches no filesystem, so the
  usual shape — build the lock on the loop, offload ``acquire`` — holds.
* **Not reentrant.** One thread acquiring twice deadlocks. This differs
  from the pre-``2.1`` lock, which appeared reentrant only because
  ``fcntl`` grants the owning process a lock it already holds — i.e. by
  way of the defect the intra-process mutex above exists to fix.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import IO, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType

_LockKey = tuple[int, int]
"""``(st_dev, st_ino)`` of a lockfile — see the module docstring."""

_intra_process_locks: dict[_LockKey, threading.Lock] = {}
"""One mutex per lockfile inode, shared by every instance.

Never pruned. An entry costs a few dozen bytes and the set is bounded by
how many distinct files the process locks, whereas dropping one while a
holder still had it would silently reopen the defect this exists to
close — a later acquirer would build a second mutex and walk in.
"""

_registry_guard = threading.Lock()
"""Serializes construction of the entries above, not their acquisition."""


def _mutex_for(key: _LockKey) -> threading.Lock:
    """The process-wide mutex for ``key``, created on first request."""
    with _registry_guard:
        mutex = _intra_process_locks.get(key)
        if mutex is None:
            mutex = threading.Lock()
            _intra_process_locks[key] = mutex
        return mutex


class FileLock:
    """Cross-platform, cross-process advisory lock on a single file.

    Args:
        filepath: The file being guarded. The lock itself is taken on a
            sibling ``<filepath>.lock`` beside the *resolved* target, so
            the target need not exist but its directory must.

    Example:
        ```python
        from dataknobs_common.locks import FileLock

        with FileLock("/var/lib/app/index.pkl"):
            ...  # read-modify-write, serialized against every holder
        ```
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        # The unresolved sibling. ``acquire`` replaces this with the
        # sibling of the *resolved* target, which is the file actually
        # locked; resolving needs the filesystem, and ``__init__`` is
        # called on the event loop. See the module docstring.
        self.lockfile = filepath + ".lock"
        self.lock_handle: IO[bytes] | None = None
        # Set between a successful ``acquire`` and its ``release``, and
        # ``None`` otherwise — which is also how ``release`` knows there
        # is nothing to hand on.
        self._mutex: threading.Lock | None = None

    def acquire(self) -> None:
        """Block until this is the only holder, in this process and beyond.

        Resolving the path and opening the lockfile are filesystem I/O,
        so they happen here rather than in ``__init__``. An async caller
        builds the lock on its event loop and offloads only this call to
        a worker thread; a stat in the constructor would stall that loop
        just before the offload meant to keep it free.

        Ordering is load-bearing in both directions. The handle is opened
        first because the mutex is keyed by the inode it reports. The
        mutex is then taken *before* the OS lock, and released *after*
        it, because a POSIX record lock is owned by the process: a second
        thread of this interpreter would be granted ``fcntl``
        immediately, so the mutex is the only thing standing between it
        and a handle this instance still holds.
        """
        # Resolve the *target*, then append the suffix. Appending first
        # and resolving after leaves the ``.lock`` component unresolved,
        # which is how a symlink and its target end up with two lockfiles.
        self.lockfile = os.path.realpath(self.filepath) + ".lock"

        handle = self._open_lockfile()
        try:
            stat = os.fstat(handle.fileno())
            mutex = _mutex_for((stat.st_dev, stat.st_ino))
            mutex.acquire()
        except BaseException:
            handle.close()
            raise

        try:
            self._lock_file(handle)
        except BaseException:
            mutex.release()
            handle.close()
            raise

        self.lock_handle = handle
        self._mutex = mutex

    def _open_lockfile(self) -> IO[bytes]:
        """Open the sibling lockfile, creating it if it is not there.

        ``O_CREAT | O_RDWR`` without ``O_TRUNC``: an exclusive record
        lock needs a writable descriptor, and truncating would discard
        the contents of a file that now outlives every holder.

        ``0o666`` is what ``open(path, "wb")`` passed before, restated
        because it is now load-bearing rather than incidental — a
        permanent lockfile has to stay openable by every uid that can
        write the directory. umask still applies, so this is a floor and
        not a guarantee; see the module docstring.
        """
        fd = os.open(self.lockfile, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            return os.fdopen(fd, "rb+")
        except BaseException:
            os.close(fd)
            raise

    def _lock_file(self, handle: IO[bytes]) -> None:
        """Take the OS-level lock on the sibling lockfile."""
        # ``sys.platform`` rather than ``platform.system()``: a type
        # checker narrows the former, so the branch it cannot verify on
        # this host is skipped instead of being reported as a module
        # without the attributes it only has on Windows.
        if sys.platform == "win32":
            import msvcrt
            import time

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    time.sleep(0.01)
                    continue
                return
        else:
            import fcntl

            fcntl.lockf(handle, fcntl.LOCK_EX)

    def release(self) -> None:
        """Hand the lock to the next holder. A no-op if not held.

        The lockfile is deliberately left in place — see the module
        docstring for why removing it is what lets two holders in.
        """
        mutex = self._mutex
        if mutex is None:
            return
        self._mutex = None
        try:
            handle = self.lock_handle
            self.lock_handle = None
            if handle is not None:
                if sys.platform == "win32":
                    import msvcrt

                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                handle.close()
        finally:
            mutex.release()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()
