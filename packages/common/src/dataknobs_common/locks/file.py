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

* **Within one process** — a :class:`threading.Lock` per canonical path.
  POSIX record locks (``fcntl.lockf``, i.e. ``fcntl(F_SETLKW)``) are
  owned by the *process*, so a second thread of the same interpreter is
  granted a lock the first already holds. Measured, not inferred:
  without this half, two threads sit inside the section simultaneously
  and neither blocks.
* **Across processes** — ``fcntl.lockf`` on POSIX, ``msvcrt.locking`` on
  Windows, on a sibling ``<path>.lock`` file.

## The lockfile outlives its holder

Releasing does **not** unlink ``<path>.lock``, and that is the whole of
why a handover is safe. Closing the handle grants the lock to a blocked
waiter, which now holds an inode with no name; unlinking at that moment
would let the next ``acquire`` create a *fresh* inode and lock that
instead, putting two holders inside the section. So a zero-byte
``.lock`` file is left beside the target permanently — the conventional
shape for exactly this reason, and cheap next to what removing it costs.

## What it does not promise

* **Advisory, and local-filesystem only.** A writer that never takes the
  lock is not stopped by it, and ``fcntl`` semantics over NFS and other
  network mounts are unreliable. Same caveat the single-file stores'
  mtime/inode identity check already carries.
* **:meth:`acquire` blocks without bound.** Correct on a worker thread,
  fatal on an event loop — never call it from a coroutine. Constructing
  the lock there is fine: ``__init__`` touches no filesystem, so the
  usual shape — build the lock on the loop, offload ``acquire`` — holds.
* **Not reentrant.** One thread acquiring twice deadlocks.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import IO, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType

_intra_process_locks: dict[str, threading.Lock] = {}
"""One mutex per canonical lockfile path, shared by every instance.

Never pruned. An entry costs a few dozen bytes and the set is bounded by
how many distinct files the process locks, whereas dropping one while a
holder still had it would silently reopen the defect this exists to
close — a later acquirer would build a second mutex and walk in.
"""

_registry_guard = threading.Lock()
"""Serializes construction of the entries above, not their acquisition."""


def _mutex_for(key: str) -> threading.Lock:
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
            sibling ``<filepath>.lock``, so the target need not exist.

    Example:
        ```python
        from dataknobs_common.locks import FileLock

        with FileLock("/var/lib/app/index.pkl"):
            ...  # read-modify-write, serialized against every holder
        ```
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lockfile = filepath + ".lock"
        self.lock_handle: IO[bytes] | None = None
        # Set between a successful ``acquire`` and its ``release``, and
        # ``None`` otherwise — which is also how ``release`` knows there
        # is nothing to hand on. Resolved in ``acquire`` rather than
        # here because resolving it is filesystem I/O; see there.
        self._mutex: threading.Lock | None = None

    def acquire(self) -> None:
        """Block until this is the only holder, in this process and beyond.

        Canonicalizing the path is filesystem I/O — ``realpath`` resolves
        every component — so it happens here rather than in ``__init__``.
        An async caller builds the lock on its event loop and offloads
        only this call to a worker thread, so a stat in the constructor
        would stall that loop just before the offload meant to keep it
        free.

        Takes the intra-process mutex first: it is the cheaper of the two
        and holding it is what makes the OS-level handle below safe to
        keep on ``self``, since only one thread of this process can be
        between here and :meth:`release`.
        """
        # Canonical so two spellings of one file — relative vs absolute,
        # a symlink and its target — contend for the same mutex instead
        # of taking one each and both proceeding.
        mutex = _mutex_for(os.path.realpath(self.lockfile))
        mutex.acquire()
        try:
            self._lock_file()
        except BaseException:
            mutex.release()
            raise
        self._mutex = mutex

    def _lock_file(self) -> None:
        """Take the OS-level lock on the sibling lockfile."""
        # ``sys.platform`` rather than ``platform.system()``: a type
        # checker narrows the former, so the branch it cannot verify on
        # this host is skipped instead of being reported as a module
        # without the attributes it only has on Windows.
        if sys.platform == "win32":
            import msvcrt
            import time

            while True:
                handle = open(self.lockfile, "wb")  # noqa: SIM115
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    time.sleep(0.01)
                    continue
                self.lock_handle = handle
                return
        else:
            import fcntl

            handle = open(self.lockfile, "wb")  # noqa: SIM115
            try:
                fcntl.lockf(handle, fcntl.LOCK_EX)
            except BaseException:
                handle.close()
                raise
            self.lock_handle = handle

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
