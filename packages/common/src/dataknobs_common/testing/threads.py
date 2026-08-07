"""Detect leaked dataknobs daemon threads in tests.

Two dataknobs constructs put work on a private daemon thread so that
synchronous code can drive asynchronous machinery:
:class:`~dataknobs_common.sync_bridge.SyncLoopBridge` runs an event loop on
one, and :func:`~dataknobs_common.async_iter.aiter_sync_in_thread` runs a
producer on another. Both are released by their owner's ``close()``. A
holder that never exposes a ``close()`` — or a caller that never calls it —
leaks the thread for the life of the process.

The leak is silent by construction. The threads are daemons, so they do not
delay interpreter exit; the owning object still behaves correctly; nothing
raises. What it does do is accumulate: a full suite run once left 32 of them
alive, which was only noticed because it made *another package's*
thread-teardown assertions fail depending on test ordering.

:func:`assert_no_leaked_bridge_threads` turns that into a direct failure at
the site responsible::

    def test_wizard_closes_its_fsm():
        with assert_no_leaked_bridge_threads():
            fsm = build_wizard_fsm()
            fsm.step({})
            fsm.close()

It measures a **delta** — threads that appear inside the block and survive
it — rather than an absolute count, so a thread leaked by an unrelated
earlier test in the same process cannot make an unrelated assertion fail.
As a session-scoped autouse fixture it covers a whole suite; around a single
call it names the leaking operation precisely.

Both dataknobs daemon-thread names are covered. Covering only the bridge
would guarantee a fresh copy of this idiom the first time someone needed the
pump — which is how three near-identical copies of it came to exist across
``common``, ``fsm``, and ``bots`` in the first place.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from dataknobs_common.async_iter import _THREAD_NAME as _PUMP_THREAD_NAME
from dataknobs_common.sync_bridge import _THREAD_NAME as _BRIDGE_THREAD_NAME

__all__ = [
    "DK_AITER_PUMP_THREAD",
    "DK_DAEMON_THREAD_NAMES",
    "DK_SYNC_BRIDGE_THREAD",
    "assert_no_leaked_bridge_threads",
    "live_dk_daemon_threads",
]

#: Name of :class:`~dataknobs_common.sync_bridge.SyncLoopBridge`'s private
#: event-loop thread. Public so a test can scope an assertion to this
#: thread alone without importing the private constant from the module that
#: defines it.
DK_SYNC_BRIDGE_THREAD: str = _BRIDGE_THREAD_NAME

#: Name of :func:`~dataknobs_common.async_iter.aiter_sync_in_thread`'s
#: producer thread.
DK_AITER_PUMP_THREAD: str = _PUMP_THREAD_NAME

#: Names of every dataknobs-managed daemon thread a ``close()`` is expected
#: to release. Sourced from the modules that create them, so renaming one
#: there cannot leave this set silently stale.
DK_DAEMON_THREAD_NAMES: frozenset[str] = frozenset(
    {DK_SYNC_BRIDGE_THREAD, DK_AITER_PUMP_THREAD}
)


def live_dk_daemon_threads(
    names: Iterable[str] | None = None,
) -> list[threading.Thread]:
    """Return the dataknobs daemon threads currently alive.

    Args:
        names: Thread names to look for. Defaults to
            :data:`DK_DAEMON_THREAD_NAMES` (both of them).

    Returns:
        The live :class:`threading.Thread` objects, in
        :func:`threading.enumerate` order.
    """
    wanted = frozenset(names) if names is not None else DK_DAEMON_THREAD_NAMES
    return [t for t in threading.enumerate() if t.name in wanted]


@contextmanager
def assert_no_leaked_bridge_threads(
    names: Iterable[str] | None = None,
    *,
    grace_seconds: float = 1.0,
) -> Iterator[None]:
    """Assert the block leaks no dataknobs daemon threads.

    Samples the live threads on entry and, on exit, fails if any thread
    that appeared inside the block is still alive. Threads that were
    already alive on entry are ignored, so this never reports another
    test's leak.

    Args:
        names: Thread names to watch. Defaults to
            :data:`DK_DAEMON_THREAD_NAMES`. Normalized once on entry, so a
            one-shot iterable (a generator expression) works: the watch set
            is needed on both entry and exit, and re-consuming an exhausted
            iterator would leave the exit check watching nothing — a guard
            that silently passes forever.
        grace_seconds: How long to wait for an apparently-leaked thread to
            finish shutting down before failing. A thread whose ``close()``
            has been called is joined well within this; a genuinely leaked
            one never exits, so this bounds only the failure path. The wait
            is shared across all candidates rather than applied to each.

    Raises:
        AssertionError: If any thread created inside the block outlives it.

    Note:
        This does not run on exceptional exit — an exception propagating
        out of the block is the more informative failure, and asserting on
        thread state while unwinding would mask it.
    """
    watched = frozenset(names) if names is not None else DK_DAEMON_THREAD_NAMES
    before = set(live_dk_daemon_threads(watched))
    yield
    leaked = [t for t in live_dk_daemon_threads(watched) if t not in before]
    if not leaked:
        return

    # Give a thread that is mid-shutdown a chance to finish. Bounded by one
    # shared deadline so a real leak of N threads costs `grace_seconds`, not
    # N times that.
    deadline = time.monotonic() + grace_seconds
    for thread in leaked:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)

    still_alive = sorted(t.name for t in leaked if t.is_alive())
    assert not still_alive, (
        f"{len(still_alive)} dataknobs daemon thread(s) leaked: "
        f"{still_alive}. Something allocated a sync bridge or an "
        f"aiter pump and was never closed — check that every object "
        f"holding one exposes close()/aclose() and that callers use it "
        f"(a context manager or test fixture is the reliable form)."
    )
