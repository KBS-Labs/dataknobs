"""Run a coroutine to completion from a synchronous caller — safely.

This is the async->sync counterpart to :func:`aiter_sync_in_thread` (the
sync->async direction). A synchronous public API that wraps an
async-first implementation faces a recurring problem: it must run a
coroutine to completion and return its result, but the obvious tools fail
in the one case that matters most. ``asyncio.run(coro)`` and
``loop.run_until_complete(coro)`` both raise (or deadlock) when a loop is
**already running on the calling thread** — exactly what happens when a
synchronous wrapper is, in turn, called from inside async code. The
``nest_asyncio`` monkey-patch "fixes" this but is rejected by the
dependency bar.

:class:`SyncLoopBridge` is the structural fix. It owns a private event
loop running on a **dedicated daemon thread**, so a coroutine handed to
:meth:`SyncLoopBridge.run` always executes on a loop that is *not* the
caller's. The caller blocks on the result like any synchronous call, and
the footgun is avoided by construction rather than by patching the event
loop — the bridge is callable from a plain sync function and from inside a
running event loop alike, with no deadlock.

Typical use is a long-lived bridge owned by a synchronous wrapper object::

    class SyncThing:
        def __init__(self) -> None:
            self._bridge = SyncLoopBridge()

        def do(self, x):
            return self._bridge.run(self._async_thing.do(x))

        def close(self):
            self._bridge.close()

For a one-off call that does not justify owning a bridge,
:func:`run_coro_sync` spins one up, runs the coroutine, and tears it down::

    result = run_coro_sync(some_coro())

The bridge costs one daemon thread for its lifetime. When many short-lived
synchronous callers each need a bridge, prefer sharing a single
long-lived bridge over spawning one per call (or per
:func:`run_coro_sync`).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from types import TracebackType
from typing import Any, Self, TypeVar

__all__ = ["SyncLoopBridge", "run_coro_sync"]

T = TypeVar("T")

# Name applied to the bridge's loop thread so tests (and debuggers) can assert
# no bridge thread is left alive after teardown.
_THREAD_NAME = "dk-sync-loop-bridge"


class SyncLoopBridge:
    """Run coroutines synchronously on a private background event loop.

    The bridge starts a daemon thread running its own
    :class:`asyncio.AbstractEventLoop` on construction.
    :meth:`run` submits a coroutine to that loop and blocks the **caller's**
    thread until it completes, returning the result or re-raising the
    coroutine's exception (with its traceback preserved). Because the loop
    runs on a separate thread, :meth:`run` is safe to call from a plain
    synchronous function *and* from inside an already-running event loop —
    the ``asyncio.run()`` / ``run_until_complete()`` "loop already running"
    footgun is avoided by construction.

    The loop thread is a ``daemon`` so it can never block process exit.
    Call :meth:`close` (or use the bridge as a context manager) for
    deterministic teardown: it stops the loop and joins the thread.

    A single bridge is reusable across many :meth:`run` calls and is the
    right shape for a long-lived synchronous wrapper. It is **not** designed
    for concurrent :meth:`run` calls that race :meth:`close` from other
    threads; ``run`` after ``close`` raises :class:`RuntimeError`.
    """

    def __init__(self, *, thread_name: str = _THREAD_NAME) -> None:
        """Start the background loop thread.

        Args:
            thread_name: Name for the loop's daemon thread. Defaults to a
                stable name tests can assert against; override to
                distinguish multiple bridges in diagnostics.
        """
        self._closed = False
        self._close_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, name=thread_name, daemon=True
        )
        self._thread.start()
        # Block construction until the loop is actually running, so the first
        # ``run`` cannot race a not-yet-started loop.
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()
        # ``run_forever`` returned -> ``close`` stopped the loop. Drain any
        # leftover async generators and close the loop here, on the loop's own
        # thread (the only thread allowed to close it cleanly).
        try:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        finally:
            self._loop.close()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` to completion on the background loop and return it.

        Blocks the calling thread until the coroutine finishes. The
        coroutine's return value is returned; an exception it raises is
        re-raised here with its original traceback. Safe to call from inside
        a running event loop (the coroutine runs on the bridge's separate
        loop, so there is no re-entrancy).

        Args:
            coro: The coroutine to run.

        Returns:
            Whatever ``coro`` returns.

        Raises:
            RuntimeError: If the bridge has been closed.
            BaseException: Whatever ``coro`` raises, re-raised in the caller.
        """
        if self._closed:
            # Close the coroutine so it does not leak / warn as never-awaited.
            coro.close()
            raise RuntimeError("SyncLoopBridge is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        """Stop the background loop and join its thread. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def run_coro_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a single coroutine to completion from synchronous code.

    Convenience wrapper that spins up a throwaway :class:`SyncLoopBridge`,
    runs ``coro`` on it, and tears it down — for one-off calls that do not
    justify owning a bridge. Like :meth:`SyncLoopBridge.run`, it is safe to
    call from inside a running event loop.

    Each call costs a short-lived daemon thread. For repeated calls from the
    same synchronous component, own a long-lived :class:`SyncLoopBridge` and
    reuse it instead.

    Args:
        coro: The coroutine to run.

    Returns:
        Whatever ``coro`` returns.

    Raises:
        BaseException: Whatever ``coro`` raises, re-raised in the caller.
    """
    with SyncLoopBridge() as bridge:
        return bridge.run(coro)
