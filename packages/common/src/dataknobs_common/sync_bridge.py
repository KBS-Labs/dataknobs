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
    right shape for a long-lived synchronous wrapper. Concurrent :meth:`run`
    calls from multiple threads are supported — each coroutine runs on the
    one shared loop and its caller blocks on its own result.

    Ordering between :meth:`run` and :meth:`close` is the caller's
    responsibility: a ``run`` issued strictly *after* ``close`` raises
    :class:`RuntimeError`, but a ``run`` that *races* an in-flight ``close``
    from another thread is undefined (it may raise, or block until its
    ``timeout`` elapses). Quiesce ``run`` callers before closing, and pass a
    ``timeout`` to ``run`` if you need a guaranteed upper bound on the wait.
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
        # Set only after the winning ``close`` has stopped the loop and joined
        # the thread, so a concurrent second closer waits for teardown to
        # finish instead of returning while the thread is still alive.
        self._closed_event = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        try:
            self._thread = threading.Thread(
                target=self._run_loop, name=thread_name, daemon=True
            )
            self._thread.start()
        except BaseException:
            # Thread creation/start failed -> close the loop we just created so
            # its self-pipe file descriptors do not leak.
            self._loop.close()
            raise
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

    def run(self, coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        """Run ``coro`` to completion on the background loop and return it.

        Blocks the calling thread until the coroutine finishes. The
        coroutine's return value is returned; an exception it raises is
        re-raised here with its original traceback. Safe to call from inside
        a running event loop (the coroutine runs on the bridge's separate
        loop, so there is no re-entrancy).

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait for the coroutine. ``None``
                (the default) waits forever. On timeout, the still-running
                coroutine is asked to cancel (best-effort — it may already be
                past its last ``await`` point) and :class:`TimeoutError` is
                raised; the bridge remains usable for further ``run`` calls.

        Returns:
            Whatever ``coro`` returns.

        Raises:
            RuntimeError: If the bridge has been closed.
            TimeoutError: If ``timeout`` elapses before the coroutine finishes.
            BaseException: Whatever ``coro`` raises, re-raised in the caller.

        Note:
            If the calling thread is interrupted (e.g. ``KeyboardInterrupt``)
            or times out while blocked here, the interrupt/timeout reaches the
            *caller*, but the coroutine keeps running on the bridge loop until
            it completes or its best-effort cancellation takes effect — it is
            not abandoned mid-flight. This mirrors
            :func:`asyncio.run_coroutine_threadsafe` semantics.
        """
        if self._closed:
            # Close the coroutine so it does not leak / warn as never-awaited.
            coro.close()
            raise RuntimeError("SyncLoopBridge is closed")
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Lost a race with ``close`` — the loop was torn down between the
            # check above and submission. Close the coroutine so it does not
            # warn as never-awaited, and report the closed state uniformly.
            coro.close()
            raise RuntimeError("SyncLoopBridge is closed") from None
        try:
            return future.result(timeout)
        except TimeoutError:
            # Best-effort: ask the loop to cancel the still-running task so it
            # does not run unbounded after the caller has stopped waiting.
            future.cancel()
            raise

    def close(self) -> None:
        """Stop the background loop and join its thread. Idempotent.

        Safe to call from any thread except the bridge's own loop thread:
        calling ``close`` from inside a coroutine running on the bridge would
        have to join the current thread, which is impossible, so that raises
        :class:`RuntimeError` rather than deadlocking. Concurrent callers all
        block until teardown completes — every ``close`` returns only once the
        loop thread is actually gone.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "SyncLoopBridge.close() must not be called from within a "
                "coroutine running on the bridge loop"
            )
        with self._close_lock:
            already_closing = self._closed
            self._closed = True
        if already_closing:
            # Another thread owns teardown; wait for it to finish so this call
            # also returns only after the thread has been joined.
            self._closed_event.wait()
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._closed_event.set()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def run_coro_sync(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
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
        timeout: Maximum seconds to wait, forwarded to
            :meth:`SyncLoopBridge.run`. ``None`` (the default) waits forever.

    Returns:
        Whatever ``coro`` returns.

    Raises:
        TimeoutError: If ``timeout`` elapses before the coroutine finishes.
        BaseException: Whatever ``coro`` raises, re-raised in the caller.
    """
    with SyncLoopBridge() as bridge:
        return bridge.run(coro, timeout=timeout)
