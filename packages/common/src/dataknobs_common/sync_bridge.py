# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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
import logging
import sys
import threading
import time
import warnings
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, ClassVar, Self, TypeVar

__all__ = [
    "BridgedOperation",
    "OperationTimeoutError",
    "SyncBridgeAdapter",
    "SyncLoopBridge",
    "bridge_thread_names",
    "bridged_operation",
    "run_coro_sync",
]

T = TypeVar("T")

# Name applied to the bridge's loop thread so tests (and debuggers) can assert
# no bridge thread is left alive after teardown.
_THREAD_NAME = "dk-sync-loop-bridge"

# Every name a bridge has actually run under in this process, including the
# ones callers supplied. The leak guard matches threads *by name*, so a name it
# has never heard of is a thread it cannot see -- which is what made
# `SyncTextEmbedder`'s `dk-sync-embedder` invisible to it at every call site,
# silently. Registering here is what keeps `thread_name=` a diagnostic label
# rather than a way out of the guard.
_thread_names: set[str] = {_THREAD_NAME}

logger = logging.getLogger(__name__)

#: Seconds teardown spends letting cancelled tasks unwind before the loop is
#: closed anyway. Bounded rather than unbounded --- ``close`` joins the loop
#: thread, so a task that refuses to cancel would otherwise hang the closing
#: caller forever, which is a worse failure than the one this drain fixes. A
#: task that cancels promptly costs a single loop iteration, not this budget.
_TEARDOWN_DRAIN_SECONDS = 5.0


def _run_loop(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    """Body of a bridge's loop thread.

    Module-level, and taking the loop and the event rather than the bridge,
    because ``Thread`` holds its target for as long as the thread runs and a
    bound method holds ``self``. As a method this was a reference from the
    live thread back to the bridge that owns it --- and a bridge nobody
    closes runs forever, so that reference never went away. An unclosed
    bridge was therefore *unreclaimable*: unreachable from any caller, alive
    in ``threading._active``, and beyond the reach of any finalizer that
    might have said so. Taking two arguments is what makes
    :meth:`SyncLoopBridge.__del__` able to run at all.
    """
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()
    # ``run_forever`` returned -> ``close`` stopped the loop. Unwind what is
    # still on it and close it here, on the loop's own thread (the only thread
    # allowed to close it cleanly). Tasks first, then async generators: a
    # cancelled task's ``finally`` can be what closes a generator, and draining
    # the generators first would close it out from under the task.
    try:
        _cancel_pending_tasks(loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()


def _cancel_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel whatever is still running on ``loop`` and let it unwind.

    ``loop.close()`` destroys a pending task outright: its ``finally`` never
    runs and the only report is ``Task was destroyed but it is pending!`` on
    stderr. That ``finally`` is where a connection goes back to its pool, a
    transaction is rolled back, a lock is released --- so without this the
    bridge leaks exactly the resources a teardown exists to reclaim.

    Two shapes reach here, and neither is exotic. A coroutine that spawned a
    task and returned without awaiting it leaves one behind on a wholly
    *successful* ``run`` --- a pool's background maintenance is this shape. And
    a coroutine cancelled by :meth:`SyncLoopBridge.run`'s timeout needs more
    than the single loop iteration it used to get if its cleanup awaits more
    than once, which realistic cleanup does.

    This is what makes ``run``'s "not abandoned mid-flight" true on the close
    path as well as the timeout path. It mirrors what :func:`asyncio.run` does
    through :class:`asyncio.Runner`, with the wait bounded --- see
    :data:`_TEARDOWN_DRAIN_SECONDS`.
    """
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()
    # ``wait`` returns the two sets rather than raising, so a task that ignores
    # cancellation costs the budget and nothing else.
    _, still_pending = loop.run_until_complete(
        asyncio.wait(pending, timeout=_TEARDOWN_DRAIN_SECONDS)
    )
    if still_pending:
        # Reported rather than swallowed: these are about to be destroyed by
        # `loop.close()` with their cleanup unrun, which is the original defect
        # surviving in the one case this cannot fix. Naming them is the
        # difference between a diagnosable leak and a silent one.
        logger.warning(
            "SyncLoopBridge teardown: %d task(s) did not finish cancelling "
            "within %.1fs and were abandoned: %s",
            len(still_pending),
            _TEARDOWN_DRAIN_SECONDS,
            ", ".join(sorted(task.get_name() for task in still_pending)),
        )


def bridge_thread_names() -> frozenset[str]:
    """Every thread name a :class:`SyncLoopBridge` has used in this process.

    Grows as bridges are constructed, so a guard resolving it late sees a
    name first used after the guard started. Always contains the default.
    """
    # `set` copy under the GIL: `add` from another thread cannot interleave.
    return frozenset(_thread_names)


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
                distinguish multiple bridges in diagnostics. Registered in
                :func:`bridge_thread_names`, so naming a bridge does not
                hide it from the leaked-thread guard.
        """
        _thread_names.add(thread_name)
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
                target=_run_loop,
                args=(self._loop, self._ready),
                name=thread_name,
                daemon=True,
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
            RuntimeError: If the bridge has been closed, or if called from
                within a coroutine already running on the bridge's own loop
                (which would self-deadlock).
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
        if threading.current_thread() is self._thread:
            # Called from a coroutine already running ON the bridge loop (e.g. a
            # transform re-entering the same synchronous wrapper). Submitting to
            # this loop and then blocking on the result would wait for a loop
            # that cannot advance until we return — a self-deadlock. Raise
            # instead, symmetric with close()'s same-thread guard. Close the
            # coroutine so it does not warn as never-awaited.
            coro.close()
            raise RuntimeError(
                "SyncLoopBridge.run() must not be called from within a "
                "coroutine running on the bridge loop (it would deadlock); "
                "await the coroutine directly instead"
            )
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

    @property
    def is_closed(self) -> bool:
        """Whether :meth:`close` has been claimed. Never rises again once true.

        For a caller deciding whether this bridge is still somewhere a
        coroutine *can* be run --- a teardown path that must not raise, most
        of all one reached from a finalizer, where an exception has nowhere to
        go and is printed as "Exception ignored". It answers about the moment
        it was asked, so a ``run`` issued on the strength of it still races an
        in-flight ``close`` exactly as the class docstring says: quiesce, or
        pass a ``timeout``.
        """
        return self._closed

    def close(self) -> None:
        """Stop the background loop and join its thread. Idempotent.

        Whatever is still running on the loop is cancelled and given a bounded
        window to unwind before the loop is closed --- see
        :func:`_cancel_pending_tasks`. That is what makes :meth:`run`'s "not
        abandoned mid-flight" hold on this path too: without it a task whose
        cancellation had been requested but not yet delivered, or one a
        coroutine spawned and never awaited, was destroyed by ``loop.close()``
        with its ``finally`` unrun. Teardown therefore costs a loop iteration
        rather than nothing, and in the pathological case up to
        :data:`_TEARDOWN_DRAIN_SECONDS`.

        Safe to call from any thread except the bridge's own loop thread:
        calling ``close`` from inside a coroutine running on the bridge would
        have to join the current thread, which is impossible, so that raises
        :class:`RuntimeError` rather than deadlocking. Concurrent callers all
        block until teardown completes — every ``close`` returns only once the
        loop thread is actually gone.

        If teardown itself raises, the exception propagates to *this* caller
        and the waiters are released rather than left blocked: an exception
        during shutdown is a problem for the closer to handle, not grounds to
        strand every other holder of the bridge forever.
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
        # The event must be set on *every* exit from here, not just the happy
        # one. This caller has already claimed teardown by setting `_closed`,
        # so every later `close()` is committed to waiting on the event; if an
        # exception between here and the set skipped it, that wait would never
        # end. The failure would present as a permanent hang in an unrelated
        # caller, with the real cause — an exception raised over here — long
        # since propagated somewhere else.
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
        finally:
            self._closed_event.set()

    def __del__(self) -> None:
        """Warn about a bridge nobody closed, and best-effort close it.

        A leaked bridge is silent by construction --- the thread is a daemon
        so it cannot delay interpreter exit, the owning object goes on
        working, and nothing raises. What it costs is a thread and an event
        loop's self-pipe descriptors for the life of the process, once per
        bridge: a server building one per tenant accumulates both and finds
        out from neither an exception nor a log line. The ``ResourceWarning``
        is what makes that visible, and it is the reason this method exists;
        the teardown below is the repair.

        Deliberately conditional, because a finalizer runs at a moment
        nobody chose. :meth:`close` joins the loop thread, and a daemon
        thread is killed rather than joined once interpreter finalization
        starts --- so a join issued from here during shutdown would wait on a
        thread that can no longer answer. The warning is unconditional; only
        the join is skipped.
        """
        # `getattr`, not an attribute read: `__del__` also runs on an object
        # whose `__init__` raised before `_closed` was ever assigned, and a
        # finalizer that raises turns a leak into unignorable noise on stderr.
        if getattr(self, "_closed", True):
            return
        # B028 is waived below: a `stacklevel` in a finalizer points at whichever
        # frame happened to trigger collection, which has no relationship to
        # the code that failed to close the bridge. `source=self` is the
        # locator that means something for a ResourceWarning, and is what
        # CPython's own `BaseEventLoop.__del__` uses for the same reason.
        warnings.warn(  # noqa: B028
            f"unclosed SyncLoopBridge (loop thread {self._thread.name!r}); "
            f"call close() or use the bridge as a context manager",
            ResourceWarning,
            source=self,
        )
        if sys.is_finalizing() or threading.current_thread() is self._thread:
            return
        try:
            self.close()
        except Exception:  # pragma: no cover - teardown is already best-effort
            # Swallowed rather than logged: an exception here is reported by
            # the interpreter as "Exception ignored in __del__" regardless,
            # and the caller has already been told about the leak by the
            # warning above. Whatever went wrong will recur, diagnosably, for
            # anyone who calls `close()` at a moment of their choosing.
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class SyncBridgeAdapter:
    """Shared shape for a synchronous wrapper over an asynchronous object.

    Three classes in three packages reached an async object from synchronous
    code by holding a :class:`SyncLoopBridge`, and each declared the same
    surface by hand: a keyword-only ``timeout``, a bridge under a per-class
    thread name, ``close()``, and the context-manager pair. Nothing declared
    the shape, so it drifted in both directions --- the third was written by
    copying the first two, and then grew ``bridge=``, ``aclose()``, lazy
    construction and a use-after-close guard that the first two did not have.
    This class is that shape, declared once. A subclass supplies only what
    varies: the object it wraps, the methods that forward to it, and --- if it
    *owns* that object --- how to close it.

    A subclass names its loop thread with :attr:`BRIDGE_THREAD_NAME` and
    forwards through :meth:`_run`::

        class SyncThing(SyncBridgeAdapter):
            BRIDGE_THREAD_NAME = "dk-sync-thing"

            def __init__(self, inner: AsyncThing, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._inner = inner

            def do(self, x: int) -> str:
                return self._run(self._inner.do(x))

    The wrapped object is deliberately **not** stored here. It is the one
    thing that genuinely varies --- each of the three names it differently and
    one of them exposes it publicly --- and a base that owned it would force a
    rename on a published attribute to buy nothing.

    "Callable from inside a running loop" means it does not *deadlock*. It
    still **blocks**: the calling thread waits on the bridge's result for the
    whole call, so every other task on the caller's loop is stalled meanwhile.
    From async code, await the wrapped object directly; a subclass of this is
    for the ``def`` sites that cannot. :meth:`aclose` exists so that an async
    holder's *teardown*, at least, does not pay that cost, and ``async with``
    is its context-manager form --- both protocols are here because both
    teardowns are.

    **Quiesce the callers before closing.** :class:`SyncLoopBridge` states the
    rule for itself and it reaches every subclass of this: a ``run`` that races
    an in-flight ``close`` from another thread is undefined. The shape that
    hits it is the one ``async with`` invites --- an async holder handing this
    to a ``def`` site through :func:`asyncio.to_thread`, which cannot cancel
    the thread it started. Cancel the holding task and the body unwinds while
    the worker is still inside a call, and teardown stops the loop under it;
    ``timeout=`` is then the only upper bound that worker has. Join or
    cancel-and-await the workers before leaving the block.

    Concurrent *closers* are safe, and are the one race this class resolves
    rather than forwards: exactly one caller of :meth:`close` or :meth:`aclose`
    tears the wrapped object down and the rest wait for it, so two holders
    closing at once cannot stop the loop under each other's teardown.
    """

    #: Loop-thread name for this class's bridge. Registered by the bridge
    #: itself, so the leaked-thread guard sees it rather than being escaped by
    #: it; it is a diagnostic label, not an opt-out. Every subclass must set
    #: it --- see :meth:`__init_subclass__` for why it is required rather than
    #: defaulted.
    BRIDGE_THREAD_NAME: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass that did not name its loop thread.

        The attribute's whole stated purpose is that a stack dump names which
        wrapper allocated the thread. A default would be the shared
        ``dk-sync-loop-bridge`` that :func:`run_coro_sync`'s throwaway bridges
        already use, so a subclass that forgot would lose exactly the
        diagnostic this exists for --- silently, while still passing the
        leaked-thread guard, because the name it reported was a real
        registered name belonging to something else.

        Inherited names are fine: an intermediate subclass names the thread
        and its specialisations share it, which is the same wrapper as far as
        a stack dump is concerned.
        """
        super().__init_subclass__(**kwargs)
        if not cls.BRIDGE_THREAD_NAME:
            raise TypeError(
                f"{cls.__name__} must set BRIDGE_THREAD_NAME; it is what names "
                f"the daemon thread this wrapper allocates"
            )

    def __init__(
        self, *, bridge: SyncLoopBridge | None = None, timeout: float | None = None
    ) -> None:
        """Args:
        bridge: A bridge to run this object's coroutines on. The default
            builds a private one on first use and ends it in :meth:`close`.
            A bridge handed in here belongs to the caller: it is shared with
            whatever else uses it, and :meth:`close` leaves it running.
        timeout: Seconds to allow each call, giving a synchronous caller an
            upper bound on a blocking wait it cannot otherwise cancel.
            ``None`` (the default) waits as long as the call takes.
        """
        self._timeout = timeout
        self._owns_bridge = bridge is None
        self._bridge = bridge
        # Guards the bridge slot *and* the closed flag together, because the
        # two are one decision: whether this object may still reach a loop.
        # Held only across the slot and the flag, never across a teardown or
        # a `run` --- see `close`, which claims under it and then releases.
        self._lock = threading.Lock()
        self._closed = False
        # Set by whichever caller claimed teardown, once the bridge is
        # actually down. The same shape as `SyncLoopBridge._closed_event` and
        # for the same reason: a second closer must wait for teardown to
        # finish rather than return while it is still in flight.
        self._teardown_done = threading.Event()

    def _ensure_bridge(self) -> SyncLoopBridge:
        """This object's bridge, built on first use.

        Deferred because construction is also how a consumer reaches whatever
        the subclass forwards *without* awaiting --- a model id, a capability
        set, a provider name --- and a daemon thread per discovery-only
        construction is a cost those callers never asked for.

        Refuses once closed, and the refusal is *inside* the lock, which is
        what makes "no bridge is constructed after teardown" an invariant
        rather than a check with a window. :meth:`_run`'s own guard cannot
        carry that alone: it reads the flag and asks for the bridge in two
        steps, and a ``close()`` landing between them finds the slot still
        empty, ends nothing, and leaves the call it raced free to allocate a
        thread nothing will ever join.
        """
        bridge = self._bridge
        if bridge is not None:
            return bridge
        with self._lock:
            if self._closed:
                raise RuntimeError(self._closed_message())
            if self._bridge is None:
                self._bridge = SyncLoopBridge(thread_name=self.BRIDGE_THREAD_NAME)
            return self._bridge

    def _closed_message(self) -> str:
        """The one wording for a refusal, so the two guards cannot disagree."""
        return f"{type(self).__name__} is closed; build another rather than reusing this one"

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on this object's bridge and return its result.

        The one place the bridge is reached, so a wrapper's forwarding methods
        cannot disagree about which loop they run on or whether ``timeout``
        applies. One copy of that decision per method is what made the
        provider adapter's version a six-method defect rather than a
        one-method one.
        """
        if self._closed:
            # A fast refusal for the common case --- an object closed before
            # it was ever used --- so a caller does not pay for the lock to be
            # told no. It is not the guarantee: `_ensure_bridge` re-checks
            # under the lock, which is what closes the interleaved case.
            coro.close()
            raise RuntimeError(self._closed_message())
        try:
            bridge = self._ensure_bridge()
        except BaseException:
            # Symmetric with the bridge's own refusal paths: a coroutine that
            # is never handed to a loop must be closed, or it warns as
            # never-awaited from wherever collection happens to run.
            coro.close()
            raise
        return bridge.run(coro, timeout=self._timeout)

    def _run_teardown(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a *teardown* coroutine on this object's bridge.

        The one call allowed after the object is marked closed, and the reason
        it exists rather than being a rule about :meth:`_ensure_bridge`: an
        owning subclass has to reach its wrapped object from inside
        :meth:`_close_inner`, which runs with ``_closed`` already set, and
        :meth:`_run` refuses there by design. A subclass reaching for the
        obvious ``self._run(...)`` in its own teardown would raise from inside
        it; this is the method that means it does not have to know that.

        Builds the bridge if there is not one, because a wrapper closed
        without ever being used still owns an object that has to be closed
        *somewhere*, and the bridge is the only loop this object has. Whatever
        it builds, :meth:`_end_bridge` then ends --- the two share the lock and
        the slot, so the teardown cannot outlive itself.
        """
        with self._lock:
            bridge = self._bridge
            if bridge is None:
                bridge = self._bridge = SyncLoopBridge(thread_name=self.BRIDGE_THREAD_NAME)
        return bridge.run(coro, timeout=self._timeout)

    def _close_inner(self) -> None:
        """Close the wrapped object, synchronously. Default: it is not ours.

        Two of the three wrappers are handed an object somebody else built and
        must not close it; the third is what a factory returns and owns what
        it wraps. That is ownership, not drift, which is why it is a hook
        rather than a shared body.

        An override reaches the wrapped object through :meth:`_run_teardown`,
        not :meth:`_run`: this runs with ``_closed`` already set, and ``_run``
        refuses there.
        """
        return None

    async def _aclose_inner(self) -> None:
        """Close the wrapped object from async code. Default: it is not ours.

        The async twin of :meth:`_close_inner`. What "async" buys here is that
        the holder's **loop** is not blocked for the teardown; it does not
        follow that the teardown belongs on that loop. A subclass whose
        wrapped object holds loop-bound state --- an HTTP session opened by an
        ``initialize()`` that went through :meth:`_run`, and therefore bound to
        the *bridge's* loop --- must still close it there, and reaches it with
        ``await asyncio.to_thread(self._close_inner)``. Awaiting the object
        directly is right only when nothing it holds is bound to a loop.
        """
        return None

    def _claim_teardown(self) -> bool:
        """Claim the right to tear this object down. ``False`` if another has.

        Atomic, because the alternative --- reading a bare flag, then setting
        it --- lets a second closer conclude the wrapped object is somebody
        else's problem and go straight on to end the bridge, under a teardown
        that is still running on it.
        """
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            return True

    def close(self) -> None:
        """Close the wrapped object if it is ours, then this object's bridge.

        Idempotent, and safe to call from several threads at once. Exactly one
        caller closes the wrapped object; the others wait for it to finish
        rather than racing past it --- a second closer that merely *arrives*
        while the first is inside the hook used to go straight on to end the
        bridge, stopping the loop under a teardown still running on it and
        leaving the first blocked on a future nothing would ever set.

        The bridge is then asked on **every** call, waiters included, because
        the bridge is already idempotent and concurrency-safe and it is the
        thing that actually holds the thread. Ending it only on the claiming
        call is how a teardown that raises part way --- reachable through the
        re-entrancy :meth:`SyncLoopBridge.run` documents by name --- leaves the
        flag set, the loop thread running, and every later ``close()``
        returning at the flag without ever reaching it. Unrecoverable, since
        nothing else holds a reference to that thread.
        """
        try:
            if self._claim_teardown():
                try:
                    self._close_inner()
                finally:
                    # Set on *every* exit, not just the happy one: once a
                    # caller has claimed teardown every other close is
                    # committed to waiting here, so an exception that skipped
                    # this would strand them permanently --- presenting as a
                    # hang in an unrelated caller, with the real cause
                    # propagated somewhere else entirely.
                    self._teardown_done.set()
            else:
                self._teardown_done.wait()
        finally:
            self._end_bridge()

    async def aclose(self) -> None:
        """Close from async code: the wrapped object first, then the bridge.

        The async twin of :meth:`close`, and the reason it exists is that
        :meth:`close` reaches the wrapped object *through* the bridge --- so an
        async holder calling it blocks its own event loop for that object's
        entire teardown, which on a real provider is an HTTP round trip. Here
        the subclass decides how to reach it (:meth:`_aclose_inner`) and the
        holder's loop stays free either way.

        What is left on the loop is :meth:`_end_bridge`: a ``stop`` and a
        thread join. Usually microseconds, but *not* guaranteed free of I/O ---
        the bridge drains its async generators on the way down, so an
        abandoned stream's ``finally`` runs inside that join. A holder that
        cannot afford it wraps this call in :func:`asyncio.to_thread`.

        Shares :meth:`_claim_teardown` and :meth:`_end_bridge` with
        :meth:`close`, so the two cannot come to disagree about whose bridge it
        is or who is tearing down. A caller that loses the claim waits off the
        loop, for the same reason the teardown itself does.
        """
        try:
            if self._claim_teardown():
                try:
                    await self._aclose_inner()
                finally:
                    self._teardown_done.set()
            else:
                await asyncio.to_thread(self._teardown_done.wait)
        finally:
            self._end_bridge()

    def _end_bridge(self) -> None:
        """End the bridge iff this object built it.

        A bridge handed to the constructor is shared --- ending it here would
        close it under whatever else is using it.

        Reads the slot under the lock so it cannot miss a bridge
        :meth:`_run_teardown` built a moment earlier, and releases it before
        closing: ``SyncLoopBridge.close`` joins a thread, and holding this
        object's lock across that join would stall every caller for its
        duration to tell them all the same thing.
        """
        with self._lock:
            bridge = self._bridge if self._owns_bridge else None
        if bridge is not None:
            bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        """Entry for an async holder, pairing with :meth:`aclose`.

        Both protocols are here because both teardowns are, and a teardown
        with no context-manager form is one a holder writes ``try``/``finally``
        around by hand --- where the cost of getting it wrong is a leaked
        daemon thread, which is the failure this whole class exists to make
        hard. A synchronous holder writes ``with`` and pays the bridged
        teardown it was always going to pay; an async holder writes
        ``async with`` and does not block its loop for it.

        Refusing one of the two --- the shape ``AsyncLLMProvider`` takes, where
        ``__enter__`` raises ``TypeError`` --- is right for an object only one
        kind of holder can have, and wrong here. A wrapper whose whole purpose
        is to be *called* synchronously is routinely built and torn down by
        async code that hands it to a ``def`` site in a worker thread, so both
        kinds of holder are real and neither entry is a mistake.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


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
        timeout: Maximum seconds to wait for ``coro``, forwarded to
            :meth:`SyncLoopBridge.run`. ``None`` (the default) waits forever.
            It bounds the coroutine, not this call: the throwaway bridge is
            torn down afterwards, and that waits up to
            :data:`_TEARDOWN_DRAIN_SECONDS` for a cancelled ``coro`` to unwind
            rather than destroying its cleanup mid-flight. Prompt cancellation
            costs one loop iteration; the worst case is ``timeout`` plus the
            drain.

    Returns:
        Whatever ``coro`` returns.

    Raises:
        TimeoutError: If ``timeout`` elapses before the coroutine finishes.
        BaseException: Whatever ``coro`` raises, re-raised in the caller.
    """
    with SyncLoopBridge() as bridge:
        return bridge.run(coro, timeout=timeout)


class OperationTimeoutError(TimeoutError):
    """A :class:`BridgedOperation`'s deadline expired.

    A subclass of the **builtin** ``TimeoutError`` --- deliberately, and not of
    ``dataknobs_common.exceptions.TimeoutError``, which is a ``DataknobsError``
    that shadows the builtin name. What :meth:`SyncLoopBridge.run` raises for
    an expired wait is the builtin, so that is what this must widen if
    ``except TimeoutError`` around a bridged call is to keep catching both.

    The distinct type exists for the caller that must tell *this operation ran
    out of time* apart from *the work raised a timeout of its own*: a batch
    executor whose per-item handler absorbs an item's failure has to let the
    operation's deadline through that handler, and a bare ``TimeoutError``
    cannot say which of the two it is.
    """


@dataclass(frozen=True)
class BridgedOperation:
    """The loop one synchronous call runs on, and the budget it runs within.

    A synchronous wrapper over an asynchronous object reaches that object more
    than once per public call --- a chunked write per chunk, a batch per item,
    a stream per record. Both of the things that govern those reaches belong
    to the *operation* rather than to the wrapper:

    **The loop**, because an object can bind state to the first loop it runs
    on and then only that loop will do. An ``asyncpg`` pool acquired by
    ``connect()`` belongs to the loop that acquired it; a second loop finds it
    unusable, with an error naming the connection rather than the loop.

    **The budget**, because a timeout applied per reach is not a bound on the
    call the caller made. Spent afresh on each round trip, a 30-second bound on
    a twenty-chunk write permits ten minutes.

    Carrying both here rather than on the wrapper is what lets one wrapper
    serve two concurrent operations: instance state would have them writing
    each other's deadline.

    Build one with :func:`bridged_operation`, which also decides whether the
    operation owns its bridge or borrows the caller's.
    """

    #: The loop this operation runs on. ``None`` when the operation's work is
    #: synchronous and reaches no loop at all --- a wrapper that fronts either
    #: flavour still has a deadline, and still needs somewhere to put it.
    bridge: SyncLoopBridge | None

    #: ``time.monotonic()`` value past which this operation stops waiting.
    #: ``None`` when the caller set no timeout, which waits for as long as the
    #: work takes.
    deadline: float | None

    #: What the operation is called, for the message a caller reads when the
    #: deadline expires several frames from where they set it.
    label: str = "The operation"

    @property
    def remaining(self) -> float | None:
        """Seconds left in the budget, or ``None`` when it is unbounded.

        Can be zero or negative: a caller past its deadline should refuse
        rather than start another reach, which is what :meth:`run` does.
        """
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Drive one coroutine on this operation's loop, within its budget.

        Args:
            coro: The coroutine to run. It is closed rather than started when
                the budget is already spent, so nothing warns about a
                coroutine that was never awaited.

        Returns:
            Whatever ``coro`` returns.

        Raises:
            OperationTimeoutError: If the budget was already spent before this
                call, or expired while waiting for it.
            RuntimeError: If this operation has no bridge, which means a
                synchronous wrapper reached its asynchronous path.
            BaseException: Whatever ``coro`` raises, re-raised in the caller.
        """
        if self.bridge is None:
            coro.close()
            raise RuntimeError(
                f"{self.label} has no operation loop --- its work was detected as "
                "synchronous, so this asynchronous path should be unreachable"
            )
        remaining = self.remaining
        if remaining is not None and remaining <= 0:
            # Already over budget: refuse without reaching the object at all,
            # so a caller past its deadline stops paying for round trips it
            # has already decided not to wait for.
            coro.close()
            raise OperationTimeoutError(f"{self.label} exceeded its timeout before this call ran")
        try:
            return self.bridge.run(coro, timeout=remaining)
        except OperationTimeoutError:
            # Someone else's deadline, always. A bridge reports an expired wait
            # as the *builtin* ``TimeoutError``; this type is only ever
            # constructed here, with the label of the operation that ran out.
            # So one arriving from the coroutine belongs to an operation nested
            # inside this one, and relabelling it would erase the deadline that
            # actually expired and name the wrong one.
            raise
        except TimeoutError:
            # The bridge raises the builtin for a wait that expired, and
            # re-raises a ``TimeoutError`` the coroutine itself raised through
            # the same channel. The deadline is what tells them apart: only
            # the first can have consumed the whole budget.
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise OperationTimeoutError(
                    f"{self.label} exceeded its timeout while waiting for this call"
                ) from None
            raise


@contextmanager
def bridged_operation(
    *,
    bridge: SyncLoopBridge | None = None,
    timeout: float | None = None,
    thread_name: str = _THREAD_NAME,
    label: str = "The operation",
    needs_loop: bool = True,
) -> Iterator[BridgedOperation]:
    """Open the loop and the time budget one synchronous call runs within.

    A caller-supplied ``bridge`` is used as-is and **left running**: it
    belongs to whoever passed it, who may be running other things on it.
    Otherwise the operation owns a bridge for its duration and leaving this
    block ends it --- including on the error paths, which is why this is a
    context manager rather than a pair of calls.

    Scope it to the **public entry point**, not to the individual reach: pass
    the yielded operation down to the private workers, so a call that reaches
    its object several times drives all of them on one loop and spends one
    budget across them.

    Args:
        bridge: A loop the caller owns. Required for an object that binds
            state to the loop that connected it, since a wrapper handed such
            an object has no way to reach that loop for itself. ``None`` (the
            default) gives the operation a bridge of its own.
        timeout: Seconds to allow the whole operation --- a bound on the
            *work*, spent across every reach the call makes. ``None`` (the
            default) waits for as long as the work takes.

            It does not cover teardown of an **owned** bridge, which happens
            after the budget is already spent: leaving this block cancels
            whatever the expired call left running and waits up to
            :data:`_TEARDOWN_DRAIN_SECONDS` for it to unwind, so the worst
            case a caller can observe is ``timeout`` plus that. A coroutine
            that cancels promptly costs one loop iteration and the difference
            is unmeasurable; one whose cleanup awaits something slow, or
            ignores cancellation, costs the whole drain. The alternative is
            destroying that cleanup mid-flight, which is the defect the drain
            exists to fix --- so the bound is deliberately on the work rather
            than on the call. A supplied bridge is not closed here and adds
            nothing.
        thread_name: Name for an owned bridge's loop thread. A diagnostic
            label, not a way out of the leak guard --- the bridge registers
            every name it runs under, so
            ``assert_no_leaked_bridge_threads`` watches this one too. The
            default is the same throwaway name :func:`run_coro_sync` uses,
            which is honest for an unnamed one-off; name it after the wrapper
            wherever a stack trace or a thread dump would otherwise be
            ambiguous.
        label: What the operation is called in a timeout message.
        needs_loop: ``False`` for a call whose work turns out to be
            synchronous. It then yields an operation carrying the deadline and
            no bridge, so no thread is allocated for a loop nothing will use.

    Yields:
        The :class:`BridgedOperation` for this call.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    if not needs_loop or bridge is not None:
        yield BridgedOperation(
            bridge=bridge if needs_loop else None, deadline=deadline, label=label
        )
        return
    with SyncLoopBridge(thread_name=thread_name) as owned:
        yield BridgedOperation(bridge=owned, deadline=deadline, label=label)
