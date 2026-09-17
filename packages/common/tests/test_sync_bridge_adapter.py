"""Behavioural tests for :class:`SyncBridgeAdapter`, the shape declared once.

Three wrappers in three packages hand-wrote this surface, and the third grew
four members the first two never got. These pin what a subclass now inherits,
so the next divergence fails here rather than being discovered by a consumer
holding two of them and counting threads.

The subclasses below are real implementations, not fakes: the thing under test
*is* the base class, and a subclass of it is the only way to exercise one.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import warnings

import pytest

from dataknobs_common import SyncBridgeAdapter, SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads, live_dk_daemon_threads

_THREAD = "dk-test-adapter"


class _Inner:
    """An async object with a teardown, so ownership is observable."""

    def __init__(self) -> None:
        self.closed = 0
        #: Which thread ran the teardown --- the bridge's loop thread when it
        #: was reached through ``close()``, the caller's when awaited by
        #: ``aclose()``. The difference between those two *is* what the async
        #: half buys, and no assertion on ``closed`` alone can see it.
        self.closed_on: str | None = None

    async def echo(self, value: int) -> int:
        await asyncio.sleep(0)
        return value

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed += 1
        self.closed_on = threading.current_thread().name


class _Borrower(SyncBridgeAdapter):
    """Wraps an object it does not own — the embedder/resolver shape."""

    BRIDGE_THREAD_NAME = _THREAD

    def __init__(self, inner: _Inner, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._inner = inner

    def echo(self, value: int) -> int:
        return self._run(self._inner.echo(value))


class _Owner(_Borrower):
    """Wraps an object it *does* own — the provider-adapter shape."""

    def _close_inner(self) -> None:
        self._ensure_bridge().run(self._inner.close())

    async def _aclose_inner(self) -> None:
        await self._inner.close()


def test_a_subclass_forwards_off_a_loop() -> None:
    with _Borrower(_Inner()) as sync:
        assert sync.echo(7) == 7


async def test_a_subclass_forwards_from_inside_a_running_loop() -> None:
    """The whole point of the bridge, inherited rather than restated."""
    with _Borrower(_Inner()) as sync:
        assert sync.echo(7) == 7


def test_constructing_one_spawns_no_thread() -> None:
    """Lazy, so a consumer who only reads a model id pays nothing.

    Two of the three adopters built their bridge eagerly in ``__init__``
    before this base existed, so constructing one to ask its ``model_id``
    cost a daemon thread for the object's life.
    """
    before = set(live_dk_daemon_threads({_THREAD}))
    sync = _Borrower(_Inner())
    try:
        assert set(live_dk_daemon_threads({_THREAD})) == before
        sync.echo(1)
        assert set(live_dk_daemon_threads({_THREAD})) != before
    finally:
        sync.close()


def test_the_subclass_names_the_thread() -> None:
    """``BRIDGE_THREAD_NAME`` is a diagnostic label, and it is registered.

    Registered by the bridge itself, so the leak guard watches it. A name the
    guard has never heard of is a thread it cannot see.

    The reference is *held* rather than dropped, because the guard
    under-reports a dropped one: ``SyncLoopBridge`` closes itself from
    ``__del__``, so an adapter whose last reference dies inside the block is
    already torn down before the check runs. What the guard names is a leak
    the test is still holding, which is the case pinned here.
    """
    leaked: _Borrower | None = None
    try:
        with pytest.raises(AssertionError, match=_THREAD):
            with assert_no_leaked_bridge_threads():
                leaked = _Borrower(_Inner())
                leaked.echo(1)
    finally:
        if leaked is not None:
            leaked.close()


def test_a_shared_bridge_survives_one_holders_close() -> None:
    """``bridge=`` is what makes two wrappers cost one thread rather than two."""
    with SyncLoopBridge(thread_name=_THREAD) as shared:
        a = _Borrower(_Inner(), bridge=shared)
        b = _Borrower(_Inner(), bridge=shared)
        assert a.echo(1) == 1
        assert b.echo(2) == 2

        a.close()
        assert b.echo(3) == 3, "closing one holder must not end a bridge it borrowed"
        b.close()
        assert shared.run(asyncio.sleep(0, result=4)) == 4


def test_an_owned_bridge_ends_with_the_adapter() -> None:
    sync = _Borrower(_Inner())
    sync.echo(1)
    assert live_dk_daemon_threads({_THREAD})
    sync.close()
    assert not live_dk_daemon_threads({_THREAD})


def test_use_after_close_refuses_and_leaks_nothing() -> None:
    """The guard exists because the failure it prevents is unrecoverable.

    Without it, an adapter closed before it was ever used would build a
    *second* bridge on first call — the first was never built, so there is no
    closed bridge left to refuse — and leak that one by construction, since
    ``close()`` has already run. The coroutine is closed rather than dropped,
    or it warns as never-awaited from wherever collection happens to land.
    """
    sync = _Borrower(_Inner())
    sync.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="_Borrower is closed"):
            sync.echo(1)
        gc.collect()
    dropped = [w for w in caught if "never awaited" in str(w.message)]
    assert not dropped, f"the refused coroutine was dropped rather than closed: {dropped}"
    assert not live_dk_daemon_threads({_THREAD})


def test_close_is_idempotent() -> None:
    inner = _Inner()
    sync = _Owner(inner)
    sync.echo(1)
    sync.close()
    sync.close()
    assert inner.closed == 1, "the wrapped object is closed once, not once per call"


def test_a_borrower_does_not_close_what_it_was_handed() -> None:
    """Ownership is the one variation the base leaves to a hook."""
    inner = _Inner()
    with _Borrower(inner) as sync:
        sync.echo(1)
    assert inner.closed == 0


def test_an_owner_closes_what_it_built() -> None:
    inner = _Inner()
    with _Owner(inner) as sync:
        sync.echo(1)
    assert inner.closed == 1


def test_a_teardown_that_raises_still_ends_the_bridge() -> None:
    """The reason ``_closed`` is set before the hook and the bridge after it.

    An early return on a flag beside the bridge is how a teardown that raises
    part way leaves the flag true, the loop thread running, and every later
    ``close()`` returning without ever reaching it — unrecoverable, since
    nothing else holds a reference to the thread.
    """

    class _Hostile(_Borrower):
        def _close_inner(self) -> None:
            raise ValueError("teardown blew up")

    sync = _Hostile(_Inner())
    sync.echo(1)
    assert live_dk_daemon_threads({_THREAD})
    with pytest.raises(ValueError, match="teardown blew up"):
        sync.close()
    assert not live_dk_daemon_threads({_THREAD}), "the thread outlived a raising teardown"


def test_close_reaches_the_inner_through_the_bridge() -> None:
    """The control for the test below: synchronously, teardown *is* bridged.

    Pinned because it is what the async half is measured against. A sync
    holder has no loop of its own to protect, so paying the bridge here is
    right — and asserting it makes the following test a difference rather
    than an unanchored claim.
    """
    inner = _Inner()
    with _Owner(inner) as sync:
        sync.echo(1)
    assert inner.closed_on == _THREAD


async def test_aclose_closes_the_inner_without_bridging_it() -> None:
    """``aclose`` is why an async holder does not stall its own loop to tear down.

    ``close()`` reaches the wrapped object *through* the bridge, so an async
    holder calling it blocks its loop for that object's entire teardown — an
    HTTP round trip on a real provider. Here it is awaited directly, which is
    checked structurally: the teardown ran on *this* thread, not on the loop
    thread the control above names. Counting ``closed`` cannot tell the two
    paths apart, since both of them close it exactly once.
    """
    inner = _Inner()
    sync = _Owner(inner)
    sync.echo(1)
    await sync.aclose()
    assert inner.closed == 1
    assert inner.closed_on == threading.current_thread().name
    assert inner.closed_on != _THREAD
    assert not live_dk_daemon_threads({_THREAD})


async def test_async_with_tears_down_the_way_aclose_does() -> None:
    """``async with`` is the form an async holder actually writes.

    Without it the holder writes the ``try``/``finally`` by hand, and the cost
    of writing it wrong is a leaked daemon thread — the failure this class
    exists to make hard, reintroduced at the one site the class does not
    cover.
    """
    inner = _Inner()
    async with _Owner(inner) as sync:
        assert sync.echo(1) == 1
    assert inner.closed == 1
    assert inner.closed_on != _THREAD, "``async with`` must not bridge the teardown"
    assert not live_dk_daemon_threads({_THREAD})


async def test_async_with_tears_down_when_the_body_raises() -> None:
    """Exception safety is the whole of what the protocol adds over ``aclose``."""
    inner = _Inner()
    with pytest.raises(ValueError, match="body blew up"):
        async with _Owner(inner) as sync:
            sync.echo(1)
            raise ValueError("body blew up")
    assert inner.closed == 1
    assert not live_dk_daemon_threads({_THREAD})


async def test_both_protocols_are_present_and_pick_different_teardowns() -> None:
    """One object, two kinds of holder — neither entry is a mistake.

    An async holder that builds one of these to hand to a ``def`` site in a
    worker thread can still write the synchronous form, and gets the bridged
    teardown when it does. That is a cost, not a bug, which is why ``with``
    is left working rather than refused the way ``AsyncLLMProvider`` refuses
    it — there, sync entry is *always* wrong; here it is the majority case.
    """
    bridged = _Inner()
    with _Owner(bridged) as sync:
        sync.echo(1)
    assert bridged.closed_on == _THREAD

    awaited = _Inner()
    async with _Owner(awaited) as sync:
        sync.echo(1)
    assert awaited.closed_on == threading.current_thread().name


async def test_aclose_is_idempotent_with_close() -> None:
    inner = _Inner()
    sync = _Owner(inner)
    sync.echo(1)
    await sync.aclose()
    sync.close()
    assert inner.closed == 1


def test_timeout_bounds_a_blocking_wait() -> None:
    """A synchronous caller has no other way to cancel one."""

    class _Slow(_Inner):
        async def echo(self, value: int) -> int:
            await asyncio.sleep(5)
            return value

    with _Borrower(_Slow(), timeout=0.05) as sync:
        with pytest.raises(TimeoutError):
            sync.echo(1)
