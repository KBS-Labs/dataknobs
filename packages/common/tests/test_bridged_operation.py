# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""One loop and one budget, for the span of one synchronous call.

A synchronous wrapper over an asynchronous object reaches that object more
than once per public call --- a chunked write per chunk, a batch per item, a
stream per record --- and both of the things governing those reaches belong to
the *operation* rather than to the wrapper.

**The loop**, because an object can bind state to the first loop it runs on.
Three wrappers in this workspace had written some version of "open a bridge for
this call, or use the caller's" by hand: ``BatchOperations``
(``dataknobs-data``) and the FSM's ``BatchExecutor`` and ``StreamExecutor``
(``dataknobs-fsm``).

**The budget**, because a timeout applied per reach is not a bound on the call
the caller made. Spent afresh on each round trip, a 30-second bound on a
twenty-chunk write permits ten minutes --- which is what ``BatchOperations``
shipped before this, and what the executors had no bound for at all.

:class:`~dataknobs_common.BridgedOperation` is that pair, and
:func:`~dataknobs_common.bridged_operation` is how a call opens one.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dataknobs_common import (
    BridgedOperation,
    OperationTimeoutError,
    SyncLoopBridge,
    bridged_operation,
)
from dataknobs_common.sync_bridge import bridge_thread_names
from dataknobs_common.testing import assert_no_leaked_bridge_threads


async def echo(value: int) -> int:
    """A coroutine that yields to the loop once and returns its argument."""
    await asyncio.sleep(0)
    return value


async def naps(seconds: float) -> str:
    """A coroutine that takes a known amount of time."""
    await asyncio.sleep(seconds)
    return "finished"


# --------------------------------------------------------------------------- #
# 1. The loop: owned for the call, or borrowed from the caller
# --------------------------------------------------------------------------- #


def test_an_owned_bridge_serves_the_call_and_ends_with_it() -> None:
    """With no ``bridge``, the operation owns one and the block ends it."""
    with assert_no_leaked_bridge_threads():
        with bridged_operation(thread_name="dk-test-op") as op:
            assert op.run(echo(7)) == 7
            assert op.bridge is not None
            owned = op.bridge

    assert owned.is_closed


def test_an_owned_bridge_ends_even_when_the_call_raises() -> None:
    """Teardown is on the error path too, which is why this is a ``with``."""
    captured: list[SyncLoopBridge] = []

    with assert_no_leaked_bridge_threads(), pytest.raises(ValueError, match="from the body"):
        with bridged_operation(thread_name="dk-test-op") as op:
            assert op.bridge is not None
            captured.append(op.bridge)
            raise ValueError("from the body")

    assert captured[0].is_closed


def test_every_reach_of_one_operation_runs_on_one_loop() -> None:
    """The point of scoping to the call rather than to the reach."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def witness() -> None:
        loops.append(asyncio.get_running_loop())

    with bridged_operation(thread_name="dk-test-op") as op:
        for _ in range(4):
            op.run(witness())

    assert len({id(loop) for loop in loops}) == 1


def test_a_supplied_bridge_is_used_and_left_running() -> None:
    """A bridge belongs to whoever passed it, who may be using it for more."""
    bridge = SyncLoopBridge(thread_name="dk-test-owner")
    try:
        with bridged_operation(bridge=bridge) as op:
            assert op.bridge is bridge
            assert op.run(echo(3)) == 3
        assert not bridge.is_closed, "a borrowed bridge was closed by the operation"

        with bridged_operation(bridge=bridge) as op:
            assert op.run(echo(4)) == 4
    finally:
        bridge.close()


def test_two_operations_can_share_one_supplied_bridge() -> None:
    """One thread serves several operations when the caller says so."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def witness() -> None:
        loops.append(asyncio.get_running_loop())

    with SyncLoopBridge(thread_name="dk-test-owner") as bridge:
        with bridged_operation(bridge=bridge) as first:
            first.run(witness())
        with bridged_operation(bridge=bridge) as second:
            second.run(witness())

    assert len({id(loop) for loop in loops}) == 1


def test_an_owned_bridges_thread_name_reaches_the_leak_guard() -> None:
    """``thread_name`` is a diagnostic label, not a way out of the guard.

    The bridge registers every name it runs under, so a wrapper that names its
    thread is still watched --- which is what keeps the guard from silently
    losing sight of a leak.
    """
    with bridged_operation(thread_name="dk-test-registered") as op:
        op.run(echo(1))

    assert "dk-test-registered" in bridge_thread_names()


# --------------------------------------------------------------------------- #
# 2. The budget: one deadline across the call, not one per reach
# --------------------------------------------------------------------------- #


def test_the_budget_is_spent_across_the_calls_reaches() -> None:
    """Four reaches under one bound do not each get the whole bound."""
    with pytest.raises(OperationTimeoutError):
        with bridged_operation(thread_name="dk-test-op", timeout=0.3) as op:
            started = time.monotonic()
            for _ in range(4):
                op.run(naps(0.15))

    elapsed = time.monotonic() - started
    assert elapsed < 0.8, f"the bound restarted per reach: {elapsed:.2f}s for a 0.3s budget"


def test_a_spent_budget_refuses_without_reaching_the_loop() -> None:
    """Past the deadline, the coroutine is closed rather than started.

    A caller who has stopped waiting should stop paying for round trips, and a
    coroutine that is closed rather than abandoned does not warn about never
    being awaited.
    """
    started = False

    async def should_not_run() -> None:
        nonlocal started
        started = True

    with bridged_operation(thread_name="dk-test-op", timeout=-1.0, label="Widget") as op:
        with pytest.raises(OperationTimeoutError, match="Widget exceeded its timeout"):
            op.run(should_not_run())

    assert not started, "the coroutine reached the loop after the deadline had passed"


def test_a_wait_that_expires_raises_the_operations_own_error() -> None:
    """The bridge raises the builtin; the operation names itself."""
    with bridged_operation(thread_name="dk-test-op", timeout=0.05, label="Widget") as op:
        with pytest.raises(OperationTimeoutError, match="Widget exceeded its timeout"):
            op.run(naps(5.0))


def test_a_timeout_the_work_itself_raises_is_not_relabelled() -> None:
    """Only the operation's own deadline becomes an ``OperationTimeoutError``.

    The bridge re-raises a ``TimeoutError`` from the coroutine through the same
    channel as an expired wait, so the deadline is what tells them apart. A
    caller distinguishing "my operation ran out of time" from "the work raised
    a timeout" gets the answer it asked for.
    """

    async def raises_a_timeout() -> None:
        raise TimeoutError("the remote call timed out")

    with bridged_operation(thread_name="dk-test-op", timeout=30.0) as op:
        with pytest.raises(TimeoutError, match="the remote call timed out") as caught:
            op.run(raises_a_timeout())

    assert not isinstance(caught.value, OperationTimeoutError)


def test_an_operation_timeout_is_catchable_as_a_plain_timeout() -> None:
    """Subclassing the builtin keeps a published ``Raises: TimeoutError`` true."""
    assert issubclass(OperationTimeoutError, TimeoutError)

    with bridged_operation(thread_name="dk-test-op", timeout=-1.0) as op:
        with pytest.raises(TimeoutError):
            op.run(echo(1))


def test_no_timeout_leaves_the_budget_unbounded() -> None:
    """The default waits for as long as the work takes."""
    with bridged_operation(thread_name="dk-test-op") as op:
        assert op.deadline is None
        assert op.remaining is None
        assert op.run(naps(0.05)) == "finished"


# --------------------------------------------------------------------------- #
# 3. An operation whose work turns out to be synchronous
# --------------------------------------------------------------------------- #


def test_a_synchronous_operation_is_charged_no_thread() -> None:
    """``needs_loop=False`` allocates no loop for work that will not reach one.

    A wrapper fronting either flavour still has a deadline to carry, so it
    still gets an operation --- just not a thread.
    """
    before = threading.active_count()

    with bridged_operation(thread_name="dk-test-op", timeout=5.0, needs_loop=False) as op:
        assert op.bridge is None
        assert op.deadline is not None
        assert threading.active_count() == before


def test_a_synchronous_operation_says_so_if_an_async_path_reaches_it() -> None:
    """A broken invariant becomes a sentence rather than a stray coroutine.

    Reaching :meth:`BridgedOperation.run` with no bridge means a wrapper took
    its asynchronous branch after declaring its work synchronous. The
    alternative symptom is a ``coroutine was never awaited`` warning several
    frames from the cause.
    """
    op = BridgedOperation(bridge=None, deadline=None, label="Widget")

    with pytest.raises(RuntimeError, match="Widget has no operation loop"):
        op.run(echo(1))


def test_a_supplied_bridge_is_ignored_when_the_work_is_synchronous() -> None:
    """Nothing reaches a loop, so nothing should be pointed at one."""
    with SyncLoopBridge(thread_name="dk-test-owner") as bridge:
        with bridged_operation(bridge=bridge, needs_loop=False) as op:
            assert op.bridge is None


def test_a_nested_operations_timeout_keeps_its_own_name() -> None:
    """An inner operation's deadline is not relabelled as the outer one's.

    ``OperationTimeoutError`` is only ever constructed by
    :meth:`BridgedOperation.run`; a bridge reports an expired wait as the
    builtin. So one arriving *from the coroutine* belongs to an operation
    nested inside this one --- a wrapper with its own budget, called from
    within the outer call's work --- and the outer operation must pass it
    through rather than name itself in it.

    **What this cannot cover.** The outer deadline governs the wait, so a
    coroutine that raises before that deadline is normally read as having
    raised before it, and the misattribution needed the two to disagree: the
    exception raised at ``deadline - epsilon`` while ``time.monotonic()``, read
    afterwards in the handler, had passed it. That window is scheduling
    latency, and it is not reproducible on demand --- which is the argument for
    deciding by *type* rather than by clock, since the type is exact where the
    clock is a race. This test pins the decision; it cannot pin the window.
    """
    inner = OperationTimeoutError("Inner exceeded its timeout while waiting for this call")

    async def raises_an_inner_operations_timeout() -> None:
        raise inner

    with bridged_operation(thread_name="dk-test-op", timeout=30.0, label="Outer") as op:
        with pytest.raises(OperationTimeoutError) as caught:
            op.run(raises_an_inner_operations_timeout())

    assert caught.value is inner, "the inner operation's error was replaced, not propagated"
    assert "Inner" in str(caught.value)
    assert "Outer" not in str(caught.value)
