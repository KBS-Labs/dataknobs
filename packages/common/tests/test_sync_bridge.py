"""Tests for the ``SyncLoopBridge`` / ``run_coro_sync`` async->sync bridge.

The bridge runs a coroutine to completion from synchronous code on a
private background event-loop thread. These tests pin the behaviors that
make it correct and safe: it returns the coroutine's value, re-raises its
exception with the traceback preserved, **works when called from inside an
already-running event loop** (the footgun ``asyncio.run`` fails on), and
tears down cleanly leaving no live thread. No mocks — real coroutines on a
real loop thread.
"""

from __future__ import annotations

import asyncio
import threading
import traceback

import pytest

from dataknobs_common import SyncLoopBridge, run_coro_sync
from dataknobs_common.sync_bridge import _THREAD_NAME


def _bridge_threads_alive() -> list[str]:
    """Names of any live bridge loop threads (empty after teardown)."""
    return [t.name for t in threading.enumerate() if t.name == _THREAD_NAME]


def test_run_returns_coroutine_value() -> None:
    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    bridge = SyncLoopBridge()
    try:
        assert bridge.run(add(2, 3)) == 5
        # Reusable across calls.
        assert bridge.run(add(10, 1)) == 11
    finally:
        bridge.close()


def test_run_reraises_exception_with_traceback() -> None:
    async def boom() -> None:
        await asyncio.sleep(0)
        raise ValueError("kaboom")

    bridge = SyncLoopBridge()
    try:
        with pytest.raises(ValueError, match="kaboom") as exc_info:
            bridge.run(boom())
    finally:
        bridge.close()

    # The coroutine's own frame is preserved in the re-raised traceback.
    rendered = "".join(traceback.format_tb(exc_info.value.__traceback__))
    assert "boom" in rendered


async def test_run_works_from_inside_a_running_loop() -> None:
    """The central correctness requirement: callable from async code.

    This test body runs inside pytest-asyncio's event loop. ``asyncio.run``
    / ``loop.run_until_complete`` would raise "cannot be called from a
    running event loop" here; the bridge runs the coroutine on its own loop
    thread, so the synchronous ``run`` completes and returns.
    """

    async def work() -> str:
        await asyncio.sleep(0)
        return "done"

    # Confirm the footgun the bridge exists to avoid is real in this context.
    footgun = work()
    with pytest.raises(RuntimeError):
        asyncio.run(footgun)
    footgun.close()  # asyncio.run raised before awaiting it; close to avoid a warning

    bridge = SyncLoopBridge()
    try:
        # Synchronous call from inside the running loop — no deadlock, no raise.
        assert bridge.run(work()) == "done"
    finally:
        bridge.close()


def test_close_joins_thread() -> None:
    bridge = SyncLoopBridge()
    assert _bridge_threads_alive() != []
    bridge.close()
    assert _bridge_threads_alive() == []


def test_close_is_idempotent() -> None:
    bridge = SyncLoopBridge()
    bridge.close()
    bridge.close()  # must not raise or hang
    assert _bridge_threads_alive() == []


def test_run_after_close_raises() -> None:
    bridge = SyncLoopBridge()
    bridge.close()

    async def work() -> int:
        return 1

    coro = work()
    with pytest.raises(RuntimeError, match="closed"):
        bridge.run(coro)


def test_context_manager_closes_on_exit() -> None:
    async def work() -> int:
        await asyncio.sleep(0)
        return 42

    with SyncLoopBridge() as bridge:
        assert bridge.run(work()) == 42
    assert _bridge_threads_alive() == []


def test_run_coro_sync_oneshot_runs_and_cleans_up() -> None:
    async def work() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_coro_sync(work()) == 7
    # The throwaway bridge's thread is torn down before returning.
    assert _bridge_threads_alive() == []


async def test_run_coro_sync_from_inside_a_running_loop() -> None:
    async def work() -> str:
        await asyncio.sleep(0)
        return "ok"

    assert run_coro_sync(work()) == "ok"
    assert _bridge_threads_alive() == []


def test_run_times_out_and_bridge_stays_usable() -> None:
    """A bounded ``timeout`` escapes a slow coroutine; the bridge survives."""

    async def slow() -> str:
        await asyncio.sleep(30)
        return "too late"

    async def quick() -> int:
        return 1

    bridge = SyncLoopBridge()
    try:
        with pytest.raises(TimeoutError):
            bridge.run(slow(), timeout=0.05)
        # The bridge is reusable after a timeout — the timed-out coroutine was
        # cancelled on the loop, not left blocking it.
        assert bridge.run(quick()) == 1
    finally:
        bridge.close()


def test_run_coro_sync_times_out_and_cleans_up() -> None:
    async def slow() -> None:
        await asyncio.sleep(30)

    with pytest.raises(TimeoutError):
        run_coro_sync(slow(), timeout=0.05)
    # The throwaway bridge is still torn down on the timeout path.
    assert _bridge_threads_alive() == []


def test_concurrent_run_from_multiple_threads() -> None:
    """Many threads submitting to one shared bridge all get correct results."""

    async def work(n: int) -> int:
        await asyncio.sleep(0.01)
        return n * n

    results: dict[int, int] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def submit(n: int) -> None:
        start.wait()  # maximize overlap of concurrent submissions
        try:
            value = bridge.run(work(n))
        except BaseException as exc:  # recorded across threads for assertion
            with lock:
                errors.append(exc)
        else:
            with lock:
                results[n] = value

    bridge = SyncLoopBridge()
    try:
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        bridge.close()

    assert errors == []
    assert results == {i: i * i for i in range(8)}


def test_concurrent_close_all_callers_wait_for_teardown() -> None:
    """Every concurrent ``close`` returns only after the thread is joined."""
    bridge = SyncLoopBridge()
    start = threading.Barrier(4)

    def closer() -> None:
        start.wait()
        bridge.close()

    threads = [threading.Thread(target=closer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All four returned; the loop thread is gone for every one of them.
    assert _bridge_threads_alive() == []


def test_close_from_within_coroutine_raises_clear_error() -> None:
    """Self-close from the loop thread fails loudly instead of deadlocking."""
    bridge = SyncLoopBridge()
    try:

        async def closes_self() -> None:
            bridge.close()  # runs on the bridge's own loop thread

        with pytest.raises(RuntimeError, match="must not be called from within"):
            bridge.run(closes_self())
    finally:
        bridge.close()
    assert _bridge_threads_alive() == []


def test_construction_failure_closes_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the loop thread fails to start, the created loop is closed (no fd leak)."""
    created: list[asyncio.AbstractEventLoop] = []
    real_new_event_loop = asyncio.new_event_loop

    def tracking_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = real_new_event_loop()
        created.append(loop)
        return loop

    def failing_start(self: threading.Thread) -> None:
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(asyncio, "new_event_loop", tracking_new_event_loop)
    monkeypatch.setattr(threading.Thread, "start", failing_start)

    with pytest.raises(RuntimeError, match="cannot start thread"):
        SyncLoopBridge()

    assert created, "expected a loop to have been created"
    assert created[0].is_closed(), "the loop must be closed on the failure path"
