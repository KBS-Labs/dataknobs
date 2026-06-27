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
