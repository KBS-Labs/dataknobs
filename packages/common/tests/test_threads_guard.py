"""Self-tests for the leaked-daemon-thread guard.

A guard that never fails is worse than no guard, because it also reports
green. These pin both directions: it fires on a real leak, and it stays
quiet on correctly-closed code and on someone else's pre-existing leak.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from dataknobs_common import SyncLoopBridge, aiter_sync_in_thread
from dataknobs_common.async_iter import _THREAD_NAME as _PUMP_THREAD
from dataknobs_common.sync_bridge import _THREAD_NAME as _BRIDGE_THREAD
from dataknobs_common.testing import (
    DK_DAEMON_THREAD_NAMES,
    assert_no_leaked_bridge_threads,
    live_dk_daemon_threads,
)


def test_covers_both_dataknobs_daemon_thread_names() -> None:
    """Both names, sourced from the modules that create them.

    Hardcoding the strings here would let a rename in ``sync_bridge`` or
    ``async_iter`` leave the guard watching for a thread that no longer
    exists — passing forever.
    """
    assert DK_DAEMON_THREAD_NAMES == {_BRIDGE_THREAD, _PUMP_THREAD}


def test_passes_when_the_bridge_is_closed() -> None:
    with assert_no_leaked_bridge_threads():
        bridge = SyncLoopBridge()
        bridge.run(_answer())
        bridge.close()


def test_fails_when_a_bridge_is_leaked() -> None:
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError, match="daemon thread"):
            with assert_no_leaked_bridge_threads():
                leaked = SyncLoopBridge()
                leaked.run(_answer())
                # deliberately not closed
    finally:
        if leaked is not None:
            leaked.close()


def test_failure_message_names_the_leaked_thread() -> None:
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError) as excinfo:
            with assert_no_leaked_bridge_threads():
                leaked = SyncLoopBridge()
                leaked.run(_answer())
        assert _BRIDGE_THREAD in str(excinfo.value)
    finally:
        if leaked is not None:
            leaked.close()


async def test_passes_when_the_aiter_pump_drains() -> None:
    """The second name is watched too, not just the bridge."""
    with assert_no_leaked_bridge_threads():
        items = [x async for x in aiter_sync_in_thread(_counter)]
    assert items == [0, 1, 2]


def test_ignores_a_leak_that_predates_the_block() -> None:
    """Delta, not absolute count.

    A thread leaked by an unrelated earlier test in the same process must
    not fail a later, well-behaved one — otherwise the guard reports the
    wrong culprit and the real one hides behind it.
    """
    pre_existing = SyncLoopBridge()
    try:
        pre_existing.run(_answer())
        assert live_dk_daemon_threads()

        with assert_no_leaked_bridge_threads():
            clean = SyncLoopBridge()
            clean.run(_answer())
            clean.close()
    finally:
        pre_existing.close()


def test_narrowing_to_one_name_ignores_the_other() -> None:
    """The ``names`` argument scopes what is watched."""
    leaked: SyncLoopBridge | None = None
    try:
        # Watching only the pump: a leaked *bridge* is not this block's
        # concern and must not fail it.
        with assert_no_leaked_bridge_threads(names={_PUMP_THREAD}):
            leaked = SyncLoopBridge()
            leaked.run(_answer())
    finally:
        if leaked is not None:
            leaked.close()


def test_live_threads_returns_thread_objects() -> None:
    bridge = SyncLoopBridge()
    try:
        bridge.run(_answer())
        live = live_dk_daemon_threads({_BRIDGE_THREAD})
        assert live
        assert all(isinstance(t, threading.Thread) for t in live)
        assert all(t.name == _BRIDGE_THREAD for t in live)
    finally:
        bridge.close()


def test_exception_inside_the_block_propagates_unmasked() -> None:
    """The original error is the more informative failure.

    Asserting on thread state while unwinding would replace a real
    exception with an AssertionError about its side effects.
    """
    with pytest.raises(RuntimeError, match="boom"):
        with assert_no_leaked_bridge_threads():
            raise RuntimeError("boom")


async def _answer() -> int:
    return 42


def _counter() -> Iterator[int]:
    yield from range(3)
