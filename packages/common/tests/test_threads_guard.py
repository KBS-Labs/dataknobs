"""Self-tests for the leaked-daemon-thread guard.

A guard that never fails is worse than no guard, because it also reports
green. These pin both directions: it fires on a real leak, and it stays
quiet on correctly-closed code and on someone else's pre-existing leak.
"""

from __future__ import annotations

import gc
import threading
import warnings
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
    assert {_BRIDGE_THREAD, _PUMP_THREAD} == DK_DAEMON_THREAD_NAMES


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
        with pytest.raises(AssertionError) as excinfo, assert_no_leaked_bridge_threads():
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
    with pytest.raises(RuntimeError, match="boom"), assert_no_leaked_bridge_threads():
        raise RuntimeError("boom")


async def _answer() -> int:
    return 42


def _counter() -> Iterator[int]:
    yield from range(3)


def test_names_may_be_a_one_shot_iterable() -> None:
    """Reproduce-first: a generator ``names`` silently disabled the guard.

    ``names`` is declared ``Iterable[str]`` and was consumed **twice** —
    once to sample on entry, once to check on exit. A ``list`` or
    ``frozenset`` survives that; a generator does not. The second pass saw
    an exhausted iterator, watched nothing, found nothing, and passed.

    For a construct whose own docstring says a guard that never fails is
    worse than no guard, silently becoming that guard is the failure mode
    most worth pinning.
    """
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError, match="daemon thread"):
            with assert_no_leaked_bridge_threads(names=(name for name in [_BRIDGE_THREAD])):
                leaked = SyncLoopBridge()
                leaked.run(_answer())
                # deliberately not closed
    finally:
        if leaked is not None:
            leaked.close()


def test_live_threads_accepts_a_one_shot_iterable() -> None:
    """The same contract on the query the guard is built from.

    Measured as a delta. Asserting the absolute result is the mistake this
    module exists to warn about: another test holding a live bridge — or
    another package's, in a combined run — would make the list longer and
    fail this for someone else's thread.
    """
    before = set(live_dk_daemon_threads({_BRIDGE_THREAD}))
    bridge = SyncLoopBridge()
    try:
        found = live_dk_daemon_threads(name for name in [_BRIDGE_THREAD])
        created = [t.name for t in found if t not in before]
        assert created == [_BRIDGE_THREAD]
    finally:
        bridge.close()


def test_a_custom_thread_name_is_watched_too() -> None:
    """A ``thread_name=`` must not be a way out of the guard.

    ``SyncTextEmbedder`` names its bridge ``dk-sync-embedder`` for
    diagnostics, and exact-name matching against a fixed pair made every
    thread that class creates invisible here — silently, at every call
    site. A name is a label, not an opt-out.
    """
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError, match="dk-custom-named-bridge"):
            with assert_no_leaked_bridge_threads():
                leaked = SyncLoopBridge(thread_name="dk-custom-named-bridge")
                leaked.run(_answer())
    finally:
        if leaked is not None:
            leaked.close()


def test_a_custom_name_registered_inside_the_block_is_still_watched() -> None:
    """The default watch set is resolved at exit as well as at entry.

    A bridge whose name this process has never seen before is created
    *inside* the block. Freezing the watch set on entry would leave that
    name unwatched for the one block that most needs it — the block that
    introduced it.
    """
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError, match="dk-first-seen-inside"):
            with assert_no_leaked_bridge_threads():
                leaked = SyncLoopBridge(thread_name="dk-first-seen-inside")
                leaked.run(_answer())
    finally:
        if leaked is not None:
            leaked.close()


def test_a_caller_supplied_watch_set_is_still_normalized_once() -> None:
    """A one-shot iterable must survive being needed at entry and exit."""
    leaked: SyncLoopBridge | None = None
    try:
        with pytest.raises(AssertionError, match=_BRIDGE_THREAD):
            with assert_no_leaked_bridge_threads(names=(n for n in [_BRIDGE_THREAD])):
                leaked = SyncLoopBridge()
                leaked.run(_answer())
    finally:
        if leaked is not None:
            leaked.close()


def test_dropping_an_unclosed_bridge_warns() -> None:
    """The leak is silent by construction; this is what makes it audible.

    A daemon thread cannot block process exit and the owning object goes
    on working, so nothing about a dropped bridge is observable from the
    outside. A long-lived server building one per tenant leaks a thread
    and an event loop's self-pipe descriptors per tenant, and finds out
    from neither an exception nor a log line.
    """
    with pytest.warns(ResourceWarning, match="unclosed SyncLoopBridge"):
        bridge = SyncLoopBridge(thread_name="dk-warns-on-drop")
        bridge.run(_answer())
        thread = bridge._thread
        del bridge
        gc.collect()

    thread.join(timeout=5.0)
    assert not thread.is_alive(), "the finalizer warned but left the thread running"


def test_dropping_a_closed_bridge_is_silent() -> None:
    """Only an *unclosed* bridge is a leak.

    A warning on correct teardown is a warning nobody keeps enabled, and
    a `ResourceWarning` nobody keeps enabled catches no leaks.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        bridge = SyncLoopBridge(thread_name="dk-silent-on-close")
        bridge.run(_answer())
        bridge.close()
        del bridge
        gc.collect()
