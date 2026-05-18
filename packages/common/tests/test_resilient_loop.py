"""Unit tests for the shared event-bus supervised-loop helper.

Deterministic: a recording fake ``sleep`` is injected (no real waiting,
back-off delays captured) and ``random`` is seeded for the jitter
assertions. Every invariant in the helper's contract is pinned here.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from dataknobs_common.events._resilient_loop import run_supervised_loop


class _RecordingSleep:
    """Awaitable that records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _counting_should_run(max_true: int):
    """should_run() that returns True max_true times, then False."""
    state = {"n": 0}

    def should_run() -> bool:
        if state["n"] >= max_true:
            return False
        state["n"] += 1
        return True

    return should_run


@pytest.mark.asyncio
async def test_backoff_is_exponential_with_jitter_and_resets_after_clean():
    """Invariant 1: exponential+jitter back-off; resets after a clean run."""
    random.seed(2026)
    sleep = _RecordingSleep()
    script = iter(
        [
            "raise",  # failure 1 -> attempt 1 -> ~[0.9, 1.1]
            "raise",  # failure 2 -> attempt 2 -> ~[1.8, 2.2]
            "ok",     # clean -> resets failure counter
            "raise",  # failure 1 again -> attempt 1 -> ~[0.9, 1.1]
            "stop",
        ]
    )

    async def one_iteration() -> None:
        action = next(script)
        if action == "raise":
            raise RuntimeError("transient")
        if action == "stop":
            raise asyncio.CancelledError

    await run_supervised_loop(
        one_iteration,
        should_run=lambda: True,
        name="test",
        base_delay=1.0,
        max_delay=30.0,
        sleep=sleep,
    )

    assert len(sleep.delays) == 3
    assert 0.9 <= sleep.delays[0] <= 1.1   # attempt 1
    assert 1.8 <= sleep.delays[1] <= 2.2   # attempt 2 (escalated)
    # After the clean iteration the counter reset, so this is attempt 1:
    assert 0.9 <= sleep.delays[2] <= 1.1


@pytest.mark.asyncio
async def test_backoff_honors_max_delay():
    """Invariant 2: the escalating delay is capped at max_delay."""
    random.seed(7)
    sleep = _RecordingSleep()
    calls = {"n": 0}

    async def one_iteration() -> None:
        calls["n"] += 1
        if calls["n"] > 8:
            raise asyncio.CancelledError
        raise RuntimeError("always fails")

    await run_supervised_loop(
        one_iteration,
        should_run=lambda: True,
        name="test",
        base_delay=1.0,
        max_delay=1.5,
        sleep=sleep,
    )

    assert sleep.delays  # backed off at least once
    assert all(d <= 1.5 for d in sleep.delays)
    # Later attempts would be huge un-capped (1*2^k); cap proves it bit.
    assert sleep.delays[-1] <= 1.5


@pytest.mark.asyncio
async def test_breaks_promptly_on_cancelled_no_reraise_no_sleep():
    """Invariant 3: CancelledError -> return (no re-raise), no back-off."""
    sleep = _RecordingSleep()
    calls = {"n": 0}

    async def one_iteration() -> None:
        calls["n"] += 1
        raise asyncio.CancelledError

    # Must NOT propagate CancelledError to the caller.
    await run_supervised_loop(
        one_iteration,
        should_run=lambda: True,
        name="test",
        sleep=sleep,
    )

    assert calls["n"] == 1
    assert sleep.delays == []  # no back-off on cancellation


@pytest.mark.asyncio
async def test_cancelled_during_backoff_sleep_breaks():
    """Cancellation while awaiting the back-off sleep also breaks cleanly."""

    async def cancelling_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    calls = {"n": 0}

    async def one_iteration() -> None:
        calls["n"] += 1
        raise RuntimeError("transient")

    await run_supervised_loop(
        one_iteration,
        should_run=lambda: True,
        name="test",
        sleep=cancelling_sleep,
    )

    # Failed once, entered back-off, got cancelled mid-sleep, returned.
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_resumes_after_transient_raise():
    """Invariant 4: a transient failure does not end the loop."""
    sleep = _RecordingSleep()
    seen: list[str] = []
    script = iter(["raise", "ok", "stop"])

    async def one_iteration() -> None:
        action = next(script)
        seen.append(action)
        if action == "raise":
            raise RuntimeError("transient")
        if action == "stop":
            raise asyncio.CancelledError

    await run_supervised_loop(
        one_iteration,
        should_run=lambda: True,
        name="test",
        sleep=sleep,
    )

    assert seen == ["raise", "ok", "stop"]
    assert len(sleep.delays) == 1  # one back-off, for the single failure


@pytest.mark.asyncio
async def test_returns_cleanly_when_should_run_flips_false():
    """Invariant 5: returns (no exception) once should_run() is false."""
    sleep = _RecordingSleep()
    calls = {"n": 0}

    async def one_iteration() -> None:
        calls["n"] += 1

    await run_supervised_loop(
        one_iteration,
        should_run=_counting_should_run(3),
        name="test",
        sleep=sleep,
    )

    assert calls["n"] == 3
    assert sleep.delays == []


@pytest.mark.asyncio
async def test_does_not_self_terminate_while_running_and_failing():
    """Invariant 6: keeps going through repeated failures (never gives up).

    Bounded by should_run() so the test terminates; the assertion is
    that the loop ran the full bound rather than bailing out early on
    the persistent failures.
    """
    sleep = _RecordingSleep()
    calls = {"n": 0}

    async def one_iteration() -> None:
        calls["n"] += 1
        raise RuntimeError("persistent outage")

    await run_supervised_loop(
        one_iteration,
        should_run=_counting_should_run(10),
        name="test",
        sleep=sleep,
    )

    assert calls["n"] == 10           # did not give up despite 10 failures
    assert len(sleep.delays) == 10    # backed off after each
