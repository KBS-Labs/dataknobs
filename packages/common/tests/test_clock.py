"""The one read of the current time, and the seam a test fixes it with.

:mod:`dataknobs_common.clock` is where the library reads the current UTC time
for a timestamp. :func:`~dataknobs_common.clock.use_clock` sets the clock for
the current context, so a test can fix every stamp the code under it makes,
the dataclass defaults included, without patching :mod:`datetime`.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

import dataknobs_common
from dataknobs_common import clock
from dataknobs_common.async_iter import aiter_sync_in_thread
from dataknobs_common.clock import now_iso, system_clock, use_clock, utc_now

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)


def test_the_default_clock_reads_aware_utc_wall_time() -> None:
    before = datetime.now(UTC)
    read = utc_now()
    after = datetime.now(UTC)
    assert read.tzinfo is UTC
    assert before <= read <= after


def test_now_iso_is_the_isoformat_of_the_read() -> None:
    with use_clock(lambda: AT):
        assert now_iso() == AT.isoformat() == "2026-01-01T12:00:00+00:00"
    assert now_iso().endswith("+00:00")


def test_system_clock_is_the_default() -> None:
    assert system_clock().tzinfo is UTC
    with use_clock(lambda: AT):
        assert system_clock() != AT  # the default is not what use_clock sets


def test_use_clock_sets_the_read_and_restores_it() -> None:
    with use_clock(lambda: AT) as given:
        assert utc_now() == AT
        assert given() == AT
    assert utc_now() != AT


def test_use_clock_nests_and_restores_in_order() -> None:
    with use_clock(lambda: AT):
        with use_clock(lambda: LATER):
            assert utc_now() == LATER
        assert utc_now() == AT
    assert utc_now() not in (AT, LATER)


def test_use_clock_restores_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), use_clock(lambda: AT):
        raise RuntimeError("inside")
    assert utc_now() != AT


def test_an_aware_clock_in_another_zone_reads_back_as_utc() -> None:
    plus_five = timezone(timedelta(hours=5))
    with use_clock(lambda: datetime(2026, 1, 1, 17, 0, tzinfo=plus_five)):
        read = utc_now()
        assert read == AT
        assert read.tzinfo is UTC
        assert now_iso() == "2026-01-01T12:00:00+00:00"


def test_a_naive_clock_is_refused() -> None:
    with use_clock(lambda: datetime(2026, 1, 1, 12, 0)), pytest.raises(ValueError, match="naive"):
        utc_now()


def test_a_clock_returning_something_else_is_refused() -> None:
    with use_clock(lambda: "2026-01-01"), pytest.raises(TypeError, match="datetime"):
        utc_now()


async def test_a_task_started_inside_use_clock_reads_it() -> None:
    with use_clock(lambda: AT):
        task = asyncio.create_task(_read())
    assert await task == AT


async def test_a_concurrent_task_outside_use_clock_does_not() -> None:
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def fixed() -> datetime:
        with use_clock(lambda: AT):
            entered.set()
            await leave.wait()
            return utc_now()

    async def outside() -> datetime:
        await entered.wait()
        try:
            return utc_now()
        finally:
            leave.set()

    inside_read, outside_read = await asyncio.gather(fixed(), outside())
    assert inside_read == AT
    assert outside_read != AT


async def test_asyncio_to_thread_carries_the_clock() -> None:
    with use_clock(lambda: AT):
        assert await asyncio.to_thread(utc_now) == AT


#: Python 3.14 can start a thread in a copy of its parent's context, and does
#: by default on a free-threaded build. Where it does, a bare thread carries
#: the clock like any task.
_THREADS_INHERIT_CONTEXT = bool(getattr(sys.flags, "thread_inherit_context", 0))


def test_a_bare_thread_reads_the_default_clock() -> None:
    """The documented limit: a thread started without the context reads the default."""
    seen: list[datetime] = []
    with use_clock(lambda: AT):
        thread = threading.Thread(target=lambda: seen.append(utc_now()))
        thread.start()
        thread.join()
    assert seen
    assert (seen[0] == AT) is _THREADS_INHERIT_CONTEXT


def test_a_thread_pool_worker_reads_the_default_clock() -> None:
    with use_clock(lambda: AT), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(utc_now).result() != AT


async def test_run_in_executor_does_not_carry_the_clock() -> None:
    """The usual asyncio hand-off to a thread is not ``to_thread``'s."""
    loop = asyncio.get_running_loop()
    with use_clock(lambda: AT):
        assert await loop.run_in_executor(None, utc_now) != AT


async def test_aiter_sync_in_thread_does_not_carry_the_clock() -> None:
    """Its producer runs on a thread of its own, started without the context."""
    with use_clock(lambda: AT):
        seen = [stamp async for stamp in aiter_sync_in_thread(lambda: iter([utc_now()]))]
    assert seen and (seen[0] == AT) is _THREADS_INHERIT_CONTEXT


def test_copy_context_run_carries_the_clock_onto_a_thread() -> None:
    """The documented way across: run the work in a copy of the context."""
    with use_clock(lambda: AT), ThreadPoolExecutor(max_workers=1) as pool:
        context = contextvars.copy_context()
        assert pool.submit(context.run, utc_now).result() == AT


def test_run_coroutine_threadsafe_carries_the_clock_to_the_loop() -> None:
    """A sync caller handing a coroutine to a loop thread: the loop sees its clock."""
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        with use_clock(lambda: AT):
            assert asyncio.run_coroutine_threadsafe(_read(), loop).result(timeout=5) == AT
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=5)
        loop.close()


def test_the_names_are_reached_from_the_module_not_the_package_door() -> None:
    assert set(clock.__all__) == {"Clock", "now_iso", "system_clock", "use_clock", "utc_now"}
    assert not set(clock.__all__) & set(dataknobs_common.__all__)


async def _read() -> datetime:
    return utc_now()
