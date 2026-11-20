"""The one read of the current time, and the seam a test fixes it with.

:mod:`dataknobs_common.clock` is where the library reads the current UTC time
for a timestamp. :func:`~dataknobs_common.clock.use_clock` sets the clock for
the current context, so a test can fix every stamp the code under it makes,
the dataclass defaults included, without patching :mod:`datetime`.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

import dataknobs_common
from dataknobs_common import clock
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


def test_a_bare_thread_reads_the_default_clock() -> None:
    """The documented limit: a thread started without the context reads the default."""
    seen: list[datetime] = []
    with use_clock(lambda: AT):
        thread = threading.Thread(target=lambda: seen.append(utc_now()))
        thread.start()
        thread.join()
    assert seen and seen[0] != AT


def test_the_names_are_reached_from_the_module_not_the_package_door() -> None:
    assert set(clock.__all__) == {"Clock", "now_iso", "system_clock", "use_clock", "utc_now"}
    assert not set(clock.__all__) & set(dataknobs_common.__all__)


async def _read() -> datetime:
    return utc_now()
