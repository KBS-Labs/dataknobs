"""``FixedClock``: the reference clock a test sets, advances and steps back.

It is a :data:`~dataknobs_common.clock.Clock`, so one object serves both ways
in: ambient through :func:`~dataknobs_common.clock.use_clock`, and injected
where a constructor takes a clock (the per-user state store's
``components["now"]``, tested in that package).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from dataknobs_common.clock import now_iso, use_clock, utc_now
from dataknobs_common.provenance import create_provenance
from dataknobs_common.testing import FixedClock

AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_it_returns_the_instant_it_was_given() -> None:
    clock = FixedClock(AT)
    assert clock() == AT
    assert clock() == AT  # reading does not move it
    assert clock.now == AT


def test_a_naive_start_is_refused() -> None:
    with pytest.raises(ValueError, match="naive"):
        FixedClock(datetime(2026, 1, 1, 12, 0))


def test_something_other_than_a_datetime_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="str"):
        FixedClock("2026-01-01T00:00:00+00:00")  # type: ignore[arg-type]
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(TypeError, match="int"):
        clock.set(1767225600)  # type: ignore[arg-type]


def test_advance_moves_it_forward() -> None:
    clock = FixedClock(AT)
    assert clock.advance(timedelta(seconds=5)) == AT + timedelta(seconds=5)
    assert clock() == AT + timedelta(seconds=5)


def test_set_can_step_it_backwards() -> None:
    clock = FixedClock(AT)
    earlier = AT - timedelta(minutes=1)
    clock.set(earlier)
    assert clock() == earlier


def test_set_refuses_a_naive_instant() -> None:
    clock = FixedClock(AT)
    with pytest.raises(ValueError, match="naive"):
        clock.set(datetime(2026, 1, 1))
    assert clock() == AT


def test_an_instant_in_another_zone_is_kept_and_read_as_utc() -> None:
    plus_five = timezone(timedelta(hours=5))
    clock = FixedClock(datetime(2026, 1, 1, 17, 0, tzinfo=plus_five))
    with use_clock(clock):
        assert utc_now() == AT
        assert utc_now().tzinfo is UTC


def test_under_use_clock_every_stamp_is_the_fixed_instant() -> None:
    clock = FixedClock(AT)
    with use_clock(clock):
        assert now_iso() == AT.isoformat()
        assert create_provenance("user:a", "manual").created_at == AT.isoformat()
        clock.advance(timedelta(seconds=1))
        assert now_iso() == (AT + timedelta(seconds=1)).isoformat()
