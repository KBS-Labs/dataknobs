# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A clock a test sets, advances and steps back.

:class:`FixedClock` is a :data:`~dataknobs_common.clock.Clock`: calling it
returns the instant it holds, and reading it does not move it. One object
serves both ways a clock goes in, ambient and injected:

```python
from datetime import UTC, datetime, timedelta

from dataknobs_common.clock import now_iso, use_clock
from dataknobs_common.testing import FixedClock

clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
with use_clock(clock):
    first = now_iso()
    clock.advance(timedelta(seconds=5))
    clock.set(datetime(2025, 12, 31, 23, 59, tzinfo=UTC))   # a step backwards
```

**Setting an earlier instant is deliberate.** A wall clock can be corrected
backwards, and code that stamps one record after another can then write a
later record with an earlier time. :meth:`FixedClock.set` reproduces that step
on demand, where waiting for it on a real clock cannot.
"""

from __future__ import annotations

from datetime import datetime, timedelta

__all__ = ["FixedClock"]


def _aware(at: datetime) -> datetime:
    if not isinstance(at, datetime):
        raise TypeError(f"FixedClock takes an aware datetime, not {type(at).__name__}: {at!r}")
    if at.utcoffset() is None:
        raise ValueError(f"FixedClock takes an aware datetime; {at.isoformat()} is naive")
    return at


class FixedClock:
    """A clock that returns the instant it holds until told otherwise.

    Args:
        at: The instant to start at; an aware ``datetime``. A naive one is
            refused with ``ValueError``, anything else with ``TypeError``.
    """

    def __init__(self, at: datetime) -> None:
        self._now = _aware(at)

    def __call__(self) -> datetime:
        """Return the instant held."""
        return self._now

    def __repr__(self) -> str:
        return f"FixedClock({self._now.isoformat()})"

    @property
    def now(self) -> datetime:
        """The instant held."""
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        """Move the clock by ``delta`` (backwards if it is negative) and return the new instant."""
        self._now = self._now + delta
        return self._now

    def set(self, at: datetime) -> datetime:
        """Hold ``at`` from now on, earlier or later than before, and return it.

        Raises:
            TypeError: ``at`` is not a ``datetime``.
            ValueError: ``at`` is naive.
        """
        self._now = _aware(at)
        return self._now
