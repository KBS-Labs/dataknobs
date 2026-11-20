# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The one place the library reads the current time for a timestamp.

:func:`utc_now` returns the current time as an aware UTC
:class:`~datetime.datetime`, and :func:`now_iso` returns it as an ISO 8601
string. Every aware timestamp dataknobs writes is read through one of them,
from a record's default to a log line's ``timestamp``.

**A test fixes the time with** :func:`use_clock`. It sets the clock for the
current context, so it reaches every stamp made under it, including a
dataclass default made deep inside a call the test did not make itself:

```python
from datetime import UTC, datetime

from dataknobs_common.clock import now_iso, use_clock

at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
with use_clock(lambda: at):
    assert now_iso() == "2026-01-01T12:00:00+00:00"
```

:class:`~dataknobs_common.testing.FixedClock` is the reference clock for
tests: it can be advanced, and set to an earlier instant on purpose.

**What a clock returns.** A :data:`Clock` returns an aware ``datetime``. A
value in another zone is converted to UTC on every read, so a stamp always
means UTC. A naive value is refused with ``ValueError``: it has no single
right reading, and guessing one is how local time gets written as if it were
UTC.

**Where the clock reaches.** The clock is held in a
:class:`~contextvars.ContextVar`. :mod:`asyncio` copies the context into each
task, and :func:`asyncio.to_thread` copies it into its worker, so a clock set
in one test does not reach a concurrent one. A
:class:`~concurrent.futures.ThreadPoolExecutor` and a bare
:class:`threading.Thread` do **not** copy the context, so a stamp made on such
a thread reads :func:`system_clock`.

**Where a constructor takes a clock, that clock wins.** A component that
accepts a :data:`Clock` (the per-user state store's ``components["now"]``)
uses the one it was given, and defaults to :func:`utc_now`, so the ambient
clock reaches it unless one was injected.

**Durations are not this clock's job.** The time between two events is
measured with :func:`time.monotonic`, which a wall-clock correction cannot
move backwards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

__all__ = [
    "Clock",
    "now_iso",
    "system_clock",
    "use_clock",
    "utc_now",
]

#: A source of the current time: called with no arguments, it returns an
#: aware :class:`~datetime.datetime`.
Clock = Callable[[], datetime]


def system_clock() -> datetime:
    """Return the wall-clock time in UTC: the clock in use unless one is set."""
    return datetime.now(UTC)


_CLOCK: ContextVar[Clock] = ContextVar("dataknobs_clock", default=system_clock)


def utc_now() -> datetime:
    """Return the current time, from the clock in use, as an aware UTC datetime.

    Raises:
        TypeError: The clock returned something other than a ``datetime``.
        ValueError: The clock returned a naive ``datetime``.
    """
    value = _CLOCK.get()()
    if not isinstance(value, datetime):
        raise TypeError(f"a clock must return a datetime, not {type(value).__name__}: {value!r}")
    if value.utcoffset() is None:
        raise ValueError(
            f"a clock must return an aware datetime; {value.isoformat()} is naive, "
            "and a naive time has no single reading as UTC"
        )
    return value.astimezone(UTC)


def now_iso() -> str:
    """Return the current time, from the clock in use, as a UTC ISO 8601 string."""
    return utc_now().isoformat()


@contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """Read the time from ``clock`` for the rest of the block, in this context.

    The clock in use before the block is restored when it ends, whether it
    ends normally or by an exception, so blocks nest.

    Args:
        clock: Called for each read; returns an aware ``datetime``.

    Yields:
        ``clock``, so ``with use_clock(FixedClock(at)) as clock:`` keeps a
        handle to advance it.
    """
    token = _CLOCK.set(clock)
    try:
        yield clock
    finally:
        _CLOCK.reset(token)
