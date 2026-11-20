# The clock

`dataknobs_common.clock` is where the library reads the current time for a
timestamp. Every aware `datetime` timestamp dataknobs writes goes through it,
from a provenance record's `created_at` to a log line's `timestamp`, so a test
can fix all of them at once. Epoch-float and naive stamps do not, and
[What the clock does not reach](#what-the-clock-does-not-reach) names them.

```python
from dataknobs_common.clock import Clock, now_iso, system_clock, use_clock, utc_now
```

The names are imported from the module; the `dataknobs_common` package does
not export them.

## Reading the time

| Name | Returns |
|---|---|
| `utc_now()` | the current time from the clock in use, as an aware UTC `datetime` |
| `now_iso()` | the same, as an ISO 8601 string (`2026-01-01T12:00:00+00:00`) |
| `system_clock()` | the wall-clock time in UTC: the clock in use unless one is set |

With no clock set, `utc_now()` is `datetime.now(UTC)` and `now_iso()` is its
`isoformat()`, microseconds included.

## Fixing the time in a test

`use_clock(clock)` reads the time from `clock` for the rest of the block. It
reaches every aware stamp made under it, including a dataclass default made deep
inside a call the test did not make itself, so nothing has to patch
`datetime`:

```python
from datetime import UTC, datetime, timedelta

from dataknobs_common.clock import now_iso, use_clock
from dataknobs_common.provenance import create_provenance
from dataknobs_common.testing import FixedClock

at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
clock = FixedClock(at)
with use_clock(clock):
    record = create_provenance("user:buyer", "manual")
    clock.advance(timedelta(seconds=5))
    later = now_iso()

assert record.created_at == "2026-01-01T12:00:00+00:00"
assert later == "2026-01-01T12:00:05+00:00"
```

The clock in use before the block comes back when it ends, normally or by an
exception, so blocks nest. `use_clock` yields the clock it was given.

### `FixedClock`

`dataknobs_common.testing.FixedClock` is the reference clock for tests. It
returns the instant it holds, and reading it does not move it.

| Member | What it does |
|---|---|
| `FixedClock(at)` | starts at `at`, an aware `datetime`; a naive one raises `ValueError`, anything else `TypeError` |
| `clock()` / `clock.now` | the instant held |
| `clock.advance(delta)` | moves it by a `timedelta` and returns the new instant |
| `clock.set(at)` | holds `at` from now on, **earlier or later**, and returns it |

Setting an earlier instant is deliberate. A wall clock can be corrected
backwards, and code that stamps one write after another then gives the later
write the earlier time. `set` reproduces that step on demand:

```python
from datetime import UTC, datetime

from dataknobs_common.clock import now_iso, use_clock
from dataknobs_common.testing import FixedClock

clock = FixedClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
with use_clock(clock):
    first = now_iso()
    clock.set(datetime(2026, 1, 1, 11, 59, tzinfo=UTC))
    second = now_iso()
assert second < first
```

## What a clock returns

A `Clock` is any callable taking no arguments and returning an aware
`datetime`.

- **A value in another zone is converted to UTC** on every read, so a stamp
  always means UTC.
- **A naive value is refused** with `ValueError`. It has no single reading as
  UTC, and guessing one is how local time gets written as if it were UTC.
- **Anything that is not a `datetime`** raises `TypeError`.

## Injected clocks

A component that takes a clock in its constructor uses the one it was given.
The per-user state store is one: `UserStateStore.from_config(config,
now=clock)` stamps `_written_at` and runs retention from `clock`. Without one
it reads `utc_now`, so `use_clock` reaches it too. An injected clock is the
more local statement, and it wins over the ambient one.

`FixedClock` serves both ways in, ambient and injected.

## Where the clock reaches

The clock is held in a `contextvars.ContextVar`.

| Where the stamp is made | Clock it reads |
|---|---|
| the block under `use_clock` | the clock set |
| an `asyncio` task created under it | the clock set (a task copies the context) |
| `asyncio.to_thread` called under it | the clock set (the worker copies the context) |
| a coroutine handed to a loop with `asyncio.run_coroutine_threadsafe` under it | the clock set (the caller's context goes with it) |
| a task created **before** the block, such as an event-bus listener | the clock it started with: a task copies the context once, when it is created |
| a concurrent task created outside it | its own: a clock set in one test does not reach another |
| `loop.run_in_executor`, a `ThreadPoolExecutor` worker, a bare `threading.Thread`, the producer of `aiter_sync_in_thread` | `system_clock`: none of them copies the context |

So a component that runs a background task, a listener or a polling loop has
to be started inside the block for its stamps to follow the clock.

To carry the clock onto a thread that does not copy the context, run the work
with `contextvars.copy_context().run(...)`.

On Python 3.14, a new thread starts in a copy of its parent's context when
`sys.flags.thread_inherit_context` is set, which is the default on a
free-threaded build. There a bare thread, a pool worker started under the
block and the `aiter_sync_in_thread` producer carry the clock too.

## Reading a stamp back

Code that compares a stamp with *now* reads *now* from the same clock, so a
test can age what it stamped by moving the clock:

- a config draft expires `max_age_hours` after its `last_updated`, by the
  clock in use;
- the per-user state store's retention runs on the store's clock, injected or
  ambient;
- a provider's `Retry-After` given as an HTTP date becomes a wait measured
  from the clock in use. Under a clock fixed in the past, a real date header
  turns into a wait of months, so a test that fixes the clock and replays a
  recorded response should fix the clock near the recording's date.

## What the clock does not reach

Two kinds of timestamp are written without it, and `use_clock` does not fix
them:

| Kind | Where | Why it has not moved |
|---|---|---|
| epoch floats from `time.time()` | a registry's `registered_at`, a wizard's stage-entry and snapshot times, a memory bank's `created_at`, reasoning observability records | a different representation; moving one changes what a reader stored or compares against |
| naive local `datetime.now()` | an llm conversation's `created_at` / `updated_at` and its nodes' `timestamp`, the prompt builders' rendered `timestamp` | local time written without a zone: moving it needs a decision about the values already stored |

The guard that keeps aware reads in this module, `tests/test_one_clock.py`,
leaves both alone on purpose, and widens when they move.

## Durations are not this clock's job

The time between two events is measured with `time.monotonic()`. A wall clock,
this one included, can be corrected backwards, and a duration taken from two
of its reads can then come out negative.
