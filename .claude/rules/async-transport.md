# Async Transport — No Blocking I/O on the Event Loop

## The Rule

An `async def` method MUST NOT perform blocking I/O on the event loop. An
async backend, adapter, store, or tool MUST either:

1. **Use an async transport** — `aioboto3`, `asyncpg`, `aiosqlite`,
   `aiohttp`, etc. — for its I/O; **or**
2. **Offload the blocking call off the loop** — wrap the synchronous work in
   `asyncio.to_thread(...)` (whole read/write) or drive a lazy/streaming
   sync iterator through `aiter_sync_in_thread(...)` (bounded, backpressured).

Never hold a synchronous `boto3` / `psycopg2` / `requests` client, and never
do blocking `open()` / `os.path` / `os.stat` / `Path.mkdir` disk I/O,
directly inside an `async def` body.

## Why

A blocking call in an `async def` stalls the **entire** event loop for its
duration — on a shared loop (a multi-tenant server, a bot handling many
concurrent conversations) one synchronous disk write or socket read freezes
every other in-flight task. The cost is invisible in single-request tests and
catastrophic under concurrency.

The harm is therefore to **co-tenants**, and that — not the presence of the
words `async def` — is what scopes this rule. Shipped code cannot see who else
is on its loop and must assume the worst. A loop built for one coroutine and
torn down after it is the one place that assumption is checkable rather than
assumed; see [Test loops](#test-loops--where-the-named-harm-cannot-occur).

## Enforcement

- **Static guard:** ruff's `ASYNC` family (`flake8-async`) is enabled in the
  root `select`, which is the **only** ruff config in the repo: no package
  declares `[tool.ruff]`, so `bin/validate.sh --config <repo-root>/pyproject.toml`,
  a bare `ruff check`, and an editor all resolve the same rules. It flags
  blocking `open()` (`ASYNC230`), `Path`/`os` calls (`ASYNC240`), `time.sleep`
  (`ASYNC251`), blocking HTTP clients (`ASYNC210`/`ASYNC212`), and subprocess
  calls (`ASYNC220`/`ASYNC221`/`ASYNC222`) inside `async def`, plus the
  `ASYNC1xx` style checks such as `ASYNC110` (async busy-wait).

  > **One config, and it stays that way.** Five packages used to carry their
  > own `[tool.ruff]` for IDE invocations, and keeping the two in step was a
  > standing obligation nothing enforced — they diverged in *both* directions,
  > enforcing rules the gate had declined while missing whole families it ran.
  > A new `ASYNC` suppression goes in the root `per-file-ignores` and nowhere
  > else; `tests/test_ruff_config_single_source.py` fails if a second copy
  > appears.

  > **ASYNC240 blind spot:** ruff reliably flags a `Path`/`os` method only
  > when it can see the receiver is a `Path` — typically a directly-
  > constructed `Path(...)` literal. The same call on an attribute-/variable-
  > bound Path it cannot type — `self._db_path.parent.mkdir(...)`,
  > `db_file.parent.mkdir(...)` — is NOT flagged. A green ASYNC lint is
  > necessary but not sufficient; attribute-Path disk I/O in an `async def`
  > must still be caught by review + `assert_no_blocking()`.
- **Runtime proof:** the `assert_no_blocking()` test construct
  (`from dataknobs_common.testing import assert_no_blocking`) activates the
  `blockbuster` detector and raises `BlockingError` if a blocking syscall
  runs on a live loop inside the block. Write the reproduce-first test
  *first*: it FAILS against the blocking code, PASSES once offloaded.

  > **blockbuster blind spot:** `blockbuster` does not patch `readline` / line
  > iteration. For line-iterating readers, additionally pin the offload with
  > a structural worker-thread-identity proof (spy `open`, assert the read
  > ran on the worker thread, not the event-loop thread).

## Suppressing a Finding

**This section governs every blocking-I/O check in the family — the whole
`ASYNC2xx` series.** That series exists to detect exactly one defect:
synchronous work running on the event loop. That is what this rule exists to
prevent, so every member of it is in scope.

The series currently comprises:

| Code | Blocking call detected |
|---|---|
| `ASYNC210` / `ASYNC212` | blocking HTTP client (generic / `httpx`) |
| `ASYNC220` / `ASYNC221` / `ASYNC222` | subprocess create / run / wait |
| `ASYNC230` | blocking `open()` |
| `ASYNC240` | blocking `Path` / `os` call |
| `ASYNC250` | blocking `input()` |
| `ASYNC251` | `time.sleep` |

**The table is illustrative; the definition is the `ASYNC2xx` series.** Scope
this section by that definition, not by the list — a check ruff adds later is
governed the day it ships, with no edit here. The subprocess members deserve
particular note: `subprocess.run(...)` inside an `async def` blocks the loop
for the child process's entire lifetime, which is the most severe form of the
defect this rule addresses.

A genuine blocking call is **fixed (offloaded), not ignored**. A per-file
`ASYNC` ignore is permitted ONLY for a verified false positive — a cheap,
one-shot, setup-time stat that is not on a hot loop, or a call also reachable
from sync contexts — and MUST carry a one-line justification. Never a blanket
ignore of an `ASYNC2xx` check, and never an ignore on a true-positive blocking
site. **A guard that ignores the defects it exists to catch is worse than no
guard**, because it also reports green.

"True positive" there means a call that blocks a loop with something to lose.
A blocking call on a loop built for one test and torn down after it is not a
false positive — it really does block — but it starves nothing, and the next
section governs it under conditions of its own. Nothing in that section
relaxes this paragraph for `packages/*/src`.

> **Do NOT add `anyio` / `trio` to satisfy `ASYNC240`.** The dependency-free
> fix is `asyncio.to_thread` around the stat/open; adding an async-filesystem
> dependency is rejected by the dependency bar.

### Test loops — where the named harm cannot occur

A shipped `async def` runs on a loop whose other occupants it cannot see,
which is why `packages/*/src` is absolute. A test's loop is not that loop:
under `pytest-asyncio` the default test loop scope is `function`, so the loop
is built for one test, carries only that test's coroutine, and is torn down
after it. A blocking `open()` writing a fixture file spends wall-clock the
test was going to spend anyway and freezes nothing, because there is nothing
else there to freeze.

That argument is narrower than it first looks, and it is the easiest one in
this document to over-apply. It licenses an exemption only when **all three**
hold:

1. **The loop belongs to this test alone.** Verify rather than assume: the
   default is `function`, but an `asyncio_default_test_loop_scope` in a
   `pyproject.toml`, or a `loop_scope=` on the test or a fixture it uses, can
   widen it. A session- or module-scoped loop has co-tenants again, and with
   them the exemption ends.
2. **The enclosing `async def` schedules no concurrency** — no
   `asyncio.gather`, no `create_task`, no `TaskGroup`. Once a test runs tasks
   concurrently the interleaving *is* the subject, and a stall in the test's
   own scaffolding can mask the race the test was written to catch.
3. **The call is the test's scaffolding, not its subject** — writing a fixture
   file, stat-ing a path, reading output back to assert on it. Not a call into
   the code under test, and not standing in for one.

#### Fix it if you can — and the fixes are ordered

A waiver is the last of four options, not the first.

1. **Drop the `async` when the function never awaits.** An `async def` fixture
   whose whole body is `tempfile.mkstemp` and `os.remove` is async by
   accident; a plain `@pytest.fixture` runs it before the loop exists. A real
   fix, and it needs no waiver.
2. **Move the work to a sync fixture** that runs before the loop, where the
   enclosing function does genuinely need to be async.
3. **Waive per file**, under the three conditions above, in the form below.
4. **Never relocate the blocking call into a sync helper called from the same
   `async def`.** The `ASYNC2xx` members fire only inside an `async def` body,
   so a one-line extraction turns the cell green while the loop blocks exactly
   as before — measured, not assumed: an `open()` and a `time.sleep()` report
   two findings inline and **zero** from a sync helper one frame away, with no
   runtime difference between the two. That is a green light bought by moving
   code, which is the failure this rule exists to make visible.

#### Where the exemption never reaches

- **`packages/*/src`.** Every `ASYNC2xx` member stays enforced, with no
  test-shaped argument available.
- **Any helper the shipped code can also reach.** If it is importable from a
  package it is `src` for this purpose, whichever directory holds it.
- **`ASYNC251` (`time.sleep`).** Not because the co-tenant argument fails —
  it holds here as anywhere — but because there is nothing to weigh against
  it. `await asyncio.sleep(...)` is a drop-in at no cost, so a waiver would
  switch off a check whose fix is free. A fixed sleep waiting for an external
  service to catch up is also a flakiness defect in its own right, which the
  loop argument neither reaches nor excuses.
- **A file whose subject is blocking behaviour.** A test importing
  `assert_no_blocking` is *about* the blocking/non-blocking distinction, and a
  per-file waiver there switches the check off over precisely the code most
  likely to need it. The offending call in such a file is normally deliberate
  — the red half of a reproduce-first pair, or a handle opened to hand into
  the code whose job is to offload it. Waive per line, `# noqa: <code>` with
  its own reason, never per file.

  This is also the one place `time.sleep` inside an `async def` is correct: a
  test asserting the detector *catches* a blocking sleep has to perform one,
  on a live loop, which is why dropping the `async` is the wrong fix there.
  Per line, named, with the assertion visible beside it.

#### The form of the waiver

Root `per-file-ignores`, one entry per file with its own reason, in the shape
the `ASYNC` and `SIM115` blocks there already use.

**Name the codes; never `ASYNC` or `ASYNC2` bare.** A file exempted for
`ASYNC230` still reports a `time.sleep` or a `subprocess.run` added to it next
year. That precision is the difference between a waiver and a hole.

The reason must say **which of the three conditions carried it**. "It is a
test" is not one of them, and that omission is the point: a directory name is
not an argument, and every condition above is a property of the code that
someone can go and check.

State the cost rather than leaving it implicit: a per-file entry also unflags
a *future* blocking call of that code in that file. That is the price of not
writing a per-line directive at every site, and it is paid per file, with the
file read first.

Where the module a test exercises has a dedicated `*_offload.py` sibling — a
reproduce-first test driving the production path inside `assert_no_blocking()`
— name it in the reason. Not a condition, since a fixture that never touches
production async code needs no sibling; but where one exists it turns "the
scaffolding is not the subject" from an argument into a file name.

### Members with no blocking semantics

`flake8-async` also ships the `ASYNC1xx` series, which is about async *style*
rather than blocking the loop. Two come up in practice:

- **`ASYNC109`** objects to a `timeout` parameter on an `async def`, preferring
  `asyncio.timeout` at the call site. A finding is a signature-shape opinion,
  not a stalled loop: the flagged code performs no I/O on the event loop, and
  "fixing" it changes a public signature. The usual shape is a contract method
  (`DistributedLock.acquire`, `RateLimiter.acquire`) or a graceful-shutdown
  method, where the parameter *is* the published interface and an
  implementation forwards it rather than timing out client-side.
- **`ASYNC110`** objects to an `await asyncio.sleep(...)` poll loop, preferring
  an event/queue primitive. `asyncio.sleep` yields to the loop, so the loop is
  not stalled; the finding is about idiom, not a blocking call.

These are outside this rule. Silencing one — including family-wide — is an
ordinary API decision, made on its own merits and recorded with its rationale
beside the `ignore`. It is not the blanket suppression prohibited above, and
it leaves the `ASYNC2xx` members fully enforced.

**Do not extend this carve-out by analogy.** It covers the `ASYNC1xx` style
checks. If a check detects synchronous work on the loop, it belongs to the
section above regardless of which number it carries.

## References

| Pattern | Reference implementation |
|---|---|
| Swap to an async transport | `AsyncS3Database` (aioboto3), `AsyncPostgresDatabase` (asyncpg), `AsyncSQLiteDatabase` (aiosqlite) |
| Offload a sync call | `AsyncDuckDBDatabase`, `AsyncFileDatabase` (`asyncio.to_thread`) |
| Pump a lazy sync iterator | `aiter_sync_in_thread` (`dataknobs_common.async_iter`) |
