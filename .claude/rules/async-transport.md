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

> **Do NOT add `anyio` / `trio` to satisfy `ASYNC240`.** The dependency-free
> fix is `asyncio.to_thread` around the stat/open; adding an async-filesystem
> dependency is rejected by the dependency bar.

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
