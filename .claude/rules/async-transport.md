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

- **Static guard:** ruff's `ASYNC` family (`flake8-async`) is enabled
  repo-wide (root `select` + each package `select`). It flags blocking
  `open()` (`ASYNC230`), `Path`/`os` calls (`ASYNC240`), `time.sleep`
  (`ASYNC251`), and blocking HTTP clients (`ASYNC210`) inside `async def`.
- **Runtime proof:** the `assert_no_blocking()` test construct
  (`from dataknobs_common.testing import assert_no_blocking`) activates the
  `blockbuster` detector and raises `BlockingError` if a blocking syscall
  runs on a live loop inside the block. Write the reproduce-first test
  *first*: it FAILS against the blocking code, PASSES once offloaded.

  > **Blind spot:** `blockbuster` does not patch `readline` / line
  > iteration. For line-iterating readers, additionally pin the offload with
  > a structural worker-thread-identity proof (spy `open`, assert the read
  > ran on the worker thread, not the event-loop thread).

## Suppressing a Finding

A genuine blocking call is **fixed (offloaded), not ignored**. A per-file
`ASYNC` ignore is permitted ONLY for a verified false positive — a cheap,
one-shot, setup-time stat that is not on a hot loop, or a call also reachable
from sync contexts — and MUST carry a one-line justification. Never a blanket
`ASYNC` ignore, and never an ignore on a true-positive blocking site.

> **Do NOT add `anyio` / `trio` to satisfy `ASYNC240`.** The dependency-free
> fix is `asyncio.to_thread` around the stat/open; adding an async-filesystem
> dependency is rejected by the dependency bar.

## References

| Pattern | Reference implementation |
|---|---|
| Swap to an async transport | `AsyncS3Database` (aioboto3), `AsyncPostgresDatabase` (asyncpg), `AsyncSQLiteDatabase` (aiosqlite) |
| Offload a sync call | `AsyncDuckDBDatabase`, `AsyncFileDatabase` (`asyncio.to_thread`) |
| Pump a lazy sync iterator | `aiter_sync_in_thread` (`dataknobs_common.async_iter`) |
