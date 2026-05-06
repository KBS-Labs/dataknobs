# Changelog

All notable changes to the dataknobs-llm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Internal
- `FileSystemPromptLibrary._load_file` uses
  `dataknobs_common.config_loading.load_yaml_or_json`. Surface is
  `ValueError` for unsupported extensions, parse failures, and read
  errors. Empty / falsy parsed payloads collapse to `{}`.

## v0.5.10 - 2026-05-06

### Execution Layer

- `ParallelLLMExecutor` gains an opt-in `fail_fast` mode (default `False`,
  no behavior change for existing consumers). When enabled at the executor
  level (`__init__(fail_fast=True)`) or per call (`execute(...,
  fail_fast=True)` / `execute_mixed(..., fail_fast=True)` /
  `execute_sequential(..., fail_fast=True)`), the executor cancels
  remaining pending tasks on the first task failure. Cancelled tasks
  return `TaskResult(success=False, error=asyncio.CancelledError(...))`,
  distinguishable from completion-failures by the error type. Under
  `execute_sequential` the loop breaks on the first failure and the
  returned list is shorter than the input list (callers can detect
  short-circuit via `len(results) < len(tasks)`).
- `ParallelLLMExecutor` accepts `default_per_task_timeout`; `LLMTask` and
  `DeterministicTask` accept a per-task `timeout` override. When set,
  each task's body is bounded by `asyncio.wait_for`, returning
  `TaskResult(success=False, error=asyncio.TimeoutError(...))` on
  overrun. With `RetryConfig`, the timeout bounds each retry attempt
  individually (total elapsed across retries remains the consumer's
  responsibility). Sync `DeterministicTask` callables run on the thread
  executor and cannot be pre-empted mid-call; the awaiter stops waiting
  but the underlying thread continues until the function returns.

## v0.5.9 - 2026-04-29

### Test Infrastructure
- Postgres integration fixtures and the `test_storage_postgres.py` asyncpg
  call site now validate interpolated SQL identifiers via
  `dataknobs_common.testing.safe_sql_ident` (regex-validated; raises
  `ValueError` on anything outside `[A-Za-z_][A-Za-z0-9_]*`). The data-package
  conftest's `pg_database` lookup also moved from f-string interpolation to
  psycopg2 `%s` parameter binding for that string-literal site. Closes R1-01.

### Fixed
- `DataknobsConversationStorage` now propagates `state.metadata` into
  `Record.metadata` when persisting conversations. SQL backends with a
  dedicated metadata column (Postgres, Elasticsearch, etc.) can now
  index and query conversation metadata via
  `list_conversations(filter_metadata={...})` and
  `count_conversations(filter_metadata=...)`. Previously the metadata
  column was `NULL` on every conversation row and `metadata.<key>`
  filters returned no matches on those backends; in-memory backend
  behaviour is unchanged.

  Pre-fix rows in production Postgres databases remain queryable via
  `data->'metadata'`. To make pre-fix rows visible to `filter_metadata`
  on Postgres, run the following one-shot backfill (idempotent):

  ```sql
  UPDATE conversations
     SET metadata = data->'metadata'
   WHERE metadata IS NULL AND data ? 'metadata';
  ```

  (Substitute the actual table name if it isn't `conversations`.)

  Rows where `state.metadata` is an empty dict at save time have their
  metadata column set to `'{}'::jsonb`, not `NULL`. This is functionally
  equivalent to `NULL` for `filter_metadata` queries (no key matches an
  empty object) and matches the `Record.metadata` contract — no
  additional `WHERE` guard is needed on the consumer side.

  `state.metadata` is typed `Dict[str, Any]` and may contain
  JSON-serializable nested values (lists, dicts, numbers, booleans,
  strings, `None`); the in-tree wizard FSM persists nested state under
  `state.metadata["wizard"]`, and rate-limit/timing middleware write
  non-string scalars. On save, `_state_to_record` deep-copies
  `state.metadata` into `Record.metadata`, so post-save mutations of
  nested values do not leak into already-persisted rows. SQL backends
  with a dedicated metadata column index top-level keys;
  `filter_metadata={"key": value}` performs equality on the top-level
  value at that key, so nested-value filtering is outside the
  `filter_metadata` contract.
