# Changelog

All notable changes to the dataknobs-llm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **History-redaction primitive** in the new module
  `dataknobs_llm.conversations.history_redaction`: the `HistoryRedaction`
  frozen `StructuredConfig` (`pattern` + `replacement`, eager-compiled in
  `__post_init__`) plus the `_compile_history_redactions` /
  `apply_history_redactions` helpers. `apply_history_redactions` is
  generalized over a `(role_of, content_of, replace_content)` accessor
  trio so callers operating on different element shapes — an `LLMMessage`
  (the middleware) and a plain `dict[str, Any]` (memory backends in
  `dataknobs-bots`) — drive one implementation; a dict-shape convenience
  wrapper `apply_history_redactions_to_dicts` covers the dict call site.
  `HistoryRedaction` is exported from `dataknobs_llm.conversations`.
- **`HistoryRedactionMiddleware`** in
  `dataknobs_llm.conversations.middleware`. New `ConversationMiddleware`
  that rewrites assistant-role message content in `process_request`
  before it reaches the provider; `process_response` is a passthrough so
  the fresh LLM response keeps its full citation set for rendering.
  Constructor accepts either a sequence of typed `HistoryRedaction`
  instances (preferred — reuses the list built for a memory backend's
  `history_redactions`) or the legacy ordered list of `{"pattern":
  <regex>, "replacement": <str>}` dicts, plus an optional `redact_roles`
  override (defaults to `("assistant",)`). Mixing the two shapes in one
  call raises `TypeError`. Each dict spec is validated up front: a
  missing `pattern` key or an empty pattern raises `ValueError` at
  construction time so config typos surface at the config-load boundary
  rather than mid-loop on the first request. Non-content fields on the
  rewritten assistant message — `tool_calls`, `tool_call_id`, `name`,
  `function_call`, `metadata` — are preserved (the clone uses
  `dataclasses.replace`), so agent / tool-use loops keep their
  invocation and pairing fields intact across the rewrite. Persisted
  conversation-tree nodes are NOT mutated — redaction is scoped to the
  in-memory message list that this turn forwards to the LLM, so the
  persisted assistant node keeps the original text and only the
  prompt-feed boundary sees the redacted form. Motivated by the
  *citation carry-over* leak: bots that emit structured citation tokens
  (`[bib:N · …]` headers, bare `bib:N` references) would otherwise have
  the model treat its own prior citations as a reusable source pool on
  the next turn, even when this turn's retrieval no longer surfaces
  those sources. Patterns are applied in declared order — list the more
  specific pattern (a bracketed header) before the more general bare
  token, or the bare-token rule will consume `bib:N` inside the bracket
  and leave a malformed `[ · vendor · …]` header. Exported from
  `dataknobs_llm.conversations`.

### Security

- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.9.0` to `>=2.12.0` to exclude PYSEC-2026-139 (CVSS 7.8,
  deserialization in the pt2 Loading Handler), flagged at the floor
  resolve by the `dependency-update` workflow. The OSV record's
  `last_affected: 2.10.0` makes 2.11.0+ unaffected per OSV semantics;
  2.12.0 was chosen as the latest stable. The bump preserves the
  prior sweep of PYSEC-2025-203/204/206 (fixed in 2.9.0),
  GHSA-887c-mr87-cxwp (CVSS 4.8, 2.8.0), GHSA-3749-ghw9-m3mg (CVSS
  3.3, 2.7.1), and CVE-2025-32434 (RCE in `torch.load`, 2.6.0).

## v0.6.0 - 2026-05-26

### Changed

- `LLMConfig` is now a frozen `StructuredConfig` (was a plain mutable
  dataclass). Fields can no longer be reassigned after construction — derive
  a varied config with `clone(**overrides)` instead. `from_dict` / `to_dict`
  are now inherited from the base.
  - `to_dict()` now emits **every** field, with unset optionals serialized as
    `None` (and `options` as `{}`), so that `from_dict(to_dict())` round-trips
    exactly. The previous hand-rolled `to_dict()` omitted `None`-valued fields;
    code that relied on those keys being absent must adjust. For a
    JSON-serialisable projection (enums rendered as their `.value`), use
    `to_json_dict()`.
  - `repr(config)` now masks `api_key` as `'***'` so the credential cannot leak
    to logs via `repr()` or an f-string. The stored value is unchanged and
    `to_dict()` still carries it for round-tripping.

### Added

- An `llm` resolver is registered into `config_registries`, so a raw `llm`
  config section (e.g. a bot's provider section) can be validated via
  `StructuredConfig.validate()` without constructing a provider.

### Security

- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.8.0` to `>=2.9.0` to exclude PYSEC-2025-203 (CVSS 7.5),
  PYSEC-2025-204 (CVSS 7.5), and PYSEC-2025-206 (CVSS 5.3), flagged at
  the floor resolve by the `dependency-update` workflow. The bump
  preserves the prior sweep of GHSA-887c-mr87-cxwp (CVSS 4.8, 2.8.0),
  GHSA-3749-ghw9-m3mg (CVSS 3.3, 2.7.1), and CVE-2025-32434 (RCE in
  `torch.load`, 2.6.0). PYSEC-2026-139 (CVSS 7.8) has no upstream fix
  yet and remains flagged; it will be addressed when a fixed release
  ships.

## v0.5.14 - 2026-05-20

## v0.5.13 - 2026-05-18

## v0.5.12 - 2026-05-13

### Security
- Bumped minimum `transformers` requirement (extra: `embeddings`) from
  `>=4.53.0` to `>=5.0.0` to exclude GHSA-69w3-r845-3855 (CVSS 6.5),
  the first CVE not covered by the prior floor. 5.0.0 is the GA release
  fixing the new issue. Verified locally via `bin/dk pr --all` — the
  three transformers usage sites in
  `fsm_integration/resources.py` (`pipeline`, `AutoTokenizer`,
  `AutoModel`) are stable across the 4.x → 5.x boundary.
- Bumped minimum `torch` requirement (extra: `embeddings`) from
  `>=2.6.0` to `>=2.8.0` to exclude GHSA-887c-mr87-cxwp (CVSS 4.8,
  fixed in 2.8.0). The bump also sweeps GHSA-3749-ghw9-m3mg (CVSS 3.3,
  fixed in 2.7.1) and CVE-2025-32434 (RCE in `torch.load`, fixed in
  2.6.0). 2.8.0 was previously deferred for GA wheel coverage; coverage
  is now in place across supported platforms.

### Fixed
- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build from
  source against modern Cython (`'build_ext' object has no attribute
  'cython_sources'`). Surfaced by the floor resolve step in the
  `dependency-update` workflow.

## v0.5.11 - 2026-05-09

### Security
- Bumped minimum `aiohttp` requirement (extras: `ollama`, `huggingface`)
  from `>=3.8.0` to `>=3.13.4` to exclude 22 known CVEs (highest
  CVSS 9.1: GHSA-63hf-3vf5-4wqf), including CVE-2024-23334 / GHSA-5m98-qgg9-wh84.
- Bumped minimum `transformers` requirement (extra: `embeddings`) from
  `>=4.30.0` to `>=4.53.0` to exclude 16 known CVEs (highest CVSS 9.0:
  PYSEC-2023-300).
- Bumped minimum `jinja2` requirement from `>=3.1.0` to `>=3.1.6` to
  exclude versions affected by GHSA-cpwx-vrp4-4pq7, GHSA-gmj6-6f8f-6699,
  GHSA-h75v-3vvj-5mfj, and GHSA-q2x7-8rv6-6q7h.
- `torch>=2.6.0` (extra: `embeddings`) is unchanged. Two newer CVEs at
  CVSS 3.3 / 4.8 are tracked but the fix versions are 2.7.1-rc1 (not
  GA) / 2.8.0; will be revisited via the weekly CVE-audit workflow once
  GA wheels are available across supported platforms.

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
