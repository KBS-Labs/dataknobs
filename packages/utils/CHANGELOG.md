# Changelog

All notable changes to the dataknobs-utils package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v1.2.16 - 2026-07-15

### Added

- `SimplifiedElasticsearchIndex.index()` gains an optional `op_type` parameter.
  Pass `op_type="create"` for an atomic insert that fails closed on a colliding
  document id: the resulting HTTP 409 is raised as the new
  `ElasticsearchConflictError` (rather than returned), so callers handle a
  create-conflict with the same `try`/`except` shape the native async client
  uses.
- `SimplifiedElasticsearchIndex.update()` and `.delete()` gain optional
  `if_seq_no` / `if_primary_term` parameters for optimistic-concurrency
  (compare-and-set) writes. When both are supplied the write proceeds only if
  the document still carries that `_seq_no`/`_primary_term`; a stale token on an
  existing document is a 409 raised as `ElasticsearchConflictError`, while a
  missing document is a 404 that returns `False` (an absent id never conflicts).
  Omitting them preserves the existing unconditional `bool`-returning behavior
  exactly.

### Fixed

- `SimplifiedElasticsearchIndex` now percent-encodes the document id in the REST
  path for `index()`, `get()`, `update()`, `delete()`, and `exists()`. A document
  id containing `/` (hierarchical keys such as `artifacts/alice/report/final`),
  or any other path-reserved character, was previously interpolated raw into the
  `_doc/<id>` path, so Elasticsearch parsed it as extra route segments and the
  operation silently failed (`index()` returned `{"_id": None, "result": "error"}`).
  Ids are now encoded with `safe=""`, so slash-delimited and other special-char
  ids round-trip correctly.

### Security

- Bumped minimum `nltk` requirement from `>=3.9.4` to `>=3.10.0` to exclude
  versions affected by GHSA-p4gq-832x-fm9v / PYSEC-2026-2078 / CVE-2026-54293
  (CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via percent-encoded
  `..%2f` sequences that bypass the `../` regex check once `url2pathname()`
  decodes them), fixed in 3.10.0. Flagged at the floor resolve by the
  `dependency-update` workflow. The related PYSEC-2026-597 / CVE-2026-12243
  (same path-traversal class) has no upstream fix and remains accepted — not
  reachable from this codebase, which loads only fixed corpus names
  (wordnet/omw-1.4/wordnet_ic) and never passes caller-controlled strings into
  `nltk.data.find()`.

## v1.2.15 - 2026-07-07

### Security

- Acknowledged GHSA-p4gq-832x-fm9v and PYSEC-2026-597 / CVE-2026-12243
  (both CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via
  percent-encoded `..%2f` sequences that bypass the `../` regex check
  once `url2pathname()` decodes them) against the `nltk>=3.9.4` floor,
  flagged at the floor resolve by the `dependency-update` workflow.
  Both affect all `nltk` versions through 3.9.4 with no upstream fix.
  Not reachable from this codebase: `resource_utils` loads only the
  fixed corpus names (`wordnet` / `omw-1.4` / `wordnet_ic`) and never
  passes caller-controlled strings into `nltk.data.find()`. The inline
  floor comment in `pyproject.toml` records the rationale.

## v1.2.14 - 2026-06-22

## v1.2.13 - 2026-05-26

### Added

- **`sslmode` parameter on `DotenvPostgresConnector` and `PostgresDB`.**
  Both now accept an optional `sslmode: str | None = None` that is
  forwarded to `psycopg2.connect` only when set (libpq's own default
  applies otherwise, preserving prior behavior). This lets sync
  Postgres consumers request a TLS mode (`"require"`, `"verify-full"`,
  …) through the connector. When an existing `DotenvPostgresConnector`
  instance is passed to `PostgresDB`, the connector's own `sslmode` is
  retained and the `PostgresDB` argument is ignored.

## v1.2.12 - 2026-05-19

## v1.2.11 - 2026-05-13

### Fixed
- Bumped minimum `psycopg2-binary` requirement from `>=2.8.6` to
  `>=2.9.10` to exclude versions that lack cp312/cp313 wheels.
  2.8.6 has no wheels past cp39, and 2.9.9 lacks cp313; falling
  back to a source build requires `pg_config`/`libpq-dev` and is
  not portable. Surfaced by the floor resolve step in the
  `dependency-update` workflow.

## v1.2.10 - 2026-05-09

### Security
- Bumped minimum `nltk` requirement from `>=3.7` to `>=3.9.4` to
  exclude versions affected by GHSA-rf74-v2fm-23pw, CVE-2026-33230,
  and CVE-2026-33231 (one DoS, two in the WordNet browser HTTP
  component).
- Bumped minimum `requests` requirement from `>=2.25.0` to
  `>=2.33.0` to exclude versions affected by PYSEC-2023-74 and
  GHSA-9hjg-9r4m-mvj7 / GHSA-9wx4-h78v-vm56 / GHSA-gc5v-m9x4-r6x2.
- Bumped minimum `lxml` requirement from `>=4.6.0` to `>=6.1.0` to
  exclude versions affected by PYSEC-2020-62, PYSEC-2021-19,
  PYSEC-2021-852, PYSEC-2022-230, and GHSA-vfmq-68hx-4jfw. This is
  a major-version bump (4.x → 6.x); the public lxml API used in
  `dataknobs_utils.xml_utils` is stable across this range.
- Bumped minimum `python-dotenv` requirement from `>=0.19.0` to
  `>=1.2.2` to exclude versions affected by GHSA-mf9w-mj56-hr94.
  This is a major-version bump (0.x → 1.x); `load_dotenv()` and
  related public APIs are unchanged.

## v1.2.9 - 2026-04-29

### Security

- **`PostgresDB.upload()` and `_create_table` now quote DataFrame column names** with `quote_ident()`. Column names were previously joined raw into INSERT and CREATE TABLE statements, allowing columns named with spaces, reserved words, or special characters to produce invalid SQL. `psql_schema_line` has been extracted as `PostgresDB._psql_schema_line(df, col)` (a `@staticmethod`) and `_build_insert_columns(columns)` has been added as a `@staticmethod` to make the SQL-building logic directly testable. The `isinstance(dtype, np.dtype)` guard replaces the broader `hasattr(dtype, "type")` check, fixing a pre-existing crash with pandas `StringDtype` columns. The float-subtype check is broadened from `np.float64` to `np.floating` so `float32` and other numpy float subtypes map to `real` instead of falling through to `varchar`. Pandas nullable numeric types (`Float32Dtype`, `Float64Dtype`, `Int64Dtype`, etc.) are now detected via `pd.api.types.is_float_dtype` / `is_integer_dtype` rather than silently producing `varchar`. `str.len().max(skipna=True) or 1` replaces `max(str.len())` to handle empty DataFrames without raising `ValueError`.

### Added

- `quote_ident(name, dialect="postgres")` in `dataknobs_utils.sql_utils`: production-grade SQL identifier quoting returning double-quoted identifiers (`"name"` with internal `"` escaped as `""`). Supports `postgres`, `sqlite`, and `duckdb` dialects (all use the same SQL-standard rule). Applied internally to `table_head()`, `upload()`, and `_create_table()` in `PostgresDB`-derived classes. Now raises `ValueError` for unsupported dialects. Removed dead `psycopg2` delegation that silently fell through to the pure-Python rule on every call.
