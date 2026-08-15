# Changelog

All notable changes to the dataknobs-utils package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- **`upload` rejected a DataFrame with default column labels, and said only
  `Invalid SQL identifier: 0`.** Rejecting is right — an unnamed SQL column is
  not something `upload` should invent a name for — but the message named
  neither the subject, nor that pandas supplies integer labels by default, nor
  that `df.columns = [...]` is the fix. Labels are now checked up front, and
  the error names every offending position with its type.

- **`PostgresRecordFetcher.get_records` inlined three identifiers unquoted, and
  one of them is caller-supplied per call.** The field list, the table name and
  the ID field name all went into the SQL through bare f-string slots, while
  every other SQL site in the module already used `quote_ident`. Because
  `fields_to_retrieve` is a per-call argument, it was a reachable injection
  vector rather than a hardening gap: a fetcher configured for one table
  returned a column from another one, plus `current_user`, through nothing but
  that parameter. The same gap broke ordinary input — a legitimate `Mixed Case`
  column name failed with `column "mixed" does not exist`.

  All three positions are now quoted. `fields_to_retrieve` is a list of column
  names, which is what the parameter already means on the other `RecordFetcher`
  implementations, so this matches the family rather than narrowing it. `ids` is
  unchanged and needs no quoting: entries pass through `str(value + offset)`,
  which rejects anything non-numeric.

- **`PostgresDB` never closed a connection, and opened a new one per call.**
  psycopg2's `with conn` is a *transaction* scope, not a close, so every
  `query` / `execute` / `upload` left its connection open — and the class had
  no `close()` at all. Nothing accumulated, because CPython reclaims the
  connection when the frame exits, which is an interpreter detail rather than
  anything the code arranged, and it does not cover a caller of the public
  `get_conn()`. What the mask did not hide was the cost: a full TCP+auth
  handshake in front of every call, measured at 79% of wall time for a trivial
  `SELECT 1`.

  `DotenvPostgresConnector` now holds one connection **per thread** and reuses
  it, reopening if it is found closed. Per thread because reuse must not become
  sharing: psycopg2's `with conn` transaction block is not re-entrant, so two
  threads on one connection raise `the connection cannot be re-entered
  recursively` — and beneath that error they would share a transaction, where
  either thread's commit commits the other's uncommitted work. `close()` reaches
  every connection the connector opened on any thread. `PostgresDB` gains
  `close()` and context-manager support; a connector passed in explicitly
  belongs to its caller and is left open.

- **`PostgresDB` generated a `CREATE TABLE` that crashed on boolean and
  timestamp columns, and typed duration columns as `integer`.**
  `_psql_schema_line` named only integer and float, and fell through to
  `df[col].str.len()` for everything else — encoding "not integer and not
  float, therefore a string". A `bool`, `datetime64[ns]`, nullable `boolean` or
  tz-aware `datetime64[ns, tz]` column reached `.str` and raised
  `AttributeError`, so the table could not be created at all; a
  `timedelta64[ns]` column was emitted as `integer`, because `timedelta64`
  subclasses `np.signedinteger` and `np.issubdtype(dtype, np.integer)` reports
  it as one.

  The ladder now maps `bool` → `boolean`, `datetime64` → `timestamp`, tz-aware
  `datetime64` → `timestamptz` (a bare `timestamp` would discard the offset),
  and `timedelta64` → `interval`, alongside the existing `integer` / `real` /
  `varchar`. An integration test uploads one column per family into a live
  PostgreSQL and reads it back, so the emitted types are checked against real
  type input functions rather than only against expected strings.

### Changed

- **Float columns are typed by width too.** `real` is 4-byte `float4`,
  carrying about 7 significant digits, and `float64` is pandas' default: a
  value with more precision was accepted and silently rounded, so
  `1.2345678901234567` read back as `1.2345679`. Unlike its integer sibling
  this never failed, which is what made it the worse of the two -- the round
  trip looked successful. 64-bit floats now map to `double precision`, by the
  same itemsize rule, so a `float32` column keeps `real`.

- **Integer columns are typed by width rather than by family.** `integer` is
  PostgreSQL's 4-byte type while pandas defaults to `int64`, so a column
  holding a value past 2<sup>31</sup> created a column its own data could not
  enter: the `CREATE TABLE` succeeded and the `INSERT` then failed. 64-bit
  integers now map to `bigint`, taken from the dtype's itemsize so a genuinely
  narrow column keeps `integer`. Nullable extension dtypes are measured by the
  numpy dtype behind them, which they do not expose an itemsize for.

  Note the interaction with `CREATE TABLE IF NOT EXISTS`: a table already
  created with the narrower column keeps it, so an existing estate has column
  widths that depend on when each table was made.

- **`_psql_schema_line` is one ladder rather than two.** It previously branched
  on `isinstance(dtype, np.dtype)` and repeated the whole ladder in each half,
  because `np.issubdtype` raises `TypeError` on every pandas `ExtensionDtype`.
  `pd.api.types` predicates answer correctly for numpy dtypes and
  `ExtensionDtype`s alike, so the split — and the drift it allowed between the
  two copies — is gone. This removes the last `np.issubdtype` calls in the
  workspace.

- **The `varchar` width is measured on the rendered value, with the renderer
  that sends it.** `upload` sends `str(value)` for every cell, so
  `_psql_varchar_width` measures `dropna().map(str).str.len()` instead of
  calling `.str` on the column directly. An object column holding non-strings
  previously raised `AttributeError`. `map(str)` rather than `astype(str)`
  because the two disagree on some object payloads — a `bytes` value measures
  3 under `astype` and is sent as the 6-character `b'abc'` — which declared the
  column narrower than the text written into it. String columns are unaffected,
  nulls are still excluded, and an empty column still yields `varchar(1)`.

- **A column of empty strings no longer emits `varchar(0)`.** The width guard
  covered "nothing to measure" but not "measures zero", so a column whose
  values are all `""` — routine in CSV and ETL loads — produced a declaration
  PostgreSQL refuses outright (`length for type varchar must be at least 1`),
  losing the whole table rather than a value. The floor is now 1 in both cases.

## v2.0.0 - 2026-08-11

### Changed

- `RequestHelper`'s request body accepts a serialized `str` or `bytes` as well
  as a mapping. The bodies are sent as `data=`, where a mapping is form-encoded
  and a string is sent verbatim — which is how the Elasticsearch helper in this
  package has always passed pre-serialized JSON, against a declared type that
  admitted only the mapping.

- The five convenience wrappers declare their real return type,
  `ServerResponse`, rather than `Any`; so does `ElasticsearchIndex._request`.
  Their docstrings already said so.

### Fixed

- **`RequestHelper.get` / `post` / `put` / `delete` / `head` sent requests with
  no timeout at all.** The convenience wrappers spell "unset" as `None` and
  passed it straight through to `request()`, which spells it as `0` and so
  never substituted the helper's configured timeout. `timeout=None` reaches
  `requests` as *wait indefinitely* — on a call the caller believed carried the
  default it configured. `request()` now treats both spellings as unset, so
  every wrapper falls back correctly; an explicitly passed timeout still wins.

- **`load_project_vars` could return `None` values and raise `TypeError` when
  asked to set the environment.** A bare `KEY` line in a `.project_vars` or
  `.env` file — no `=` at all — is reported by `python-dotenv` as `None`, which
  the declared `dict[str, str]` return type does not admit and which
  `os.environ` cannot hold. Such entries are now dropped rather than coerced:
  `KEY=` remains how an empty value is spelled, so coercing would turn a
  malformed line into a plausible-looking one.

- Raised the `nltk` floor to `>=3.10.2`, excluding the broken 3.10.1
  release. 3.10.1 shipped an import-security hook (`nltk/inisec.py`) that
  blocked any module whose install path resolved under the current working
  directory, without excluding an in-tree virtualenv. With the venv inside
  the project — uv's default layout — every nltk-initiated `import regex`
  raised `ImportError: Blocked import of regex from current working
  directory`, which in this workspace made six of the nine workspace packages fail to import at all;
  running with `cwd=/` blocked the standard library. The `PYTHONSAFEPATH`
  workaround named in the error message does not help, because the check
  is path containment rather than `sys.path` membership. Upstream removed
  the module in 3.10.2.

## v1.2.18 - 2026-07-29

## v1.2.17 - 2026-07-20

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
