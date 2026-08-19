# Changelog

All notable changes to the dataknobs-xization package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v2.1.0 - 2026-08-19

### Security

- **A `DocumentFileRef` can no longer read outside a `LocalDocumentSource`'s
  root.** `read_bytes` and `read_streaming` composed `root / ref.path` and
  opened the result unchecked, so a caller-built ref carrying `..` read a file
  from outside the tree — and an absolute `ref.path` discarded the root
  outright, which is the wider of the two spellings rather than a narrower
  case. Both readers now raise `PathEscapeError` before any filesystem call,
  so an escaping ref fails identically whether or not it names something that
  exists.
  Nesting is untouched — `sub/a.md` and an interior `a/../b` still read, and
  so does an absolute ref that lands back inside the root, because
  containment is judged on where the ref lands rather than on how it is
  spelled. The `DocumentSource` protocol now states the rule, so a
  consumer-written implementation over a bounded store inherits it.

- **A glob pattern could enumerate outside a `LocalDocumentSource`'s root.**
  `iter_files` handed each pattern straight to `Path.glob`, which treats `..`
  as an ordinary literal segment and descends through it — so a pattern of
  `../secrets/*.env` yielded refs for files outside the tree, each carrying
  that file's real size and its resolved absolute `source_uri`, which travel
  onward into chunk metadata. Patterns arrive from `IngestionConfig`, the
  same config plane every other guard in this area exists for.

  The claim that no check was needed here — that the class "already treats
  the root as its boundary" because every ref is derived with
  `relative_to(root)` — does not hold: `relative_to` re-expresses a path
  lexically and enforces nothing, returning `../outside/x` rather than
  raising. A match outside the root is now skipped and logged, and a ref's
  path is recorded in its canonical spelling so one file has one ref path
  whichever pattern found it. **Breaking** for a configuration whose patterns
  deliberately reached outside the declared root — such an ingest previously
  enumerated those files and then failed on the first read, since reading was
  already bounded; it now consistently ignores them.

## v2.0.0 - 2026-08-11

### Changed

- **A `chunker:` or transform key written `module.path:Name` now resolves
  instead of failing as an unregistered plugin.** The gate deciding whether a
  key names a class to import or a registry entry tested only for `.`, so a
  path using the other separator this workspace accepts was never handed to the
  resolver at all — it fell through to a registry lookup and reported that no
  such plugin was registered, which is not what went wrong. Both separators are
  recognised now. Registry keys are plain identifiers (`markdown_tree`,
  `merge_small`) and contain neither, and a registered key wins in any case:
  a path is resolved only for a key the registry does not already hold, so
  widening the gate cannot shadow an existing entry.

- **Resolving one of those keys fails as a `ConfigurationError`.** This site
  raised `ImportError`, `AttributeError` or `TypeError` depending on how the
  path was wrong, so a caller wrapping chunker construction in
  `except ConfigurationError` — which catches every other dotted-path fault in
  the workspace — caught none of them. They are now `DottedPathError` (path
  unresolvable) and `DottedPathTypeError` (resolved, not a `Chunker` /
  `ChunkTransform` subclass), both `ConfigurationError` subclasses, raised by
  the shared `dataknobs_common.imports` resolver rather than a local copy of
  it. Catch the old three and you catch nothing; catch `ConfigurationError`.

### Security

- **A config file's resolved path and the parser's text no longer reach the
  error message.** `KnowledgeBaseConfig.load` reported a failure by relaying
  both, and a parser reports a syntax error by quoting the line it choked on —
  an unterminated quote on an `api_key` puts the key in the text. The message
  now names the file and the exception class, with the parser's own words on
  `__cause__`. Matches what `dataknobs-config` does for the same failure, so
  the two loaders cannot disagree about it.


### Fixed

- `MultiAuthorityData.get_unique_vals_df` raised `TypeError` for columns with
  a pandas extension dtype (nullable `Int64`/`Float64`/`boolean`, `category`,
  `string`). Integer detection moved from `np.issubdtype` to
  `pd.api.types.is_integer_dtype`, so these columns produce a unique-value
  frame like any other — integer columns indexed by value, others by position.

  **Behaviour change for `timedelta64` columns.** `np.timedelta64` subclasses
  `np.signedinteger`, so the old check treated a timedelta column as integer
  and used the raw timedelta values as row IDs. It is now treated as
  non-integer and gets a positional 0..n-1 index. This is the only column type
  whose branch changed without previously raising; `datetime64`, `bool`, and
  the signed/unsigned numpy integer and float dtypes are unaffected.
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

## v1.3.14 - 2026-07-29

## v1.3.13 - 2026-07-20

## v1.3.12 - 2026-07-15

### Security

- Bumped minimum `nltk` requirement from `>=3.9.4` to `>=3.10.0` to exclude
  versions affected by GHSA-p4gq-832x-fm9v / PYSEC-2026-2078 / CVE-2026-54293
  (CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via percent-encoded
  `..%2f` sequences that bypass the `../` regex check once `url2pathname()`
  decodes them), now fixed in 3.10.0 — previously acknowledged as unfixed in
  v1.3.11. Flagged at the floor resolve by the `dependency-update` workflow. The
  related PYSEC-2026-597 / CVE-2026-12243 (same path-traversal class) still has
  no upstream fix and remains accepted — not reachable from this codebase, which
  loads only fixed corpus names (wordnet/omw-1.4/wordnet_ic) and never passes
  caller-controlled strings into `nltk.data.find()`.

## v1.3.11 - 2026-07-07

### Security

- Acknowledged GHSA-p4gq-832x-fm9v and PYSEC-2026-597 / CVE-2026-12243
  (both CVSS 7.5, path traversal in `nltk.data.find()` / `load()` via
  percent-encoded `..%2f` sequences that bypass the `../` regex check
  once `url2pathname()` decodes them) against the `nltk>=3.9.4` floor,
  flagged at the floor resolve by the `dependency-update` workflow.
  Both affect all `nltk` versions through 3.9.4 with no upstream fix.
  Not reachable from this codebase: no `nltk.data.find()` / `load()`
  call site takes caller-controlled input. The inline floor comment in
  `pyproject.toml` records the rationale.

## v1.3.10 - 2026-06-29

## v1.3.9 - 2026-06-22

### Changed

- ruff's `ASYNC` lint family (`flake8-async`) is now enforced for this
  package, so blocking I/O on the event loop inside `async def` code is
  caught at lint time. See the `async-transport` authoring rule.

- **`LocalDocumentSource.iter_files` no longer blocks the event loop.**
  The `Path.glob` walk and per-path `stat` are blocking filesystem
  calls; they are now collected in a single worker-thread hop via
  `asyncio.to_thread`, matching the already-offloaded `read_bytes` /
  `read_streaming`. Only the lightweight `DocumentFileRef` list is
  materialized — file contents are still read lazily per file.
  Behavior and ordering are unchanged.
- **`DirectoryProcessor.process_async` no longer blocks the loop when
  ingesting YAML/CSV files.** The YAML/CSV → markdown conversion is
  offloaded via `asyncio.to_thread`: `ContentTransformer` decides
  path-vs-inline-content with `Path(content).exists()` (a blocking
  `stat`), which previously ran on the event loop even though the
  content was already read into memory through the offloaded source.
  Markdown and (non-streaming) JSON paths were already loop-safe. Output
  is unchanged.
- **`DirectoryProcessor.process_async` no longer blocks the loop when
  streaming large local JSON/JSONL files.** The streaming branch hands
  the on-disk path to a *lazy* synchronous chunker generator that
  `open`/`gzip.open`s the file and reads forward on every chunk pull —
  previously each pull ran on the event loop. The generator is now driven
  on a worker thread and its chunks are pumped to the async consumer
  across a bounded queue, so the file open, gzip decompression, and every
  read happen off the loop. Streaming is preserved (chunks are not
  buffered whole-file), backpressure keeps memory bounded, and abandoned
  iteration tears the worker thread down and releases the file handle.
  gzip handling and path/format dispatch are unchanged. The remote
  single-JSON-tree branch (whole-tree parse of an in-memory buffer) is
  driven through the same worker-thread primitive — that parse is
  CPU-bound rather than blocking I/O, but a large tree would still stall
  the loop, so it is offloaded too.

Together these make the local-filesystem async directory-ingest path
(`process_async`, and `RAGKnowledgeBase.load_from_directory` above it)
loop-friendly across markdown, YAML, CSV, and JSON corpora — including
large streamed JSON/JSONL.

## v1.3.8 - 2026-06-02

## v1.3.7 - 2026-05-20

## v1.3.6 - 2026-05-18

### Added

- **`BackendDocumentSource(file_filter=)`** — optional keyword-only
  `Callable[[KnowledgeFile], bool]` predicate. Evaluated in
  `iter_files` *after* the glob/pattern match (and applied even when
  `patterns` is empty), it restricts enumeration to a subset of the
  backend's files. `None` (default) enumerates every matching file —
  behavior-identical to prior releases, so no existing caller
  changes. This is the source-layer seam that lets a per-file delta
  re-ingest (`dataknobs-bots`
  `KnowledgeIngestionManager.ingest_changes`) re-embed only the
  changed files while reusing the full pattern/chunking pipeline.

## v1.3.5 - 2026-05-09

### Fixed

- **`MarkdownChunker._create_chunk` no longer lets caller-supplied
  node metadata overwrite the chunker-supplied `node_type`** in
  `ChunkMetadata.custom`. Defense-in-depth: the md_parser callers
  do not currently set `node_type` in node metadata, so the path
  is practically unreachable today. The safeguard becomes zero
  marginal cost once
  `dataknobs_common.metadata.enforce_immutable_keys` exists, and
  emits a `WARNING` if a colliding override is ever attempted.

- **`ChunkMetadata.to_dict()` no longer lets `custom` overwrite
  structured fields.** Pre-fix, `to_dict` ended with
  `**self.custom`, so a custom entry sharing a key with a
  structured field (`headings`, `chunk_index`, `chunk_size`,
  `line_number`, `content_length`, etc.) silently overwrote the
  structured value in the serialized dict — same vulnerability
  class as the `_create_chunk` `node_type` defense, but covering
  the entire system-field surface. Post-fix, `**self.custom` is
  unpacked FIRST so structured fields win.

### Security
- Bumped minimum `nltk` requirement from `>=3.9.1` to `>=3.9.4` to
  exclude versions affected by GHSA-rf74-v2fm-23pw, CVE-2026-33230,
  and CVE-2026-33231 (one DoS, two in the WordNet browser HTTP
  component).

### Changed
- `KnowledgeBaseConfig._load_file` raises `IngestionConfigError` for
  malformed or unreadable config files. `yaml.YAMLError`,
  `json.JSONDecodeError`, and `OSError` no longer escape; callers
  should catch `IngestionConfigError`.

### Internal
- `KnowledgeBaseConfig._load_file` uses
  `dataknobs_common.config_loading.load_yaml_or_json`. Surface is
  `IngestionConfigError`.

## v1.3.4 - 2026-05-06

## v1.3.3 - 2026-04-23

### Added
- `DocumentSource` async protocol plus `DocumentFileRef` dataclass
  and `LocalDocumentSource` / `BackendDocumentSource` implementations
  (`dataknobs_xization.ingestion.source`). Decouples ingestion from
  the local filesystem so the same pattern-based pipeline can drive
  any storage backend. `BackendDocumentSource` derives a literal
  prefix from configured patterns and passes it to
  `backend.list_files(prefix=...)`.
- `DirectoryProcessor` dispatches `.md`, `.markdown`, `.txt`,
  `.yaml`, `.yml`, `.csv`, `.json`, `.jsonl`, `.ndjson` (plus `.gz`
  variants for JSON). YAML and CSV are transformed to markdown via
  `ContentTransformer` before chunking.
- `DirectoryProcessor.files_skipped` counter exposing the number of
  config files, excluded paths, and unsupported-extension files
  skipped during iteration.
- `KnowledgeBaseConfig.load()` resolves a config file from the
  directory root (`knowledge_base.(yaml|yml|json)`) and, as a
  fallback, from a `_metadata/` subdirectory. Symmetric with
  `RAGKnowledgeBase.ingest_from_backend`'s backend-side lookup.
- `ProcessedDocument.source_path` — source-relative file path
  (stable across local and backend sources; suitable for metadata
  filtering).
- JSONL streaming from non-local `DocumentSource`s parses one
  object per line directly from the async byte iterator, without
  buffering the full file.

### Changed
- `DirectoryProcessor.process()` is a thin sync wrapper over
  `process_async()`. The sync API is unchanged for callers; async
  callers should prefer `process_async()` directly. `process()`
  cannot be called from inside a running event loop.
- `DirectoryProcessor` constructor accepts a `DocumentSource` in
  addition to `str | Path` for `root_dir`. When a `DocumentSource`
  is passed directly, `processor.root_dir` is `None`; use
  `processor.source` to access the underlying source.
- `JSONChunker.stream_chunks` and
  `dataknobs_utils.json_utils.stream_json_data` accept file-like
  objects (`TextIO` / `BinaryIO`) in addition to paths. Existing
  path-based callers are unaffected.
