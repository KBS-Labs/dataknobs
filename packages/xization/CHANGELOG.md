# Changelog

All notable changes to the dataknobs-xization package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
