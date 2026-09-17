# DirectoryProcessor — Async-Primary API

`DirectoryProcessor` walks a `DocumentSource`, applies pattern-based
chunking per the supplied `KnowledgeBaseConfig`, and yields
`ProcessedDocument` values. It handles markdown, JSON, and JSONL files
and streams large JSON automatically.

The processor is **async-primary**: `process_async()` is the primary
API. `process()` is a thin sync wrapper that collects the async
iterator through
[`run_coro_sync`](https://kbs-labs.github.io/dataknobs/packages/common/guides/sync-bridge/),
so it is callable from plain synchronous code and from inside a running
event loop alike.

## Constructor

```python
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
    LocalDocumentSource,
)

DirectoryProcessor(
    config: KnowledgeBaseConfig,
    root_dir: str | Path | DocumentSource,
    chunker: Chunker | None = None,
)
```

`root_dir` accepts any of:

- A `str` or `pathlib.Path` — wrapped automatically in a
  `LocalDocumentSource`
- A `DocumentSource` instance — `LocalDocumentSource`,
  `BackendDocumentSource`, or a custom implementation

When a `DocumentSource` is passed directly, `processor.root_dir` is
`None` and `processor.source` returns the provided source.

## Async API — `process_async()`

```python
async def main():
    processor = DirectoryProcessor(config, "./docs")

    async for doc in processor.process_async():
        if doc.has_errors:
            print(f"Error: {doc.source_file}: {doc.errors}")
            continue
        print(f"{doc.source_file}: {doc.chunk_count} chunks")
```

Yields each `ProcessedDocument` as it is read from the source. Uses
streaming for JSON files above `STREAMING_THRESHOLD_BYTES` (10 MB) to
cap memory use.

## Sync Wrapper — `process()`

```python
processor = DirectoryProcessor(config, "./docs")
for doc in processor.process():
    print(doc.source_file)
```

`process()` drives `process_async()` on a private loop supplied by
`run_coro_sync`, never on the caller's, so a caller already inside a
running event loop gets the documents rather than `RuntimeError:
asyncio.run() cannot be called from a running event loop`.

It **blocks** the calling thread for the whole walk either way. An async
caller therefore stalls every other task on its loop for the duration —
`process_async()` is still the right call from async code, and this
wrapper is for the `def` sites that cannot await. One coroutine is
driven, so the loop is a throwaway: `process()` holds nothing you have
to close.

The collection is unchanged: the returned iterator is over a list that
is already complete, so `files_skipped` is final as soon as the call
returns, and the streaming `process_async()` offers does not survive the
wrapper.

## ProcessedDocument

```python
@dataclass
class ProcessedDocument:
    source_file: str
    document_type: Literal["markdown", "json", "jsonl"]
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any]
    errors: list[str]

    @property
    def chunk_count(self) -> int: ...

    @property
    def has_errors(self) -> bool: ...
```

## Pattern Handling

Patterns come from `config.patterns` (enabled entries only). When no
patterns are configured, the processor falls back to the default set:

```python
_DEFAULT_PATTERNS = (
    "**/*.md",
    "**/*.json",
    "**/*.jsonl",
    "**/*.ndjson",
    "**/*.json.gz",
    "**/*.jsonl.gz",
    "**/*.ndjson.gz",
)
```

Deduplication across patterns is handled by the processor before
dispatch — a file matching two patterns is processed once.

Excluded files (`config.exclude_patterns`) and config files
(`knowledge_base.json`, `.yaml`, `.yml`) are skipped before dispatch.

## Per-Pattern Chunking

Pattern configs can override the default chunker:

```python
config = KnowledgeBaseConfig(
    name="docs",
    default_chunking={"max_chunk_size": 500},
    patterns=[
        FilePatternConfig(pattern="**/*.md"),
        FilePatternConfig(
            pattern="api/*.json",
            chunking={"max_chunk_size": 1000},
            text_fields=["title", "description"],
        ),
    ],
)
```

Chunker instances are cached by serialized config — the default chunker
is built once in `__init__`, and per-pattern chunkers are built only
when a pattern overrides the default.

## Using Any Backend via DocumentSource

```python
from dataknobs_xization.ingestion import (
    BackendDocumentSource,
    DirectoryProcessor,
)

source = BackendDocumentSource(backend, "my-domain")
processor = DirectoryProcessor(config, source)

async for doc in processor.process_async():
    ...
```

See [DocumentSource](document-source.md) for backend-side details.

## Migration Notes

The previous implementation was sync-only and dispatched by file
extension directly against a `Path`. Since the Phase 1 refactor:

- `process()` is now a thin wrapper over `process_async()`.
- Callers that construct a `DirectoryProcessor` with `str | Path`
  behave unchanged.
- Callers that reach into `processor.root_dir` should expect `None`
  when a custom `DocumentSource` was passed; check `processor.source`
  instead.

## Related

- [DocumentSource](document-source.md) — async protocol for file
  enumeration and reads
- [Knowledge Base Ingestion (overview)](https://kbs-labs.github.io/dataknobs/packages/xization/ingestion/) — configuration
  reference and end-to-end examples
