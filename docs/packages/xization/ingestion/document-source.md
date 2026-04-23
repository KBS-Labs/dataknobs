# DocumentSource — Storage-Agnostic Ingestion

`DocumentSource` is the async protocol that `DirectoryProcessor` uses to
enumerate and read files. It decouples ingestion from the local
filesystem so the same pattern-based chunking, excludes, and per-pattern
metadata pipeline can drive any storage backend (local `Path`, in-memory,
S3, etc.).

## Overview

The protocol has two built-in implementations:

- **`LocalDocumentSource`** — backed by a local `pathlib.Path`. Uses
  `Path.glob` for pattern matching.
- **`BackendDocumentSource`** — backed by any
  `KnowledgeResourceBackend` (file / memory / S3). Uses `list_files()`
  plus in-Python glob matching.

Both are async-native. `DirectoryProcessor` drives them through
`process_async()`; the sync `process()` wrapper collects results via
`asyncio.run()`.

## Protocol

```python
from dataknobs_xization.ingestion import (
    DocumentSource,
    DocumentFileRef,
    LocalDocumentSource,
    BackendDocumentSource,
)


class DocumentSource(Protocol):
    async def iter_files(
        self, patterns: Iterable[str]
    ) -> AsyncIterator[DocumentFileRef]:
        """Yield DocumentFileRef for each file matching any pattern."""

    async def read_bytes(self, ref: DocumentFileRef) -> bytes:
        """Read the full contents of a file."""

    async def read_streaming(
        self, ref: DocumentFileRef, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream file contents in byte-sized chunks."""
```

Pattern semantics follow shell-style globs that mirror `pathlib.Path.glob`:

- `*` — any characters except `/`
- `?` — a single non-`/` character
- `**` — any sequence of characters including `/`
- `**/` at segment boundaries — any depth (including zero)

## DocumentFileRef

```python
@dataclass(frozen=True)
class DocumentFileRef:
    path: str         # Relative to the source root, forward slashes
    size_bytes: int   # File size; -1 when the source cannot report it
    source_uri: str   # Opaque identifier for logging/metadata
```

- **Local sources** set `source_uri` to `file:///abs/path`.
- **Backend sources** set `source_uri` to `{backend.uri}/{domain_id}/{path}`
  when the backend exposes a `uri` attribute, otherwise
  `{BackendClassName}://{domain_id}/{path}`.

## LocalDocumentSource

```python
from pathlib import Path
from dataknobs_xization.ingestion import LocalDocumentSource

source = LocalDocumentSource(Path("./docs"))

async for ref in source.iter_files(["**/*.md", "**/*.json"]):
    content = await source.read_bytes(ref)
    print(ref.path, len(content))
```

`iter_files()` calls `Path.glob(pattern)` per pattern and filters to
files only. Directories are skipped. The same file may be yielded more
than once if it matches multiple patterns — `DirectoryProcessor`
deduplicates by `path` before processing.

`read_streaming()` reads via a thread + bounded asyncio queue so the
event loop is not blocked on file I/O. Chunks are yielded incrementally.

## BackendDocumentSource

```python
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_xization.ingestion import BackendDocumentSource

backend = InMemoryKnowledgeBackend()
await backend.initialize()
await backend.create_kb("my-domain")
await backend.put_file("my-domain", "intro.md", b"# Intro\n")

source = BackendDocumentSource(backend, "my-domain")
async for ref in source.iter_files(["**/*.md"]):
    content = await source.read_bytes(ref)
    print(ref.path, len(content))
```

`iter_files()` calls `backend.list_files(domain_id)` once and filters in
Python via a glob-to-regex compiler that matches `Path.glob` semantics.
When `patterns` is empty, all files under the domain are yielded.

`read_streaming()` prefers `backend.stream_file()` when the backend
exposes it; otherwise it falls back to a full `get_file()` followed by
in-memory chunking.

## End-to-end Example — Arbitrary Backend

```python
from dataknobs_xization.ingestion import (
    BackendDocumentSource,
    DirectoryProcessor,
    FilePatternConfig,
    KnowledgeBaseConfig,
)

config = KnowledgeBaseConfig(
    name="my-kb",
    patterns=[FilePatternConfig(pattern="**/*.md")],
    exclude_patterns=["**/drafts/**"],
)

source = BackendDocumentSource(backend, "my-domain")
processor = DirectoryProcessor(config, source)

async for doc in processor.process_async():
    print(doc.source_file, doc.chunk_count)
```

## End-to-end Example — RAG Ingest

`RAGKnowledgeBase.ingest_from_backend()` wires the above together:

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase

kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding_provider": "echo",
    "embedding_model": "test",
})

stats = await kb.ingest_from_backend(backend, "my-domain", config=config)
print(stats["total_files"], stats["total_chunks"])
```

When `config` is `None`, `ingest_from_backend` attempts to load
`_metadata/knowledge_base.(yaml|yml|json)` from the backend's domain
namespace, falling back to `KnowledgeBaseConfig` defaults.

## Writing a Custom DocumentSource

The protocol is runtime-checkable. Any object with
`iter_files`, `read_bytes`, and `read_streaming` works:

```python
class MyCustomSource:
    async def iter_files(self, patterns):
        for path in my_enumeration_logic(patterns):
            yield DocumentFileRef(path=path, size_bytes=-1, source_uri=f"custom://{path}")

    async def read_bytes(self, ref):
        return await my_fetch(ref.path)

    async def read_streaming(self, ref, chunk_size=8192):
        data = await self.read_bytes(ref)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

processor = DirectoryProcessor(config, MyCustomSource())
```

## Related

- [DirectoryProcessor](directory-processor.md) — async-primary processor
  that consumes `DocumentSource`
- [Knowledge Base Ingestion (overview)](../ingestion.md) — configuration
  reference and end-to-end examples
- [RAG Ingestion Guide](../../bots/knowledge/ingestion-guide.md) —
  consumer-facing guide covering local/backend/event-driven ingestion
