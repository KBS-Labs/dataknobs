# Pluggable Chunking Abstraction

The `dataknobs_xization.chunking` module provides a pluggable abstraction for
document chunking.  Instead of calling `chunk_markdown_tree()` directly,
consuming code can use `create_chunker()` to resolve the chunker implementation
from configuration — enabling custom chunking strategies without modifying
framework code.

## Overview

- **`Chunker`** ABC defines the interface: `chunk(content, document_info) -> list[Chunk]`
- **`ChunkTransform`** ABC for composable post-processing: `transform(chunks, document_info) -> list[Chunk]`
- **`CompositeChunker`** wraps a chunker + ordered transforms into a pipeline
- **`MarkdownTreeChunker`** is the default implementation, wrapping the existing `MarkdownChunker`
- **`chunker_registry`** and **`transform_registry`** are `PluginRegistry` singletons
- **`create_chunker(config)`** creates a chunker from a config dict, wrapping in `CompositeChunker` when `transforms` is present
- Custom chunkers/transforms can be registered by name or resolved via dotted import path
- **Source positions**: `char_start`/`char_end` on `ChunkMetadata` track character offsets into the original document

## Quick Start

```python
from dataknobs_xization.chunking import create_chunker

# Default — uses MarkdownTreeChunker
chunker = create_chunker({"max_chunk_size": 800})
chunks = chunker.chunk("# Title\nBody text.")

# Explicit selection
chunker = create_chunker({"chunker": "markdown_tree", "max_chunk_size": 800})

# Custom implementation via dotted import path
chunker = create_chunker({"chunker": "my_project.chunkers.RFCChunker"})
```

## Configuration

The `chunker` key in the config dict selects the implementation.  When absent,
it defaults to `"markdown_tree"`.  Remaining keys are forwarded to the
chunker's `from_config()` classmethod (if present) or its constructor.

### In Bot YAML

```yaml
knowledge_base:
  chunking:
    chunker: markdown_tree    # default, can be omitted
    max_chunk_size: 800
    combine_under_heading: true
    generate_embeddings: true
```

```yaml
# Custom chunker:
knowledge_base:
  chunking:
    chunker: my_project.chunkers.RFCChunker
    max_chunk_size: 1200
```

### With Transform Pipeline

```yaml
knowledge_base:
  chunking:
    chunker: markdown_tree
    max_chunk_size: 800
    transforms:
      - merge_small:
          min_size: 200
          max_size: 800
      - quality_filter:
          min_content_chars: 50
          min_words: 5
```

When `transforms` is present, `create_chunker()` wraps the resolved chunker
in a `CompositeChunker` that applies each transform in order.  When absent,
no wrapper is created (zero overhead).

### In Ingestion Config

```yaml
name: product-docs
default_chunking:
  chunker: markdown_tree
  max_chunk_size: 500

patterns:
  - pattern: "rfcs/**/*.md"
    chunking:
      chunker: my_project.chunkers.RFCChunker
      max_chunk_size: 1200
```

## Writing a Custom Chunker

Subclass `Chunker` and implement the `chunk()` method:

```python
from dataknobs_xization.chunking import Chunker, DocumentInfo
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata

class PlaintextChunker(Chunker):
    """Simple chunker that splits on double newlines."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            Chunk(
                text=para,
                metadata=ChunkMetadata(
                    chunk_index=i,
                    chunk_size=len(para),
                    content_length=len(para),
                ),
            )
            for i, para in enumerate(paragraphs)
        ]

    @classmethod
    def from_config(cls, config: dict) -> "PlaintextChunker":
        return cls(max_chunk_size=config.get("max_chunk_size", 1000))
```

### Registration

Register at import time so YAML configs can reference it by name:

```python
from dataknobs_xization.chunking import register_chunker

register_chunker("plaintext", PlaintextChunker)
```

Or use the dotted import path in config without explicit registration:

```yaml
chunking:
  chunker: my_project.chunkers.PlaintextChunker
```

## Transform Pipeline

Transforms are composable post-processing steps that run after the chunker
produces its initial chunks.  They can merge, split, filter, reorder, or
enrich chunks without requiring a custom `Chunker` subclass.

### Built-in Transforms

| Transform | Registry Key | Purpose |
|-----------|-------------|---------|
| `MergeSmallChunks` | `merge_small` | Combine adjacent chunks below `min_size`, respecting heading boundaries. `max_size` prevents cascade merging. |
| `SplitLargeChunks` | `split_large` | Re-split oversized chunks using boundary-aware splitting. |
| `QualityFilterTransform` | `quality_filter` | Wraps `ChunkQualityFilter` as a pipeline stage. |

### Writing a Custom Transform

```python
from dataknobs_xization.chunking import ChunkTransform, register_transform

class RelevanceFilter(ChunkTransform):
    """Filter chunks by relevance score."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def transform(self, chunks, document_info=None):
        return [c for c in chunks if self._score(c) >= self.threshold]

    def _score(self, chunk):
        # Domain-specific relevance scoring
        ...

    @classmethod
    def from_config(cls, config):
        return cls(threshold=config.get("threshold", 0.5))

register_transform("relevance", RelevanceFilter)
```

**Position contract:** Transforms that merge or split chunks must maintain
valid `char_start`/`char_end` on the resulting chunks.  Merges use
`min(char_start)` / `max(char_end)`; splits adjust offsets relative to
the original.  Filters and enrichment transforms need no position changes.

### CompositeChunker

`CompositeChunker` wraps an inner chunker + ordered transforms.  After all
transforms run, `chunk_index` is re-numbered to ensure a clean `0..N`
sequence.  `create_chunker()` creates this automatically when `transforms`
is present in the config.

## Source Positions

Each `Chunk` carries `char_start` and `char_end` on its metadata, tracking
the character offsets into the original source document.  These enable:

- Highlighting the exact source span a chunk came from
- Building citation links to precise document locations
- Mapping retrieval results back to source for verification
- "Show in context" features for RAG results

Positions use Python slice semantics: `source[chunk.metadata.char_start:chunk.metadata.char_end]`
gives the source span the chunk was derived from.

```python
chunker = create_chunker({"max_chunk_size": 500})
source = "# Title\n\nBody text here."
chunks = chunker.chunk(source)

for chunk in chunks:
    print(f"Chunk at [{chunk.metadata.char_start}:{chunk.metadata.char_end}]")
    print(f"  Source span: {source[chunk.metadata.char_start:chunk.metadata.char_end]!r}")
```

**Note:** For body chunks produced with `combine_under_heading=True`, positions
are linearly interpolated when nodes are split, and may be approximate if the
source nodes were non-contiguous.  `0` indicates position is unknown.

## API Reference

### Chunker (ABC)

```python
class Chunker(ABC):
    @abstractmethod
    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]: ...
```

### DocumentInfo

```python
@dataclass
class DocumentInfo:
    source: str = ""                       # Source identifier
    content_type: str = "text/markdown"    # MIME-ish hint
    metadata: dict[str, Any] = ...         # Additional metadata
```

### MarkdownTreeChunker

Default chunker wrapping the existing `MarkdownChunker`.  Accepts all
`MarkdownChunker` parameters plus a `from_config()` classmethod.

Config keys: `max_chunk_size`, `heading_inclusion`, `combine_under_heading`,
`quality_filter`, `generate_embeddings`.

### ChunkTransform (ABC)

```python
class ChunkTransform(ABC):
    @abstractmethod
    def transform(
        self,
        chunks: list[Chunk],
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]: ...
```

### CompositeChunker

```python
class CompositeChunker(Chunker):
    def __init__(self, inner: Chunker, transforms: list[ChunkTransform]): ...
```

### create_chunker(config)

Create a chunker from config.  The `chunker` key selects the implementation
(default: `"markdown_tree"`).  The `transforms` key adds a post-processing
pipeline (wrapping in `CompositeChunker`).  Supports registry keys and
dotted import paths for both chunkers and transforms.

### register_chunker(key, factory, override=False)

Register a custom chunker class under a short name.

### register_transform(key, factory, override=False)

Register a custom transform class under a short name.

### split_text(text, max_size, boundaries=None)

Public utility for boundary-aware text splitting with position tracking.
Returns `list[tuple[str, int, int]]` — `(chunk_text, rel_start, rel_end)`.

### chunker_registry / transform_registry

`PluginRegistry` singletons for chunkers and transforms.  Built-in
implementations are lazily registered on first access.

## Integration

### RAGKnowledgeBase

`RAGKnowledgeBase` resolves the chunker from `chunking_config` automatically:

```python
# Config-driven (YAML or dict)
kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
    "chunking": {"chunker": "markdown_tree", "max_chunk_size": 800},
})

# Or inject a pre-built chunker directly
from my_project.chunkers import RFCChunker

kb = RAGKnowledgeBase(
    vector_store=vs,
    embedding_provider=ep,
    chunker=RFCChunker(max_chunk_size=1200),
)
```

### DirectoryProcessor

`DirectoryProcessor` uses `create_chunker()` for markdown files, resolving
the chunker from the per-file or default chunking config.

## Related

- [Markdown Chunking](markdown-chunking.md) — Low-level `MarkdownChunker` and `chunk_markdown_tree()`
- [Knowledge Base Ingestion](ingestion.md) — Batch directory processing
- [Quality Filtering](quality-filtering.md) — Filtering low-quality chunks
