# JSON Chunking

The JSON chunking module provides utilities for chunking JSON and JSONL documents into semantically meaningful chunks suitable for RAG applications.

## Overview

The JSON chunker handles various structures:

- **Arrays of objects**: Each object becomes one or more chunks
- **Nested objects**: Flattened with path-based keys
- **JSONL/NDJSON files**: Streaming support for large files
- **Compressed files**: Support for `.json.gz` and `.jsonl.gz`

## Quick Start

```python
from dataknobs_xization.json import JSONChunker, JSONChunkConfig

# Configure chunker
config = JSONChunkConfig(
    max_chunk_size=1000,
    text_fields=["title", "description"],
)

chunker = JSONChunker(config)

# Chunk JSON data
data = [
    {"title": "Introduction", "description": "Getting started guide"},
    {"title": "Advanced Usage", "description": "In-depth techniques"},
]

for chunk in chunker.chunk(data, source="docs.json"):
    print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
```

## Configuration

### JSONChunkConfig

```python
config = JSONChunkConfig(
    # Text generation
    max_chunk_size=1000,          # Maximum chunk size in characters
    text_template=None,           # Jinja2 template for text generation
    text_fields=None,             # Specific fields to include

    # Field handling
    nested_separator=".",         # Separator for flattened paths
    array_handling="expand",      # "expand", "join", or "first"
    include_field_names=True,     # Include field names in text
    skip_technical_fields=True,   # Skip _id, timestamp, etc.
)
```

## Text Generation Strategies

### Template-based

```python
config = JSONChunkConfig(
    text_template="""
# {{ name }}

{{ description }}

**Category:** {{ category }}
"""
)
```

### Field Selection

```python
config = JSONChunkConfig(
    text_fields=["title", "body", "summary"],
    include_field_names=True,
)
```

### Automatic (Default)

When no template or fields specified, automatically generates text from all non-technical fields.

## Streaming Large Files

```python
from dataknobs_xization.json import JSONChunker

chunker = JSONChunker()

# Stream JSONL file without loading into memory
for chunk in chunker.stream_chunks("large_data.jsonl.gz"):
    await store_in_vector_db(chunk.text, chunk.metadata)
```

## API Reference

### JSONChunker

```python
class JSONChunker:
    def chunk(
        self,
        data: dict | list,
        source: str | None = None,
    ) -> Iterator[JSONChunk]:
        """Chunk JSON data into JSONChunk objects."""

    def stream_chunks(
        self,
        file_path: str | Path,
    ) -> Iterator[JSONChunk]:
        """Stream chunks from JSON/JSONL file."""
```

### JSONChunk

```python
@dataclass
class JSONChunk:
    text: str                    # Generated text
    metadata: dict[str, Any]     # Flattened metadata
    source_path: str             # Path within JSON (e.g., "[0].items[2]")
    source_file: str             # Original file path
    embedding_text: str          # Optimized text for embedding
    chunk_index: int             # Sequential index
```

## Use Cases

### RAG Knowledge Base

```python
config = JSONChunkConfig(
    text_fields=["title", "content"],
    max_chunk_size=500,
)

chunker = JSONChunker(config)

for chunk in chunker.stream_chunks("products.jsonl"):
    embedding = await embed(chunk.embedding_text or chunk.text)
    await vector_store.add(chunk.text, embedding, chunk.metadata)
```

### API Documentation

```python
config = JSONChunkConfig(
    text_template="""
## {{ method }} {{ path }}

{{ description }}

**Parameters:**
{% for param in parameters %}
- {{ param.name }}: {{ param.description }}
{% endfor %}
"""
)
```

## Related

- [Markdown Chunking](markdown-chunking.md) - Chunking markdown documents
- [Ingestion Module](ingestion.md) - Directory processing
- [Quality Filtering](quality-filtering.md) - Filtering low-quality chunks
