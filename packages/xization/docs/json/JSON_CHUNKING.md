# JSON Chunking for RAG Applications

This module provides comprehensive utilities for chunking JSON and JSONL documents into semantically meaningful chunks suitable for RAG (Retrieval-Augmented Generation) applications.

## Overview

The JSON chunking system handles various JSON structures:

1. **Arrays of objects**: Each object becomes one or more chunks
2. **Nested objects**: Flattened with path-based keys
3. **Large text fields**: Split with configurable chunking
4. **JSONL/NDJSON files**: Streaming support for large files

## Key Features

- **Configurable text generation**: Templates, field selection, or auto-detection
- **Smart field handling**: Skip technical fields, include field names
- **Array expansion**: Each array item becomes a separate chunk
- **Streaming support**: Process JSONL files without loading into memory
- **GZIP support**: Read compressed `.json.gz` and `.jsonl.gz` files
- **Embedding text generation**: Create optimized text for vector embeddings

## Installation

```bash
uv pip install dataknobs-xization
```

## Quick Start

### Basic Usage

```python
from dataknobs_xization.json import JSONChunker, JSONChunkConfig

# Configure chunker
config = JSONChunkConfig(
    max_chunk_size=1000,
    text_fields=["title", "description"],
)

chunker = JSONChunker(config)

# Chunk a JSON document
data = [
    {"title": "Introduction", "description": "Getting started guide"},
    {"title": "Advanced Usage", "description": "In-depth techniques"},
]

for chunk in chunker.chunk(data, source="docs.json"):
    print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
```

### Streaming JSONL Files

```python
from dataknobs_xization.json import JSONChunker

chunker = JSONChunker()

# Stream large JSONL file
for chunk in chunker.stream_chunks("large_data.jsonl"):
    store_in_vector_db(chunk.text, chunk.embedding_text, chunk.metadata)
```

### Using Templates

```python
config = JSONChunkConfig(
    text_template="{{ title }}: {{ description }}\n\nCategory: {{ category }}"
)

chunker = JSONChunker(config)
chunks = list(chunker.chunk(data))
```

## Configuration

### JSONChunkConfig

```python
from dataknobs_xization.json import JSONChunkConfig

config = JSONChunkConfig(
    # Text generation
    max_chunk_size=1000,          # Maximum chunk size in characters
    text_template=None,           # Jinja2 template for text generation
    text_fields=None,             # Specific fields to include in text

    # Field handling
    nested_separator=".",         # Separator for flattened paths (e.g., "user.name")
    array_handling="expand",      # "expand" each item, "join" into single chunk, or "first"
    include_field_names=True,     # Include field names in generated text
    skip_technical_fields=True,   # Skip _id, timestamp, metadata fields
)
```

### Text Generation Strategies

#### 1. Template-based

```python
config = JSONChunkConfig(
    text_template="""
# {{ name }}

{{ description }}

**Category:** {{ category }}
**Price:** ${{ price }}
"""
)
```

#### 2. Field Selection

```python
config = JSONChunkConfig(
    text_fields=["title", "body", "summary"],
    include_field_names=True,  # "title: My Title\nbody: Content..."
)
```

#### 3. Automatic (Default)

When no template or fields specified, automatically generates text from all non-technical fields:

```python
config = JSONChunkConfig(
    skip_technical_fields=True,  # Excludes _id, created_at, etc.
)
```

### Array Handling

#### Expand (Default)

Each array item becomes a separate chunk:

```python
config = JSONChunkConfig(array_handling="expand")

data = [{"title": "A"}, {"title": "B"}]
# Results in 2 chunks
```

#### Join

Combine array items into a single chunk:

```python
config = JSONChunkConfig(array_handling="join")

data = [{"title": "A"}, {"title": "B"}]
# Results in 1 chunk with both items
```

## API Reference

### JSONChunker

```python
class JSONChunker:
    def __init__(self, config: JSONChunkConfig | None = None):
        """Initialize with optional configuration."""

    def chunk(
        self,
        data: dict | list,
        source: str | None = None,
    ) -> Iterator[JSONChunk]:
        """Chunk JSON data into JSONChunk objects.

        Args:
            data: JSON data (dict or list)
            source: Optional source identifier

        Yields:
            JSONChunk objects
        """

    def stream_chunks(
        self,
        file_path: str | Path,
    ) -> Iterator[JSONChunk]:
        """Stream chunks from a JSON or JSONL file.

        Supports .json, .jsonl, .ndjson formats.
        Also supports gzipped files (.json.gz, .jsonl.gz).

        Args:
            file_path: Path to JSON file

        Yields:
            JSONChunk objects
        """
```

### JSONChunk

```python
@dataclass
class JSONChunk:
    text: str                    # Generated text for display/search
    metadata: dict[str, Any]     # All original fields (flattened)
    source_path: str             # Path within JSON (e.g., "[0].items[2]")
    source_file: str             # Original file path (if from file)
    embedding_text: str          # Enriched text optimized for embedding
    chunk_index: int             # Sequential index

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
```

## Use Cases

### RAG Knowledge Base Loading

```python
from dataknobs_xization.json import JSONChunker, JSONChunkConfig

config = JSONChunkConfig(
    text_fields=["title", "content", "summary"],
    max_chunk_size=500,
)

chunker = JSONChunker(config)

# Load product catalog
for chunk in chunker.stream_chunks("products.jsonl"):
    embedding = await embed(chunk.embedding_text or chunk.text)
    await vector_store.add(
        id=f"product_{chunk.chunk_index}",
        text=chunk.text,
        embedding=embedding,
        metadata=chunk.metadata,
    )
```

### API Documentation Processing

```python
config = JSONChunkConfig(
    text_template="""
## {{ method }} {{ path }}

{{ description }}

**Parameters:**
{% for param in parameters %}
- {{ param.name }}: {{ param.description }}
{% endfor %}
""",
)

chunker = JSONChunker(config)

# Process OpenAPI spec endpoints
for chunk in chunker.chunk(openapi_spec["paths"]):
    await kb.add_chunk(chunk)
```

### FAQ Processing

```python
config = JSONChunkConfig(
    text_template="Q: {{ question }}\n\nA: {{ answer }}",
    embedding_prefix="FAQ: ",
)

chunker = JSONChunker(config)

for chunk in chunker.chunk(faq_data):
    # chunk.embedding_text = "FAQ: Q: How do I...?\n\nA: You can..."
    await index_faq(chunk)
```

## Streaming Large Files

For large JSONL files, use streaming to avoid memory issues:

```python
from dataknobs_xization.json import JSONChunker

chunker = JSONChunker()

# Process 100GB JSONL file
chunk_count = 0
for chunk in chunker.stream_chunks("massive_dataset.jsonl.gz"):
    await process_chunk(chunk)
    chunk_count += 1

    if chunk_count % 10000 == 0:
        print(f"Processed {chunk_count} chunks...")

print(f"Total: {chunk_count} chunks")
```

## Integration with Ingestion Module

The JSON chunker integrates with the ingestion module for batch processing:

```python
from dataknobs_xization.ingestion import DirectoryProcessor, KnowledgeBaseConfig

config = KnowledgeBaseConfig(
    name="product-catalog",
    patterns=[
        FilePatternConfig(
            pattern="**/*.json",
            text_fields=["name", "description"],
        ),
    ],
)

processor = DirectoryProcessor(config, "./data")

for doc in processor.process():
    print(f"Processed {doc.source_file}: {doc.chunk_count} chunks")
```

## Testing

```bash
cd packages/xization
uv run pytest tests/test_json_chunker.py -v
```

## Related

- [Markdown Chunking](MARKDOWN_CHUNKING.md) - Chunking markdown documents
- [Ingestion Module](../ingestion/INGESTION.md) - Directory processing
- [Quality Filtering](RAG_QUALITY_FILTERING.md) - Filtering low-quality chunks
