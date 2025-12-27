# Knowledge Base Ingestion

This module provides configuration and processing utilities for ingesting documents from directories into knowledge bases. It supports markdown, JSON, and JSONL files with configurable chunking, patterns, and metadata.

## Overview

The ingestion system consists of two main components:

1. **KnowledgeBaseConfig**: Configuration for how documents should be processed
2. **DirectoryProcessor**: Processes documents according to the configuration

## Key Features

- **Pattern-based file selection**: Glob patterns for including/excluding files
- **Per-pattern configuration**: Different chunking/text extraction per file type
- **Config file support**: Load configuration from `knowledge_base.json` or `.yaml`
- **Streaming for large files**: JSONL files processed without loading into memory
- **Automatic type detection**: Handles markdown, JSON, and JSONL automatically
- **Document-level metadata**: Attach metadata to all chunks from a file

## Installation

```bash
uv pip install dataknobs-xization
```

## Quick Start

### Basic Directory Processing

```python
from dataknobs_xization.ingestion import process_directory

# Process all files in directory
for doc in process_directory("./docs"):
    print(f"{doc.source_file}: {doc.chunk_count} chunks")
    for chunk in doc.chunks:
        print(f"  - {chunk['text'][:50]}...")
```

### With Configuration

```python
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
    FilePatternConfig,
)

config = KnowledgeBaseConfig(
    name="product-docs",
    default_chunking={
        "max_chunk_size": 500,
        "chunk_overlap": 50,
    },
    patterns=[
        FilePatternConfig(
            pattern="api/**/*.json",
            text_fields=["title", "description"],
        ),
        FilePatternConfig(
            pattern="**/*.md",
        ),
    ],
    exclude_patterns=["**/drafts/**", "**/test/**"],
)

processor = DirectoryProcessor(config, "./docs")

for doc in processor.process():
    if doc.has_errors:
        print(f"Error processing {doc.source_file}: {doc.errors}")
    else:
        print(f"Processed {doc.source_file}: {doc.chunk_count} chunks")
```

### Loading Config from File

```python
from dataknobs_xization.ingestion import KnowledgeBaseConfig, DirectoryProcessor

# Loads from ./docs/knowledge_base.json or knowledge_base.yaml
config = KnowledgeBaseConfig.load("./docs")

processor = DirectoryProcessor(config, "./docs")
```

## Configuration

### KnowledgeBaseConfig

```python
from dataknobs_xization.ingestion import KnowledgeBaseConfig

config = KnowledgeBaseConfig(
    # Identity
    name="my-knowledge-base",

    # Default chunking for all files
    default_chunking={
        "max_chunk_size": 500,
        "chunk_overlap": 50,
        "combine_under_heading": True,
    },

    # Quality filtering (optional)
    default_quality_filter={
        "min_content_length": 50,
        "min_word_count": 10,
    },

    # File patterns (optional - defaults to **/*.md, **/*.json, **/*.jsonl)
    patterns=[
        FilePatternConfig(pattern="**/*.md"),
        FilePatternConfig(pattern="api/**/*.json", text_fields=["title"]),
    ],

    # Exclusions
    exclude_patterns=[
        "**/drafts/**",
        "**/.git/**",
        "*.tmp",
    ],

    # Default metadata for all chunks
    default_metadata={
        "version": "1.0",
        "source": "documentation",
    },
)
```

### FilePatternConfig

```python
from dataknobs_xization.ingestion import FilePatternConfig

pattern_config = FilePatternConfig(
    # Required
    pattern="api/**/*.json",          # Glob pattern

    # Optional
    enabled=True,                      # Enable/disable this pattern
    chunking={                         # Override default chunking
        "max_chunk_size": 800,
    },
    text_template="{{ title }}: {{ description }}",  # Jinja2 template for JSON
    text_fields=["title", "body"],     # Fields to extract for text
    metadata_fields=["author", "date"], # Fields to include in metadata
)
```

### Config File Format

Create `knowledge_base.json` or `knowledge_base.yaml` in your docs directory:

**JSON:**
```json
{
  "name": "product-docs",
  "default_chunking": {
    "max_chunk_size": 500,
    "chunk_overlap": 50
  },
  "patterns": [
    {
      "pattern": "api/**/*.json",
      "text_fields": ["title", "description"]
    },
    {
      "pattern": "**/*.md"
    }
  ],
  "exclude_patterns": ["**/drafts/**"]
}
```

**YAML:**
```yaml
name: product-docs
default_chunking:
  max_chunk_size: 500
  chunk_overlap: 50

patterns:
  - pattern: "api/**/*.json"
    text_fields:
      - title
      - description
  - pattern: "**/*.md"

exclude_patterns:
  - "**/drafts/**"
```

## API Reference

### KnowledgeBaseConfig

```python
class KnowledgeBaseConfig:
    @classmethod
    def load(cls, directory: str | Path) -> KnowledgeBaseConfig:
        """Load config from directory's knowledge_base.json/yaml file.

        If no config file exists, returns default configuration.
        """

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBaseConfig:
        """Create config from dictionary."""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""

    def get_pattern_config(self, file_path: str | Path) -> FilePatternConfig | None:
        """Get the matching pattern config for a file path."""

    def is_excluded(self, file_path: str | Path) -> bool:
        """Check if a file should be excluded."""

    def get_chunking_config(self, file_path: str | Path) -> dict:
        """Get chunking config for a file (pattern override + defaults)."""

    def get_metadata(self, file_path: str | Path) -> dict:
        """Get metadata for a file including default metadata."""
```

### DirectoryProcessor

```python
class DirectoryProcessor:
    def __init__(
        self,
        config: KnowledgeBaseConfig,
        root_dir: str | Path,
    ):
        """Initialize processor.

        Args:
            config: Knowledge base configuration
            root_dir: Root directory to process
        """

    def process(self) -> Iterator[ProcessedDocument]:
        """Process all documents in the directory.

        Yields ProcessedDocument for each file. Uses streaming
        for large JSON files to avoid memory exhaustion.

        Yields:
            ProcessedDocument for each processed file
        """
```

### ProcessedDocument

```python
@dataclass
class ProcessedDocument:
    source_file: str              # Path to source file
    document_type: str            # "markdown", "json", or "jsonl"
    chunks: list[dict[str, Any]]  # List of processed chunks
    metadata: dict[str, Any]      # Document-level metadata
    errors: list[str]             # Errors encountered during processing

    @property
    def chunk_count(self) -> int:
        """Number of chunks in this document."""

    @property
    def has_errors(self) -> bool:
        """Whether processing encountered errors."""
```

### Chunk Format

Each chunk in `ProcessedDocument.chunks` contains:

```python
{
    "text": "The chunk text content",
    "embedding_text": "Optimized text for embeddings",
    "chunk_index": 0,
    "source_path": "[0].items[1]",  # For JSON
    "metadata": {
        "heading_path": "Section > Subsection",  # For markdown
        "headings": ["Section", "Subsection"],
        # ... additional metadata
    }
}
```

## Use Cases

### RAG Knowledge Base Loading

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase, KnowledgeBaseConfig

# Create knowledge base
kb = await RAGKnowledgeBase.from_config(config)

# Load documents using ingestion config
kb_config = KnowledgeBaseConfig(
    name="docs",
    patterns=[
        FilePatternConfig(pattern="**/*.md"),
        FilePatternConfig(pattern="api/*.json", text_fields=["title", "body"]),
    ],
)

results = await kb.load_from_directory("./docs", config=kb_config)
print(f"Loaded {results['total_chunks']} chunks from {results['total_files']} files")
```

### Batch Processing with Progress

```python
from dataknobs_xization.ingestion import DirectoryProcessor, KnowledgeBaseConfig

config = KnowledgeBaseConfig.load("./docs")
processor = DirectoryProcessor(config, "./docs")

total_chunks = 0
for doc in processor.process():
    total_chunks += doc.chunk_count
    print(f"[{total_chunks:,} chunks] {doc.source_file}")

print(f"Total: {total_chunks:,} chunks")
```

### Custom Processing Pipeline

```python
from dataknobs_xization.ingestion import process_directory

async def process_documents(directory: str):
    for doc in process_directory(directory):
        # Skip documents with errors
        if doc.has_errors:
            log_errors(doc.source_file, doc.errors)
            continue

        # Process chunks
        for chunk in doc.chunks:
            # Generate embedding
            embedding = await embed(chunk["embedding_text"] or chunk["text"])

            # Store in database
            await db.insert({
                "id": f"{doc.source_file}_{chunk['chunk_index']}",
                "text": chunk["text"],
                "embedding": embedding,
                "source": doc.source_file,
                "type": doc.document_type,
                **chunk["metadata"],
                **doc.metadata,
            })
```

## File Type Handling

### Markdown Files

- Parsed into tree structure preserving heading hierarchy
- Chunked with configurable size and overlap
- Quality filtering available
- Heading metadata preserved

### JSON Files

- Arrays: Each item becomes one or more chunks
- Objects: Flattened or template-based text generation
- Supports text_template and text_fields configuration

### JSONL/NDJSON Files

- Streamed line-by-line for memory efficiency
- Supports gzipped files (`.jsonl.gz`)
- Each line becomes one or more chunks

## Testing

```bash
cd packages/xization
uv run pytest tests/test_ingestion.py -v
```

## Related

- [JSON Chunking](../json/JSON_CHUNKING.md) - JSON-specific chunking
- [Markdown Chunking](../markdown/MARKDOWN_CHUNKING.md) - Markdown-specific chunking
- [Quality Filtering](../markdown/RAG_QUALITY_FILTERING.md) - Filtering low-quality chunks
