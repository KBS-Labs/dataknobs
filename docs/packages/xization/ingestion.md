# Knowledge Base Ingestion

The ingestion module provides configuration and processing utilities for loading documents from directories into knowledge bases.

## Overview

The ingestion system provides:

- **KnowledgeBaseConfig**: Configuration for document processing
- **DirectoryProcessor**: Processes documents according to configuration
- **Pattern matching**: Glob patterns for file selection
- **Per-file configuration**: Different settings per file type

## Quick Start

```python
from dataknobs_xization.ingestion import process_directory

# Process all files in directory
for doc in process_directory("./docs"):
    print(f"{doc.source_file}: {doc.chunk_count} chunks")
```

## Configuration

### KnowledgeBaseConfig

```python
from dataknobs_xization.ingestion import KnowledgeBaseConfig, FilePatternConfig

config = KnowledgeBaseConfig(
    name="product-docs",

    # Default chunking for all files
    default_chunking={
        "max_chunk_size": 500,
        "chunk_overlap": 50,
    },

    # File patterns
    patterns=[
        FilePatternConfig(
            pattern="api/**/*.json",
            text_fields=["title", "description"],
        ),
        FilePatternConfig(pattern="**/*.md"),
    ],

    # Exclusions
    exclude_patterns=["**/drafts/**", "**/.git/**"],

    # Default metadata
    default_metadata={"version": "1.0"},
)
```

### Config File

Create `knowledge_base.json` or `knowledge_base.yaml` in your docs directory:

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

## Processing Documents

### DirectoryProcessor

```python
from dataknobs_xization.ingestion import DirectoryProcessor, KnowledgeBaseConfig

config = KnowledgeBaseConfig.load("./docs")
processor = DirectoryProcessor(config, "./docs")

for doc in processor.process():
    if doc.has_errors:
        print(f"Error: {doc.source_file}: {doc.errors}")
    else:
        print(f"Processed: {doc.source_file}: {doc.chunk_count} chunks")
        for chunk in doc.chunks:
            print(f"  - {chunk['text'][:50]}...")
```

### ProcessedDocument

```python
@dataclass
class ProcessedDocument:
    source_file: str              # Path to source file
    document_type: str            # "markdown", "json", or "jsonl"
    chunks: list[dict[str, Any]]  # Processed chunks
    metadata: dict[str, Any]      # Document metadata
    errors: list[str]             # Processing errors

    @property
    def chunk_count(self) -> int: ...

    @property
    def has_errors(self) -> bool: ...
```

## Integration with RAGKnowledgeBase

```python
from dataknobs_bots.knowledge import RAGKnowledgeBase, KnowledgeBaseConfig

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
print(f"Loaded {results['total_chunks']} chunks")
```

## Supported File Types

| Type | Extension | Processing |
|------|-----------|------------|
| Markdown | `.md` | Tree-based chunking with heading preservation |
| JSON | `.json` | Template or field-based text generation |
| JSONL | `.jsonl`, `.ndjson` | Streaming line-by-line processing |
| Compressed | `.json.gz`, `.jsonl.gz` | Automatic decompression |

## Related

- [JSON Chunking](json-chunking.md) - JSON-specific chunking
- [Markdown Chunking](markdown-chunking.md) - Markdown-specific chunking
- [Quality Filtering](quality-filtering.md) - Filtering low-quality chunks
