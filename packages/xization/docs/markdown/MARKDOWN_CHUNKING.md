# Markdown Chunking for RAG Applications

This package provides comprehensive utilities for splitting and chunking markdown documents into semantically meaningful chunks suitable for RAG (Retrieval-Augmented Generation) applications.

## Overview

The markdown chunking system consists of three main components:

1. **Parser (`md_parser.py`)**: Converts markdown text into a Tree structure that preserves heading hierarchy
2. **Chunker (`md_chunker.py`)**: Generates chunks from the Tree structure with configurable parameters
3. **Streaming Processor (`md_streaming.py`)**: Handles large documents with memory management

## Key Features

- **Preserves heading hierarchy**: Each chunk maintains the full path of headings from root to its content
- **Flexible heading inclusion**: Include headings in text, metadata, both, or neither
- **Multiple output formats**: Markdown, plain text, or structured JSON
- **Configurable chunk sizing**: Control maximum chunk size with smart boundary detection
- **Streaming support**: Process large documents without loading entirely into memory
- **Tree-based structure**: Uses the `Tree` data structure from `dataknobs_structures`

## Installation

The markdown chunking utilities are part of the `dataknobs-xization` package:

```bash
uv pip install dataknobs-xization
```

## Quick Start

### Basic Usage

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree

# Parse markdown into tree structure
markdown_text = """
# Introduction
This is the introduction.

## Background
Some background information.
"""

tree = parse_markdown(markdown_text)

# Generate chunks
chunks = chunk_markdown_tree(tree, max_chunk_size=500)

for chunk in chunks:
    print(f"Chunk {chunk.metadata.chunk_index}:")
    print(f"Headings: {chunk.metadata.get_heading_path()}")
    print(f"Text: {chunk.text}")
    print()
```

### Streaming Large Files

```python
from dataknobs_xization import stream_markdown_file

# Process large file incrementally
for chunk in stream_markdown_file("large_document.md", max_chunk_size=1000):
    # Process each chunk as it becomes available
    store_in_database(chunk.text, chunk.metadata)
```

## Architecture

### Tree Structure

The parser builds a Tree where:
- Root node represents the document
- Heading nodes are parents to their sub-headings and body text
- Body text nodes are leaf nodes containing the actual content

Example structure:
```
ROOT
├─ H1: Introduction
│  ├─ Body: "This is the introduction."
│  └─ H2: Background
│     └─ Body: "Some background information."
```

### MarkdownNode Data Structure

Each tree node contains a `MarkdownNode` with:
- `text`: The text content
- `level`: Heading level (1-6) or 0 for body text
- `node_type`: 'heading' or 'body'
- `line_number`: Source line number (for debugging)

### Chunk Data Structure

Each chunk contains:
- `text`: The chunk text (with or without headings based on configuration)
- `metadata`: A `ChunkMetadata` object with:
  - `headings`: List of heading texts from root to chunk
  - `heading_levels`: Corresponding heading levels
  - `line_number`: Starting line number
  - `chunk_index`: Sequential index
  - `chunk_size`: Size in characters
  - `custom`: Dictionary for custom metadata

## API Reference

### Parsing

#### `parse_markdown(source, max_line_length=None, preserve_empty_lines=False)`

Parse markdown content into a tree structure.

**Parameters:**
- `source`: Markdown content as string, file object, or line iterator
- `max_line_length`: Maximum length for body text lines (None for unlimited)
- `preserve_empty_lines`: Whether to preserve empty lines

**Returns:** Tree with root node containing the document structure

**Example:**
```python
from dataknobs_xization import parse_markdown

tree = parse_markdown("# Title\nBody text.", max_line_length=100)
```

#### `MarkdownParser`

Class for parsing markdown with configurable options.

```python
from dataknobs_xization import MarkdownParser

parser = MarkdownParser(max_line_length=1000, preserve_empty_lines=False)
tree = parser.parse(markdown_content)
```

### Chunking

#### `chunk_markdown_tree(tree, max_chunk_size=1000, heading_inclusion=HeadingInclusion.BOTH, chunk_format=ChunkFormat.MARKDOWN, combine_under_heading=True)`

Generate chunks from a markdown tree.

**Parameters:**
- `tree`: Tree structure built from markdown
- `max_chunk_size`: Maximum size of chunk text in characters
- `heading_inclusion`: How to include headings (BOTH, IN_TEXT, IN_METADATA, NONE)
- `chunk_format`: Output format (MARKDOWN, PLAIN, DICT)
- `combine_under_heading`: Whether to combine body text under same heading

**Returns:** List of Chunk objects

**Example:**
```python
from dataknobs_xization import chunk_markdown_tree, HeadingInclusion, ChunkFormat

chunks = chunk_markdown_tree(
    tree,
    max_chunk_size=500,
    heading_inclusion=HeadingInclusion.BOTH,
    chunk_format=ChunkFormat.MARKDOWN
)
```

#### `MarkdownChunker`

Class for chunking with configurable parameters.

```python
from dataknobs_xization import MarkdownChunker, HeadingInclusion

chunker = MarkdownChunker(
    max_chunk_size=1000,
    heading_inclusion=HeadingInclusion.BOTH
)

chunks = list(chunker.chunk(tree))
```

#### Text Splitting Strategy

When body text under a single heading exceeds `max_chunk_size`, the chunker splits
it using a priority-based boundary detection strategy. Within each window of
`max_chunk_size` characters, it searches backward for the best available break point:

1. **Paragraph boundary** (`\n\n`) — highest priority, preserves paragraph-level coherence
2. **Sentence boundary** (`. `, `! `, `? `, `.\n`, `!\n`, `?\n`) — keeps sentences intact
3. **Word boundary** (space) — avoids splitting mid-word
4. **Hard cut** at `max_chunk_size` — only when no boundary exists in the window

The backward search (last occurrence within the window) ensures each chunk is as
large as possible while still ending at a clean boundary. Atomic constructs (code
blocks, tables, lists) are never split, even if they exceed `max_chunk_size`.

### Streaming

#### `stream_markdown_file(file_path, max_chunk_size=1000, heading_inclusion=HeadingInclusion.BOTH, chunk_format=ChunkFormat.MARKDOWN)`

Stream chunks from a markdown file.

**Parameters:**
- `file_path`: Path to markdown file
- `max_chunk_size`: Maximum size of chunk text
- `heading_inclusion`: How to include headings
- `chunk_format`: Output format

**Yields:** Chunk objects

**Example:**
```python
from dataknobs_xization import stream_markdown_file

for chunk in stream_markdown_file("document.md", max_chunk_size=500):
    print(chunk.to_dict())
```

#### `stream_markdown_string(content, ...)`

Stream chunks from a markdown string.

```python
from dataknobs_xization import stream_markdown_string

markdown = "# Title\nBody text."
for chunk in stream_markdown_string(markdown):
    print(chunk.text)
```

#### `StreamingMarkdownProcessor` and `AdaptiveStreamingProcessor`

Classes for streaming processing with memory management.

```python
from dataknobs_xization import AdaptiveStreamingProcessor

processor = AdaptiveStreamingProcessor(
    max_chunk_size=1000,
    memory_limit_nodes=10000,
    adaptive_threshold=0.8
)

chunks = list(processor.process_file("large_document.md"))
```

## Command-Line Interface

The package includes a CLI for testing and demonstration.

### Commands

#### `info` - Show document information

```bash
uv run python packages/xization/scripts/md_cli.py info document.md
```

Output:
```
Document Information
==================================================
Total nodes: 49
Heading nodes: 24
Body text nodes: 24
Tree depth: 2

Heading levels:
  Level 1: 3
  Level 2: 7
  ...
```

#### `chunk` - Chunk a markdown document

```bash
# Basic chunking
uv run python packages/xization/scripts/md_cli.py chunk document.md

# With custom parameters
uv run python packages/xization/scripts/md_cli.py chunk document.md \
  --max-size 500 \
  --headings both \
  --show-metadata

# Output as JSON
uv run python packages/xization/scripts/md_cli.py chunk document.md \
  --output-format json \
  --output chunks.json
```

#### `parse` - Parse and show tree structure

```bash
uv run python packages/xization/scripts/md_cli.py parse document.md --show-tree
```

### CLI Options

**chunk command:**
- `--max-size`: Maximum chunk size in characters (default: 1000)
- `--headings`: How to include headings: both, text, metadata, none (default: both)
- `--output-format`: Output format: markdown, plain, json (default: markdown)
- `--output`: Output file (default: stdout)
- `--show-metadata`: Show chunk metadata
- `--separator`: Separator between chunks (default: \n---\n)

## Configuration Options

### HeadingInclusion

Controls how headings are included in chunks:

- `HeadingInclusion.BOTH`: Include in both text and metadata (default)
- `HeadingInclusion.IN_TEXT`: Include only in chunk text
- `HeadingInclusion.IN_METADATA`: Include only in metadata
- `HeadingInclusion.NONE`: Don't include headings

### ChunkFormat

Controls output format:

- `ChunkFormat.MARKDOWN`: Include headings as markdown (e.g., `# Title`)
- `ChunkFormat.PLAIN`: Plain text without markdown formatting
- `ChunkFormat.DICT`: Return as dictionary

## Use Cases

### RAG Vector Store Loading

```python
from dataknobs_xization import stream_markdown_file
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

for chunk in stream_markdown_file("documentation.md", max_chunk_size=500):
    collection.add(
        documents=[chunk.text],
        metadatas=[chunk.metadata.to_dict()],
        ids=[f"chunk_{chunk.metadata.chunk_index}"]
    )
```

### Contextual Retrieval

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree, HeadingInclusion

tree = parse_markdown(document_text)

# Include headings in metadata for context, but not in text
chunks = chunk_markdown_tree(
    tree,
    heading_inclusion=HeadingInclusion.IN_METADATA,
    max_chunk_size=500
)

for chunk in chunks:
    # Heading context available in metadata for filtering/ranking
    context_path = chunk.metadata.get_heading_path()
    # Body text is clean without heading markup
    clean_text = chunk.text
```

### Document Analysis

```python
from dataknobs_xization import parse_markdown

tree = parse_markdown(document)

# Find all sections about a specific topic
relevant_sections = tree.find_nodes(
    lambda n: n.data.is_heading() and "security" in n.data.text.lower()
)

# Get body text under each relevant section
for section in relevant_sections:
    body_nodes = section.find_nodes(lambda n: n.data.is_body())
    for body in body_nodes:
        print(body.data.text)
```

## Memory Management

For large documents, use the `AdaptiveStreamingProcessor`:

```python
from dataknobs_xization import AdaptiveStreamingProcessor

processor = AdaptiveStreamingProcessor(
    max_chunk_size=1000,
    memory_limit_nodes=10000,  # Maximum nodes to keep in memory
    adaptive_threshold=0.8      # Trigger chunking at 80% of limit
)

# Process very large file
for chunk in processor.process_file("massive_documentation.md"):
    # Chunks are yielded as they're ready
    # Memory usage remains bounded
    process_chunk(chunk)
```

## Testing

Run the test suite:

```bash
cd packages/xization
uv run pytest tests/test_md_parser.py tests/test_md_chunker.py tests/test_md_streaming.py -v
```

## Examples

See `example_document.md` for a sample markdown file demonstrating various heading levels and structures.

## Pluggable Chunker Abstraction

The `dataknobs_xization.chunking` module provides a pluggable abstraction over
chunking implementations.  Instead of calling `chunk_markdown_tree()` directly,
consuming code (RAGKnowledgeBase, DirectoryProcessor, custom pipelines) can use
`create_chunker()` to resolve the implementation from configuration.

### Quick Start

```python
from dataknobs_xization.chunking import create_chunker

# Default — wraps MarkdownChunker with content-level interface
chunker = create_chunker({"max_chunk_size": 800})
chunks = chunker.chunk("# Title\nBody text.")

# Explicit selection
chunker = create_chunker({"chunker": "markdown_tree", "max_chunk_size": 800})

# Custom implementation via dotted import path
chunker = create_chunker({"chunker": "my_project.chunkers.RFCChunker"})
```

### Writing a Custom Chunker

Subclass `Chunker` and implement the `chunk()` method:

```python
from dataknobs_xization.chunking import Chunker, DocumentInfo, register_chunker
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata

class PlaintextChunker(Chunker):
    """Simple chunker that splits on double newlines."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk(self, content: str, document_info: DocumentInfo | None = None) -> list[Chunk]:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            Chunk(
                text=para,
                metadata=ChunkMetadata(chunk_index=i, chunk_size=len(para), content_length=len(para)),
            )
            for i, para in enumerate(paragraphs)
        ]

    @classmethod
    def from_config(cls, config: dict) -> "PlaintextChunker":
        return cls(max_chunk_size=config.get("max_chunk_size", 1000))

# Register for use in YAML configs
register_chunker("plaintext", PlaintextChunker)
```

### Using in Bot YAML Configuration

```yaml
knowledge_base:
  chunking:
    chunker: markdown_tree    # default, can be omitted
    max_chunk_size: 800

# Or with a custom chunker:
knowledge_base:
  chunking:
    chunker: my_project.chunkers.RFCChunker
    max_chunk_size: 1200
```

### Key Classes

| Class | Module | Role |
|-------|--------|------|
| `Chunker` | `chunking.base` | ABC — subclass to create custom chunkers |
| `DocumentInfo` | `chunking.base` | Document metadata passed to `chunk()` |
| `MarkdownTreeChunker` | `chunking.markdown` | Default — wraps `MarkdownChunker` |
| `chunker_registry` | `chunking.registry` | `PluginRegistry[Chunker]` singleton |
| `create_chunker()` | `chunking.registry` | Config-driven factory function |
| `register_chunker()` | `chunking.registry` | Register custom implementations |

The registry uses `PluginRegistry` from `dataknobs-common` with `config_key="chunker"`,
`config_key_default="markdown_tree"`, and `from_config` classmethod detection.

## Integration with Other Dataknobs Packages

The markdown chunking utilities integrate seamlessly with other dataknobs packages:

- Uses `Tree` from `dataknobs-structures` for representing document hierarchy
- `PluginRegistry` from `dataknobs-common` for chunker registration and resolution
- `RAGKnowledgeBase` from `dataknobs-bots` resolves chunkers from config automatically
- `DirectoryProcessor` from `dataknobs-xization` uses the same chunker abstraction
- Compatible with `dataknobs-data` for data pipeline integration

## Contributing

Contributions are welcome! Please ensure:

- All tests pass
- New features include tests
- Code follows the existing style
- Documentation is updated

## License

Part of the dataknobs project. See main repository for license information.
