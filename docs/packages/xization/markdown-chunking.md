# Markdown Chunking for RAG

The markdown chunking utilities provide comprehensive tools for parsing and chunking markdown documents while preserving semantic structure - essential for RAG (Retrieval-Augmented Generation) applications.

## Overview

When building RAG systems, proper document chunking is critical. These utilities:

- **Preserve heading hierarchy** - Each chunk maintains its full heading context
- **Handle special constructs** - Code blocks, tables, and lists kept intact
- **Flexible configuration** - Control chunk size, overlap, and heading inclusion
- **Stream large documents** - Process files larger than available memory
- **Rich metadata** - Track construct types, line numbers, and more

## Quick Start

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
    print(f"Headings: {chunk.metadata.get_heading_path()}")
    print(f"Text: {chunk.text}\n")
```

## Supported Markdown Constructs

### Code Blocks
- **Fenced code blocks** with language tags (` ```python `)
- **Indented code blocks** (4 spaces or tab)
- Preserved as atomic units (never split mid-function)
- Language metadata included in chunks

```python
# Code blocks stay together even if they exceed chunk size
markdown = """
# Example

```python
def long_function():
    # This entire code block
    # will be kept together
    return result
```
"""
```

### Lists
- **Unordered lists** (`-`, `*`, `+`)
- **Ordered lists** (`1.`, `2.`, `3.`)
- Multi-line list items
- Preserved as complete units

### Tables
- Standard markdown tables
- Alignment markers supported
- Never split across rows
- Row count in metadata

### Other Constructs
- **Blockquotes** - Preserved as semantic units
- **Horizontal rules** - Recognized as section boundaries

## Key Features

### Tree-Based Structure

The parser builds a Tree (from `dataknobs-structures`) where:
- Heading nodes parent their sub-headings and content
- Body text and special constructs are leaf nodes
- Full heading path retrievable by traversing to root

```python
tree = parse_markdown(markdown)

# Find all code blocks
code_nodes = tree.find_nodes(lambda n: n.data.is_code())

# Get heading context for any node
for node in code_nodes:
    path = node.get_path()  # From root to this node
```

### Heading Inclusion Options

Control how headings appear in chunks:

```python
from dataknobs_xization import HeadingInclusion

# Include in both text and metadata (default)
chunks = chunk_markdown_tree(tree, heading_inclusion=HeadingInclusion.BOTH)

# Include only in metadata (clean body text)
chunks = chunk_markdown_tree(tree, heading_inclusion=HeadingInclusion.IN_METADATA)

# Include only in text
chunks = chunk_markdown_tree(tree, heading_inclusion=HeadingInclusion.IN_TEXT)

# Exclude headings
chunks = chunk_markdown_tree(tree, heading_inclusion=HeadingInclusion.NONE)
```

### Chunk Metadata

Each chunk includes rich metadata:

```python
{
    "text": "def example(): ...",
    "metadata": {
        "headings": ["Chapter 1", "Code Examples"],
        "heading_levels": [1, 2],
        "line_number": 15,
        "chunk_index": 3,
        "chunk_size": 120,
        "node_type": "code",
        "language": "python",
        "fence_type": "```"
    }
}
```

### Streaming for Large Documents

Process files larger than available memory:

```python
from dataknobs_xization import stream_markdown_file

for chunk in stream_markdown_file("large_doc.md", max_chunk_size=1000):
    # Chunks yielded incrementally
    store_in_vector_db(chunk.text, chunk.metadata.to_dict())
```

## Use Cases

### Vector Store Loading

```python
from dataknobs_xization import stream_markdown_file
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

for chunk in stream_markdown_file("documentation.md"):
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

# Keep headings in metadata for context
chunks = chunk_markdown_tree(
    tree,
    heading_inclusion=HeadingInclusion.IN_METADATA,
    max_chunk_size=500
)

for chunk in chunks:
    # Use heading path for filtering/ranking
    context = chunk.metadata.get_heading_path()
    # Body text is clean
    text = chunk.text
```

### Special Construct Handling

```python
# Filter chunks by type
code_chunks = [
    c for c in chunks
    if c.metadata.custom.get("node_type") == "code"
]

# Handle code specially (e.g., syntax highlighting)
for chunk in code_chunks:
    language = chunk.metadata.custom.get("language", "")
    highlighted = syntax_highlight(chunk.text, language)
```

## Command-Line Interface

The package includes a CLI for testing and demonstration:

```bash
# Analyze document
uv run python -m dataknobs_xization.markdown.md_cli info document.md

# Chunk document
uv run python -m dataknobs_xization.markdown.md_cli chunk document.md \
  --max-size 500 \
  --overlap 50 \
  --show-metadata

# Output as JSON
uv run python -m dataknobs_xization.markdown.md_cli chunk document.md \
  --output-format json \
  --output chunks.json

# Parse and view tree
uv run python -m dataknobs_xization.markdown.md_cli parse document.md --show-tree
```

## API Reference

### Parser

::: dataknobs_xization.markdown.md_parser.MarkdownParser

::: dataknobs_xization.markdown.md_parser.MarkdownNode

::: dataknobs_xization.markdown.md_parser.parse_markdown

### Chunker

::: dataknobs_xization.markdown.md_chunker.MarkdownChunker

::: dataknobs_xization.markdown.md_chunker.Chunk

::: dataknobs_xization.markdown.md_chunker.ChunkMetadata

::: dataknobs_xization.markdown.md_chunker.chunk_markdown_tree

### Streaming

::: dataknobs_xization.markdown.md_streaming.StreamingMarkdownProcessor

::: dataknobs_xization.markdown.md_streaming.AdaptiveStreamingProcessor

::: dataknobs_xization.markdown.md_streaming.stream_markdown_file

::: dataknobs_xization.markdown.md_streaming.stream_markdown_string

## Examples

See the [examples directory](https://github.com/kbs-labs/dataknobs/tree/main/packages/xization/examples/markdown) for:

- Comprehensive usage examples
- Sample markdown documents
- RAG integration patterns

## Architecture

The chunking system consists of three main components:

1. **Parser (`md_parser.py`)**: Converts markdown to Tree structure
2. **Chunker (`md_chunker.py`)**: Generates chunks with configurable parameters
3. **Streaming (`md_streaming.py`)**: Handles large documents with memory management

All components use absolute import paths and are thoroughly tested (93+ passing tests).

## Best Practices

1. **Choose appropriate chunk sizes** - Balance between context and precision (500-1000 chars typical)
2. **Use heading metadata** - Essential for relevance ranking
3. **Handle constructs specially** - Code/tables have different retrieval needs
4. **Test with your data** - Chunk strategies vary by content type
5. **Monitor chunk distribution** - Ensure even coverage of document

## Related Packages

- **dataknobs-structures**: Provides the Tree data structure
- **dataknobs-utils**: Additional text processing utilities
- **dataknobs-data**: For storing chunks in various backends
