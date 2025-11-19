# Heading Enrichment for RAG Embeddings

This module provides utilities to enrich chunk content with heading context for improved semantic search, while keeping headings out of the displayed content.

## Overview

When embedding chunks for RAG, the embedding model only sees the chunk text. A chunk like "See the example below" provides poor semantic signal without context. But if we know it's under `Patterns > Chain-of-Thought > Example`, we can enrich the embedding text:

```
Chain-of-Thought: Example: See the example below
```

This gives the embedding model semantic context about what the "example" refers to, improving retrieval accuracy.

## Quick Start

```python
from dataknobs_xization import build_enriched_text

# Build text for embedding
embedding_text = build_enriched_text(
    heading_path=["Patterns", "Chain-of-Thought", "Example"],
    content="See the example below..."
)
# Returns: "Chain-of-Thought: Example: See the example below..."
```

## Key Concepts

### The Problem

Without enrichment:
```python
# User query: "chain of thought example"
# Chunk text: "See the example below"
# Similarity: LOW (no mention of chain-of-thought)
```

With enrichment:
```python
# User query: "chain of thought example"
# Embedding text: "Chain-of-Thought: Example: See the example below"
# Similarity: HIGH (chain-of-thought in embedding)
```

### The Solution

The `build_enriched_text` function:
1. Walks backward from the deepest heading
2. Includes headings until the first multi-word heading
3. Prepends these headings to the content

This approach:
- Provides semantic context
- Avoids over-weighting deep single-word labels
- Keeps displayed content clean

## Core Functions

### build_enriched_text

```python
from dataknobs_xization import build_enriched_text

# Example 1: Single-word deepest heading
text = build_enriched_text(
    ["Patterns", "Chain-of-Thought", "Example"],
    "code here"
)
# Returns: "Chain-of-Thought: Example: code here"

# Example 2: Multi-word deepest heading (stops immediately)
text = build_enriched_text(
    ["API Reference", "Authentication", "OAuth 2.0"],
    "Configure OAuth..."
)
# Returns: "OAuth 2.0: Configure OAuth..."

# Example 3: All multi-word headings
text = build_enriched_text(
    ["Getting Started", "Quick Start"],
    "Install the package..."
)
# Returns: "Quick Start: Install the package..."

# Example 4: Single heading
text = build_enriched_text(
    ["Setup"],
    "Install steps..."
)
# Returns: "Setup: Install steps..."

# Example 5: No headings
text = build_enriched_text(
    [],
    "Standalone content"
)
# Returns: "Standalone content"
```

### enrich_chunk

Convenience function that creates fully enriched chunk data:

```python
from dataknobs_xization import enrich_chunk

enriched = enrich_chunk(
    content="Configure OAuth settings...",
    headings=["API", "Authentication", "OAuth 2.0"],
    heading_levels=[1, 2, 3]
)

print(enriched.content)        # "Configure OAuth settings..."
print(enriched.embedding_text) # "OAuth 2.0: Configure OAuth settings..."
print(enriched.heading_path)   # ["API", "Authentication", "OAuth 2.0"]
print(enriched.heading_display)# "API > Authentication > OAuth 2.0"
print(enriched.content_length) # 27
```

### EnrichedChunkData

```python
from dataknobs_xization import EnrichedChunkData

@dataclass
class EnrichedChunkData:
    content: str          # Clean content text (for display)
    embedding_text: str   # Enriched text (for embedding)
    heading_path: list[str]
    heading_display: str  # "A > B > C" format
    content_length: int
```

## Display Utilities

### format_heading_display

Format heading path for display:

```python
from dataknobs_xization import format_heading_display

display = format_heading_display(
    ["Chapter", "Section", "Subsection"],
    separator=" > "
)
# Returns: "Chapter > Section > Subsection"

display = format_heading_display(
    ["Chapter", "Section"],
    separator=" / "
)
# Returns: "Chapter / Section"
```

### get_dynamic_heading_display

Get heading display based on content length:

```python
from dataknobs_xization import get_dynamic_heading_display

# Small content: full path
display = get_dynamic_heading_display(
    ["A", "B", "C"],
    content_length=100,
    small_threshold=200
)
# Returns: "A > B > C"

# Medium content: last 2 headings
display = get_dynamic_heading_display(
    ["A", "B", "C"],
    content_length=500,
    small_threshold=200,
    medium_threshold=800
)
# Returns: "B > C"

# Large content: no headings
display = get_dynamic_heading_display(
    ["A", "B", "C"],
    content_length=1000,
    medium_threshold=800
)
# Returns: ""
```

### get_relevant_headings_for_display

Get heading list for display:

```python
from dataknobs_xization import get_relevant_headings_for_display

# Small chunk: full heading path
headings = get_relevant_headings_for_display(
    ["A", "B", "C", "D"],
    content_length=100,
    small_threshold=200
)
# Returns: ["A", "B", "C", "D"]

# Medium chunk: last 2 levels
headings = get_relevant_headings_for_display(
    ["A", "B", "C", "D"],
    content_length=500,
    small_threshold=200,
    medium_threshold=800
)
# Returns: ["C", "D"]

# Large chunk: no headings
headings = get_relevant_headings_for_display(
    ["A", "B", "C", "D"],
    content_length=1000
)
# Returns: []
```

## Integration with Ingestion

### Basic Pipeline

```python
from dataknobs_xization import (
    MarkdownChunker,
    ChunkQualityFilter,
    build_enriched_text,
)

async def ingest_with_enrichment(content, vector_store, embedder):
    # Chunk and filter
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content)

    quality_filter = ChunkQualityFilter()
    chunks = quality_filter.filter_chunks(chunks)

    for chunk in chunks:
        # Create enriched text for embedding
        embedding_text = build_enriched_text(
            chunk.metadata.headings,
            chunk.text
        )

        # Embed the enriched text
        vector = await embedder.embed(embedding_text)

        # Store original text for display
        await vector_store.add(
            vector=vector,
            text=chunk.text,  # Clean text
            metadata={
                "headings": chunk.metadata.headings,
                "heading_levels": chunk.metadata.heading_levels,
            }
        )
```

### Full Enrichment Pipeline

```python
from dataknobs_xization import (
    MarkdownChunker,
    ChunkQualityFilter,
    enrich_chunk,
    extract_heading_metadata,
)

async def full_enrichment_pipeline(markdown_files, vector_store, embedder):
    chunker = MarkdownChunker()
    quality_filter = ChunkQualityFilter()

    total_indexed = 0

    for file_path in markdown_files:
        with open(file_path) as f:
            content = f.read()

        # Chunk and filter
        chunks = chunker.chunk(content)
        chunks = quality_filter.filter_chunks(chunks)

        for chunk in chunks:
            # Full enrichment
            enriched = enrich_chunk(
                content=chunk.text,
                headings=chunk.metadata.headings,
                heading_levels=chunk.metadata.heading_levels,
            )

            # Embed enriched text
            vector = await embedder.embed(enriched.embedding_text)

            # Extract metadata
            heading_meta = extract_heading_metadata(
                chunk.metadata.headings,
                chunk.metadata.heading_levels
            )

            # Store with rich metadata
            await vector_store.add(
                vector=vector,
                text=enriched.content,
                metadata={
                    **heading_meta,
                    "source": file_path,
                    "content_length": enriched.content_length,
                }
            )

            total_indexed += 1

    return total_indexed
```

## Algorithm Details

### Heading Selection Algorithm

The `build_enriched_text` function uses this algorithm:

```python
def build_enriched_text(heading_path, content):
    if not heading_path:
        return content

    # Walk backwards from deepest heading
    relevant_headings = []
    for heading in reversed(heading_path):
        relevant_headings.insert(0, heading)
        # Stop after multi-word heading
        if len(heading.split()) > 1:
            break

    # Prepend to content
    if relevant_headings:
        prefix = ": ".join(relevant_headings)
        return f"{prefix}: {content}"

    return content
```

### Why This Approach?

**Problem with full path:**
```
"Getting Started > Installation > Requirements > System: Python 3.10+"
```
Too much noise, dilutes semantic signal.

**Problem with deepest only:**
```
"System: Python 3.10+"
```
"System" alone provides no context.

**Our approach:**
```
"Requirements > System: Python 3.10+"
```
"Requirements" is multi-word stop point, providing semantic context without noise.

## Helper Functions

### is_multiword

Check if a heading contains multiple words:

```python
from dataknobs_xization import is_multiword

is_multiword("Setup")           # False
is_multiword("Getting Started") # True
is_multiword("OAuth 2.0")       # True
```

### format_heading_for_display

Format headings with different styles:

```python
from dataknobs_xization import format_heading_for_display

# Path style
format_heading_for_display(
    ["A", "B", "C"],
    format_style="path"
)
# Returns: "A > B > C"

# Markdown style
format_heading_for_display(
    ["A", "B", "C"],
    heading_levels=[1, 2, 3],
    format_style="markdown"
)
# Returns: "# A\n## B\n### C"
```

## API Reference

### Core Functions

```python
def build_enriched_text(heading_path: list[str], content: str) -> str:
    """Build text for embedding with relevant heading context."""

def enrich_chunk(
    content: str,
    headings: list[str],
    heading_levels: list[int],
) -> EnrichedChunkData:
    """Create fully enriched chunk data from raw components."""

def extract_heading_metadata(
    headings: list[str],
    heading_levels: list[int],
    separator: str = " > ",
) -> dict[str, Any]:
    """Extract heading metadata for storage."""
```

### Display Functions

```python
def format_heading_display(
    heading_path: list[str],
    separator: str = " > ",
) -> str:
    """Format a heading path for display."""

def get_dynamic_heading_display(
    heading_path: list[str],
    content_length: int,
    small_threshold: int = 200,
    medium_threshold: int = 800,
) -> str:
    """Get heading display based on content length."""

def get_relevant_headings_for_display(
    heading_path: list[str],
    content_length: int,
    small_threshold: int = 200,
    medium_threshold: int = 800,
) -> list[str]:
    """Get headings to display based on content length."""

def format_heading_for_display(
    headings: list[str],
    heading_levels: list[int] | None = None,
    format_style: str = "markdown",
) -> str:
    """Format headings for display in LLM context."""

def is_multiword(heading: str) -> bool:
    """Check if a heading contains multiple words."""
```

### Data Classes

```python
@dataclass
class EnrichedChunkData:
    content: str          # Clean content text
    embedding_text: str   # Text for embedding
    heading_path: list[str]
    heading_display: str
    content_length: int
```

## Related

- [Markdown Chunking](markdown-chunking.md) - Core chunking functionality
- [Quality Filtering](quality-filtering.md) - Filter low-quality chunks
