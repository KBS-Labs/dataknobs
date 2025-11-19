# Chunk Quality Filtering for RAG

This module provides filtering utilities to identify and remove low-quality chunks during ingestion that would not contribute meaningful content to RAG retrieval.

## Overview

When chunking markdown documents for RAG, not all chunks are equally valuable. Some may be:
- Too short to provide useful context
- Heading-only without body content
- Mostly punctuation or formatting
- Single-word labels

The `ChunkQualityFilter` helps ensure that only meaningful content is indexed, reducing noise and improving retrieval quality.

## Quick Start

```python
from dataknobs_xization import ChunkQualityFilter, ChunkQualityConfig

# Create filter with default settings
quality_filter = ChunkQualityFilter()

# Filter chunks during ingestion
filtered_chunks = quality_filter.filter_chunks(chunks)

# Or check individual chunks
for chunk in chunks:
    if quality_filter.is_valid(chunk):
        # Include in index
        pass
```

## Configuration

### ChunkQualityConfig

```python
from dataknobs_xization import ChunkQualityConfig

config = ChunkQualityConfig(
    min_content_chars=50,       # Minimum characters of content
    min_alphanumeric_ratio=0.3, # Minimum ratio of letters/numbers
    skip_heading_only=True,     # Filter heading-only chunks
    min_words=5,                # Minimum word count
    allow_code_blocks=True,     # Lenient filtering for code
    allow_tables=True,          # Lenient filtering for tables
)
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `min_content_chars` | 50 | Minimum characters of non-heading content |
| `min_alphanumeric_ratio` | 0.3 | Minimum ratio of alphanumeric to total chars |
| `skip_heading_only` | True | Skip chunks with only headings |
| `min_words` | 5 | Minimum word count for content |
| `allow_code_blocks` | True | Allow short code blocks that would otherwise be filtered |
| `allow_tables` | True | Allow short tables that would otherwise be filtered |

## Usage

### Basic Filtering

```python
from dataknobs_xization import (
    MarkdownChunker,
    ChunkQualityFilter,
    ChunkQualityConfig
)

# Create chunker and filter
chunker = MarkdownChunker(config)
quality_filter = ChunkQualityFilter(ChunkQualityConfig(
    min_content_chars=50,
    min_words=5
))

# Process document
chunks = chunker.chunk(markdown_content)
quality_chunks = quality_filter.filter_chunks(chunks)

print(f"Original: {len(chunks)} chunks")
print(f"After filtering: {len(quality_chunks)} chunks")
```

### Custom Configuration

```python
# Strict filtering for high-quality content
strict_config = ChunkQualityConfig(
    min_content_chars=100,
    min_alphanumeric_ratio=0.5,
    min_words=10,
    allow_code_blocks=False,  # Also filter short code blocks
)

strict_filter = ChunkQualityFilter(strict_config)

# Lenient filtering to keep more content
lenient_config = ChunkQualityConfig(
    min_content_chars=20,
    min_alphanumeric_ratio=0.2,
    min_words=3,
    skip_heading_only=False,  # Keep heading-only chunks
)

lenient_filter = ChunkQualityFilter(lenient_config)
```

### Debugging Rejections

Use `get_rejection_reason()` to understand why chunks are filtered:

```python
quality_filter = ChunkQualityFilter()

for chunk in chunks:
    if not quality_filter.is_valid(chunk):
        reason = quality_filter.get_rejection_reason(chunk)
        print(f"Rejected: {reason}")
        print(f"Content: {chunk.text[:100]}...")
```

**Example output:**
```
Rejected: Content too short (25 < 50 chars)
Rejected: Heading-only chunk (no body content)
Rejected: Word count too low (3 < 5 words)
Rejected: Alphanumeric ratio below threshold (0.3)
```

## Filter Logic

### Content Checks

1. **Heading-only check**: Removes chunks that only contain heading text without body content
2. **Length check**: Ensures minimum character count
3. **Alphanumeric ratio**: Filters chunks that are mostly punctuation/formatting
4. **Word count**: Ensures minimum meaningful words

### Special Handling

**Code blocks** (`node_type="code"`):
- Given lenient filtering when `allow_code_blocks=True`
- Only require at least one non-whitespace line
- Valuable even when short (e.g., single function definitions)

**Tables** (`node_type="table"`):
- Given lenient filtering when `allow_tables=True`
- Only require header row and at least one data row
- Compact but information-rich

## Integration with Ingestion Pipeline

### Full Pipeline Example

```python
from dataknobs_xization import (
    MarkdownChunker,
    ChunkQualityFilter,
    ChunkQualityConfig,
    build_enriched_text,
)

async def ingest_markdown(file_path: str, vector_store, embedding_model):
    """Ingest markdown file with quality filtering."""

    # 1. Read and chunk
    with open(file_path) as f:
        content = f.read()

    chunker = MarkdownChunker()
    chunks = chunker.chunk(content)

    # 2. Filter for quality
    quality_filter = ChunkQualityFilter(ChunkQualityConfig(
        min_content_chars=50,
        min_words=5
    ))
    quality_chunks = quality_filter.filter_chunks(chunks)

    print(f"Filtered {len(chunks) - len(quality_chunks)} low-quality chunks")

    # 3. Enrich and embed
    for chunk in quality_chunks:
        embedding_text = build_enriched_text(
            chunk.metadata.headings,
            chunk.text
        )
        vector = await embedding_model.embed(embedding_text)

        await vector_store.add(
            vector=vector,
            text=chunk.text,
            metadata={
                "headings": chunk.metadata.headings,
                "source": file_path
            }
        )

    return len(quality_chunks)
```

### With Rejection Logging

```python
import logging

logger = logging.getLogger(__name__)

def ingest_with_logging(chunks, quality_filter):
    """Ingest with detailed rejection logging."""

    accepted = []
    rejected = []

    for chunk in chunks:
        if quality_filter.is_valid(chunk):
            accepted.append(chunk)
        else:
            reason = quality_filter.get_rejection_reason(chunk)
            rejected.append((chunk, reason))
            logger.debug(f"Rejected chunk: {reason}")

    logger.info(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    return accepted, rejected
```

## Best Practices

### Choosing Thresholds

**For documentation/technical content:**
```python
ChunkQualityConfig(
    min_content_chars=50,
    min_words=5,
    allow_code_blocks=True,
    allow_tables=True,
)
```

**For prose/articles:**
```python
ChunkQualityConfig(
    min_content_chars=100,
    min_words=15,
    min_alphanumeric_ratio=0.5,
)
```

**For API reference:**
```python
ChunkQualityConfig(
    min_content_chars=30,
    min_words=3,
    allow_code_blocks=True,
    skip_heading_only=False,  # API headings may be meaningful
)
```

### Monitoring Filter Effectiveness

```python
def analyze_filtering(chunks, config):
    """Analyze filter effectiveness."""

    filter_obj = ChunkQualityFilter(config)

    stats = {
        "total": len(chunks),
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {}
    }

    for chunk in chunks:
        if filter_obj.is_valid(chunk):
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1
            reason = filter_obj.get_rejection_reason(chunk)
            stats["rejection_reasons"][reason] = \
                stats["rejection_reasons"].get(reason, 0) + 1

    stats["acceptance_rate"] = stats["accepted"] / stats["total"]

    return stats

# Example output:
# {
#     "total": 150,
#     "accepted": 120,
#     "rejected": 30,
#     "acceptance_rate": 0.8,
#     "rejection_reasons": {
#         "Content too short (25 < 50 chars)": 15,
#         "Heading-only chunk (no body content)": 10,
#         "Word count too low (3 < 5 words)": 5
#     }
# }
```

## API Reference

### ChunkQualityFilter

```python
class ChunkQualityFilter:
    """Filter for identifying and removing low-quality chunks."""

    def __init__(self, config: ChunkQualityConfig | None = None):
        """Initialize with optional configuration."""

    def is_valid(self, chunk: Chunk) -> bool:
        """Check if a chunk meets quality thresholds."""

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter a list of chunks, keeping only valid ones."""

    def get_rejection_reason(self, chunk: Chunk) -> str | None:
        """Get the reason a chunk would be rejected."""
```

### ChunkQualityConfig

```python
@dataclass
class ChunkQualityConfig:
    """Configuration for chunk quality filtering."""

    min_content_chars: int = 50
    min_alphanumeric_ratio: float = 0.3
    skip_heading_only: bool = True
    min_words: int = 5
    allow_code_blocks: bool = True
    allow_tables: bool = True
```

## Related

- [Markdown Chunking](markdown-chunking.md) - Core chunking functionality
- [Heading Enrichment](heading-enrichment.md) - Enrich chunks for better embeddings
