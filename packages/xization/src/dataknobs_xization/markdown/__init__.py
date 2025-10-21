"""Markdown chunking utilities for RAG applications.

This module provides comprehensive utilities for parsing and chunking markdown
documents while preserving semantic structure and heading hierarchy.
"""

from dataknobs_xization.markdown.md_chunker import (
    Chunk,
    ChunkFormat,
    ChunkMetadata,
    HeadingInclusion,
    MarkdownChunker,
    chunk_markdown_tree,
)
from dataknobs_xization.markdown.md_parser import (
    MarkdownNode,
    MarkdownParser,
    parse_markdown,
)
from dataknobs_xization.markdown.md_streaming import (
    AdaptiveStreamingProcessor,
    StreamingMarkdownProcessor,
    stream_markdown_file,
    stream_markdown_string,
)

__all__ = [
    # Parser
    "MarkdownNode",
    "MarkdownParser",
    "parse_markdown",
    # Chunker
    "Chunk",
    "ChunkFormat",
    "ChunkMetadata",
    "HeadingInclusion",
    "MarkdownChunker",
    "chunk_markdown_tree",
    # Streaming
    "AdaptiveStreamingProcessor",
    "StreamingMarkdownProcessor",
    "stream_markdown_file",
    "stream_markdown_string",
]
