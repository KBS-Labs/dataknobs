"""Text normalization and tokenization tools."""

from dataknobs_xization import (
    annotations,
    authorities,
    lexicon,
    markdown,
    masking_tokenizer,
    normalize,
)
from dataknobs_xization.markdown import (
    AdaptiveStreamingProcessor,
    Chunk,
    ChunkFormat,
    ChunkMetadata,
    ChunkQualityConfig,
    ChunkQualityFilter,
    EnrichedChunkData,
    HeadingInclusion,
    MarkdownChunker,
    MarkdownNode,
    MarkdownParser,
    StreamingMarkdownProcessor,
    build_enriched_text,
    chunk_markdown_tree,
    format_heading_display,
    get_dynamic_heading_display,
    is_multiword,
    parse_markdown,
    stream_markdown_file,
    stream_markdown_string,
)
from dataknobs_xization.masking_tokenizer import CharacterFeatures, TextFeatures

__version__ = "1.0.0"

__all__ = [
    # Existing exports
    "CharacterFeatures",
    "TextFeatures",
    "annotations",
    "authorities",
    "lexicon",
    "masking_tokenizer",
    "normalize",
    # Markdown module
    "markdown",
    # Markdown chunking classes and functions
    "AdaptiveStreamingProcessor",
    "Chunk",
    "ChunkFormat",
    "ChunkMetadata",
    "ChunkQualityConfig",
    "ChunkQualityFilter",
    "EnrichedChunkData",
    "HeadingInclusion",
    "MarkdownChunker",
    "MarkdownNode",
    "MarkdownParser",
    "StreamingMarkdownProcessor",
    "build_enriched_text",
    "chunk_markdown_tree",
    "format_heading_display",
    "get_dynamic_heading_display",
    "is_multiword",
    "parse_markdown",
    "stream_markdown_file",
    "stream_markdown_string",
]
