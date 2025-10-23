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
    HeadingInclusion,
    MarkdownChunker,
    MarkdownNode,
    MarkdownParser,
    StreamingMarkdownProcessor,
    chunk_markdown_tree,
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
    "HeadingInclusion",
    "MarkdownChunker",
    "MarkdownNode",
    "MarkdownParser",
    "StreamingMarkdownProcessor",
    "chunk_markdown_tree",
    "parse_markdown",
    "stream_markdown_file",
    "stream_markdown_string",
]
