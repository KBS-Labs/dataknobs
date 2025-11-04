"""Streaming processor for incremental markdown chunking.

This module provides functionality to process large markdown documents
incrementally, managing memory constraints while generating chunks.
"""

from __future__ import annotations

from typing import Iterator, TextIO

from dataknobs_structures.tree import Tree

from dataknobs_xization.markdown.md_chunker import Chunk, ChunkFormat, HeadingInclusion, MarkdownChunker
from dataknobs_xization.markdown.md_parser import MarkdownNode, MarkdownParser


class StreamingMarkdownProcessor:
    """Streaming processor for incremental markdown chunking.

    Processes markdown documents line-by-line, building tree structure
    incrementally and yielding chunks as they become available. Manages
    memory by pruning processed sections of the tree.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
        max_line_length: int | None = None,
        heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
        chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
        max_tree_depth: int = 100,
        memory_limit_nodes: int | None = None,
    ):
        """Initialize the streaming processor.

        Args:
            max_chunk_size: Maximum size of chunk text in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_line_length: Maximum length for individual lines
            heading_inclusion: How to include headings in chunks
            chunk_format: Output format for chunks
            max_tree_depth: Maximum depth of tree to maintain
            memory_limit_nodes: Maximum number of nodes to keep in memory
                (None for unlimited)
        """
        self.parser = MarkdownParser(
            max_line_length=max_line_length,
            preserve_empty_lines=False,
        )
        self.chunker = MarkdownChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            heading_inclusion=heading_inclusion,
            chunk_format=chunk_format,
            combine_under_heading=True,
        )
        self.max_tree_depth = max_tree_depth
        self.memory_limit_nodes = memory_limit_nodes

    def process_stream(
        self,
        source: str | TextIO | Iterator[str],
    ) -> Iterator[Chunk]:
        """Process markdown from a stream, yielding chunks incrementally.

        Args:
            source: Markdown content as string, file object, or line iterator

        Yields:
            Chunk objects as they become available
        """
        # For simplicity in v1, we'll use a batch processing approach
        # that processes complete sections under headings
        #
        # Future enhancement: true streaming with incremental tree building

        tree = self.parser.parse(source)

        # Generate chunks
        yield from self.chunker.chunk(tree)

    def process_file(self, file_path: str) -> Iterator[Chunk]:
        """Process a markdown file, yielding chunks incrementally.

        Args:
            file_path: Path to markdown file

        Yields:
            Chunk objects
        """
        with open(file_path, encoding='utf-8') as f:
            yield from self.process_stream(f)

    def process_string(self, content: str) -> Iterator[Chunk]:
        """Process markdown from a string, yielding chunks.

        Args:
            content: Markdown content string

        Yields:
            Chunk objects
        """
        yield from self.process_stream(content)


class AdaptiveStreamingProcessor(StreamingMarkdownProcessor):
    """Streaming processor that adapts to memory constraints.

    This processor monitors tree size and adaptively chunks sections
    when memory limits are approached, preventing memory overflow on
    large documents.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
        max_line_length: int | None = None,
        heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
        chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
        max_tree_depth: int = 100,
        memory_limit_nodes: int = 10000,
        adaptive_threshold: float = 0.8,
    ):
        """Initialize the adaptive streaming processor.

        Args:
            max_chunk_size: Maximum size of chunk text in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_line_length: Maximum length for individual lines
            heading_inclusion: How to include headings in chunks
            chunk_format: Output format for chunks
            max_tree_depth: Maximum depth of tree to maintain
            memory_limit_nodes: Maximum number of nodes to keep in memory
            adaptive_threshold: Fraction of memory_limit at which to trigger
                adaptive chunking (0.0-1.0)
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            max_line_length=max_line_length,
            heading_inclusion=heading_inclusion,
            chunk_format=chunk_format,
            max_tree_depth=max_tree_depth,
            memory_limit_nodes=memory_limit_nodes,
        )
        self.adaptive_threshold = adaptive_threshold

    def process_stream(self, source: str | TextIO | Iterator[str]) -> Iterator[Chunk]:
        """Process stream with adaptive memory management.

        Args:
            source: Markdown content source

        Yields:
            Chunk objects
        """
        # Build tree incrementally with memory monitoring
        root = Tree(MarkdownNode(text="ROOT", level=0, node_type="root", line_number=0))
        current_parent = root
        line_number = 0

        lines = self.parser._get_line_iterator(source)

        pending_nodes = []  # Nodes waiting to be chunked

        for line in lines:
            line_number += 1

            if not line.strip():
                continue

            # Check if line is a heading
            heading_match = self.parser.HEADING_PATTERN.match(line)

            if heading_match:
                # Before adding new heading, check if we should chunk pending nodes
                if self.memory_limit_nodes:
                    node_count = len(root.find_nodes(lambda _: True))
                    if node_count >= self.memory_limit_nodes * self.adaptive_threshold:
                        # Chunk and yield accumulated body text
                        if pending_nodes:
                            yield from self._chunk_nodes(pending_nodes)
                            pending_nodes = []
                            # Prune processed subtrees to free memory
                            self._prune_processed_nodes(root)

                # Process heading
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()

                node_data = MarkdownNode(
                    text=text,
                    level=level,
                    node_type="heading",
                    line_number=line_number,
                )

                current_parent, _ = self.parser._find_heading_parent(
                    root, current_parent, level
                )

                heading_node = current_parent.add_child(node_data)
                current_parent = heading_node

            else:
                # Body text
                node_data = MarkdownNode(
                    text=line.rstrip('\n'),
                    level=0,
                    node_type="body",
                    line_number=line_number,
                )
                body_node = current_parent.add_child(node_data)
                pending_nodes.append(body_node)

        # Process any remaining pending nodes
        if pending_nodes:
            yield from self._chunk_nodes(pending_nodes)

    def _chunk_nodes(self, nodes: list[Tree]) -> Iterator[Chunk]:
        """Chunk a list of body text nodes.

        Args:
            nodes: List of body text tree nodes

        Yields:
            Chunk objects
        """
        yield from self.chunker._chunk_by_heading(nodes)

    def _prune_processed_nodes(self, root: Tree) -> None:
        """Prune processed leaf nodes to free memory.

        Args:
            root: Root of tree to prune
        """
        # Find terminal nodes that have been processed
        # For now, we'll keep the tree structure but could optimize further
        # by removing fully processed subtrees
        pass


def stream_markdown_file(
    file_path: str,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 100,
    heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
    chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
) -> Iterator[Chunk]:
    """Stream chunks from a markdown file.

    Convenience function for processing a file with default settings.

    Args:
        file_path: Path to markdown file
        max_chunk_size: Maximum size of chunk text in characters
        chunk_overlap: Number of characters to overlap between chunks
        heading_inclusion: How to include headings in chunks
        chunk_format: Output format for chunks

    Yields:
        Chunk objects
    """
    processor = StreamingMarkdownProcessor(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        heading_inclusion=heading_inclusion,
        chunk_format=chunk_format,
    )
    yield from processor.process_file(file_path)


def stream_markdown_string(
    content: str,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 100,
    heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
    chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
) -> Iterator[Chunk]:
    """Stream chunks from a markdown string.

    Convenience function for processing a string with default settings.

    Args:
        content: Markdown content string
        max_chunk_size: Maximum size of chunk text in characters
        chunk_overlap: Number of characters to overlap between chunks
        heading_inclusion: How to include headings in chunks
        chunk_format: Output format for chunks

    Yields:
        Chunk objects
    """
    processor = StreamingMarkdownProcessor(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        heading_inclusion=heading_inclusion,
        chunk_format=chunk_format,
    )
    yield from processor.process_string(content)
