"""Markdown chunker for generating RAG-optimized chunks from tree structures.

This module provides functionality to traverse markdown tree structures and
generate chunks suitable for RAG (Retrieval-Augmented Generation) applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from dataknobs_structures.tree import Tree

from dataknobs_xization.markdown.md_parser import MarkdownNode
from dataknobs_xization.markdown.enrichment import build_enriched_text
from dataknobs_xization.markdown.filters import ChunkQualityConfig, ChunkQualityFilter


class ChunkFormat(Enum):
    """Output format for chunk text."""

    MARKDOWN = "markdown"  # Include headings as markdown
    PLAIN = "plain"  # Plain text without markdown formatting
    DICT = "dict"  # Return as dictionary


class HeadingInclusion(Enum):
    """Strategy for including headings in chunks."""

    IN_TEXT = "in_text"  # Include headings in chunk text
    IN_METADATA = "in_metadata"  # Include headings only in metadata
    BOTH = "both"  # Include in both text and metadata
    NONE = "none"  # Don't include headings


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk.

    Attributes:
        headings: List of heading texts from root to chunk
        heading_levels: List of heading levels corresponding to headings
        line_number: Starting line number in source document
        chunk_index: Index of this chunk in the sequence
        chunk_size: Size of chunk text in characters
        content_length: Size of content without headings (for quality decisions)
        heading_display: Formatted heading path for display
        embedding_text: Heading-enriched text for embedding (optional)
        custom: Additional custom metadata
    """

    headings: list[str] = field(default_factory=list)
    heading_levels: list[int] = field(default_factory=list)
    line_number: int = 0
    chunk_index: int = 0
    chunk_size: int = 0
    content_length: int = 0
    heading_display: str = ""
    embedding_text: str = ""
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        result = {
            "headings": self.headings,
            "heading_levels": self.heading_levels,
            "line_number": self.line_number,
            "chunk_index": self.chunk_index,
            "chunk_size": self.chunk_size,
            "content_length": self.content_length,
            "heading_display": self.heading_display,
            **self.custom,
        }
        # Only include embedding_text if it was generated
        if self.embedding_text:
            result["embedding_text"] = self.embedding_text
        return result

    def get_heading_path(self, separator: str = " > ") -> str:
        """Get heading hierarchy as a single string.

        Args:
            separator: String to use between headings

        Returns:
            Formatted heading path
        """
        return separator.join(self.headings)


@dataclass
class Chunk:
    """A chunk of text with associated metadata.

    Attributes:
        text: The chunk text content
        metadata: Metadata for this chunk
    """

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    def to_markdown(self, include_headings: bool = True) -> str:
        """Convert chunk to markdown format.

        Args:
            include_headings: Whether to include heading hierarchy

        Returns:
            Markdown-formatted string
        """
        if not include_headings or not self.metadata.headings:
            return self.text

        # Build heading hierarchy
        lines = []
        for heading, level in zip(
            self.metadata.headings, self.metadata.heading_levels
        ):
            lines.append(f"{'#' * level} {heading}")

        # Add body text
        if self.text:
            lines.append("")
            lines.append(self.text)

        return "\n".join(lines)


class MarkdownChunker:
    """Chunker for generating chunks from markdown tree structures.

    Traverses a Tree built from markdown and generates chunks with
    configurable size, heading inclusion, and output format.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
        heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
        chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
        combine_under_heading: bool = True,
        quality_filter: ChunkQualityConfig | None = None,
        generate_embeddings: bool = False,
    ):
        """Initialize the markdown chunker.

        Args:
            max_chunk_size: Maximum size of chunk text in characters
            chunk_overlap: Number of characters to overlap between chunks
            heading_inclusion: How to include headings in chunks
            chunk_format: Output format for chunks
            combine_under_heading: Whether to combine body text under same heading
            quality_filter: Optional config for filtering low-quality chunks
            generate_embeddings: Whether to generate heading-enriched embedding text
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_inclusion = heading_inclusion
        self.chunk_format = chunk_format
        self.combine_under_heading = combine_under_heading
        self.generate_embeddings = generate_embeddings
        self._chunk_index = 0

        # Initialize quality filter if config provided
        self._quality_filter = None
        if quality_filter is not None:
            self._quality_filter = ChunkQualityFilter(quality_filter)

    def chunk(self, tree: Tree) -> Iterator[Chunk]:
        """Generate chunks from a markdown tree.

        Args:
            tree: Tree structure built from markdown

        Yields:
            Chunk objects with text and metadata
        """
        self._chunk_index = 0

        # Get all terminal (leaf) nodes - not headings or root
        terminal_nodes = tree.collect_terminal_nodes(
            accept_node_fn=lambda n: (
                isinstance(n.data, MarkdownNode)
                and not n.data.is_heading()
                and n.data.node_type != "root"
            )
        )

        if self.combine_under_heading:
            # Group terminal nodes by their parent heading
            chunk_iter = self._chunk_by_heading(terminal_nodes)
        else:
            # Process each terminal node individually
            chunk_iter = self._chunk_individually(terminal_nodes)

        # Apply quality filter if configured
        for chunk in chunk_iter:
            if self._quality_filter is None or self._quality_filter.is_valid(chunk):
                yield chunk

    def _chunk_by_heading(self, terminal_nodes: list[Tree]) -> Iterator[Chunk]:
        """Group nodes under same heading and chunk them.

        Args:
            terminal_nodes: List of terminal tree nodes

        Yields:
            Chunk objects
        """
        # Group nodes by their immediate parent
        parent_groups: dict[Tree, list[Tree]] = {}
        for node in terminal_nodes:
            parent = node.parent
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append(node)

        # Process each group
        for parent, nodes in parent_groups.items():
            # Get heading path for this group
            headings, levels = self._get_heading_path(parent)

            # Separate atomic constructs from regular body text
            atomic_nodes = [n for n in nodes if n.data.is_atomic()]
            body_nodes = [n for n in nodes if not n.data.is_atomic()]

            # Process body text nodes (can be combined and split)
            if body_nodes:
                combined_text = "\n".join(
                    node.data.text for node in body_nodes if node.data.text.strip()
                )

                if combined_text.strip():
                    for chunk_text in self._split_text(combined_text):
                        yield self._create_chunk(
                            text=chunk_text,
                            headings=headings,
                            heading_levels=levels,
                            line_number=body_nodes[0].data.line_number if body_nodes else 0,
                        )

            # Process atomic constructs (keep as complete units)
            for atomic_node in atomic_nodes:
                # Don't split atomic constructs, even if they exceed max_chunk_size
                yield self._create_chunk(
                    text=atomic_node.data.text,
                    headings=headings,
                    heading_levels=levels,
                    line_number=atomic_node.data.line_number,
                    metadata=atomic_node.data.metadata,
                    node_type=atomic_node.data.node_type,
                )

    def _chunk_individually(self, terminal_nodes: list[Tree]) -> Iterator[Chunk]:
        """Process each terminal node individually.

        Args:
            terminal_nodes: List of terminal tree nodes

        Yields:
            Chunk objects
        """
        for node in terminal_nodes:
            if not node.data.text.strip():
                continue

            headings, levels = self._get_heading_path(node.parent)

            # Atomic constructs are kept whole
            if node.data.is_atomic():
                yield self._create_chunk(
                    text=node.data.text,
                    headings=headings,
                    heading_levels=levels,
                    line_number=node.data.line_number,
                    metadata=node.data.metadata,
                    node_type=node.data.node_type,
                )
            else:
                # Regular body text can be split
                for chunk_text in self._split_text(node.data.text):
                    yield self._create_chunk(
                        text=chunk_text,
                        headings=headings,
                        heading_levels=levels,
                        line_number=node.data.line_number,
                    )

    def _get_heading_path(self, node: Tree | None) -> tuple[list[str], list[int]]:
        """Get the heading path from root to this node.

        Args:
            node: Tree node to get path for

        Returns:
            Tuple of (heading_texts, heading_levels)
        """
        headings = []
        levels = []

        current = node
        while current is not None:
            if isinstance(current.data, MarkdownNode):
                if current.data.is_heading():
                    headings.insert(0, current.data.text)
                    levels.insert(0, current.data.level)
            current = current.parent

        return headings, levels

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting max_chunk_size.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # If not at the end, try to break at a good boundary
            if end < len(text):
                # Try to break at paragraph boundary (double newline)
                break_pos = text.rfind("\n\n", start, end)
                if break_pos > start:
                    end = break_pos + 2
                else:
                    # Try to break at sentence boundary
                    for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                        break_pos = text.rfind(punct, start, end)
                        if break_pos > start:
                            end = break_pos + len(punct)
                            break
                    else:
                        # Try to break at word boundary
                        break_pos = text.rfind(" ", start, end)
                        if break_pos > start:
                            end = break_pos + 1

            chunks.append(text[start:end].strip())

            # Move start position, accounting for overlap
            start = max(start + 1, end - self.chunk_overlap)

        return [c for c in chunks if c]  # Filter out empty chunks

    def _create_chunk(
        self,
        text: str,
        headings: list[str],
        heading_levels: list[int],
        line_number: int,
        metadata: dict[str, Any] | None = None,
        node_type: str = "body",
    ) -> Chunk:
        """Create a chunk with appropriate format and metadata.

        Args:
            text: Body text for chunk
            headings: List of heading texts
            heading_levels: List of heading levels
            line_number: Source line number
            metadata: Optional metadata from the source node
            node_type: Type of node ('body', 'code', 'list', 'table', etc.)

        Returns:
            Formatted Chunk object
        """
        # Store content length before adding headings
        content_length = len(text)

        # Build chunk text based on heading inclusion setting
        chunk_text = text

        if self.heading_inclusion in (HeadingInclusion.IN_TEXT, HeadingInclusion.BOTH):
            # Prepend headings to text
            heading_lines = []
            for heading, level in zip(headings, heading_levels):
                if self.chunk_format == ChunkFormat.MARKDOWN:
                    heading_lines.append(f"{'#' * level} {heading}")
                else:
                    heading_lines.append(heading)

            if heading_lines:
                chunk_text = "\n".join(heading_lines) + "\n\n" + text

        # Create custom metadata dict with node type and additional metadata
        custom_metadata = {"node_type": node_type}
        if metadata:
            custom_metadata.update(metadata)

        # Generate heading display string
        heading_display = " > ".join(headings) if headings else ""

        # Generate embedding text if enabled
        embedding_text = ""
        if self.generate_embeddings:
            embedding_text = build_enriched_text(headings, text)

        # Determine which headings to include in metadata
        include_headings = self.heading_inclusion in (
            HeadingInclusion.IN_METADATA,
            HeadingInclusion.BOTH,
        )

        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            headings=headings if include_headings else [],
            heading_levels=heading_levels if include_headings else [],
            line_number=line_number,
            chunk_index=self._chunk_index,
            chunk_size=len(chunk_text),
            content_length=content_length,
            heading_display=heading_display,
            embedding_text=embedding_text,
            custom=custom_metadata,
        )

        self._chunk_index += 1

        return Chunk(text=chunk_text, metadata=chunk_metadata)


def chunk_markdown_tree(
    tree: Tree,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 100,
    heading_inclusion: HeadingInclusion = HeadingInclusion.BOTH,
    chunk_format: ChunkFormat = ChunkFormat.MARKDOWN,
    combine_under_heading: bool = True,
    quality_filter: ChunkQualityConfig | None = None,
    generate_embeddings: bool = False,
) -> list[Chunk]:
    """Generate chunks from a markdown tree.

    Convenience function for creating and using a MarkdownChunker.

    Args:
        tree: Tree structure built from markdown
        max_chunk_size: Maximum size of chunk text in characters
        chunk_overlap: Number of characters to overlap between chunks
        heading_inclusion: How to include headings in chunks
        chunk_format: Output format for chunks
        combine_under_heading: Whether to combine body text under same heading
        quality_filter: Optional config for filtering low-quality chunks
        generate_embeddings: Whether to generate heading-enriched embedding text

    Returns:
        List of Chunk objects
    """
    chunker = MarkdownChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        heading_inclusion=heading_inclusion,
        chunk_format=chunk_format,
        combine_under_heading=combine_under_heading,
        quality_filter=quality_filter,
        generate_embeddings=generate_embeddings,
    )
    return list(chunker.chunk(tree))
