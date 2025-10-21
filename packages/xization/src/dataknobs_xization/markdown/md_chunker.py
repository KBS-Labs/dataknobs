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
        custom: Additional custom metadata
    """

    headings: list[str] = field(default_factory=list)
    heading_levels: list[int] = field(default_factory=list)
    line_number: int = 0
    chunk_index: int = 0
    chunk_size: int = 0
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "headings": self.headings,
            "heading_levels": self.heading_levels,
            "line_number": self.line_number,
            "chunk_index": self.chunk_index,
            "chunk_size": self.chunk_size,
            **self.custom,
        }

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
        for i, (heading, level) in enumerate(
            zip(self.metadata.headings, self.metadata.heading_levels)
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
    ):
        """Initialize the markdown chunker.

        Args:
            max_chunk_size: Maximum size of chunk text in characters
            chunk_overlap: Number of characters to overlap between chunks
            heading_inclusion: How to include headings in chunks
            chunk_format: Output format for chunks
            combine_under_heading: Whether to combine body text under same heading
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_inclusion = heading_inclusion
        self.chunk_format = chunk_format
        self.combine_under_heading = combine_under_heading
        self._chunk_index = 0

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
            yield from self._chunk_by_heading(terminal_nodes)
        else:
            # Process each terminal node individually
            yield from self._chunk_individually(terminal_nodes)

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

        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            headings=headings if self.heading_inclusion in (
                HeadingInclusion.IN_METADATA,
                HeadingInclusion.BOTH,
            ) else [],
            heading_levels=heading_levels if self.heading_inclusion in (
                HeadingInclusion.IN_METADATA,
                HeadingInclusion.BOTH,
            ) else [],
            line_number=line_number,
            chunk_index=self._chunk_index,
            chunk_size=len(chunk_text),
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

    Returns:
        List of Chunk objects
    """
    chunker = MarkdownChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        heading_inclusion=heading_inclusion,
        chunk_format=chunk_format,
        combine_under_heading=combine_under_heading,
    )
    return list(chunker.chunk(tree))
