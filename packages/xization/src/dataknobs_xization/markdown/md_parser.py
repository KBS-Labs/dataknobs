"""Markdown parser for converting markdown documents into tree structures.

This module provides functionality to parse markdown text and build a Tree
structure that preserves the document's heading hierarchy and body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, TextIO

from dataknobs_structures.tree import Tree


@dataclass
class MarkdownNode:
    """Data container for markdown tree nodes.

    Represents various markdown constructs in the document.

    Attributes:
        text: The text content
        level: Heading level (1-6) or 0 for non-headings
        node_type: Type of node ('heading', 'body', 'code', 'list', 'table',
                   'blockquote', 'horizontal_rule')
        line_number: Line number in source document (for debugging)
        metadata: Additional metadata about the node (e.g., language for code blocks,
                  list type, etc.)
    """

    text: str
    level: int = 0  # 0 for body text, 1-6 for headings
    node_type: str = "body"
    line_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the node."""
        if self.node_type == "heading":
            return f"H{self.level}: {self.text}"
        elif self.node_type == "code":
            lang = self.metadata.get("language", "")
            return f"Code({lang}): {self.text[:50]}..."
        elif self.node_type == "list":
            list_type = self.metadata.get("list_type", "unordered")
            return f"List({list_type}): {len(self.text.splitlines())} items"
        elif self.node_type == "table":
            return f"Table: {len(self.text.splitlines())} rows"
        elif self.node_type == "blockquote":
            return f"Quote: {self.text[:50]}..."
        elif self.node_type == "horizontal_rule":
            return "---"
        return self.text

    def is_heading(self) -> bool:
        """Check if this node represents a heading."""
        return self.node_type == "heading"

    def is_body(self) -> bool:
        """Check if this node represents body text."""
        return self.node_type == "body"

    def is_code(self) -> bool:
        """Check if this node represents a code block."""
        return self.node_type == "code"

    def is_list(self) -> bool:
        """Check if this node represents a list."""
        return self.node_type == "list"

    def is_table(self) -> bool:
        """Check if this node represents a table."""
        return self.node_type == "table"

    def is_blockquote(self) -> bool:
        """Check if this node represents a blockquote."""
        return self.node_type == "blockquote"

    def is_horizontal_rule(self) -> bool:
        """Check if this node represents a horizontal rule."""
        return self.node_type == "horizontal_rule"

    def is_atomic(self) -> bool:
        """Check if this node should not be split during chunking."""
        return self.node_type in ("code", "table", "list", "horizontal_rule")


class MarkdownParser:
    """Parser for converting markdown text into a tree structure.

    The parser builds a Tree where:
    - Heading nodes are parents to their sub-headings and body text
    - Special constructs (code, lists, tables) are leaf nodes
    - The hierarchy mirrors the markdown heading structure
    """

    # Regex patterns for markdown constructs
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    FENCED_CODE_START = re.compile(r'^```(\w*).*$')
    FENCED_CODE_END = re.compile(r'^```\s*$')
    INDENTED_CODE = re.compile(r'^(    |\t)(.*)$')
    HORIZONTAL_RULE = re.compile(r'^(\*{3,}|-{3,}|_{3,})\s*$')
    UNORDERED_LIST = re.compile(r'^(\s*)([-*+])\s+(.+)$')
    ORDERED_LIST = re.compile(r'^(\s*)(\d+\.)\s+(.+)$')
    BLOCKQUOTE = re.compile(r'^>\s*(.*)$')
    TABLE_ROW = re.compile(r'^\|(.+)\|$')
    TABLE_SEPARATOR = re.compile(r'^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$')

    def __init__(
        self,
        max_line_length: int | None = None,
        preserve_empty_lines: bool = False,
    ):
        """Initialize the markdown parser.

        Args:
            max_line_length: Maximum length for body text lines (None for unlimited)
            preserve_empty_lines: Whether to preserve empty lines in output
        """
        self.max_line_length = max_line_length
        self.preserve_empty_lines = preserve_empty_lines
        self._current_line = 0

    def parse(self, source: str | TextIO | Iterator[str]) -> Tree:
        """Parse markdown content into a tree structure.

        Args:
            source: Markdown content as string, file object, or line iterator

        Returns:
            Tree with root node containing the document structure
        """
        # Create root node
        root = Tree(MarkdownNode(text="ROOT", level=0, node_type="root", line_number=0))

        # Convert source to list for lookahead capability
        lines = list(self._get_line_iterator(source))

        # Track current position in tree
        current_parent = root
        self._current_line = 0

        # State tracking for multi-line constructs
        i = 0
        while i < len(lines):
            self._current_line = i + 1
            line = lines[i]

            # Skip empty lines unless preserving them
            if not line.strip():
                if self.preserve_empty_lines:
                    node_data = MarkdownNode(
                        text="",
                        level=0,
                        node_type="body",
                        line_number=self._current_line,
                    )
                    current_parent.add_child(node_data)
                i += 1
                continue

            # Check for heading
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()

                node_data = MarkdownNode(
                    text=text,
                    level=level,
                    node_type="heading",
                    line_number=self._current_line,
                )

                current_parent, _ = self._find_heading_parent(
                    root, current_parent, level
                )

                heading_node = current_parent.add_child(node_data)
                current_parent = heading_node
                i += 1
                continue

            # Check for horizontal rule
            if self.HORIZONTAL_RULE.match(line):
                node_data = MarkdownNode(
                    text=line.strip(),
                    level=0,
                    node_type="horizontal_rule",
                    line_number=self._current_line,
                )
                current_parent.add_child(node_data)
                i += 1
                continue

            # Check for fenced code block
            code_match = self.FENCED_CODE_START.match(line)
            if code_match:
                code_lines, lines_consumed = self._parse_fenced_code_block(lines, i)
                if code_lines:
                    language = code_match.group(1) or ""
                    node_data = MarkdownNode(
                        text="\n".join(code_lines),
                        level=0,
                        node_type="code",
                        line_number=self._current_line,
                        metadata={"language": language, "fence_type": "```"},
                    )
                    current_parent.add_child(node_data)
                i += lines_consumed
                continue

            # Check for table
            if self.TABLE_ROW.match(line) and i + 1 < len(lines):
                if self.TABLE_SEPARATOR.match(lines[i + 1]):
                    table_lines, lines_consumed = self._parse_table(lines, i)
                    if table_lines:
                        node_data = MarkdownNode(
                            text="\n".join(table_lines),
                            level=0,
                            node_type="table",
                            line_number=self._current_line,
                            metadata={"rows": len(table_lines)},
                        )
                        current_parent.add_child(node_data)
                    i += lines_consumed
                    continue

            # Check for list
            list_match = self.UNORDERED_LIST.match(line) or self.ORDERED_LIST.match(line)
            if list_match:
                list_lines, lines_consumed, list_type = self._parse_list(lines, i)
                if list_lines:
                    node_data = MarkdownNode(
                        text="\n".join(list_lines),
                        level=0,
                        node_type="list",
                        line_number=self._current_line,
                        metadata={"list_type": list_type, "items": len(list_lines)},
                    )
                    current_parent.add_child(node_data)
                i += lines_consumed
                continue

            # Check for blockquote
            if self.BLOCKQUOTE.match(line):
                quote_lines, lines_consumed = self._parse_blockquote(lines, i)
                if quote_lines:
                    node_data = MarkdownNode(
                        text="\n".join(quote_lines),
                        level=0,
                        node_type="blockquote",
                        line_number=self._current_line,
                    )
                    current_parent.add_child(node_data)
                i += lines_consumed
                continue

            # Check for indented code block (4 spaces or tab)
            if self.INDENTED_CODE.match(line):
                code_lines, lines_consumed = self._parse_indented_code_block(lines, i)
                if code_lines:
                    node_data = MarkdownNode(
                        text="\n".join(code_lines),
                        level=0,
                        node_type="code",
                        line_number=self._current_line,
                        metadata={"language": "", "fence_type": "indent"},
                    )
                    current_parent.add_child(node_data)
                i += lines_consumed
                continue

            # Default: body text
            text = line.rstrip('\n')

            if self.max_line_length and len(text) > self.max_line_length:
                for chunk in self._split_text_intelligently(text):
                    node_data = MarkdownNode(
                        text=chunk,
                        level=0,
                        node_type="body",
                        line_number=self._current_line,
                    )
                    current_parent.add_child(node_data)
            else:
                node_data = MarkdownNode(
                    text=text,
                    level=0,
                    node_type="body",
                    line_number=self._current_line,
                )
                current_parent.add_child(node_data)

            i += 1

        return root

    def _get_line_iterator(self, source: str | TextIO | Iterator[str]) -> Iterator[str]:
        """Convert various source types to line iterator.

        Args:
            source: Input source

        Returns:
            Iterator over lines
        """
        if isinstance(source, str):
            return iter(source.splitlines())
        elif hasattr(source, 'read'):
            # File-like object
            return iter(source)
        else:
            # Already an iterator
            return source

    def _parse_fenced_code_block(
        self,
        lines: list[str],
        start_idx: int,
    ) -> tuple[list[str], int]:
        """Parse a fenced code block.

        Args:
            lines: All lines in the document
            start_idx: Index of the opening fence

        Returns:
            Tuple of (code_lines, lines_consumed)
        """
        code_lines = []
        i = start_idx + 1

        while i < len(lines):
            if self.FENCED_CODE_END.match(lines[i]):
                return code_lines, i - start_idx + 1
            code_lines.append(lines[i].rstrip('\n'))
            i += 1

        # No closing fence found, treat as code anyway
        return code_lines, i - start_idx

    def _parse_indented_code_block(
        self,
        lines: list[str],
        start_idx: int,
    ) -> tuple[list[str], int]:
        """Parse an indented code block.

        Args:
            lines: All lines in the document
            start_idx: Index of first indented line

        Returns:
            Tuple of (code_lines, lines_consumed)
        """
        code_lines = []
        i = start_idx

        while i < len(lines):
            # Empty lines within code block are allowed
            if not lines[i].strip():
                code_lines.append("")
                i += 1
                continue

            # Check if still indented
            match = self.INDENTED_CODE.match(lines[i])
            if not match:
                break

            code_lines.append(match.group(2))
            i += 1

        # Remove trailing empty lines
        while code_lines and not code_lines[-1]:
            code_lines.pop()

        return code_lines, i - start_idx

    def _parse_table(
        self,
        lines: list[str],
        start_idx: int,
    ) -> tuple[list[str], int]:
        """Parse a markdown table.

        Args:
            lines: All lines in the document
            start_idx: Index of table header row

        Returns:
            Tuple of (table_lines, lines_consumed)
        """
        table_lines = []
        i = start_idx

        while i < len(lines):
            if not self.TABLE_ROW.match(lines[i]):
                break
            table_lines.append(lines[i].rstrip('\n'))
            i += 1

        return table_lines, i - start_idx

    def _parse_list(
        self,
        lines: list[str],
        start_idx: int,
    ) -> tuple[list[str], int, str]:
        """Parse a markdown list (ordered or unordered).

        Args:
            lines: All lines in the document
            start_idx: Index of first list item

        Returns:
            Tuple of (list_lines, lines_consumed, list_type)
        """
        list_lines = []
        i = start_idx

        # Determine list type from first line
        first_line = lines[start_idx]
        is_ordered = bool(self.ORDERED_LIST.match(first_line))
        list_type = "ordered" if is_ordered else "unordered"

        while i < len(lines):
            line = lines[i]

            # Check for list item
            if is_ordered:
                match = self.ORDERED_LIST.match(line)
            else:
                match = self.UNORDERED_LIST.match(line)

            if match:
                list_lines.append(line.rstrip('\n'))
                i += 1
            elif line.strip() and line.startswith(('  ', '\t')):
                # Continuation line (indented)
                if list_lines:
                    list_lines.append(line.rstrip('\n'))
                    i += 1
                else:
                    break
            elif not line.strip():
                # Empty line might be part of list or end it
                # Look ahead to see if more list items follow
                if i + 1 < len(lines):
                    next_match = (self.ORDERED_LIST.match(lines[i + 1])
                                  if is_ordered
                                  else self.UNORDERED_LIST.match(lines[i + 1]))
                    if next_match:
                        list_lines.append("")
                        i += 1
                        continue
                break
            else:
                break

        # Remove trailing empty lines
        while list_lines and not list_lines[-1]:
            list_lines.pop()

        return list_lines, i - start_idx, list_type

    def _parse_blockquote(
        self,
        lines: list[str],
        start_idx: int,
    ) -> tuple[list[str], int]:
        """Parse a blockquote.

        Args:
            lines: All lines in the document
            start_idx: Index of first quote line

        Returns:
            Tuple of (quote_lines, lines_consumed)
        """
        quote_lines = []
        i = start_idx

        while i < len(lines):
            match = self.BLOCKQUOTE.match(lines[i])
            if match:
                quote_lines.append(match.group(1))
                i += 1
            elif not lines[i].strip():
                # Empty line might continue the quote
                if i + 1 < len(lines) and self.BLOCKQUOTE.match(lines[i + 1]):
                    quote_lines.append("")
                    i += 1
                else:
                    break
            else:
                break

        return quote_lines, i - start_idx

    def _find_heading_parent(
        self,
        root: Tree,
        current_parent: Tree,
        new_level: int,
    ) -> tuple[Tree, int]:
        """Find the appropriate parent node for a heading at the given level.

        Args:
            root: Root node of the tree
            current_parent: Current parent node
            new_level: Level of the new heading

        Returns:
            Tuple of (parent_node, parent_level)
        """
        # If new heading is deeper than current level, it's a child
        if new_level > current_parent.data.level:
            return current_parent, current_parent.data.level

        # Otherwise, traverse up to find appropriate level
        node = current_parent
        while node is not None and node != root:
            if node.data.is_heading() and node.data.level < new_level:
                return node, node.data.level
            node = node.parent

        # Default to root
        return root, 0

    def _split_text_intelligently(self, text: str) -> list[str]:
        """Split text at sentence boundaries or other natural break points.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not self.max_line_length or len(text) <= self.max_line_length:
            return [text]

        chunks = []

        # Try to split at sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', text)
        current_chunk = ""

        for segment in sentences:
            if len(current_chunk) + len(segment) <= self.max_line_length:
                current_chunk += segment
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk)

        # If we still have chunks that are too long, split at spaces
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.max_line_length:
                final_chunks.append(chunk)
            else:
                # Split at word boundaries
                words = chunk.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= self.max_line_length:
                        current += (" " if current else "") + word
                    else:
                        if current:
                            final_chunks.append(current)
                        current = word
                if current:
                    final_chunks.append(current)

        return final_chunks if final_chunks else [text]


def parse_markdown(
    source: str | TextIO | Iterator[str],
    max_line_length: int | None = None,
    preserve_empty_lines: bool = False,
) -> Tree:
    """Parse markdown content into a tree structure.

    Convenience function for creating and using a MarkdownParser.

    Args:
        source: Markdown content as string, file object, or line iterator
        max_line_length: Maximum length for body text lines (None for unlimited)
        preserve_empty_lines: Whether to preserve empty lines in output

    Returns:
        Tree with root node containing the document structure
    """
    parser = MarkdownParser(
        max_line_length=max_line_length,
        preserve_empty_lines=preserve_empty_lines,
    )
    return parser.parse(source)
