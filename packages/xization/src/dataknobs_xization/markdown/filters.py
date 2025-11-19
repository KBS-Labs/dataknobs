"""Quality filters for markdown chunks.

This module provides filtering utilities to identify and remove low-quality
chunks that would not contribute meaningful content to RAG retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataknobs_xization.markdown.md_chunker import Chunk


@dataclass
class ChunkQualityConfig:
    """Configuration for chunk quality filtering.

    Attributes:
        min_content_chars: Minimum characters of non-heading content
        min_alphanumeric_ratio: Minimum ratio of alphanumeric to total chars
        skip_heading_only: Skip chunks with only headings (no body content)
        min_words: Minimum word count for content
        allow_code_blocks: Allow short code blocks that would otherwise be filtered
        allow_tables: Allow short tables that would otherwise be filtered
    """

    min_content_chars: int = 50
    min_alphanumeric_ratio: float = 0.3
    skip_heading_only: bool = True
    min_words: int = 5
    allow_code_blocks: bool = True
    allow_tables: bool = True


class ChunkQualityFilter:
    """Filter for identifying and removing low-quality chunks.

    This filter helps ensure that only meaningful content is indexed
    for RAG retrieval, reducing noise and improving retrieval quality.
    """

    def __init__(self, config: ChunkQualityConfig | None = None):
        """Initialize the quality filter.

        Args:
            config: Quality configuration, uses defaults if not provided
        """
        self.config = config or ChunkQualityConfig()

    def is_valid(self, chunk: Chunk) -> bool:
        """Check if a chunk meets quality thresholds.

        Args:
            chunk: The chunk to evaluate

        Returns:
            True if chunk should be kept, False if it should be filtered
        """
        # Get node type from custom metadata
        node_type = chunk.metadata.custom.get("node_type", "body")

        # Special handling for code blocks and tables
        if node_type == "code" and self.config.allow_code_blocks:
            return self._is_valid_code_block(chunk)
        if node_type == "table" and self.config.allow_tables:
            return self._is_valid_table(chunk)

        # Extract content without heading markers
        content = self._extract_content_text(chunk.text)

        # Check for heading-only chunks
        if self.config.skip_heading_only and not content.strip():
            return False

        # Check minimum content length
        if len(content) < self.config.min_content_chars:
            return False

        # Check alphanumeric ratio
        if not self._meets_alphanumeric_threshold(content):
            return False

        # Check word count
        if not self._meets_word_count(content):
            return False

        return True

    def _extract_content_text(self, text: str) -> str:
        """Extract content text, removing markdown heading markers.

        Args:
            text: Raw chunk text

        Returns:
            Content without heading lines
        """
        lines = text.split("\n")
        content_lines = []

        for line in lines:
            # Skip markdown heading lines
            if re.match(r"^#+\s+", line):
                continue
            content_lines.append(line)

        return "\n".join(content_lines)

    def _meets_alphanumeric_threshold(self, text: str) -> bool:
        """Check if text meets minimum alphanumeric ratio.

        Args:
            text: Text to check

        Returns:
            True if ratio is met
        """
        if not text:
            return False

        alphanumeric_count = sum(1 for c in text if c.isalnum())
        total_count = len(text)

        if total_count == 0:
            return False

        ratio = alphanumeric_count / total_count
        return ratio >= self.config.min_alphanumeric_ratio

    def _meets_word_count(self, text: str) -> bool:
        """Check if text meets minimum word count.

        Args:
            text: Text to check

        Returns:
            True if word count is met
        """
        words = text.split()
        return len(words) >= self.config.min_words

    def _is_valid_code_block(self, chunk: Chunk) -> bool:
        """Check if a code block chunk is valid.

        Code blocks are given more lenient filtering since they may be
        short but still valuable (e.g., single function definitions).

        Args:
            chunk: Code block chunk

        Returns:
            True if code block should be kept
        """
        # Code blocks must have at least some content
        content = chunk.text.strip()
        if not content:
            return False

        # Allow code blocks with at least one non-whitespace line
        lines = [line for line in content.split("\n") if line.strip()]
        return len(lines) >= 1

    def _is_valid_table(self, chunk: Chunk) -> bool:
        """Check if a table chunk is valid.

        Tables are given more lenient filtering since they may be
        compact but information-rich.

        Args:
            chunk: Table chunk

        Returns:
            True if table should be kept
        """
        # Tables must have at least some content
        content = chunk.text.strip()
        if not content:
            return False

        # Tables should have at least header row and one data row
        lines = [line for line in content.split("\n") if line.strip()]
        return len(lines) >= 2

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter a list of chunks, keeping only valid ones.

        Args:
            chunks: List of chunks to filter

        Returns:
            List of chunks that pass quality thresholds
        """
        return [chunk for chunk in chunks if self.is_valid(chunk)]

    def get_rejection_reason(self, chunk: Chunk) -> str | None:
        """Get the reason a chunk would be rejected.

        Useful for debugging and understanding filtering behavior.

        Args:
            chunk: The chunk to evaluate

        Returns:
            Rejection reason string, or None if chunk is valid
        """
        node_type = chunk.metadata.custom.get("node_type", "body")

        if node_type == "code" and self.config.allow_code_blocks:
            if not self._is_valid_code_block(chunk):
                return "Empty code block"
            return None

        if node_type == "table" and self.config.allow_tables:
            if not self._is_valid_table(chunk):
                return "Empty or single-row table"
            return None

        content = self._extract_content_text(chunk.text)

        if self.config.skip_heading_only and not content.strip():
            return "Heading-only chunk (no body content)"

        if len(content) < self.config.min_content_chars:
            return f"Content too short ({len(content)} < {self.config.min_content_chars} chars)"

        if not self._meets_alphanumeric_threshold(content):
            return f"Alphanumeric ratio below threshold ({self.config.min_alphanumeric_ratio})"

        if not self._meets_word_count(content):
            words = len(content.split())
            return f"Word count too low ({words} < {self.config.min_words} words)"

        return None
