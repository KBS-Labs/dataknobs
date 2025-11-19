"""Tests for markdown chunk quality filtering."""

import pytest

from dataknobs_xization.markdown.filters import (
    ChunkQualityConfig,
    ChunkQualityFilter,
)
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata


class TestChunkQualityConfig:
    """Tests for ChunkQualityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkQualityConfig()
        assert config.min_content_chars == 50
        assert config.min_alphanumeric_ratio == 0.3
        assert config.skip_heading_only is True
        assert config.min_words == 5
        assert config.allow_code_blocks is True
        assert config.allow_tables is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkQualityConfig(
            min_content_chars=100,
            min_alphanumeric_ratio=0.5,
            skip_heading_only=False,
            min_words=10,
            allow_code_blocks=False,
            allow_tables=False,
        )
        assert config.min_content_chars == 100
        assert config.min_alphanumeric_ratio == 0.5
        assert config.skip_heading_only is False
        assert config.min_words == 10
        assert config.allow_code_blocks is False
        assert config.allow_tables is False


class TestChunkQualityFilter:
    """Tests for ChunkQualityFilter."""

    def _make_chunk(
        self,
        text: str,
        headings: list[str] | None = None,
        node_type: str = "body",
    ) -> Chunk:
        """Helper to create a chunk with given text."""
        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                headings=headings or ["Test"],
                heading_levels=[1] * len(headings or ["Test"]),
                custom={"node_type": node_type},
            ),
        )

    def test_valid_chunk(self):
        """Test that valid chunks pass filtering."""
        filter_obj = ChunkQualityFilter()
        chunk = self._make_chunk(
            "This is a valid chunk with enough content and words to pass the filter."
        )
        assert filter_obj.is_valid(chunk) is True

    def test_too_short_content(self):
        """Test that chunks with too few characters are filtered."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(min_content_chars=50))
        chunk = self._make_chunk("Short")  # Only 5 chars
        assert filter_obj.is_valid(chunk) is False

    def test_too_few_words(self):
        """Test that chunks with too few words are filtered."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(min_words=5))
        chunk = self._make_chunk("Only three words")  # Only 3 words
        assert filter_obj.is_valid(chunk) is False

    def test_low_alphanumeric_ratio(self):
        """Test that chunks with low alphanumeric ratio are filtered."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(min_alphanumeric_ratio=0.5))
        # Lots of punctuation
        chunk = self._make_chunk("---...---...---...---...---...")
        assert filter_obj.is_valid(chunk) is False

    def test_heading_only_chunk_filtered(self):
        """Test that heading-only chunks are filtered by default."""
        filter_obj = ChunkQualityFilter()
        # Text that matches heading exactly
        chunk = self._make_chunk("Test", headings=["Test"])
        assert filter_obj.is_valid(chunk) is False

    def test_heading_only_chunk_allowed(self):
        """Test that heading-only chunks can be allowed."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(
            skip_heading_only=False,
            min_content_chars=10,
            min_words=3,
        ))
        chunk = self._make_chunk(
            "Test with enough content and words", headings=["Test"]
        )
        assert filter_obj.is_valid(chunk) is True

    def test_code_blocks_allowed(self):
        """Test that code blocks bypass word count check."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(allow_code_blocks=True))
        chunk = self._make_chunk("```python\nprint('hello')\n```", node_type="code")
        assert filter_obj.is_valid(chunk) is True

    def test_code_blocks_not_allowed(self):
        """Test that code blocks can be filtered when disabled."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(allow_code_blocks=False))
        chunk = self._make_chunk("```python\nx\n```", node_type="code")  # Will be filtered as body
        assert filter_obj.is_valid(chunk) is False

    def test_tables_allowed(self):
        """Test that tables bypass word count check."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(allow_tables=True))
        chunk = self._make_chunk("| A | B |\n|---|---|\n| 1 | 2 |", node_type="table")
        assert filter_obj.is_valid(chunk) is True

    def test_tables_not_allowed(self):
        """Test that tables can be filtered when disabled."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(allow_tables=False))
        chunk = self._make_chunk("| A |\n|---|", node_type="table")  # Will be filtered as body
        assert filter_obj.is_valid(chunk) is False

    def test_filter_chunks_list(self):
        """Test filtering a list of chunks."""
        filter_obj = ChunkQualityFilter()
        chunks = [
            self._make_chunk("This is a valid chunk with enough content to pass."),
            self._make_chunk("Short"),  # Too short
            self._make_chunk(
                "Another valid chunk with sufficient words and characters."
            ),
            self._make_chunk("---"),  # Low alphanumeric
        ]
        filtered = filter_obj.filter_chunks(chunks)
        assert len(filtered) == 2

    def test_filter_empty_list(self):
        """Test filtering an empty list."""
        filter_obj = ChunkQualityFilter()
        filtered = filter_obj.filter_chunks([])
        assert filtered == []

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped before checking content length."""
        filter_obj = ChunkQualityFilter(ChunkQualityConfig(min_content_chars=10))
        chunk = self._make_chunk("   a   ")  # Only 1 char after stripping
        assert filter_obj.is_valid(chunk) is False

    def test_multiline_content(self):
        """Test chunks with multiple lines."""
        filter_obj = ChunkQualityFilter()
        chunk = self._make_chunk(
            "Line one with content.\n\nLine two with more content.\n\nLine three."
        )
        assert filter_obj.is_valid(chunk) is True

    def test_mixed_content_types(self):
        """Test chunk with mixed content including code."""
        filter_obj = ChunkQualityFilter()
        chunk = self._make_chunk(
            "Here is some code:\n\n```python\ndef hello():\n    print('world')\n```"
        )
        assert filter_obj.is_valid(chunk) is True

    def test_edge_case_exact_minimum(self):
        """Test chunk at exactly minimum thresholds."""
        config = ChunkQualityConfig(
            min_content_chars=10,
            min_words=3,
            min_alphanumeric_ratio=0.3,
        )
        filter_obj = ChunkQualityFilter(config)
        # Exactly 10 chars, 3 words
        chunk = self._make_chunk("One two th")
        assert filter_obj.is_valid(chunk) is True
