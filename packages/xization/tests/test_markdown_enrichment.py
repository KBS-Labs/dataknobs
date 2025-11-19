"""Tests for markdown heading enrichment utilities."""

import pytest

from dataknobs_xization.markdown.enrichment import (
    build_enriched_text,
    format_heading_display,
    get_dynamic_heading_display,
    is_multiword,
)


class TestIsMultiword:
    """Tests for is_multiword function."""

    def test_single_word(self):
        """Test single word returns False."""
        assert is_multiword("Patterns") is False
        assert is_multiword("Introduction") is False
        assert is_multiword("API") is False

    def test_multi_word(self):
        """Test multi-word returns True."""
        assert is_multiword("Chain of Thought") is True
        assert is_multiword("Getting Started") is True
        assert is_multiword("Two Words") is True

    def test_empty_string(self):
        """Test empty string returns False."""
        assert is_multiword("") is False

    def test_whitespace_only(self):
        """Test whitespace-only returns False."""
        assert is_multiword("   ") is False

    def test_hyphenated_word(self):
        """Test hyphenated word counts as single word."""
        assert is_multiword("chain-of-thought") is False

    def test_leading_trailing_whitespace(self):
        """Test whitespace is stripped."""
        assert is_multiword("  Patterns  ") is False
        assert is_multiword("  Two Words  ") is True


class TestBuildEnrichedText:
    """Tests for build_enriched_text function."""

    def test_empty_heading_path(self):
        """Test with no headings returns content only."""
        result = build_enriched_text([], "Some content here")
        assert result == "Some content here"

    def test_single_multiword_heading(self):
        """Test with single multi-word heading."""
        result = build_enriched_text(["Chain of Thought"], "Example code")
        assert result == "Chain of Thought: Example code"

    def test_single_singleword_heading(self):
        """Test with single single-word heading."""
        result = build_enriched_text(["Patterns"], "Example code")
        assert result == "Patterns: Example code"

    def test_path_with_multiword_at_end(self):
        """Test path ending in multi-word heading."""
        result = build_enriched_text(
            ["Patterns", "Chain of Thought", "Example"],
            "code here"
        )
        # Should include from "Chain of Thought" onwards
        assert result == "Chain of Thought: Example: code here"

    def test_all_single_word_headings(self):
        """Test path with all single-word headings."""
        result = build_enriched_text(
            ["Patterns", "Advanced", "Examples"],
            "content here"
        )
        # All single words, so include all
        assert result == "Patterns: Advanced: Examples: content here"

    def test_multiword_in_middle(self):
        """Test path with multi-word heading in middle."""
        result = build_enriched_text(
            ["Topics", "Chain of Thought", "Basics", "Example"],
            "content"
        )
        # Should include from first multi-word (Chain of Thought) to end
        assert result == "Chain of Thought: Basics: Example: content"

    def test_multiple_multiword_headings(self):
        """Test path with multiple multi-word headings."""
        result = build_enriched_text(
            ["Topics", "Chain of Thought", "Zero Shot Learning"],
            "content"
        )
        # Should stop at first multi-word going backwards (Zero Shot Learning)
        # Then include Chain of Thought
        # Actually - the algorithm goes backwards and stops at first multiword
        assert result == "Zero Shot Learning: content"

    def test_preserves_content_exactly(self):
        """Test that content is not modified."""
        content = "  Some content\nwith newlines\n\nand spacing  "
        result = build_enriched_text(["Heading"], content)
        assert result == f"Heading: {content}"


class TestFormatHeadingDisplay:
    """Tests for format_heading_display function."""

    def test_empty_path(self):
        """Test empty heading path."""
        result = format_heading_display([])
        assert result == ""

    def test_single_heading(self):
        """Test single heading."""
        result = format_heading_display(["Introduction"])
        assert result == "Introduction"

    def test_multiple_headings(self):
        """Test multiple headings joined with separator."""
        result = format_heading_display(["Chapter 1", "Section A", "Details"])
        assert result == "Chapter 1 > Section A > Details"

    def test_custom_separator(self):
        """Test custom separator."""
        result = format_heading_display(["A", "B", "C"], separator=" / ")
        assert result == "A / B / C"


class TestGetDynamicHeadingDisplay:
    """Tests for get_dynamic_heading_display function."""

    def test_empty_path(self):
        """Test empty heading path returns empty string."""
        result = get_dynamic_heading_display([], 100)
        assert result == ""

    def test_small_chunk_full_path(self):
        """Test small chunks get full heading path."""
        result = get_dynamic_heading_display(
            ["Chapter", "Section", "Subsection"],
            content_length=150,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "Chapter > Section > Subsection"

    def test_medium_chunk_last_two(self):
        """Test medium chunks get last two headings."""
        result = get_dynamic_heading_display(
            ["Chapter", "Section", "Subsection"],
            content_length=500,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "Section > Subsection"

    def test_large_chunk_empty(self):
        """Test large chunks get no heading display."""
        result = get_dynamic_heading_display(
            ["Chapter", "Section", "Subsection"],
            content_length=1000,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == ""

    def test_single_heading_all_sizes(self):
        """Test single heading at all content sizes."""
        # Small - get full path (1 heading)
        result = get_dynamic_heading_display(
            ["Chapter"],
            content_length=100,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "Chapter"

        # Medium - get last 2, but only 1 exists
        result = get_dynamic_heading_display(
            ["Chapter"],
            content_length=500,
        )
        assert result == "Chapter"

        # Large - no display
        result = get_dynamic_heading_display(
            ["Chapter"],
            content_length=1000,
        )
        assert result == ""

    def test_two_headings_medium(self):
        """Test two headings at medium size."""
        result = get_dynamic_heading_display(
            ["Chapter", "Section"],
            content_length=500,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "Chapter > Section"

    def test_exact_thresholds(self):
        """Test behavior at exact threshold boundaries."""
        # At exactly small threshold - should be small
        result = get_dynamic_heading_display(
            ["A", "B", "C"],
            content_length=200,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "A > B > C"

        # At exactly medium threshold - should be medium
        result = get_dynamic_heading_display(
            ["A", "B", "C"],
            content_length=800,
            small_threshold=200,
            medium_threshold=800,
        )
        assert result == "B > C"

    def test_default_thresholds(self):
        """Test with default thresholds."""
        # Very small
        result = get_dynamic_heading_display(
            ["A", "B", "C"],
            content_length=50,
        )
        assert result == "A > B > C"

        # Medium (between 200 and 800)
        result = get_dynamic_heading_display(
            ["A", "B", "C"],
            content_length=400,
        )
        assert result == "B > C"

        # Large
        result = get_dynamic_heading_display(
            ["A", "B", "C"],
            content_length=1200,
        )
        assert result == ""
