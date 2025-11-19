"""Tests for retrieval utilities (ChunkMerger and ContextFormatter)."""

import pytest

from dataknobs_bots.knowledge.retrieval import (
    ChunkMerger,
    ContextFormatter,
    FormatterConfig,
    MergedChunk,
    MergerConfig,
)


class TestMergerConfig:
    """Tests for MergerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MergerConfig()
        assert config.max_merged_size == 2000
        assert config.preserve_order is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MergerConfig(max_merged_size=5000, preserve_order=False)
        assert config.max_merged_size == 5000
        assert config.preserve_order is False


class TestMergedChunk:
    """Tests for MergedChunk."""

    def test_creation(self):
        """Test creating a merged chunk."""
        chunk = MergedChunk(
            text="text1\n\ntext2",
            heading_path=["Chapter", "Section"],
            source="test.md",
            heading_display="Chapter > Section",
            chunks=[{"text": "text1"}, {"text": "text2"}],
            avg_similarity=0.95,
            content_length=100,
        )
        assert len(chunk.chunks) == 2
        assert chunk.heading_path == ["Chapter", "Section"]
        assert chunk.source == "test.md"
        assert chunk.avg_similarity == 0.95

    def test_merged_text(self):
        """Test getting merged text."""
        chunk = MergedChunk(
            text="First paragraph.\n\nSecond paragraph.",
            heading_path=[],
            source="",
            heading_display="",
            chunks=[{"text": "First paragraph."}, {"text": "Second paragraph."}],
            avg_similarity=0.9,
            content_length=33,
        )
        assert chunk.text == "First paragraph.\n\nSecond paragraph."


class TestChunkMerger:
    """Tests for ChunkMerger."""

    def _make_result(
        self,
        text: str,
        heading_path: list[str],
        source: str = "test.md",
        similarity: float = 0.9,
        chunk_index: int = 0,
    ) -> dict:
        """Helper to create a search result."""
        return {
            "text": text,
            "heading_path": heading_path,
            "source": source,
            "similarity": similarity,
            "metadata": {"chunk_index": chunk_index},
        }

    def test_merge_adjacent_chunks(self):
        """Test merging adjacent chunks with same heading."""
        merger = ChunkMerger()
        results = [
            self._make_result("First chunk.", ["Chapter", "Section"], chunk_index=0),
            self._make_result("Second chunk.", ["Chapter", "Section"], chunk_index=1),
        ]
        merged = merger.merge(results)

        assert len(merged) == 1
        assert merged[0].text == "First chunk.\n\nSecond chunk."
        assert len(merged[0].chunks) == 2

    def test_no_merge_different_headings(self):
        """Test that chunks with different headings are not merged."""
        merger = ChunkMerger()
        results = [
            self._make_result("First chunk.", ["Chapter", "Section A"]),
            self._make_result("Second chunk.", ["Chapter", "Section B"]),
        ]
        merged = merger.merge(results)

        assert len(merged) == 2

    def test_no_merge_different_sources(self):
        """Test that chunks from different sources are not merged."""
        merger = ChunkMerger()
        results = [
            self._make_result("First chunk.", ["Chapter"], source="file1.md"),
            self._make_result("Second chunk.", ["Chapter"], source="file2.md"),
        ]
        merged = merger.merge(results)

        assert len(merged) == 2

    def test_max_merged_size_limit(self):
        """Test that merging respects max size limit."""
        merger = ChunkMerger(MergerConfig(max_merged_size=50))
        results = [
            self._make_result("A" * 30, ["Chapter"], chunk_index=0),
            self._make_result("B" * 30, ["Chapter"], chunk_index=1),
        ]
        merged = merger.merge(results)

        # Should not merge because combined size > 50
        assert len(merged) == 2

    def test_preserve_order(self):
        """Test that merged chunks preserve chunk order."""
        merger = ChunkMerger(MergerConfig(preserve_order=True))
        results = [
            self._make_result("Third.", ["Section"], chunk_index=2, similarity=0.95),
            self._make_result("First.", ["Section"], chunk_index=0, similarity=0.8),
            self._make_result("Second.", ["Section"], chunk_index=1, similarity=0.7),
        ]
        merged = merger.merge(results)

        assert len(merged) == 1
        assert merged[0].text == "First.\n\nSecond.\n\nThird."

    def test_avg_similarity_preserved(self):
        """Test that average similarity is calculated correctly."""
        merger = ChunkMerger()
        results = [
            self._make_result("First.", ["Section"], chunk_index=0, similarity=0.7),
            self._make_result("Second.", ["Section"], chunk_index=1, similarity=0.9),
        ]
        merged = merger.merge(results)

        # Average of 0.7 and 0.9 is 0.8
        assert merged[0].avg_similarity == 0.8

    def test_empty_results(self):
        """Test merging empty results."""
        merger = ChunkMerger()
        merged = merger.merge([])
        assert merged == []

    def test_single_result(self):
        """Test merging single result."""
        merger = ChunkMerger()
        results = [self._make_result("Only one.", ["Section"])]
        merged = merger.merge(results)

        assert len(merged) == 1
        assert merged[0].text == "Only one."

    def test_to_result_list(self):
        """Test converting merged chunks back to result list."""
        merger = ChunkMerger()
        merged_chunks = [
            MergedChunk(
                text="Text 1\n\nText 2",
                heading_path=["Chapter", "Section"],
                source="test.md",
                heading_display="Chapter > Section",
                chunks=[{"text": "Text 1"}, {"text": "Text 2"}],
                avg_similarity=0.9,
                content_length=14,
            )
        ]
        results = merger.to_result_list(merged_chunks)

        assert len(results) == 1
        assert results[0]["text"] == "Text 1\n\nText 2"
        assert results[0]["heading_path"] == "Chapter > Section"
        assert results[0]["similarity"] == 0.9

    def test_complex_merge_scenario(self):
        """Test complex scenario with multiple groups."""
        merger = ChunkMerger()
        results = [
            self._make_result("A1", ["A"], "file1.md", chunk_index=0),
            self._make_result("A2", ["A"], "file1.md", chunk_index=1),
            self._make_result("B1", ["B"], "file1.md", chunk_index=0),
            self._make_result("A3", ["A"], "file2.md", chunk_index=0),
        ]
        merged = merger.merge(results)

        # Should have 3 groups: A in file1, B in file1, A in file2
        assert len(merged) == 3


class TestFormatterConfig:
    """Tests for FormatterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FormatterConfig()
        assert config.small_chunk_threshold == 200
        assert config.medium_chunk_threshold == 800
        assert config.include_scores is False
        assert config.include_source is True
        assert config.group_by_source is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FormatterConfig(
            small_chunk_threshold=100,
            medium_chunk_threshold=500,
            include_scores=True,
            include_source=False,
            group_by_source=True,
        )
        assert config.small_chunk_threshold == 100
        assert config.include_scores is True
        assert config.group_by_source is True


class TestContextFormatter:
    """Tests for ContextFormatter."""

    def _make_result(
        self,
        text: str,
        heading_path: list[str] | None = None,
        source: str = "test.md",
        similarity: float = 0.9,
    ) -> dict:
        """Helper to create a search result."""
        return {
            "text": text,
            "heading_path": heading_path or [],
            "source": source,
            "similarity": similarity,
            "metadata": {"headings": heading_path or []},
        }

    def test_basic_formatting(self):
        """Test basic formatting of results."""
        formatter = ContextFormatter()
        results = [
            self._make_result("This is the content.", ["Chapter", "Section"]),
        ]
        output = formatter.format(results)

        assert "This is the content." in output
        assert "test.md" in output

    def test_dynamic_heading_small(self):
        """Test that small chunks include headings."""
        formatter = ContextFormatter(
            FormatterConfig(small_chunk_threshold=200, medium_chunk_threshold=800)
        )
        results = [
            self._make_result(
                "Short content",  # < 200 chars
                ["Chapter", "Section", "Subsection"],
            ),
        ]
        output = formatter.format(results)

        # Should include content and source
        assert "Short content" in output
        assert "test.md" in output

    def test_dynamic_heading_medium(self):
        """Test that medium chunks are formatted."""
        formatter = ContextFormatter(
            FormatterConfig(small_chunk_threshold=50, medium_chunk_threshold=500)
        )
        results = [
            self._make_result(
                "A" * 200,  # Medium size content
                ["Chapter", "Section", "Subsection"],
            ),
        ]
        output = formatter.format(results)

        # Should include content
        assert "A" * 50 in output

    def test_include_scores(self):
        """Test including scores in output."""
        formatter = ContextFormatter(FormatterConfig(include_scores=True))
        results = [
            self._make_result("Content here.", similarity=0.95),
        ]
        output = formatter.format(results)

        assert "0.95" in output

    def test_exclude_source(self):
        """Test excluding source from output."""
        formatter = ContextFormatter(FormatterConfig(include_source=False))
        results = [
            self._make_result("Content here.", source="should_not_appear.md"),
        ]
        output = formatter.format(results)

        assert "should_not_appear.md" not in output

    def test_multiple_results(self):
        """Test formatting multiple results."""
        formatter = ContextFormatter()
        results = [
            self._make_result("First result.", ["A"]),
            self._make_result("Second result.", ["B"]),
        ]
        output = formatter.format(results)

        assert "First result." in output
        assert "Second result." in output

    def test_empty_results(self):
        """Test formatting empty results."""
        formatter = ContextFormatter()
        output = formatter.format([])
        assert output == ""

    def test_wrap_for_prompt(self):
        """Test wrapping context in XML tags."""
        formatter = ContextFormatter()
        context = "Some context here"
        wrapped = formatter.wrap_for_prompt(context)

        assert wrapped.startswith("<knowledge_base>")
        assert wrapped.endswith("</knowledge_base>")
        assert "Some context here" in wrapped

    def test_wrap_custom_tag(self):
        """Test wrapping with custom tag."""
        formatter = ContextFormatter()
        context = "Some context"
        wrapped = formatter.wrap_for_prompt(context, tag="custom_context")

        assert "<custom_context>" in wrapped
        assert "</custom_context>" in wrapped

    def test_group_by_source(self):
        """Test grouping results by source."""
        formatter = ContextFormatter(FormatterConfig(group_by_source=True))
        results = [
            self._make_result("From file1 first", source="file1.md"),
            self._make_result("From file2", source="file2.md"),
            self._make_result("From file1 second", source="file1.md"),
        ]
        output = formatter.format(results)

        # Results should be grouped
        assert output.count("file1.md") <= 2  # Should appear fewer times
        assert "file2.md" in output

    def test_no_heading_path(self):
        """Test formatting results with no heading path."""
        formatter = ContextFormatter()
        results = [
            self._make_result("Content without headings", heading_path=[]),
        ]
        output = formatter.format(results)

        assert "Content without headings" in output
