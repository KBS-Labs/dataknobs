"""Tests for markdown streaming module."""

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from dataknobs_xization.markdown.md_chunker import ChunkFormat, HeadingInclusion
from dataknobs_xization.markdown.md_streaming import (
    AdaptiveStreamingProcessor,
    StreamingMarkdownProcessor,
    stream_markdown_file,
    stream_markdown_string,
)


class TestStreamingMarkdownProcessor:
    """Tests for StreamingMarkdownProcessor class."""

    def test_process_string_basic(self):
        """Test processing a simple markdown string."""
        markdown = """# Title
Body text."""

        processor = StreamingMarkdownProcessor()
        chunks = list(processor.process_string(markdown))

        assert len(chunks) > 0
        assert chunks[0].text is not None

    def test_process_string_multiple_sections(self):
        """Test processing markdown with multiple sections."""
        markdown = """# Section 1
Text for section 1.

# Section 2
Text for section 2.

# Section 3
Text for section 3."""

        processor = StreamingMarkdownProcessor()
        chunks = list(processor.process_string(markdown))

        assert len(chunks) == 3

    def test_process_stream_from_stringio(self):
        """Test processing from StringIO object."""
        markdown = """# Title
Body text."""
        stream = StringIO(markdown)

        processor = StreamingMarkdownProcessor()
        chunks = list(processor.process_stream(stream))

        assert len(chunks) > 0

    def test_process_stream_from_iterator(self):
        """Test processing from line iterator."""
        lines = ["# Title", "Body text."]

        processor = StreamingMarkdownProcessor()
        chunks = list(processor.process_stream(iter(lines)))

        assert len(chunks) > 0

    def test_process_file(self):
        """Test processing from a file."""
        markdown = """# Title
Body text for testing file processing."""

        # Create a temporary file
        with NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(markdown)
            temp_path = f.name

        try:
            processor = StreamingMarkdownProcessor()
            chunks = list(processor.process_file(temp_path))

            assert len(chunks) > 0
            assert "Body text" in chunks[0].text
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_max_chunk_size_configuration(self):
        """Test configuring max chunk size."""
        long_text = "This is a sentence. " * 100
        markdown = f"# Title\n{long_text}"

        processor = StreamingMarkdownProcessor(max_chunk_size=100)
        chunks = list(processor.process_string(markdown))

        # Should be split into multiple chunks
        assert len(chunks) > 1

    def test_heading_inclusion_configuration(self):
        """Test configuring heading inclusion."""
        markdown = """# Title
Body text."""

        processor = StreamingMarkdownProcessor(
            heading_inclusion=HeadingInclusion.IN_METADATA,
        )
        chunks = list(processor.process_string(markdown))

        assert len(chunks) > 0
        # Headings should be in metadata
        assert chunks[0].metadata.headings == ["Title"]
        # Body should not have heading
        assert chunks[0].text.strip() == "Body text."

    def test_chunk_format_configuration(self):
        """Test configuring chunk format."""
        markdown = """# Title
Body text."""

        processor = StreamingMarkdownProcessor(
            chunk_format=ChunkFormat.PLAIN,
            heading_inclusion=HeadingInclusion.IN_TEXT,
        )
        chunks = list(processor.process_string(markdown))

        assert len(chunks) > 0
        # Should not have markdown formatting
        assert "# " not in chunks[0].text

    def test_chunk_overlap_configuration(self):
        """Test configuring chunk overlap."""
        long_text = "Sentence. " * 50
        markdown = f"# Title\n{long_text}"

        processor = StreamingMarkdownProcessor(
            max_chunk_size=100,
            chunk_overlap=20,
        )
        chunks = list(processor.process_string(markdown))

        # Should have multiple chunks
        assert len(chunks) > 1

    def test_max_line_length_configuration(self):
        """Test configuring max line length."""
        long_line = "Word " * 100
        markdown = f"# Title\n{long_line}"

        processor = StreamingMarkdownProcessor(max_line_length=50)
        chunks = list(processor.process_string(markdown))

        # Parser should split long lines
        assert len(chunks) > 0


class TestAdaptiveStreamingProcessor:
    """Tests for AdaptiveStreamingProcessor class."""

    def test_adaptive_processor_basic(self):
        """Test basic adaptive processing."""
        markdown = """# Title
Body text."""

        processor = AdaptiveStreamingProcessor()
        chunks = list(processor.process_string(markdown))

        assert len(chunks) > 0

    def test_adaptive_processor_large_document(self):
        """Test adaptive processing with larger document."""
        # Create a larger document
        sections = []
        for i in range(20):
            sections.append(f"# Section {i}")
            sections.append(f"Body text for section {i}.")
            sections.append("")

        markdown = "\n".join(sections)

        processor = AdaptiveStreamingProcessor(
            memory_limit_nodes=100,
            adaptive_threshold=0.5,
        )
        chunks = list(processor.process_string(markdown))

        # Should process all sections
        assert len(chunks) > 0

    def test_adaptive_threshold_configuration(self):
        """Test configuring adaptive threshold."""
        markdown = """# Title
Body text."""

        processor = AdaptiveStreamingProcessor(
            memory_limit_nodes=1000,
            adaptive_threshold=0.8,
        )
        chunks = list(processor.process_string(markdown))

        assert len(chunks) > 0

    def test_adaptive_processor_with_nested_headings(self):
        """Test adaptive processor with nested heading structure."""
        markdown = """# Chapter 1
## Section 1.1
Body for 1.1.

### Subsection 1.1.1
Body for 1.1.1.

## Section 1.2
Body for 1.2.

# Chapter 2
Body for chapter 2."""

        processor = AdaptiveStreamingProcessor()
        chunks = list(processor.process_string(markdown))

        # Should process all sections
        assert len(chunks) > 0

        # Check that heading hierarchy is preserved
        for chunk in chunks:
            if chunk.metadata.headings:
                # Headings should be in order
                assert len(chunk.metadata.headings) == len(chunk.metadata.heading_levels)


class TestStreamMarkdownFileFunction:
    """Tests for stream_markdown_file convenience function."""

    def test_stream_markdown_file_basic(self):
        """Test basic file streaming."""
        markdown = """# Title
Body text."""

        with NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(markdown)
            temp_path = f.name

        try:
            chunks = list(stream_markdown_file(temp_path))
            assert len(chunks) > 0
        finally:
            Path(temp_path).unlink()

    def test_stream_markdown_file_with_options(self):
        """Test file streaming with custom options."""
        markdown = """# Title
Body text."""

        with NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(markdown)
            temp_path = f.name

        try:
            chunks = list(stream_markdown_file(
                temp_path,
                max_chunk_size=50,
                heading_inclusion=HeadingInclusion.IN_TEXT,
            ))
            assert len(chunks) > 0
            assert "Title" in chunks[0].text
        finally:
            Path(temp_path).unlink()


class TestStreamMarkdownStringFunction:
    """Tests for stream_markdown_string convenience function."""

    def test_stream_markdown_string_basic(self):
        """Test basic string streaming."""
        markdown = """# Title
Body text."""

        chunks = list(stream_markdown_string(markdown))
        assert len(chunks) > 0

    def test_stream_markdown_string_with_options(self):
        """Test string streaming with custom options."""
        markdown = """# Title
Body text."""

        chunks = list(stream_markdown_string(
            markdown,
            max_chunk_size=50,
            chunk_overlap=10,
            heading_inclusion=HeadingInclusion.BOTH,
        ))

        assert len(chunks) > 0
        # Should have headings in both text and metadata
        assert "Title" in chunks[0].text
        assert chunks[0].metadata.headings == ["Title"]

    def test_stream_markdown_string_empty(self):
        """Test streaming empty markdown."""
        markdown = ""

        chunks = list(stream_markdown_string(markdown))
        # Should handle empty input gracefully
        assert len(chunks) == 0

    def test_stream_markdown_string_headings_only(self):
        """Test streaming markdown with only headings (no body)."""
        markdown = """# Title
## Subtitle
### Subsubtitle"""

        chunks = list(stream_markdown_string(markdown))
        # Should handle headings without body text
        assert len(chunks) == 0  # No body text to chunk

    def test_stream_markdown_string_complex_document(self):
        """Test streaming a complex document."""
        markdown = """# Introduction
This is the introduction section with some text.

## Background
Some background information here.

### Historical Context
Details about the history.

## Motivation
Why this work is important.

# Methods
Description of the methods used.

## Experimental Setup
How the experiment was set up.

## Data Collection
How data was collected.

# Results
The results of the study.

## Findings
What was discovered.

### Key Insights
Important insights from the findings.

# Conclusion
Summary and conclusions."""

        chunks = list(stream_markdown_string(markdown))

        # Should have chunks for each section with body text
        assert len(chunks) > 0

        # Verify that chunks have proper metadata
        for chunk in chunks:
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.line_number > 0
            assert chunk.metadata.chunk_size > 0
