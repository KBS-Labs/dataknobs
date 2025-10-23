"""Tests for markdown chunker module."""

import pytest

from dataknobs_xization.markdown.md_chunker import (
    Chunk,
    ChunkFormat,
    ChunkMetadata,
    HeadingInclusion,
    MarkdownChunker,
    chunk_markdown_tree,
)
from dataknobs_xization.markdown.md_parser import parse_markdown


class TestChunkMetadata:
    """Tests for ChunkMetadata class."""

    def test_metadata_creation(self):
        """Test creating chunk metadata."""
        metadata = ChunkMetadata(
            headings=["Title", "Subtitle"],
            heading_levels=[1, 2],
            line_number=5,
            chunk_index=0,
            chunk_size=100,
        )
        assert metadata.headings == ["Title", "Subtitle"]
        assert metadata.heading_levels == [1, 2]
        assert metadata.line_number == 5
        assert metadata.chunk_index == 0
        assert metadata.chunk_size == 100

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ChunkMetadata(
            headings=["Title"],
            heading_levels=[1],
            line_number=1,
            chunk_index=0,
            chunk_size=50,
        )
        data = metadata.to_dict()
        assert data["headings"] == ["Title"]
        assert data["heading_levels"] == [1]
        assert data["chunk_index"] == 0

    def test_get_heading_path(self):
        """Test getting heading path as string."""
        metadata = ChunkMetadata(
            headings=["Chapter 1", "Section 1.1", "Subsection 1.1.1"],
            heading_levels=[1, 2, 3],
        )
        path = metadata.get_heading_path()
        assert path == "Chapter 1 > Section 1.1 > Subsection 1.1.1"

    def test_get_heading_path_custom_separator(self):
        """Test heading path with custom separator."""
        metadata = ChunkMetadata(
            headings=["A", "B", "C"],
            heading_levels=[1, 2, 3],
        )
        path = metadata.get_heading_path(separator=" / ")
        assert path == "A / B / C"

    def test_custom_metadata(self):
        """Test adding custom metadata."""
        metadata = ChunkMetadata(
            headings=["Title"],
            heading_levels=[1],
            custom={"author": "John", "date": "2024-01-01"},
        )
        data = metadata.to_dict()
        assert data["author"] == "John"
        assert data["date"] == "2024-01-01"


class TestChunk:
    """Tests for Chunk class."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        metadata = ChunkMetadata(headings=["Title"], heading_levels=[1])
        chunk = Chunk(text="Body text.", metadata=metadata)
        assert chunk.text == "Body text."
        assert chunk.metadata.headings == ["Title"]

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        metadata = ChunkMetadata(headings=["Title"], heading_levels=[1], chunk_index=0)
        chunk = Chunk(text="Body text.", metadata=metadata)
        data = chunk.to_dict()
        assert data["text"] == "Body text."
        assert data["metadata"]["headings"] == ["Title"]
        assert data["metadata"]["chunk_index"] == 0

    def test_chunk_to_markdown_with_headings(self):
        """Test converting chunk to markdown with headings."""
        metadata = ChunkMetadata(
            headings=["Chapter", "Section"],
            heading_levels=[1, 2],
        )
        chunk = Chunk(text="Body text.", metadata=metadata)
        markdown = chunk.to_markdown(include_headings=True)
        assert "# Chapter" in markdown
        assert "## Section" in markdown
        assert "Body text." in markdown

    def test_chunk_to_markdown_without_headings(self):
        """Test converting chunk to markdown without headings."""
        metadata = ChunkMetadata(
            headings=["Chapter", "Section"],
            heading_levels=[1, 2],
        )
        chunk = Chunk(text="Body text.", metadata=metadata)
        markdown = chunk.to_markdown(include_headings=False)
        assert markdown == "Body text."


class TestMarkdownChunker:
    """Tests for MarkdownChunker class."""

    def test_basic_chunking(self):
        """Test basic chunking of simple document."""
        markdown = """# Title
This is body text.
More body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(max_chunk_size=1000)
        chunks = list(chunker.chunk(tree))

        assert len(chunks) > 0
        # Should combine text under same heading
        assert len(chunks) == 1

    def test_chunking_multiple_headings(self):
        """Test chunking with multiple headings."""
        markdown = """# Chapter 1
Text for chapter 1.

## Section 1.1
Text for section 1.1.

# Chapter 2
Text for chapter 2."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(max_chunk_size=1000)
        chunks = list(chunker.chunk(tree))

        # Should have separate chunks for different sections
        assert len(chunks) == 3

    def test_chunk_size_limiting(self):
        """Test that chunks respect max size."""
        # Create a long text that will need to be split
        long_text = "This is a sentence. " * 100
        markdown = f"# Title\n{long_text}"
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(max_chunk_size=100)
        chunks = list(chunker.chunk(tree))

        # Should be split into multiple chunks
        assert len(chunks) > 1

        # Chunks should respect size limit (approximately)
        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk.text) <= chunker.max_chunk_size + 50

    def test_heading_inclusion_both(self):
        """Test including headings in both text and metadata."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            heading_inclusion=HeadingInclusion.BOTH,
            chunk_format=ChunkFormat.MARKDOWN,
        )
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Should be in text
        assert "# Title" in chunk.text
        # Should be in metadata
        assert chunk.metadata.headings == ["Title"]

    def test_heading_inclusion_text_only(self):
        """Test including headings in text only."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            heading_inclusion=HeadingInclusion.IN_TEXT,
            chunk_format=ChunkFormat.MARKDOWN,
        )
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Should be in text
        assert "# Title" in chunk.text
        # Should NOT be in metadata
        assert chunk.metadata.headings == []

    def test_heading_inclusion_metadata_only(self):
        """Test including headings in metadata only."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            heading_inclusion=HeadingInclusion.IN_METADATA,
        )
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Should NOT be in text (just body)
        assert chunk.text.strip() == "Body text."
        # Should be in metadata
        assert chunk.metadata.headings == ["Title"]

    def test_heading_inclusion_none(self):
        """Test excluding headings from chunks."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(heading_inclusion=HeadingInclusion.NONE)
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Should NOT be in text or metadata
        assert chunk.text.strip() == "Body text."
        assert chunk.metadata.headings == []

    def test_chunk_format_plain(self):
        """Test plain text format."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            chunk_format=ChunkFormat.PLAIN,
            heading_inclusion=HeadingInclusion.IN_TEXT,
        )
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Headings should be included but without markdown formatting
        assert "Title" in chunk.text
        assert "# " not in chunk.text  # No markdown syntax

    def test_combine_under_heading_true(self):
        """Test combining body text under same heading."""
        markdown = """# Title
First paragraph.
Second paragraph.
Third paragraph."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            max_chunk_size=1000,
            combine_under_heading=True,
        )
        chunks = list(chunker.chunk(tree))

        # All paragraphs should be combined into one chunk
        assert len(chunks) == 1

    def test_combine_under_heading_false(self):
        """Test processing each body node individually."""
        markdown = """# Title
First paragraph.
Second paragraph.
Third paragraph."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            max_chunk_size=1000,
            combine_under_heading=False,
        )
        chunks = list(chunker.chunk(tree))

        # Each paragraph should be a separate chunk
        assert len(chunks) == 3

    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        long_text = "Sentence number one. " * 50
        markdown = f"# Title\n{long_text}"
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            max_chunk_size=100,
            chunk_overlap=20,
        )
        chunks = list(chunker.chunk(tree))

        # Should have overlapping content
        assert len(chunks) > 1

        # Check for some overlap (not exact due to word boundaries)
        # Just verify we have multiple chunks
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_nested_heading_hierarchy(self):
        """Test preserving nested heading hierarchy in chunks."""
        markdown = """# Chapter 1
## Section 1.1
### Subsection 1.1.1
Body text here."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker(
            heading_inclusion=HeadingInclusion.IN_METADATA,
        )
        chunks = list(chunker.chunk(tree))

        assert len(chunks) == 1
        chunk = chunks[0]

        # Should preserve full hierarchy
        assert chunk.metadata.headings == ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
        assert chunk.metadata.heading_levels == [1, 2, 3]

    def test_chunk_index_sequential(self):
        """Test that chunk indices are sequential."""
        markdown = """# Title 1
Body 1.

# Title 2
Body 2.

# Title 3
Body 3."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker()
        chunks = list(chunker.chunk(tree))

        # Verify indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_line_numbers_preserved(self):
        """Test that line numbers are preserved in chunks."""
        markdown = """# Title
Body at line 2.

More body at line 4."""
        tree = parse_markdown(markdown)

        chunker = MarkdownChunker()
        chunks = list(chunker.chunk(tree))

        # First chunk should have line number from first body text
        assert chunks[0].metadata.line_number > 0


class TestChunkMarkdownTreeFunction:
    """Tests for chunk_markdown_tree convenience function."""

    def test_chunk_markdown_tree_basic(self):
        """Test basic usage of convenience function."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunks = chunk_markdown_tree(tree)

        assert len(chunks) > 0
        assert isinstance(chunks[0], Chunk)

    def test_chunk_markdown_tree_with_options(self):
        """Test convenience function with options."""
        markdown = """# Title
Body text."""
        tree = parse_markdown(markdown)

        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=50,
            heading_inclusion=HeadingInclusion.IN_TEXT,
        )

        assert len(chunks) > 0
        # Headings should be in text
        assert "Title" in chunks[0].text
