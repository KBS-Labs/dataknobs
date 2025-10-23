"""Tests for markdown parser module."""

import pytest

from dataknobs_xization.markdown.md_parser import MarkdownNode, MarkdownParser, parse_markdown


class TestMarkdownNode:
    """Tests for MarkdownNode class."""

    def test_heading_node_creation(self):
        """Test creating a heading node."""
        node = MarkdownNode(text="Introduction", level=1, node_type="heading", line_number=1)
        assert node.text == "Introduction"
        assert node.level == 1
        assert node.node_type == "heading"
        assert node.is_heading()
        assert not node.is_body()

    def test_body_node_creation(self):
        """Test creating a body text node."""
        node = MarkdownNode(text="This is body text.", level=0, node_type="body", line_number=5)
        assert node.text == "This is body text."
        assert node.level == 0
        assert node.node_type == "body"
        assert node.is_body()
        assert not node.is_heading()

    def test_node_string_representation(self):
        """Test string representation of nodes."""
        heading = MarkdownNode(text="Title", level=2, node_type="heading")
        body = MarkdownNode(text="Content", level=0, node_type="body")
        assert str(heading) == "H2: Title"
        assert str(body) == "Content"


class TestMarkdownParser:
    """Tests for MarkdownParser class."""

    def test_simple_heading_parsing(self):
        """Test parsing simple headings."""
        markdown = """# Heading 1
## Heading 2
### Heading 3"""
        tree = parse_markdown(markdown)

        # Check that headings were parsed correctly
        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 3
        assert headings[0].data.text == "Heading 1"
        assert headings[0].data.level == 1
        assert headings[1].data.text == "Heading 2"
        assert headings[1].data.level == 2

    def test_heading_hierarchy(self):
        """Test that heading hierarchy is preserved in tree."""
        markdown = """# Level 1
## Level 2
### Level 3
## Another Level 2"""
        tree = parse_markdown(markdown)

        # Find Level 1 heading
        level1 = tree.find_nodes(
            lambda n: n.data.is_heading() and n.data.level == 1,
            only_first=True,
        )[0]

        # Level 1 should have 2 children (both Level 2 headings)
        assert level1.num_children == 2

        # First child should be Level 2 with one child (Level 3)
        level2_first = level1.children[0]
        assert level2_first.data.level == 2
        assert level2_first.data.text == "Level 2"
        assert level2_first.num_children == 1

        # Second child should be Level 2 with no children
        level2_second = level1.children[1]
        assert level2_second.data.level == 2
        assert level2_second.data.text == "Another Level 2"
        assert level2_second.num_children == 0

    def test_body_text_parsing(self):
        """Test parsing body text under headings."""
        markdown = """# Title
This is some body text.
More body text.

## Subtitle
Nested body text."""
        tree = parse_markdown(markdown)

        # Find body nodes
        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        assert len(body_nodes) == 3

        # Check body text content
        body_texts = [node.data.text for node in body_nodes]
        assert "This is some body text." in body_texts
        assert "More body text." in body_texts
        assert "Nested body text." in body_texts

    def test_empty_lines_filtering(self):
        """Test that empty lines are filtered by default."""
        markdown = """# Title

Body text.

More text."""
        tree = parse_markdown(markdown, preserve_empty_lines=False)

        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        # Should only have 2 body nodes (empty lines filtered)
        assert len(body_nodes) == 2

    def test_empty_lines_preservation(self):
        """Test preserving empty lines."""
        markdown = """# Title

Body text."""
        tree = parse_markdown(markdown, preserve_empty_lines=True)

        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        # Should have 2 body nodes (including empty line)
        assert len(body_nodes) == 2

    def test_max_line_length_splitting(self):
        """Test that long lines are split appropriately."""
        long_text = "This is a very long sentence. " * 20
        markdown = f"# Title\n{long_text}"

        parser = MarkdownParser(max_line_length=100)
        tree = parser.parse(markdown)

        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        # Should be split into multiple nodes
        assert len(body_nodes) > 1
        # Each should be under max length
        for node in body_nodes:
            assert len(node.data.text) <= 100 or len(node.data.text.split()) == 1

    def test_heading_levels_1_to_6(self):
        """Test parsing all heading levels."""
        markdown = """# H1
## H2
### H3
#### H4
##### H5
###### H6"""
        tree = parse_markdown(markdown)

        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 6

        for i, heading in enumerate(headings, 1):
            assert heading.data.level == i

    def test_complex_document_structure(self):
        """Test parsing a complex document with multiple levels."""
        markdown = """# Introduction
This is the introduction.

## Background
Background information here.

### Historical Context
Some history.

## Methods
Description of methods.

### Data Collection
How data was collected.

### Analysis
How data was analyzed.

# Results
The results section.

## Findings
What we found."""

        tree = parse_markdown(markdown)

        # Verify structure
        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 8

        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        assert len(body_nodes) == 8

    def test_heading_with_special_characters(self):
        """Test headings with special characters."""
        markdown = """# Title with *emphasis* and **bold**
## Title with `code`
### Title with [link](url)"""
        tree = parse_markdown(markdown)

        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 3
        # Text should preserve markdown formatting
        assert "**bold**" in headings[0].data.text
        assert "`code`" in headings[1].data.text

    def test_parsing_from_file_like_object(self):
        """Test parsing from file-like object."""
        from io import StringIO

        markdown = """# Title
Body text."""
        file_obj = StringIO(markdown)

        tree = parse_markdown(file_obj)
        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 1
        assert headings[0].data.text == "Title"

    def test_parsing_from_iterator(self):
        """Test parsing from line iterator."""
        lines = ["# Title", "Body text."]
        tree = parse_markdown(iter(lines))

        headings = tree.find_nodes(lambda n: n.data.is_heading())
        assert len(headings) == 1
        assert headings[0].data.text == "Title"


class TestParseMarkdownFunction:
    """Tests for parse_markdown convenience function."""

    def test_parse_simple_markdown(self):
        """Test parsing simple markdown string."""
        markdown = "# Title\nBody text."
        tree = parse_markdown(markdown)

        assert tree is not None
        assert tree.has_children()

    def test_parse_with_options(self):
        """Test parsing with options."""
        markdown = "# Title\n\nBody text."
        tree = parse_markdown(markdown, preserve_empty_lines=True)

        body_nodes = tree.find_nodes(lambda n: n.data.is_body())
        assert len(body_nodes) == 2  # includes empty line
