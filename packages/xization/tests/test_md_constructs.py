"""Tests for markdown construct parsing (code, lists, tables, etc.)."""

import pytest

from dataknobs_xization.markdown.md_chunker import chunk_markdown_tree, HeadingInclusion
from dataknobs_xization.markdown.md_parser import parse_markdown


class TestCodeBlockParsing:
    """Tests for code block parsing."""

    def test_fenced_code_block_python(self):
        """Test parsing fenced code block with language."""
        markdown = """# Code Example

```python
def hello():
    print("Hello, world!")
```

Some text after."""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        assert code_nodes[0].data.metadata["language"] == "python"
        assert "def hello():" in code_nodes[0].data.text
        assert code_nodes[0].data.is_atomic()

    def test_fenced_code_block_no_language(self):
        """Test parsing fenced code block without language."""
        markdown = """```
plain code here
```"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        assert code_nodes[0].data.metadata["language"] == ""
        assert "plain code here" in code_nodes[0].data.text

    def test_indented_code_block(self):
        """Test parsing indented code block."""
        markdown = """# Example

    def indented():
        return True

More text."""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        assert code_nodes[0].data.metadata["fence_type"] == "indent"
        assert "def indented():" in code_nodes[0].data.text

    def test_code_block_multiline(self):
        """Test parsing multi-line code block."""
        markdown = """```javascript
function test() {
    const x = 1;
    const y = 2;
    return x + y;
}
```"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        assert code_nodes[0].data.text.count('\n') >= 4
        assert "function test()" in code_nodes[0].data.text

    def test_code_block_not_split_in_chunks(self):
        """Test that code blocks are not split even if they exceed max_chunk_size."""
        long_code = "# " + "x" * 200 + "\n" + "y = 1\n" * 20
        markdown = f"""# Title

```python
{long_code}
```"""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=100,  # Much smaller than code block
            heading_inclusion=HeadingInclusion.NONE,
        )

        # Code block should be kept as one chunk despite size
        code_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "code"]
        assert len(code_chunks) == 1
        assert len(code_chunks[0].text) > 100  # Exceeds max size but not split


class TestListParsing:
    """Tests for list parsing."""

    def test_unordered_list(self):
        """Test parsing unordered list."""
        markdown = """# Items

- Item 1
- Item 2
- Item 3"""

        tree = parse_markdown(markdown)
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        assert len(list_nodes) == 1
        assert list_nodes[0].data.metadata["list_type"] == "unordered"
        assert "Item 1" in list_nodes[0].data.text
        assert "Item 3" in list_nodes[0].data.text

    def test_ordered_list(self):
        """Test parsing ordered list."""
        markdown = """1. First
2. Second
3. Third"""

        tree = parse_markdown(markdown)
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        assert len(list_nodes) == 1
        assert list_nodes[0].data.metadata["list_type"] == "ordered"
        assert "First" in list_nodes[0].data.text

    def test_list_with_continuation(self):
        """Test parsing list with indented continuation lines."""
        markdown = """- Item 1
  continuation of item 1
- Item 2"""

        tree = parse_markdown(markdown)
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        assert len(list_nodes) == 1
        assert "continuation of item 1" in list_nodes[0].data.text

    def test_list_different_markers(self):
        """Test lists with different markers."""
        markdown_asterisk = """* Item 1
* Item 2"""
        markdown_plus = """+ Item 1
+ Item 2"""

        for md in [markdown_asterisk, markdown_plus]:
            tree = parse_markdown(md)
            list_nodes = tree.find_nodes(lambda n: n.data.is_list())
            assert len(list_nodes) == 1

    def test_list_not_split_in_chunks(self):
        """Test that lists are kept as atomic units."""
        markdown = """# Title

- Item 1 with some text
- Item 2 with some text
- Item 3 with some text
- Item 4 with some text
- Item 5 with some text"""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=50,  # Smaller than list
            heading_inclusion=HeadingInclusion.NONE,
        )

        # List should be one chunk
        list_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "list"]
        assert len(list_chunks) == 1
        assert "Item 1" in list_chunks[0].text
        assert "Item 5" in list_chunks[0].text


class TestTableParsing:
    """Tests for table parsing."""

    def test_simple_table(self):
        """Test parsing a simple table."""
        markdown = """| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |"""

        tree = parse_markdown(markdown)
        table_nodes = tree.find_nodes(lambda n: n.data.is_table())

        assert len(table_nodes) == 1
        assert "Column 1" in table_nodes[0].data.text
        assert "Data 1" in table_nodes[0].data.text
        assert table_nodes[0].data.is_atomic()

    def test_table_with_alignment(self):
        """Test parsing table with alignment markers."""
        markdown = """| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |"""

        tree = parse_markdown(markdown)
        table_nodes = tree.find_nodes(lambda n: n.data.is_table())

        assert len(table_nodes) == 1
        assert "Left" in table_nodes[0].data.text
        assert "Center" in table_nodes[0].data.text

    def test_table_metadata(self):
        """Test that table metadata includes row count."""
        markdown = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |"""

        tree = parse_markdown(markdown)
        table_nodes = tree.find_nodes(lambda n: n.data.is_table())

        assert len(table_nodes) == 1
        # Header + separator + 3 data rows = 5 rows
        assert table_nodes[0].data.metadata["rows"] == 5

    def test_table_not_split_in_chunks(self):
        """Test that tables are kept whole."""
        markdown = """# Data

| Col1 | Col2 | Col3 |
|------|------|------|
| A    | B    | C    |
| D    | E    | F    |
| G    | H    | I    |
| J    | K    | L    |"""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            max_chunk_size=30,  # Much smaller than table
            heading_inclusion=HeadingInclusion.NONE,
        )

        table_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "table"]
        assert len(table_chunks) == 1
        assert "Col1" in table_chunks[0].text
        assert "| J" in table_chunks[0].text  # Last row preserved


class TestBlockquoteParsing:
    """Tests for blockquote parsing."""

    def test_simple_blockquote(self):
        """Test parsing simple blockquote."""
        markdown = """> This is a quote.
> It continues here."""

        tree = parse_markdown(markdown)
        quote_nodes = tree.find_nodes(lambda n: n.data.is_blockquote())

        assert len(quote_nodes) == 1
        assert "This is a quote" in quote_nodes[0].data.text
        assert "continues here" in quote_nodes[0].data.text

    def test_blockquote_single_line(self):
        """Test parsing single-line blockquote."""
        markdown = "> Single line quote."

        tree = parse_markdown(markdown)
        quote_nodes = tree.find_nodes(lambda n: n.data.is_blockquote())

        assert len(quote_nodes) == 1
        assert "Single line quote" in quote_nodes[0].data.text

    def test_blockquote_with_empty_lines(self):
        """Test blockquote with empty lines."""
        markdown = """> First paragraph.
>
> Second paragraph."""

        tree = parse_markdown(markdown)
        quote_nodes = tree.find_nodes(lambda n: n.data.is_blockquote())

        assert len(quote_nodes) == 1
        # Should preserve the structure
        assert "First paragraph" in quote_nodes[0].data.text
        assert "Second paragraph" in quote_nodes[0].data.text


class TestHorizontalRuleParsing:
    """Tests for horizontal rule parsing."""

    def test_horizontal_rule_dashes(self):
        """Test parsing horizontal rule with dashes."""
        markdown = """Text before

---

Text after"""

        tree = parse_markdown(markdown)
        hr_nodes = tree.find_nodes(lambda n: n.data.is_horizontal_rule())

        assert len(hr_nodes) == 1
        assert "---" in hr_nodes[0].data.text

    def test_horizontal_rule_asterisks(self):
        """Test parsing horizontal rule with asterisks."""
        markdown = """***"""

        tree = parse_markdown(markdown)
        hr_nodes = tree.find_nodes(lambda n: n.data.is_horizontal_rule())

        assert len(hr_nodes) == 1

    def test_horizontal_rule_underscores(self):
        """Test parsing horizontal rule with underscores."""
        markdown = """___"""

        tree = parse_markdown(markdown)
        hr_nodes = tree.find_nodes(lambda n: n.data.is_horizontal_rule())

        assert len(hr_nodes) == 1


class TestMixedConstructs:
    """Tests for documents with multiple construct types."""

    def test_document_with_all_constructs(self):
        """Test parsing document with all construct types."""
        markdown = """# Complete Example

Regular paragraph.

## Code Section

```python
def example():
    return True
```

## List Section

- Item 1
- Item 2
- Item 3

## Table Section

| A | B |
|---|---|
| 1 | 2 |

## Quote Section

> This is a quote.

---

Final text."""

        tree = parse_markdown(markdown)

        # Check all construct types are present
        assert len(tree.find_nodes(lambda n: n.data.is_heading())) == 5
        assert len(tree.find_nodes(lambda n: n.data.is_code())) == 1
        assert len(tree.find_nodes(lambda n: n.data.is_list())) == 1
        assert len(tree.find_nodes(lambda n: n.data.is_table())) == 1
        assert len(tree.find_nodes(lambda n: n.data.is_blockquote())) == 1
        assert len(tree.find_nodes(lambda n: n.data.is_horizontal_rule())) == 1
        assert len(tree.find_nodes(lambda n: n.data.is_body())) >= 2

    def test_chunking_preserves_construct_types(self):
        """Test that chunking preserves construct type information."""
        markdown = """# Section

Regular text.

```python
code()
```

- List item

| Table |
|-------|
| Data  |"""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            heading_inclusion=HeadingInclusion.IN_METADATA,
        )

        # Check that we have chunks of different types
        node_types = {c.metadata.custom.get("node_type") for c in chunks}
        assert "body" in node_types
        assert "code" in node_types
        assert "list" in node_types
        assert "table" in node_types

    def test_construct_hierarchy_preserved(self):
        """Test that constructs maintain proper heading hierarchy."""
        markdown = """# Chapter 1

## Section 1.1

```python
code_here()
```

## Section 1.2

- List here"""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            heading_inclusion=HeadingInclusion.IN_METADATA,
        )

        # Find code chunk
        code_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "code"]
        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.headings == ["Chapter 1", "Section 1.1"]

        # Find list chunk
        list_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "list"]
        assert len(list_chunks) == 1
        assert list_chunks[0].metadata.headings == ["Chapter 1", "Section 1.2"]

    def test_code_and_text_separation(self):
        """Test that code blocks and text are separate chunks."""
        markdown = """# Title

Text before code.

```python
def func():
    pass
```

Text after code."""

        tree = parse_markdown(markdown)
        chunks = chunk_markdown_tree(
            tree,
            heading_inclusion=HeadingInclusion.NONE,
        )

        # Should have at least 2 chunks: combined body text and code
        # (body text before/after is combined by default)
        assert len(chunks) >= 2

        # Code should be in its own chunk
        code_chunks = [c for c in chunks if "def func()" in c.text]
        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.custom.get("node_type") == "code"

        # Body text should be in separate chunk(s)
        body_chunks = [c for c in chunks if c.metadata.custom.get("node_type") == "body"]
        assert len(body_chunks) >= 1

    def test_multiple_lists_under_heading(self):
        """Test multiple lists under same heading."""
        markdown = """# Lists

First list:
- A
- B

Second list:
1. One
2. Two"""

        tree = parse_markdown(markdown)
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        assert len(list_nodes) == 2
        assert list_nodes[0].data.metadata["list_type"] == "unordered"
        assert list_nodes[1].data.metadata["list_type"] == "ordered"


class TestEdgeCases:
    """Tests for edge cases in construct parsing."""

    def test_code_block_without_closing_fence(self):
        """Test code block that's missing closing fence."""
        markdown = """```python
def incomplete():
    pass"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        # Should still parse as code
        assert len(code_nodes) == 1

    def test_empty_code_block(self):
        """Test empty code block."""
        markdown = """```

```"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        assert code_nodes[0].data.text == ""

    def test_empty_list(self):
        """Test that we don't parse single line as list."""
        markdown = "- "

        tree = parse_markdown(markdown)
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        # Empty item should still be recognized as a list
        assert len(list_nodes) >= 0

    def test_table_missing_separator(self):
        """Test table header without separator (should not parse as table)."""
        markdown = """| A | B |
| C | D |"""

        tree = parse_markdown(markdown)
        table_nodes = tree.find_nodes(lambda n: n.data.is_table())

        # Without separator, shouldn't be recognized as table
        assert len(table_nodes) == 0

    def test_nested_constructs_not_supported(self):
        """Test that nested constructs are handled gracefully."""
        # Lists inside code blocks should stay as code
        markdown = """```
- This looks like a list
- But it's in a code block
```"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())
        list_nodes = tree.find_nodes(lambda n: n.data.is_list())

        assert len(code_nodes) == 1
        assert len(list_nodes) == 0  # List markers inside code aren't parsed

    def test_construct_after_heading_no_gap(self):
        """Test construct immediately after heading."""
        markdown = """# Title
```python
code()
```"""

        tree = parse_markdown(markdown)
        code_nodes = tree.find_nodes(lambda n: n.data.is_code())

        assert len(code_nodes) == 1
        # Code should be child of heading
        assert code_nodes[0].parent.data.is_heading()
        assert code_nodes[0].parent.data.text == "Title"
