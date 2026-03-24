"""Tests for the HTMLConverter class."""

from pathlib import Path

import pytest

from dataknobs_xization import HTMLConverter, HTMLConverterConfig, html_to_markdown


class TestHTMLConverterInit:
    """Test HTMLConverter initialization."""

    def test_default_init(self):
        """Test default initialization values."""
        converter = HTMLConverter()
        assert converter.config.base_heading_level == 1
        assert converter.config.include_links is True
        assert converter.config.strip_nav is True
        assert converter.config.strip_scripts is True
        assert converter.config.preserve_code_blocks is True

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = HTMLConverterConfig(base_heading_level=2, include_links=False)
        converter = HTMLConverter(config=config)
        assert converter.config.base_heading_level == 2
        assert converter.config.include_links is False

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        converter = HTMLConverter(base_heading_level=3, wrap_width=80)
        assert converter.config.base_heading_level == 3
        assert converter.config.wrap_width == 80

    def test_invalid_content_type(self):
        """Test that invalid content type raises TypeError."""
        converter = HTMLConverter()
        with pytest.raises(TypeError, match="content must be str or Path"):
            converter.convert(12345)  # type: ignore[arg-type]

    def test_invalid_base_heading_level(self):
        """Test that invalid base_heading_level raises ValueError."""
        with pytest.raises(ValueError, match="base_heading_level must be between 1 and 6"):
            HTMLConverterConfig(base_heading_level=0)
        with pytest.raises(ValueError, match="base_heading_level must be between 1 and 6"):
            HTMLConverterConfig(base_heading_level=7)

    def test_invalid_link_style(self):
        """Test that invalid link_style raises ValueError."""
        with pytest.raises(ValueError, match="link_style must be one of"):
            HTMLConverterConfig(link_style="unknown")


class TestStandardHTMLHeadings:
    """Test standard HTML heading conversion."""

    def test_h1_heading(self):
        """Test h1 converts to # heading."""
        md = html_to_markdown("<h1>Title</h1>")
        assert "# Title" in md

    def test_h2_heading(self):
        """Test h2 converts to ## heading."""
        md = html_to_markdown("<h2>Section</h2>")
        assert "## Section" in md

    def test_h3_heading(self):
        """Test h3 converts to ### heading."""
        md = html_to_markdown("<h3>Subsection</h3>")
        assert "### Subsection" in md

    def test_heading_level_offset(self):
        """Test base_heading_level offsets all headings."""
        md = html_to_markdown("<h1>Title</h1><h2>Sub</h2>", base_heading_level=2)
        assert "## Title" in md
        assert "### Sub" in md

    def test_multiple_headings(self):
        """Test multiple headings in sequence."""
        html = "<h1>One</h1><h2>Two</h2><h3>Three</h3>"
        md = html_to_markdown(html)
        assert "# One" in md
        assert "## Two" in md
        assert "### Three" in md

    def test_heading_level_clamped_at_6(self):
        """Test heading levels are clamped to a maximum of 6."""
        converter = HTMLConverter(base_heading_level=5)
        md = converter.convert("<h1>A</h1><h2>B</h2><h3>C</h3>")
        # h1 + offset 5 -> 5, h2 + offset 5 -> 6, h3 + offset 5 -> clamped to 6
        assert "##### A" in md
        assert "###### B" in md
        assert "###### C" in md
        assert "#######" not in md


class TestStandardHTMLParagraphs:
    """Test paragraph conversion."""

    def test_simple_paragraph(self):
        """Test basic paragraph text."""
        md = html_to_markdown("<p>Hello world.</p>")
        assert "Hello world." in md

    def test_multiple_paragraphs(self):
        """Test paragraphs are separated by blank lines."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        md = html_to_markdown(html)
        assert "First paragraph." in md
        assert "Second paragraph." in md
        # Should have blank line between
        lines = md.split("\n")
        first_idx = next(i for i, line in enumerate(lines) if "First" in line)
        second_idx = next(i for i, line in enumerate(lines) if "Second" in line)
        assert second_idx > first_idx + 1

    def test_paragraph_with_inline_markup(self):
        """Test paragraph with bold/italic/code."""
        html = "<p>This is <strong>bold</strong> and <em>italic</em> and <code>code</code>.</p>"
        md = html_to_markdown(html)
        assert "**bold**" in md
        assert "*italic*" in md
        assert "`code`" in md


class TestStandardHTMLLinks:
    """Test link conversion."""

    def test_inline_link(self):
        """Test anchor tag converts to markdown link."""
        html = '<p>See <a href="https://example.com">example</a>.</p>'
        md = html_to_markdown(html)
        assert "[example](https://example.com)" in md

    def test_link_disabled(self):
        """Test links can be disabled."""
        html = '<p>See <a href="https://example.com">example</a>.</p>'
        md = html_to_markdown(html, include_links=False)
        assert "[example]" not in md
        assert "example" in md

    def test_link_text_style(self):
        """Test link_style=text strips URLs."""
        converter = HTMLConverter(link_style="text")
        md = converter.convert('<p><a href="https://x.com">click</a></p>')
        assert "click" in md
        assert "https://x.com" not in md

    def test_link_reference_style(self):
        """Test link_style=reference produces reference-style links."""
        converter = HTMLConverter(link_style="reference")
        html = (
            '<p>See <a href="https://example.com">example</a>'
            ' and <a href="https://other.com">other</a>.</p>'
        )
        md = converter.convert(html)
        assert "[example][1]" in md
        assert "[other][2]" in md
        assert "[1]: https://example.com" in md
        assert "[2]: https://other.com" in md

    def test_link_reference_style_no_inline_urls(self):
        """Test reference-style links don't include inline URLs."""
        converter = HTMLConverter(link_style="reference")
        md = converter.convert('<p><a href="https://x.com">click</a></p>')
        assert "(https://x.com)" not in md
        assert "[1]: https://x.com" in md


class TestStandardHTMLLists:
    """Test list conversion."""

    def test_unordered_list(self):
        """Test ul converts to dash-prefixed items."""
        html = "<ul><li>First</li><li>Second</li><li>Third</li></ul>"
        md = html_to_markdown(html)
        assert "- First" in md
        assert "- Second" in md
        assert "- Third" in md

    def test_ordered_list(self):
        """Test ol converts to numbered items."""
        html = "<ol><li>Alpha</li><li>Beta</li></ol>"
        md = html_to_markdown(html)
        assert "1. Alpha" in md
        assert "2. Beta" in md

    def test_nested_list(self):
        """Test nested lists use indentation."""
        html = "<ul><li>Outer<ul><li>Inner</li></ul></li></ul>"
        md = html_to_markdown(html)
        assert "- Outer" in md
        assert "  - Inner" in md


class TestStandardHTMLCodeBlocks:
    """Test code block conversion."""

    def test_pre_block(self):
        """Test pre tag converts to fenced code block."""
        html = "<pre>function hello() {\n  return 'world';\n}</pre>"
        md = html_to_markdown(html)
        assert "```" in md
        assert "function hello()" in md

    def test_pre_with_language(self):
        """Test pre>code with language class."""
        html = '<pre><code class="language-python">print("hello")</code></pre>'
        md = html_to_markdown(html)
        assert "```python" in md
        assert 'print("hello")' in md

    def test_code_blocks_disabled(self):
        """Test code blocks can be stripped."""
        converter = HTMLConverter(preserve_code_blocks=False)
        md = converter.convert("<pre>some code</pre>")
        assert "```" not in md


class TestStandardHTMLTables:
    """Test table conversion."""

    def test_simple_table(self):
        """Test basic table with header."""
        html = """
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>foo</td><td>bar</td></tr>
            <tr><td>baz</td><td>qux</td></tr>
        </table>
        """
        md = html_to_markdown(html)
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| foo | bar |" in md
        assert "| baz | qux |" in md

    def test_table_no_header(self):
        """Test table without th elements."""
        html = """
        <table>
            <tr><td>a</td><td>b</td></tr>
            <tr><td>c</td><td>d</td></tr>
        </table>
        """
        md = html_to_markdown(html)
        assert "| a | b |" in md
        assert "---" in md


class TestStandardHTMLBlockquotes:
    """Test blockquote conversion."""

    def test_blockquote(self):
        """Test blockquote converts to > prefix."""
        html = "<blockquote><p>Quoted text.</p></blockquote>"
        md = html_to_markdown(html)
        assert "> Quoted text." in md


class TestStandardHTMLMisc:
    """Test miscellaneous HTML elements."""

    def test_horizontal_rule(self):
        """Test hr converts to ---."""
        html = "<p>Above</p><hr><p>Below</p>"
        md = html_to_markdown(html)
        assert "---" in md

    def test_definition_list(self):
        """Test dl/dt/dd conversion."""
        html = "<dl><dt>Term</dt><dd>Definition</dd></dl>"
        md = html_to_markdown(html)
        assert "**Term**" in md
        assert ": Definition" in md

    def test_strip_scripts(self):
        """Test script and style tags are removed."""
        html = "<p>Content</p><script>alert('x')</script><style>.x{}</style>"
        md = html_to_markdown(html)
        assert "Content" in md
        assert "alert" not in md
        assert ".x{}" not in md

    def test_strip_nav(self):
        """Test navigation elements are removed."""
        html = "<nav>Menu</nav><p>Content</p><footer>Footer</footer>"
        md = html_to_markdown(html)
        assert "Content" in md
        assert "Menu" not in md
        assert "Footer" not in md

    def test_nested_divs(self):
        """Test content inside nested divs is extracted."""
        html = "<div><div><p>Deep content.</p></div></div>"
        md = html_to_markdown(html)
        assert "Deep content." in md

    def test_images(self):
        """Test image conversion."""
        html = '<p>See <img src="pic.png" alt="a picture">.</p>'
        md = html_to_markdown(html)
        assert "![a picture](pic.png)" in md

    def test_images_stripped(self):
        """Test images can be stripped."""
        converter = HTMLConverter(strip_images=True)
        md = converter.convert('<p><img src="pic.png" alt="pic"></p>')
        assert "![" not in md


class TestStandardHTMLFullDocument:
    """Test full HTML document conversion."""

    def test_full_document(self):
        """Test converting a complete HTML document."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <main>
                <h1>Main Title</h1>
                <p>Introduction paragraph.</p>
                <h2>Section One</h2>
                <p>Content with <strong>bold</strong>.</p>
                <ul>
                    <li>Item A</li>
                    <li>Item B</li>
                </ul>
            </main>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        md = html_to_markdown(html)
        assert "# Main Title" in md
        assert "## Section One" in md
        assert "Introduction paragraph." in md
        assert "**bold**" in md
        assert "- Item A" in md
        assert "Home" not in md  # nav stripped
        assert "Copyright" not in md  # footer stripped

    def test_title_override(self):
        """Test explicit title replaces document h1."""
        html = "<h1>Original</h1><p>Content.</p>"
        md = html_to_markdown(html, title="Override Title")
        assert "# Override Title" in md
        assert "Original" not in md  # h1 removed to avoid duplication


class TestRFCDetection:
    """Test IETF RFC format detection."""

    def test_detects_rfcmarkup_class(self):
        """Test detection via rfcmarkup div class."""
        html = '<div class="rfcmarkup"><pre>content</pre></div>'
        converter = HTMLConverter()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        assert converter._is_rfc_markup(soup) is True

    def test_detects_heading_spans_in_pre(self):
        """Test detection via h2 span inside pre."""
        html = '<pre><span class="h2">1. Introduction</span></pre>'
        converter = HTMLConverter()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        assert converter._is_rfc_markup(soup) is True

    def test_standard_html_not_detected_as_rfc(self):
        """Test standard HTML is not detected as RFC."""
        html = "<h1>Title</h1><p>Content.</p>"
        converter = HTMLConverter()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        assert converter._is_rfc_markup(soup) is False


class TestRFCConversion:
    """Test IETF RFC markup conversion."""

    def _rfc_html(self, body_content: str) -> str:
        """Wrap content in minimal RFC markup structure."""
        return f"""
        <div class="rfcmarkup">
            <div class="rfcmarkup"><pre>{body_content}</pre></div>
        </div>
        """

    def test_rfc_heading_extraction(self):
        """Test RFC span headings convert to markdown headings."""
        html = self._rfc_html("""
<span class="h1">The OAuth 2.0 Authorization Framework</span>

<span class="h2"><a class="selflink" id="section-1" href="#section-1">1</a>.  Introduction</span>

   OAuth addresses issues by introducing an authorization layer.
""")
        md = html_to_markdown(html)
        assert "# The OAuth 2.0 Authorization Framework" in md
        assert "## 1. Introduction" in md
        assert "OAuth addresses issues" in md

    def test_rfc_h3_heading(self):
        """Test h3 RFC headings convert correctly."""
        html = self._rfc_html("""
<span class="h1">Title</span>

<span class="h3"><a class="selflink" id="section-1.1" href="#section-1.1">1.1</a>.  Roles</span>

   OAuth defines four roles.
""")
        md = html_to_markdown(html)
        assert "### 1.1. Roles" in md
        assert "OAuth defines four roles." in md

    def test_rfc_page_footer_stripped(self):
        """Test RFC page footers are removed."""
        grey_footer = (
            "Hardt                        "
            "Standards Track                    [Page 4]"
        )
        grey_header = (
            '<a href="/doc/html/rfc6749">RFC 6749</a>'
            "                        OAuth 2.0"
            "                   October 2012"
        )
        html = self._rfc_html(
            '<span class="h1">Title</span>\n\n'
            '<span class="h2">'
            '<a id="section-1" href="#section-1">1</a>'
            ".  Introduction</span>\n\n"
            "   Content here.\n\n"
            f'<span class="grey">{grey_footer}</span></pre>\n'
            "<hr class='noprint'/>"
            '<pre class=\'newpage\'>'
            '<span id="page-5" ></span>\n'
            f'<span class="grey">{grey_header}</span>\n\n'
            "   More content here.\n"
        )
        md = html_to_markdown(html)
        assert "Content here." in md
        assert "More content here." in md
        assert "[Page 4]" not in md
        assert "Standards Track" not in md

    def test_rfc_list_items(self):
        """Test RFC-style bullet list (o  prefix) conversion."""
        html = self._rfc_html("""
<span class="h1">Title</span>

<span class="h2"><a id="section-1" href="#section-1">1</a>.  Overview</span>

   Key points:

   o  First item in the list that spans
      multiple lines.

   o  Second item.

   o  Third item.
""")
        md = html_to_markdown(html)
        assert "- First item in the list that spans multiple lines." in md
        assert "- Second item." in md
        assert "- Third item." in md

    def test_rfc_links_preserved(self):
        """Test external RFC links are preserved."""
        html = self._rfc_html("""
<span class="h1">Title</span>

<span class="h2"><a id="section-1" href="#section-1">1</a>.  Info</span>

   See <a href="https://www.rfc-editor.org/info/rfc6749">https://www.rfc-editor.org/info/rfc6749</a>.
""")
        md = html_to_markdown(html)
        assert "rfc-editor.org" in md

    def test_rfc_internal_links_text_only(self):
        """Test internal RFC links are rendered as text only."""
        html = self._rfc_html("""
<span class="h1">Title</span>

<span class="h2"><a id="section-1" href="#section-1">1</a>.  Info</span>

   As described in <a href="/doc/html/rfc5849">RFC 5849</a>.
""")
        md = html_to_markdown(html)
        assert "RFC 5849" in md
        # Internal links should not be markdown links
        assert "[RFC 5849](/doc/html/rfc5849)" not in md

    def test_rfc_toc_stripped(self):
        """Test Table of Contents is stripped from output."""
        html = self._rfc_html("""
<span class="h1">Title</span>

Table of Contents

   <a href="#section-1">1</a>. Introduction ....<a href="#page-4">4</a>
      <a href="#section-1.1">1.1</a>. Roles .....<a href="#page-6">6</a>

<span class="h2"><a id="section-1" href="#section-1">1</a>.  Introduction</span>

   Real content here.
""")
        md = html_to_markdown(html)
        assert "Real content here." in md
        # ToC entries should not appear
        assert "......" not in md

    def test_rfc_ascii_art_as_code_block(self):
        """Test ASCII art diagrams become code blocks."""
        html = self._rfc_html("""
<span class="h1">Title</span>

<span class="h2"><a id="section-1" href="#section-1">1</a>.  Flow</span>

     +--------+                               +---------------+
     |        |--( A )- Authorization Request-&gt;|   Resource    |
     |        |                               |     Owner     |
     |        |&lt;-(B)-- Authorization Grant ---|               |
     |        |                               +---------------+
     | Client |
     +--------+
""")
        md = html_to_markdown(html)
        assert "```" in md
        assert "+--------+" in md


class TestFrontmatter:
    """Test YAML frontmatter generation."""

    def test_frontmatter(self):
        """Test frontmatter is prepended to output."""
        md = html_to_markdown(
            "<h1>Title</h1><p>Content.</p>",
            frontmatter={"source": "https://example.com", "type": "article"},
        )
        assert md.startswith("---\n")
        assert "source: https://example.com" in md
        assert "type: article" in md
        assert "---\n" in md
        # Content follows frontmatter
        assert "# Title" in md

    def test_frontmatter_with_list(self):
        """Test frontmatter with list values."""
        md = html_to_markdown(
            "<p>Content.</p>",
            frontmatter={"tags": ["oauth", "security"]},
        )
        assert "  - oauth" in md
        assert "  - security" in md


class TestFileInput:
    """Test reading HTML from files."""

    def test_read_from_path(self, tmp_path: Path):
        """Test converting HTML from a file path."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<h1>File Title</h1><p>File content.</p>")
        md = html_to_markdown(html_file)
        assert "# File Title" in md
        assert "File content." in md

    def test_read_from_path_object(self, tmp_path: Path):
        """Test converting HTML from a Path object."""
        html_file = tmp_path / "section.html"
        html_file.write_text("<h2>Section</h2><p>Text.</p>")
        converter = HTMLConverter()
        md = converter.convert(html_file)
        assert "## Section" in md


class TestConvenienceFunction:
    """Test the html_to_markdown convenience function."""

    def test_basic_conversion(self):
        """Test convenience function produces correct output."""
        md = html_to_markdown("<h1>Title</h1><p>Hello.</p>")
        assert "# Title" in md
        assert "Hello." in md

    def test_with_title(self):
        """Test convenience function with explicit title."""
        md = html_to_markdown("<p>Content.</p>", title="My Document")
        assert "# My Document" in md
        assert "Content." in md

    def test_heading_level(self):
        """Test convenience function with heading level override."""
        md = html_to_markdown("<h1>Title</h1>", base_heading_level=2)
        assert "## Title" in md


class TestContentTransformerHTML:
    """Test HTML support through ContentTransformer."""

    def test_transform_html_method(self):
        """Test ContentTransformer.transform_html() works."""
        from dataknobs_xization import ContentTransformer

        transformer = ContentTransformer()
        md = transformer.transform_html("<h2>Section</h2><p>Text.</p>")
        assert "Section" in md
        assert "Text." in md

    def test_transform_dispatch_html(self):
        """Test ContentTransformer.transform() dispatches to HTML."""
        from dataknobs_xization import ContentTransformer

        transformer = ContentTransformer()
        md = transformer.transform("<p>Content.</p>", format="html")
        assert "Content." in md

    def test_transform_unsupported_format(self):
        """Test unsupported format raises ValueError."""
        from dataknobs_xization import ContentTransformer

        transformer = ContentTransformer()
        with pytest.raises(ValueError, match="Unsupported format"):
            transformer.transform("data", format="xml")


class TestWhitespaceNormalization:
    """Test output whitespace normalization."""

    def test_no_excessive_blank_lines(self):
        """Test that output doesn't have more than 2 consecutive blank lines."""
        html = "<h1>A</h1><p>B</p><h2>C</h2><p>D</p>"
        md = html_to_markdown(html)
        assert "\n\n\n\n" not in md

    def test_trailing_newline(self):
        """Test output ends with a single newline."""
        md = html_to_markdown("<p>Content.</p>")
        assert md.endswith("\n")
        assert not md.endswith("\n\n")
