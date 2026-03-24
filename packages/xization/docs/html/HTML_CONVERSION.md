# HTML to Markdown Conversion

The HTML conversion module converts HTML documents into well-structured markdown suitable for RAG ingestion and chunking with `MarkdownChunker`. It supports both standard HTML (semantic tags) and IETF RFC markup (pre-formatted text with span-based headings).

## Overview

When building RAG systems, HTML content from web pages, documentation sites, or standards bodies needs to be converted into markdown before chunking. The `HTMLConverter` class handles this conversion, auto-detecting the document format and applying the appropriate strategy.

### Supported Formats

| Format | Detection | Description |
|--------|-----------|-------------|
| **Standard HTML** | Default | Semantic tags: `<h1>`-`<h6>`, `<p>`, `<ul>`, `<ol>`, `<table>`, `<pre>`, `<blockquote>`, `<dl>` |
| **IETF RFC markup** | Auto-detected | Pre-formatted text with `<span class="h1">`-`<span class="h6">` headings inside `<pre>` tags, as served by datatracker.ietf.org |

## Quick Start

```python
from dataknobs_xization import HTMLConverter, html_to_markdown

# Simple conversion
markdown = html_to_markdown("<h1>Title</h1><p>Hello world.</p>")

# With configuration
converter = HTMLConverter(base_heading_level=2, wrap_width=80)
markdown = converter.convert(html_string)

# From a file
from pathlib import Path
markdown = converter.convert(Path("document.html"))

# RFC documents are auto-detected
markdown = converter.convert(rfc_html, title="RFC 6749")
```

## HTMLConverter Class

### Initialization

```python
from dataknobs_xization import HTMLConverter, HTMLConverterConfig

# Using keyword arguments
converter = HTMLConverter(
    base_heading_level=1,       # Minimum heading level in output (1 = #)
    include_links=True,         # Preserve hyperlinks as markdown links
    strip_nav=True,             # Remove <nav>, <header>, <footer>
    strip_scripts=True,         # Remove <script> and <style>
    preserve_code_blocks=True,  # Keep <pre>/<code> as fenced blocks
    link_style="inline",        # "inline", "reference", or "text"
    strip_images=False,         # Whether to remove images
    wrap_width=0,               # Wrap paragraph text (0 = no wrap)
    frontmatter=None,           # Optional YAML frontmatter dict
)

# Or using a config object
config = HTMLConverterConfig(base_heading_level=2, include_links=False)
converter = HTMLConverter(config=config)
```

### Converting HTML

```python
# From a string
markdown = converter.convert("<h1>Title</h1><p>Content.</p>")

# From a file path
markdown = converter.convert(Path("page.html"))

# With an explicit title (prepended as top-level heading)
markdown = converter.convert(html_string, title="My Document")
```

## Standard HTML Conversion

The converter handles all common semantic HTML elements:

### Headings

```html
<h1>Title</h1>
<h2>Section</h2>
<h3>Subsection</h3>
```

Converts to:

```markdown
# Title

## Section

### Subsection
```

Use `base_heading_level` to offset all headings (e.g., `base_heading_level=2` turns `<h1>` into `##`).

### Inline Markup

| HTML | Markdown |
|------|----------|
| `<strong>bold</strong>` / `<b>bold</b>` | `**bold**` |
| `<em>italic</em>` / `<i>italic</i>` | `*italic*` |
| `<code>code</code>` | `` `code` `` |
| `<a href="url">text</a>` | `[text](url)` |
| `<img src="url" alt="alt">` | `![alt](url)` |

### Lists

Unordered and ordered lists, including nested lists:

```html
<ul>
  <li>First</li>
  <li>Second
    <ul><li>Nested</li></ul>
  </li>
</ul>
```

Becomes:

```markdown
- First
- Second
  - Nested
```

### Tables

```html
<table>
  <tr><th>Name</th><th>Value</th></tr>
  <tr><td>foo</td><td>bar</td></tr>
</table>
```

Becomes:

```markdown
| Name | Value |
| --- | --- |
| foo | bar |
```

### Code Blocks

`<pre>` tags convert to fenced code blocks. Language detection uses `<code class="language-python">`:

```html
<pre><code class="language-python">print("hello")</code></pre>
```

Becomes:

````markdown
```python
print("hello")
```
````

### Other Elements

- **Blockquotes**: `<blockquote>` becomes `>` prefixed lines
- **Definition lists**: `<dl>/<dt>/<dd>` becomes bold terms with `: definition` lines
- **Horizontal rules**: `<hr>` becomes `---`
- **Figures**: `<figcaption>` text rendered in italics

### Element Stripping

By default, these elements are removed:

- `<script>`, `<style>`, `<noscript>`, `<svg>`, `<iframe>` (always when `strip_scripts=True`)
- `<nav>`, `<header>`, `<footer>` (when `strip_nav=True`)

## IETF RFC Conversion

The converter auto-detects IETF RFC markup by looking for:

1. A `<div class="rfcmarkup">` container
2. `<span class="h1">` through `<span class="h6">` heading elements inside `<pre>` tags

### RFC-Specific Handling

- **Heading spans**: `<span class="h2">1. Introduction</span>` becomes `## 1. Introduction`
- **Page breaks**: `<hr>` elements and page footer/header lines are stripped
- **Table of Contents**: Auto-detected and removed from output
- **Preamble**: Abstract, Status, and Copyright sections before the first heading are skipped
- **Bullet lists**: RFC-style `o  ` prefixed items become markdown bullet lists
- **ASCII art**: Box-drawing diagrams are wrapped in fenced code blocks
- **Links**: External links (`http://`, `https://`) are preserved; internal RFC links render as plain text

### Example: Converting an RFC

```python
from dataknobs_xization import html_to_markdown

# Fetch RFC HTML from datatracker.ietf.org
markdown = html_to_markdown(rfc_html, title="RFC 6749 - OAuth 2.0")
```

The converter extracts the document title from the first `<span class="h1">` if no title is provided.

## YAML Frontmatter

Prepend YAML frontmatter to the output:

```python
markdown = html_to_markdown(
    html_content,
    frontmatter={
        "source": "https://example.com/page",
        "type": "article",
        "tags": ["oauth", "security"],
    },
)
```

Output:

```markdown
---
source: https://example.com/page
type: article
tags:
  - oauth
  - security
---

# Title
...
```

## ContentTransformer Integration

HTML conversion is integrated into the `ContentTransformer` dispatch:

```python
from dataknobs_xization import ContentTransformer

transformer = ContentTransformer(base_heading_level=2)

# Via the generic dispatch method
markdown = transformer.transform(html_string, format="html")

# Or directly
markdown = transformer.transform_html(html_string, title="My Page")
```

## Convenience Function

For quick one-off conversions:

```python
from dataknobs_xization import html_to_markdown

markdown = html_to_markdown(
    content,                    # HTML string or Path
    title="Document Title",     # Optional title
    base_heading_level=1,       # Starting heading level
    include_links=True,         # Preserve hyperlinks
    frontmatter={"key": "val"}, # Optional YAML frontmatter
)
```

## Integration with RAG Pipeline

The typical flow for HTML-to-RAG ingestion:

```python
from dataknobs_xization import HTMLConverter, parse_markdown, chunk_markdown_tree

# Step 1: Convert HTML to markdown
converter = HTMLConverter(base_heading_level=1)
markdown = converter.convert(html_content, title="My Document")

# Step 2: Parse markdown into tree
tree = parse_markdown(markdown)

# Step 3: Chunk for vector storage
chunks = chunk_markdown_tree(tree, max_chunk_size=500, chunk_overlap=50)

for chunk in chunks:
    # Store in vector database
    pass
```

## API Reference

### HTMLConverterConfig

```python
@dataclass
class HTMLConverterConfig:
    base_heading_level: int = 1
    include_links: bool = True
    strip_nav: bool = True
    strip_scripts: bool = True
    preserve_code_blocks: bool = True
    link_style: str = "inline"       # "inline", "reference", or "text"
    strip_images: bool = False
    wrap_width: int = 0              # 0 = no wrapping
    frontmatter: dict[str, Any] | None = None
```

### HTMLConverter

```python
class HTMLConverter:
    def __init__(
        self,
        config: HTMLConverterConfig | None = None,
        **kwargs: Any,
    ) -> None: ...

    def convert(
        self,
        content: str | Path,
        title: str | None = None,
    ) -> str: ...
```

### html_to_markdown

```python
def html_to_markdown(
    content: str | Path,
    title: str | None = None,
    base_heading_level: int = 1,
    include_links: bool = True,
    frontmatter: dict[str, Any] | None = None,
) -> str: ...
```
