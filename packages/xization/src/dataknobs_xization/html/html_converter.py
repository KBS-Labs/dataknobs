"""HTML to markdown conversion.

Converts HTML documents into well-structured markdown suitable for RAG ingestion
and chunking with MarkdownChunker. Supports both standard HTML (semantic tags)
and IETF RFC markup (pre-formatted text with span-based headings).

Example:
    >>> converter = HTMLConverter()
    >>> markdown = converter.convert("<h1>Title</h1><p>Content here.</p>")
    >>> print(markdown)
    # Title
    <BLANKLINE>
    Content here.

    >>> # RFC-style documents are auto-detected
    >>> markdown = converter.convert(rfc_html, title="RFC 6749")
"""

import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


@dataclass
class HTMLConverterConfig:
    """Configuration for HTML to markdown conversion.

    Attributes:
        base_heading_level: Minimum heading level in output (1 = #, 2 = ##, etc.)
        include_links: Whether to preserve hyperlinks as markdown links.
        strip_nav: Remove <nav>, <header>, <footer> elements.
        strip_scripts: Remove <script> and <style> elements.
        preserve_code_blocks: Keep <pre>/<code> content as fenced code blocks.
        link_style: How to render links - "inline" for ``[text](url)``,
            "reference" for reference-style links, or "text" for just the text.
        strip_images: Whether to remove images from output.
        wrap_width: Wrap paragraph text at this width. 0 for no wrapping.
        frontmatter: Optional YAML frontmatter dict to prepend to output.
    """

    base_heading_level: int = 1
    include_links: bool = True
    strip_nav: bool = True
    strip_scripts: bool = True
    preserve_code_blocks: bool = True
    link_style: Literal["inline", "reference", "text"] = "inline"
    strip_images: bool = False
    wrap_width: int = 0
    frontmatter: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.base_heading_level <= 6:
            raise ValueError(
                f"base_heading_level must be between 1 and 6, got {self.base_heading_level}"
            )
        valid_link_styles = ("inline", "reference", "text")
        if self.link_style not in valid_link_styles:
            raise ValueError(
                f"link_style must be one of {valid_link_styles}, got {self.link_style!r}"
            )


# Tags whose content should be removed entirely.
_STRIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}
_NAV_TAGS = {"nav", "header", "footer"}

# Standard heading tags and their levels.
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# RFC heading span classes and their levels.
_RFC_HEADING_CLASSES = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Pattern matching RFC page footer lines like "Hardt   Standards Track   [Page 5]"
_RFC_PAGE_FOOTER_RE = re.compile(
    r"^\s*\S+\s+(?:Standards Track|Informational|Experimental|Best Current Practice"
    r"|Historic)\s+\[Page \d+\]\s*$"
)

# Pattern matching RFC page header lines like "RFC 6749   OAuth 2.0   October 2012"
_RFC_PAGE_HEADER_RE = re.compile(r"^\s*RFC \d+\s+.+\s+\w+ \d{4}\s*$")

# Pattern matching RFC section number prefixes like "1.", "1.1.", "10.12."
_RFC_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$")

# Detect ASCII art: lines with box-drawing characters (require + or | anchors).
# Matches "+---+", "|   |", "+===+", and similar box-drawing patterns.
# Deliberately excludes bare "--" and "->" which are common in prose.
_ASCII_ART_RE = re.compile(r"[+|][-=]+[+|]|[+|][-=]{2,}|[-=]{2,}[+|]")


class HTMLConverter:
    """Convert HTML content to well-structured markdown.

    Supports standard HTML with semantic tags (h1-h6, p, ul, ol, table, pre, etc.)
    and auto-detects IETF RFC markup format (pre-formatted text with span-based
    headings).

    The converter produces markdown compatible with MarkdownParser and MarkdownChunker
    for downstream RAG ingestion.

    Example:
        >>> converter = HTMLConverter()
        >>> md = converter.convert("<h2>Overview</h2><p>Details here.</p>")
        >>> print(md)
        ## Overview
        <BLANKLINE>
        Details here.
    """

    def __init__(self, config: HTMLConverterConfig | None = None, **kwargs: Any):
        """Initialize the converter.

        Args:
            config: Converter configuration. If None, uses defaults.
            **kwargs: Override individual config fields (e.g., base_heading_level=2).
        """
        if config is not None:
            self.config = config
        else:
            self.config = HTMLConverterConfig(**kwargs)

    def convert(self, content: str | Path, title: str | None = None) -> str:
        """Convert HTML content to markdown.

        Auto-detects whether the document is standard HTML or IETF RFC markup
        and applies the appropriate conversion strategy.

        Note:
            Each call uses internal state for reference-style link collection.
            Do not call ``convert()`` concurrently on the same instance from
            multiple threads. Create separate ``HTMLConverter`` instances for
            concurrent use, or use the ``html_to_markdown()`` convenience
            function which creates a fresh instance per call.

        Args:
            content: HTML string or path to an HTML file.
            title: Optional document title. If provided, prepended as a top-level
                heading. For RFC documents, extracted automatically if not provided.

        Returns:
            Well-structured markdown string.
        """
        if isinstance(content, Path):
            content = content.read_text(encoding="utf-8")
        elif not isinstance(content, str):
            raise TypeError(f"content must be str or Path, got {type(content).__name__}")

        # Per-conversion state for reference-style link collection.
        self._link_references: list[tuple[str, str]] = []

        soup = BeautifulSoup(content, "html.parser")

        # Strip unwanted elements before detection or conversion.
        self._strip_elements(soup)

        # Detect format and dispatch.
        if self._is_rfc_markup(soup):
            logger.debug("Detected IETF RFC markup format")
            result = self._convert_rfc(soup, title)
        else:
            logger.debug("Using standard HTML conversion")
            result = self._convert_standard(soup, title)

        # Append reference-style link definitions.
        if self._link_references:
            result = result.rstrip("\n") + "\n\n"
            for idx, (_text, href) in enumerate(self._link_references, 1):
                result += f"[{idx}]: {href}\n"

        # Prepend frontmatter if configured.
        if self.config.frontmatter:
            fm = self._render_frontmatter(self.config.frontmatter)
            result = fm + "\n" + result

        return self._normalize_whitespace(result)

    # ── Format Detection ──────────────────────────────────────────────

    def _is_rfc_markup(self, soup: BeautifulSoup) -> bool:
        """Detect IETF RFC markup by looking for characteristic patterns."""
        # Check for rfcmarkup class
        if soup.find("div", class_="rfcmarkup"):
            return True
        # Check for RFC heading spans inside pre tags
        for cls in ("h1", "h2", "h3"):
            if soup.find("span", class_=cls):
                pre_parent = soup.find("span", class_=cls)
                if pre_parent and pre_parent.find_parent("pre"):
                    return True
        return False

    # ── Standard HTML Conversion ──────────────────────────────────────

    def _convert_standard(self, soup: BeautifulSoup, title: str | None) -> str:
        """Convert standard semantic HTML to markdown."""
        lines: list[str] = []

        if title:
            level = self.config.base_heading_level
            lines.append(f"{'#' * level} {title}")
            lines.append("")
            # Remove h1 elements to avoid duplicating the title.
            for h1 in soup.find_all("h1"):
                h1.decompose()

        # Find the main content area, falling back to body or the whole document.
        body = (
            soup.find("main")
            or soup.find("article")
            or soup.find("body")
            or soup
        )

        self._process_element(body, lines)
        return "\n".join(lines)

    def _process_element(self, element: Tag, lines: list[str], depth: int = 0) -> None:
        """Recursively process an HTML element into markdown lines."""
        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    lines.append(text)
                continue

            if not isinstance(child, Tag):
                continue

            tag = child.name

            # Headings
            if tag in _HEADING_TAGS:
                self._process_heading(child, lines)

            # Paragraphs
            elif tag == "p":
                self._process_paragraph(child, lines)

            # Lists
            elif tag in ("ul", "ol"):
                self._process_list(child, lines, ordered=(tag == "ol"))
                lines.append("")

            # Code blocks
            elif tag == "pre":
                self._process_code_block(child, lines)

            # Blockquotes
            elif tag == "blockquote":
                self._process_blockquote(child, lines)

            # Tables
            elif tag == "table":
                self._process_table(child, lines)

            # Horizontal rules
            elif tag == "hr":
                lines.append("---")
                lines.append("")

            # Definition lists
            elif tag == "dl":
                self._process_definition_list(child, lines)

            # Divs, sections, articles — recurse into them
            elif tag in ("div", "section", "article", "main", "aside", "figure"):
                self._process_element(child, lines, depth + 1)

            # Figcaption
            elif tag == "figcaption":
                text = self._inline_text(child)
                if text:
                    lines.append(f"*{text}*")
                    lines.append("")

            # Skip known non-content tags
            elif tag in ("br",):
                lines.append("")

    def _process_heading(self, tag: Tag, lines: list[str]) -> None:
        """Convert a heading tag to markdown."""
        level = _HEADING_TAGS.get(tag.name, 2)
        level = min(6, max(1, level + self.config.base_heading_level - 1))
        text = self._inline_text(tag).strip()
        if text:
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")

    def _process_paragraph(self, tag: Tag, lines: list[str]) -> None:
        """Convert a paragraph to markdown."""
        text = self._inline_text(tag).strip()
        if not text:
            return
        if self.config.wrap_width > 0:
            text = textwrap.fill(text, width=self.config.wrap_width)
        lines.append(text)
        lines.append("")

    def _process_list(
        self, tag: Tag, lines: list[str], ordered: bool = False, indent: int = 0
    ) -> None:
        """Convert ul/ol to markdown list items."""
        counter = 1
        indent_str = "  " * indent

        for child in tag.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "li":
                prefix = f"{counter}." if ordered else "-"
                # Check for nested lists
                nested_list = child.find(["ul", "ol"], recursive=False)
                if nested_list:
                    # Get text before the nested list
                    text_parts = []
                    for sub in child.children:
                        if isinstance(sub, Tag) and sub.name in ("ul", "ol"):
                            break
                        text_parts.append(
                            self._inline_text(sub) if isinstance(sub, Tag)
                            else str(sub)
                        )
                    text = " ".join(t.strip() for t in text_parts if t.strip())
                    lines.append(f"{indent_str}{prefix} {text}")
                    self._process_list(
                        nested_list, lines,
                        ordered=(nested_list.name == "ol"),
                        indent=indent + 1,
                    )
                else:
                    text = self._inline_text(child).strip()
                    lines.append(f"{indent_str}{prefix} {text}")
                counter += 1

    def _process_code_block(self, tag: Tag, lines: list[str]) -> None:
        """Convert pre/code to a fenced code block."""
        if not self.config.preserve_code_blocks:
            return

        # Detect language from code tag class
        code_tag = tag.find("code")
        language = ""
        if code_tag and isinstance(code_tag, Tag):
            classes = code_tag.get("class", [])
            if isinstance(classes, list):
                for cls in classes:
                    if cls.startswith("language-"):
                        language = cls[9:]
                        break

        text = tag.get_text()
        # Strip leading/trailing blank lines
        text = text.strip("\n")

        lines.append(f"```{language}")
        lines.append(text)
        lines.append("```")
        lines.append("")

    def _process_blockquote(self, tag: Tag, lines: list[str]) -> None:
        """Convert blockquote to markdown."""
        inner_lines: list[str] = []
        self._process_element(tag, inner_lines)
        for line in inner_lines:
            lines.append(f"> {line}" if line else ">")
        lines.append("")

    def _process_table(self, tag: Tag, lines: list[str]) -> None:
        """Convert an HTML table to a markdown table."""
        rows: list[list[str]] = []

        for tr in tag.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                cells.append(self._inline_text(cell).strip())
            if cells:
                rows.append(cells)

        if not rows:
            return

        # Normalize column count
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # First row is always treated as header in markdown tables.
        header = rows[0]
        body = rows[1:]

        # Render
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    def _process_definition_list(self, tag: Tag, lines: list[str]) -> None:
        """Convert dl/dt/dd to markdown."""
        for child in tag.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "dt":
                text = self._inline_text(child).strip()
                lines.append(f"**{text}**")
            elif child.name == "dd":
                text = self._inline_text(child).strip()
                lines.append(f": {text}")
                lines.append("")

    def _inline_text(self, element: Tag | NavigableString) -> str:
        """Extract inline text from an element, converting inline markup."""
        if isinstance(element, NavigableString):
            return str(element)

        parts: list[str] = []
        for child in element.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name in ("strong", "b"):
                    inner = self._inline_text(child)
                    parts.append(f"**{inner}**")
                elif child.name in ("em", "i"):
                    inner = self._inline_text(child)
                    parts.append(f"*{inner}*")
                elif child.name == "code":
                    inner = child.get_text()
                    parts.append(f"`{inner}`")
                elif child.name == "a" and self.config.include_links:
                    inner = self._inline_text(child)
                    href = child.get("href", "")
                    if href and self.config.link_style == "inline":
                        parts.append(f"[{inner}]({href})")
                    elif href and self.config.link_style == "reference":
                        self._link_references.append((inner, href))
                        ref_idx = len(self._link_references)
                        parts.append(f"[{inner}][{ref_idx}]")
                    else:
                        parts.append(inner)
                elif child.name == "img" and not self.config.strip_images:
                    alt = child.get("alt", "")
                    src = child.get("src", "")
                    parts.append(f"![{alt}]({src})")
                elif child.name == "br":
                    parts.append("\n")
                elif child.name in ("sup", "sub", "span"):
                    parts.append(self._inline_text(child))
                else:
                    # For other inline elements, just get text
                    parts.append(self._inline_text(child))

        return "".join(parts)

    # ── RFC Conversion ────────────────────────────────────────────────

    def _convert_rfc(self, soup: BeautifulSoup, title: str | None) -> str:
        """Convert IETF RFC markup to markdown.

        RFC HTML from datatracker.ietf.org uses pre-formatted text with
        span-based headings (class="h1", "h2", etc.) and inline links.
        Page breaks are marked with <hr> and newpage pre elements.
        """
        lines: list[str] = []

        # Extract title from the document if not provided.
        if not title:
            title_tag = soup.find("span", class_="h1")
            if title_tag:
                title = self._rfc_heading_text(title_tag)

        if title:
            level = self.config.base_heading_level
            lines.append(f"{'#' * level} {title}")
            lines.append("")

        # Find the rfcmarkup content div
        content_div = soup.find("div", class_="rfcmarkup")
        if not content_div:
            content_div = soup.find("body") or soup
        # If nested rfcmarkup, use the inner one
        inner = content_div.find("div", class_="rfcmarkup")
        if inner:
            content_div = inner

        # Collect all text from pre elements, processing heading spans
        raw_text = self._extract_rfc_text(content_div)

        # Parse the raw text into structured sections
        self._parse_rfc_text(raw_text, lines)

        return "\n".join(lines)

    def _extract_rfc_text(self, container: Tag) -> str:
        """Extract text from RFC markup, replacing heading spans with markers."""
        parts: list[str] = []

        for child in container.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if text.strip():
                    parts.append(text)
                continue

            if not isinstance(child, Tag):
                continue

            # Skip page-break hr elements
            if child.name == "hr":
                continue

            if child.name == "pre":
                self._extract_rfc_pre(child, parts)
            elif child.name == "div":
                # Recurse into divs
                parts.append(self._extract_rfc_text(child))

        return "\n".join(parts)

    def _extract_rfc_pre(self, pre: Tag, parts: list[str]) -> None:
        """Extract text from a single RFC <pre> element."""
        segment_parts: list[str] = []

        for child in pre.children:
            if isinstance(child, NavigableString):
                segment_parts.append(str(child))
            elif isinstance(child, Tag):
                # Heading spans get special markers
                heading_level = self._rfc_heading_level(child)
                if heading_level:
                    text = self._rfc_heading_text(child)
                    segment_parts.append(f"\x00HEADING:{heading_level}:{text}\x00")
                # Page footer/header spans (class="grey") or page anchors — skip
                elif child.name == "span" and (
                    "grey" in (child.get("class") or [])
                    or child.get("id", "").startswith("page-")
                ):
                    continue
                # Links — extract text with href
                elif child.name == "a":
                    link_text = child.get_text()
                    href = child.get("href", "")
                    if href and self.config.include_links:
                        # Only include href for external links
                        if href.startswith(("http://", "https://")):
                            if self.config.link_style == "reference":
                                self._link_references.append((link_text, href))
                                ref_idx = len(self._link_references)
                                segment_parts.append(f"[{link_text}][{ref_idx}]")
                            else:
                                segment_parts.append(f"[{link_text}]({href})")
                        else:
                            segment_parts.append(link_text)
                    else:
                        segment_parts.append(link_text)
                else:
                    segment_parts.append(child.get_text())

        parts.append("".join(segment_parts))

    def _rfc_heading_level(self, tag: Tag) -> int | None:
        """Get heading level from an RFC span class, or None."""
        if tag.name != "span":
            return None
        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            if cls in _RFC_HEADING_CLASSES:
                return _RFC_HEADING_CLASSES[cls]
        return None

    def _rfc_heading_text(self, tag: Tag) -> str:
        """Extract clean heading text from an RFC heading span."""
        text = tag.get_text().strip()
        # Remove section number prefix (e.g., "1.  Introduction" -> "Introduction")
        # but keep it available for reference
        match = _RFC_SECTION_NUM_RE.match(text)
        if match:
            section_num = match.group(1)
            heading_text = match.group(2).strip()
            return f"{section_num}. {heading_text}"
        return text

    def _parse_rfc_text(self, raw: str, lines: list[str]) -> None:
        """Parse extracted RFC text into markdown sections."""
        # Split on heading markers
        current_lines: list[str] = []
        in_toc = False
        # Skip everything until the first non-h1 section heading.
        # This strips the preamble (Abstract, Status, Copyright, ToC).
        skip_preamble = True

        for line in raw.split("\n"):
            # Check for heading marker
            if "\x00HEADING:" in line:
                # Flush any accumulated content
                if not skip_preamble:
                    self._flush_rfc_content(current_lines, lines)
                current_lines = []

                # Extract heading info
                match = re.search(r"\x00HEADING:(\d+):(.+?)\x00", line)
                if match:
                    level = int(match.group(1))
                    text = match.group(2).strip()

                    # Skip the h1 title — already added
                    if level == 1:
                        continue

                    adjusted = min(6, max(1, level + self.config.base_heading_level - 1))
                    lines.append(f"{'#' * adjusted} {text}")
                    lines.append("")
                    skip_preamble = False

                    # Detect Table of Contents
                    in_toc = "table of contents" in text.lower()
                continue

            # Skip preamble content (before first section heading)
            if skip_preamble:
                continue

            # Skip Table of Contents body
            if in_toc:
                continue

            # Skip page footers and headers
            stripped = line.strip()
            if _RFC_PAGE_FOOTER_RE.match(stripped):
                continue
            if _RFC_PAGE_HEADER_RE.match(stripped):
                continue

            # Skip "NewPage" artifacts from HTML comments
            if stripped == "NewPage":
                continue

            current_lines.append(line)

        # Flush remaining content
        self._flush_rfc_content(current_lines, lines)

    def _flush_rfc_content(self, raw_lines: list[str], out: list[str]) -> None:
        """Process accumulated RFC content lines into markdown."""
        if not raw_lines:
            return

        # Remove leading/trailing blank lines
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()

        if not raw_lines:
            return

        # Group lines into paragraphs and special blocks
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            stripped = line.strip()

            # Blank line — paragraph break
            if not stripped:
                out.append("")
                i += 1
                continue

            # Detect ASCII art / diagrams (indented + box-drawing chars)
            if self._is_ascii_art_block(raw_lines, i):
                block, i = self._collect_ascii_art(raw_lines, i)
                out.append("```")
                out.extend(block)
                out.append("```")
                out.append("")
                continue

            # RFC-style bullet lists (lines starting with "o  ")
            if re.match(r"^\s{3}o\s{2}", line):
                items, i = self._collect_rfc_list(raw_lines, i)
                for item in items:
                    out.append(f"- {item}")
                out.append("")
                continue

            # Regular paragraph text (indented 3 spaces in RFC format)
            para, i = self._collect_paragraph(raw_lines, i)
            if para:
                text = " ".join(para)
                if self.config.wrap_width > 0:
                    text = textwrap.fill(text, width=self.config.wrap_width)
                out.append(text)
                out.append("")

    def _is_ascii_art_block(self, lines: list[str], start: int) -> bool:
        """Check if lines starting at index form an ASCII art block."""
        # Look for characteristic ASCII art patterns in a window.
        # Require at least 2 lines with box-drawing/arrow characters.
        window = lines[start:start + 5]
        art_lines = sum(1 for line in window if _ASCII_ART_RE.search(line))
        return art_lines >= 2

    def _collect_ascii_art(
        self, lines: list[str], start: int
    ) -> tuple[list[str], int]:
        """Collect contiguous ASCII art lines."""
        block: list[str] = []
        i = start
        blank_count = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                blank_count += 1
                if blank_count > 1:
                    break
                block.append("")
                i += 1
                continue

            # Check if this line looks like it's still part of the diagram
            is_art = bool(_ASCII_ART_RE.search(line))
            is_indented = line.startswith("     ")  # 5+ spaces

            if is_art or (is_indented and blank_count <= 1):
                blank_count = 0
                block.append(line.rstrip())
                i += 1
            else:
                break

        # Safety: always advance at least one line
        if i == start:
            i = start + 1

        # Remove trailing blank lines from block
        while block and not block[-1].strip():
            block.pop()

        return block, i

    def _collect_rfc_list(
        self, lines: list[str], start: int
    ) -> tuple[list[str], int]:
        """Collect RFC-style list items (lines starting with 'o  ')."""
        items: list[str] = []
        i = start
        current_item: list[str] = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # New list item
            if re.match(r"^\s{3}o\s{2}", line):
                if current_item:
                    items.append(" ".join(current_item))
                current_item = [re.sub(r"^\s{3}o\s{2,}", "", line).strip()]
                i += 1
                continue

            # Continuation line (indented, non-blank)
            if stripped and line.startswith("      "):
                current_item.append(stripped)
                i += 1
                continue

            # Blank line might separate items or end the list
            if not stripped:
                # Look ahead to see if list continues
                if i + 1 < len(lines) and re.match(r"^\s{3}o\s{2}", lines[i + 1]):
                    i += 1
                    continue
                break

            # Non-list content
            break

        if current_item:
            items.append(" ".join(current_item))

        return items, i

    def _collect_paragraph(
        self, lines: list[str], start: int
    ) -> tuple[list[str], int]:
        """Collect paragraph lines until a blank line or special element."""
        words: list[str] = []
        i = start

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                break

            # Stop at list items
            if re.match(r"^\s{3}o\s{2}", line):
                break

            # Stop at ASCII art
            if self._is_ascii_art_block(lines, i):
                break

            words.append(stripped)
            i += 1

        return words, i

    # ── Utility Methods ───────────────────────────────────────────────

    def _strip_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup."""
        # Always strip scripts/styles
        if self.config.strip_scripts:
            for tag in soup.find_all(list(_STRIP_TAGS)):
                tag.decompose()

        # Optionally strip navigation elements
        if self.config.strip_nav:
            for tag in soup.find_all(list(_NAV_TAGS)):
                tag.decompose()

    def _render_frontmatter(self, metadata: dict[str, Any]) -> str:
        """Render a YAML frontmatter block."""
        try:
            import yaml
            body = yaml.safe_dump(
                metadata, default_flow_style=False, sort_keys=False,
            ).rstrip("\n")
            return f"---\n{body}\n---"
        except ImportError:
            # Fallback for simple scalar/list metadata when PyYAML is absent.
            lines = ["---"]
            for key, value in metadata.items():
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("---")
            return "\n".join(lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Clean up excessive blank lines and trailing whitespace."""
        # Collapse 3+ consecutive blank lines to 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        # Remove leading blank lines
        while lines and not lines[0]:
            lines.pop(0)
        # Ensure single trailing newline
        while lines and not lines[-1]:
            lines.pop()
        if lines:
            lines.append("")
        return "\n".join(lines)


def html_to_markdown(
    content: str | Path,
    title: str | None = None,
    base_heading_level: int = 1,
    include_links: bool = True,
    frontmatter: dict[str, Any] | None = None,
) -> str:
    """Convert HTML content to markdown.

    Convenience function that creates an HTMLConverter and converts in one call.

    Args:
        content: HTML string or path to an HTML file.
        title: Optional document title.
        base_heading_level: Starting heading level (default: 1).
        include_links: Whether to preserve hyperlinks (default: True).
        frontmatter: Optional YAML frontmatter dict.

    Returns:
        Markdown formatted string.

    Example:
        >>> md = html_to_markdown("<h1>Title</h1><p>Hello world.</p>")
        >>> print(md)
        # Title
        <BLANKLINE>
        Hello world.
    """
    converter = HTMLConverter(
        base_heading_level=base_heading_level,
        include_links=include_links,
        frontmatter=frontmatter,
    )
    return converter.convert(content, title=title)
