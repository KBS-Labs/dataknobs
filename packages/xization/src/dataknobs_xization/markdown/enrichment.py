"""Heading enrichment utilities for RAG-optimized chunk embeddings.

This module provides utilities to enrich chunk content with heading context
for improved semantic search, while keeping headings out of the displayed content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def is_multiword(heading: str) -> bool:
    """Check if a heading contains multiple words.

    Args:
        heading: The heading text to check

    Returns:
        True if the heading has more than one word
    """
    return len(heading.split()) > 1


def format_heading_display(
    heading_path: list[str],
    separator: str = " > ",
) -> str:
    """Format a heading path for display.

    Args:
        heading_path: List of headings from root to chunk
        separator: Separator to use between headings

    Returns:
        Formatted heading path string
    """
    if not heading_path:
        return ""
    return separator.join(heading_path)


def get_dynamic_heading_display(
    heading_path: list[str],
    content_length: int,
    small_threshold: int = 200,
    medium_threshold: int = 800,
) -> str:
    """Get heading display based on content length.

    Dynamic heading inclusion:
    - Small chunks (< small_threshold): Full heading path
    - Medium chunks (< medium_threshold): Last 2 headings
    - Large chunks: No headings

    Args:
        heading_path: List of headings from root to chunk
        content_length: Length of chunk content in characters
        small_threshold: Max chars for "small" chunks
        medium_threshold: Max chars for "medium" chunks

    Returns:
        Formatted heading display string
    """
    if not heading_path:
        return ""

    if content_length <= small_threshold:
        # Small chunks: include full heading path
        return format_heading_display(heading_path)
    elif content_length <= medium_threshold:
        # Medium chunks: include last 2 heading levels
        relevant = heading_path[-2:] if len(heading_path) > 2 else heading_path
        return format_heading_display(relevant)
    else:
        # Large chunks: omit headings
        return ""


@dataclass
class EnrichedChunkData:
    """Data for a chunk enriched with heading context.

    Attributes:
        content: Clean content text (no headings)
        embedding_text: Text to use for embedding (heading-enriched)
        heading_path: List of headings from root to chunk
        heading_display: Formatted heading path for display
        content_length: Length of clean content in characters
    """

    content: str
    embedding_text: str
    heading_path: list[str]
    heading_display: str
    content_length: int


def build_enriched_text(heading_path: list[str], content: str) -> str:
    """Build text for embedding with relevant heading context.

    Uses a modified approach where headings are included up from the chunk
    until and including the first multi-word heading. This provides semantic
    context without over-weighting deep, single-word labels like "Example".

    Args:
        heading_path: List of heading texts from root to chunk
        content: The chunk content text

    Returns:
        Enriched text suitable for embedding

    Examples:
        >>> build_enriched_text(["Patterns", "Chain-of-Thought", "Example"], "code here")
        'Chain-of-Thought: Example: code here'

        >>> build_enriched_text(["Setup"], "install steps")
        'Setup: install steps'

        >>> build_enriched_text(["API Reference", "Authentication", "OAuth 2.0"], "...")
        'Authentication: OAuth 2.0: ...'

        >>> build_enriched_text([], "standalone content")
        'standalone content'
    """
    if not heading_path:
        return content

    # Walk backwards from deepest heading to find relevant context
    relevant_headings = []
    for heading in reversed(heading_path):
        relevant_headings.insert(0, heading)
        # Stop after including a multi-word heading
        if len(heading.split()) > 1:
            break

    # Build the enriched text
    if relevant_headings:
        prefix = ": ".join(relevant_headings)
        return f"{prefix}: {content}"

    return content


def extract_heading_metadata(
    headings: list[str],
    heading_levels: list[int],
    separator: str = " > ",
) -> dict[str, Any]:
    """Extract heading metadata for storage.

    Args:
        headings: List of heading texts from root to chunk
        heading_levels: Corresponding heading levels (1-6)
        separator: Separator for display string

    Returns:
        Dictionary with heading metadata fields
    """
    return {
        "heading_path": headings,
        "heading_levels": heading_levels,
        "heading_display": separator.join(headings) if headings else "",
        "heading_depth": len(headings),
    }


def get_relevant_headings_for_display(
    heading_path: list[str],
    content_length: int,
    small_threshold: int = 200,
    medium_threshold: int = 800,
) -> list[str]:
    """Get headings to display based on content length.

    Implements dynamic heading inclusion:
    - Small chunks: Full heading path (need context)
    - Medium chunks: Last 2 heading levels
    - Large chunks: No headings (content is self-contained)

    Args:
        heading_path: List of heading texts from root to chunk
        content_length: Length of chunk content in characters
        small_threshold: Max chars for "small" chunks
        medium_threshold: Max chars for "medium" chunks

    Returns:
        List of headings to display
    """
    if not heading_path:
        return []

    if content_length < small_threshold:
        # Small chunks: include full heading path
        return heading_path
    elif content_length < medium_threshold:
        # Medium chunks: include last 2 heading levels
        return heading_path[-2:] if len(heading_path) > 2 else heading_path
    else:
        # Large chunks: omit headings
        return []


def format_heading_for_display(
    headings: list[str],
    heading_levels: list[int] | None = None,
    format_style: str = "markdown",
) -> str:
    """Format headings for display in LLM context.

    Args:
        headings: List of heading texts to display
        heading_levels: Corresponding levels (used for markdown format)
        format_style: "markdown" for # syntax, "path" for > separated

    Returns:
        Formatted heading string
    """
    if not headings:
        return ""

    if format_style == "path":
        return " > ".join(headings)

    if format_style == "markdown" and heading_levels:
        lines = []
        for heading, level in zip(headings, heading_levels):
            lines.append(f"{'#' * level} {heading}")
        return "\n".join(lines)

    # Default: just join with separator
    return " > ".join(headings)


def enrich_chunk(
    content: str,
    headings: list[str],
    heading_levels: list[int],
) -> EnrichedChunkData:
    """Create fully enriched chunk data from raw components.

    Convenience function that combines all enrichment operations.

    Args:
        content: Raw chunk content text
        headings: List of heading texts from root to chunk
        heading_levels: Corresponding heading levels

    Returns:
        EnrichedChunkData with all computed fields
    """
    embedding_text = build_enriched_text(headings, content)

    return EnrichedChunkData(
        content=content,
        embedding_text=embedding_text,
        heading_path=headings,
        heading_display=" > ".join(headings) if headings else "",
        content_length=len(content),
    )
