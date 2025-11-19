"""Context formatting utilities for RAG retrieval.

This module provides formatting for retrieved chunks to optimize
LLM context window usage and improve comprehension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataknobs_bots.knowledge.retrieval.merger import MergedChunk


@dataclass
class FormatterConfig:
    """Configuration for context formatting.

    Attributes:
        small_chunk_threshold: Max chars for "small" chunks (full heading path)
        medium_chunk_threshold: Max chars for "medium" chunks (last 2 headings)
        include_scores: Whether to include similarity scores
        include_source: Whether to include source file information
        group_by_source: Whether to group chunks by source file
    """

    small_chunk_threshold: int = 200
    medium_chunk_threshold: int = 800
    include_scores: bool = False
    include_source: bool = True
    group_by_source: bool = False


class ContextFormatter:
    """Formats retrieved chunks for LLM context with dynamic heading inclusion.

    This formatter applies intelligent heading inclusion based on content
    size to optimize token usage while maintaining context clarity:
    - Small chunks: Full heading path (need context)
    - Medium chunks: Last 2 heading levels
    - Large chunks: No headings (content is self-contained)

    Example:
        ```python
        formatter = ContextFormatter(FormatterConfig(
            small_chunk_threshold=200,
            include_scores=True
        ))

        # Format standard results
        context = formatter.format(results)

        # Format merged chunks
        context = formatter.format_merged(merged_chunks)
        ```
    """

    def __init__(self, config: FormatterConfig | None = None):
        """Initialize the context formatter.

        Args:
            config: Formatter configuration, uses defaults if not provided
        """
        self.config = config or FormatterConfig()

    def format(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM context.

        Args:
            results: Search results from RAGKnowledgeBase.query()

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        if self.config.group_by_source:
            return self._format_grouped_by_source(results)

        formatted_chunks = []
        for i, result in enumerate(results, 1):
            formatted = self._format_result(result, i)
            formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    def format_merged(self, merged_chunks: list[MergedChunk]) -> str:
        """Format merged chunks for LLM context.

        Args:
            merged_chunks: Merged chunks from ChunkMerger

        Returns:
            Formatted context string
        """
        if not merged_chunks:
            return ""

        # Convert to result format and use standard formatting
        results = []
        for chunk in merged_chunks:
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "heading_path": chunk.heading_display,
                "similarity": chunk.avg_similarity,
                "metadata": {
                    "headings": chunk.heading_path,
                    "content_length": chunk.content_length,
                },
            })

        return self.format(results)

    def _format_result(self, result: dict[str, Any], index: int) -> str:
        """Format a single result with dynamic heading inclusion.

        Args:
            result: Search result dictionary
            index: Result index for numbering

        Returns:
            Formatted chunk string
        """
        text = result.get("text", "")
        source = result.get("source", "")
        similarity = result.get("similarity", 0.0)
        metadata = result.get("metadata", {})

        # Get heading information
        headings = metadata.get("headings", [])
        if not headings:
            heading_path = result.get("heading_path", "")
            if isinstance(heading_path, str) and heading_path:
                headings = heading_path.split(" > ")

        # Determine content length for heading decision
        content_length = metadata.get("content_length", len(text))

        # Get headings to display based on content size
        display_headings = self._get_display_headings(headings, content_length)

        # Build formatted chunk
        lines = []

        # Add index and heading
        if display_headings:
            heading_str = " > ".join(display_headings)
            if self.config.include_scores:
                lines.append(f"[{index}] [{similarity:.2f}] {heading_str}")
            else:
                lines.append(f"[{index}] {heading_str}")
        else:
            if self.config.include_scores:
                lines.append(f"[{index}] [{similarity:.2f}]")
            else:
                lines.append(f"[{index}]")

        # Add content
        lines.append(text.strip())

        # Add source
        if self.config.include_source and source:
            lines.append(f"(Source: {source})")

        return "\n".join(lines)

    def _get_display_headings(
        self,
        headings: list[str],
        content_length: int,
    ) -> list[str]:
        """Get headings to display based on content length.

        Implements dynamic heading inclusion:
        - Small chunks: Full heading path
        - Medium chunks: Last 2 heading levels
        - Large chunks: No headings

        Args:
            headings: Full heading path
            content_length: Length of content in characters

        Returns:
            List of headings to display
        """
        if not headings:
            return []

        if content_length < self.config.small_chunk_threshold:
            # Small chunks: include full heading path
            return headings
        elif content_length < self.config.medium_chunk_threshold:
            # Medium chunks: include last 2 heading levels
            return headings[-2:] if len(headings) > 2 else headings
        else:
            # Large chunks: omit headings (content is self-contained)
            return []

    def _format_grouped_by_source(self, results: list[dict[str, Any]]) -> str:
        """Format results grouped by source file.

        Args:
            results: Search results

        Returns:
            Formatted context string with source grouping
        """
        from collections import defaultdict

        # Group by source
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            source = result.get("source", "unknown")
            groups[source].append(result)

        # Format each group
        formatted_groups = []
        chunk_index = 1

        for source, source_results in groups.items():
            group_lines = [f"## Source: {source}"]

            for result in source_results:
                formatted = self._format_result(result, chunk_index)
                # Remove source line since we're grouping
                lines = formatted.split("\n")
                lines = [line for line in lines if not line.startswith("(Source:")]
                group_lines.append("\n".join(lines))
                chunk_index += 1

            formatted_groups.append("\n\n".join(group_lines))

        return "\n\n---\n\n".join(formatted_groups)

    def wrap_for_prompt(self, context: str, tag: str = "knowledge_base") -> str:
        """Wrap formatted context in XML tags for prompt injection.

        Args:
            context: Formatted context string
            tag: Tag name to wrap with

        Returns:
            Context wrapped in XML tags
        """
        if not context:
            return ""
        return f"<{tag}>\n{context}\n</{tag}>"
