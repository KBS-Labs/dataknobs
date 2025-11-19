"""Chunk merging utilities for RAG retrieval optimization.

This module provides functionality to merge adjacent chunks that share
the same heading path, improving context coherence for LLM consumption.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class MergerConfig:
    """Configuration for chunk merging.

    Attributes:
        max_merged_size: Maximum size of merged chunk content in characters
        preserve_order: Whether to preserve positional ordering within groups
    """

    max_merged_size: int = 2000
    preserve_order: bool = True


@dataclass
class MergedChunk:
    """A merged chunk combining multiple related chunks.

    Attributes:
        text: Combined text content
        source: Source file path
        heading_path: Shared heading path
        heading_display: Formatted heading display string
        chunks: Original chunks that were merged
        avg_similarity: Average similarity score of merged chunks
        content_length: Total content length
    """

    text: str
    source: str
    heading_path: list[str]
    heading_display: str
    chunks: list[dict[str, Any]]
    avg_similarity: float
    content_length: int


class ChunkMerger:
    """Merges adjacent chunks sharing the same heading path.

    This merger groups search results by their heading path and source,
    then combines them into coherent context units while respecting
    size limits.

    Example:
        ```python
        merger = ChunkMerger(MergerConfig(max_merged_size=2000))
        results = await kb.query("How do I configure auth?", k=10)
        merged = merger.merge(results)

        for chunk in merged:
            print(f"[{chunk.avg_similarity:.2f}] {chunk.heading_display}")
            print(chunk.text)
        ```
    """

    def __init__(self, config: MergerConfig | None = None):
        """Initialize the chunk merger.

        Args:
            config: Merger configuration, uses defaults if not provided
        """
        self.config = config or MergerConfig()

    def merge(self, results: list[dict[str, Any]]) -> list[MergedChunk]:
        """Merge search results by shared heading path.

        Groups chunks by (source, heading_path) and merges those that
        share identical heading paths. Chunks are ordered by their
        position within the document.

        Args:
            results: Search results from RAGKnowledgeBase.query()
                Each result should have:
                - text: Chunk content
                - source: Source file
                - heading_path: Heading hierarchy string or list
                - similarity: Similarity score
                - metadata: Full chunk metadata

        Returns:
            List of MergedChunk objects, sorted by average similarity
        """
        if not results:
            return []

        # Group chunks by (source, heading_path)
        groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)

        for result in results:
            source = result.get("source", "")
            heading_path = self._normalize_heading_path(result)
            key = (source, tuple(heading_path))
            groups[key].append(result)

        # Merge each group
        merged_chunks = []
        for (source, heading_path_tuple), chunks in groups.items():
            heading_path = list(heading_path_tuple)

            # Sort by position if available
            if self.config.preserve_order:
                chunks = self._sort_by_position(chunks)

            # Merge chunks respecting size limit
            merged = self._merge_chunk_group(chunks, source, heading_path)
            merged_chunks.extend(merged)

        # Sort by average similarity (descending)
        merged_chunks.sort(key=lambda c: c.avg_similarity, reverse=True)

        return merged_chunks

    def _normalize_heading_path(self, result: dict[str, Any]) -> list[str]:
        """Extract and normalize heading path from result.

        Args:
            result: Search result dictionary

        Returns:
            List of heading strings
        """
        # Try to get from metadata first (may have list format)
        metadata = result.get("metadata", {})
        headings = metadata.get("headings", [])
        if headings:
            return headings

        # Fall back to heading_path string
        heading_path = result.get("heading_path", "")
        if isinstance(heading_path, list):
            return heading_path
        elif heading_path:
            return heading_path.split(" > ")

        return []

    def _sort_by_position(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort chunks by their position in the document.

        Args:
            chunks: List of chunk results

        Returns:
            Sorted list
        """
        def get_position(chunk: dict[str, Any]) -> int:
            metadata = chunk.get("metadata", {})
            # Try chunk_index first, then line_number
            return metadata.get("chunk_index", metadata.get("line_number", 0))

        return sorted(chunks, key=get_position)

    def _merge_chunk_group(
        self,
        chunks: list[dict[str, Any]],
        source: str,
        heading_path: list[str],
    ) -> list[MergedChunk]:
        """Merge a group of chunks with the same heading path.

        Combines chunks until max_merged_size is reached, then starts
        a new merged chunk. Overflow chunks are returned as separate
        merged chunks.

        Args:
            chunks: Chunks to merge
            source: Source file path
            heading_path: Shared heading path

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged_results = []
        current_chunks: list[dict[str, Any]] = []
        current_size = 0

        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_size = len(chunk_text)

            # Check if adding this chunk would exceed the limit
            if current_size + chunk_size > self.config.max_merged_size and current_chunks:
                # Save current merge and start new one
                merged_results.append(
                    self._create_merged_chunk(current_chunks, source, heading_path)
                )
                current_chunks = []
                current_size = 0

            current_chunks.append(chunk)
            current_size += chunk_size

        # Don't forget the last group
        if current_chunks:
            merged_results.append(
                self._create_merged_chunk(current_chunks, source, heading_path)
            )

        return merged_results

    def _create_merged_chunk(
        self,
        chunks: list[dict[str, Any]],
        source: str,
        heading_path: list[str],
    ) -> MergedChunk:
        """Create a MergedChunk from a list of chunks.

        Args:
            chunks: Chunks to combine
            source: Source file path
            heading_path: Shared heading path

        Returns:
            MergedChunk object
        """
        # Combine text with double newline separator
        texts = [chunk.get("text", "") for chunk in chunks]
        combined_text = "\n\n".join(text.strip() for text in texts if text.strip())

        # Calculate average similarity
        similarities = [chunk.get("similarity", 0.0) for chunk in chunks]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Build heading display
        heading_display = " > ".join(heading_path) if heading_path else ""

        return MergedChunk(
            text=combined_text,
            source=source,
            heading_path=heading_path,
            heading_display=heading_display,
            chunks=chunks,
            avg_similarity=avg_similarity,
            content_length=len(combined_text),
        )

    def to_result_list(self, merged_chunks: list[MergedChunk]) -> list[dict[str, Any]]:
        """Convert merged chunks back to result list format.

        Useful for compatibility with existing code that expects
        the standard result format.

        Args:
            merged_chunks: List of merged chunks

        Returns:
            List of result dictionaries
        """
        results = []
        for merged in merged_chunks:
            results.append({
                "text": merged.text,
                "source": merged.source,
                "heading_path": merged.heading_display,
                "similarity": merged.avg_similarity,
                "metadata": {
                    "headings": merged.heading_path,
                    "content_length": merged.content_length,
                    "merged_count": len(merged.chunks),
                },
            })
        return results
