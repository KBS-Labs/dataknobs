"""Built-in chunk transforms for common post-processing operations."""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_xization.chunking.base import ChunkTransform, DocumentInfo
from dataknobs_xization.chunking.utils import split_text
from dataknobs_xization.markdown.md_chunker import Chunk, ChunkMetadata
from dataknobs_xization.markdown.filters import ChunkQualityConfig, ChunkQualityFilter

logger = logging.getLogger(__name__)


def _positions_known(chunk: Chunk) -> bool:
    """Return True if the chunk has meaningful position data."""
    return chunk.metadata.char_start != 0 or chunk.metadata.char_end != 0


class MergeSmallChunks(ChunkTransform):
    """Merge adjacent chunks that fall below a minimum size threshold.

    Only merges chunks with identical heading paths to preserve
    heading-level coherence.  When ``max_size`` is set, a merge is
    skipped if the result would exceed that limit, preventing cascade
    merging into unboundedly large chunks.
    """

    def __init__(self, min_size: int = 200, max_size: int | None = None):
        self.min_size = min_size
        self.max_size = max_size

    def transform(
        self,
        chunks: list[Chunk],
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        result: list[Chunk] = []
        pending: Chunk | None = None

        for chunk in chunks:
            if pending is None:
                pending = chunk
                continue

            # Only merge if same heading path and at least one is small
            same_headings = pending.metadata.headings == chunk.metadata.headings
            pending_small = pending.metadata.chunk_size < self.min_size
            chunk_small = chunk.metadata.chunk_size < self.min_size
            would_exceed = (
                self.max_size is not None
                and (pending.metadata.chunk_size + chunk.metadata.chunk_size + 2)
                > self.max_size
            )

            if same_headings and (pending_small or chunk_small) and not would_exceed:
                pending = self._merge(pending, chunk)
            else:
                result.append(pending)
                pending = chunk

        if pending is not None:
            result.append(pending)

        return result

    @staticmethod
    def _merge(a: Chunk, b: Chunk) -> Chunk:
        """Merge two chunks into one, maintaining position invariants.

        When position data is unknown on both inputs (``char_start``
        and ``char_end`` are both 0), the merged chunk also gets 0/0.
        ``embedding_text`` is cleared because the heading-enriched text
        from the original chunker is no longer valid for the merged
        content.
        """
        merged_text = a.text + "\n\n" + b.text
        if _positions_known(a) or _positions_known(b):
            char_start = min(a.metadata.char_start, b.metadata.char_start)
            char_end = max(a.metadata.char_end, b.metadata.char_end)
        else:
            char_start = 0
            char_end = 0
        return Chunk(
            text=merged_text,
            metadata=ChunkMetadata(
                headings=a.metadata.headings,
                heading_levels=a.metadata.heading_levels,
                line_number=a.metadata.line_number,
                char_start=char_start,
                char_end=char_end,
                chunk_index=0,  # Re-numbered by CompositeChunker
                chunk_size=len(merged_text),
                content_length=a.metadata.content_length + b.metadata.content_length + 2,
                heading_display=a.metadata.heading_display,
                embedding_text="",  # Invalidated by merge
                custom=a.metadata.custom,
            ),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MergeSmallChunks:
        return cls(
            min_size=config.get("min_size", 200),
            max_size=config.get("max_size"),
        )


class SplitLargeChunks(ChunkTransform):
    """Re-split chunks that exceed a maximum size limit.

    Uses boundary-aware splitting via
    :func:`~dataknobs_xization.chunking.utils.split_text`.
    ``embedding_text`` is cleared on split chunks because the
    heading-enriched text from the original chunker is no longer
    valid for the fragment.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size

    def transform(
        self,
        chunks: list[Chunk],
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        result: list[Chunk] = []
        for chunk in chunks:
            if chunk.metadata.chunk_size <= self.max_size:
                result.append(chunk)
                continue

            has_positions = _positions_known(chunk)

            # Split and create new chunks with adjusted positions.
            # embedding_text is cleared because the heading-enriched
            # text from the original chunker is no longer valid for
            # the split fragment.
            for text, rel_start, rel_end in split_text(chunk.text, self.max_size):
                if has_positions:
                    span_len = chunk.metadata.char_end - chunk.metadata.char_start
                    text_len = len(chunk.text) or 1
                    abs_start = chunk.metadata.char_start + int(
                        rel_start * span_len / text_len
                    )
                    abs_end = chunk.metadata.char_start + int(
                        rel_end * span_len / text_len
                    )
                else:
                    abs_start = 0
                    abs_end = 0

                result.append(Chunk(
                    text=text,
                    metadata=ChunkMetadata(
                        headings=list(chunk.metadata.headings),
                        heading_levels=list(chunk.metadata.heading_levels),
                        line_number=chunk.metadata.line_number,
                        char_start=abs_start,
                        char_end=abs_end,
                        chunk_index=0,  # Re-numbered by CompositeChunker
                        chunk_size=len(text),
                        content_length=len(text),
                        heading_display=chunk.metadata.heading_display,
                        embedding_text="",  # Invalidated by split
                        custom=dict(chunk.metadata.custom),
                    ),
                ))
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SplitLargeChunks:
        return cls(max_size=config.get("max_size", 1000))


class QualityFilterTransform(ChunkTransform):
    """Filter chunks using :class:`ChunkQualityFilter` as a pipeline stage.

    This wraps the existing quality filter as a composable transform,
    allowing it to be used at any point in a transform pipeline rather
    than only as the final stage inside ``MarkdownChunker``.
    """

    def __init__(self, quality_filter: ChunkQualityConfig | dict[str, Any]):
        if isinstance(quality_filter, dict):
            quality_filter = ChunkQualityConfig(**quality_filter)
        self._filter = ChunkQualityFilter(quality_filter)

    def transform(
        self,
        chunks: list[Chunk],
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        return [c for c in chunks if self._filter.is_valid(c)]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QualityFilterTransform:
        return cls(quality_filter=config)
