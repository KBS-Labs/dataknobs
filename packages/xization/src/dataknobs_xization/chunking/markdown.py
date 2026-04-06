"""Default chunker wrapping the existing MarkdownChunker."""

from __future__ import annotations

from typing import Any

from dataknobs_xization.chunking.base import Chunker, DocumentInfo
from dataknobs_xization.markdown.md_chunker import (
    Chunk,
    HeadingInclusion,
    MarkdownChunker,
)
from dataknobs_xization.markdown.md_parser import parse_markdown
from dataknobs_xization.markdown.filters import ChunkQualityConfig


class MarkdownTreeChunker(Chunker):
    """Chunker that parses markdown and chunks by heading structure.

    This is the default chunker — it wraps the existing
    :class:`~dataknobs_xization.markdown.md_chunker.MarkdownChunker`
    with a content-level interface suitable for the chunker registry.

    All ``MarkdownChunker``-specific parameters (``heading_inclusion``,
    ``combine_under_heading``, etc.) are accepted as constructor kwargs
    and also via ``from_config``.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        heading_inclusion: HeadingInclusion = HeadingInclusion.IN_METADATA,
        combine_under_heading: bool = True,
        quality_filter: ChunkQualityConfig | dict[str, Any] | None = None,
        generate_embeddings: bool = True,
    ):
        qf: ChunkQualityConfig | None = None
        if isinstance(quality_filter, ChunkQualityConfig):
            qf = quality_filter
        elif isinstance(quality_filter, dict):
            qf = ChunkQualityConfig(**quality_filter)

        self._inner = MarkdownChunker(
            max_chunk_size=max_chunk_size,
            heading_inclusion=heading_inclusion,
            combine_under_heading=combine_under_heading,
            quality_filter=qf,
            generate_embeddings=generate_embeddings,
        )

    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        """Parse markdown content and chunk by heading structure."""
        tree = parse_markdown(content)
        return list(self._inner.chunk(tree))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MarkdownTreeChunker:
        """Create from a chunking configuration dictionary.

        Recognised keys (all optional):
            - ``max_chunk_size`` (int): Maximum chunk size in characters.
            - ``heading_inclusion`` (str): One of ``"in_text"``,
              ``"in_metadata"``, ``"both"``, ``"none"``.
            - ``combine_under_heading`` (bool): Combine text under same
              heading.
            - ``quality_filter`` (dict): Kwargs for
              :class:`ChunkQualityConfig`.
            - ``generate_embeddings`` (bool): Generate heading-enriched
              embedding text.
        """
        hi_raw = config.get("heading_inclusion")
        if hi_raw is not None:
            try:
                hi = HeadingInclusion(hi_raw)
            except ValueError:
                valid = [e.value for e in HeadingInclusion]
                raise ValueError(
                    f"Invalid 'heading_inclusion' value {hi_raw!r}. "
                    f"Valid values: {valid}"
                ) from None
        else:
            hi = HeadingInclusion.IN_METADATA

        return cls(
            max_chunk_size=config.get("max_chunk_size", 500),
            heading_inclusion=hi,
            combine_under_heading=config.get("combine_under_heading", True),
            quality_filter=config.get("quality_filter"),
            generate_embeddings=config.get("generate_embeddings", True),
        )
