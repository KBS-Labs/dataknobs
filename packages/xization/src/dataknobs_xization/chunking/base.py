"""Base chunker and transform abstractions for pluggable document chunking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataknobs_xization.markdown.md_chunker import Chunk


@dataclass
class DocumentInfo:
    """Metadata about the document being chunked.

    Passed to :meth:`Chunker.chunk` so implementations can make
    format-aware decisions (e.g., dispatch to different parsing
    strategies based on ``content_type``).

    Attributes:
        source: Source identifier (filename, URL, label, etc.)
        content_type: MIME-ish content type hint (default ``text/markdown``)
        metadata: Arbitrary additional metadata about the document
    """

    source: str = ""
    content_type: str = "text/markdown"
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    """Base class for document chunkers.

    Implementations transform raw document content into a list of
    :class:`~dataknobs_xization.markdown.md_chunker.Chunk` objects
    suitable for embedding and retrieval.

    Subclasses must implement :meth:`chunk`.  They may also provide a
    ``from_config`` classmethod for config-driven construction — the
    chunker registry's ``create()`` will detect and use it automatically.

    **ChunkMetadata field contract for framework consumers:**

    ``RAGKnowledgeBase`` and ``DirectoryProcessor`` access the following
    fields on each returned ``Chunk``.  Custom chunkers should populate
    at least these; remaining fields may be left at their defaults.

    - ``chunk.text`` — *required*; the chunk content.
    - ``chunk.metadata.embedding_text`` — if non-empty, used as the text
      to embed; otherwise ``chunk.text`` is used.
    - ``chunk.metadata.chunk_index`` — sequential index.  Note:
      consumers (``RAGKnowledgeBase``, ``DirectoryProcessor``,
      ``CompositeChunker``) re-number this to ``0..N`` in their
      output, so the chunker-assigned value is not preserved.
    - ``chunk.metadata.chunk_size`` — size of ``chunk.text`` in chars.
    - ``chunk.metadata.content_length`` — content size excluding headings.
    - ``chunk.metadata.headings`` — heading path list (may be empty).
    - ``chunk.metadata.heading_levels`` — parallel list of heading levels.
    - ``chunk.metadata.heading_display`` — formatted heading path string.
    - ``chunk.metadata.line_number`` — source line number (0 if unknown).
    - ``chunk.metadata.char_start`` — character offset of source span
      start (0 if unknown).  For body chunks produced under
      ``combine_under_heading=True``, positions are linearly
      interpolated and may be approximate.
    - ``chunk.metadata.char_end`` — character offset of source span end,
      exclusive (0 if unknown).  Same interpolation caveat.

    **Config forwarding:** When created via ``create_chunker()``, the
    ``chunker`` routing key is stripped from the config dict before it
    reaches ``from_config()``.  Unknown keys (e.g. ``quality_filter``)
    are forwarded and should be silently ignored if not applicable.
    """

    @abstractmethod
    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        """Chunk document content into retrieval-sized pieces.

        Args:
            content: Raw document content (text).
            document_info: Optional metadata about the document being
                chunked.  Implementations may use this for format-aware
                dispatch or to enrich chunk metadata.

        Returns:
            List of chunks with text and metadata.
        """
        ...


class ChunkTransform(ABC):
    """Base class for chunk post-processing transforms.

    Transforms receive a list of chunks and return a modified list.
    They may merge, split, filter, reorder, or enrich chunks.

    **Position contract:** Transforms that merge or split chunks must
    maintain valid ``char_start``/``char_end`` on the resulting chunks:

    - **Merge:** use ``min(char_start)`` / ``max(char_end)`` of the
      merged chunks.
    - **Split:** adjust offsets relative to the original chunk's
      ``char_start``.
    - **Filter/enrich:** no position changes needed.

    Subclasses must implement :meth:`transform`.  They may also provide
    a ``from_config`` classmethod for config-driven construction — the
    transform registry's ``create()`` will detect and use it
    automatically.
    """

    @abstractmethod
    def transform(
        self,
        chunks: list[Chunk],
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        """Apply the transform to a list of chunks.

        Args:
            chunks: Input chunks to transform.
            document_info: Optional metadata about the source document.

        Returns:
            Transformed list of chunks.
        """
        ...
