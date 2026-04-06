"""Composite chunker that applies a transform pipeline after initial chunking."""

from __future__ import annotations


from dataknobs_xization.chunking.base import ChunkTransform, Chunker, DocumentInfo
from dataknobs_xization.markdown.md_chunker import Chunk


class CompositeChunker(Chunker):
    """Chunker that wraps an inner chunker and applies ordered transforms.

    The inner chunker produces initial chunks; each transform processes
    the list in sequence.  After all transforms, ``chunk_index`` is
    re-numbered to ensure a clean ``0..N`` sequence.

    When no transforms are configured, prefer using the inner chunker
    directly to avoid the wrapper overhead.

    .. note::

       Transforms that merge or split chunks clear ``embedding_text``
       on the affected chunks because the heading-enriched text from
       the original chunker is no longer valid.  Consumers that need
       ``embedding_text`` should regenerate it after the pipeline runs,
       or ensure their transforms preserve/rebuild it.
    """

    def __init__(
        self,
        inner: Chunker,
        transforms: list[ChunkTransform],
    ):
        self._inner = inner
        self._transforms = list(transforms)

    def chunk(
        self,
        content: str,
        document_info: DocumentInfo | None = None,
    ) -> list[Chunk]:
        """Chunk content and apply the transform pipeline."""
        chunks = self._inner.chunk(content, document_info)
        for transform in self._transforms:
            chunks = transform.transform(chunks, document_info)
        # Re-index after all transforms
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
        return chunks
