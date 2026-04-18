"""Vector knowledge base grounded source.

Wraps an existing :class:`KnowledgeBase` as a :class:`GroundedSource`,
using ``text_queries`` from :class:`RetrievalIntent` for semantic search.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from typing import Any

from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
)

from dataknobs_bots.knowledge.base import KnowledgeBase

logger = logging.getLogger(__name__)


def default_dedup_key(r: dict[str, Any]) -> Hashable:
    """Default dedup key: ``(source, chunk_index)`` tuple.

    Matches the historical hardcoded behavior for markdown-chunked
    corpora produced by :class:`RAGKnowledgeBase` + ``MarkdownChunker``.
    Uses safe ``.get`` access so records missing either key do not raise
    — the ``id(r)`` fallback for missing ``chunk_index`` disables dedup
    entirely for results whose metadata doesn't carry chunk positions.
    """
    source = r.get("source", "")
    chunk_index = r.get("metadata", {}).get("chunk_index", id(r))
    return (source, chunk_index)


def default_source_id(r: dict[str, Any]) -> str:
    """Default source_id template: ``"{source}:chunk_{chunk_index}"``.

    Matches the historical hardcoded behavior. Consumers with
    non-markdown corpora should pass a custom ``source_id_fn`` to
    construct identities from their own structural keys.
    """
    source = r.get("source", "")
    chunk_index = r.get("metadata", {}).get("chunk_index", "")
    return f"{source}:chunk_{chunk_index}"


def default_metadata(r: dict[str, Any]) -> dict[str, Any]:
    """Default metadata surface: ``heading_path`` + ``source`` + raw metadata.

    Matches the historical hardcoded behavior. Non-markdown consumers
    reading ``result.metadata["heading_path"]`` / ``["source"]`` get
    empty strings (same as before). Consumers who want a clean
    non-markdown surface should pass a custom ``metadata_fn`` to
    replace this dict entirely.
    """
    return {
        "heading_path": r.get("heading_path", ""),
        "source": r.get("source", ""),
        **r.get("metadata", {}),
    }


class VectorKnowledgeSource(GroundedSource):
    """Grounded source backed by a :class:`KnowledgeBase`.

    Uses ``intent.text_queries`` for semantic similarity search. Raw
    KB results are translated into :class:`SourceResult` instances via
    a small identity layer that can be customized for non-markdown
    corpora.

    Filters:
        Structured filters keyed by this source's name in
        ``intent.filters`` are passed through to the vector store as
        metadata filters. Filter semantics are scalar-on-scalar equality
        across all built-in vector store backends (Memory, FAISS,
        pgvector). Sources that need richer filter semantics
        (list-contains, boolean composition, ranges) should compose a
        separate :class:`GroundedSource` implementation alongside the
        vector source.

    Identity, dedup, and metadata surface:
        Three callables control how raw KB results are translated:

        - ``dedup_key(raw)`` returns a hashable key used to deduplicate
          results across multiple text queries. Default: the historical
          ``(source, chunk_index)`` tuple, matching markdown-chunked
          corpora from :class:`RAGKnowledgeBase`.
        - ``source_id_fn(raw)`` returns the string ``source_id`` that
          appears on the emitted :class:`SourceResult`. Default: the
          historical ``f"{source}:chunk_{chunk_index}"`` template.
        - ``metadata_fn(raw)`` returns the ``metadata`` dict attached to
          the emitted :class:`SourceResult`. Default: ``heading_path`` +
          ``source`` keys spread with the raw metadata (matches
          historical behavior — non-markdown consumers see empty strings
          for the first two keys).

        Consumers whose corpus has a natural structural identity
        (entity id, symbol name, URL, etc.) should pass all three
        callables so the vector source emits real identity and clean
        metadata rather than synthetic file/chunk pairs and empty-string
        surface keys.

    Optionally carries a :attr:`topic_index` for heading-tree or
    cluster-based retrieval. When present, the grounded strategy uses
    the topic index instead of standard text_queries. The same
    ``source_id_fn`` and ``metadata_fn`` callables are threaded through
    the topic-index path (``dedup_key`` is not — topic-index callers
    do their own dedup).

    Example — a product catalog where each record is a product with
    its own SKU (no markdown chunking)::

        def dedup_by_sku(r):
            return r["metadata"].get("sku") or id(r)

        def sku_as_id(r):
            return r["metadata"].get("sku", "")

        def product_metadata(r):
            m = r.get("metadata", {})
            return {
                "sku": m.get("sku"),
                "category": m.get("category"),
                "price": m.get("price"),
            }

        source = VectorKnowledgeSource(
            kb=kb,
            name="catalog",
            dedup_key=dedup_by_sku,
            source_id_fn=sku_as_id,
            metadata_fn=product_metadata,
        )

    Args:
        kb: The knowledge base to query.
        name: Unique source name for provenance tracking.
        topic_index: Optional topic index for structured retrieval.
        dedup_key: Optional callable to extract a hashable dedup key
            from a raw KB result dict. Default: historical
            ``(source, chunk_index)``.
        source_id_fn: Optional callable to format the ``source_id`` on
            the emitted :class:`SourceResult`. Default: historical
            ``"{source}:chunk_{index}"`` template.
        metadata_fn: Optional callable to build the metadata dict
            attached to the emitted :class:`SourceResult`. Default:
            historical ``heading_path`` + ``source`` keys plus raw
            metadata spread.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        name: str = "knowledge_base",
        topic_index: Any | None = None,
        dedup_key: Callable[[dict[str, Any]], Hashable] | None = None,
        source_id_fn: Callable[[dict[str, Any]], str] | None = None,
        metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._kb = kb
        self._name = name
        self.topic_index = topic_index
        self._dedup_key = dedup_key or default_dedup_key
        self._source_id_fn = source_id_fn or default_source_id
        self._metadata_fn = metadata_fn or default_metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "vector_kb"

    def get_schema(self) -> SourceSchema | None:
        """Vector sources have no structured filter dimensions."""
        return None

    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        """Execute each text query via KB semantic search.

        Args:
            intent: Uses ``text_queries`` for semantic search and the
                ``filters[self.name]`` slice as metadata filters for the
                underlying vector store.
            top_k: Max results per query.
            score_threshold: Minimum similarity score.

        Returns:
            Deduplicated results sorted by similarity (descending).
        """
        all_results: list[SourceResult] = []
        seen: set[Hashable] = set()

        # Pick up the filter slice keyed by our source name, matching the
        # convention DatabaseSource uses (database.py:300). Empty slice
        # or missing slice both mean "no filter" — fall through to the
        # KB's no-filter path so existing consumers see unchanged
        # behavior.
        source_filters = intent.filters.get(self._name) or None

        for tq in intent.text_queries:
            try:
                raw_results = await self._kb.query(
                    tq, k=top_k, filter_metadata=source_filters,
                )
            except Exception:
                logger.warning(
                    "KB query failed for '%s' in source '%s'",
                    tq, self._name, exc_info=True,
                )
                continue

            for r in raw_results:
                similarity = r.get("similarity", 1.0)
                if similarity < score_threshold:
                    continue

                # Identity callables are consumer-supplied and therefore
                # may raise. Isolate each record: on failure, log and
                # skip the record so retrieval degrades gracefully rather
                # than aborting the entire turn (dropping results from
                # sibling sources too).
                try:
                    key = self._dedup_key(r)
                    if key in seen:
                        continue
                    source_id = self._source_id_fn(r)
                    metadata = self._metadata_fn(r)
                except Exception:
                    logger.warning(
                        "Identity callable raised for a result in "
                        "source '%s' (query=%r); skipping record",
                        self._name, tq, exc_info=True,
                    )
                    continue

                seen.add(key)
                all_results.append(SourceResult(
                    content=r.get("text", ""),
                    source_id=source_id,
                    source_name=self._name,
                    source_type="vector_kb",
                    relevance=similarity,
                    metadata=metadata,
                ))

        # Sort by relevance descending
        all_results.sort(key=lambda r: r.relevance, reverse=True)
        return all_results

    async def close(self) -> None:
        """Close the underlying knowledge base."""
        await self._kb.close()

    def providers(self) -> dict[str, Any]:
        """Delegate to the KB's provider registry."""
        return self._kb.providers()

    def set_provider(self, role: str, provider: Any) -> bool:
        """Delegate to the KB's provider injection."""
        return self._kb.set_provider(role, provider)
