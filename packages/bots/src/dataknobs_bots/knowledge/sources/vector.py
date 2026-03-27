"""Vector knowledge base grounded source.

Wraps an existing :class:`KnowledgeBase` as a :class:`GroundedSource`,
using ``text_queries`` from :class:`RetrievalIntent` for semantic search.
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
    SourceSchema,
)

from dataknobs_bots.knowledge.base import KnowledgeBase

logger = logging.getLogger(__name__)


class VectorKnowledgeSource(GroundedSource):
    """Grounded source backed by a :class:`KnowledgeBase`.

    Uses ``intent.text_queries`` for semantic similarity search.  Has
    no filter schema — all filtering is similarity-based.  This is the
    adapter that makes existing ``RAGKnowledgeBase`` instances usable
    in the grounded source pipeline.

    Optionally carries a :attr:`topic_index` for heading-tree or
    cluster-based retrieval.  When present, the grounded strategy
    uses the topic index instead of standard text_queries.

    Args:
        kb: The knowledge base to query.
        name: Unique source name for provenance tracking.
        topic_index: Optional topic index for structured retrieval.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        name: str = "knowledge_base",
        topic_index: Any | None = None,
    ) -> None:
        self._kb = kb
        self._name = name
        self.topic_index = topic_index

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
            intent: Only ``text_queries`` is used.
            top_k: Max results per query.
            score_threshold: Minimum similarity score.

        Returns:
            Deduplicated results sorted by similarity (descending).
        """
        all_results: list[SourceResult] = []
        seen: set[tuple[str, int]] = set()

        for tq in intent.text_queries:
            try:
                raw_results = await self._kb.query(tq, k=top_k)
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

                # Deduplicate by (source_file, chunk_index)
                source_file = r.get("source", "")
                metadata = r.get("metadata", {})
                chunk_index = metadata.get("chunk_index", id(r))
                key = (source_file, chunk_index)
                if key in seen:
                    continue
                seen.add(key)

                all_results.append(SourceResult(
                    content=r.get("text", ""),
                    source_id=f"{source_file}:chunk_{chunk_index}",
                    source_name=self._name,
                    source_type="vector_kb",
                    relevance=similarity,
                    metadata={
                        "heading_path": r.get("heading_path", ""),
                        "source": source_file,
                        **metadata,
                    },
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
