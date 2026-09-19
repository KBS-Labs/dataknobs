# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A semantic index, seen through the retrieval stack's source interface.

An **adapter, not a base class**. The index must stay usable without the
retrieval stack --- a caller holding a vocabulary and a store should not have
to know what a grounded pipeline is --- so the dependency runs this way round:
this module knows about both, and neither knows about this one.

It declares no schema. A semantic index has no structured filter dimensions to
extract intent against; it takes the text queries every source receives and
answers with the nearest rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataknobs_data.sources.base import GroundedSource, SourceResult

if TYPE_CHECKING:
    from dataknobs_data.sources.base import RetrievalIntent
    from dataknobs_data.vector.semantic_index import SemanticIndex

__all__ = ["SemanticIndexSource"]

#: What this source calls itself in a composed pipeline's provenance.
SOURCE_TYPE = "semantic_index"


class SemanticIndexSource(GroundedSource):
    """One :class:`~dataknobs_data.vector.semantic_index.SemanticIndex`, as a source.

    Example:
        ```python
        source = SemanticIndexSource(index, name="catalog")
        results = await source.query(RetrievalIntent(text_queries=["acme widget"]))
        ```
    """

    def __init__(self, index: SemanticIndex, *, name: str = "semantic_index") -> None:
        """Wrap an index that is already built.

        Args:
            index: The index to search. Its lifecycle is its owner's --- this
                adapter opens nothing and closes nothing, which is why
                :meth:`close` stays the base class's no-op.
            name: What this source is called in provenance and in the
                ``filters`` slice keyed by source name. Defaulted so the
                single-source case needs no argument, and settable because a
                pipeline holding two indexes needs to tell them apart.
        """
        self._index = index
        self._name = name

    @property
    def name(self) -> str:
        """This source's identifier within a composed pipeline."""
        return self._name

    @property
    def source_type(self) -> str:
        """The category string a result carries back."""
        return SOURCE_TYPE

    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        """Search the index for every text query, best first.

        The queries are searched as a batch rather than in a loop, which is
        what the index's batch member is for and what lets an embedder send
        one request instead of N.

        **Merged across queries, deduplicated by id, keeping the best score.**
        Two phrasings of one question that both reach a row should not put
        that row in the answer twice, and the score a caller acts on is the
        best evidence found rather than whichever query ran last.

        Args:
            intent: The extracted intent. Only ``text_queries`` is read ---
                this source declares no schema, so it has no ``filters``
                slice of its own.
            top_k: Maximum results to return, after the merge.
            score_threshold: Minimum relevance to include. The index applies
                it per query as well, so a hit below it never reaches here.

                **Passed through as given, including ``0.0``.** Written
                ``score_threshold or None`` this argument turned the filter
                off for exactly one value --- the documented default, and a
                meaningful cut rather than an absent one, since cosine
                similarity runs to ``-1`` and zero is *drop anything pointing
                the wrong way*. A caller who said nothing got the opposite of
                what the signature says they asked for.

        Returns:
            Results sorted by relevance, descending, at most *top_k* of them.
        """
        queries = [text for text in intent.text_queries if text]
        if not queries:
            return []

        best: dict[str, SourceResult] = {}
        batches = await self._index.search_batch(queries, k=top_k, threshold=score_threshold)
        for hits in batches:
            for hit in hits:
                result = SourceResult(
                    content=hit.source_text or "",
                    source_id=hit.record.id or "",
                    source_name=self._name,
                    source_type=SOURCE_TYPE,
                    relevance=hit.score,
                    metadata=dict(hit.metadata),
                )
                previous = best.get(result.source_id)
                if previous is None or result.relevance > previous.relevance:
                    best[result.source_id] = result

        ranked = sorted(best.values(), key=lambda result: result.relevance, reverse=True)
        return ranked[:top_k]
