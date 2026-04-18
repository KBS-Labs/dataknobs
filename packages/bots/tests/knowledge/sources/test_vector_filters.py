"""Behavioral tests for filter passthrough in ``VectorKnowledgeSource``.

Covers Item 95: ``intent.filters[self._name]`` must flow through to
``KnowledgeBase.query(filter_metadata=...)`` with scalar-equality
semantics.
"""

from __future__ import annotations

from typing import Any

from dataknobs_data.sources.base import RetrievalIntent

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource


class FilteringKnowledgeBase(KnowledgeBase):
    """Test KB that respects ``filter_metadata`` on ``query()``.

    Returns records where every filter key/value pair matches the
    record's metadata (scalar equality). Matches the real
    ``RAGKnowledgeBase`` + ``MemoryVectorStore`` filter semantics.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.last_filter: dict[str, Any] | None = None
        self.query_count = 0

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.query_count += 1
        self.last_filter = filter_metadata
        if not filter_metadata:
            return self._records[:k]
        matched = [
            r for r in self._records
            if all(
                r.get("metadata", {}).get(fk) == fv
                for fk, fv in filter_metadata.items()
            )
        ]
        return matched[:k]

    async def close(self) -> None:
        pass


def _record(
    text: str,
    source: str,
    chunk_index: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md = {"chunk_index": chunk_index}
    if metadata:
        md.update(metadata)
    return {
        "text": text,
        "source": source,
        "heading_path": "",
        "similarity": 0.9,
        "metadata": md,
    }


async def test_filter_slice_forwarded_to_kb() -> None:
    """Slice keyed by source name flows through as ``filter_metadata``."""
    kb = FilteringKnowledgeBase([
        _record("tension claim", "fd-02.md", 0, {"entity_type": "tension"}),
        _record("gap claim", "fd-02.md", 1, {"entity_type": "gap"}),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["claims"],
        filters={"docs": {"entity_type": "tension"}},
    )
    results = await source.query(intent)

    assert kb.last_filter == {"entity_type": "tension"}
    assert len(results) == 1
    assert results[0].content == "tension claim"


async def test_no_filter_slice_backward_compat() -> None:
    """Empty ``intent.filters`` passes ``None`` (preserves old behavior)."""
    kb = FilteringKnowledgeBase([
        _record("a", "doc.md", 0),
        _record("b", "doc.md", 1),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(text_queries=["anything"])
    results = await source.query(intent)

    assert kb.last_filter is None
    assert len(results) == 2


async def test_filter_slice_for_different_source_ignored() -> None:
    """Filter slice keyed by a different source name is ignored."""
    kb = FilteringKnowledgeBase([
        _record("a", "doc.md", 0, {"entity_type": "tension"}),
        _record("b", "doc.md", 1, {"entity_type": "gap"}),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["anything"],
        filters={"other_source": {"entity_type": "tension"}},
    )
    results = await source.query(intent)

    assert kb.last_filter is None
    assert len(results) == 2


async def test_empty_slice_treated_as_no_filter() -> None:
    """``intent.filters[name] == {}`` should not narrow results to nothing."""
    kb = FilteringKnowledgeBase([
        _record("a", "doc.md", 0),
        _record("b", "doc.md", 1),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["anything"],
        filters={"docs": {}},
    )
    results = await source.query(intent)

    assert kb.last_filter is None
    assert len(results) == 2


async def test_multi_key_filter_forwarded() -> None:
    """Multi-key filter dicts flow through unchanged (AND semantics)."""
    kb = FilteringKnowledgeBase([
        _record(
            "match",
            "fd-02.md",
            0,
            {"entity_type": "gap_analysis", "domain": "FD-02"},
        ),
        _record(
            "wrong domain",
            "fd-02.md",
            1,
            {"entity_type": "gap_analysis", "domain": "FD-03"},
        ),
        _record(
            "wrong type",
            "fd-02.md",
            2,
            {"entity_type": "tension", "domain": "FD-02"},
        ),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["claims"],
        filters={
            "docs": {"entity_type": "gap_analysis", "domain": "FD-02"},
        },
    )
    results = await source.query(intent)

    assert kb.last_filter == {
        "entity_type": "gap_analysis",
        "domain": "FD-02",
    }
    assert len(results) == 1
    assert results[0].content == "match"
