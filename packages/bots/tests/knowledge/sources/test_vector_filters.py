"""Behavioral tests for filter passthrough in ``VectorKnowledgeSource``.

Covers Item 95: ``intent.filters[self._name]`` must flow through to
``KnowledgeBase.query(filter_metadata=...)`` with scalar-equality
semantics.
"""

from __future__ import annotations

from typing import Any

from dataknobs_data.sources.base import RetrievalIntent

from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource
from dataknobs_bots.testing import ScriptedKnowledgeBase


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
    kb = ScriptedKnowledgeBase([
        _record("alpha entry", "doc-1.md", 0, {"category": "alpha"}),
        _record("beta entry", "doc-1.md", 1, {"category": "beta"}),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["entries"],
        filters={"docs": {"category": "alpha"}},
    )
    results = await source.query(intent)

    assert kb.last_filter == {"category": "alpha"}
    assert len(results) == 1
    assert results[0].content == "alpha entry"


async def test_no_filter_slice_backward_compat() -> None:
    """Empty ``intent.filters`` passes ``None`` (preserves old behavior)."""
    kb = ScriptedKnowledgeBase([
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
    kb = ScriptedKnowledgeBase([
        _record("a", "doc.md", 0, {"category": "alpha"}),
        _record("b", "doc.md", 1, {"category": "beta"}),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["anything"],
        filters={"other_source": {"category": "alpha"}},
    )
    results = await source.query(intent)

    assert kb.last_filter is None
    assert len(results) == 2


async def test_empty_slice_treated_as_no_filter() -> None:
    """``intent.filters[name] == {}`` should not narrow results to nothing."""
    kb = ScriptedKnowledgeBase([
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
    kb = ScriptedKnowledgeBase([
        _record(
            "match",
            "doc-1.md",
            0,
            {"category": "premium", "section": "intro"},
        ),
        _record(
            "wrong section",
            "doc-1.md",
            1,
            {"category": "premium", "section": "advanced"},
        ),
        _record(
            "wrong category",
            "doc-1.md",
            2,
            {"category": "standard", "section": "intro"},
        ),
    ])
    source = VectorKnowledgeSource(kb, name="docs")

    intent = RetrievalIntent(
        text_queries=["entries"],
        filters={
            "docs": {"category": "premium", "section": "intro"},
        },
    )
    results = await source.query(intent)

    assert kb.last_filter == {
        "category": "premium",
        "section": "intro",
    }
    assert len(results) == 1
    assert results[0].content == "match"
