"""An index that cannot run must not be reported as a vocabulary gap.

The grounded retrieval loop has one fallback from the topic-index path to
plain text retrieval, and two very different conditions reach it. An index
that ran and matched nothing is a vocabulary gap, which is what the
fallback is *for*; the turn is served and nothing is wrong. An index that
could not run at all -- no embedder, no way to fetch seeds -- is a wiring
fault that will take the same branch on every turn until someone fixes it.

Both used to arrive as an empty list, so both were logged as
"topic index returned empty", at INFO, naming the wrong cause for one of
them. The operator's only signal that a source's whole retrieval strategy
was dead pointed at the corpus instead of at the configuration.

A third condition reaches neither: an index that *broke*. It must not fall
back at all -- the loop's own guard drops such a source, with its cause --
and that disposition survives only while the catch stays narrow.

These pin the distinction from the consumer's side: one fallback, two
records, and the unresolvable one says what is missing.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dataknobs_bots.knowledge.base import KnowledgeBase
from dataknobs_bots.knowledge.sources.factory import build_topic_index
from dataknobs_bots.knowledge.sources.heading_tree import (
    HeadingTreeConfig,
    HeadingTreeIndex,
)
from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource
from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig
from dataknobs_data.sources.base import RetrievalIntent, SourceResult

#: The module holding the topic-index branch, so a record from anywhere
#: else cannot satisfy the assertions below.
STRATEGY_LOGGER = "dataknobs_bots.reasoning.grounded"


class _KnowledgeBase(KnowledgeBase):
    """A knowledge base with one document, reachable by text query.

    Real rather than scripted-empty: these tests turn on the fallback
    still *serving the turn*, so the text path has to have something to
    find.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def query(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [
            {
                "text": "Session tokens expire after thirty minutes.",
                "source": "security.md",
                "similarity": 0.81,
                "metadata": {"headings": ["Security"], "chunk_index": 0},
            }
        ][:k]

    async def close(self) -> None:
        pass


def _chunk(chunk_id: str, content: str, headings: list[str]) -> SourceResult:
    return SourceResult(
        content=content,
        source_id=chunk_id,
        source_name="docs",
        source_type="vector_kb",
        relevance=1.0,
        metadata={"headings": headings, "heading_levels": list(range(1, len(headings) + 1))},
    )


async def _retrieve(index: HeadingTreeIndex, kb: _KnowledgeBase) -> dict[str, list[SourceResult]]:
    strategy = GroundedReasoning(config=GroundedReasoningConfig())
    strategy.add_source(VectorKnowledgeSource(kb, name="docs", topic_index=index))
    return await strategy._retrieve_from_sources(
        RetrievalIntent(text_queries=["session tokens"]),
        user_message="how long do session tokens last",
    )


async def test_an_unresolvable_topic_index_falls_back_and_names_the_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lazy index with no way to fetch seeds cannot run, and says which.

    The turn is still served -- that is the half this must not regress --
    but the record is a WARNING naming the index, not the INFO that
    reports a corpus with nothing on the topic.
    """
    kb = _KnowledgeBase()
    index = HeadingTreeIndex(
        config=HeadingTreeConfig(entry_strategy="vector"),
        source_name="docs",
    )

    with caplog.at_level(logging.DEBUG, logger=STRATEGY_LOGGER):
        results = await _retrieve(index, kb)

    # The turn is served by the fallback, exactly as before.
    assert [r.content for r in results["docs"]] == ["Session tokens expire after thirty minutes."]
    assert kb.queries, "the fallback did not run"

    records = [r for r in caplog.records if r.name == STRATEGY_LOGGER]
    assert records, "the fallback was taken with no record at all"
    assert not any("returned empty" in r.getMessage() for r in records), (
        "an index that could not run was reported as one that found nothing: "
        f"{[r.getMessage() for r in records]}"
    )
    named = [r for r in records if r.levelno >= logging.WARNING]
    assert named, (
        "a dead retrieval strategy was reported below WARNING: "
        f"{[(r.levelname, r.getMessage()) for r in records]}"
    )
    assert "docs" in named[0].getMessage()


async def test_an_empty_topic_index_still_reports_a_vocabulary_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other branch, unchanged -- and the reason the first one matters.

    Without this the test above is satisfied by warning about every
    fallback, which would collapse the two conditions again in the other
    direction.
    """
    kb = _KnowledgeBase()
    index = HeadingTreeIndex.from_chunks(
        [_chunk("intro", "Getting started with the toolkit", ["Introduction"])],
        config=HeadingTreeConfig(entry_strategy="heading_match"),
        source_name="docs",
    )

    with caplog.at_level(logging.DEBUG, logger=STRATEGY_LOGGER):
        results = await _retrieve(index, kb)

    assert [r.content for r in results["docs"]] == ["Session tokens expire after thirty minutes."]
    messages = [r.getMessage() for r in caplog.records if r.name == STRATEGY_LOGGER]
    assert any("returned empty" in m for m in messages), messages
    assert not any(
        r.levelno >= logging.WARNING for r in caplog.records if r.name == STRATEGY_LOGGER
    ), "a working index with no match was reported as a fault"


async def test_a_cluster_index_over_a_kb_that_cannot_embed_names_the_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The configuration that reaches this in practice, built by the factory.

    ``build_topic_index`` always threads a ``vector_query_fn``, so the
    seeding half of this condition is not reachable through config. The
    embedder half is: a ``cluster`` index takes its embedder from the KB,
    and ``_build_embedder`` returns ``None`` for a KB with no ``embed``
    method. Nothing rejects that pairing at build time --- the index is
    constructed, logged as built, and then cannot resolve anything.

    So the whole strategy a config author asked for was silently inert,
    and the only record said the corpus had nothing on the topic.
    """
    kb = _KnowledgeBase()
    assert not hasattr(kb, "embed"), "this test needs a KB the factory cannot embed with"

    index = build_topic_index({"type": "cluster"}, kb, source_name="docs")
    assert index is not None, "the factory built no index at all"

    with caplog.at_level(logging.DEBUG, logger=STRATEGY_LOGGER):
        results = await _retrieve(index, kb)

    assert [r.content for r in results["docs"]] == ["Session tokens expire after thirty minutes."]
    named = [
        r for r in caplog.records if r.name == STRATEGY_LOGGER and r.levelno >= logging.WARNING
    ]
    assert named, [(r.levelname, r.getMessage()) for r in caplog.records]
    assert "embedder" in named[0].getMessage()


async def test_an_index_that_breaks_drops_the_source_instead_of_falling_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The third condition, and the reason the catch is narrow.

    An index that *broke* is neither of the two above. It must not fall
    back, because a fallback would serve the turn from a source whose
    retrieval strategy just failed for an unknown reason and report
    nothing wrong; the loop's own guard drops such a source, with its
    cause, exactly as it does for a source that raises without an index.

    That disposition is bought entirely by ``except StrategyUnavailable``
    being narrow. Widening it to ``except Exception`` would turn every
    broken index into a silent fallback --- the defect this whole change
    removed, reintroduced one layer up. This is the test that fails if
    someone does.
    """

    async def failing_seeds(
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        raise RuntimeError("vector store unreachable")

    kb = _KnowledgeBase()
    index = HeadingTreeIndex(
        vector_query_fn=failing_seeds,
        config=HeadingTreeConfig(entry_strategy="vector"),
        source_name="docs",
    )

    with caplog.at_level(logging.DEBUG, logger=STRATEGY_LOGGER):
        results = await _retrieve(index, kb)

    assert "docs" not in results, "a source whose index broke was kept for the turn"
    assert kb.queries == [], (
        "the fallback ran for an index that broke, which is the "
        "unavailable-index disposition applied to the wrong condition"
    )

    records = [r for r in caplog.records if r.name == STRATEGY_LOGGER]
    assert not any("returned empty" in r.getMessage() for r in records), (
        f"an index that broke was reported as one that found nothing: "
        f"{[r.getMessage() for r in records]}"
    )
    dropped = [r for r in records if "skipping" in r.getMessage()]
    assert dropped, [(r.levelname, r.getMessage()) for r in records]
    assert dropped[0].exc_info is not None, "the source was dropped without its cause"
