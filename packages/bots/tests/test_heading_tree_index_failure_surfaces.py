"""A heading-tree index that cannot run must not read as one with no match.

``HeadingTreeIndex`` is one of the two implementations of the ``TopicIndex``
protocol, and it answered the protocol's unstated failure question the same
wrong way its sibling did: a lazy index with no way to fetch seeds returned
an empty list, and a vector query that raised was logged and turned into
one too.

Both matter because of what the caller does with an empty list. The
grounded retrieval loop reads it as a vocabulary gap and reroutes the turn
to plain text retrieval, reporting that the index "returned empty" --- so a
dead retrieval strategy and an unreachable store both arrived as a claim
about the *corpus*, on every turn, at INFO.

These pin the three answers apart: cannot run at all
(``StrategyUnavailable``, the caller falls back and says why), broke while
running (propagates, the caller drops the source with its cause), and ran
and matched nothing (``[]``, which still means exactly that).

The sibling file is ``packages/data/tests/test_cluster_index_failure_surfaces.py``;
these are deliberately the same cases, because the two implementations owe
the same contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_bots.knowledge.sources.heading_tree import (
    HeadingTreeConfig,
    HeadingTreeIndex,
)
from dataknobs_data.sources.base import SourceResult, StrategyUnavailable


def _chunk(chunk_id: str, headings: list[str], levels: list[int], content: str) -> SourceResult:
    return SourceResult(
        content=content,
        source_id=chunk_id,
        source_name="kb",
        source_type="vector_kb",
        relevance=1.0,
        metadata={"headings": headings, "heading_levels": levels},
    )


#: A security section with one child, so an expanded region has something
#: to expand *to* --- an index that matched only its own seed would satisfy
#: the control test without doing the thing under test.
_CHUNKS = [
    _chunk("intro", ["1. Introduction"], [1], "OAuth provides authorization flows"),
    _chunk(
        "sec",
        ["10. Security Considerations"],
        [1],
        "This section describes security considerations",
    ),
    _chunk(
        "csrf",
        ["10. Security Considerations", "10.12 CSRF"],
        [1, 2],
        "CSRF attacks exploit trust in authorized users",
    ),
]


async def _seed_fn(
    query: str,
    top_k: int,
    *,
    filter_metadata: dict[str, Any] | None = None,
) -> list[SourceResult]:
    """Return every chunk as a seed, which is what a reachable store does."""
    return _CHUNKS[:top_k]


async def test_a_lazy_index_with_no_vector_query_fn_says_so() -> None:
    """Not an empty result: empty means "ran, matched nothing".

    Lazy mode seeds through the vector path whatever the entry strategy,
    so this holds for all three rather than only for ``"vector"``.
    """
    for strategy in ("vector", "both", "heading_match"):
        index = HeadingTreeIndex(
            config=HeadingTreeConfig(entry_strategy=strategy),
            source_name="kb",
        )
        with pytest.raises(StrategyUnavailable, match="no vector_query_fn") as caught:
            await index.resolve("security")
        assert "kb" in str(caught.value), "the message does not say which source"


async def test_an_eager_index_with_no_vector_query_fn_still_resolves() -> None:
    """The asymmetry: only lazy mode seeds, so only lazy mode needs it.

    A guard that required the callable unconditionally would break every
    eagerly-built index, which is the ordinary way to build one.
    """
    index = HeadingTreeIndex.from_chunks(
        _CHUNKS,
        config=HeadingTreeConfig(entry_strategy="heading_match"),
        source_name="kb",
    )

    results = await index.resolve("security considerations")

    assert {r.source_id for r in results} >= {"sec", "csrf"}


async def test_a_failing_vector_query_does_not_read_as_no_seeds() -> None:
    """A store that cannot be reached is not a store with no seeds in it.

    This is the swallow that the sibling index had removed and this one
    kept: it caught everything the vector query raised, logged at WARNING,
    and returned ``[]`` --- which the retrieval loop then reported as a
    vocabulary gap. The loop already drops a source that raises, with its
    cause, which is the right disposition for a store that is down.
    """

    async def failing_seeds(
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        raise RuntimeError("vector store unreachable")

    index = HeadingTreeIndex(
        vector_query_fn=failing_seeds,
        config=HeadingTreeConfig(entry_strategy="vector"),
        source_name="kb",
    )

    with pytest.raises(RuntimeError, match="vector store unreachable"):
        await index.resolve("security")


async def test_a_working_index_with_no_match_still_answers_empty() -> None:
    """The negative control: empty still means empty.

    Without this the tests above are satisfied by an index that raises
    indiscriminately, and the distinction they exist to draw is not
    actually drawn.
    """
    index = HeadingTreeIndex(
        vector_query_fn=_seed_fn,
        config=HeadingTreeConfig(entry_strategy="heading_match"),
        source_name="kb",
    )

    assert await index.resolve("quantum chromodynamics") == []
