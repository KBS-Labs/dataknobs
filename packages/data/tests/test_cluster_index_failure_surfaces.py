"""A topic index that cannot run must not read as one that found no topics.

``ClusterTopicIndex.resolve`` wrapped the two calls its answer depends on
-- embedding the query, and fetching seeds from the vector store -- logged
whatever they raised, and returned an empty list. Every way of failing
therefore arrived at the caller as "this index has nothing on that topic".

That is worse here than it is for a plain source, because of what the
caller does next. The grounded retrieval loop treats an empty topic index
as a vocabulary gap and *falls back* to plain text retrieval, logging that
the index returned empty. So a broken embedder did not merely read as a
quiet index: it silently rerouted retrieval to a different strategy, and
said so in a message naming the wrong cause.

The loop already guards each source and drops one that raises, with its
cause. These pin the two whole-operation failures against that guard, and
keep the one place a caught failure is still right -- a single seed chunk
that will not embed, which is dropped so the rest can still cluster.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_data.sources.base import SourceResult
from dataknobs_data.sources.cluster_index import ClusterTopicConfig, ClusterTopicIndex


def _chunk(chunk_id: str, content: str) -> SourceResult:
    return SourceResult(
        content=content,
        source_id=chunk_id,
        source_name="kb",
        source_type="vector_kb",
        relevance=1.0,
    )


#: Three chunks over two topics, with unit embeddings on separate axes so
#: the clustering is decided by the vectors rather than by tuning. Two of
#: them share a topic, so dropping the third still leaves a pool that can
#: cluster -- which is what the per-chunk skip has to be shown against.
_CHUNKS = [
    _chunk("a1", "authentication login security tokens"),
    _chunk("a2", "authentication password hashing security"),
    _chunk("b1", "database query optimization indexes"),
]

_EMBEDDINGS = {
    "a1": [1.0, 0.0, 0.0, 0.0],
    "a2": [0.99, 0.1, 0.0, 0.0],
    "b1": [0.0, 1.0, 0.0, 0.0],
}

_CONFIG = ClusterTopicConfig(cluster_threshold=0.5)


async def _embed_near_a(text: str) -> list[float]:
    """Embed anything as a vector near the first cluster."""
    return [0.95, 0.05, 0.0, 0.0]


async def _seed_fn(
    query: str,
    top_k: int,
    *,
    filter_metadata: dict[str, Any] | None = None,
) -> list[SourceResult]:
    """Return every chunk as a seed, which is what a reachable store does."""
    return _CHUNKS[:top_k]


def _lazy(**kwargs: Any) -> ClusterTopicIndex:
    return ClusterTopicIndex(config=_CONFIG, source_name="kb", **kwargs)


async def test_a_failing_query_embedding_does_not_read_as_no_topics() -> None:
    """An embedder that raises is not an index with nothing on the topic.

    This is the failure the caller most needs to see, because the empty
    list it used to get is the one value the retrieval loop reads as a
    vocabulary gap worth falling back from.
    """

    async def failing_embed(text: str) -> list[float]:
        raise RuntimeError("embedding service unreachable")

    idx = _lazy(embed_fn=failing_embed, vector_query_fn=_seed_fn)

    with pytest.raises(RuntimeError, match="embedding service unreachable"):
        await idx.resolve("authentication")


async def test_a_failing_seed_fetch_does_not_read_as_no_seeds() -> None:
    """A vector store that cannot be reached is not a store with no seeds.

    The seed fetch is this index's retrieval call, so this is the exact
    shape the sibling ``DatabaseSource`` fix addressed one file over.
    """

    async def failing_seeds(
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        raise RuntimeError("vector store unreachable")

    idx = _lazy(embed_fn=_embed_near_a, vector_query_fn=failing_seeds)

    with pytest.raises(RuntimeError, match="vector store unreachable"):
        await idx.resolve("authentication")


async def test_every_seed_failing_to_embed_does_not_read_as_no_seeds() -> None:
    """A pool where nothing embedded is not a pool with nothing in it.

    Seeds are embedded one at a time so a single bad chunk can be dropped.
    When every one of them fails, the per-chunk tolerance has stopped
    describing what happened -- the embedder is broken -- and returning an
    empty pool reports the wrong one.
    """

    async def embed_query_only(text: str) -> list[float]:
        if text == "authentication":
            return [0.95, 0.05, 0.0, 0.0]
        raise RuntimeError("embedding service unreachable")

    idx = _lazy(embed_fn=embed_query_only, vector_query_fn=_seed_fn)

    with pytest.raises(RuntimeError, match="seed chunk"):
        await idx.resolve("authentication")


async def test_one_seed_failing_to_embed_is_still_skipped() -> None:
    """The per-chunk tolerance survives: one bad seed does not sink the turn.

    This is the counterpart to the test above and the reason that one
    cannot simply propagate: dropping a chunk that will not embed is what
    lets the rest of the pool cluster, and is the case the catch was
    written for.
    """
    by_content = {c.content: _EMBEDDINGS[c.source_id] for c in _CHUNKS}

    async def embed_all_but_b1(text: str) -> list[float]:
        if text.startswith("database"):
            raise RuntimeError("that one chunk will not embed")
        if text == "authentication":
            return [0.95, 0.05, 0.0, 0.0]
        return by_content[text]

    idx = _lazy(embed_fn=embed_all_but_b1, vector_query_fn=_seed_fn)

    results = await idx.resolve("authentication")

    # Sorted because the index orders by relevance, which is not what this
    # is about: the claim is that the two embeddable chunks survived the
    # third being dropped.
    assert sorted(r.source_id for r in results) == ["a1", "a2"]


async def test_a_working_index_with_no_match_still_answers_empty() -> None:
    """The negative control: empty still means empty.

    Without this the tests above are satisfied by an index that raises
    indiscriminately, and the distinction they exist to draw is not
    actually drawn.
    """

    async def embed_far_from_both(text: str) -> list[float]:
        if text == "unrelated":
            return [0.0, 0.0, 1.0, 0.0]
        return _EMBEDDINGS.get("a1", [1.0, 0.0, 0.0, 0.0])

    idx = _lazy(embed_fn=embed_far_from_both, vector_query_fn=_seed_fn)

    assert await idx.resolve("unrelated") == []
