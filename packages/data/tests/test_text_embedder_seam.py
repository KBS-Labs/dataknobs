"""The one shape for "turn text into vectors", and the cache that wraps it.

Eight incompatible embedding-callable spellings across this package, none of
them matching what ``AsyncLLMProvider.embed`` returns, is what
:class:`TextEmbedder` replaces. These tests pin the three properties the rest
of the seam is allowed to assume: the protocol is satisfiable, the published
test double actually satisfies it, and :class:`CachedEmbedder` keys on the
model as well as the text.

That last one is the reason this file exists rather than being folded into a
larger suite. A cache keyed on text alone does not fail loudly after a model
swap --- every lookup *succeeds*, and hands back vectors from a model that is
no longer in use, in a vector space the new model knows nothing about. There
is no exception to catch and no log line to read: the similarities are simply
wrong. So the invalidation is tested directly rather than inferred from the
key's construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import (
    CachedEmbedder,
    TextEmbedder,
    VectorCache,
    embedding_cache_key,
)


class RecordingCache:
    """A ``VectorCache`` that also says what was asked of it.

    Not a mock: it is a real cache with real hit/miss behaviour, keyed the way
    the shipped ``dataknobs-llm`` caches key. The call log is what lets a test
    assert that a hit did *not* reach the inner embedder, which is the only
    externally visible difference between a working cache and a broken one.
    """

    def __init__(self) -> None:
        self.store: dict[str, list[float]] = {}
        self.get_calls: list[tuple[str, list[str]]] = []
        self.put_calls: list[tuple[str, list[str]]] = []

    async def get_batch(self, model: str, texts: list[str]) -> list[list[float] | None]:
        self.get_calls.append((model, list(texts)))
        return [self.store.get(embedding_cache_key(model, t)) for t in texts]

    async def put_batch(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        self.put_calls.append((model, list(texts)))
        for text, vec in zip(texts, vectors, strict=True):
            self.store[embedding_cache_key(model, text)] = vec


class CountingEmbedder:
    """A ``TextEmbedder`` that counts the texts it was actually asked to embed."""

    def __init__(self, dimensions: int = 4, model_id: str = "counter") -> None:
        self._dimensions = dimensions
        self._model_id = model_id
        self.seen: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts):
        self.seen.append(list(texts))
        return [[float(len(t))] * self._dimensions for t in texts]


# --------------------------------------------------------------------------
# The protocol, and the double that has to satisfy it
# --------------------------------------------------------------------------


def test_deterministic_embedder_satisfies_the_protocol() -> None:
    """The published test double is checkable against the published protocol."""
    assert isinstance(DeterministicEmbedder(dimensions=8), TextEmbedder)


def test_recording_cache_satisfies_the_cache_port() -> None:
    """``VectorCache`` is satisfied structurally --- nothing subclasses it."""
    assert isinstance(RecordingCache(), VectorCache)


async def test_embed_returns_one_vector_per_text_in_order() -> None:
    embedder = DeterministicEmbedder(dimensions=8)

    vectors = await embedder.embed(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)
    assert vectors[0] == (await embedder.embed(["alpha"]))[0]


async def test_empty_batch_is_not_an_error() -> None:
    """An empty input is an empty output, not a raise --- callers batch blindly."""
    assert await DeterministicEmbedder().embed([]) == []


async def test_same_text_embeds_identically_across_instances() -> None:
    """Stability is across processes, so two instances must agree."""
    first = await DeterministicEmbedder(dimensions=16).embed(["repeatable"])
    second = await DeterministicEmbedder(dimensions=16).embed(["repeatable"])

    assert first == second


async def test_distinct_texts_are_discriminable_by_cosine() -> None:
    """The property ``text_embedding`` does not have, and the reason for the new draw.

    ``text_embedding`` draws every component from ``[0, 1)``, so any two of its
    vectors are highly cosine-similar whatever their texts --- documented on
    ``chroma_embedding_function`` as "unusable for asserting a *ranking*". A
    published embedder aimed at resolver tests has to survive the ranking case,
    so unrelated texts must land near-orthogonal rather than near-parallel.
    """
    vectors = await DeterministicEmbedder(dimensions=64).embed(["alpha", "beta", "gamma", "delta"])
    matrix = np.array(vectors)

    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    similarities = matrix @ matrix.T
    off_diagonal = similarities[~np.eye(len(vectors), dtype=bool)]
    assert np.all(np.abs(off_diagonal) < 0.5), similarities


async def test_model_id_changes_the_vector_space() -> None:
    """Two model identities must not put one text in the same place.

    This is what makes the double usable for testing a staleness path: if a
    ``model_id`` swap produced identical vectors, no test could distinguish
    "invalidated and re-embedded" from "served the old vector".
    """
    old = await DeterministicEmbedder(dimensions=16, model_id="v1").embed(["shared"])
    new = await DeterministicEmbedder(dimensions=16, model_id="v2").embed(["shared"])

    assert old != new


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        DeterministicEmbedder(dimensions=0)


# --------------------------------------------------------------------------
# CachedEmbedder — acceptance criterion 4
# --------------------------------------------------------------------------


async def test_a_model_swap_invalidates_rather_than_serving_stale() -> None:
    """THE reproduce-first test for this change: the key includes ``model_id``.

    Written against a key of ``(model_id, text)``. Run against a
    ``CachedEmbedder`` that passed a constant model instead of
    ``inner.model_id``, it fails on the final assertion --- the swapped-in
    embedder is never called and ``v2`` receives ``v1``'s vectors, silently.
    That is the defect this asserts against, and it has no louder symptom.
    """
    cache = RecordingCache()

    first = CountingEmbedder(model_id="embed-v1")
    warmed = await CachedEmbedder(first, cache).embed(["shared text"])
    assert first.seen == [["shared text"]]

    # Same cache, same text, different model. A cache keyed on text alone hits
    # here and hands back `first`'s vector.
    second = CountingEmbedder(model_id="embed-v2")
    after_swap = await CachedEmbedder(second, cache).embed(["shared text"])

    assert second.seen == [["shared text"]], (
        "the swapped-in embedder was never called — the cache served a vector "
        "produced by a different model, in a different vector space"
    )
    assert cache.get_calls[-1][0] == "embed-v2"
    # The lookup itself must have carried the new identity, not merely have
    # missed by luck.
    assert cache.get_calls[-1] == ("embed-v2", ["shared text"])
    assert len(after_swap) == len(warmed) == 1


async def test_cache_hit_does_not_reach_the_inner_embedder() -> None:
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, RecordingCache())

    first = await embedder.embed(["a", "b"])
    second = await embedder.embed(["a", "b"])

    assert first == second
    assert inner.seen == [["a", "b"]], "second call should have been served entirely from cache"


async def test_only_the_misses_are_embedded() -> None:
    """A partial hit embeds the missing texts and nothing else."""
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, RecordingCache())

    await embedder.embed(["a"])
    result = await embedder.embed(["a", "b"])

    assert inner.seen == [["a"], ["b"]]
    assert len(result) == 2


async def test_repeated_text_in_one_batch_costs_one_embedding() -> None:
    """Positions, not texts: a repeat is two output slots and one embedding."""
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, RecordingCache())

    result = await embedder.embed(["dup", "other", "dup"])

    assert inner.seen == [["dup", "other"]]
    assert result[0] == result[2]
    assert result[0] != result[1]


async def test_cached_embedder_forwards_identity_unchanged() -> None:
    """A caller must not be able to tell from the metadata that a cache was there."""
    inner = DeterministicEmbedder(dimensions=32, model_id="inner-model")
    embedder = CachedEmbedder(inner, RecordingCache())

    assert embedder.model_id == "inner-model"
    assert embedder.dimensions == 32
    assert isinstance(embedder, TextEmbedder)


async def test_cached_embedder_returns_what_the_inner_embedder_would() -> None:
    inner = DeterministicEmbedder(dimensions=8)
    cached = CachedEmbedder(inner, RecordingCache())

    assert await cached.embed(["x", "y"]) == await inner.embed(["x", "y"])


async def test_empty_batch_never_touches_the_cache() -> None:
    cache = RecordingCache()

    assert await CachedEmbedder(CountingEmbedder(), cache).embed([]) == []
    assert cache.get_calls == []
    assert cache.put_calls == []


def test_cache_key_separates_the_model_from_the_text() -> None:
    """The null byte is what stops ``("ab", "c")`` and ``("a", "bc")`` colliding."""
    assert embedding_cache_key("ab", "c") != embedding_cache_key("a", "bc")
    assert embedding_cache_key("m", "t") == embedding_cache_key("m", "t")


class TruncatingCache(RecordingCache):
    """A ``VectorCache`` that answers with fewer entries than it was asked for.

    The failure a real cache reaches by an ordinary route: a backend that
    drops a row on a partial read, a paginated store returning one page, a
    batch size clamped somewhere below the request. Nothing about it looks
    like an error from the caller's side --- it returns a list, of vectors,
    in order.
    """

    async def get_batch(self, model: str, texts: list[str]) -> list[list[float] | None]:
        answer = await super().get_batch(model, texts)
        return answer[:-1]


class OverlongCache(RecordingCache):
    """The other direction, which fails just as quietly."""

    async def get_batch(self, model: str, texts: list[str]) -> list[list[float] | None]:
        answer = await super().get_batch(model, texts)
        return [*answer, None]


async def test_a_short_cache_answer_is_refused_rather_than_returned() -> None:
    """The cache side owes the same guarantee: one vector per input, in order.

    ``zip(..., strict=True)`` guards the *inner embedder*'s answer, and the
    cache's went unchecked --- so a cache returning three entries for four
    texts produced three vectors for four texts. Silently, and misaligned:
    every caller pairing the result back against its own input list gets the
    wrong vector for every text after the dropped one, and stores it.
    """
    embedder = CachedEmbedder(CountingEmbedder(), TruncatingCache())

    with pytest.raises(ValueError, match="3 entries for 4 texts"):
        await embedder.embed(["a", "b", "c", "d"])


async def test_an_overlong_cache_answer_is_refused_too() -> None:
    """The other direction, which is not the same check.

    A length guard written only for the short case passes the direction
    nobody thought about.
    """
    embedder = CachedEmbedder(CountingEmbedder(), OverlongCache())

    with pytest.raises(ValueError, match="5 entries for 4 texts"):
        await embedder.embed(["a", "b", "c", "d"])
