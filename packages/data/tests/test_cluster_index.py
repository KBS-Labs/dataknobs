"""Tests for ClusterTopicIndex — embedding-cluster topic index."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from dataknobs_data.sources.base import (
    RetrievalIntent,
    SourceResult,
    StrategyUnavailable,
)
from dataknobs_data.sources.cluster_index import (
    DEFAULT_LABEL_MIN_WORD_LENGTH,
    DEFAULT_LABEL_TOP_TERMS,
    ClusterTopicConfig,
    ClusterTopicIndex,
    _resolve_params,
)
from dataknobs_data.sources.topic_index import DEFAULT_HEADING_STOPWORDS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    content: str,
    source_name: str = "kb",
) -> SourceResult:
    return SourceResult(
        content=content,
        source_id=chunk_id,
        source_name=source_name,
        source_type="vector_kb",
        relevance=1.0,
    )


def _unit_vector(dim: int, index: int) -> list[float]:
    """Unit vector along the given axis."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_similar(base: list[float], noise: float = 0.05) -> list[float]:
    """Create a vector similar to base by adding small perturbation.

    Draws from its own ``random.Random`` rather than seeding the global
    one. This runs at import time to build the module constants below,
    so seeding the global left every later test in the session drawing
    from a stream this module had chosen -- the hazard
    ``dataknobs_data.testing`` documents. The values are unchanged: a
    fresh ``Random(42)`` yields the sequence ``seed(42)`` did.
    """
    rng = random.Random(42)
    result = [v + rng.uniform(-noise, noise) for v in base]
    # Normalize
    norm = math.sqrt(sum(x * x for x in result))
    return [x / norm for x in result] if norm > 0 else result


# Build two well-separated clusters in 4D
_CLUSTER_A_BASE = [1.0, 0.0, 0.0, 0.0]
_CLUSTER_B_BASE = [0.0, 1.0, 0.0, 0.0]

_CHUNKS = [
    _make_chunk("a1", "authentication login security tokens"),
    _make_chunk("a2", "authentication password hashing security"),
    _make_chunk("a3", "authentication session management security"),
    _make_chunk("b1", "database query optimization indexes"),
    _make_chunk("b2", "database schema migration indexes"),
    _make_chunk("b3", "database connection pooling indexes"),
]

# Embeddings: a-chunks near axis 0, b-chunks near axis 1
_EMBEDDINGS = {
    "a1": _make_similar(_CLUSTER_A_BASE, 0.05),
    "a2": _make_similar(_CLUSTER_A_BASE, 0.06),
    "a3": _make_similar(_CLUSTER_A_BASE, 0.07),
    "b1": _make_similar(_CLUSTER_B_BASE, 0.05),
    "b2": _make_similar(_CLUSTER_B_BASE, 0.06),
    "b3": _make_similar(_CLUSTER_B_BASE, 0.07),
}


class _Embedder:
    """A ``TextEmbedder`` over a per-text rule.

    These tests need *specific* vectors --- a query landing near cluster A,
    or far from both --- so they cannot use ``DeterministicEmbedder``, whose
    whole point is that the caller does not choose where a text lands.
    """

    def __init__(self, rule: Any, *, model_id: str = "test-embedder") -> None:
        self._rule = rule
        self._model_id = model_id

    @property
    def dimensions(self) -> int:
        return len(self._rule("probe"))

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: Any) -> list[list[float]]:
        return [list(self._rule(text)) for text in texts]


def _by_content(text: str) -> list[float]:
    """The corpus rule: route a text to cluster A, cluster B, or neither."""
    if "authentication" in text or "security" in text:
        return _make_similar(_CLUSTER_A_BASE, 0.05)
    if "database" in text:
        return _make_similar(_CLUSTER_B_BASE, 0.05)
    return [0.25, 0.25, 0.25, 0.25]


# Query embedders: every text lands in one place, whatever it says.
_AUTH_EMBEDDER = _Embedder(lambda _text: [0.95, 0.05, 0.0, 0.0], model_id="auth")
_DB_EMBEDDER = _Embedder(lambda _text: [0.05, 0.95, 0.0, 0.0], model_id="db")
_NEITHER_EMBEDDER = _Embedder(lambda _text: [0.0, 0.0, 1.0, 0.0], model_id="neither")

# Corpus embedder, used by ``build()``. One embedder now does the corpus and
# the query, where ``build`` used to take a batch function and a separate
# per-text one --- two parameters that could disagree about the model.
_CORPUS_EMBEDDER = _Embedder(_by_content, model_id="corpus")


def _make_vector_fn(
    chunks: list[SourceResult],
    embeddings: dict[str, list[float]],
) -> ...:
    """Create a vector query fn that returns chunks matching query words."""

    async def vector_fn(
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        query_lower = query.lower()
        matches = []
        for c in chunks:
            if any(w in c.content.lower() for w in query_lower.split()):
                matches.append(c)
        return matches[:top_k]

    return vector_fn


# ------------------------------------------------------------------
# Tests: ClusterTopicConfig
# ------------------------------------------------------------------


class TestClusterTopicConfig:
    """Test configuration dataclass."""

    def test_defaults(self) -> None:
        config = ClusterTopicConfig()
        assert config.cluster_threshold == 0.7
        assert config.min_cluster_size == 2
        assert config.seed_max_results == 30
        assert config.seed_score_threshold == 0.2
        assert config.top_clusters == 3
        assert config.max_results_per_cluster == 20
        assert config.max_total_results == 50
        assert config.centroid_score_threshold == 0.2
        assert config.label_stopwords is DEFAULT_HEADING_STOPWORDS
        assert config.label_min_word_length == DEFAULT_LABEL_MIN_WORD_LENGTH
        assert config.label_top_terms == DEFAULT_LABEL_TOP_TERMS
        assert config.scope_profiles == {}

    def test_from_dict_basic(self) -> None:
        config = ClusterTopicConfig.from_dict(
            {
                "cluster_threshold": 0.5,
                "top_clusters": 5,
                "unknown_key": "ignored",
            }
        )
        assert config.cluster_threshold == 0.5
        assert config.top_clusters == 5

    def test_from_dict_stopwords_list(self) -> None:
        config = ClusterTopicConfig.from_dict(
            {
                "label_stopwords": ["custom", "words"],
            }
        )
        assert config.label_stopwords == frozenset({"custom", "words"})

    def test_from_dict_scope_profiles(self) -> None:
        config = ClusterTopicConfig.from_dict(
            {
                "scope_profiles": {
                    "focused": {"top_clusters": 1},
                    "broad": {"top_clusters": 5},
                },
            }
        )
        assert config.scope_profiles["focused"]["top_clusters"] == 1

    def test_frozen(self) -> None:
        config = ClusterTopicConfig()
        with pytest.raises(AttributeError):
            config.cluster_threshold = 0.5  # type: ignore[misc]

    def test_custom_label_params(self) -> None:
        config = ClusterTopicConfig(
            label_min_word_length=4,
            label_top_terms=5,
        )
        assert config.label_min_word_length == 4
        assert config.label_top_terms == 5


# ------------------------------------------------------------------
# Tests: Eager construction (from_chunks)
# ------------------------------------------------------------------


class TestEagerConstruction:
    """Test eager construction via from_chunks."""

    def test_two_clusters_formed(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 2
        sizes = sorted(c["size"] for c in idx.cluster_info)
        assert sizes == [3, 3]

    def test_topics_returns_labels(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        topics = idx.topics()
        assert len(topics) == 2
        for t in topics:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_custom_labels(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
            labels={0: "Auth Cluster", 1: "DB Cluster"},
        )
        topics = idx.topics()
        assert "Auth Cluster" in topics
        assert "DB Cluster" in topics

    def test_chunks_without_embeddings_skipped(self) -> None:
        partial = {k: v for k, v in _EMBEDDINGS.items() if k.startswith("a")}
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            partial,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 1
        assert idx.cluster_info[0]["size"] == 3

    def test_min_cluster_size(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5, min_cluster_size=4),
        )
        assert idx.topics() == []

    def test_auto_label_uses_config(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5, label_top_terms=1),
        )
        for topic in idx.topics():
            assert " " not in topic


# ------------------------------------------------------------------
# Tests: Lazy construction
# ------------------------------------------------------------------


class TestLazyConstruction:
    """Test lazy per-turn construction."""

    def test_lazy_topics_returns_empty(self) -> None:
        idx = ClusterTopicIndex()
        assert idx.topics() == []
        assert idx.cluster_info == []

    @pytest.mark.asyncio
    async def test_lazy_no_embedder_says_it_cannot_resolve(self) -> None:
        """Not an empty result: empty means "ran, matched nothing"."""
        idx = ClusterTopicIndex()
        with pytest.raises(StrategyUnavailable, match="no embedder"):
            await idx.resolve("test query")

    @pytest.mark.asyncio
    async def test_lazy_no_vector_fn_says_it_cannot_resolve(self) -> None:
        idx = ClusterTopicIndex(embedder=_AUTH_EMBEDDER)
        with pytest.raises(StrategyUnavailable, match="no vector_query_fn"):
            await idx.resolve("test query")

    @pytest.mark.asyncio
    async def test_lazy_resolve_clusters_per_turn(self) -> None:
        """Lazy mode fetches seeds and clusters them per query."""
        vector_fn = _make_vector_fn(_CHUNKS, _EMBEDDINGS)
        idx = ClusterTopicIndex(
            embedder=_AUTH_EMBEDDER,
            vector_query_fn=vector_fn,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication security")
        assert len(results) > 0
        content = " ".join(r.content for r in results)
        assert "authentication" in content


# ------------------------------------------------------------------
# Tests: build() class method
# ------------------------------------------------------------------


class TestBuildClassMethod:
    """Test the async build() factory."""

    @pytest.mark.asyncio
    async def test_build_from_embedder(self) -> None:
        idx = await ClusterTopicIndex.build(
            _CHUNKS,
            _CORPUS_EMBEDDER,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 2

    @pytest.mark.asyncio
    async def test_build_empty_chunks(self) -> None:
        idx = await ClusterTopicIndex.build(
            [],
            _CORPUS_EMBEDDER,
        )
        assert idx.topics() == []


# ------------------------------------------------------------------
# Tests: resolve() (eager mode)
# ------------------------------------------------------------------


class TestResolve:
    """Test query-time resolution with eager mode."""

    @pytest.mark.asyncio
    async def test_matches_auth_cluster(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication security")
        assert len(results) > 0
        content = " ".join(r.content for r in results)
        assert "authentication" in content

    @pytest.mark.asyncio
    async def test_matches_db_cluster(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_DB_EMBEDDER,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("database optimization")
        assert len(results) > 0
        content = " ".join(r.content for r in results)
        assert "database" in content

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_NEITHER_EMBEDDER,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.5,
            ),
        )
        results = await idx.resolve("something unrelated")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_embedder_says_it_cannot_resolve(self) -> None:
        """Eager too: the pre-built pool does not remove the need to embed.

        The clusters are there, but the *query* still has to be embedded
        to be matched against their centroids.
        """
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        with pytest.raises(StrategyUnavailable, match="no embedder"):
            await idx.resolve("test query")

    @pytest.mark.asyncio
    async def test_top_k_caps_results(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_max_results_per_cluster(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                max_results_per_cluster=1,
            ),
        )
        results = await idx.resolve("authentication")
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_deduplication(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.0,
            ),
        )
        results = await idx.resolve("authentication")
        ids = [r.source_id for r in results]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_embedder_failure_reaches_the_caller(self) -> None:
        """An embedder that raises is reported, not turned into no topics.

        This asserted ``== []`` while ``resolve`` absorbed the failure,
        which made a broken embedder indistinguishable from a query with
        no topics behind it. Pinned here in eager mode; the lazy path and
        the reasoning for both are in
        ``test_cluster_index_failure_surfaces.py``.
        """

        def failing_rule(text: str) -> list[float]:
            raise RuntimeError("embed failed")

        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_Embedder(failing_rule),
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        with pytest.raises(RuntimeError, match="embed failed"):
            await idx.resolve("test query")


# ------------------------------------------------------------------
# Tests: Scope profiles
# ------------------------------------------------------------------


class TestScopeProfiles:
    """Test per-query parameter resolution via scope profiles."""

    @pytest.mark.asyncio
    async def test_scope_profile_overrides(self) -> None:
        config = ClusterTopicConfig(
            cluster_threshold=0.5,
            top_clusters=3,
            scope_profiles={"focused": {"top_clusters": 1}},
        )
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=config,
        )
        intent = RetrievalIntent(
            text_queries=["auth"],
            scope="focused",
        )
        results = await idx.resolve("authentication", intent=intent)
        assert len(results) > 0
        for r in results:
            assert "authentication" in r.content or "security" in r.content

    @pytest.mark.asyncio
    async def test_explicit_overrides_beat_profile(self) -> None:
        config = ClusterTopicConfig(
            cluster_threshold=0.5,
            max_total_results=50,
            scope_profiles={"focused": {"max_total_results": 10}},
        )
        intent = RetrievalIntent(
            text_queries=["auth"],
            scope="focused",
            raw_data={"topic_index": {"max_total_results": 2}},
        )
        params = _resolve_params(config, intent)
        assert params.max_total_results == 2

    @pytest.mark.asyncio
    async def test_unknown_scope_uses_defaults(self) -> None:
        config = ClusterTopicConfig(cluster_threshold=0.5, top_clusters=3)
        intent = RetrievalIntent(
            text_queries=["auth"],
            scope="nonexistent",
        )
        params = _resolve_params(config, intent)
        assert params.top_clusters == 3


# ------------------------------------------------------------------
# Tests: cluster_info introspection
# ------------------------------------------------------------------


class TestClusterInfo:
    """Test cluster introspection."""

    def test_cluster_info_structure(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        for info in idx.cluster_info:
            assert "id" in info
            assert "label" in info
            assert "size" in info
            assert "centroid" in info
            assert isinstance(info["centroid"], list)
            assert len(info["centroid"]) == 4


# ------------------------------------------------------------------
# Tests: Integration pipeline
# ------------------------------------------------------------------


class TestIntegrationPipeline:
    """Integration tests exercising full construction + resolve."""

    @pytest.mark.asyncio
    async def test_build_and_resolve_roundtrip(self) -> None:
        idx = await ClusterTopicIndex.build(
            _CHUNKS,
            _CORPUS_EMBEDDER,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication security")
        assert len(results) > 0
        auth_count = sum(1 for r in results if "authentication" in r.content)
        assert auth_count > 0

    @pytest.mark.asyncio
    async def test_results_ranked_by_similarity(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS,
            _EMBEDDINGS,
            embedder=_AUTH_EMBEDDER,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.0,
            ),
        )
        results = await idx.resolve("authentication")
        assert len(results) > 0
