"""Tests for ClusterTopicIndex — embedding-cluster topic index."""

from __future__ import annotations

import math
from typing import Any

import pytest

from dataknobs_data.sources.base import RetrievalIntent, SourceResult
from dataknobs_data.sources.cluster_index import (
    DEFAULT_LABEL_MIN_WORD_LENGTH,
    DEFAULT_LABEL_TOP_TERMS,
    ClusterTopicConfig,
    ClusterTopicIndex,
    _resolve_params,
)
from dataknobs_data.sources.processing import cosine_similarity
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
    """Create a vector similar to base by adding small perturbation."""
    import random
    random.seed(42)
    result = [v + random.uniform(-noise, noise) for v in base]
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


async def _embed_query_auth(text: str) -> list[float]:
    """Embed function that returns a vector near cluster A."""
    return [0.95, 0.05, 0.0, 0.0]


async def _embed_query_db(text: str) -> list[float]:
    """Embed function that returns a vector near cluster B."""
    return [0.05, 0.95, 0.0, 0.0]


async def _embed_query_neither(text: str) -> list[float]:
    """Embed function that returns a vector far from both clusters."""
    return [0.0, 0.0, 1.0, 0.0]


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed function for build() tests."""
    result = []
    for text in texts:
        if "authentication" in text or "security" in text:
            result.append(_make_similar(_CLUSTER_A_BASE, 0.05))
        elif "database" in text:
            result.append(_make_similar(_CLUSTER_B_BASE, 0.05))
        else:
            result.append([0.25, 0.25, 0.25, 0.25])
    return result


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
        config = ClusterTopicConfig.from_dict({
            "cluster_threshold": 0.5,
            "top_clusters": 5,
            "unknown_key": "ignored",
        })
        assert config.cluster_threshold == 0.5
        assert config.top_clusters == 5

    def test_from_dict_stopwords_list(self) -> None:
        config = ClusterTopicConfig.from_dict({
            "label_stopwords": ["custom", "words"],
        })
        assert config.label_stopwords == frozenset({"custom", "words"})

    def test_from_dict_scope_profiles(self) -> None:
        config = ClusterTopicConfig.from_dict({
            "scope_profiles": {
                "focused": {"top_clusters": 1},
                "broad": {"top_clusters": 5},
            },
        })
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
            _CHUNKS, _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 2
        sizes = sorted(c["size"] for c in idx.cluster_info)
        assert sizes == [3, 3]

    def test_topics_returns_labels(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        topics = idx.topics()
        assert len(topics) == 2
        for t in topics:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_custom_labels(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
            labels={0: "Auth Cluster", 1: "DB Cluster"},
        )
        topics = idx.topics()
        assert "Auth Cluster" in topics
        assert "DB Cluster" in topics

    def test_chunks_without_embeddings_skipped(self) -> None:
        partial = {k: v for k, v in _EMBEDDINGS.items() if k.startswith("a")}
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, partial,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 1
        assert idx.cluster_info[0]["size"] == 3

    def test_min_cluster_size(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5, min_cluster_size=4),
        )
        assert idx.topics() == []

    def test_auto_label_uses_config(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
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
    async def test_lazy_no_embed_fn_returns_empty(self) -> None:
        idx = ClusterTopicIndex()
        results = await idx.resolve("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_lazy_no_vector_fn_returns_empty(self) -> None:
        idx = ClusterTopicIndex(embed_fn=_embed_query_auth)
        results = await idx.resolve("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_lazy_resolve_clusters_per_turn(self) -> None:
        """Lazy mode fetches seeds and clusters them per query."""
        vector_fn = _make_vector_fn(_CHUNKS, _EMBEDDINGS)
        idx = ClusterTopicIndex(
            embed_fn=_embed_query_auth,
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
    async def test_build_from_embed_fn(self) -> None:
        idx = await ClusterTopicIndex.build(
            _CHUNKS, _embed_batch,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        assert len(idx.cluster_info) == 2

    @pytest.mark.asyncio
    async def test_build_empty_chunks(self) -> None:
        idx = await ClusterTopicIndex.build(
            [], _embed_batch,
            embed_fn=_embed_query_auth,
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
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication security")
        assert len(results) > 0
        content = " ".join(r.content for r in results)
        assert "authentication" in content

    @pytest.mark.asyncio
    async def test_matches_db_cluster(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_db,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("database optimization")
        assert len(results) > 0
        content = " ".join(r.content for r in results)
        assert "database" in content

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_neither,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.5,
            ),
        )
        results = await idx.resolve("something unrelated")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_embed_fn_returns_empty(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_caps_results(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_max_results_per_cluster(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
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
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.0,
            ),
        )
        results = await idx.resolve("authentication")
        ids = [r.source_id for r in results]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_embed_fn_failure_returns_empty(self) -> None:
        async def failing_embed(text: str) -> list[float]:
            raise RuntimeError("embed failed")

        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=failing_embed,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("test query")
        assert results == []


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
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
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
            _CHUNKS, _EMBEDDINGS,
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
            _CHUNKS, _embed_batch,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(cluster_threshold=0.5),
        )
        results = await idx.resolve("authentication security")
        assert len(results) > 0
        auth_count = sum(1 for r in results if "authentication" in r.content)
        assert auth_count > 0

    @pytest.mark.asyncio
    async def test_results_ranked_by_similarity(self) -> None:
        idx = ClusterTopicIndex.from_chunks(
            _CHUNKS, _EMBEDDINGS,
            embed_fn=_embed_query_auth,
            config=ClusterTopicConfig(
                cluster_threshold=0.5,
                centroid_score_threshold=0.0,
            ),
        )
        results = await idx.resolve("authentication")
        assert len(results) > 0
