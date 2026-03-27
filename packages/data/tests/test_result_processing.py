"""Tests for the result processing pipeline.

Tests Level 1 processors (CrossSourceNormalizer, RelativeRelevanceFilter,
QueryRelevanceRanker), StrategyChain, ResultPipeline, and build_pipeline.
"""

from __future__ import annotations

import pytest

from dataknobs_data.sources.base import RetrievalIntent, SourceResult
from dataknobs_data.sources.processing import (
    CrossSourceNormalizer,
    EmbeddingClusterer,
    QueryClusterScorer,
    QueryRelevanceRanker,
    RelativeRelevanceFilter,
    ResultPipeline,
    StrategyChain,
    StrategyUnavailable,
    TermOverlapClusterer,
    TfidfClusterer,
    build_pipeline,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _result(
    content: str = "text",
    source_name: str = "src_a",
    source_type: str = "vector_kb",
    relevance: float = 0.5,
    source_id: str = "",
) -> SourceResult:
    return SourceResult(
        content=content,
        source_id=source_id or f"{source_name}_{relevance}",
        source_name=source_name,
        source_type=source_type,
        relevance=relevance,
    )


def _intent(queries: list[str] | None = None) -> RetrievalIntent:
    return RetrievalIntent(text_queries=queries or ["test query"])


# ------------------------------------------------------------------
# CrossSourceNormalizer
# ------------------------------------------------------------------


class TestCrossSourceNormalizer:
    """Tests for CrossSourceNormalizer."""

    @pytest.mark.asyncio
    async def test_min_max_scales_per_source(self) -> None:
        results = [
            _result(source_name="kb", relevance=0.9),
            _result(source_name="kb", relevance=0.3),
            _result(source_name="db", relevance=0.6),
            _result(source_name="db", relevance=0.2),
        ]
        norm = CrossSourceNormalizer(strategy="min_max")
        out = await norm.process(results, _intent(), "test")

        # KB: 0.9 -> 1.0, 0.3 -> 0.0
        assert out[0].relevance == pytest.approx(1.0)
        assert out[1].relevance == pytest.approx(0.0)
        # DB: 0.6 -> 1.0, 0.2 -> 0.0
        assert out[2].relevance == pytest.approx(1.0)
        assert out[3].relevance == pytest.approx(0.0)
        # Raw relevance preserved in metadata
        assert out[0].metadata["_raw_relevance"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_min_max_same_scores_become_one(self) -> None:
        """When all scores in a source are equal, normalize to 1.0."""
        results = [
            _result(source_name="kb", relevance=0.5, source_id="a"),
            _result(source_name="kb", relevance=0.5, source_id="b"),
            _result(source_name="db", relevance=0.8, source_id="c"),
        ]
        norm = CrossSourceNormalizer(strategy="min_max")
        out = await norm.process(results, _intent(), "test")
        # KB: all same -> all 1.0
        assert out[0].relevance == pytest.approx(1.0)
        assert out[1].relevance == pytest.approx(1.0)
        # DB: single value -> 1.0
        assert out[2].relevance == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_rank_normalization(self) -> None:
        results = [
            _result(source_name="kb", relevance=0.9),
            _result(source_name="kb", relevance=0.5),
            _result(source_name="db", relevance=0.8),
        ]
        norm = CrossSourceNormalizer(strategy="rank")
        out = await norm.process(results, _intent(), "test")
        # KB: 0.9 -> rank 1 (1.0), 0.5 -> rank 2 (0.5)
        assert out[0].relevance == pytest.approx(1.0)
        assert out[1].relevance == pytest.approx(0.5)
        # DB: only one result -> rank 1 (1.0)
        assert out[2].relevance == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_z_score_normalization(self) -> None:
        results = [
            _result(source_name="kb", relevance=0.9),
            _result(source_name="kb", relevance=0.1),
            _result(source_name="db", relevance=0.5),
        ]
        norm = CrossSourceNormalizer(strategy="z_score")
        out = await norm.process(results, _intent(), "test")
        # High score should be above 0.5, low below
        assert out[0].relevance > 0.5
        assert out[1].relevance < 0.5

    @pytest.mark.asyncio
    async def test_single_source_passthrough(self) -> None:
        results = [
            _result(source_name="kb", relevance=0.9),
            _result(source_name="kb", relevance=0.3),
        ]
        norm = CrossSourceNormalizer(strategy="min_max")
        out = await norm.process(results, _intent(), "test")
        # Single source -> no normalization
        assert out[0].relevance == pytest.approx(0.9)
        assert out[1].relevance == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        norm = CrossSourceNormalizer(strategy="min_max")
        out = await norm.process([], _intent(), "test")
        assert out == []

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self) -> None:
        norm = CrossSourceNormalizer(strategy="nonexistent")
        with pytest.raises(StrategyUnavailable, match="nonexistent"):
            await norm.process(
                [_result(source_name="a"), _result(source_name="b")],
                _intent(),
                "test",
            )


# ------------------------------------------------------------------
# RelativeRelevanceFilter
# ------------------------------------------------------------------


class TestRelativeRelevanceFilter:
    """Tests for RelativeRelevanceFilter."""

    @pytest.mark.asyncio
    async def test_drops_weak_results(self) -> None:
        results = [
            _result(relevance=0.9, source_id="a"),
            _result(relevance=0.8, source_id="b"),
            _result(relevance=0.3, source_id="c"),
            _result(relevance=0.1, source_id="d"),
        ]
        filt = RelativeRelevanceFilter(threshold=0.5, min_results=1)
        out = await filt.process(results, _intent(), "test")
        # Cutoff: 0.5 * 0.9 = 0.45. 0.3 and 0.1 below.
        assert len(out) == 2
        assert {r.source_id for r in out} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_respects_min_results_floor(self) -> None:
        results = [
            _result(relevance=0.9, source_id="a"),
            _result(relevance=0.1, source_id="b"),
            _result(relevance=0.05, source_id="c"),
        ]
        filt = RelativeRelevanceFilter(threshold=0.9, min_results=3)
        out = await filt.process(results, _intent(), "test")
        # Only 0.9 passes threshold, but min_results=3 keeps all
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        filt = RelativeRelevanceFilter()
        out = await filt.process([], _intent(), "test")
        assert out == []

    @pytest.mark.asyncio
    async def test_all_same_score(self) -> None:
        results = [
            _result(relevance=0.5, source_id="a"),
            _result(relevance=0.5, source_id="b"),
            _result(relevance=0.5, source_id="c"),
        ]
        filt = RelativeRelevanceFilter(threshold=0.5, min_results=1)
        out = await filt.process(results, _intent(), "test")
        # All equal -> all pass (0.5 * 0.5 = 0.25 cutoff)
        assert len(out) == 3


# ------------------------------------------------------------------
# QueryRelevanceRanker
# ------------------------------------------------------------------


class TestQueryRelevanceRanker:
    """Tests for QueryRelevanceRanker."""

    @pytest.mark.asyncio
    async def test_boosts_relevant_content(self) -> None:
        results = [
            _result(
                content="OAuth token refresh mechanism details",
                relevance=0.6,
                source_id="a",
            ),
            _result(
                content="CSRF protection and security risks in OAuth",
                relevance=0.5,
                source_id="b",
            ),
        ]
        ranker = QueryRelevanceRanker(boost_weight=0.5)
        out = await ranker.process(
            results, _intent(), "What security risks should I be aware of?"
        )
        # "security risks" overlaps more with result b
        assert out[0].source_id == "b"

    @pytest.mark.asyncio
    async def test_preserves_order_with_zero_weight(self) -> None:
        results = [
            _result(content="first", relevance=0.9, source_id="a"),
            _result(content="second", relevance=0.3, source_id="b"),
        ]
        ranker = QueryRelevanceRanker(boost_weight=0.0)
        out = await ranker.process(results, _intent(), "anything")
        assert out[0].source_id == "a"
        assert out[1].source_id == "b"

    @pytest.mark.asyncio
    async def test_empty_message_passthrough(self) -> None:
        results = [_result(content="text", relevance=0.5)]
        ranker = QueryRelevanceRanker(boost_weight=0.5)
        out = await ranker.process(results, _intent(), "")
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_query_overlap_in_metadata(self) -> None:
        results = [
            _result(content="security risks in authentication", relevance=0.5),
        ]
        ranker = QueryRelevanceRanker(boost_weight=0.3)
        out = await ranker.process(
            results, _intent(), "security risks"
        )
        assert "_query_overlap" in out[0].metadata
        assert out[0].metadata["_query_overlap"] > 0


# ------------------------------------------------------------------
# StrategyChain
# ------------------------------------------------------------------


class _AlwaysUnavailable:
    """Test processor that always raises StrategyUnavailable."""

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        raise StrategyUnavailable("always unavailable")


class _AlwaysErrors:
    """Test processor that raises a real error (not StrategyUnavailable)."""

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        msg = "real error"
        raise RuntimeError(msg)


class _AddMetadata:
    """Test processor that adds a marker to metadata."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    async def process(
        self,
        results: list[SourceResult],
        intent: RetrievalIntent,
        user_message: str,
    ) -> list[SourceResult]:
        return [
            SourceResult(
                content=r.content,
                source_id=r.source_id,
                source_name=r.source_name,
                source_type=r.source_type,
                relevance=r.relevance,
                metadata={**r.metadata, "_processed_by": self.marker},
            )
            for r in results
        ]


class TestStrategyChain:
    """Tests for StrategyChain."""

    @pytest.mark.asyncio
    async def test_first_viable_wins(self) -> None:
        chain = StrategyChain(
            strategies=[_AddMetadata("first"), _AddMetadata("second")],
            name="test",
        )
        results = [_result()]
        out = await chain.process(results, _intent(), "test")
        assert out[0].metadata["_processed_by"] == "first"

    @pytest.mark.asyncio
    async def test_fallback_on_unavailable(self) -> None:
        chain = StrategyChain(
            strategies=[_AlwaysUnavailable(), _AddMetadata("fallback")],
            name="test",
        )
        results = [_result()]
        out = await chain.process(results, _intent(), "test")
        assert out[0].metadata["_processed_by"] == "fallback"

    @pytest.mark.asyncio
    async def test_exhausted_passthrough(self) -> None:
        chain = StrategyChain(
            strategies=[_AlwaysUnavailable(), _AlwaysUnavailable()],
            name="test",
        )
        results = [_result(relevance=0.7)]
        out = await chain.process(results, _intent(), "test")
        # Original results returned unmodified
        assert out[0].relevance == pytest.approx(0.7)
        assert "_processed_by" not in out[0].metadata

    @pytest.mark.asyncio
    async def test_real_error_propagates(self) -> None:
        chain = StrategyChain(
            strategies=[_AlwaysErrors(), _AddMetadata("backup")],
            name="test",
        )
        with pytest.raises(RuntimeError, match="real error"):
            await chain.process([_result()], _intent(), "test")

    @pytest.mark.asyncio
    async def test_empty_strategies(self) -> None:
        chain = StrategyChain(strategies=[], name="empty")
        results = [_result()]
        out = await chain.process(results, _intent(), "test")
        assert len(out) == 1


# ------------------------------------------------------------------
# ResultPipeline
# ------------------------------------------------------------------


class TestResultPipeline:
    """Tests for ResultPipeline composition."""

    @pytest.mark.asyncio
    async def test_pipeline_runs_stages_in_order(self) -> None:
        pipeline = ResultPipeline(stages=[
            CrossSourceNormalizer(strategy="min_max"),
            RelativeRelevanceFilter(threshold=0.3, min_results=1),
        ])
        results = [
            _result(source_name="a", relevance=0.9, source_id="x"),
            _result(source_name="a", relevance=0.1, source_id="y"),
            _result(source_name="b", relevance=0.8, source_id="z"),
        ]
        out = await pipeline.process(results, _intent(), "test")
        # After normalization: a has [1.0, 0.0], b has [1.0]
        # After filter (0.3 * 1.0 = 0.3): a[0.0] dropped
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_pipeline_with_strategy_chain(self) -> None:
        pipeline = ResultPipeline(stages=[
            StrategyChain(
                strategies=[_AlwaysUnavailable(), _AddMetadata("chain_ok")],
                name="test",
            ),
        ])
        out = await pipeline.process([_result()], _intent(), "test")
        assert out[0].metadata["_processed_by"] == "chain_ok"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self) -> None:
        pipeline = ResultPipeline()
        results = [_result()]
        out = await pipeline.process(results, _intent(), "test")
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        pipeline = ResultPipeline(stages=[
            CrossSourceNormalizer(strategy="min_max"),
            RelativeRelevanceFilter(threshold=0.5),
        ])
        out = await pipeline.process([], _intent(), "test")
        assert out == []


# ------------------------------------------------------------------
# build_pipeline factory
# ------------------------------------------------------------------


class TestBuildPipeline:
    """Tests for the build_pipeline factory function."""

    def test_empty_config_returns_none(self) -> None:
        assert build_pipeline({}) is None
        assert build_pipeline(None) is None

    def test_normalize_shorthand(self) -> None:
        pipeline = build_pipeline({"normalize_strategy": "min_max"})
        assert pipeline is not None
        assert len(pipeline.stages) == 1

    def test_normalize_chain(self) -> None:
        pipeline = build_pipeline({
            "normalize_strategy": [
                {"method": "z_score"},
                {"method": "min_max"},
            ],
        })
        assert pipeline is not None
        assert len(pipeline.stages) == 1
        stage = pipeline.stages[0]
        assert isinstance(stage, StrategyChain)
        assert len(stage.strategies) == 2

    def test_all_level1_stages(self) -> None:
        pipeline = build_pipeline({
            "normalize_strategy": "rank",
            "relative_threshold": 0.4,
            "min_results": 2,
            "query_rerank_weight": 0.3,
        })
        assert pipeline is not None
        assert len(pipeline.stages) == 3  # normalize + filter + ranker

    def test_filter_only(self) -> None:
        pipeline = build_pipeline({"relative_threshold": 0.5})
        assert pipeline is not None
        assert len(pipeline.stages) == 1
        stage = pipeline.stages[0]
        assert isinstance(stage, RelativeRelevanceFilter)
        assert stage.threshold == pytest.approx(0.5)

    def test_no_active_stages_returns_none(self) -> None:
        # Config with only irrelevant keys
        assert build_pipeline({"cluster_min_size": 2}) is None

    @pytest.mark.asyncio
    async def test_built_pipeline_runs(self) -> None:
        pipeline = build_pipeline({
            "normalize_strategy": "min_max",
            "relative_threshold": 0.3,
            "min_results": 1,
        })
        assert pipeline is not None
        results = [
            _result(source_name="a", relevance=0.9, source_id="x"),
            _result(source_name="a", relevance=0.1, source_id="y"),
            _result(source_name="b", relevance=0.8, source_id="z"),
        ]
        out = await pipeline.process(results, _intent(), "test")
        # Normalization + filtering should reduce count
        assert len(out) < len(results) or all(r.relevance > 0 for r in out)


# ------------------------------------------------------------------
# TermOverlapClusterer
# ------------------------------------------------------------------


class TestTermOverlapClusterer:
    """Tests for TermOverlapClusterer."""

    @pytest.mark.asyncio
    async def test_groups_similar_content(self) -> None:
        results = [
            _result(content="OAuth token security risks and vulnerabilities", source_id="a"),
            _result(content="Security risks in token handling and storage", source_id="b"),
            _result(content="Database migration patterns and best practices", source_id="c"),
            _result(content="Migration strategies for database schemas", source_id="d"),
        ]
        clusterer = TermOverlapClusterer(similarity_threshold=0.15, min_cluster_size=2)
        out = await clusterer.process(results, _intent(), "test")

        # Security results should cluster together, migration results together
        assert out[0].metadata["cluster_id"] >= 0
        assert all("cluster_id" in r.metadata for r in out)
        # At least 2 distinct clusters (or 1 cluster + unclustered)
        cluster_ids = {r.metadata["cluster_id"] for r in out}
        assert len(cluster_ids) >= 2

    @pytest.mark.asyncio
    async def test_metadata_annotated(self) -> None:
        results = [
            _result(content="authentication tokens and security", source_id="a"),
            _result(content="security authentication methods", source_id="b"),
        ]
        clusterer = TermOverlapClusterer(similarity_threshold=0.1, min_cluster_size=2)
        out = await clusterer.process(results, _intent(), "test")

        for r in out:
            assert "cluster_id" in r.metadata
            assert "cluster_label" in r.metadata
            assert "cluster_size" in r.metadata

    @pytest.mark.asyncio
    async def test_too_few_results_passthrough(self) -> None:
        results = [_result(content="only one")]
        clusterer = TermOverlapClusterer(min_cluster_size=2)
        out = await clusterer.process(results, _intent(), "test")
        assert len(out) == 1
        assert "cluster_id" not in out[0].metadata

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        results = [
            _result(content="OAuth security risks", source_id="a"),
            _result(content="Security risk assessment", source_id="b"),
            _result(content="Database performance tuning", source_id="c"),
        ]
        clusterer = TermOverlapClusterer(similarity_threshold=0.15, min_cluster_size=2)
        out1 = await clusterer.process(results, _intent(), "test")
        out2 = await clusterer.process(results, _intent(), "test")
        assert [r.metadata.get("cluster_id") for r in out1] == [
            r.metadata.get("cluster_id") for r in out2
        ]


# ------------------------------------------------------------------
# TfidfClusterer
# ------------------------------------------------------------------


class TestTfidfClusterer:
    """Tests for TfidfClusterer."""

    @pytest.mark.asyncio
    async def test_groups_similar_content(self) -> None:
        results = [
            _result(content="OAuth token security risks and vulnerabilities", source_id="a"),
            _result(content="Security risks in token handling and storage", source_id="b"),
            _result(content="Database migration patterns and best practices", source_id="c"),
            _result(content="Migration strategies for database schemas", source_id="d"),
        ]
        clusterer = TfidfClusterer(similarity_threshold=0.1, min_cluster_size=2)
        out = await clusterer.process(results, _intent(), "test")

        assert all("cluster_id" in r.metadata for r in out)
        cluster_ids = {r.metadata["cluster_id"] for r in out if r.metadata["cluster_id"] >= 0}
        assert len(cluster_ids) >= 1

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        results = [
            _result(content="OAuth security token", source_id="a"),
            _result(content="Security token management", source_id="b"),
        ]
        clusterer = TfidfClusterer(similarity_threshold=0.1, min_cluster_size=2)
        out1 = await clusterer.process(results, _intent(), "test")
        out2 = await clusterer.process(results, _intent(), "test")
        assert [r.metadata.get("cluster_id") for r in out1] == [
            r.metadata.get("cluster_id") for r in out2
        ]


# ------------------------------------------------------------------
# EmbeddingClusterer
# ------------------------------------------------------------------


class TestEmbeddingClusterer:
    """Tests for EmbeddingClusterer."""

    @pytest.mark.asyncio
    async def test_no_embed_fn_raises(self) -> None:
        clusterer = EmbeddingClusterer(embed_fn=None)
        with pytest.raises(StrategyUnavailable, match="No embed_fn"):
            await clusterer.process([_result(), _result()], _intent(), "test")

    @pytest.mark.asyncio
    async def test_clusters_with_embed_fn(self) -> None:
        async def mock_embed(texts: list[str]) -> list[list[float]]:
            """Produce embeddings that cluster security vs database."""
            embeddings = []
            for t in texts:
                if "security" in t.lower():
                    embeddings.append([1.0, 0.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0, 0.0])
            return embeddings

        results = [
            _result(content="OAuth security risks", source_id="a"),
            _result(content="Security token handling", source_id="b"),
            _result(content="Database migration", source_id="c"),
        ]
        clusterer = EmbeddingClusterer(
            similarity_threshold=0.5, min_cluster_size=2, embed_fn=mock_embed,
        )
        out = await clusterer.process(results, _intent(), "test")

        # Security results should cluster, database separate
        sec_ids = {out[0].metadata["cluster_id"], out[1].metadata["cluster_id"]}
        assert len(sec_ids) == 1  # Same cluster
        assert sec_ids != {-1}


# ------------------------------------------------------------------
# QueryClusterScorer
# ------------------------------------------------------------------


class TestQueryClusterScorer:
    """Tests for QueryClusterScorer."""

    @pytest.mark.asyncio
    async def test_scores_with_terms(self) -> None:
        """Term-based scoring when no embed_fn."""
        results = [
            SourceResult(content="security risks in OAuth", source_id="a",
                         source_name="kb", source_type="vector_kb", relevance=0.8,
                         metadata={"cluster_id": 0, "cluster_label": "security", "cluster_size": 2}),
            SourceResult(content="token security vulnerabilities", source_id="b",
                         source_name="kb", source_type="vector_kb", relevance=0.7,
                         metadata={"cluster_id": 0, "cluster_label": "security", "cluster_size": 2}),
            SourceResult(content="database migration patterns", source_id="c",
                         source_name="kb", source_type="vector_kb", relevance=0.5,
                         metadata={"cluster_id": 1, "cluster_label": "database", "cluster_size": 1}),
        ]
        scorer = QueryClusterScorer(embed_fn=None)
        out = await scorer.process(
            results, _intent(), "What security risks should I worry about?",
        )

        # Security cluster should score higher than database
        sec_score = out[0].metadata["cluster_query_score"]
        db_result = [r for r in out if r.source_id == "c"][0]
        db_score = db_result.metadata["cluster_query_score"]
        assert sec_score > db_score

    @pytest.mark.asyncio
    async def test_reorders_by_cluster_score(self) -> None:
        """Higher-scoring clusters appear first."""
        results = [
            SourceResult(content="database migration", source_id="a",
                         source_name="kb", source_type="vector_kb", relevance=0.9,
                         metadata={"cluster_id": 0, "cluster_label": "db", "cluster_size": 1}),
            SourceResult(content="security risks vulnerabilities", source_id="b",
                         source_name="kb", source_type="vector_kb", relevance=0.5,
                         metadata={"cluster_id": 1, "cluster_label": "sec", "cluster_size": 1}),
        ]
        scorer = QueryClusterScorer(embed_fn=None)
        out = await scorer.process(
            results, _intent(), "security risks",
        )
        # Security result should come first despite lower relevance
        assert out[0].source_id == "b"

    @pytest.mark.asyncio
    async def test_scores_with_embeddings(self) -> None:
        async def mock_embed(texts: list[str]) -> list[list[float]]:
            embeddings = []
            for t in texts:
                if "security" in t.lower():
                    embeddings.append([1.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0])
            return embeddings

        results = [
            SourceResult(content="security risks", source_id="a",
                         source_name="kb", source_type="vector_kb", relevance=0.8,
                         metadata={"cluster_id": 0, "cluster_label": "sec", "cluster_size": 1}),
            SourceResult(content="database migration", source_id="b",
                         source_name="kb", source_type="vector_kb", relevance=0.5,
                         metadata={"cluster_id": 1, "cluster_label": "db", "cluster_size": 1}),
        ]
        scorer = QueryClusterScorer(embed_fn=mock_embed)
        out = await scorer.process(
            results, _intent(), "security concerns",
        )
        sec = [r for r in out if r.source_id == "a"][0]
        db = [r for r in out if r.source_id == "b"][0]
        assert sec.metadata["cluster_query_score"] > db.metadata["cluster_query_score"]


# ------------------------------------------------------------------
# build_pipeline with clustering
# ------------------------------------------------------------------


class TestBuildPipelineWithClustering:
    """Tests for build_pipeline with cluster_strategy config."""

    def test_cluster_shorthand(self) -> None:
        pipeline = build_pipeline({"cluster_strategy": "term_overlap"})
        assert pipeline is not None
        # cluster + query scorer = 2 stages
        assert len(pipeline.stages) == 2

    def test_cluster_chain(self) -> None:
        pipeline = build_pipeline({
            "cluster_strategy": [
                {"method": "embedding"},
                {"method": "tfidf"},
            ],
        })
        assert pipeline is not None
        # StrategyChain(cluster) + QueryClusterScorer = 2 stages
        assert len(pipeline.stages) == 2
        assert isinstance(pipeline.stages[0], StrategyChain)

    def test_full_pipeline(self) -> None:
        pipeline = build_pipeline({
            "normalize_strategy": "min_max",
            "relative_threshold": 0.3,
            "query_rerank_weight": 0.2,
            "cluster_strategy": "tfidf",
        })
        assert pipeline is not None
        # normalize + filter + ranker + cluster + scorer = 5
        assert len(pipeline.stages) == 5

    @pytest.mark.asyncio
    async def test_cluster_pipeline_runs(self) -> None:
        pipeline = build_pipeline({
            "cluster_strategy": "term_overlap",
            "cluster_threshold": 0.15,
            "cluster_min_size": 2,
        })
        assert pipeline is not None
        results = [
            _result(content="security risks in authentication", source_id="a"),
            _result(content="authentication security measures", source_id="b"),
            _result(content="database performance optimization", source_id="c"),
        ]
        out = await pipeline.process(
            results, _intent(), "security risks",
        )
        # All results should have cluster metadata
        assert all("cluster_id" in r.metadata for r in out)
        # All results should have query scores
        assert all("cluster_query_score" in r.metadata for r in out)
