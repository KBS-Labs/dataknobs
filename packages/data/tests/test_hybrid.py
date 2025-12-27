"""Tests for hybrid search module."""

import pytest
from unittest.mock import MagicMock

from dataknobs_data.vector.hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
    _normalize_scores,
)


class TestFusionStrategy:
    """Tests for FusionStrategy enum."""

    def test_fusion_strategies_exist(self):
        """Test that all expected fusion strategies are defined."""
        assert FusionStrategy.RRF.value == "rrf"
        assert FusionStrategy.WEIGHTED_SUM.value == "weighted_sum"
        assert FusionStrategy.NATIVE.value == "native"


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridSearchConfig()
        assert config.text_weight == 0.5
        assert config.vector_weight == 0.5
        assert config.fusion_strategy == FusionStrategy.RRF
        assert config.rrf_k == 60
        assert config.text_fields is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridSearchConfig(
            text_weight=0.7,
            vector_weight=0.3,
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            rrf_k=40,
            text_fields=["title", "content"],
        )
        assert config.text_weight == 0.7
        assert config.vector_weight == 0.3
        assert config.fusion_strategy == FusionStrategy.WEIGHTED_SUM
        assert config.rrf_k == 40
        assert config.text_fields == ["title", "content"]

    def test_invalid_text_weight(self):
        """Test that invalid text_weight raises error."""
        with pytest.raises(ValueError, match="text_weight must be between"):
            HybridSearchConfig(text_weight=1.5)
        with pytest.raises(ValueError, match="text_weight must be between"):
            HybridSearchConfig(text_weight=-0.1)

    def test_invalid_vector_weight(self):
        """Test that invalid vector_weight raises error."""
        with pytest.raises(ValueError, match="vector_weight must be between"):
            HybridSearchConfig(vector_weight=2.0)

    def test_invalid_rrf_k(self):
        """Test that invalid rrf_k raises error."""
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            HybridSearchConfig(rrf_k=0)
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            HybridSearchConfig(rrf_k=-10)

    def test_normalize_weights(self):
        """Test weight normalization."""
        config = HybridSearchConfig(text_weight=0.3, vector_weight=0.7)
        text_w, vector_w = config.normalize_weights()
        assert text_w == 0.3
        assert vector_w == 0.7

    def test_normalize_weights_unequal(self):
        """Test weight normalization with unequal weights."""
        config = HybridSearchConfig(text_weight=0.25, vector_weight=0.75)
        text_w, vector_w = config.normalize_weights()
        assert text_w == 0.25
        assert vector_w == 0.75

    def test_normalize_weights_zero(self):
        """Test weight normalization when both are zero."""
        config = HybridSearchConfig(text_weight=0.0, vector_weight=0.0)
        text_w, vector_w = config.normalize_weights()
        assert text_w == 0.5
        assert vector_w == 0.5


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a hybrid search result."""
        mock_record = MagicMock()
        mock_record.id = "test-id"

        result = HybridSearchResult(
            record=mock_record,
            combined_score=0.85,
            text_score=0.9,
            vector_score=0.8,
            text_rank=1,
            vector_rank=2,
        )
        assert result.combined_score == 0.85
        assert result.text_score == 0.9
        assert result.vector_score == 0.8
        assert result.text_rank == 1
        assert result.vector_rank == 2

    def test_result_sorting(self):
        """Test that results sort by combined score descending."""
        mock_record = MagicMock()
        mock_record.id = "test-id"

        result1 = HybridSearchResult(record=mock_record, combined_score=0.5)
        result2 = HybridSearchResult(record=mock_record, combined_score=0.8)
        result3 = HybridSearchResult(record=mock_record, combined_score=0.3)

        results = sorted([result1, result2, result3])
        assert results[0].combined_score == 0.8
        assert results[1].combined_score == 0.5
        assert results[2].combined_score == 0.3

    def test_result_repr(self):
        """Test string representation of result."""
        mock_record = MagicMock()
        mock_record.id = "doc-123"

        result = HybridSearchResult(
            record=mock_record,
            combined_score=0.75,
            text_score=0.8,
            vector_score=0.7,
        )
        repr_str = repr(result)
        assert "0.75" in repr_str
        assert "doc-123" in repr_str


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_basic_fusion(self):
        """Test basic RRF with overlapping results."""
        text_results = [
            ("doc1", 0.9),
            ("doc2", 0.8),
            ("doc3", 0.7),
        ]
        vector_results = [
            ("doc2", 0.95),
            ("doc1", 0.85),
            ("doc4", 0.75),
        ]

        fused = reciprocal_rank_fusion(text_results, vector_results, k=60)

        # Convert to dict for easier checking
        scores = dict(fused)

        # doc1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        # doc2: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        # Both doc1 and doc2 appear in both lists, should have highest scores
        assert "doc1" in scores
        assert "doc2" in scores
        assert "doc3" in scores
        assert "doc4" in scores

        # doc1 and doc2 should have similar high scores (appear in both)
        assert scores["doc1"] > scores["doc3"]
        assert scores["doc2"] > scores["doc4"]

    def test_disjoint_results(self):
        """Test RRF with completely disjoint result sets."""
        text_results = [("doc1", 0.9), ("doc2", 0.8)]
        vector_results = [("doc3", 0.95), ("doc4", 0.85)]

        fused = reciprocal_rank_fusion(text_results, vector_results, k=60)
        scores = dict(fused)

        assert len(scores) == 4
        # All docs should have same score within their source rank
        assert scores["doc1"] == scores["doc3"]  # Both rank 1
        assert scores["doc2"] == scores["doc4"]  # Both rank 2

    def test_empty_text_results(self):
        """Test RRF with empty text results."""
        text_results = []
        vector_results = [("doc1", 0.9), ("doc2", 0.8)]

        fused = reciprocal_rank_fusion(text_results, vector_results, k=60)
        scores = dict(fused)

        assert len(scores) == 2
        assert scores["doc1"] > scores["doc2"]

    def test_empty_vector_results(self):
        """Test RRF with empty vector results."""
        text_results = [("doc1", 0.9), ("doc2", 0.8)]
        vector_results = []

        fused = reciprocal_rank_fusion(text_results, vector_results, k=60)
        scores = dict(fused)

        assert len(scores) == 2
        assert scores["doc1"] > scores["doc2"]

    def test_both_empty(self):
        """Test RRF with both empty result sets."""
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_weighted_rrf(self):
        """Test RRF with different weights."""
        text_results = [("doc1", 0.9)]
        vector_results = [("doc2", 0.9)]

        # With equal weights
        fused_equal = reciprocal_rank_fusion(
            text_results, vector_results, k=60, text_weight=1.0, vector_weight=1.0
        )
        scores_equal = dict(fused_equal)
        assert scores_equal["doc1"] == scores_equal["doc2"]

        # With text weight higher
        fused_text = reciprocal_rank_fusion(
            text_results, vector_results, k=60, text_weight=2.0, vector_weight=1.0
        )
        scores_text = dict(fused_text)
        assert scores_text["doc1"] > scores_text["doc2"]

    def test_k_parameter_effect(self):
        """Test that k parameter affects score distribution."""
        text_results = [("doc1", 0.9), ("doc2", 0.8)]
        vector_results = []

        # Lower k = more weight to top ranks
        fused_low_k = reciprocal_rank_fusion(text_results, vector_results, k=10)
        scores_low_k = dict(fused_low_k)

        # Higher k = more uniform distribution
        fused_high_k = reciprocal_rank_fusion(text_results, vector_results, k=100)
        scores_high_k = dict(fused_high_k)

        # Ratio between rank 1 and rank 2 should be higher with lower k
        ratio_low = scores_low_k["doc1"] / scores_low_k["doc2"]
        ratio_high = scores_high_k["doc1"] / scores_high_k["doc2"]
        assert ratio_low > ratio_high


class TestWeightedScoreFusion:
    """Tests for weighted_score_fusion function."""

    def test_basic_weighted_fusion(self):
        """Test basic weighted score fusion."""
        text_results = [("doc1", 0.8), ("doc2", 0.6)]
        vector_results = [("doc1", 0.9), ("doc3", 0.7)]

        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.5, vector_weight=0.5,
            normalize_scores=False,
        )
        scores = dict(fused)

        # doc1: 0.5 * 0.8 + 0.5 * 0.9 = 0.85
        assert abs(scores["doc1"] - 0.85) < 0.001

    def test_weighted_fusion_with_normalization(self):
        """Test weighted fusion with score normalization."""
        text_results = [("doc1", 100), ("doc2", 50)]
        vector_results = [("doc1", 0.9), ("doc2", 0.5)]

        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.5, vector_weight=0.5,
            normalize_scores=True,
        )
        scores = dict(fused)

        # After normalization, both should be in 0-1 range
        # doc1 gets max in both, so should have score of 1.0
        assert scores["doc1"] == 1.0
        # doc2 gets 0 in text (min) and 0 in vector (min after norm)
        assert scores["doc2"] == 0.0

    def test_different_weight_ratios(self):
        """Test fusion with different weight ratios."""
        text_results = [("doc1", 1.0)]
        vector_results = [("doc1", 0.0)]

        # All weight on text
        fused_text = weighted_score_fusion(
            text_results, vector_results,
            text_weight=1.0, vector_weight=0.0,
            normalize_scores=False,
        )
        assert dict(fused_text)["doc1"] == 1.0

        # All weight on vector
        fused_vector = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.0, vector_weight=1.0,
            normalize_scores=False,
        )
        assert dict(fused_vector)["doc1"] == 0.0


class TestNormalizeScores:
    """Tests for _normalize_scores helper function."""

    def test_normalize_empty(self):
        """Test normalizing empty scores."""
        result = _normalize_scores({})
        assert result == {}

    def test_normalize_single_value(self):
        """Test normalizing single value."""
        result = _normalize_scores({"doc1": 0.5})
        assert result["doc1"] == 1.0

    def test_normalize_same_values(self):
        """Test normalizing identical values."""
        result = _normalize_scores({"doc1": 0.5, "doc2": 0.5})
        assert result["doc1"] == 1.0
        assert result["doc2"] == 1.0

    def test_normalize_range(self):
        """Test normalizing values to 0-1 range."""
        result = _normalize_scores({"doc1": 0, "doc2": 50, "doc3": 100})
        assert result["doc1"] == 0.0
        assert result["doc2"] == 0.5
        assert result["doc3"] == 1.0
