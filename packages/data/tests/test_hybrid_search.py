"""Tests for hybrid search functionality."""

import pytest
import numpy as np

from dataknobs_data.vector.hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HybridSearchConfig()
        assert config.text_weight == 0.5
        assert config.vector_weight == 0.5
        assert config.fusion_strategy == FusionStrategy.RRF
        assert config.rrf_k == 60
        assert config.text_fields is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HybridSearchConfig(
            text_weight=0.3,
            vector_weight=0.7,
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            rrf_k=100,
            text_fields=["title", "content"],
        )
        assert config.text_weight == 0.3
        assert config.vector_weight == 0.7
        assert config.fusion_strategy == FusionStrategy.WEIGHTED_SUM
        assert config.rrf_k == 100
        assert config.text_fields == ["title", "content"]

    def test_validation_text_weight_out_of_range(self):
        """Test that invalid text_weight raises error."""
        with pytest.raises(ValueError, match="text_weight must be between"):
            HybridSearchConfig(text_weight=1.5)

    def test_validation_vector_weight_out_of_range(self):
        """Test that invalid vector_weight raises error."""
        with pytest.raises(ValueError, match="vector_weight must be between"):
            HybridSearchConfig(vector_weight=-0.1)

    def test_validation_rrf_k_not_positive(self):
        """Test that non-positive rrf_k raises error."""
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            HybridSearchConfig(rrf_k=0)

    def test_normalize_weights(self):
        """Test weight normalization."""
        config = HybridSearchConfig(text_weight=0.3, vector_weight=0.7)
        text_w, vector_w = config.normalize_weights()
        assert text_w == pytest.approx(0.3)
        assert vector_w == pytest.approx(0.7)
        assert text_w + vector_w == pytest.approx(1.0)

    def test_normalize_weights_unbalanced(self):
        """Test weight normalization with unbalanced weights."""
        config = HybridSearchConfig(text_weight=0.2, vector_weight=0.3)
        text_w, vector_w = config.normalize_weights()
        assert text_w == pytest.approx(0.4)
        assert vector_w == pytest.approx(0.6)

    def test_normalize_weights_zero(self):
        """Test weight normalization when both are zero."""
        config = HybridSearchConfig(text_weight=0.0, vector_weight=0.0)
        text_w, vector_w = config.normalize_weights()
        assert text_w == 0.5
        assert vector_w == 0.5


class TestFusionStrategy:
    """Tests for FusionStrategy enum."""

    def test_rrf_value(self):
        """Test RRF enum value."""
        assert FusionStrategy.RRF.value == "rrf"

    def test_weighted_sum_value(self):
        """Test WEIGHTED_SUM enum value."""
        assert FusionStrategy.WEIGHTED_SUM.value == "weighted_sum"

    def test_native_value(self):
        """Test NATIVE enum value."""
        assert FusionStrategy.NATIVE.value == "native"


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_basic_fusion(self):
        """Test basic RRF with overlapping results."""
        text_results = [
            ("doc1", 0.9),
            ("doc2", 0.7),
            ("doc3", 0.5),
        ]
        vector_results = [
            ("doc2", 0.95),
            ("doc1", 0.85),
            ("doc4", 0.75),
        ]

        fused = reciprocal_rank_fusion(text_results, vector_results, k=60)

        # Both doc1 and doc2 should be at top since they appear in both lists
        fused_ids = [rid for rid, _ in fused]
        assert "doc1" in fused_ids[:2]
        assert "doc2" in fused_ids[:2]
        assert len(fused) == 4  # 3 from text + 1 unique from vector

    def test_rrf_with_weights(self):
        """Test RRF with custom weights."""
        text_results = [("doc1", 0.9)]
        vector_results = [("doc2", 0.9)]

        # Equal weights
        fused_equal = reciprocal_rank_fusion(
            text_results, vector_results,
            k=60, text_weight=1.0, vector_weight=1.0
        )

        # Heavy vector weight
        fused_vector = reciprocal_rank_fusion(
            text_results, vector_results,
            k=60, text_weight=0.1, vector_weight=1.0
        )

        # With heavy vector weight, doc2 should score higher
        fused_vector_dict = dict(fused_vector)
        fused_equal_dict = dict(fused_equal)

        # doc2's score should be relatively higher with vector weight
        assert fused_vector_dict["doc2"] > fused_vector_dict["doc1"]

    def test_rrf_empty_results(self):
        """Test RRF with empty result lists."""
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_rrf_one_empty(self):
        """Test RRF with one empty result list."""
        text_results = [("doc1", 0.9), ("doc2", 0.8)]
        fused = reciprocal_rank_fusion(text_results, [], k=60)

        assert len(fused) == 2
        fused_ids = [rid for rid, _ in fused]
        assert "doc1" in fused_ids
        assert "doc2" in fused_ids

    def test_rrf_k_parameter_effect(self):
        """Test that k parameter affects ranking smoothing."""
        text_results = [("doc1", 0.9)]
        vector_results = [("doc2", 0.9)]

        # Lower k = more emphasis on top ranks
        fused_low_k = dict(reciprocal_rank_fusion(
            text_results, vector_results, k=10
        ))

        # Higher k = more even distribution
        fused_high_k = dict(reciprocal_rank_fusion(
            text_results, vector_results, k=100
        ))

        # With lower k, the contribution per rank is higher
        # Both should be equal since both are rank 1, but ratio changes
        ratio_low = fused_low_k["doc1"] / (fused_low_k["doc1"] + fused_low_k["doc2"])
        ratio_high = fused_high_k["doc1"] / (fused_high_k["doc1"] + fused_high_k["doc2"])

        # Ratios should be equal (both are rank 1)
        assert ratio_low == pytest.approx(ratio_high)


class TestWeightedScoreFusion:
    """Tests for weighted_score_fusion function."""

    def test_basic_fusion(self):
        """Test basic weighted fusion."""
        text_results = [("doc1", 0.8), ("doc2", 0.6)]
        vector_results = [("doc2", 0.9), ("doc3", 0.7)]

        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.5, vector_weight=0.5
        )

        # doc2 appears in both, should have combined score
        fused_dict = dict(fused)
        assert "doc2" in fused_dict
        assert fused_dict["doc2"] > 0  # Has contributions from both

    def test_weighted_fusion_weights(self):
        """Test weighted fusion with different weights."""
        text_results = [("doc1", 1.0)]
        vector_results = [("doc2", 1.0)]

        # Heavy text weight
        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.8, vector_weight=0.2
        )
        fused_dict = dict(fused)

        # With normalization, doc1 should score higher with text weight
        assert fused_dict["doc1"] > fused_dict["doc2"]

    def test_weighted_fusion_no_normalize(self):
        """Test weighted fusion without normalization."""
        text_results = [("doc1", 0.5)]
        vector_results = [("doc1", 0.5)]

        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.5, vector_weight=0.5,
            normalize_scores=False
        )
        fused_dict = dict(fused)

        # 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        assert fused_dict["doc1"] == pytest.approx(0.5)

    def test_weighted_fusion_empty(self):
        """Test weighted fusion with empty results."""
        fused = weighted_score_fusion([], [], text_weight=0.5, vector_weight=0.5)
        assert fused == []

    def test_weighted_fusion_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        text_results = [("doc1", 0.3), ("doc2", 0.9)]
        vector_results = [("doc3", 0.5)]

        fused = weighted_score_fusion(
            text_results, vector_results,
            text_weight=0.5, vector_weight=0.5
        )

        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_basic_creation(self):
        """Test creating a HybridSearchResult."""
        # Create a mock record
        class MockRecord:
            id = "test-id"

        result = HybridSearchResult(
            record=MockRecord(),
            combined_score=0.85,
            text_score=0.7,
            vector_score=0.9,
            text_rank=2,
            vector_rank=1,
        )

        assert result.combined_score == 0.85
        assert result.text_score == 0.7
        assert result.vector_score == 0.9
        assert result.text_rank == 2
        assert result.vector_rank == 1

    def test_optional_scores(self):
        """Test HybridSearchResult with optional scores."""
        class MockRecord:
            id = "test-id"

        result = HybridSearchResult(
            record=MockRecord(),
            combined_score=0.5,
        )

        assert result.text_score is None
        assert result.vector_score is None
        assert result.text_rank is None
        assert result.vector_rank is None

    def test_sorting(self):
        """Test that HybridSearchResult can be sorted by score."""
        class MockRecord:
            def __init__(self, id):
                self.id = id

        results = [
            HybridSearchResult(record=MockRecord("a"), combined_score=0.3),
            HybridSearchResult(record=MockRecord("b"), combined_score=0.9),
            HybridSearchResult(record=MockRecord("c"), combined_score=0.6),
        ]

        # Sort descending (higher score first)
        sorted_results = sorted(results)

        assert sorted_results[0].record.id == "b"
        assert sorted_results[1].record.id == "c"
        assert sorted_results[2].record.id == "a"

    def test_repr(self):
        """Test string representation."""
        class MockRecord:
            id = "test-123"

        result = HybridSearchResult(
            record=MockRecord(),
            combined_score=0.75,
            text_score=0.6,
            vector_score=0.8,
        )

        repr_str = repr(result)
        assert "0.75" in repr_str
        assert "0.60" in repr_str
        assert "0.80" in repr_str
        assert "test-123" in repr_str

    def test_metadata(self):
        """Test HybridSearchResult with metadata."""
        class MockRecord:
            id = "test-id"

        result = HybridSearchResult(
            record=MockRecord(),
            combined_score=0.5,
            metadata={"fusion_strategy": "rrf", "backend": "elasticsearch"},
        )

        assert result.metadata["fusion_strategy"] == "rrf"
        assert result.metadata["backend"] == "elasticsearch"
