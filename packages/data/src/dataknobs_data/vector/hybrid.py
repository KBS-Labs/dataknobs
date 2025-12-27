"""Hybrid search types and fusion algorithms.

This module provides types and utilities for combining keyword (text) search
with vector (semantic) search for improved retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..records import Record


class FusionStrategy(Enum):
    """Strategy for combining text and vector search results."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Weighted score combination
    NATIVE = "native"  # Use backend's native hybrid implementation


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search operations.

    Attributes:
        text_weight: Weight for text search scores (0.0 to 1.0)
        vector_weight: Weight for vector search scores (0.0 to 1.0)
        fusion_strategy: Strategy for combining results
        rrf_k: Constant k for RRF formula (default 60)
        text_fields: Fields to search for text matching (None = all text fields)
    """

    text_weight: float = 0.5
    vector_weight: float = 0.5
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60
    text_fields: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.text_weight <= 1.0:
            raise ValueError(f"text_weight must be between 0 and 1, got {self.text_weight}")
        if not 0.0 <= self.vector_weight <= 1.0:
            raise ValueError(f"vector_weight must be between 0 and 1, got {self.vector_weight}")
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {self.rrf_k}")

    def normalize_weights(self) -> tuple[float, float]:
        """Get normalized weights that sum to 1.0.

        Returns:
            Tuple of (normalized_text_weight, normalized_vector_weight)
        """
        total = self.text_weight + self.vector_weight
        if total == 0:
            return 0.5, 0.5
        return self.text_weight / total, self.vector_weight / total


@dataclass
class HybridSearchResult:
    """Result from a hybrid search operation.

    Attributes:
        record: The matched record
        combined_score: Final combined score after fusion
        text_score: Score from text search (None if not matched by text)
        vector_score: Score from vector search (None if not matched by vector)
        text_rank: Rank in text search results (None if not matched)
        vector_rank: Rank in vector search results (None if not matched)
        metadata: Additional result metadata
    """

    record: Record
    combined_score: float
    text_score: float | None = None
    vector_score: float | None = None
    text_rank: int | None = None
    vector_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: HybridSearchResult) -> bool:
        """Enable sorting by combined score (descending)."""
        return self.combined_score > other.combined_score

    def __repr__(self) -> str:
        """String representation of the result."""
        text_str = f"{self.text_score:.4f}" if self.text_score is not None else "N/A"
        vector_str = f"{self.vector_score:.4f}" if self.vector_score is not None else "N/A"
        return (
            f"HybridSearchResult(score={self.combined_score:.4f}, "
            f"text={text_str}, "
            f"vector={vector_str}, "
            f"record_id={self.record.id})"
        )


def reciprocal_rank_fusion(
    text_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    k: int = 60,
    text_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Combine ranked results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank)) for each ranking where the item appears.

    This is a robust fusion method that doesn't require score normalization
    and handles different score distributions well.

    Args:
        text_results: List of (record_id, score) from text search, ordered by score desc
        vector_results: List of (record_id, score) from vector search, ordered by score desc
        k: Ranking constant (default 60). Higher k reduces the impact of high ranks.
        text_weight: Weight multiplier for text search ranks
        vector_weight: Weight multiplier for vector search ranks

    Returns:
        List of (record_id, rrf_score) ordered by RRF score descending
    """
    scores: dict[str, float] = {}

    # Add text search contributions
    for rank, (record_id, _score) in enumerate(text_results, start=1):
        rrf_contribution = text_weight / (k + rank)
        scores[record_id] = scores.get(record_id, 0.0) + rrf_contribution

    # Add vector search contributions
    for rank, (record_id, _score) in enumerate(vector_results, start=1):
        rrf_contribution = vector_weight / (k + rank)
        scores[record_id] = scores.get(record_id, 0.0) + rrf_contribution

    # Sort by RRF score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def weighted_score_fusion(
    text_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    text_weight: float = 0.5,
    vector_weight: float = 0.5,
    normalize_scores: bool = True,
) -> list[tuple[str, float]]:
    """Combine results using weighted score sum.

    Combined score = text_weight * text_score + vector_weight * vector_score

    Args:
        text_results: List of (record_id, score) from text search
        vector_results: List of (record_id, score) from vector search
        text_weight: Weight for text scores (0.0 to 1.0)
        vector_weight: Weight for vector scores (0.0 to 1.0)
        normalize_scores: Whether to normalize scores to 0-1 range before combining

    Returns:
        List of (record_id, combined_score) ordered by score descending
    """
    # Build score maps
    text_scores = dict(text_results)
    vector_scores = dict(vector_results)

    # Optionally normalize scores
    if normalize_scores:
        text_scores = _normalize_scores(text_scores)
        vector_scores = _normalize_scores(vector_scores)

    # Get all unique record IDs
    all_ids = set(text_scores.keys()) | set(vector_scores.keys())

    # Compute combined scores
    combined: dict[str, float] = {}
    for record_id in all_ids:
        text_score = text_scores.get(record_id, 0.0)
        vector_score = vector_scores.get(record_id, 0.0)
        combined[record_id] = text_weight * text_score + vector_weight * vector_score

    # Sort by combined score descending
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to 0-1 range using min-max normalization.

    Args:
        scores: Dictionary mapping record IDs to scores

    Returns:
        Dictionary with normalized scores
    """
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        # All scores are the same
        return dict.fromkeys(scores, 1.0)

    return {
        k: (v - min_score) / (max_score - min_score)
        for k, v in scores.items()
    }
