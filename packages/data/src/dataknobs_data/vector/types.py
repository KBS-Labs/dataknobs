"""Core types and data structures for vector operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..records import Record


class DistanceMetric(Enum):
    """Enumeration of supported vector distance metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    INNER_PRODUCT = "inner_product"  # Alias for dot_product
    L2 = "l2"  # Alias for euclidean
    L1 = "l1"  # Manhattan distance

    def get_aliases(self) -> list[str]:
        """Get alternative names for this metric."""
        aliases: dict[DistanceMetric, list[str]] = {
            DistanceMetric.COSINE: ["cosine_similarity", "cos"],
            DistanceMetric.EUCLIDEAN: ["l2", "euclidean_distance"],
            DistanceMetric.DOT_PRODUCT: ["inner_product", "ip"],
            DistanceMetric.L1: ["manhattan", "l1_distance"],
        }
        return aliases.get(self, [])


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search operation."""

    record: Record
    score: float
    source_text: str | None = None
    vector_field: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: VectorSearchResult) -> bool:
        """Enable sorting by score."""
        return self.score < other.score

    def __repr__(self) -> str:
        """String representation of the result."""
        return (
            f"VectorSearchResult(score={self.score:.4f}, "
            f"record_id={self.record.id}, "
            f"vector_field={self.vector_field})"
        )


@dataclass
class VectorConfig:
    """Configuration for vector operations."""

    dimensions: int
    metric: DistanceMetric = DistanceMetric.COSINE
    normalize: bool = False
    source_field: str | None = None
    model_name: str | None = None
    model_version: str | None = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.dimensions}")

        if self.dimensions > 65536:  # Common maximum for vector databases
            raise ValueError(
                f"Dimensions {self.dimensions} exceeds maximum supported (65536)"
            )


@dataclass
class VectorIndexConfig:
    """Configuration for vector index creation."""

    index_type: str = "auto"  # auto, flat, ivfflat, hnsw
    lists: int | None = None  # For IVFFlat
    m: int | None = None  # For HNSW
    ef_construction: int | None = None  # For HNSW
    ef_search: int | None = None  # For HNSW search
    probes: int | None = None  # For IVFFlat search
    quantization: str | None = None  # none, scalar, product

    def get_optimal_params(self, num_vectors: int) -> dict[str, Any]:
        """Get optimal index parameters based on dataset size."""
        params = {}

        if self.index_type == "auto":
            # Auto-select based on dataset size
            if num_vectors < 10_000:
                params["type"] = "flat"
            elif num_vectors < 1_000_000:
                params["type"] = "ivfflat"
                params["lists"] = self.lists or max(num_vectors // 1000, 100)
                params["probes"] = self.probes or 10
            else:
                params["type"] = "hnsw"
                params["m"] = self.m or 16
                params["ef_construction"] = self.ef_construction or 200
                params["ef_search"] = self.ef_search or 64
        else:
            params["type"] = self.index_type
            if self.index_type == "ivfflat":
                params["lists"] = self.lists or 100
                params["probes"] = self.probes or 10
            elif self.index_type == "hnsw":
                params["m"] = self.m or 16
                params["ef_construction"] = self.ef_construction or 200
                params["ef_search"] = self.ef_search or 64

        if self.quantization:
            params["quantization"] = self.quantization

        return params


@dataclass
class VectorMetadata:
    """Metadata associated with vector fields."""

    dimensions: int
    source_field: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    index_type: str | None = None
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimensions": self.dimensions,
            "source_field": self.source_field,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            } if self.model_name else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "index_type": self.index_type,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorMetadata:
        """Create from dictionary representation."""
        model_info = data.get("model", {})
        return cls(
            dimensions=data["dimensions"],
            source_field=data.get("source_field"),
            model_name=model_info.get("name") if model_info else None,
            model_version=model_info.get("version") if model_info else None,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            index_type=data.get("index_type"),
            metric=data.get("metric"),
        )
