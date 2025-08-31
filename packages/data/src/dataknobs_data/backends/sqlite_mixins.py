"""SQLite-specific mixins for vector support and other functionality."""

from __future__ import annotations

import json
import logging

import numpy as np

from typing import TYPE_CHECKING
from ..fields import VectorField
from ..vector.types import DistanceMetric

if TYPE_CHECKING:
    from ..records import Record


logger = logging.getLogger(__name__)


class SQLiteVectorSupport:
    """Vector support for SQLite using JSON storage and Python-based similarity."""

    def __init__(self):
        """Initialize vector support tracking."""
        self._vector_dimensions = {}
        self._vector_fields = {}

    def _has_vector_fields(self, record: Record) -> bool:
        """Check if record has vector fields.
        
        Args:
            record: Record to check
            
        Returns:
            True if record has vector fields
        """
        return any(isinstance(field, VectorField)
                   for field in record.fields.values())

    def _extract_vector_dimensions(self, record: Record) -> dict[str, int]:
        """Extract dimensions from vector fields in a record.
        
        Args:
            record: Record containing potential vector fields
            
        Returns:
            Dictionary mapping field names to dimensions
        """
        dimensions = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                if field.value is not None:
                    if isinstance(field.value, np.ndarray):
                        dimensions[name] = field.value.shape[0]
                    elif isinstance(field.value, list):
                        dimensions[name] = len(field.value)
                elif field.dimensions:
                    dimensions[name] = field.dimensions
        return dimensions

    def _update_vector_dimensions(self, record: Record) -> None:
        """Update tracked vector dimensions from a record.
        
        Args:
            record: Record containing vector fields
        """
        dimensions = self._extract_vector_dimensions(record)
        self._vector_dimensions.update(dimensions)

        # Track which fields are vectors
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                self._vector_fields[name] = {
                    "dimensions": dimensions.get(name),
                    "source_field": field.source_field,
                    "model_name": field.model_name,
                    "model_version": field.model_version,
                }

    def _serialize_vector(self, vector: np.ndarray | list) -> str:
        """Serialize a vector to JSON string for storage.
        
        Args:
            vector: Vector as numpy array or list
            
        Returns:
            JSON string representation
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return json.dumps(vector)

    def _deserialize_vector(self, vector_str: str) -> np.ndarray | None:
        """Deserialize a vector from JSON string.
        
        Args:
            vector_str: JSON string representation
            
        Returns:
            Numpy array
        """
        if not vector_str:
            return None
        try:
            vector_list = json.loads(vector_str)
            return np.array(vector_list, dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _compute_similarity(
        self,
        vec1: np.ndarray | None,
        vec2: np.ndarray | None,
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> float:
        """Compute similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric to use
            
        Returns:
            Similarity score (higher is more similar)
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Ensure vectors are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)  # type: ignore[unreachable]
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)  # type: ignore[unreachable]

        # Check dimensions match
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")

        if metric == DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        elif metric == DistanceMetric.EUCLIDEAN:
            # Convert Euclidean distance to similarity (inverse)
            distance = float(np.linalg.norm(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        elif metric == DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            return float(np.dot(vec1, vec2))

        else:
            raise ValueError(f"Unsupported metric: {metric}")

