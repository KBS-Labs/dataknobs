"""Common base implementation for vector stores."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dataknobs_config import ConfigurableBase

from ..types import DistanceMetric

if TYPE_CHECKING:
    import numpy as np


class VectorStoreBase(ConfigurableBase):
    """Base implementation with common functionality for all vector stores.
    
    This class provides:
    - Configuration parsing following the database pattern
    - Common parameter extraction
    - Shared utility methods
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize vector store with configuration.
        
        Args:
            config: Configuration dictionary with backend-specific parameters
        """
        # ConfigurableBase doesn't have __init__, so don't call super().__init__()
        self.config = config or {}
        self._parse_common_config()
        self._parse_backend_config()
        self._initialized = False

    def _parse_common_config(self) -> None:
        """Parse common configuration parameters.
        
        Extracts parameters that are common to all vector stores:
        - dimensions: Vector dimensions
        - metric: Distance metric
        - persist_path: Path for persistent storage
        - batch_size: Batch size for operations
        - index_params: Index-specific parameters
        - search_params: Search-specific parameters
        """
        # Extract dimensions (required for most stores)
        self.dimensions = self.config.get("dimensions", 0)

        # Extract and parse metric
        metric = self.config.get("metric", "cosine")
        if isinstance(metric, str):
            self.metric = DistanceMetric(metric)
        else:
            self.metric = metric

        # Extract paths and sizes (expand ~ to home directory)
        persist_path = self.config.get("persist_path")
        self.persist_path = Path(persist_path).expanduser() if persist_path else None
        self.batch_size = self.config.get("batch_size", 100)

        # Debug logging for path resolution
        import logging
        logger = logging.getLogger(__name__)
        if persist_path:
            logger.info(f"VectorStore persist_path: {persist_path} -> {self.persist_path} (exists: {os.path.exists(self.persist_path) if self.persist_path else False})")

        # Extract parameter dictionaries
        self.index_params = self.config.get("index_params", {})
        self.search_params = self.config.get("search_params", {})

        # Store any additional metadata
        self.metadata = self.config.get("metadata", {})

    def _parse_backend_config(self) -> None:
        """Parse backend-specific configuration.
        
        Override this method in subclasses to handle backend-specific parameters.
        """
        pass

    def _validate_dimensions(self) -> None:
        """Validate vector dimensions.
        
        Raises:
            ValueError: If dimensions are invalid
        """
        if self.dimensions <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.dimensions}")
        if self.dimensions > 65536:
            raise ValueError(f"Dimensions {self.dimensions} exceeds maximum (65536)")

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector for cosine similarity.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        import numpy as np

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate similarity between two vectors based on configured metric.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        import numpy as np

        if self.metric == DistanceMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            # Convert distance to similarity
            distance = float(np.linalg.norm(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            # Dot product
            return float(np.dot(vec1, vec2))

        elif self.metric == DistanceMetric.L1:
            # Manhattan distance to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            return 1.0 / (1.0 + distance)

        else:
            # Default to cosine
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _convert_distance_to_score(self, distance: float) -> float:
        """Convert a distance to a similarity score based on metric.
        
        Args:
            distance: Distance value
            
        Returns:
            Similarity score (higher is more similar)
        """
        if self.metric == DistanceMetric.COSINE:
            # Cosine distance is 1 - similarity
            return 1.0 - distance
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            # Convert distance to similarity
            return 1.0 / (1.0 + distance)
        elif self.metric == DistanceMetric.L1:
            # Manhattan distance to similarity
            return 1.0 / (1.0 + distance)
        else:
            # For dot product and others, higher is better
            return distance

    def _prepare_vector(self, vector: np.ndarray | list[float] | list[np.ndarray], normalize: bool = False) -> np.ndarray:
        """Prepare a vector for storage or search.
        
        Args:
            vector: Input vector (numpy array, list of floats, or list of arrays)
            normalize: Whether to normalize for cosine similarity
            
        Returns:
            Prepared numpy array
        """
        import numpy as np

        # Convert to numpy array
        if isinstance(vector, list):
            if len(vector) > 0 and isinstance(vector[0], np.ndarray):
                # List of arrays - stack them
                vector = np.vstack(vector).astype(np.float32)
            else:
                # List of floats
                vector = np.array(vector, dtype=np.float32)
        else:
            vector = np.asarray(vector, dtype=np.float32)

        # Ensure vector is an ndarray at this point
        assert isinstance(vector, np.ndarray)

        # Ensure correct shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Normalize if needed (e.g., for cosine similarity)
        if normalize or self.metric == DistanceMetric.COSINE:
            # Apply normalization for cosine similarity
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vector = vector / norms

        return cast("np.ndarray", vector)

    def _apply_metadata_filter(self, candidates: list[tuple[Any, dict]], filter: dict[str, Any]) -> list[tuple[Any, dict]]:
        """Apply metadata filter to candidates.
        
        Args:
            candidates: List of (id, metadata) tuples
            filter: Filter criteria as key-value pairs
            
        Returns:
            Filtered list of candidates
        """
        if not filter:
            return candidates

        filtered = []
        for item_id, metadata in candidates:
            # Check if all filter conditions match
            match = all(
                metadata.get(key) == value
                for key, value in filter.items()
            )
            if match:
                filtered.append((item_id, metadata))

        return filtered

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"dimensions={self.dimensions}, "
            f"metric={self.metric.value})"
        )
