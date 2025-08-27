"""Vector store support for DataKnobs data package.

This module provides vector field types, operations, and backend integrations
for working with vector embeddings and semantic search.
"""

from ..fields import VectorField
from .exceptions import (
    VectorBackendError,
    VectorDimensionError,
    VectorError,
    VectorIndexError,
    VectorNotSupportedError,
    VectorValidationError,
)
from .mixins import VectorCapable, VectorOperationsMixin, VectorSyncMixin
from .operations import (
    batch_compute_distances,
    chunk_vectors,
    compute_distance,
    compute_similarity,
    estimate_memory_usage,
    normalize_vector,
    validate_vector_dimensions,
)
from .types import (
    DistanceMetric,
    VectorConfig,
    VectorIndexConfig,
    VectorMetadata,
    VectorSearchResult,
)

__all__ = [
    # Field
    "VectorField",
    # Types
    "DistanceMetric",
    "VectorConfig",
    "VectorIndexConfig",
    "VectorMetadata",
    "VectorSearchResult",
    # Mixins
    "VectorCapable",
    "VectorOperationsMixin",
    "VectorSyncMixin",
    # Operations
    "batch_compute_distances",
    "chunk_vectors",
    "compute_distance",
    "compute_similarity",
    "estimate_memory_usage",
    "normalize_vector",
    "validate_vector_dimensions",
    # Exceptions
    "VectorBackendError",
    "VectorDimensionError",
    "VectorError",
    "VectorIndexError",
    "VectorNotSupportedError",
    "VectorValidationError",
]