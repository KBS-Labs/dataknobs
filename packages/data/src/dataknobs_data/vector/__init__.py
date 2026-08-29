"""Vector store support for DataKnobs data package.

This module provides vector field types, operations, and backend integrations
for working with vector embeddings and semantic search.
"""

from ..fields import VectorField
from .content import (
    CONTENT_HASH_KEY,
    DEFAULT_FIELD_SEPARATOR,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    assemble_source_text,
    compute_content_hash,
    content_hash_metadata,
    recompute_content_hash,
)
from .exceptions import (
    VectorBackendError,
    VectorDimensionError,
    VectorDomainScopeError,
    VectorError,
    VectorIndexError,
    VectorNotSupportedError,
    VectorValidationError,
)
from .migration import IncrementalVectorizer, VectorMigration
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
from .sync import VectorTextSynchronizer
from .tracker import ChangeTracker
from .types import (
    DistanceMetric,
    VectorConfig,
    VectorIndexConfig,
    VectorMetadata,
    VectorSearchResult,
)
from .hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
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
    # Hybrid Search
    "FusionStrategy",
    "HybridSearchConfig",
    "HybridSearchResult",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
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
    # Synchronization
    "VectorTextSynchronizer",
    "ChangeTracker",
    # Staleness — how source fields become the text a vector was built from
    "CONTENT_HASH_KEY",
    "DEFAULT_FIELD_SEPARATOR",
    "FIELD_SEPARATOR_KEY",
    "SOURCE_FIELDS_KEY",
    "assemble_source_text",
    "compute_content_hash",
    "content_hash_metadata",
    "recompute_content_hash",
    # Migration
    "VectorMigration",
    "IncrementalVectorizer",
    # Exceptions
    "VectorBackendError",
    "VectorDimensionError",
    "VectorDomainScopeError",
    "VectorError",
    "VectorIndexError",
    "VectorNotSupportedError",
    "VectorValidationError",
]
