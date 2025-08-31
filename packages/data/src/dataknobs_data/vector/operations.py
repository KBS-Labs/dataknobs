"""Base vector operations and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .types import DistanceMetric


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

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


def compute_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: DistanceMetric,
) -> float:
    """Compute distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector
        metric: Distance metric to use

    Returns:
        Distance value
    """
    import numpy as np

    if metric == DistanceMetric.COSINE:
        # Cosine similarity (1 - similarity for distance)
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot / (norm1 * norm2)
        return float(1 - similarity)

    elif metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
        # Euclidean/L2 distance
        return float(np.linalg.norm(vec1 - vec2))

    elif metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
        # Negative dot product (for distance-based sorting)
        return float(-np.dot(vec1, vec2))

    elif metric == DistanceMetric.L1:
        # Manhattan/L1 distance
        return float(np.sum(np.abs(vec1 - vec2)))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def compute_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: DistanceMetric,
) -> float:
    """Compute similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector
        metric: Distance metric to use

    Returns:
        Similarity value (higher is more similar)
    """
    import numpy as np

    if metric == DistanceMetric.COSINE:
        # Cosine similarity
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    elif metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
        # Convert distance to similarity (inverse)
        distance = np.linalg.norm(vec1 - vec2)
        return float(1.0 / (1.0 + distance))

    elif metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
        # Direct dot product
        return float(np.dot(vec1, vec2))

    elif metric == DistanceMetric.L1:
        # Convert L1 distance to similarity
        distance = np.sum(np.abs(vec1 - vec2))
        return float(1.0 / (1.0 + distance))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def batch_compute_distances(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    """Compute distances from query vector to multiple vectors.

    Args:
        query_vector: Query vector (1D)
        vectors: Matrix of vectors (2D)
        metric: Distance metric to use

    Returns:
        Array of distances
    """
    import numpy as np

    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, -1)

    if metric == DistanceMetric.COSINE:
        # Batch cosine similarity
        dots = np.dot(vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vectors, axis=1)

        # Handle zero norms
        valid = (vector_norms != 0) & (query_norm != 0)
        similarities = np.zeros(len(vectors))
        similarities[valid] = dots[valid] / (vector_norms[valid] * query_norm)

        # Convert to distances
        return 1 - similarities

    elif metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
        # Batch Euclidean distance
        return np.linalg.norm(vectors - query_vector, axis=1)

    elif metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
        # Negative dot product for distance
        return -np.dot(vectors, query_vector)

    elif metric == DistanceMetric.L1:
        # Batch L1 distance
        return np.sum(np.abs(vectors - query_vector), axis=1)

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def validate_vector_dimensions(
    vector: np.ndarray | list[float],
    expected_dims: int,
    field_name: str | None = None,
) -> np.ndarray:
    """Validate and convert vector to proper format.

    Args:
        vector: Vector to validate
        expected_dims: Expected number of dimensions
        field_name: Optional field name for error messages

    Returns:
        Validated numpy array

    Raises:
        ValueError: If dimensions don't match
    """
    import numpy as np

    if isinstance(vector, list):
        vector = np.array(vector, dtype=np.float32)

    actual_dims = len(vector) if vector.ndim == 1 else vector.shape[-1]

    if actual_dims != expected_dims:
        field_str = f" for field '{field_name}'" if field_name else ""
        raise ValueError(
            f"Vector dimension mismatch{field_str}: "
            f"expected {expected_dims}, got {actual_dims}"
        )

    return vector


def chunk_vectors(
    vectors: np.ndarray | list[np.ndarray],
    chunk_size: int,
) -> list[np.ndarray]:
    """Split vectors into chunks for batch processing.

    Args:
        vectors: Vectors to chunk
        chunk_size: Maximum chunk size

    Returns:
        List of vector chunks
    """
    import numpy as np

    if isinstance(vectors, list):
        # List of individual vectors
        chunks = []
        for i in range(0, len(vectors), chunk_size):
            chunk = vectors[i:i + chunk_size]
            chunks.append(np.array(chunk))
        return chunks
    else:
        # Numpy array
        chunks = []
        for i in range(0, len(vectors), chunk_size):
            chunks.append(vectors[i:i + chunk_size])
        return chunks


def estimate_memory_usage(
    num_vectors: int,
    dimensions: int,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Estimate memory usage for vector storage.

    Args:
        num_vectors: Number of vectors
        dimensions: Number of dimensions per vector
        dtype: Data type for vectors

    Returns:
        Dictionary with memory estimates
    """
    bytes_per_element = {
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
    }.get(dtype, 4)

    vector_bytes = num_vectors * dimensions * bytes_per_element

    # Add overhead estimates
    index_overhead = vector_bytes * 0.1  # ~10% for basic index
    metadata_overhead = num_vectors * 100  # ~100 bytes per vector for metadata

    total_bytes = vector_bytes + index_overhead + metadata_overhead

    return {
        "vector_storage_mb": vector_bytes / (1024 * 1024),
        "index_overhead_mb": index_overhead / (1024 * 1024),
        "metadata_overhead_mb": metadata_overhead / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }
