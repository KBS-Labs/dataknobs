"""Elasticsearch-specific vector utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .types import DistanceMetric

logger = logging.getLogger(__name__)


def get_similarity_for_metric(metric: DistanceMetric) -> str:
    """Get Elasticsearch similarity function for a distance metric.
    
    Args:
        metric: Distance metric
        
    Returns:
        Elasticsearch similarity function name
    """
    mapping = {
        DistanceMetric.COSINE: "cosine",
        DistanceMetric.DOT_PRODUCT: "dot_product",
        DistanceMetric.EUCLIDEAN: "l2_norm",
        DistanceMetric.INNER_PRODUCT: "dot_product",
    }

    similarity = mapping.get(metric, "cosine")
    logger.debug(f"Using similarity '{similarity}' for metric {metric}")
    return similarity


def build_knn_query(
    query_vector: np.ndarray | list[float],
    field_name: str,
    k: int = 10,
    num_candidates: int | None = None,
    filter_query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a KNN query for Elasticsearch.
    
    Args:
        query_vector: Query vector
        field_name: Name of the vector field (will be prefixed with 'data.')
        k: Number of results to return
        num_candidates: Number of candidates to consider (default: k * 10)
        filter_query: Optional filter query
        
    Returns:
        Elasticsearch KNN query
    """
    # Convert numpy array to list if needed
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    # Default num_candidates if not specified
    if num_candidates is None:
        num_candidates = max(k * 10, 100)

    # Build the KNN query
    knn_query = {
        "field": f"data.{field_name}",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }

    # Add filter if provided
    if filter_query:
        knn_query["filter"] = filter_query

    return {"knn": knn_query}


def build_script_score_query(
    query_vector: np.ndarray | list[float],
    field_name: str,
    metric: DistanceMetric = DistanceMetric.COSINE,
    filter_query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a script_score query for exact vector search.
    
    Args:
        query_vector: Query vector
        field_name: Name of the vector field
        metric: Distance metric to use
        filter_query: Optional filter query
        
    Returns:
        Elasticsearch script_score query
    """
    # Convert numpy array to list if needed
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    # Build the script based on metric
    field_path = f"data.{field_name}"

    if metric == DistanceMetric.COSINE:
        script_source = f"cosineSimilarity(params.query_vector, '{field_path}') + 1.0"
    elif metric == DistanceMetric.DOT_PRODUCT or metric == DistanceMetric.INNER_PRODUCT:
        script_source = f"dotProduct(params.query_vector, '{field_path}')"
    elif metric == DistanceMetric.EUCLIDEAN:
        script_source = f"1 / (1 + l2norm(params.query_vector, '{field_path}'))"
    else:
        # Default to cosine
        script_source = f"cosineSimilarity(params.query_vector, '{field_path}') + 1.0"

    # Build the query
    base_query = filter_query if filter_query else {"match_all": {}}

    return {
        "script_score": {
            "query": base_query,
            "script": {
                "source": script_source,
                "params": {
                    "query_vector": query_vector
                }
            }
        }
    }


def build_hybrid_query(
    text_query: str,
    query_vector: np.ndarray | list[float],
    text_fields: list[str],
    vector_field: str,
    text_boost: float = 1.0,
    vector_boost: float = 1.0,
    k: int = 10,
) -> dict[str, Any]:
    """Build a hybrid text + vector search query.
    
    Args:
        text_query: Text query string
        query_vector: Query vector
        text_fields: Fields to search for text
        vector_field: Vector field name
        text_boost: Boost for text search
        vector_boost: Boost for vector search
        k: Number of results for KNN
        
    Returns:
        Elasticsearch hybrid query
    """
    # Convert numpy array to list if needed
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    # Build text query
    text_query_clause = {
        "multi_match": {
            "query": text_query,
            "fields": [f"data.{field}" for field in text_fields],
            "boost": text_boost,
        }
    }

    # Build KNN query
    knn_clause = {
        "field": f"data.{vector_field}",
        "query_vector": query_vector,
        "k": k,
        "boost": vector_boost,
    }

    # Combine with bool query
    return {
        "bool": {
            "should": [text_query_clause],
        },
        "knn": knn_clause,
    }


def format_vector_for_elasticsearch(vector: np.ndarray | list[float]) -> list[float]:
    """Format a vector for Elasticsearch storage.
    
    Args:
        vector: Vector to format
        
    Returns:
        List of floats suitable for Elasticsearch
    """
    if isinstance(vector, np.ndarray):
        # Convert to list and ensure float32
        result: list[float] = vector.astype(np.float32).tolist()
        return result
    elif isinstance(vector, list):
        # Ensure all values are floats
        return [float(v) for v in vector]
    else:
        raise ValueError(f"Unsupported vector type: {type(vector)}")


def parse_elasticsearch_vector(value: Any) -> np.ndarray | None:
    """Parse a vector value from Elasticsearch.
    
    Args:
        value: Value from Elasticsearch document
        
    Returns:
        Numpy array or None
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, np.ndarray):
        return value.astype(np.float32)
    else:
        logger.warning(f"Unexpected vector value type: {type(value)}")
        return None


def get_vector_mapping(
    dimensions: int,
    similarity: str = "cosine",
    index: bool = True,
) -> dict[str, Any]:
    """Get Elasticsearch mapping for a vector field.
    
    Args:
        dimensions: Number of dimensions
        similarity: Similarity metric (cosine, dot_product, l2_norm)
        index: Whether to index the field for KNN search
        
    Returns:
        Mapping dictionary for the field
    """
    return {
        "type": "dense_vector",
        "dims": dimensions,
        "index": index,
        "similarity": similarity,
    }


def estimate_index_parameters(num_vectors: int) -> dict[str, Any]:
    """Estimate optimal index parameters based on dataset size.
    
    Args:
        num_vectors: Expected number of vectors
        
    Returns:
        Dictionary of index parameters
    """
    # HNSW parameters based on dataset size
    if num_vectors < 10000:
        # Small dataset - prioritize accuracy
        return {
            "index.knn": True,
            "index.knn.algo_param.ef_construction": 200,
            "index.knn.algo_param.m": 16,
        }
    elif num_vectors < 100000:
        # Medium dataset - balance
        return {
            "index.knn": True,
            "index.knn.algo_param.ef_construction": 100,
            "index.knn.algo_param.m": 16,
        }
    else:
        # Large dataset - prioritize speed
        return {
            "index.knn": True,
            "index.knn.algo_param.ef_construction": 50,
            "index.knn.algo_param.m": 8,
        }


def validate_vector_dimensions(vector: np.ndarray | list[float], expected_dims: int) -> bool:
    """Validate that a vector has the expected dimensions.
    
    Args:
        vector: Vector to validate
        expected_dims: Expected number of dimensions
        
    Returns:
        True if dimensions match
    """
    if isinstance(vector, np.ndarray):
        actual_dims = vector.shape[0] if vector.ndim == 1 else vector.shape[-1]
    elif isinstance(vector, list):
        actual_dims = len(vector)
    else:
        # Unsupported vector type
        return False  # type: ignore[unreachable]

    if actual_dims != expected_dims:
        logger.warning(f"Vector dimension mismatch: expected {expected_dims}, got {actual_dims}")
        return False

    return True
