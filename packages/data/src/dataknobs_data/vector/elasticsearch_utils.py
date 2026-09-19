# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Elasticsearch-specific vector utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .types import DistanceMetric

logger = logging.getLogger(__name__)


#: The ``dense_vector`` similarities Elasticsearch offers, keyed on the
#: canonical member so the table has one entry per metric rather than one per
#: spelling. ``L1`` is absent because Elasticsearch has no Manhattan
#: similarity --- not because the table forgot it.
_SIMILARITIES: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.DOT_PRODUCT: "dot_product",
    DistanceMetric.EUCLIDEAN: "l2_norm",
}


def get_similarity_for_metric(metric: DistanceMetric | str) -> str:
    """The Elasticsearch ``dense_vector`` similarity for a distance metric.

    Keyed on :meth:`DistanceMetric.canonical` and refusing what it cannot
    serve. It was keyed on the member and ended in ``.get(metric, "cosine")``,
    which is the pgvector defect in another file: ``L2`` and ``INNER_PRODUCT``
    are spellings the table did not list, so an explicit
    ``create_vector_index(metric="l2")`` built a mapping with
    ``similarity: cosine`` and reported success. Every vector written into
    that field was then ranked under a metric nobody asked for, and the only
    way to notice was to read the mapping back.

    Args:
        metric: A member, member value, or published alias.

    Returns:
        The similarity name for the field mapping.

    Raises:
        ValueError: If the name is not an accepted spelling, or names a
            metric Elasticsearch has no similarity for.
    """
    canonical = DistanceMetric.resolve(metric).canonical()
    try:
        similarity = _SIMILARITIES[canonical]
    except KeyError:
        offered = ", ".join(sorted(m.value for m in _SIMILARITIES))
        raise ValueError(
            f"Elasticsearch has no dense_vector similarity for {canonical.value!r}; "
            f"it offers: {offered}"
        ) from None
    logger.debug("Using similarity '%s' for metric %s", similarity, canonical)
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

    # Build the KNN query. Annotated because ``filter`` below puts a nested
    # query object in, which the inferred value type from the four scalars
    # does not admit.
    knn_query: dict[str, Any] = {
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
    metric: DistanceMetric | str = DistanceMetric.COSINE,
    filter_query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a script_score query for exact vector search.

    The query-side twin of :func:`get_similarity_for_metric`, and keyed the
    same way for the same reason. That one builds the ``dense_vector``
    mapping; this builds the query that reads it, so the two are one decision
    about a metric taken in two places. It branched on the **member** and
    ended in ``else: # Default to cosine``, which meant ``L2`` --- a member a
    consumer reaches by configuring ``metric="l2"`` --- mapped to ``l2_norm``
    at one door and was queried with ``cosineSimilarity`` at the other, with
    nothing reporting the disagreement.

    Args:
        query_vector: Query vector
        field_name: Name of the vector field
        metric: A member, member value, or published alias.
        filter_query: Optional filter query

    Returns:
        Elasticsearch script_score query

    Raises:
        ValueError: If the name is not an accepted spelling, or names a
            metric Elasticsearch has no painless function for.
    """
    # Convert numpy array to list if needed
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    # Build the script based on metric
    field_path = f"data.{field_name}"

    # Keyed on the canonical member, so there is one entry per family and no
    # spelling can be missed. ``L1`` is absent for the reason it is absent
    # from ``_SIMILARITIES``: painless has ``cosineSimilarity``,
    # ``dotProduct`` and ``l2norm`` and no Manhattan function, so a refusal
    # is the honest answer and it is the one the mapping door already gives.
    canonical = DistanceMetric.resolve(metric).canonical()
    scripts = {
        DistanceMetric.COSINE: f"cosineSimilarity(params.query_vector, '{field_path}') + 1.0",
        DistanceMetric.DOT_PRODUCT: f"dotProduct(params.query_vector, '{field_path}')",
        DistanceMetric.EUCLIDEAN: f"1 / (1 + l2norm(params.query_vector, '{field_path}'))",
    }
    try:
        script_source = scripts[canonical]
    except KeyError:
        offered = ", ".join(sorted(m.value for m in scripts))
        raise ValueError(
            f"Elasticsearch has no script_score function for {canonical.value!r}; "
            f"it offers: {offered}"
        ) from None

    # Build the query
    base_query = filter_query if filter_query else {"match_all": {}}

    return {
        "script_score": {
            "query": base_query,
            "script": {"source": script_source, "params": {"query_vector": query_vector}},
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
