"""Mixins and protocols for vector-capable databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from ..fields import FieldType
from .hybrid import (
    FusionStrategy,
    HybridSearchConfig,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from .types import DistanceMetric, VectorSearchResult

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable
    from ..query import Query
    from ..records import Record


class VectorCapable(Protocol):
    """Protocol for backends that can handle vector operations."""

    async def has_vector_support(self) -> bool:
        """Check if backend has vector support available.

        Returns:
            True if vector operations are supported
        """
        ...

    async def enable_vector_support(self) -> bool:
        """Enable vector support (install extensions, configure indices, etc.).

        Returns:
            True if vector support was successfully enabled
        """
        ...

    async def detect_vector_fields(self, record: Record) -> list[str]:
        """Detect vector fields in a record.

        Args:
            record: Record to examine

        Returns:
            List of field names that contain vectors
        """
        return [
            field_name
            for field_name, field_obj in record.fields.items()
            if field_obj.type in (FieldType.VECTOR, FieldType.SPARSE_VECTOR)
        ]

    def get_vector_config(self) -> dict[str, Any]:
        """Get vector-specific configuration for this backend.

        Returns:
            Dictionary of vector configuration options
        """
        return {}


class VectorOperationsMixin(ABC):
    """Mixin providing vector operations for databases.

    This mixin should be added to database backend classes that support
    vector operations. It provides abstract methods that must be implemented
    by the concrete backend class.
    """

    @abstractmethod
    async def vector_search(
        self,
        query_vector: np.ndarray | list[float],
        vector_field: str = "embedding",
        k: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE,
        filter: Query | None = None,
        include_source: bool = True,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: The vector to search for
            vector_field: Name of the vector field to search
            k: Number of results to return
            metric: Distance metric to use
            filter: Optional query filter to apply before vector search
            include_source: Whether to include source text in results
            score_threshold: Optional minimum similarity score

        Returns:
            List of search results ordered by similarity
        """
        pass

    @abstractmethod
    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model

        Returns:
            List of record IDs that were processed
        """
        pass

    async def update_vector(
        self,
        record_id: str,
        vector_field: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector field for a specific record.

        Args:
            record_id: ID of the record to update
            vector_field: Name of the vector field
            vector: New vector value
            metadata: Optional metadata to attach

        Returns:
            True if update was successful
        """
        # Default implementation using standard update
        from ..fields import VectorField

        record = await self.read(record_id)  # type: ignore
        if not record:
            return False

        # Update the vector field
        record.fields[vector_field] = VectorField(
            name=vector_field,
            value=vector,
            metadata=metadata,
        )

        return await self.update(record_id, record) is not None  # type: ignore

    async def delete_from_index(
        self,
        record_id: str,
        vector_field: str = "embedding",
    ) -> bool:
        """Remove a record from the vector index.

        Args:
            record_id: ID of the record to remove
            vector_field: Name of the vector field

        Returns:
            True if deletion was successful
        """
        # Default implementation using standard delete
        return await self.delete(record_id)  # type: ignore

    async def create_vector_index(
        self,
        vector_field: str = "embedding",
        dimensions: int | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        index_type: str = "auto",
        **kwargs: Any,
    ) -> bool:
        """Create an index for vector similarity search.

        Args:
            vector_field: Name of the vector field to index
            dimensions: Number of dimensions (if known)
            metric: Distance metric for the index
            index_type: Type of index to create
            **kwargs: Backend-specific index parameters

        Returns:
            True if index was created successfully
        """
        # Default no-op implementation
        return True

    async def drop_vector_index(
        self,
        vector_field: str = "embedding",
    ) -> bool:
        """Drop a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            True if index was dropped successfully
        """
        # Default no-op implementation
        return True

    async def get_vector_index_stats(
        self,
        vector_field: str = "embedding",
    ) -> dict[str, Any]:
        """Get statistics about a vector index.

        Args:
            vector_field: Name of the vector field

        Returns:
            Dictionary of index statistics
        """
        return {
            "field": vector_field,
            "indexed": False,
            "vector_count": 0,
        }

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray | list[float],
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        k: int = 10,
        config: HybridSearchConfig | None = None,
        filter: Query | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining text and vector similarity.

        This method combines traditional text search with vector similarity search
        using configurable fusion strategies. The default implementation performs
        both searches and merges results client-side. Backends with native hybrid
        search support (like Elasticsearch) can override this for better performance.

        Args:
            query_text: Text query for keyword/text matching
            query_vector: Vector for semantic similarity search
            text_fields: Fields to search for text matching (default: search all text fields)
            vector_field: Name of the vector field to search
            k: Number of results to return
            config: Hybrid search configuration (weights, fusion strategy)
            filter: Optional additional filters to apply
            metric: Distance metric for vector search

        Returns:
            List of HybridSearchResult ordered by combined score (descending)

        Example:
            ```python
            from dataknobs_data.vector import HybridSearchConfig, FusionStrategy

            # Default RRF fusion
            results = await db.hybrid_search(
                query_text="machine learning",
                query_vector=embedding,
                text_fields=["title", "content"],
                k=10,
            )

            # Custom weighted fusion
            config = HybridSearchConfig(
                text_weight=0.3,
                vector_weight=0.7,
                fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            )
            results = await db.hybrid_search(
                query_text="machine learning",
                query_vector=embedding,
                config=config,
            )
            ```
        """
        config = config or HybridSearchConfig()

        # If using NATIVE strategy but backend doesn't support it, fall back to RRF
        if config.fusion_strategy == FusionStrategy.NATIVE:
            if not await self._supports_native_hybrid():  # type: ignore[attr-defined]
                config = HybridSearchConfig(
                    text_weight=config.text_weight,
                    vector_weight=config.vector_weight,
                    fusion_strategy=FusionStrategy.RRF,
                    rrf_k=config.rrf_k,
                    text_fields=config.text_fields,
                )

        # Use config.text_fields if provided, otherwise use parameter
        search_text_fields = config.text_fields or text_fields

        # Get more results for fusion (we'll filter to k after combining)
        fetch_k = min(k * 3, 100)

        # Perform text search
        text_results = await self._text_search_for_hybrid(
            query_text=query_text,
            text_fields=search_text_fields,
            k=fetch_k,
            filter=filter,
        )

        # Perform vector search
        vector_results = await self.vector_search(
            query_vector=query_vector,
            vector_field=vector_field,
            k=fetch_k,
            metric=metric,
            filter=filter,
        )

        # Build ID->Record and ID->score maps
        records_by_id: dict[str, Record] = {}
        text_scores: list[tuple[str, float]] = []
        vector_scores: list[tuple[str, float]] = []

        for record, score in text_results:
            record_id = record.id or record.storage_id
            if record_id:
                records_by_id[record_id] = record
                text_scores.append((record_id, score))

        for result in vector_results:
            record_id = result.record.id or result.record.storage_id
            if record_id:
                records_by_id[record_id] = result.record
                vector_scores.append((record_id, result.score))

        # Fuse results
        if config.fusion_strategy == FusionStrategy.RRF:
            fused = reciprocal_rank_fusion(
                text_results=text_scores,
                vector_results=vector_scores,
                k=config.rrf_k,
                text_weight=config.text_weight,
                vector_weight=config.vector_weight,
            )
        else:  # WEIGHTED_SUM
            text_w, vector_w = config.normalize_weights()
            fused = weighted_score_fusion(
                text_results=text_scores,
                vector_results=vector_scores,
                text_weight=text_w,
                vector_weight=vector_w,
                normalize_scores=True,
            )

        # Build HybridSearchResult objects
        text_score_map = dict(text_scores)
        vector_score_map = dict(vector_scores)
        text_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(text_scores)}
        vector_rank_map = {rid: i + 1 for i, (rid, _) in enumerate(vector_scores)}

        results: list[HybridSearchResult] = []
        for record_id, combined_score in fused[:k]:
            if record_id not in records_by_id:
                continue

            results.append(HybridSearchResult(
                record=records_by_id[record_id],
                combined_score=combined_score,
                text_score=text_score_map.get(record_id),
                vector_score=vector_score_map.get(record_id),
                text_rank=text_rank_map.get(record_id),
                vector_rank=vector_rank_map.get(record_id),
                metadata={
                    "fusion_strategy": config.fusion_strategy.value,
                    "text_weight": config.text_weight,
                    "vector_weight": config.vector_weight,
                },
            ))

        return results

    async def _text_search_for_hybrid(
        self,
        query_text: str,
        text_fields: list[str] | None,
        k: int,
        filter: Query | None = None,
    ) -> list[tuple[Record, float]]:
        """Perform text search for hybrid search fusion.

        Default implementation uses LIKE query on text fields.
        Backends can override for better text search (e.g., full-text search).

        Args:
            query_text: Text to search for
            text_fields: Fields to search in
            k: Maximum results to return
            filter: Additional filters

        Returns:
            List of (record, score) tuples ordered by relevance
        """
        from ..query import Filter, Operator, Query

        # Build text search query
        query = filter.copy() if filter else Query()
        query.limit_value = k

        # Add text matching filters
        # For simple implementation, use LIKE on each text field with OR logic
        # This is a basic implementation; backends should override for better text search
        if text_fields:
            # Use first field for simplicity in default implementation
            # Backends with full-text search should override this
            for field in text_fields[:1]:  # Only use first field to avoid complex OR
                query.filters.append(Filter(
                    field=field,
                    operator=Operator.LIKE,
                    value=f"%{query_text}%",
                ))

        # Perform search
        records = await self.search(query)  # type: ignore[attr-defined]

        # Assign basic scores based on match quality
        results: list[tuple[Record, float]] = []
        query_lower = query_text.lower()
        for i, record in enumerate(records):
            # Calculate a simple relevance score
            score = 1.0 / (i + 1)  # Rank-based score

            # Boost exact matches
            for field in (text_fields or []):
                value = record.get_value(field)
                if value and isinstance(value, str):
                    if query_lower in value.lower():
                        score *= 1.5
                    if query_lower == value.lower():
                        score *= 2.0

            results.append((record, min(score, 1.0)))

        return results

    async def _supports_native_hybrid(self) -> bool:
        """Check if this backend supports native hybrid search.

        Override in backends that have native hybrid search support
        (e.g., Elasticsearch with RRF).

        Returns:
            True if native hybrid search is supported
        """
        return False


class VectorSyncMixin:
    """Mixin for synchronizing vectors with source text."""

    async def sync_vectors_with_text(
        self,
        records: list[Record],
        text_fields: list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        force: bool = False,
    ) -> int:
        """Synchronize vector embeddings with text content.

        Args:
            records: Records to synchronize
            text_fields: Text fields to generate vectors from
            vector_field: Vector field to update
            embedding_fn: Embedding function
            force: Force re-generation even if vectors exist

        Returns:
            Number of records updated
        """
        if not embedding_fn:
            raise ValueError("Embedding function is required for vector synchronization")

        updated = 0
        for record in records:
            # Check if vector needs update
            needs_update = force or vector_field not in record.fields

            if not needs_update:
                # Check if source fields changed
                vector_meta = record.fields[vector_field].metadata
                source_fields = vector_meta.get("source_field", "").split(",")
                needs_update = set(source_fields) != set(text_fields)

            if needs_update:
                # Concatenate text fields
                text_content = " ".join([
                    str(record.get_value(field))
                    for field in text_fields
                    if record.get_value(field)
                ])

                # Generate embedding
                if text_content:
                    from ..fields import VectorField

                    result = embedding_fn([text_content])
                    # Handle both sync and async embedding functions
                    if hasattr(result, '__await__'):
                        embeddings = await result  # type: ignore[misc]
                    else:
                        embeddings = result
                    record.fields[vector_field] = VectorField(
                        name=vector_field,
                        value=embeddings[0],
                        source_field=",".join(text_fields),
                    )
                    updated += 1

        return updated
