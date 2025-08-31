"""Mixins and protocols for vector-capable databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from ..fields import FieldType
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
