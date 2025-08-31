"""Base class for specialized vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ...fields import VectorField
from ...records import Record
from ..types import VectorSearchResult
from .common import VectorStoreBase

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Callable


class VectorStore(ABC, VectorStoreBase):
    """Abstract base class for specialized vector stores.
    
    This provides a dedicated vector storage backend that can be used
    independently or alongside traditional databases. It inherits from
    VectorStoreBase which provides common configuration parsing and
    utility methods.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create index, connect, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources."""
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the store.
        
        Args:
            vectors: Vector(s) to add
            ids: Optional IDs for vectors (generated if not provided)
            metadata: Optional metadata for each vector
            
        Returns:
            List of IDs for the added vectors
        """
        pass

    @abstractmethod
    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray, dict[str, Any] | None]]:
        """Retrieve vectors by ID.
        
        Args:
            ids: Vector IDs to retrieve
            include_metadata: Whether to include metadata
            
        Returns:
            List of (vector, metadata) tuples
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID.
        
        Args:
            ids: Vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            include_metadata: Whether to include metadata
            
        Returns:
            List of (id, score, metadata) tuples
        """
        pass

    @abstractmethod
    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors.
        
        Args:
            ids: Vector IDs to update
            metadata: New metadata for each vector
            
        Returns:
            Number of vectors updated
        """
        pass

    @abstractmethod
    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the store.
        
        Args:
            filter: Optional metadata filter
            
        Returns:
            Number of vectors matching filter
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors from the store."""
        pass

    async def update_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Update existing vectors by ID.
        
        This is a convenience method that deletes and re-adds vectors.
        Some vector stores may override this with a more efficient implementation.
        
        Args:
            vectors: New vector values
            ids: IDs of vectors to update
            metadata: Optional new metadata
            
        Returns:
            List of updated IDs
        """
        # Delete existing vectors
        await self.delete_vectors(ids)

        # Add new vectors with same IDs
        return await self.add_vectors(vectors, ids, metadata)

    # Higher-level convenience methods

    async def add_records(
        self,
        records: list[Record],
        vector_field: str = "embedding",
        include_fields: list[str] | None = None,
    ) -> list[str]:
        """Add records with vector fields to the store.
        
        Args:
            records: Records containing vector fields
            vector_field: Name of the vector field
            include_fields: Fields to include in metadata
            
        Returns:
            List of IDs for added vectors
        """
        vectors = []
        ids = []
        metadatas = []

        for record in records:
            # Extract vector
            if vector_field not in record.fields:
                continue

            vector_obj = record.fields[vector_field]
            if not isinstance(vector_obj, VectorField):
                continue
            
            # Skip records without IDs
            if record.id is None:
                continue

            vectors.append(vector_obj.value)
            ids.append(record.id)

            # Build metadata
            metadata = {"record_id": record.id}

            # Add source field if present
            if vector_obj.source_field:
                metadata["source_field"] = vector_obj.source_field
                # Include source text if available
                if vector_obj.source_field in record.fields:
                    metadata["source_text"] = record.get_value(vector_obj.source_field)

            # Add model info if present
            if vector_obj.model_name:
                metadata["model_name"] = vector_obj.model_name
            if vector_obj.model_version:
                metadata["model_version"] = vector_obj.model_version

            # Add requested fields
            if include_fields:
                for field_name in include_fields:
                    if field_name in record.fields:
                        metadata[field_name] = record.get_value(field_name)

            metadatas.append(metadata)

        if vectors:
            return await self.add_vectors(vectors, ids=ids, metadata=metadatas)
        return []

    async def search_similar_records(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        fetch_records: Callable[[list[str]], list[Record]] | None = None,
    ) -> list[VectorSearchResult]:
        """Search and return results as VectorSearchResult objects.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            fetch_records: Optional function to fetch full records
            
        Returns:
            List of VectorSearchResult objects
        """
        results = await self.search(
            query_vector, k=k, filter=filter, include_metadata=True
        )

        search_results = []
        record_ids = []

        for vector_id, _score, metadata in results:
            record_id = metadata.get("record_id", vector_id) if metadata else vector_id
            record_ids.append(record_id)

        # Fetch full records if function provided
        records_map = {}
        if fetch_records and record_ids:
            records = fetch_records(record_ids)
            records_map = {r.id: r for r in records}

        for vector_id, score, metadata in results:
            record_id = metadata.get("record_id", vector_id) if metadata else vector_id

            # Get or create record
            if record_id in records_map:
                record = records_map[record_id]
            else:
                # Create minimal record with metadata
                record = Record({"id": record_id})
                if metadata:
                    for key, value in metadata.items():
                        if key not in ["record_id", "source_text", "source_field"]:
                            record.fields[key] = value

            # Extract source text
            source_text = None
            if metadata:
                source_text = metadata.get("source_text")

            search_results.append(
                VectorSearchResult(
                    record=record,
                    score=score,
                    source_text=source_text,
                    vector_field=metadata.get("source_field") if metadata else None,
                    metadata=metadata or {},
                )
            )

        return search_results

    async def bulk_embed_and_store(
        self,
        texts: list[str],
        embedding_fn: Callable[[list[str]], np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Embed texts and store vectors.
        
        Args:
            texts: Texts to embed
            embedding_fn: Function to generate embeddings
            ids: Optional IDs for vectors
            metadata: Optional metadata for each vector
            batch_size: Batch size for embedding
            
        Returns:
            List of IDs for added vectors
        """
        batch_size = batch_size or self.batch_size
        all_ids = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids else None
            batch_metadata = metadata[i:i + batch_size] if metadata else None

            # Generate embeddings
            embeddings = embedding_fn(batch_texts)

            # Add source text to metadata
            if batch_metadata is None:
                batch_metadata = [{} for _ in batch_texts]

            for j, text in enumerate(batch_texts):
                batch_metadata[j]["source_text"] = text

            # Store vectors
            stored_ids = await self.add_vectors(
                embeddings, ids=batch_ids, metadata=batch_metadata
            )
            all_ids.extend(stored_ids)

        return all_ids
