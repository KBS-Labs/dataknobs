"""In-memory vector store implementation."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from .base import VectorStore


class MemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and development.
    
    This implementation stores vectors in memory using numpy arrays
    and performs brute-force search. Suitable for small datasets
    and testing scenarios.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize memory vector store."""
        super().__init__(config)
        self.vectors = {}  # id -> vector
        self.metadata_store = {}  # id -> metadata

    async def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True

    async def close(self) -> None:
        """Close the store."""
        self._initialized = False

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to memory."""
        if not self._initialized:
            await self.initialize()

        # Convert to numpy array
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Store vectors and metadata
        for i, vector_id in enumerate(ids):
            self.vectors[vector_id] = vectors[i]
            if metadata and i < len(metadata):
                self.metadata_store[vector_id] = metadata[i]
            else:
                self.metadata_store[vector_id] = {}

        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray, dict[str, Any] | None]]:
        """Get vectors by ID."""
        if not self._initialized:
            await self.initialize()

        results = []
        for vector_id in ids:
            if vector_id in self.vectors:
                vector = self.vectors[vector_id]
                meta = self.metadata_store.get(vector_id) if include_metadata else None
                results.append((vector, meta))
            else:
                results.append((None, None))

        return results

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        deleted = 0
        for vector_id in ids:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                self.metadata_store.pop(vector_id, None)
                deleted += 1

        return deleted

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors using brute force."""
        if not self._initialized:
            await self.initialize()

        if not self.vectors:
            return []

        # Prepare query
        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Filter candidates
        candidates = []
        for vector_id, vector in self.vectors.items():
            # Apply metadata filter
            if filter:
                meta = self.metadata_store.get(vector_id, {})
                match = all(
                    meta.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue

            candidates.append((vector_id, vector))

        if not candidates:
            return []

        # Calculate distances using common method
        scores = []
        for vector_id, vector in candidates:
            score = self._calculate_similarity(query[0], vector)
            scores.append((vector_id, score))

        # Sort by score (descending for similarity)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for vector_id, score in scores[:k]:
            meta = self.metadata_store.get(vector_id) if include_metadata else None
            results.append((vector_id, score, meta))

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for vectors."""
        if not self._initialized:
            await self.initialize()

        updated = 0
        for vector_id, meta in zip(ids, metadata, strict=False):
            if vector_id in self.vectors:
                self.metadata_store[vector_id] = meta
                updated += 1

        return updated

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors."""
        if not self._initialized:
            await self.initialize()

        if filter is None:
            return len(self.vectors)

        # Count with filter
        count = 0
        for vector_id in self.vectors:
            meta = self.metadata_store.get(vector_id, {})
            match = all(
                meta.get(key) == value
                for key, value in filter.items()
            )
            if match:
                count += 1

        return count

    async def clear(self) -> None:
        """Clear all vectors."""
        if not self._initialized:
            await self.initialize()

        self.vectors.clear()
        self.metadata_store.clear()
