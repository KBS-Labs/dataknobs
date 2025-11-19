"""Faiss vector store implementation."""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore

if TYPE_CHECKING:
    import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorStore(VectorStore):
    """Faiss-based vector store for efficient similarity search.
    
    Faiss is a library for efficient similarity search and clustering of dense vectors.
    It provides various index types optimized for different use cases:
    - Flat: Exact search, best for small datasets
    - IVF: Inverted file index, good for medium datasets  
    - HNSW: Hierarchical navigable small world, good for large datasets
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Faiss vector store."""
        if not FAISS_AVAILABLE:
            raise ImportError(
                "Faiss is not installed. Install with: pip install faiss-cpu"
            )

        super().__init__(config)
        self.index = None
        self.id_map = {}  # Map from our IDs to Faiss internal indices
        self.metadata_store = {}  # Store metadata separately
        self.next_idx = 0

    def _parse_backend_config(self) -> None:
        """Parse Faiss-specific configuration."""
        # Determine index type
        self.index_type = self.index_params.get("type", "auto")
        if "index_type" in self.config:
            self.index_type = self.config["index_type"]

        self.nlist = self.index_params.get("nlist", 100)  # For IVF
        self.m = self.index_params.get("m", 32)  # For HNSW
        self.ef_construction = self.index_params.get("ef_construction", 200)  # For HNSW
        self.ef_search = self.index_params.get("ef_search", 50)  # For HNSW search
        self.nprobe = self.search_params.get("nprobe", 10)  # For IVF search

    async def initialize(self) -> None:
        """Initialize Faiss index."""
        if self._initialized:
            return

        # Create index based on type and metric
        self.index = self._create_index()

        # Load existing index if persist path exists
        if self.persist_path and os.path.exists(self.persist_path):
            await self.load()

        self._initialized = True

    def _create_index(self) -> Any:
        """Create Faiss index based on configuration."""
        dimensions = self.dimensions

        # Map distance metrics
        if self.metric == DistanceMetric.COSINE:
            # For cosine similarity, we'll normalize vectors and use inner product
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
            metric = faiss.METRIC_L2
        elif self.metric in (DistanceMetric.DOT_PRODUCT, DistanceMetric.INNER_PRODUCT):
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        # Auto-select index type based on expected dataset size
        if self.index_type == "auto":
            # Use flat for small dimensions/datasets
            if dimensions < 100:
                self.index_type = "flat"
            else:
                self.index_type = "ivfflat"

        # Create index
        if self.index_type == "flat":
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dimensions)
            else:
                index = faiss.IndexFlatL2(dimensions)

        elif self.index_type == "ivfflat":
            # Create quantizer
            quantizer = faiss.IndexFlatL2(dimensions)
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexIVFFlat(quantizer, dimensions, self.nlist, metric)
            else:
                index = faiss.IndexIVFFlat(quantizer, dimensions, self.nlist)

        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimensions, self.m, metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

        elif self.index_type == "ivfpq":
            # Product quantization for compression
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(dimensions)
            index = faiss.IndexIVFPQ(quantizer, dimensions, self.nlist, m, 8)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Wrap with IDMap to maintain our own IDs
        index = faiss.IndexIDMap(index)

        return index

    async def close(self) -> None:
        """Save and close the index."""
        if self.persist_path and self._initialized:
            await self.save()
        self._initialized = False

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the index."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Prepare vectors using common method
        vectors = self._prepare_vector(vectors, normalize=(self.metric == DistanceMetric.COSINE))

        # For Faiss, we need to ensure vectors are C-contiguous
        if not vectors.flags['C_CONTIGUOUS']:
            vectors = np.ascontiguousarray(vectors)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Generate metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]

        # Train index if needed (for IVF types)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if len(vectors) >= self.nlist:
                self.index.train(vectors)
            else:
                # Not enough vectors to train, use flat index temporarily
                pass

        # Map IDs to internal indices
        internal_ids = []
        for i, ext_id in enumerate(ids):
            internal_id = self.next_idx
            self.next_idx += 1
            self.id_map[ext_id] = internal_id
            self.metadata_store[internal_id] = metadata[i]
            internal_ids.append(internal_id)

        # Add to index with internal IDs
        internal_ids_array = np.array(internal_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, internal_ids_array)

        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID."""
        if not self._initialized:
            await self.initialize()


        results: list[tuple[np.ndarray | None, dict[str, Any] | None]] = []
        for ext_id in ids:
            if ext_id not in self.id_map:
                results.append((None, None))
                continue

            internal_id = self.id_map[ext_id]

            # Reconstruct vector from index
            try:
                vector = self.index.reconstruct(int(internal_id))
                metadata = self.metadata_store.get(internal_id) if include_metadata else None
                results.append((vector, metadata))
            except Exception:
                results.append((None, None))

        return results

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Get internal IDs
        internal_ids = []
        for ext_id in ids:
            if ext_id in self.id_map:
                internal_id = self.id_map[ext_id]
                internal_ids.append(internal_id)
                del self.id_map[ext_id]
                if internal_id in self.metadata_store:
                    del self.metadata_store[internal_id]

        if internal_ids:
            # Remove from index
            internal_ids_array = np.array(internal_ids, dtype=np.int64)
            removed = self.index.remove_ids(internal_ids_array)
            return removed

        return 0

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, dict[str, Any] | None]]:
        """Search for similar vectors."""
        if not self._initialized:
            await self.initialize()

        # Prepare query vector using common method
        query = self._prepare_vector(query_vector, normalize=(self.metric == DistanceMetric.COSINE))

        # Set search parameters for IVF
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than we have
        if k == 0:
            return []

        scores, indices = self.index.search(query, k)

        # Convert results
        results = []
        reverse_id_map = {v: k for k, v in self.id_map.items()}

        for i in range(len(indices[0])):
            internal_id = indices[0][i]
            if internal_id == -1:  # No result
                continue

            score = float(scores[0][i])

            # Convert score based on metric
            if self.metric == DistanceMetric.COSINE:
                # Inner product of normalized vectors = cosine similarity
                score = score  # noqa: PLW0127 - Keep for clarity
            elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
                # Convert distance to similarity score
                score = 1.0 / (1.0 + score)

            # Get external ID
            ext_id = reverse_id_map.get(internal_id, str(internal_id))

            # Apply metadata filter if provided
            metadata = self.metadata_store.get(internal_id) if include_metadata else None
            if filter and metadata:
                # Simple key-value matching
                match = all(
                    metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue

            results.append((ext_id, score, metadata))

        return results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        updated = 0
        for ext_id, meta in zip(ids, metadata, strict=False):
            if ext_id in self.id_map:
                internal_id = self.id_map[ext_id]
                self.metadata_store[internal_id] = meta
                updated += 1

        return updated

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the store."""
        if not self._initialized:
            await self.initialize()

        if filter is None:
            return self.index.ntotal

        # Count with filter
        count = 0
        for metadata in self.metadata_store.values():
            match = all(
                metadata.get(key) == value
                for key, value in filter.items()
            )
            if match:
                count += 1

        return count

    async def clear(self) -> None:
        """Clear all vectors from the store."""
        if not self._initialized:
            await self.initialize()

        # Reset everything
        self.index = self._create_index()
        self.id_map.clear()
        self.metadata_store.clear()
        self.next_idx = 0

    async def save(self) -> None:
        """Save index and metadata to disk."""
        if not self.persist_path:
            return

        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)

        # Create directory if needed
        parent_dir = os.path.dirname(persist_path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save index
        faiss.write_index(self.index, persist_path_str)

        # Save metadata and mappings
        metadata_path = persist_path_str + ".meta"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "metadata_store": self.metadata_store,
                "next_idx": self.next_idx,
                "config": {
                    "dimensions": self.dimensions,
                    "metric": self.metric.value,
                    "index_type": self.index_type,
                }
            }, f)

    async def load(self) -> None:
        """Load index and metadata from disk."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.persist_path or not os.path.exists(self.persist_path):
            logger.debug(f"FAISS: No persist path or file not found: {self.persist_path}")
            return

        # Convert Path to string for FAISS
        persist_path_str = str(self.persist_path)

        # Load index
        self.index = faiss.read_index(persist_path_str)
        logger.info(f"FAISS: Loaded index from {persist_path_str} with {self.index.ntotal} vectors")

        # Load metadata and mappings
        metadata_path = persist_path_str + ".meta"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.metadata_store = data["metadata_store"]
                self.next_idx = data["next_idx"]
            logger.info(f"FAISS: Loaded metadata with {len(self.id_map)} entries")
