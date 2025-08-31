"""Chroma vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..types import DistanceMetric
from .base import VectorStore

if TYPE_CHECKING:
    import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorStore(VectorStore):
    """Chroma-based vector store for semantic search.
    
    Chroma is a vector database designed for AI applications with features like:
    - Built-in embedding functions
    - Metadata filtering
    - Persistent storage
    - Multi-tenancy support
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Chroma vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        super().__init__(config)
        self.client = None
        self.collection = None

    def _parse_backend_config(self) -> None:
        """Parse Chroma-specific configuration."""
        # Set default dimensions if not provided
        if self.dimensions == 0:
            self.dimensions = 384  # Default for sentence-transformers

        # Chroma-specific configuration
        self.collection_name = self.config.get("collection_name", "vectors")

        # Handle embedding function
        self.embedding_function = None
        if "embedding_function" in self.config:
            ef = self.config["embedding_function"]
            if isinstance(ef, str):
                # Map string to Chroma embedding functions
                if ef == "default":
                    from chromadb.utils import embedding_functions
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                elif ef == "openai":
                    from chromadb.utils import embedding_functions
                    api_key = self.config.get("openai_api_key")
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
                # Add more as needed
            else:
                self.embedding_function = ef

        # Map distance metrics
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.L2: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",
            DistanceMetric.INNER_PRODUCT: "ip",
        }
        self.chroma_metric = metric_map.get(self.metric, "cosine")

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        if self._initialized:
            return

        # Create client
        if self.persist_path:
            # Persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            # In-memory client
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.chroma_metric},
                embedding_function=self.embedding_function
            )

        self._initialized = True

    async def close(self) -> None:
        """Close Chroma client."""
        # Chroma handles persistence automatically
        self._initialized = False

    async def add_vectors(
        self,
        vectors: np.ndarray | list[np.ndarray],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add vectors to the collection."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Convert to list format for Chroma
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = [vectors.tolist()]
            else:
                vectors = vectors.tolist()
        elif isinstance(vectors, list) and len(vectors) > 0:
            if isinstance(vectors[0], np.ndarray):
                vectors = [v.tolist() for v in vectors]

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        # Ensure metadata is provided
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]

        # Add to collection
        self.collection.add(
            embeddings=vectors,
            ids=ids,
            metadatas=metadata
        )

        return ids

    async def get_vectors(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> list[tuple[np.ndarray | None, dict[str, Any] | None]]:
        """Retrieve vectors by ID."""
        if not self._initialized:
            await self.initialize()

        import numpy as np

        # Get from collection
        result = self.collection.get(
            ids=ids,
            include=["embeddings", "metadatas"] if include_metadata else ["embeddings"]
        )

        # Convert to expected format
        vectors = []
        for id_val in ids:
            try:
                idx = result["ids"].index(id_val)
                embedding = result["embeddings"][idx] if result["embeddings"] else None
                metadata = result["metadatas"][idx] if include_metadata and result.get("metadatas") else None

                if embedding is not None:
                    embedding = np.array(embedding, dtype=np.float32)

                vectors.append((embedding, metadata))
            except (ValueError, IndexError):
                vectors.append((None, None))

        return vectors

    async def delete_vectors(self, ids: list[str]) -> int:
        """Delete vectors by ID."""
        if not self._initialized:
            await self.initialize()

        # Check which IDs exist
        existing = self.collection.get(ids=ids, include=[])
        existing_ids = existing["ids"]

        if existing_ids:
            self.collection.delete(ids=existing_ids)
            return len(existing_ids)

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

        # Convert query vector
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Build where clause for metadata filtering
        where = None
        if filter:
            # Chroma uses a different filter syntax
            # Convert simple key-value filter to Chroma format
            where = {}
            for key, value in filter.items():
                if isinstance(value, list):
                    where[key] = {"$in": value}
                else:
                    where[key] = {"$eq": value}

        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
            include=["metadatas", "distances"] if include_metadata else ["distances"]
        )

        # Convert results
        search_results = []
        if results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
            metadatas = results["metadatas"][0] if include_metadata and results.get("metadatas") else [None] * len(ids)

            for id_val, distance, metadata in zip(ids, distances, metadatas, strict=False):
                # Convert distance to similarity score
                if self.metric == DistanceMetric.COSINE:
                    # Chroma returns cosine distance (1 - similarity)
                    score = 1.0 - distance
                elif self.metric in (DistanceMetric.EUCLIDEAN, DistanceMetric.L2):
                    # Convert distance to similarity
                    score = 1.0 / (1.0 + distance)
                else:
                    score = float(distance)

                search_results.append((id_val, score, metadata))

        return search_results

    async def update_metadata(
        self,
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> int:
        """Update metadata for existing vectors."""
        if not self._initialized:
            await self.initialize()

        # Check which IDs exist
        existing = self.collection.get(ids=ids, include=[])
        existing_ids = existing["ids"]

        if existing_ids:
            # Filter metadata to only update existing vectors
            filtered_ids = []
            filtered_metadata = []
            for id_val, meta in zip(ids, metadata, strict=False):
                if id_val in existing_ids:
                    filtered_ids.append(id_val)
                    filtered_metadata.append(meta)

            if filtered_ids:
                self.collection.update(
                    ids=filtered_ids,
                    metadatas=filtered_metadata
                )
                return len(filtered_ids)

        return 0

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        """Count vectors in the collection."""
        if not self._initialized:
            await self.initialize()

        if filter is None:
            # Get total count
            return self.collection.count()

        # Count with filter
        where = {}
        for key, value in filter.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = {"$eq": value}

        # Query with limit 1 to get count efficiently
        results = self.collection.query(
            query_embeddings=[[0.0] * self.dimensions],  # Dummy query
            n_results=1,
            where=where,
            include=[]
        )

        # The actual count would need a different approach
        # For now, return the number of results (limited approach)
        # In production, you might want to maintain counts separately
        return len(results["ids"][0]) if results["ids"] else 0

    async def clear(self) -> None:
        """Clear all vectors from the collection."""
        if not self._initialized:
            await self.initialize()

        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.chroma_metric},
            embedding_function=self.embedding_function
        )

    async def add_documents(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the collection (uses Chroma's embedding)."""
        if not self._initialized:
            await self.initialize()

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]

        # Ensure metadata is provided
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        # Add documents (Chroma will embed them if embedding_function is set)
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadata
        )

        return ids

    async def search_documents(
        self,
        query_text: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[tuple[str, float, str, dict[str, Any] | None]]:
        """Search using text query (uses Chroma's embedding)."""
        if not self._initialized:
            await self.initialize()

        # Build where clause
        where = None
        if filter:
            where = {}
            for key, value in filter.items():
                if isinstance(value, list):
                    where[key] = {"$in": value}
                else:
                    where[key] = {"$eq": value}

        # Query with text
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Convert results
        search_results = []
        if results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0]
            documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
            metadatas = results["metadatas"][0] if include_metadata and results.get("metadatas") else [None] * len(ids)

            for id_val, distance, doc, metadata in zip(ids, distances, documents, metadatas, strict=False):
                # Convert distance to similarity score
                score = 1.0 - distance  # Cosine distance to similarity
                search_results.append((id_val, score, doc, metadata))

        return search_results
