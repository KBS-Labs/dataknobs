"""Specialized vector store implementations."""

from typing import Type

from dataknobs_common import Registry

from .base import VectorStore


class VectorBackendRegistry(Registry[Type[VectorStore]]):
    """Registry of available vector store backends.

    This registry manages vector store backend classes and their metadata.
    Backends are auto-registered on import if their dependencies are available.
    """

    def __init__(self) -> None:
        """Initialize the vector backend registry."""
        super().__init__("vector_backends", enable_metrics=True)
        self._register_builtin_backends()

    def _register_builtin_backends(self) -> None:
        """Auto-register all available built-in vector backends."""
        # Memory backend (always available)
        try:
            from .memory import MemoryVectorStore

            self.register(
                "memory",
                MemoryVectorStore,
                metadata={
                    "description": "In-memory vector storage for testing",
                    "persistent": False,
                    "requires_install": False,
                    "config_options": {
                        "dimensions": "Vector dimensions (required)",
                        "metric": "Distance metric: cosine, euclidean, dot_product",
                    },
                },
            )
        except ImportError:
            pass

        # Faiss backend
        try:
            from .faiss import FaissVectorStore

            self.register(
                "faiss",
                FaissVectorStore,
                metadata={
                    "description": "Facebook AI Similarity Search - efficient vector search",
                    "persistent": True,
                    "requires_install": "pip install faiss-cpu",
                    "config_options": {
                        "dimensions": "Vector dimensions (required)",
                        "metric": "Distance metric: cosine, euclidean, dot_product",
                        "index_type": "Index type: flat, ivfflat, hnsw, auto",
                        "persist_path": "Path to save/load index",
                        "nlist": "Number of clusters for IVF index",
                        "m": "Number of connections for HNSW",
                    },
                },
            )
        except ImportError:
            pass

        # Chroma backend
        try:
            from .chroma import ChromaVectorStore

            self.register(
                "chroma",
                ChromaVectorStore,
                metadata={
                    "description": "ChromaDB - AI-native vector database",
                    "persistent": True,
                    "requires_install": "pip install chromadb",
                    "config_options": {
                        "collection_name": "Name of the collection",
                        "persist_path": "Path for persistent storage",
                        "embedding_function": "Embedding function name or object",
                        "metric": "Distance metric: cosine, euclidean, dot_product",
                    },
                },
            )
            self.register("chromadb", ChromaVectorStore)  # Alias
        except ImportError:
            pass


# Create singleton instance BEFORE importing factory to avoid circular import
vector_backends = VectorBackendRegistry()

# Now import factory (which will import vector_backends from this module)
from .factory import VectorStoreFactory  # noqa: E402


__all__ = [
    "VectorStore",
    "VectorStoreFactory",
    "VectorBackendRegistry",
    "vector_backends",
]

# Import specialized stores when available
try:
    from .faiss import FaissVectorStore

    __all__ += ["FaissVectorStore"]
except ImportError:
    pass

try:
    from .chroma import ChromaVectorStore

    __all__ += ["ChromaVectorStore"]
except ImportError:
    pass
