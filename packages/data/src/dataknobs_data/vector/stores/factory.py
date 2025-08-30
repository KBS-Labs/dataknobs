"""Factory for creating vector store backends."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from .base import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory(FactoryBase):
    """Factory for creating vector store backends dynamically.
    
    This factory allows creating different vector store implementations
    based on configuration, supporting specialized vector databases.
    
    Configuration Options:
        backend (str): Backend type (faiss, chroma, memory)
        dimensions (int): Vector dimensions (required for some backends)
        **kwargs: Backend-specific configuration options
        
    Example Configuration:
        vector_stores:
          - name: main_vectors
            factory: vector_store
            backend: faiss
            dimensions: 768
            index_type: ivfflat
            persist_path: ./vectors/main
            
          - name: doc_search
            factory: vector_store
            backend: chroma
            collection_name: documents
            persist_path: ./chroma_db
    """

    def create(self, **config) -> VectorStore:
        """Create a vector store instance based on configuration.
        
        Args:
            **config: Configuration including 'backend' field and backend-specific options
            
        Returns:
            Instance of appropriate vector store backend
            
        Raises:
            ValueError: If backend type is not recognized or not available
        """
        backend_type = config.pop("backend", "memory").lower()

        logger.info(f"Creating vector store with backend: {backend_type}")

        if backend_type == "memory":
            # Simple in-memory implementation
            from .memory import MemoryVectorStore
            return MemoryVectorStore(config)

        elif backend_type == "faiss":
            try:
                from .faiss import FaissVectorStore
                return FaissVectorStore(config)
            except ImportError as e:
                raise ValueError(
                    "Faiss backend requires faiss-cpu. "
                    "Install with: pip install faiss-cpu"
                ) from e

        elif backend_type in ("chroma", "chromadb"):
            try:
                from .chroma import ChromaVectorStore
                return ChromaVectorStore(config)
            except ImportError as e:
                raise ValueError(
                    "Chroma backend requires chromadb. "
                    "Install with: pip install chromadb"
                ) from e

        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: memory, faiss, chroma"
            )

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """Get information about a specific backend.
        
        Args:
            backend_type: Name of the backend
            
        Returns:
            Dictionary with backend information
        """
        info = {
            "memory": {
                "description": "In-memory vector storage for testing",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product",
                }
            },
            "faiss": {
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
                }
            },
            "chroma": {
                "description": "ChromaDB - AI-native vector database",
                "persistent": True,
                "requires_install": "pip install chromadb",
                "config_options": {
                    "collection_name": "Name of the collection",
                    "persist_path": "Path for persistent storage",
                    "embedding_function": "Embedding function name or object",
                    "metric": "Distance metric: cosine, euclidean, dot_product",
                }
            },
        }

        return info.get(backend_type.lower(), {
            "description": "Unknown backend",
            "error": f"Backend '{backend_type}' not recognized"
        })


# Create singleton instance for registration
vector_store_factory = VectorStoreFactory()
