"""Factory for creating vector store backends."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from . import vector_backends
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

    def create(self, **config: Any) -> VectorStore:
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

        # Get backend class from registry
        try:
            backend_class = vector_backends.get(backend_type)
        except Exception as e:
            # Backend not found - provide helpful error message
            available = vector_backends.list_keys()
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: {', '.join(sorted(set(available)))}"
            ) from e

        # Create and return backend instance
        try:
            return backend_class(config)
        except ImportError as e:
            # Convert ImportError to ValueError with expected format
            # Extract package name from "pip install X" in error message
            import re
            match = re.search(r'pip install ([\w-]+)', str(e))
            if match:
                package = match.group(1)
                raise ValueError(
                    f"{backend_type.capitalize()} backend requires {package}"
                ) from e
            else:
                # Fallback if pattern doesn't match
                raise ValueError(
                    f"Backend '{backend_type}' has missing dependencies"
                ) from e

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """Get information about a specific backend.

        Args:
            backend_type: Name of the backend

        Returns:
            Dictionary with backend information from registry metadata
        """
        # Normalize to lowercase for case-insensitive lookup
        backend_type = backend_type.lower()

        # Check if backend exists first
        if not vector_backends.has(backend_type):
            return {
                "description": "Unknown backend",
                "error": f"Backend '{backend_type}' not recognized",
            }

        # Get metadata from registry
        metrics = vector_backends.get_metrics(backend_type)
        return metrics.get("metadata", {})


# Create singleton instance for registration
vector_store_factory = VectorStoreFactory()
