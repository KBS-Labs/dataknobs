"""Factory for creating vector store backends."""

import re
from typing import Any

from dataknobs_config import FactoryBase

from dataknobs_data.backend_selection import (
    available_backends,
    backend_available,
    backend_info,
    select_backend,
)

from . import vector_backends
from .base import VectorStore


class VectorStoreFactory(FactoryBase):
    """Factory for creating vector store backends dynamically.

    This factory allows creating different vector store implementations
    based on configuration, supporting specialized vector databases.

    Configuration Options:
        backend (str): Backend type. ``get_available_backends()`` reports the
            registered names; a config that omits the key falls back to
            ``memory`` and says so at WARNING.
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
        backend_class, backend_type, options = select_backend(
            config, vector_backends, kind="vector store"
        )

        # Create and return backend instance
        try:
            store: VectorStore = backend_class(options)
            return store
        except ImportError as e:
            # Convert ImportError to ValueError with expected format
            # Extract package name from "pip install X" in error message
            match = re.search(r"pip install ([\w-]+)", str(e))
            if match:
                package = match.group(1)
                raise ValueError(f"{backend_type.capitalize()} backend requires {package}") from e
            else:
                # Fallback if pattern doesn't match
                raise ValueError(f"Backend '{backend_type}' has missing dependencies") from e

    def get_available_backends(self) -> list[str]:
        """List the vector store backends this factory can create.

        Returns:
            Sorted canonical names. Registration aliases are collapsed, so a
            backend appears once however many spellings ``create`` accepts.
        """
        return available_backends(vector_backends)

    def is_backend_available(self, backend_type: str) -> bool:
        """Whether a vector store backend can be created here.

        Every store in this package defers its driver's ``ImportError`` to
        construction, so registration probes the declared driver instead --
        which is what makes this the guard the documentation describes
        rather than a restatement of "is the name known".

        Args:
            backend_type: Backend name or registration alias

        Returns:
            True when ``create(backend=backend_type)`` can resolve it.
        """
        return backend_available(vector_backends, backend_type)

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """Get information about a specific backend.

        Args:
            backend_type: Name of the backend

        Returns:
            Dictionary with backend information from registry metadata
        """
        return backend_info(vector_backends, backend_type)


# Create singleton instance for registration
vector_store_factory = VectorStoreFactory()
