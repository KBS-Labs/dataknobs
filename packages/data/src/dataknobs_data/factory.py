"""Backend factory for dynamic database creation."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from dataknobs_data.backends import async_backends, sync_backends
from dataknobs_data.database import SyncDatabase

# Import the VectorStoreFactory from vector.stores.factory
from dataknobs_data.vector.stores.factory import VectorStoreFactory


logger = logging.getLogger(__name__)


class DatabaseFactory(FactoryBase):
    """Factory for creating database backends dynamically.
    
    This factory allows creating different database implementations
    based on configuration, supporting all available backends.
    
    Configuration Options:
        backend (str): Backend type (memory, file, postgres, elasticsearch, s3)
        **kwargs: Backend-specific configuration options
        
    Example Configuration:
        databases:
          - name: main
            factory: database
            backend: postgres
            host: localhost
            database: myapp
            
          - name: cache
            factory: database
            backend: memory
            
          - name: archive
            factory: database
            backend: s3
            bucket: my-archive-bucket
            prefix: archives/
    """

    def create(self, **config: Any) -> SyncDatabase:
        """Create a database instance based on configuration.

        Args:
            **config: Configuration including 'backend' field and backend-specific options

        Returns:
            Instance of appropriate database backend

        Raises:
            ValueError: If backend type is not recognized or not available
        """
        backend_type = config.pop("backend", "memory").lower()

        logger.info(f"Creating database with backend: {backend_type}")

        # Check if vector_enabled is set
        vector_enabled = config.get("vector_enabled", False)

        if vector_enabled:
            # All backends now have vector support (some native, some via Python)
            logger.debug(f"Vector support enabled for backend: {backend_type}")

        # Get backend class from registry
        try:
            backend_class = sync_backends.get(backend_type)
        except Exception as e:
            # Backend not found - provide helpful error message
            available = sync_backends.list_keys()
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: {', '.join(sorted(set(available)))}"
            ) from e

        # Create and return backend instance
        return backend_class.from_config(config)


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
        if not sync_backends.has(backend_type):
            return {
                "description": "Unknown backend",
                "error": f"Backend '{backend_type}' not recognized",
            }

        # Get metadata from registry
        metrics = sync_backends.get_metrics(backend_type)
        return metrics.get("metadata", {})


class AsyncDatabaseFactory(FactoryBase):
    """Factory for creating async database backends.
    
    Note: Currently only some backends support async operations.
    """

    def create(self, **config: Any) -> Any:
        """Create an async database instance.

        Args:
            **config: Configuration including 'backend' field

        Returns:
            Instance of appropriate async database backend

        Raises:
            ValueError: If backend doesn't support async operations
        """
        backend_type = config.pop("backend", "memory").lower()

        # Check if vector_enabled is set
        vector_enabled = config.get("vector_enabled", False)

        if vector_enabled:
            # All backends now have vector support (some native, some via Python)
            logger.debug(f"Vector support enabled for async backend: {backend_type}")

        # Get backend class from registry
        try:
            backend_class = async_backends.get(backend_type)
        except Exception as e:
            # Backend not found - provide helpful error message
            available = async_backends.list_keys()
            raise ValueError(
                f"Backend '{backend_type}' does not support async operations yet. "
                f"Available async backends: {', '.join(sorted(set(available)))}"
            ) from e

        # Create and return backend instance
        return backend_class.from_config(config)


# TODO: Add AsyncVectorStoreFactory when async vector stores are implemented
# The async vector store implementations (AsyncFaissVectorStore, AsyncChromaVectorStore, 
# AsyncMemoryVectorStore) and base class (AsyncVectorStore) need to be created first.


# Create singleton instances for registration
database_factory = DatabaseFactory()
async_database_factory = AsyncDatabaseFactory()
vector_store_factory = VectorStoreFactory()
# TODO: add an 'async_vector_store_factory = AsyncVectorStoreFactory()' when async vector stores are implemented
