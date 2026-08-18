"""Backend factory for dynamic database creation."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from dataknobs_data.backend_selection import (
    available_backends,
    backend_available,
    backend_info,
    build_backend,
    select_backend,
)
from dataknobs_data.backends import async_backends, sync_backends
from dataknobs_data.database import SyncDatabase

# Import the VectorStoreFactory from vector.stores.factory
from dataknobs_data.vector.stores.factory import VectorStoreFactory


logger = logging.getLogger(__name__)


def _async_backend_unavailable(backend_type: str, available: str) -> str:
    """An unrecognised async backend usually exists, without an async variant.

    Different enough from "you typed it wrong" to be worth its own sentence,
    which is why the async factory does not share the default text.
    """
    return (
        f"Backend '{backend_type}' does not support async operations yet. "
        f"Available async backends: {available}"
    )


class DatabaseFactory(FactoryBase):
    """Factory for creating database backends dynamically.

    This factory allows creating different database implementations
    based on configuration, supporting all available backends.

    Configuration Options:
        backend (str): Backend type. ``get_available_backends()`` reports the
            registered names; a config that omits the key falls back to
            ``memory`` and says so at WARNING.
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
        backend_class, backend_type, options = select_backend(
            config, sync_backends, kind="database"
        )
        database: SyncDatabase = build_backend(
            backend_class, options, kind="database", backend_type=backend_type
        )
        return database

    def get_available_backends(self) -> list[str]:
        """List the backends this factory can create.

        Returns:
            Sorted canonical names. Registration aliases are collapsed, so a
            backend appears once however many spellings ``create`` accepts.
        """
        return available_backends(sync_backends)

    def is_backend_available(self, backend_type: str) -> bool:
        """Whether a backend can be created under this installation.

        Registration probes the backend's declared driver, so a registered
        name is one whose optional dependency is present. A backend that is
        known but uninstalled reports False here and still describes itself
        through :meth:`get_backend_info`, including what to install.

        Args:
            backend_type: Backend name or registration alias

        Returns:
            True when ``create(backend=backend_type)`` can resolve it.
        """
        return backend_available(sync_backends, backend_type)

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """Get information about a specific backend.

        Args:
            backend_type: Name of the backend

        Returns:
            Dictionary with backend information from registry metadata
        """
        return backend_info(sync_backends, backend_type)


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
        backend_class, backend_type, options = select_backend(
            config,
            async_backends,
            kind="async database",
            unknown_message=_async_backend_unavailable,
        )
        return build_backend(
            backend_class, options, kind="async database", backend_type=backend_type
        )

    def get_available_backends(self) -> list[str]:
        """List the backends this factory can create.

        Returns:
            Sorted canonical names of the backends with an async variant.
            Registration aliases are collapsed.
        """
        return available_backends(async_backends)

    def is_backend_available(self, backend_type: str) -> bool:
        """Whether a backend has an async variant under this installation.

        Two ways to be absent: no async variant exists, or its driver is
        not installed. Both report False; :meth:`get_backend_info` tells
        them apart, since only the second describes itself.

        Args:
            backend_type: Backend name or registration alias

        Returns:
            True when ``create(backend=backend_type)`` can resolve it.
        """
        return backend_available(async_backends, backend_type)

    def get_backend_info(self, backend_type: str) -> dict[str, Any]:
        """Get information about a specific async backend.

        Args:
            backend_type: Name of the backend

        Returns:
            Dictionary with backend information from registry metadata
        """
        return backend_info(async_backends, backend_type)


# TODO: Add AsyncVectorStoreFactory when async vector stores are implemented
# The async vector store implementations (AsyncFaissVectorStore, AsyncChromaVectorStore,
# AsyncMemoryVectorStore) and base class (AsyncVectorStore) need to be created first.


# Create singleton instances for registration
database_factory = DatabaseFactory()
async_database_factory = AsyncDatabaseFactory()
vector_store_factory = VectorStoreFactory()
# TODO: add an 'async_vector_store_factory = AsyncVectorStoreFactory()' when async vector stores are implemented
