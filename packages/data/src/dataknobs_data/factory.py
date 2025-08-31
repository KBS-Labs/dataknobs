"""Backend factory for dynamic database creation."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

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

    def create(self, **config) -> SyncDatabase:
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

        if backend_type in ("memory", "mem"):
            from dataknobs_data.backends.memory import SyncMemoryDatabase
            return SyncMemoryDatabase.from_config(config)

        elif backend_type == "file":
            from dataknobs_data.backends.file import SyncFileDatabase
            return SyncFileDatabase.from_config(config)

        elif backend_type in ("postgres", "postgresql", "pg"):
            try:
                from dataknobs_data.backends.postgres import SyncPostgresDatabase
                return SyncPostgresDatabase.from_config(config)
            except ImportError as e:
                raise ValueError(
                    "PostgreSQL backend requires psycopg2. "
                    "Install with: pip install dataknobs-data[postgres]"
                ) from e

        elif backend_type in ("elasticsearch", "es"):
            try:
                from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
                return SyncElasticsearchDatabase.from_config(config)
            except ImportError as e:
                raise ValueError(
                    "Elasticsearch backend requires elasticsearch package. "
                    "Install with: pip install dataknobs-data[elasticsearch]"
                ) from e

        elif backend_type == "sqlite":
            from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
            return SyncSQLiteDatabase.from_config(config)

        elif backend_type == "s3":
            try:
                from dataknobs_data.backends.s3 import SyncS3Database
                return SyncS3Database.from_config(config)
            except ImportError as e:
                raise ValueError(
                    "S3 backend requires boto3. "
                    "Install with: pip install dataknobs-data[s3]"
                ) from e

        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: memory, file, postgres, elasticsearch, sqlite, s3"
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
                "description": "In-memory storage for testing and caching",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "initial_data": "Optional initial data dictionary"
                }
            },
            "file": {
                "description": "File-based storage (JSON, CSV, Parquet)",
                "persistent": True,
                "requires_install": False,
                "config_options": {
                    "path": "Path to the file (required)",
                    "format": "File format: json, csv, parquet (default: json)",
                    "compression": "Optional compression: gzip, bz2, xz"
                }
            },
            "postgres": {
                "description": "PostgreSQL database backend with native vector support (pgvector)",
                "persistent": True,
                "requires_install": "pip install dataknobs-data[postgres]",
                "vector_support": True,
                "config_options": {
                    "host": "Database host (required)",
                    "port": "Database port (default: 5432)",
                    "database": "Database name (required)",
                    "user": "Username (required)",
                    "password": "Password (required)",
                    "table": "Table name (default: records)",
                    "vector_enabled": "Enable vector support (default: False)",
                    "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)"
                }
            },
            "elasticsearch": {
                "description": "Elasticsearch search engine backend with native KNN vector support",
                "persistent": True,
                "requires_install": "pip install dataknobs-data[elasticsearch]",
                "vector_support": True,
                "config_options": {
                    "hosts": "List of host URLs (required)",
                    "index": "Index name (required)",
                    "doc_type": "Document type (default: _doc)",
                    "username": "Optional username",
                    "password": "Optional password",
                    "vector_enabled": "Enable vector support (default: False)",
                    "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)"
                }
            },
            "sqlite": {
                "description": "SQLite database backend with Python-based vector support",
                "persistent": True,
                "requires_install": False,
                "vector_support": True,
                "config_options": {
                    "path": "Path to database file (required)",
                    "table": "Table name (default: records)",
                    "vector_enabled": "Enable vector support (default: False)",
                    "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)"
                }
            },
            "s3": {
                "description": "AWS S3 object storage backend",
                "persistent": True,
                "requires_install": "pip install dataknobs-data[s3]",
                "config_options": {
                    "bucket": "S3 bucket name (required)",
                    "prefix": "Object key prefix (default: records/)",
                    "region": "AWS region (default: us-east-1)",
                    "endpoint_url": "Custom endpoint for S3-compatible services",
                    "access_key_id": "AWS access key (or use IAM role)",
                    "secret_access_key": "AWS secret key (or use IAM role)"
                }
            }
        }

        return info.get(backend_type.lower(), {
            "description": "Unknown backend",
            "error": f"Backend '{backend_type}' not recognized"
        })


class AsyncDatabaseFactory(FactoryBase):
    """Factory for creating async database backends.
    
    Note: Currently only some backends support async operations.
    """

    def create(self, **config) -> Any:
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

        if backend_type in ("memory", "mem"):
            from dataknobs_data.backends.memory import AsyncMemoryDatabase
            return AsyncMemoryDatabase.from_config(config)

        elif backend_type == "file":
            from dataknobs_data.backends.file import AsyncFileDatabase
            return AsyncFileDatabase.from_config(config)

        elif backend_type in ("postgres", "postgresql", "pg"):
            from dataknobs_data.backends.postgres import AsyncPostgresDatabase
            return AsyncPostgresDatabase.from_config(config)

        elif backend_type in ("elasticsearch", "es"):
            from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
            return AsyncElasticsearchDatabase.from_config(config)

        elif backend_type == "s3":
            from dataknobs_data.backends.s3_async import AsyncS3Database
            return AsyncS3Database.from_config(config)

        elif backend_type == "sqlite":
            from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
            return AsyncSQLiteDatabase.from_config(config)

        else:
            raise ValueError(
                f"Backend '{backend_type}' does not support async operations yet. "
                f"Available async backends: memory, file, postgres, elasticsearch, s3, sqlite"
            )


# TODO: Add AsyncVectorStoreFactory when async vector stores are implemented
# The async vector store implementations (AsyncFaissVectorStore, AsyncChromaVectorStore, 
# AsyncMemoryVectorStore) and base class (AsyncVectorStore) need to be created first.


# Create singleton instances for registration
database_factory = DatabaseFactory()
async_database_factory = AsyncDatabaseFactory()
vector_store_factory = VectorStoreFactory()
# TODO: add an 'async_vector_store_factory = AsyncVectorStoreFactory()' when async vector stores are implemented
