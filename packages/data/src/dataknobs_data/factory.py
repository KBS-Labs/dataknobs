"""Backend factory for dynamic database creation."""

import logging
from typing import Any

from dataknobs_config import FactoryBase

from dataknobs_data.database import SyncDatabase

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
        
        # Check if vector_enabled is set (for future database backend integration)
        if config.get("vector_enabled", False):
            # Currently, vector support for database backends is not yet implemented
            # This will be added when database backends integrate VectorOperationsMixin
            raise ValueError(
                f"Vector-enabled mode for database backend '{backend_type}' is not yet implemented. "
                f"Use dedicated vector stores instead: faiss, chroma, memory_vector"
            )

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

        elif backend_type == "s3":
            try:
                from dataknobs_data.backends.s3 import SyncS3Database
                return SyncS3Database.from_config(config)
            except ImportError as e:
                raise ValueError(
                    "S3 backend requires boto3. "
                    "Install with: pip install dataknobs-data[s3]"
                ) from e
                
        # Vector store backends
        elif backend_type == "faiss":
            try:
                from dataknobs_data.vector.stores.faiss import FaissVectorStore
                return FaissVectorStore(config)
            except ImportError as e:
                raise ValueError(
                    "Faiss backend requires faiss-cpu. "
                    "Install with: pip install faiss-cpu"
                ) from e
                
        elif backend_type in ("chroma", "chromadb"):
            try:
                from dataknobs_data.vector.stores.chroma import ChromaVectorStore
                return ChromaVectorStore(config)
            except ImportError as e:
                raise ValueError(
                    "Chroma backend requires chromadb. "
                    "Install with: pip install chromadb"
                ) from e
                
        elif backend_type == "memory_vector":
            from dataknobs_data.vector.stores.memory import MemoryVectorStore
            return MemoryVectorStore(config)

        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available backends: memory, file, postgres, elasticsearch, s3, "
                f"faiss, chroma, memory_vector"
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
                "description": "PostgreSQL database backend",
                "persistent": True,
                "requires_install": "pip install dataknobs-data[postgres]",
                "config_options": {
                    "host": "Database host (required)",
                    "port": "Database port (default: 5432)",
                    "database": "Database name (required)",
                    "user": "Username (required)",
                    "password": "Password (required)",
                    "table": "Table name (default: records)"
                }
            },
            "elasticsearch": {
                "description": "Elasticsearch search engine backend",
                "persistent": True,
                "requires_install": "pip install dataknobs-data[elasticsearch]",
                "config_options": {
                    "hosts": "List of host URLs (required)",
                    "index": "Index name (required)",
                    "doc_type": "Document type (default: _doc)",
                    "username": "Optional username",
                    "password": "Optional password"
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
            },
            "faiss": {
                "description": "Facebook AI Similarity Search vector store",
                "persistent": False,
                "requires_install": "pip install faiss-cpu",
                "config_options": {
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                    "index_type": "Index type: flat, ivfflat, hnsw (default: auto)",
                    "nlist": "Number of clusters for IVF index",
                    "m": "Number of connections for HNSW"
                }
            },
            "chroma": {
                "description": "ChromaDB vector database",
                "persistent": True,
                "requires_install": "pip install chromadb",
                "config_options": {
                    "collection_name": "Collection name (default: vectors)",
                    "persist_directory": "Directory for persistence (optional)",
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)"
                }
            },
            "memory_vector": {
                "description": "In-memory vector store for testing",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "dimensions": "Vector dimensions (required)",
                    "metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)"
                }
            }
        }
        
        # Note about future vector support for database backends
        if backend_type.lower() in ["postgres", "elasticsearch"]:
            base_info = info.get(backend_type.lower(), {})
            base_info["vector_support_planned"] = True
            base_info["vector_note"] = (
                "Vector support for this backend is planned but not yet implemented. "
                "Use dedicated vector stores (faiss, chroma, memory_vector) for now."
            )
            return base_info

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

        else:
            raise ValueError(
                f"Backend '{backend_type}' does not support async operations yet. "
                f"Available async backends: memory, file, postgres, elasticsearch, s3"
            )


# Create singleton instances for registration
database_factory = DatabaseFactory()
async_database_factory = AsyncDatabaseFactory()
