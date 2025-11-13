"""Database backend implementations."""

from typing import Type

from dataknobs_common import Registry

from ..database import AsyncDatabase, SyncDatabase

# Import memory backends for backward compatibility
try:
    from .memory import AsyncMemoryDatabase, SyncMemoryDatabase
except ImportError:
    AsyncMemoryDatabase = None  # type: ignore
    SyncMemoryDatabase = None  # type: ignore


class BackendRegistry(Registry[Type[SyncDatabase]]):
    """Registry of available sync database backends.

    This registry manages sync database backend classes and their metadata.
    Backends are auto-registered on import if their dependencies are available.
    """

    def __init__(self) -> None:
        """Initialize the backend registry."""
        super().__init__("sync_backends", enable_metrics=True)
        self._register_builtin_backends()

    def _register_builtin_backends(self) -> None:
        """Auto-register all available built-in backends."""
        # Memory backend (always available)
        try:
            from .memory import SyncMemoryDatabase

            self.register(
                "memory",
                SyncMemoryDatabase,
                metadata={
                    "description": "In-memory storage for testing and caching",
                    "persistent": False,
                    "requires_install": False,
                    "config_options": {
                        "initial_data": "Optional initial data dictionary"
                    },
                },
            )
            self.register("mem", SyncMemoryDatabase)  # Alias
        except ImportError:
            pass

        # File backend
        try:
            from .file import SyncFileDatabase

            self.register(
                "file",
                SyncFileDatabase,
                metadata={
                    "description": "File-based storage (JSON, CSV, Parquet)",
                    "persistent": True,
                    "requires_install": False,
                    "vector_support": False,
                    "config_options": {
                        "path": "Path to the file (required)",
                        "format": "File format: json, csv, parquet (default: json)",
                        "compression": "Optional compression: gzip, bz2, xz",
                    },
                },
            )
        except ImportError:
            pass

        # SQLite backend
        try:
            from .sqlite import SyncSQLiteDatabase

            self.register(
                "sqlite",
                SyncSQLiteDatabase,
                metadata={
                    "description": "SQLite database backend with Python-based vector support",
                    "persistent": True,
                    "requires_install": False,
                    "vector_support": True,
                    "config_options": {
                        "path": "Path to database file (required)",
                        "table": "Table name (default: records)",
                        "vector_enabled": "Enable vector support (default: False)",
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("sqlite3", SyncSQLiteDatabase)  # Alias
        except ImportError:
            pass

        # PostgreSQL backend
        try:
            from .postgres import SyncPostgresDatabase

            self.register(
                "postgres",
                SyncPostgresDatabase,
                metadata={
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
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("postgresql", SyncPostgresDatabase)  # Alias
            self.register("pg", SyncPostgresDatabase)  # Alias
        except ImportError:
            pass

        # Elasticsearch backend
        try:
            from .elasticsearch import SyncElasticsearchDatabase

            self.register(
                "elasticsearch",
                SyncElasticsearchDatabase,
                metadata={
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
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("es", SyncElasticsearchDatabase)  # Alias
        except ImportError:
            pass

        # S3 backend
        try:
            from .s3 import SyncS3Database

            self.register(
                "s3",
                SyncS3Database,
                metadata={
                    "description": "AWS S3 object storage backend",
                    "persistent": True,
                    "requires_install": "pip install dataknobs-data[s3]",
                    "vector_support": False,
                    "config_options": {
                        "bucket": "S3 bucket name (required)",
                        "prefix": "Object key prefix (default: records/)",
                        "region": "AWS region (default: us-east-1)",
                        "endpoint_url": "Custom endpoint for S3-compatible services",
                        "access_key_id": "AWS access key (or use IAM role)",
                        "secret_access_key": "AWS secret key (or use IAM role)",
                    },
                },
            )
        except ImportError:
            pass

        # DuckDB backend
        try:
            from .duckdb import SyncDuckDBDatabase

            self.register(
                "duckdb",
                SyncDuckDBDatabase,
                metadata={
                    "description": "DuckDB database backend for analytical workloads with columnar storage",
                    "persistent": True,
                    "requires_install": "pip install duckdb",
                    "vector_support": False,
                    "config_options": {
                        "path": "Path to database file (required, use :memory: for in-memory)",
                        "table": "Table name (default: records)",
                        "timeout": "Connection timeout in seconds (default: 5.0)",
                        "read_only": "Open database in read-only mode (default: False)",
                    },
                },
            )
        except ImportError:
            pass


class AsyncBackendRegistry(Registry[Type[AsyncDatabase]]):
    """Registry of available async database backends.

    This registry manages async database backend classes and their metadata.
    Backends are auto-registered on import if their dependencies are available.
    """

    def __init__(self) -> None:
        """Initialize the async backend registry."""
        super().__init__("async_backends", enable_metrics=True)
        self._register_builtin_backends()

    def _register_builtin_backends(self) -> None:
        """Auto-register all available built-in async backends."""
        # Memory backend (always available)
        try:
            from .memory import AsyncMemoryDatabase

            self.register(
                "memory",
                AsyncMemoryDatabase,
                metadata={
                    "description": "In-memory storage for testing and caching",
                    "persistent": False,
                    "requires_install": False,
                    "config_options": {
                        "initial_data": "Optional initial data dictionary"
                    },
                },
            )
            self.register("mem", AsyncMemoryDatabase)  # Alias
        except ImportError:
            pass

        # File backend
        try:
            from .file import AsyncFileDatabase

            self.register(
                "file",
                AsyncFileDatabase,
                metadata={
                    "description": "File-based storage (JSON, CSV, Parquet)",
                    "persistent": True,
                    "requires_install": False,
                    "vector_support": False,
                    "config_options": {
                        "path": "Path to the file (required)",
                        "format": "File format: json, csv, parquet (default: json)",
                        "compression": "Optional compression: gzip, bz2, xz",
                    },
                },
            )
        except ImportError:
            pass

        # SQLite backend
        try:
            from .sqlite_async import AsyncSQLiteDatabase

            self.register(
                "sqlite",
                AsyncSQLiteDatabase,
                metadata={
                    "description": "SQLite database backend with Python-based vector support",
                    "persistent": True,
                    "requires_install": False,
                    "vector_support": True,
                    "config_options": {
                        "path": "Path to database file (required)",
                        "table": "Table name (default: records)",
                        "vector_enabled": "Enable vector support (default: False)",
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("sqlite3", AsyncSQLiteDatabase)  # Alias
        except ImportError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"AsyncSQLiteDatabase not available: {e}")

        # PostgreSQL backend
        try:
            from .postgres import AsyncPostgresDatabase

            self.register(
                "postgres",
                AsyncPostgresDatabase,
                metadata={
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
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("postgresql", AsyncPostgresDatabase)  # Alias
            self.register("pg", AsyncPostgresDatabase)  # Alias
        except ImportError:
            pass

        # Elasticsearch backend
        try:
            from .elasticsearch_async import AsyncElasticsearchDatabase

            self.register(
                "elasticsearch",
                AsyncElasticsearchDatabase,
                metadata={
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
                        "vector_metric": "Distance metric for vectors: cosine, euclidean, dot_product (default: cosine)",
                    },
                },
            )
            self.register("es", AsyncElasticsearchDatabase)  # Alias
        except ImportError:
            pass

        # S3 backend
        try:
            from .s3_async import AsyncS3Database

            self.register(
                "s3",
                AsyncS3Database,
                metadata={
                    "description": "AWS S3 object storage backend",
                    "persistent": True,
                    "requires_install": "pip install dataknobs-data[s3]",
                    "vector_support": False,
                    "config_options": {
                        "bucket": "S3 bucket name (required)",
                        "prefix": "Object key prefix (default: records/)",
                        "region": "AWS region (default: us-east-1)",
                        "endpoint_url": "Custom endpoint for S3-compatible services",
                        "access_key_id": "AWS access key (or use IAM role)",
                        "secret_access_key": "AWS secret key (or use IAM role)",
                    },
                },
            )
        except ImportError:
            pass

        # DuckDB backend
        try:
            from .duckdb import AsyncDuckDBDatabase

            self.register(
                "duckdb",
                AsyncDuckDBDatabase,
                metadata={
                    "description": "DuckDB database backend for analytical workloads with columnar storage",
                    "persistent": True,
                    "requires_install": "pip install duckdb",
                    "vector_support": False,
                    "config_options": {
                        "path": "Path to database file (required, use :memory: for in-memory)",
                        "table": "Table name (default: records)",
                        "timeout": "Connection timeout in seconds (default: 5.0)",
                        "max_workers": "Number of threads in pool (default: 4)",
                        "read_only": "Open database in read-only mode (default: False)",
                    },
                },
            )
        except ImportError:
            pass


# Create singleton instances
sync_backends = BackendRegistry()
async_backends = AsyncBackendRegistry()

# Backward compatibility: expose old dict-based API
BACKEND_REGISTRY = {key: async_backends.get(key) for key in async_backends.list_keys()}
SYNC_BACKEND_REGISTRY = {key: sync_backends.get(key) for key in sync_backends.list_keys()}


def register_backend(
    name: str,
    async_class: Type[AsyncDatabase] | None = None,
    sync_class: Type[SyncDatabase] | None = None,
) -> None:
    """Register a backend implementation.

    This function is maintained for backward compatibility.
    Prefer using sync_backends.register() or async_backends.register() directly.

    Args:
        name: Backend name
        async_class: Async database class
        sync_class: Sync database class
    """
    if async_class:
        async_backends.register(name, async_class, allow_overwrite=True)
    if sync_class:
        sync_backends.register(name, sync_class, allow_overwrite=True)


__all__ = [
    "BackendRegistry",
    "AsyncBackendRegistry",
    "sync_backends",
    "async_backends",
    "BACKEND_REGISTRY",  # Backward compatibility
    "SYNC_BACKEND_REGISTRY",  # Backward compatibility
    "register_backend",  # Backward compatibility
    "AsyncMemoryDatabase",  # Backward compatibility
    "SyncMemoryDatabase",  # Backward compatibility
]
