"""Database backend implementations."""

from __future__ import annotations

import logging
from typing import Any, Type

from dataknobs_common.registry import PluginRegistry

from ..database import AsyncDatabase, SyncDatabase

logger = logging.getLogger(__name__)

# Import memory backends for backward compatibility
try:
    from .memory import AsyncMemoryDatabase, SyncMemoryDatabase
except ImportError:
    AsyncMemoryDatabase = None  # type: ignore[assignment,misc]
    SyncMemoryDatabase = None  # type: ignore[assignment,misc]


# ------------------------------------------------------------------
# Sync backend registry
# ------------------------------------------------------------------


def _register_sync_backends(
    registry: PluginRegistry[Type[SyncDatabase]],
) -> None:
    """Auto-register all available built-in sync backends."""
    # Memory backend (always available)
    try:
        from .memory import SyncMemoryDatabase

        registry.register(
            "memory",
            SyncMemoryDatabase,
            metadata={
                "description": "In-memory storage for testing and caching",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "initial_data": "Optional initial data dictionary",
                },
            },
        )
        registry.register("mem", SyncMemoryDatabase)  # Alias
    except ImportError:
        pass

    # File backend
    try:
        from .file import SyncFileDatabase

        registry.register(
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

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("sqlite3", SyncSQLiteDatabase)  # Alias
    except ImportError:
        pass

    # PostgreSQL backend
    try:
        from .postgres import SyncPostgresDatabase

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("postgresql", SyncPostgresDatabase)  # Alias
        registry.register("pg", SyncPostgresDatabase)  # Alias
    except ImportError:
        pass

    # Elasticsearch backend
    try:
        from .elasticsearch import SyncElasticsearchDatabase

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("es", SyncElasticsearchDatabase)  # Alias
    except ImportError:
        pass

    # S3 backend
    try:
        from .s3 import SyncS3Database

        registry.register(
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

        registry.register(
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


# ------------------------------------------------------------------
# Async backend registry
# ------------------------------------------------------------------


def _register_async_backends(
    registry: PluginRegistry[Type[AsyncDatabase]],
) -> None:
    """Auto-register all available built-in async backends."""
    # Memory backend (always available)
    try:
        from .memory import AsyncMemoryDatabase

        registry.register(
            "memory",
            AsyncMemoryDatabase,
            metadata={
                "description": "In-memory storage for testing and caching",
                "persistent": False,
                "requires_install": False,
                "config_options": {
                    "initial_data": "Optional initial data dictionary",
                },
            },
        )
        registry.register("mem", AsyncMemoryDatabase)  # Alias
    except ImportError:
        pass

    # File backend
    try:
        from .file import AsyncFileDatabase

        registry.register(
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

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("sqlite3", AsyncSQLiteDatabase)  # Alias
    except ImportError as e:
        logger.debug("AsyncSQLiteDatabase not available: %s", e)

    # PostgreSQL backend
    try:
        from .postgres import AsyncPostgresDatabase

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("postgresql", AsyncPostgresDatabase)  # Alias
        registry.register("pg", AsyncPostgresDatabase)  # Alias
    except ImportError:
        pass

    # Elasticsearch backend
    try:
        from .elasticsearch_async import AsyncElasticsearchDatabase

        registry.register(
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
                    "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
                },
            },
        )
        registry.register("es", AsyncElasticsearchDatabase)  # Alias
    except ImportError:
        pass

    # S3 backend
    try:
        from .s3_async import AsyncS3Database

        registry.register(
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

        registry.register(
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


# ------------------------------------------------------------------
# Singleton instances
# ------------------------------------------------------------------

sync_backends: PluginRegistry[Type[SyncDatabase]] = PluginRegistry(
    "sync_backends",
    canonicalize_keys=True,
    on_first_access=_register_sync_backends,
)

async_backends: PluginRegistry[Type[AsyncDatabase]] = PluginRegistry(
    "async_backends",
    canonicalize_keys=True,
    on_first_access=_register_async_backends,
)


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------

# Lazily populated on first access — avoids importing all backends at
# module load time.  Use sync_backends / async_backends directly instead.


def _build_compat_dict(
    registry: PluginRegistry[Any],
) -> dict[str, Any]:
    """Build a backward-compat dict from a PluginRegistry."""
    return {key: registry.get_factory(key) for key in registry.list_keys()}


class _LazyBackendDict(dict):  # type: ignore[type-arg]
    """Dict that populates itself on first access for backward compat."""

    def __init__(self, registry: PluginRegistry[Any]) -> None:
        super().__init__()
        self._registry = registry
        self._populated = False

    def _ensure_populated(self) -> None:
        if not self._populated:
            self._populated = True
            self.update(_build_compat_dict(self._registry))

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_populated()
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._ensure_populated()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure_populated()
        return super().__contains__(key)

    def __iter__(self):  # type: ignore[override]
        self._ensure_populated()
        return super().__iter__()

    def keys(self):  # type: ignore[override]
        self._ensure_populated()
        return super().keys()

    def values(self):  # type: ignore[override]
        self._ensure_populated()
        return super().values()

    def items(self):  # type: ignore[override]
        self._ensure_populated()
        return super().items()

    def __len__(self) -> int:
        self._ensure_populated()
        return super().__len__()


BACKEND_REGISTRY: dict[str, Type[AsyncDatabase]] = _LazyBackendDict(async_backends)  # type: ignore[assignment]
SYNC_BACKEND_REGISTRY: dict[str, Type[SyncDatabase]] = _LazyBackendDict(sync_backends)  # type: ignore[assignment]


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
        async_backends.register(name, async_class, override=True)
    if sync_class:
        sync_backends.register(name, sync_class, override=True)


# Keep BackendRegistry and AsyncBackendRegistry as aliases for type compat
BackendRegistry = PluginRegistry[Type[SyncDatabase]]
AsyncBackendRegistry = PluginRegistry[Type[AsyncDatabase]]

__all__ = [
    "BackendRegistry",
    "AsyncBackendRegistry",
    "sync_backends",
    "async_backends",
    "BACKEND_REGISTRY",
    "SYNC_BACKEND_REGISTRY",
    "register_backend",
    "AsyncMemoryDatabase",
    "SyncMemoryDatabase",
]
