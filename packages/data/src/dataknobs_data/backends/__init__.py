"""Database backend implementations.

Migration note (v0.5.0):
    The deprecated ``BACKEND_REGISTRY`` and ``SYNC_BACKEND_REGISTRY``
    backward-compatibility shims have been removed.  Use
    ``sync_backends`` and ``async_backends`` directly instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import import_module
from typing import Any, Type

from dataknobs_common.registry import PluginRegistry

from ..backend_selection import module_installed, register_backend
from ..database import AsyncDatabase, SyncDatabase

logger = logging.getLogger(__name__)


def _load(module: str, attr: str) -> Callable[[], Any]:
    """Defer ``from <module> import <attr>`` until the driver is known present.

    Registration used to be a ``try: import ... except ImportError: pass``
    per backend, which conflated two questions: whether the driver is
    installed, and whether the module happens to raise on import. Backends
    answered the first differently depending on where they put their driver
    import, so ``registered`` meant "installed" for some and "importable"
    for others. Handing the import over as a thunk lets
    :func:`register_backend` ask the first question on its own terms.
    """

    def load() -> Any:
        return getattr(import_module(module, __name__), attr)

    return load


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
    *,
    installed: Callable[[str], bool] = module_installed,
) -> None:
    """Register the built-in sync backends this installation can build.

    Args:
        registry: The registry to populate.
        installed: The "is this module importable?" predicate, forwarded to
            :func:`register_backend`. Injectable so a test can describe an
            environment the one it runs in is not.
    """
    register_backend(
        registry,
        "memory",
        _load(".memory", "SyncMemoryDatabase"),
        metadata={
            "description": "In-memory storage for testing and caching",
            "persistent": False,
            "requires_install": False,
            "config_options": {
                "initial_data": "Optional initial data dictionary",
            },
        },
        aliases=("mem",),
        installed=installed,
    )

    register_backend(
        registry,
        "file",
        _load(".file", "SyncFileDatabase"),
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
        installed=installed,
    )

    register_backend(
        registry,
        "sqlite",
        _load(".sqlite", "SyncSQLiteDatabase"),
        metadata={
            "description": "SQLite database backend with Python-based vector support",
            "persistent": True,
            # The sync variant is on stdlib ``sqlite3``; only the async one
            # needs a driver installed.
            "requires_install": False,
            "vector_support": True,
            "config_options": {
                "path": "Path to database file (required)",
                "table": "Table name (default: records)",
                "vector_enabled": "Enable vector support (default: False)",
                "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
            },
        },
        aliases=("sqlite3",),
        installed=installed,
    )

    register_backend(
        registry,
        "postgres",
        _load(".postgres", "SyncPostgresDatabase"),
        metadata={
            "description": "PostgreSQL database backend with native vector support (pgvector)",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[postgres]",
            # Both, because the module imports both at top level: the sync
            # class is unimportable without the async driver present.
            "requires_module": ("psycopg2", "asyncpg"),
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
        aliases=("postgresql", "pg"),
        installed=installed,
    )

    register_backend(
        registry,
        "elasticsearch",
        _load(".elasticsearch", "SyncElasticsearchDatabase"),
        metadata={
            "description": "Elasticsearch search engine backend with native KNN vector support",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[elasticsearch]",
            "requires_module": "elasticsearch",
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
        aliases=("es",),
        installed=installed,
    )

    register_backend(
        registry,
        "s3",
        _load(".s3", "SyncS3Database"),
        metadata={
            "description": "AWS S3 object storage backend",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[s3]",
            # boto3 is imported lazily inside the pool helper, so the module
            # imports cleanly without it and only fails on first use. That
            # is the shape that made "registered" mean nothing here.
            "requires_module": "boto3",
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
        installed=installed,
    )

    register_backend(
        registry,
        "duckdb",
        _load(".duckdb", "SyncDuckDBDatabase"),
        metadata={
            "description": "DuckDB database backend for analytical workloads with columnar storage",
            "persistent": True,
            "requires_install": "pip install duckdb",
            "requires_module": "duckdb",
            "vector_support": False,
            "config_options": {
                "path": "Path to database file (required, use :memory: for in-memory)",
                "table": "Table name (default: records)",
                "timeout": "Connection timeout in seconds (default: 5.0)",
                "read_only": "Open database in read-only mode (default: False)",
            },
        },
        installed=installed,
    )


# ------------------------------------------------------------------
# Async backend registry
# ------------------------------------------------------------------


def _register_async_backends(
    registry: PluginRegistry[Type[AsyncDatabase]],
    *,
    installed: Callable[[str], bool] = module_installed,
) -> None:
    """Register the built-in async backends this installation can build.

    Args:
        registry: The registry to populate.
        installed: The "is this module importable?" predicate, forwarded to
            :func:`register_backend`. Injectable so a test can describe an
            environment the one it runs in is not.
    """
    register_backend(
        registry,
        "memory",
        _load(".memory", "AsyncMemoryDatabase"),
        metadata={
            "description": "In-memory storage for testing and caching",
            "persistent": False,
            "requires_install": False,
            "config_options": {
                "initial_data": "Optional initial data dictionary",
            },
        },
        aliases=("mem",),
        installed=installed,
    )

    register_backend(
        registry,
        "file",
        _load(".file", "AsyncFileDatabase"),
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
        installed=installed,
    )

    register_backend(
        registry,
        "sqlite",
        _load(".sqlite_async", "AsyncSQLiteDatabase"),
        metadata={
            "description": "SQLite database backend with Python-based vector support",
            "persistent": True,
            # Unlike the sync variant, this one needs a driver: it is on
            # aiosqlite, which ships in the ``sqlite`` extra. Recorded as
            # False before, which is why it looked unconditional.
            "requires_install": "pip install dataknobs-data[sqlite]",
            "requires_module": "aiosqlite",
            "vector_support": True,
            "config_options": {
                "path": "Path to database file (required)",
                "table": "Table name (default: records)",
                "vector_enabled": "Enable vector support (default: False)",
                "vector_metric": "Distance metric: cosine, euclidean, dot_product (default: cosine)",
            },
        },
        aliases=("sqlite3",),
        installed=installed,
    )

    register_backend(
        registry,
        "postgres",
        _load(".postgres", "AsyncPostgresDatabase"),
        metadata={
            "description": "PostgreSQL database backend with native vector support (pgvector)",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[postgres]",
            "requires_module": ("psycopg2", "asyncpg"),
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
        aliases=("postgresql", "pg"),
        installed=installed,
    )

    register_backend(
        registry,
        "elasticsearch",
        _load(".elasticsearch_async", "AsyncElasticsearchDatabase"),
        metadata={
            "description": "Elasticsearch search engine backend with native KNN vector support",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[elasticsearch]",
            # Imported inside the pool helper rather than at module top
            # level, so this module too imports cleanly without its driver.
            "requires_module": "elasticsearch",
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
        aliases=("es",),
        installed=installed,
    )

    register_backend(
        registry,
        "s3",
        _load(".s3_async", "AsyncS3Database"),
        metadata={
            "description": "AWS S3 object storage backend",
            "persistent": True,
            "requires_install": "pip install dataknobs-data[s3]",
            "requires_module": "aioboto3",
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
        installed=installed,
    )

    register_backend(
        registry,
        "duckdb",
        _load(".duckdb", "AsyncDuckDBDatabase"),
        metadata={
            "description": "DuckDB database backend for analytical workloads with columnar storage",
            "persistent": True,
            "requires_install": "pip install duckdb",
            "requires_module": "duckdb",
            "vector_support": False,
            "config_options": {
                "path": "Path to database file (required, use :memory: for in-memory)",
                "table": "Table name (default: records)",
                "timeout": "Connection timeout in seconds (default: 5.0)",
                "max_workers": "Number of threads in pool (default: 4)",
                "read_only": "Open database in read-only mode (default: False)",
            },
        },
        installed=installed,
    )


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


__all__ = [
    "sync_backends",
    "async_backends",
    "AsyncMemoryDatabase",
    "SyncMemoryDatabase",
]
