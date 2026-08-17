"""Database backend implementations.

Migration note (v0.5.0):
    The deprecated ``BACKEND_REGISTRY`` and ``SYNC_BACKEND_REGISTRY``
    backward-compatibility shims have been removed.  Use
    ``sync_backends`` and ``async_backends`` directly instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Type

from dataknobs_common.registry import PluginRegistry

from ..backend_selection import module_installed, register_backend
from ..database import AsyncDatabase, SyncDatabase

logger = logging.getLogger(__name__)


# --- Deferred loaders ------------------------------------------------
#
# Each imports its backend class on first call rather than at module
# scope. Registration used to be a ``try: import ... except ImportError:
# pass`` per backend, which conflated two questions: whether the driver is
# installed, and whether the module happens to raise on import. Backends
# answered the first differently depending on where they put their driver
# import, so ``registered`` meant "installed" for some and "importable"
# for others. Handing the import over as a thunk lets
# :func:`register_backend` ask the first question on its own terms.
#
# Spelled as an import *statement* rather than ``import_module(name)``
# because the module and the attribute are literals in this source, not a
# dotted path arriving from configuration: there is no separator to pick,
# no typo that a user could make, and nothing for
# ``dataknobs_common.imports`` to own. Written statically the class stays
# a symbol -- mypy checks it, a rename updates it, and a misspelling fails
# at authoring time rather than at registration.


def _sync_memory() -> type[SyncDatabase]:
    from .memory import SyncMemoryDatabase

    return SyncMemoryDatabase


def _sync_file() -> type[SyncDatabase]:
    from .file import SyncFileDatabase

    return SyncFileDatabase


def _sync_sqlite() -> type[SyncDatabase]:
    from .sqlite import SyncSQLiteDatabase

    return SyncSQLiteDatabase


def _sync_postgres() -> type[SyncDatabase]:
    from .postgres import SyncPostgresDatabase

    return SyncPostgresDatabase


def _sync_elasticsearch() -> type[SyncDatabase]:
    from .elasticsearch import SyncElasticsearchDatabase

    return SyncElasticsearchDatabase


def _sync_s3() -> type[SyncDatabase]:
    from .s3 import SyncS3Database

    return SyncS3Database


def _sync_duckdb() -> type[SyncDatabase]:
    from .duckdb import SyncDuckDBDatabase

    return SyncDuckDBDatabase


def _async_memory() -> type[AsyncDatabase]:
    from .memory import AsyncMemoryDatabase

    return AsyncMemoryDatabase


def _async_file() -> type[AsyncDatabase]:
    from .file import AsyncFileDatabase

    return AsyncFileDatabase


def _async_sqlite() -> type[AsyncDatabase]:
    from .sqlite_async import AsyncSQLiteDatabase

    return AsyncSQLiteDatabase


def _async_postgres() -> type[AsyncDatabase]:
    from .postgres import AsyncPostgresDatabase

    return AsyncPostgresDatabase


def _async_elasticsearch() -> type[AsyncDatabase]:
    from .elasticsearch_async import AsyncElasticsearchDatabase

    return AsyncElasticsearchDatabase


def _async_s3() -> type[AsyncDatabase]:
    from .s3_async import AsyncS3Database

    return AsyncS3Database


def _async_duckdb() -> type[AsyncDatabase]:
    from .duckdb import AsyncDuckDBDatabase

    return AsyncDuckDBDatabase


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
        _sync_memory,
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
        _sync_file,
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
        _sync_sqlite,
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
        _sync_postgres,
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
        _sync_elasticsearch,
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
        _sync_s3,
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
        _sync_duckdb,
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
        _async_memory,
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
        _async_file,
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
        _async_sqlite,
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
        _async_postgres,
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
        _async_elasticsearch,
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
        _async_s3,
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
        _async_duckdb,
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
