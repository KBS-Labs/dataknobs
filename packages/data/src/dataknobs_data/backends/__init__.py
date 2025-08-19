"""Database backend implementations."""

from typing import Dict, Type

from ..database import AsyncDatabase, SyncDatabase

BACKEND_REGISTRY: dict[str, type[AsyncDatabase]] = {}
SYNC_BACKEND_REGISTRY: dict[str, type[SyncDatabase]] = {}


def register_backend(
    name: str, async_class: type[AsyncDatabase] = None, sync_class: type[SyncDatabase] = None
):
    """Register a backend implementation.

    Args:
        name: Backend name
        async_class: Async database class
        sync_class: Sync database class
    """
    if async_class:
        BACKEND_REGISTRY[name] = async_class
    if sync_class:
        SYNC_BACKEND_REGISTRY[name] = sync_class


# Import and register backends
try:
    from .memory import AsyncMemoryDatabase, SyncMemoryDatabase

    register_backend("memory", AsyncMemoryDatabase, SyncMemoryDatabase)
except ImportError:
    pass

try:
    from .file import AsyncFileDatabase, SyncFileDatabase

    register_backend("file", AsyncFileDatabase, SyncFileDatabase)
except ImportError:
    pass

try:
    from .s3 import SyncS3Database
    from .s3_async import AsyncS3Database

    register_backend("s3", AsyncS3Database, SyncS3Database)
except ImportError:
    pass

try:
    from .postgres import AsyncPostgresDatabase, SyncPostgresDatabase

    register_backend("postgres", AsyncPostgresDatabase, SyncPostgresDatabase)
    register_backend("postgresql", AsyncPostgresDatabase, SyncPostgresDatabase)
except ImportError:
    pass

try:
    from .elasticsearch import SyncElasticsearchDatabase
    from .elasticsearch_async import AsyncElasticsearchDatabase

    register_backend("elasticsearch", AsyncElasticsearchDatabase, SyncElasticsearchDatabase)
    register_backend("es", AsyncElasticsearchDatabase, SyncElasticsearchDatabase)
except ImportError:
    pass

try:
    from .sqlite import SyncSQLiteDatabase
    from .sqlite_async import AsyncSQLiteDatabase

    register_backend("sqlite", AsyncSQLiteDatabase, SyncSQLiteDatabase)
    register_backend("sqlite3", AsyncSQLiteDatabase, SyncSQLiteDatabase)
except ImportError:
    pass


__all__ = [
    "BACKEND_REGISTRY",
    "SYNC_BACKEND_REGISTRY",
    "register_backend",
]
