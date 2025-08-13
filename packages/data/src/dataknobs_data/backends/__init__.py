"""Database backend implementations."""

from typing import Dict, Type

from ..database import Database, SyncDatabase

BACKEND_REGISTRY: Dict[str, Type[Database]] = {}
SYNC_BACKEND_REGISTRY: Dict[str, Type[SyncDatabase]] = {}


def register_backend(name: str, async_class: Type[Database] = None, sync_class: Type[SyncDatabase] = None):
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
    from .memory import MemoryDatabase, SyncMemoryDatabase
    register_backend("memory", MemoryDatabase, SyncMemoryDatabase)
except ImportError:
    pass

try:
    from .file import FileDatabase, SyncFileDatabase
    register_backend("file", FileDatabase, SyncFileDatabase)
except ImportError:
    pass

try:
    from .s3 import S3Database, SyncS3Database
    register_backend("s3", S3Database, SyncS3Database)
except ImportError:
    pass

try:
    from .postgres import PostgresDatabase, SyncPostgresDatabase
    register_backend("postgres", PostgresDatabase, SyncPostgresDatabase)
    register_backend("postgresql", PostgresDatabase, SyncPostgresDatabase)
except ImportError:
    pass

try:
    from .elasticsearch import ElasticsearchDatabase, SyncElasticsearchDatabase
    register_backend("elasticsearch", ElasticsearchDatabase, SyncElasticsearchDatabase)
    register_backend("es", ElasticsearchDatabase, SyncElasticsearchDatabase)
except ImportError:
    pass


__all__ = [
    "BACKEND_REGISTRY",
    "SYNC_BACKEND_REGISTRY",
    "register_backend",
]