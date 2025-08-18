"""Example classes for dataknobs-config.

These classes demonstrate how to use the config system's object construction
features and are also used in tests.
"""

from .cache import Cache, CacheFactory, create_cache
from .database import Database, DatabaseFactory
from .services import ServiceManager, ServiceRegistry

__all__ = [
    "Cache",
    "CacheFactory",
    "Database",
    "DatabaseFactory",
    "ServiceManager",
    "ServiceRegistry",
    "create_cache",
]
