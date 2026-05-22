"""Example classes for dataknobs-config.

These classes demonstrate how to use the config system's object construction
features and are also used in tests.
"""

from .cache import Cache, CacheFactory, create_cache
from .database import Database, DatabaseFactory
from .services import ServiceManager, ServiceRegistry
from .widgets import (
    AsyncWidget,
    AsyncWidgetFactory,
    PlainWidget,
    SyncWidget,
    WidgetConfig,
)

__all__ = [
    "AsyncWidget",
    "AsyncWidgetFactory",
    "Cache",
    "CacheFactory",
    "Database",
    "DatabaseFactory",
    "PlainWidget",
    "ServiceManager",
    "ServiceRegistry",
    "SyncWidget",
    "WidgetConfig",
    "create_cache",
]
