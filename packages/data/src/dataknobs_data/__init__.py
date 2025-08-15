"""DataKnobs Data Package - Unified data abstraction layer."""

from .database import Database, SyncDatabase
from .exceptions import (
    BackendNotFoundError,
    ConcurrencyError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
    DataknobsDataError,
    FieldTypeError,
    MigrationError,
    QueryError,
    RecordNotFoundError,
    RecordValidationError,
    SerializationError,
    TransactionError,
)
from .factory import DatabaseFactory, database_factory, async_database_factory

# Import core modules
from .fields import Field, FieldType
from .query import Filter, Operator, Query, SortOrder, SortSpec
from .records import Record
from .streaming import StreamConfig, StreamResult

# Import v2 modules as the primary validation and migration modules
from . import validation_v2 as validation
from . import migration_v2 as migration

# Also make v2 modules available for backwards compatibility during transition
from . import validation_v2
from . import migration_v2

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Database",
    "SyncDatabase",
    "Record",
    "Field",
    "FieldType",
    "Query",
    "Filter",
    "Operator",
    "SortOrder",
    "SortSpec",
    # Streaming
    "StreamConfig",
    "StreamResult",
    # Factory
    "DatabaseFactory",
    "database_factory",
    "async_database_factory",
    # Validation and Migration modules
    "validation",
    "migration",
    "validation_v2",
    "migration_v2",
    # Exceptions
    "DataknobsDataError",
    "RecordNotFoundError",
    "RecordValidationError",
    "FieldTypeError",
    "DatabaseConnectionError",
    "DatabaseOperationError",
    "QueryError",
    "SerializationError",
    "BackendNotFoundError",
    "ConfigurationError",
    "ConcurrencyError",
    "TransactionError",
    "MigrationError",
]
