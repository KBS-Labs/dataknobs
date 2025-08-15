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

# Import v2 modules for easier access
from . import validation_v2
from . import migration_v2
from .fields import Field, FieldType
from .query import Filter, Operator, Query, SortOrder, SortSpec
from .records import Record

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
    # Factory
    "DatabaseFactory",
    "database_factory",
    "async_database_factory",
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
