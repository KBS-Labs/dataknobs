"""DataKnobs Data Package - Unified data abstraction layer."""

# Import validation and migration modules
from . import migration, validation
from .database import AsyncDatabase, SyncDatabase
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
from .factory import AsyncDatabaseFactory, DatabaseFactory, async_database_factory, database_factory

# Import core modules
from .fields import Field, FieldType, VectorField
from .query import Filter, Operator, Query, SortOrder, SortSpec
from .query_logic import (
    ComplexQuery,
    Condition,
    FilterCondition,
    LogicCondition,
    LogicOperator,
    QueryBuilder,
)
from .records import Record
from .streaming import StreamConfig, StreamProcessor, StreamResult

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "AsyncDatabase",
    "SyncDatabase",
    "Record",
    "Field",
    "FieldType",
    "VectorField",
    "Query",
    "Filter",
    "Operator",
    "SortOrder",
    "SortSpec",
    # Boolean logic
    "ComplexQuery",
    "QueryBuilder",
    "LogicOperator",
    "Condition",
    "FilterCondition",
    "LogicCondition",
    # Streaming
    "StreamConfig",
    "StreamResult",
    "StreamProcessor",
    # Factory
    "DatabaseFactory",
    "AsyncDatabaseFactory",
    "database_factory",
    "async_database_factory",
    # Validation and Migration modules
    "validation",
    "migration",
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
