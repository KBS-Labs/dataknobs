"""DataKnobs Data Package - Unified data abstraction layer.

The `dataknobs-data` package provides a unified interface for working with various
database backends, including SQLite, PostgreSQL, Elasticsearch, and S3. It offers
structured data storage, querying, validation, migration, and vector search capabilities.

Modules:
    database: Core Database classes (SyncDatabase, AsyncDatabase) providing the main API
    records: Record class for structured data with fields and metadata
    fields: Field types and definitions for data validation
    schema: Database schema definitions and field schemas
    query: Query building with filters, operators, and sorting
    query_logic: Complex boolean logic queries with AND/OR/NOT operators
    factory: Database factory functions for creating database instances
    streaming: Streaming operations for large-scale data processing
    validation: Data validation with schemas and constraints
    migration: Data migration tools for moving between backends
    exceptions: Custom exceptions for error handling

Quick Examples:

    Create and query a database:

    ```python
    from dataknobs_data import database_factory, Record, Query, Operator, Filter

    # Create an in-memory database
    db = database_factory("memory")

    # Add records
    db.add(Record({"name": "Alice", "age": 30}))
    db.add(Record({"name": "Bob", "age": 25}))

    # Query with filters
    query = Query(filters=[Filter("age", Operator.GT, 25)])
    results = db.search(query)
    print(results)  # [Record with Alice's data]
    ```

    Use schemas for validation:

    ```python
    from dataknobs_data import database_factory, Record, FieldType
    from dataknobs_data.schema import DatabaseSchema

    # Define schema
    schema = DatabaseSchema.create(
        name=FieldType.STRING,
        age=FieldType.INTEGER,
        email=FieldType.STRING
    )

    # Create database with schema
    db = database_factory("memory", config={"schema": schema})
    db.add(Record({"name": "Alice", "age": 30, "email": "alice@example.com"}))
    ```

    Stream large datasets:

    ```python
    from dataknobs_data import database_factory, StreamConfig

    db = database_factory("sqlite", config={"path": "large_data.db"})

    # Stream records in batches
    config = StreamConfig(batch_size=100)
    for batch in db.stream(config=config):
        process_batch(batch.records)
    ```

Design Philosophy:

    1. **Backend Agnostic** - Write once, deploy anywhere with multiple backend support
    2. **Type Safe** - Strong typing with schema validation and field type checking
    3. **Async Ready** - Full async/await support for high-performance applications
    4. **Composable** - Mix and match features like validation, migration, and vector search

Installation:

    ```bash
    pip install dataknobs-data
    ```

For detailed documentation, see the individual module docstrings and the online
documentation at https://docs.kbs-labs.com/dataknobs
"""

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
