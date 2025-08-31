"""Custom exceptions for the dataknobs_data package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataknobs_data.query import Query


class DataknobsDataError(Exception):
    """Base exception for all dataknobs_data errors."""

    pass


class RecordNotFoundError(DataknobsDataError):
    """Raised when a requested record is not found."""

    def __init__(self, id: str):
        self.id = id
        super().__init__(f"Record with ID '{id}' not found")


class RecordValidationError(DataknobsDataError):
    """Raised when record validation fails."""

    def __init__(self, message: str, field_name: str | None = None):
        self.field_name = field_name
        if field_name:
            message = f"Field '{field_name}': {message}"
        super().__init__(message)


class FieldTypeError(DataknobsDataError):
    """Raised when a field type operation fails."""

    def __init__(self, field_name: str, expected_type: str, actual_type: str):
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Field '{field_name}' type mismatch: expected {expected_type}, got {actual_type}"
        )


class DatabaseError(DataknobsDataError):
    """General database error."""

    pass


class DatabaseConnectionError(DataknobsDataError):
    """Raised when database connection fails."""

    def __init__(self, backend: str, message: str):
        self.backend = backend
        super().__init__(f"Failed to connect to {backend} backend: {message}")


class DatabaseOperationError(DataknobsDataError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, message: str):
        self.operation = operation
        super().__init__(f"Database operation '{operation}' failed: {message}")


class QueryError(DataknobsDataError):
    """Raised when query execution fails."""

    def __init__(self, message: str, query: Query | None = None):
        self.query = query
        super().__init__(f"Query error: {message}")


class SerializationError(DataknobsDataError):
    """Raised when serialization/deserialization fails."""

    def __init__(self, format: str, message: str):
        self.format = format
        super().__init__(f"Serialization error ({format}): {message}")


class DataFormatError(DataknobsDataError):
    """Raised when data format is invalid or unsupported."""

    def __init__(self, format: str, message: str):
        self.format = format
        super().__init__(f"Data format error ({format}): {message}")


class BackendNotFoundError(DataknobsDataError):
    """Raised when a requested backend is not available."""

    def __init__(self, backend: str, available: list | None = None):
        self.backend = backend
        self.available = available or []
        message = f"Backend '{backend}' not found"
        if self.available:
            message += f". Available backends: {', '.join(self.available)}"
        super().__init__(message)


class ConfigurationError(DataknobsDataError):
    """Raised when configuration is invalid."""

    def __init__(self, parameter: str, message: str):
        self.parameter = parameter
        super().__init__(f"Configuration error for '{parameter}': {message}")


class ConcurrencyError(DataknobsDataError):
    """Raised when a concurrency conflict occurs."""

    def __init__(self, message: str):
        super().__init__(f"Concurrency error: {message}")


class TransactionError(DataknobsDataError):
    """Raised when a transaction fails."""

    def __init__(self, message: str):
        super().__init__(f"Transaction error: {message}")


class MigrationError(DataknobsDataError):
    """Raised when data migration fails."""

    def __init__(self, source: str, target: str, message: str):
        self.source = source
        self.target = target
        super().__init__(f"Migration from {source} to {target} failed: {message}")
