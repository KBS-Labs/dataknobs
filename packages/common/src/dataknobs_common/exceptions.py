"""Common exception hierarchy for all dataknobs packages.

This module provides a unified exception framework that all dataknobs packages
can extend. It supports both simple exceptions and context-rich exceptions with
detailed error information.

The exception hierarchy supports:
- Simple error messages for straightforward cases
- Context dictionaries for rich error information
- Details dictionaries (FSM-style) for structured error data
- Package-specific extensions

Example:
    ```python
    from dataknobs_common.exceptions import ValidationError, NotFoundError

    # Simple exception
    raise ValidationError("Invalid email format")

    # Context-rich exception
    raise NotFoundError(
        "User not found",
        context={"user_id": "123", "attempted_at": "2024-11-08"}
    )

    # Catch any dataknobs error
    try:
        operation()
    except DataknobsError as e:
        logger.error(f"Error: {e}")
        if e.context:
            logger.error(f"Context: {e.context}")
    ```

Package-Specific Extensions:
    ```python
    from dataknobs_common.exceptions import DataknobsError

    class MyPackageError(DataknobsError):
        '''Base exception for mypackage.'''
        pass

    class SpecificError(MyPackageError):
        '''Specific error with custom context.'''
        def __init__(self, item_id: str, message: str):
            super().__init__(
                f"Item '{item_id}': {message}",
                context={"item_id": item_id}
            )
    ```
"""

from typing import Any, Dict


class DataknobsError(Exception):
    """Base exception for all dataknobs packages.

    This is the root exception that all dataknobs packages should extend.
    It supports optional context data for rich error information, making
    debugging and error handling more effective.

    Attributes:
        context: Dictionary containing contextual information about the error
        details: Alias for context (FSM-style compatibility)

    Args:
        message: Human-readable error message
        context: Optional dictionary with error context (field names, IDs, etc.)
        details: Alternative to context (both are supported for compatibility)

    Example:
        ```python
        error = DataknobsError(
            "Operation failed",
            context={"operation": "save", "item_id": "123"}
        )
        str(error)
        # 'Operation failed'
        error.context
        # {'operation': 'save', 'item_id': '123'}
        ```
    """

    def __init__(
        self,
        message: str,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ):
        """Initialize the exception with optional context.

        Args:
            message: Error message
            context: Optional context dictionary
            details: Optional details dictionary (merged with context)
        """
        super().__init__(message)
        # Support both context and details parameters
        # Details takes precedence if both are provided
        self.context = details or context or {}
        # Alias for FSM-style compatibility
        self.details = self.context


class ValidationError(DataknobsError):
    """Raised when validation fails.

    Use this exception when data or configuration fails validation checks.
    Common scenarios include:
    - Invalid input data
    - Schema validation failures
    - Constraint violations
    - Type mismatches

    Example:
        ```python
        raise ValidationError(
            "Email format invalid",
            context={"field": "email", "value": "not-an-email"}
        )
        ```
    """

    pass


class ConfigurationError(DataknobsError):
    """Raised when configuration is invalid or missing.

    Use this exception for configuration-related errors including:
    - Missing required configuration
    - Invalid configuration values
    - Configuration file not found
    - Circular references in configuration

    Example:
        ```python
        raise ConfigurationError(
            "Database configuration missing",
            context={"config_key": "database.primary", "available_keys": ["cache", "auth"]}
        )
        ```
    """

    pass


class ResourceError(DataknobsError):
    """Raised when resource operations fail.

    Use this exception for resource management failures including:
    - Resource acquisition failures
    - Connection errors
    - Resource pool exhaustion
    - Timeout errors

    Example:
        ```python
        raise ResourceError(
            "Failed to acquire database connection",
            context={"pool_size": 10, "active_connections": 10, "timeout": 30}
        )
        ```
    """

    pass


class NotFoundError(DataknobsError):
    """Raised when a requested item is not found.

    Use this exception when looking up items by ID, name, or key and they
    don't exist. Common scenarios include:
    - Record not found in database
    - Configuration key not found
    - File not found
    - Resource not registered

    Example:
        ```python
        raise NotFoundError(
            "Record not found",
            context={"record_id": "user-123", "table": "users"}
        )
        ```
    """

    pass


class OperationError(DataknobsError):
    """Raised when an operation fails.

    Use this exception for general operation failures that don't fit
    other categories. Common scenarios include:
    - Database operation failures
    - File I/O errors
    - Network operation failures
    - State transition errors

    Example:
        ```python
        raise OperationError(
            "Failed to save record",
            context={"operation": "update", "backend": "postgres", "error": "connection lost"}
        )
        ```
    """

    pass


class ConcurrencyError(DataknobsError):
    """Raised when concurrent operation conflicts occur.

    Use this exception for concurrency-related failures including:
    - Lock acquisition failures
    - Transaction conflicts
    - Race conditions
    - Optimistic locking failures

    Example:
        ```python
        raise ConcurrencyError(
            "Record modified by another process",
            context={"record_id": "123", "expected_version": 5, "actual_version": 6}
        )
        ```
    """

    pass


class SerializationError(DataknobsError):
    """Raised when serialization or deserialization fails.

    Use this exception for data format conversion errors including:
    - JSON encoding/decoding failures
    - Invalid data format
    - Schema mismatch
    - Type conversion errors

    Example:
        ```python
        raise SerializationError(
            "Cannot deserialize data",
            context={"format": "json", "field": "created_at", "value": "invalid-date"}
        )
        ```
    """

    pass


class TimeoutError(DataknobsError):
    """Raised when an operation times out.

    Use this exception when operations exceed their time limit including:
    - Connection timeouts
    - Query timeouts
    - Resource acquisition timeouts
    - Operation execution timeouts

    Example:
        ```python
        raise TimeoutError(
            "Database query timed out",
            context={"query": "SELECT * FROM large_table", "timeout_seconds": 30}
        )
        ```
    """

    pass


__all__ = [
    "DataknobsError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "NotFoundError",
    "OperationError",
    "ConcurrencyError",
    "SerializationError",
    "TimeoutError",
]
