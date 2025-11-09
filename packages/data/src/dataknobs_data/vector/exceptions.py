"""Vector-specific exceptions.

This module defines exception types for vector operations,
built on the common exception framework from dataknobs_common.
"""

from __future__ import annotations

from dataknobs_common import (
    DataknobsError,
    OperationError,
    ResourceError,
    ValidationError,
)

# Create VectorError as alias to DataknobsError for backward compatibility
VectorError = DataknobsError


class VectorDimensionError(ValidationError):
    """Raised when vector dimensions don't match expectations."""

    def __init__(self, expected: int, actual: int, field_name: str | None = None):
        """Initialize dimension error.

        Args:
            expected: Expected number of dimensions
            actual: Actual number of dimensions
            field_name: Optional field name for context
        """
        self.expected = expected
        self.actual = actual
        self.field_name = field_name

        message = f"Vector dimension mismatch: expected {expected}, got {actual}"
        if field_name:
            message = f"{message} for field '{field_name}'"

        context = {"expected": expected, "actual": actual}
        if field_name:
            context["field_name"] = field_name

        super().__init__(message, context=context)


class VectorBackendError(ResourceError):
    """Raised when vector backend operations fail."""

    pass


class VectorIndexError(OperationError):
    """Raised when vector index operations fail."""

    pass


class VectorNotSupportedError(OperationError):
    """Raised when vector operations are not supported by backend."""

    def __init__(self, backend: str, operation: str | None = None):
        """Initialize not supported error.

        Args:
            backend: Name of the backend
            operation: Optional specific operation that's not supported
        """
        self.backend = backend
        self.operation = operation

        message = f"Vector operations not supported by {backend} backend"
        if operation:
            message = f"{message}: {operation}"

        context = {"backend": backend}
        if operation:
            context["operation"] = operation

        super().__init__(message, context=context)


class VectorValidationError(ValidationError):
    """Raised when vector validation fails."""

    pass
