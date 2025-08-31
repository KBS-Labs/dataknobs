"""Vector-specific exceptions."""

from __future__ import annotations


class VectorError(Exception):
    """Base exception for vector operations."""

    pass


class VectorDimensionError(VectorError):
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

        super().__init__(message)


class VectorBackendError(VectorError):
    """Raised when vector backend operations fail."""

    pass


class VectorIndexError(VectorError):
    """Raised when vector index operations fail."""

    pass


class VectorNotSupportedError(VectorError):
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

        super().__init__(message)


class VectorValidationError(VectorError):
    """Raised when vector validation fails."""

    pass
