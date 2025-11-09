"""Custom exceptions for the LLM package.

This module defines exception types for the LLM package,
built on the common exception framework from dataknobs_common.
"""

from dataknobs_common import (
    DataknobsError,
    OperationError,
    ResourceError,
)

# Create LLMError as alias to DataknobsError for backward compatibility
LLMError = DataknobsError


class VersioningError(OperationError):
    """Base exception for versioning-related errors."""

    pass


class RateLimitError(OperationError):
    """Exception raised when rate limit is exceeded."""

    pass


class StorageError(ResourceError):
    """Exception raised for storage operation errors."""

    pass


class SchemaVersionError(OperationError):
    """Exception raised for schema version incompatibilities."""

    pass
