"""Custom exceptions for the LLM package.

This module defines exception types for the LLM package,
built on the common exception framework from dataknobs_common.
"""

from dataknobs_common import (
    DataknobsError,
    OperationError,
    ResourceError,
)
from dataknobs_common.exceptions import RateLimitError

# Create LLMError as alias to DataknobsError for backward compatibility
LLMError = DataknobsError

__all__ = [
    "LLMError",
    "RateLimitError",
    "SchemaVersionError",
    "StorageError",
    "VersioningError",
]


class VersioningError(OperationError):
    """Base exception for versioning-related errors."""

    pass


class StorageError(ResourceError):
    """Exception raised for storage operation errors."""

    pass


class SchemaVersionError(OperationError):
    """Exception raised for schema version incompatibilities."""

    pass
