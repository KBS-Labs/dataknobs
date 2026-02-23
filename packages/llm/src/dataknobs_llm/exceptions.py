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
    "ToolsNotSupportedError",
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


class ToolsNotSupportedError(OperationError):
    """Model does not support tool/function calling."""

    def __init__(self, model: str, suggestion: str = "") -> None:
        self.model = model
        self.suggestion = suggestion
        msg = f"Model '{model}' does not support tools."
        if suggestion:
            msg = f"{msg} {suggestion}"
        super().__init__(msg)
