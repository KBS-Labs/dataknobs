"""Custom exceptions for the LLM package.

This module defines exception types for the LLM package,
built on the common exception framework from dataknobs_common.
"""

from dataknobs_common import (
    DataknobsError,
    OperationError,
    ResourceError,
)
from dataknobs_common.exceptions import RateLimitError, ValidationError

# Create LLMError as alias to DataknobsError for backward compatibility
LLMError = DataknobsError

__all__ = [
    "ContextLengthExceededError",
    "LLMError",
    "RateLimitError",
    "ResponseQueueExhaustedError",
    "SchemaVersionError",
    "StorageError",
    "ToolsNotSupportedError",
    "VersioningError",
]


class ContextLengthExceededError(ValidationError):
    """The request exceeded the model's input context window.

    A specialization of :class:`~dataknobs_common.exceptions.ValidationError`
    (context overflow is a 400 / invalid request), so existing
    ``except ValidationError`` handlers keep matching. Catch this narrower type
    to react to context overflow specifically — compact history and retry,
    switch to a larger-context model, or surface a distinct message. The
    original vendor SDK error is preserved on ``__cause__``.
    """

    pass


class VersioningError(OperationError):
    """Base exception for versioning-related errors."""

    pass


class StorageError(ResourceError):
    """Exception raised for storage operation errors."""

    pass


class SchemaVersionError(OperationError):
    """Exception raised for schema version incompatibilities."""

    pass


class ResponseQueueExhaustedError(OperationError):
    """EchoProvider response queue exhausted in strict mode.

    Raised when ``strict=True`` and a ``complete()`` call is made after
    all scripted responses have been consumed.  Indicates the test
    scripted fewer responses than the code actually needed.
    """

    def __init__(self, call_count: int) -> None:
        self.call_count = call_count
        super().__init__(
            f"EchoProvider response queue exhausted after {call_count} "
            f"call(s) (strict mode). The test scripted fewer responses "
            f"than the code requires. Either add more responses to the "
            f"queue or set strict=False to fall back to echo behavior."
        )


class ToolsNotSupportedError(OperationError):
    """Model does not support tool/function calling."""

    def __init__(self, model: str, suggestion: str = "") -> None:
        self.model = model
        self.suggestion = suggestion
        msg = f"Model '{model}' does not support tools."
        if suggestion:
            msg = f"{msg} {suggestion}"
        super().__init__(msg)
