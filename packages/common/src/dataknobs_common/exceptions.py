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

from enum import StrEnum
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


class ConsentRequiredError(DataknobsError):
    """Raised when access to a consent-gated resource is refused.

    A fail-closed *policy denial*, not an operation failure: the caller has
    not granted the consent scope a resource requires, so the read or write is
    refused before it runs. A top-level sibling of :class:`ValidationError` /
    :class:`NotFoundError` (not an :class:`OperationError` — nothing failed;
    the operation was declined by policy).

    Attributes:
        scope: The consent scope that was required but not granted.
        user_id: The user the access was attempted for (optional; may be an
            opaque identifier — handle with the same care as the id itself).

    Example:
        ```python
        raise ConsentRequiredError(
            "Consent scope 'analytics' not granted",
            scope="analytics",
        )
        ```
    """

    def __init__(
        self,
        message: str = "Consent required",
        scope: str | None = None,
        user_id: str | None = None,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the consent-required error.

        Args:
            message: Error message.
            scope: The consent scope that was required but not granted.
            user_id: Optional user identifier the access was attempted for.
            context: Optional context dictionary (merged with ``scope`` /
                ``user_id``).
            details: Optional details dictionary (takes precedence per the
                base contract).
        """
        merged: Dict[str, Any] = {}
        if scope is not None:
            merged["scope"] = scope
        if user_id is not None:
            merged["user_id"] = user_id
        if context:
            merged.update(context)
        super().__init__(message, context=merged, details=details)
        self.scope = scope
        self.user_id = user_id


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


class RateLimitError(OperationError):
    """Raised when a rate limit is exceeded.

    Use this exception when an operation cannot proceed because a rate limit
    has been reached. Includes an optional ``retry_after`` hint indicating
    how many seconds the caller should wait before retrying.

    Attributes:
        retry_after: Optional number of seconds to wait before retrying.

    Example:
        ```python
        raise RateLimitError(
            "API rate limit exceeded",
            retry_after=2.5,
            context={"category": "api_write", "limit": 10, "interval": 60}
        )
        ```
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        context: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Error message.
            retry_after: Optional seconds to wait before retrying.
            context: Optional context dictionary.
            details: Optional details dictionary (merged with context).
        """
        super().__init__(message, context=context, details=details)
        self.retry_after = retry_after


class DottedPathReason(StrEnum):
    """The complete :attr:`DottedPathError.reason` vocabulary.

    Normalized in the constructor for the same reason
    ``PackResolutionReason`` is: a plain string stays acceptable, but an
    unrecognized one is a typo rather than a new vocabulary member.
    """

    #: The reference is not of the form ``module:name`` / ``module.name``.
    MALFORMED = "malformed"
    #: A module was not found — the target itself, an ancestor package of it,
    #: or something it imports at its top level. **An environment condition:**
    #: something is not installed. This is the reason a config key documented
    #: as ``optional`` may reasonably swallow.
    #:
    #: A missing *transitive* dependency lands here rather than in
    #: :attr:`IMPORT_FAILED` deliberately. A tool whose module imports an
    #: uninstalled SDK is exactly the optional-dependency case, and telling it
    #: apart from a mistyped path would need the deployment's intent, which
    #: this layer does not have.
    MODULE_NOT_FOUND = "module_not_found"
    #: The module was found, and **executing it raised** something other than
    #: a missing module. **A defect, not an environment condition:** the code
    #: is present and broken. Split from :attr:`MODULE_NOT_FOUND` so that a
    #: caller skipping absent optional dependencies does not also skip a
    #: module that is installed and raising — the two want opposite responses,
    #: and one is never safe to swallow silently.
    IMPORT_FAILED = "import_failed"
    #: The module imported; it has no such attribute.
    ATTRIBUTE_NOT_FOUND = "attribute_not_found"
    #: The attribute resolved and is not callable.
    NOT_CALLABLE = "not_callable"


class DottedPathError(ConfigurationError):
    """Raised when a dotted path from configuration cannot be resolved.

    Carries the offending ``ref`` and a machine-readable :attr:`reason`,
    following ``PackResolutionError``'s shape.

    Deliberately a **sibling** of :class:`DottedPathTypeError`, not its
    parent. The two mean different things to a caller: a resolution failure
    is transient and environmental (a module that is not installed, a typo
    in a path), and a config key documented as ``optional`` may reasonably
    swallow it; a shape mismatch means the path resolved to the wrong kind
    of object, which is never optional. Were the shape error a subclass, the
    obvious lenient handler::

        except DottedPathError:
            if optional:
                return None

    would swallow it too, and ``optional: true`` would silently grow to
    cover misfiled specs. As siblings, that handler cannot match a shape
    mismatch at all, so the distinction holds by construction rather than by
    remembering to order the ``except`` clauses.
    """

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        reason: DottedPathReason | str,
        **context: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Bounded description — see the module docstring of
                :mod:`dataknobs_common.imports` for why the underlying
                exception's text does not belong here.
            ref: The dotted path that failed, as written in configuration.
            reason: A :class:`DottedPathReason` member (or its value).
            **context: Extra context keys, merged into ``context``.
        """
        reason = DottedPathReason(reason)
        super().__init__(message, context={"ref": ref, "reason": reason, **context})
        self.ref = ref
        self.reason = reason


class DottedPathTypeError(ConfigurationError):
    """Raised when a dotted path resolves to an object of the wrong shape.

    The sibling of :class:`DottedPathError` — see that class for why the two
    are not related by inheritance. This one is **never** optional: a path
    that resolved successfully but named the wrong kind of object is a
    programmer error in the configuration's layout, and the only safe
    response is to surface it at config-load time.
    """

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        expected: type,
        **context: Any,
    ) -> None:
        """Initialize the error.

        Args:
            message: Bounded description of the mismatch.
            ref: The dotted path that resolved, as written in configuration.
            expected: The base class or protocol the target had to satisfy.
            **context: Extra context keys, merged into ``context``.
        """
        super().__init__(message, context={"ref": ref, "expected": expected, **context})
        self.ref = ref
        self.expected = expected


__all__ = [
    "DataknobsError",
    "ValidationError",
    "ConfigurationError",
    "DottedPathError",
    "DottedPathReason",
    "DottedPathTypeError",
    "ResourceError",
    "NotFoundError",
    "ConsentRequiredError",
    "OperationError",
    "ConcurrencyError",
    "SerializationError",
    "TimeoutError",
    "RateLimitError",
]
