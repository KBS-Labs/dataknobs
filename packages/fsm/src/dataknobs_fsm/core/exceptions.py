"""FSM Core Exceptions Module.

This module defines core exception types used throughout the FSM system.
Now built on the common exception framework from dataknobs_common.
"""

from typing import Any, Dict

from dataknobs_common import (
    ConcurrencyError as BaseConcurrencyError,
    ConfigurationError,
    DataknobsError,
    OperationError,
    ResourceError as BaseResourceError,
    TimeoutError as BaseTimeoutError,
    ValidationError as BaseValidationError,
)

# Create FSMError as alias to DataknobsError for backward compatibility
FSMError = DataknobsError


class InvalidConfigurationError(ConfigurationError):
    """Raised when FSM configuration is invalid."""
    pass


class StateExecutionError(OperationError):
    """Raised when state execution fails."""

    def __init__(self, state_name: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(
            f"State '{state_name}' execution failed: {message}",
            context=details
        )
        self.state_name = state_name


class TransitionError(OperationError):
    """Raised when state transition fails."""

    def __init__(self, from_state: str, to_state: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(
            f"Transition from '{from_state}' to '{to_state}' failed: {message}",
            context=details
        )
        self.from_state = from_state
        self.to_state = to_state


class ResourceError(BaseResourceError):
    """FSM-specific resource error with resource_id tracking.

    Extends the common ResourceError with FSM-specific resource tracking.
    """

    def __init__(self, resource_id: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(
            f"Resource '{resource_id}' error: {message}",
            context=details
        )
        self.resource_id = resource_id


# Use common ValidationError directly (no FSM-specific behavior needed)
ValidationError = BaseValidationError


class FunctionError(OperationError):
    """Raised when a function execution fails.

    This represents deterministic failures in user-defined functions
    (transforms, validators, conditions) that should not be retried.
    These are code errors, not transient failures.
    """

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        details: Dict[str, Any] | None = None,
    ):
        if function_name:
            message = f"Function '{function_name}' failed: {message}"
        super().__init__(message, context=details)
        self.function_name = function_name
        self.from_state = from_state
        self.to_state = to_state


# Use common TimeoutError directly (no FSM-specific behavior needed)
TimeoutError = BaseTimeoutError


# Use common ConcurrencyError directly (no FSM-specific behavior needed)
ConcurrencyError = BaseConcurrencyError


class CircuitBreakerError(BaseConcurrencyError):
    """Raised when circuit breaker is open."""

    def __init__(self, wait_time: float | None = None, details: Dict[str, Any] | None = None):
        if wait_time:
            message = f"Circuit breaker is open (wait {wait_time:.1f}s)"
        else:
            message = "Circuit breaker is open"
        super().__init__(message, context=details)
        self.wait_time = wait_time


class ETLError(OperationError):
    """Raised when ETL operations fail."""
    pass


class BulkheadTimeoutError(BaseTimeoutError):
    """Raised when bulkhead queue times out."""
    pass
