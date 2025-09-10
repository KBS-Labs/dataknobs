"""FSM Core Exceptions Module.

This module defines core exception types used throughout the FSM system.
"""

from typing import Any, Dict


class FSMError(Exception):
    """Base exception for all FSM-related errors."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class InvalidConfigurationError(FSMError):
    """Raised when FSM configuration is invalid."""
    pass


class StateExecutionError(FSMError):
    """Raised when state execution fails."""
    
    def __init__(self, state_name: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(f"State '{state_name}' execution failed: {message}", details)
        self.state_name = state_name


class TransitionError(FSMError):
    """Raised when state transition fails."""
    
    def __init__(self, from_state: str, to_state: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(f"Transition from '{from_state}' to '{to_state}' failed: {message}", details)
        self.from_state = from_state
        self.to_state = to_state


class ResourceError(FSMError):
    """Raised when resource operations fail."""
    
    def __init__(self, resource_id: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(f"Resource '{resource_id}' error: {message}", details)
        self.resource_id = resource_id


class ValidationError(FSMError):
    """Raised when data validation fails."""
    pass


class TimeoutError(FSMError):
    """Raised when operation times out."""
    pass


class ConcurrencyError(FSMError):
    """Raised when concurrent execution fails."""
    pass
