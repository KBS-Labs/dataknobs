"""Base interfaces and classes for FSM functions.

This module defines the interfaces for:
- Validation functions (check data validity)
- Transform functions (modify data)
- State test functions (determine next state)
- End state test functions (check if processing should end)
- Resources (external systems and services)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, TypeVar
from enum import Enum
from dataclasses import dataclass, field

T = TypeVar("T")


class FunctionType(Enum):
    """Types of functions in the FSM."""
    
    VALIDATION = "validation"
    TRANSFORM = "transform"
    STATE_TEST = "state_test"
    END_STATE_TEST = "end_state_test"


class ExecutionResult:
    """Result of function execution."""
    
    def __init__(
        self,
        success: bool,
        data: Any | None = None,
        error: str | None = None,
        metadata: Dict[str, Any] | None = None
    ):
        """Initialize execution result.
        
        Args:
            success: Whether execution succeeded.
            data: Result data if successful.
            error: Error message if failed.
            metadata: Additional metadata about execution.
        """
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
    
    @classmethod
    def success_result(cls, data: Any, metadata: Dict[str, Any] | None = None) -> 'ExecutionResult':
        """Create a successful result.
        
        Args:
            data: The result data.
            metadata: Optional metadata.
            
        Returns:
            A successful ExecutionResult.
        """
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def failure_result(cls, error: str, metadata: Dict[str, Any] | None = None) -> 'ExecutionResult':
        """Create a failure result.

        Args:
            error: The error message.
            metadata: Optional metadata.

        Returns:
            A failed ExecutionResult.
        """
        return cls(success=False, error=error, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata
        }

    def __json__(self) -> Dict[str, Any]:
        """Support JSON serialization.

        Returns:
            Dictionary representation for JSON.
        """
        return self.to_dict()


@dataclass
class FunctionContext:
    """Context passed to functions during execution."""
    state_name: str
    function_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)  # Shared variables
    network_name: str | None = None  # Current network for scoping


class IValidationFunction(ABC):
    """Interface for validation functions."""
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] | None = None) -> ExecutionResult:
        """Validate data according to function logic.
        
        Args:
            data: The data to validate.
            context: Optional execution context.
            
        Returns:
            ExecutionResult with validation outcome.
        """
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules this function implements.
        
        Returns:
            Dictionary describing the validation rules.
        """
        pass


class ITransformFunction(ABC):
    """Interface for transform functions."""
    
    @abstractmethod
    def transform(self, data: Any, context: Dict[str, Any] | None = None) -> ExecutionResult:
        """Transform data according to function logic.
        
        Args:
            data: The data to transform.
            context: Optional execution context.
            
        Returns:
            ExecutionResult with transformed data.
        """
        pass
    
    @abstractmethod
    def get_transform_description(self) -> str:
        """Get a description of the transformation.
        
        Returns:
            String describing what this transform does.
        """
        pass


class IStateTestFunction(ABC):
    """Interface for state test functions."""
    
    @abstractmethod
    def test(self, data: Any, context: Dict[str, Any] | None = None) -> Tuple[bool, str | None]:
        """Test if a condition is met for state transition.
        
        Args:
            data: The data to test.
            context: Optional execution context.
            
        Returns:
            Tuple of (test_passed, reason).
        """
        pass
    
    @abstractmethod
    def get_test_description(self) -> str:
        """Get a description of what this test checks.
        
        Returns:
            String describing the test condition.
        """
        pass


class IEndStateTestFunction(ABC):
    """Interface for end state test functions."""
    
    @abstractmethod
    def should_end(self, data: Any, context: Dict[str, Any] | None = None) -> Tuple[bool, str | None]:
        """Test if processing should end.
        
        Args:
            data: The current data.
            context: Optional execution context.
            
        Returns:
            Tuple of (should_end, reason).
        """
        pass
    
    @abstractmethod
    def get_end_condition(self) -> str:
        """Get a description of the end condition.
        
        Returns:
            String describing when processing ends.
        """
        pass


class ResourceStatus(Enum):
    """Status of a resource."""
    
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ResourceConfig:
    """Configuration for a resource."""
    
    name: str
    type: str
    connection_params: Dict[str, Any]
    pool_size: int | None = None
    timeout: float | None = None
    retry_policy: Dict[str, Any] | None = None
    health_check_interval: float | None = None


class IResource(ABC):
    """Interface for external resources."""
    
    @abstractmethod
    async def initialize(self, config: ResourceConfig) -> None:
        """Initialize the resource.
        
        Args:
            config: Resource configuration.
        """
        pass
    
    @abstractmethod
    async def acquire(self, timeout: float | None = None) -> Any:
        """Acquire a connection/handle to the resource.
        
        Args:
            timeout: Optional timeout for acquisition.
            
        Returns:
            A resource handle/connection.
        """
        pass
    
    @abstractmethod
    async def release(self, handle: Any) -> None:
        """Release a resource handle/connection.
        
        Args:
            handle: The handle to release.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the resource is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the resource and cleanup."""
        pass
    
    @abstractmethod
    def get_status(self) -> ResourceStatus:
        """Get the current resource status.
        
        Returns:
            Current ResourceStatus.
        """
        pass


# Exception classes

class FSMError(Exception):
    """Base exception for FSM errors."""
    pass


class ValidationError(FSMError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, validation_errors: List[str] | None = None):
        """Initialize validation error.
        
        Args:
            message: Error message.
            validation_errors: List of specific validation errors.
        """
        super().__init__(message)
        self.validation_errors = validation_errors or []


class TransformError(FSMError):
    """Raised when transformation fails."""
    pass


class StateTransitionError(FSMError):
    """Raised when state transition fails."""
    
    def __init__(self, message: str, from_state: str, to_state: str | None = None):
        """Initialize state transition error.
        
        Args:
            message: Error message.
            from_state: The state transitioning from.
            to_state: The state attempting to transition to.
        """
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state


class ResourceError(FSMError):
    """Raised when resource operations fail."""
    
    def __init__(self, message: str, resource_name: str, operation: str):
        """Initialize resource error.
        
        Args:
            message: Error message.
            resource_name: Name of the resource.
            operation: The operation that failed.
        """
        super().__init__(message)
        self.resource_name = resource_name
        self.operation = operation


class ConfigurationError(FSMError):
    """Raised when configuration is invalid."""
    pass


# Base implementations

class BaseFunction:
    """Base class for functions with common functionality."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize base function.
        
        Args:
            name: Function name.
            description: Function description.
        """
        self.name = name
        self.description = description
        self.execution_count = 0
        self.error_count = 0
    
    def _record_execution(self, success: bool) -> None:
        """Record execution statistics.
        
        Args:
            success: Whether execution succeeded.
        """
        self.execution_count += 1
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics.
        
        Returns:
            Dictionary with execution stats.
        """
        return {
            "executions": self.execution_count,
            "errors": self.error_count,
            "success_rate": float(  # type: ignore
                (self.execution_count - self.error_count) / self.execution_count
                if self.execution_count > 0 else 0
            )
        }


class CompositeFunction(BaseFunction):
    """Base class for functions that compose multiple sub-functions."""
    
    def __init__(self, name: str, functions: List[BaseFunction], description: str = ""):
        """Initialize composite function.
        
        Args:
            name: Function name.
            functions: List of sub-functions to compose.
            description: Function description.
        """
        super().__init__(name, description)
        self.functions = functions
    
    def add_function(self, function: BaseFunction) -> None:
        """Add a function to the composite.
        
        Args:
            function: Function to add.
        """
        self.functions.append(function)
    
    def remove_function(self, function_name: str) -> bool:
        """Remove a function from the composite.
        
        Args:
            function_name: Name of function to remove.
            
        Returns:
            True if removed, False if not found.
        """
        for i, func in enumerate(self.functions):
            if func.name == function_name:
                self.functions.pop(i)
                return True
        return False


# Simple Function class for basic use
class Function(ABC):
    """Abstract base class for simple functions."""
    
    @abstractmethod
    def execute(self, data: Any, context: 'FunctionContext') -> Any:
        """Execute the function.
        
        Args:
            data: Input data.
            context: Function context.
            
        Returns:
            Function result.
        """
        pass


# FunctionRegistry for managing functions
class FunctionRegistry:
    """Registry for managing FSM functions."""
    
    def __init__(self):
        """Initialize function registry."""
        self.functions: Dict[str, Any] = {}
        self.validators: Dict[str, IValidationFunction] = {}
        self.transforms: Dict[str, ITransformFunction] = {}
    
    def register(self, name: str, function: Any) -> None:
        """Register a function.
        
        Args:
            name: Function name.
            function: Function instance.
        """
        if isinstance(function, Function):
            self.functions[name] = function
        elif isinstance(function, IValidationFunction):
            self.validators[name] = function
        elif isinstance(function, ITransformFunction):
            self.transforms[name] = function
        else:
            # Store as generic function
            self.functions[name] = function
    
    def get_function(self, name: str) -> Any | None:
        """Get a function by name.
        
        Args:
            name: Function name.
            
        Returns:
            Function instance or None.
        """
        # Check all registries
        if name in self.functions:
            return self.functions[name]
        elif name in self.validators:
            return self.validators[name]
        elif name in self.transforms:
            return self.transforms[name]
        return None
    
    def remove(self, name: str) -> bool:
        """Remove a function.
        
        Args:
            name: Function name.
            
        Returns:
            True if removed.
        """
        if name in self.functions:
            del self.functions[name]
            return True
        elif name in self.validators:
            del self.validators[name]
            return True
        elif name in self.transforms:
            del self.transforms[name]
            return True
        return False
    
    def list_functions(self) -> List[str]:
        """List all registered functions.
        
        Returns:
            List of function names.
        """
        all_names = []
        all_names.extend(self.functions.keys())
        all_names.extend(self.validators.keys())
        all_names.extend(self.transforms.keys())
        return sorted(all_names)
    
    def clear(self) -> None:
        """Clear all registered functions."""
        self.functions.clear()
        self.validators.clear()
        self.transforms.clear()


# Alias FunctionError to StateTransitionError for compatibility
FunctionError = StateTransitionError
