"""Base interfaces and classes for FSM functions.

This module defines the interfaces for:
- Validation functions (check data validity)
- Transform functions (modify data)
- State test functions (determine next state)
- End state test functions (check if processing should end)
- Resources (external systems and services)
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, TypeVar

from dataknobs_common.exceptions import (
    ConfigurationError as BaseConfigurationError,
    DataknobsError,
    OperationError,
    ResourceError as BaseResourceError,
    ValidationError as BaseValidationError,
)
from dataknobs_common.structured_config import StructuredConfig

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
    """Context passed to functions during execution.

    Resources a state or arc declares are injected into ``resources`` keyed by
    resource **name**. When the declaration is role-based (an arc's
    ``{role: name}`` map), that map is also exposed via
    ``metadata['resource_roles']`` so a role-bound function reusable across arcs
    can resolve its logical role. The :meth:`require_resource` /
    :meth:`resource_for_role` accessors cover both models without hand-rolling
    dict plumbing.
    """
    state_name: str
    function_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)  # Shared variables
    network_name: str | None = None  # Current network for scoping

    def require_resource(self, name: str) -> Any:
        """Return the injected resource named ``name`` or raise.

        Resources are injected by the engine from the state's or arc's
        ``resources`` declaration — never smuggled through the data payload. A
        missing resource is a wiring error (the resource was not declared, or no
        provider is registered for it). This is the single error contract shared
        by the database function library, so the message is identical whether
        the lookup happens from a library function, an arc function, or a state
        function.

        Args:
            name: The declared resource name.

        Returns:
            The injected resource.

        Raises:
            TransformError: If no resource is registered under ``name``.
        """
        resource = self.resources.get(name)
        if resource is None:
            raise TransformError(
                f"Resource '{name}' not found in context.resources "
                f"(is it declared in the state's or arc's 'resources'?)"
            )
        return resource

    def resource_for_role(self, role: str) -> Any:
        """Resolve a logical ``role`` to its bound resource.

        Uses the arc's ``{role: name}`` map (exposed via
        ``metadata['resource_roles']``) to resolve the role to a resource name,
        then returns the injected resource under that name. This lets a single
        function be reused across arcs that bind the same role to different
        concrete resources.

        Args:
            role: The logical role name (e.g. ``"database"``).

        Returns:
            The injected resource bound to ``role``.

        Raises:
            TransformError: If the role is not bound, or the bound resource is
                not injected.
        """
        roles = self.metadata.get("resource_roles") or {}
        name = roles.get(role)
        if name is None:
            raise TransformError(
                f"Role '{role}' is not bound to a resource "
                f"(declare it in the arc's 'resources', e.g. {{'{role}': '<name>'}})"
            )
        return self.require_resource(name)


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
    def transform(
        self,
        data: Any,
        context: "FunctionContext | Dict[str, Any] | None" = None,
    ) -> ExecutionResult:
        """Transform data according to function logic.

        Args:
            data: The data to transform.
            context: Optional execution context. The FSM engines always pass a
                ``FunctionContext`` (carrying injected ``resources`` and the
                ``resource_roles`` map); a plain ``dict`` is accepted for
                lightweight/standalone invocation.
            
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
    def test(
        self,
        data: Any,
        context: "FunctionContext | Dict[str, Any] | None" = None,
    ) -> Tuple[bool, str | None]:
        """Test if a condition is met for state transition.

        Args:
            data: The data to test.
            context: Optional execution context. The FSM engines always pass a
                ``FunctionContext`` (carrying injected ``resources`` and the
                ``resource_roles`` map); a plain ``dict`` is accepted for
                lightweight/standalone invocation.
            
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


def as_state_test_callable(func: Any) -> Any:
    """Return the callable form of a resolved arc-condition / pre-test function.

    A bare ``IStateTestFunction`` instance carries its condition logic on
    ``.test(data, context) -> (passed, reason)`` and is not itself callable, so
    the execution engines (which invoke every pre-test uniformly as
    ``func(data, context)``) cannot dispatch it. Return the bound ``.test``
    method for a bare interface instance; every already-callable form — plain
    predicates, :class:`FunctionWrapper`/``InterfaceWrapper``, and the config
    builder's resolved adapters — passes through unchanged.

    This mirrors ``FunctionWrapper._normalize_interface_callable`` for the one
    path that bypasses it: the async engine's ``custom_functions`` merge
    (``AsyncExecutionEngine._get_merged_functions``) stores engine-injected
    functions raw. It is deliberately scoped to ``IStateTestFunction`` only —
    the transform path has its own deterministic ``ITransformFunction`` dispatch
    (``_is_interface_transform``/``_invoke_state_transform``), so normalizing
    other interfaces here would convert a bare transform instance into a bound
    method and silently bypass that resource-injecting dispatch.

    A bare ``IValidationFunction`` used directly as an arc condition is likewise
    not normalized here. That shape is unusual — validators belong on a state's
    ``(pre_)validators``, where the manager build path normalizes all four
    interfaces — so an interface-as-condition reference is expected to be an
    ``IStateTestFunction``. A bare validator instance reaching this path stays
    non-callable and surfaces as a record error rather than being silently
    reinterpreted as a condition.
    """
    if isinstance(func, IStateTestFunction):
        return func.test
    return func


class ResourceStatus(Enum):
    """Status of a resource."""
    
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ResourceConfig(StructuredConfig):
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
#
# These predate the migration of the package's exceptions onto the shared
# `dataknobs_common` hierarchy and were left behind by it, so for a while
# they formed a second hierarchy rooted at a plain `Exception` that reused
# four names `dataknobs_fsm.core.exceptions` also defines as unrelated
# types. Each is now *also* the common type that describes what happened,
# which does two things: `except DataknobsError` reaches them, and anything
# that classifies an exception rather than just reporting it -- retry logic
# keyed on a base, an HTTP boundary mapping types onto statuses -- reads the
# same answer here as it does everywhere else in dataknobs.
#
# `FSMError` is kept as their common base so no existing `except FSMError`
# clause catches less than it did.


def _warn_deprecated(name: str, guidance: str) -> None:
    """Emit the notice for a legacy name that nothing in the package raises."""
    warnings.warn(
        f"dataknobs_fsm.functions.base.{name} is deprecated and is raised "
        f"nowhere in this package; {guidance}",
        DeprecationWarning,
        stacklevel=3,
    )


class FSMError(DataknobsError):
    """Base exception for the errors raised by the functions layer.

    Deprecated as a name to raise or to catch on. It duplicates
    :data:`dataknobs_fsm.core.exceptions.FSMError`, which is an alias of
    ``DataknobsError`` and so means something considerably broader, and
    nothing raises this one directly. It remains the base of the types below
    purely so existing ``except FSMError`` clauses are unaffected; new code
    should catch ``DataknobsError``, which now reaches these too, or the
    specific common type for the condition it handles.
    """

    def __init__(self, message: str, *args: Any, **kwargs: Any):
        if type(self) is FSMError:
            _warn_deprecated(
                "FSMError",
                "catch dataknobs_common.DataknobsError instead, which now "
                "reaches every error this package raises.",
            )
        super().__init__(message, *args, **kwargs)


class ValidationError(BaseValidationError, FSMError):
    """Raised when validation fails.

    Also a :class:`dataknobs_common.exceptions.ValidationError`: the
    condition is that data the caller supplied did not validate, which is
    what that type describes and how a caller should render it.
    """

    def __init__(self, message: str, validation_errors: List[str] | None = None):
        """Initialize validation error.

        Args:
            message: Error message.
            validation_errors: List of specific validation errors.
        """
        super().__init__(
            message,
            context={"validation_errors": list(validation_errors)}
            if validation_errors
            else None,
        )
        self.validation_errors = validation_errors or []


class TransformError(OperationError, FSMError):
    """Raised when transformation fails.

    An :class:`dataknobs_common.exceptions.OperationError`: a transform that
    fails is a failed operation, and unlike a validation failure it is not
    the caller's input that is at fault.
    """
    pass


class StateTransitionError(OperationError, FSMError):
    """Raised when state transition fails.

    Deprecated, and raised nowhere in this package. Its alias below,
    ``FunctionError``, is the reason to prefer the ``core.exceptions``
    types: that name means a failed *transition* here and a failed
    *function* there.
    """

    def __init__(self, message: str, from_state: str, to_state: str | None = None):
        """Initialize state transition error.

        Args:
            message: Error message.
            from_state: The state transitioning from.
            to_state: The state attempting to transition to.
        """
        if type(self) is StateTransitionError:
            _warn_deprecated(
                "StateTransitionError (also exported as FunctionError)",
                "use dataknobs_fsm.core.exceptions.TransitionError for a "
                "failed transition, or dataknobs_fsm.core.exceptions."
                "FunctionError for a failed function -- the FunctionError "
                "alias here conflates the two.",
            )
        super().__init__(
            message, context={"from_state": from_state, "to_state": to_state}
        )
        self.from_state = from_state
        self.to_state = to_state


class ResourceError(BaseResourceError, FSMError):
    """Raised when resource operations fail.

    Also a :class:`dataknobs_common.exceptions.ResourceError`, which is what
    a caller reads to tell "the deployment could not reach something" apart
    from "the request was wrong". Note that the message may carry
    infrastructure detail -- a connection string from a failed connect --
    so a caller rendering this to an untrusted client should mask it.
    """

    def __init__(self, message: str, resource_name: str, operation: str):
        """Initialize resource error.

        Args:
            message: Error message.
            resource_name: Name of the resource.
            operation: The operation that failed.
        """
        super().__init__(
            message,
            context={"resource_name": resource_name, "operation": operation},
        )
        self.resource_name = resource_name
        self.operation = operation


class ConfigurationError(BaseConfigurationError, FSMError):
    """Raised when configuration is invalid.

    Deprecated, and raised nowhere in this package -- every ``raise
    ConfigurationError`` in it already uses the ``dataknobs_common`` type
    this now extends.
    """

    def __init__(self, message: str, *args: Any, **kwargs: Any):
        if type(self) is ConfigurationError:
            _warn_deprecated(
                "ConfigurationError",
                "use dataknobs_common.ConfigurationError, which this now "
                "extends and which every raise site in this package already "
                "uses.",
            )
        super().__init__(message, *args, **kwargs)


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


# Alias FunctionError to StateTransitionError for compatibility.
#
# Deprecated along with what it points at, and the sharpest reason to prefer
# `core.exceptions`: that module also exports a `FunctionError`, but it is an
# `OperationError` about a user-supplied function failing, not a transition.
# Same name, two conditions, depending on which module you imported from.
FunctionError = StateTransitionError
