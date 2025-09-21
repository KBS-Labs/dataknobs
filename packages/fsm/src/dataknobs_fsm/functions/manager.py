"""Unified function management for FSM.

This module provides a central, robust system for managing both sync and async functions
across all FSM components. It handles function registration, wrapping, resolution, and execution
in a consistent manner.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, Union, Protocol, runtime_checkable
from enum import Enum
import logging

from dataknobs_fsm.functions.base import (
    IValidationFunction,
    ITransformFunction,
    IStateTestFunction,
    IEndStateTestFunction,
    ExecutionResult
)

logger = logging.getLogger(__name__)


class FunctionSource(Enum):
    """Source of a function definition."""
    REGISTERED = "registered"  # Explicitly registered function
    INLINE = "inline"  # Inline code string
    BUILTIN = "builtin"  # Built-in FSM function
    REFERENCE = "reference"  # Reference to registered function


@runtime_checkable
class AsyncCallable(Protocol):
    """Protocol for async callable objects."""
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the async function."""
        ...


class FunctionWrapper:
    """Unified wrapper for all function types.

    This wrapper handles both sync and async functions uniformly,
    preserving their async nature and providing consistent interfaces.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        source: FunctionSource = FunctionSource.REGISTERED,
        interface: type | None = None
    ):
        """Initialize function wrapper.

        Args:
            func: The actual function (sync or async)
            name: Function name for identification
            source: Where the function came from
            interface: Optional interface the function should implement
        """
        self.func = func
        self.name = name
        self.source = source
        self.interface = interface

        # Determine if function is async
        self._is_async = self._check_async(func)

        # Store original function metadata
        self.__name__ = getattr(func, '__name__', name)
        self.__doc__ = getattr(func, '__doc__', '')

    def _check_async(self, func: Callable) -> bool:
        """Check if a function is async.

        Args:
            func: Function to check

        Returns:
            True if async, False otherwise
        """
        # Direct coroutine function check
        if asyncio.iscoroutinefunction(func):
            return True

        # Check for async __call__ method (for callable objects)
        # But not for regular functions which also have __call__
        if callable(func) and not inspect.isfunction(func) and not inspect.ismethod(func):
            # Check if the __call__ method itself is async
            try:
                if asyncio.iscoroutinefunction(func.__call__):  # type: ignore[operator]
                    return True
            except AttributeError:
                pass

        return False

    @property
    def is_async(self) -> bool:
        """Check if wrapped function is async."""
        return self._is_async

    async def execute_async(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function asynchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if self._is_async:
            # Direct async execution
            result = await self.func(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.func, *args, **kwargs)

        return result

    def execute_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function synchronously.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            RuntimeError: If trying to execute async function synchronously
        """
        if self._is_async:
            raise RuntimeError(
                f"Cannot execute async function '{self.name}' synchronously. "
                "Use execute_async instead."
            )

        return self.func(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function.

        This preserves the async nature of the wrapped function.
        """
        if self._is_async:
            # Return coroutine for async functions
            return self.execute_async(*args, **kwargs)
        else:
            # Direct call for sync functions
            return self.func(*args, **kwargs)

    # Make wrapper detectable as async when wrapping async functions
    def __getattr__(self, name):
        """Forward attribute access to wrapped function."""
        if name == '_is_coroutine' and self._is_async:
            # Mark as coroutine function for asyncio detection
            return asyncio.coroutines._is_coroutine
        return getattr(self.func, name)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FunctionWrapper(name={self.name}, "
            f"async={self._is_async}, source={self.source.value})"
        )


class InterfaceWrapper:
    """Wrapper that adapts functions to specific FSM interfaces."""

    def __init__(self, wrapper: FunctionWrapper, interface: type):
        """Initialize interface wrapper.

        Args:
            wrapper: The function wrapper
            interface: The interface to implement
        """
        self.wrapper = wrapper
        self.interface = interface
        self._setup_interface_methods()

    def _setup_interface_methods(self):
        """Set up methods based on interface."""
        if self.interface == ITransformFunction:
            self.transform = self._create_method('transform')
            self.get_transform_description = lambda: f"Transform: {self.wrapper.name}"

        elif self.interface == IValidationFunction:
            self.validate = self._create_method('validate')
            self.get_validation_rules = lambda: {"name": self.wrapper.name}

        elif self.interface == IStateTestFunction:
            self.test = self._create_test_method()
            self.get_test_description = lambda: f"Test: {self.wrapper.name}"

        elif self.interface == IEndStateTestFunction:
            self.should_end = self._create_test_method()
            self.get_end_condition = lambda: f"End test: {self.wrapper.name}"

    def _create_method(self, method_name: str):
        """Create an interface method that wraps the function.

        Args:
            method_name: Name of the interface method

        Returns:
            Method that calls the wrapped function
        """
        # Check if the function expects a single state argument (common for inline lambdas)
        import inspect
        func = self.wrapper.func
        try:
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            # If function takes only 1 param, it likely expects a state object
            expects_state_obj = param_count == 1
        except Exception:
            # Can't determine signature, assume standard (data, context)
            expects_state_obj = False

        if self.wrapper.is_async:
            async def async_method(data: Any, context: Dict[str, Any] | None = None) -> Any:
                if expects_state_obj:
                    # Wrap data for functions expecting state.data pattern
                    from dataknobs_fsm.core.data_wrapper import wrap_for_lambda
                    state_obj = wrap_for_lambda(data)
                    result = await self.wrapper.execute_async(state_obj)
                else:
                    result = await self.wrapper.execute_async(data, context)
                if method_name in ['validate', 'transform']:
                    # Wrap in ExecutionResult if needed
                    if not isinstance(result, ExecutionResult):
                        return ExecutionResult.success_result(result)
                return result
            return async_method
        else:
            def sync_method(data: Any, context: Dict[str, Any] | None = None) -> Any:
                if expects_state_obj:
                    # Wrap data for functions expecting state.data pattern
                    from dataknobs_fsm.core.data_wrapper import wrap_for_lambda
                    state_obj = wrap_for_lambda(data)
                    result = self.wrapper.execute_sync(state_obj)
                else:
                    result = self.wrapper.execute_sync(data, context)
                if method_name in ['validate', 'transform']:
                    # Wrap in ExecutionResult if needed
                    if not isinstance(result, ExecutionResult):
                        return ExecutionResult.success_result(result)
                return result
            return sync_method

    def _create_test_method(self):
        """Create a test method that returns (bool, reason)."""
        # Check if the function expects a single state argument (common for inline lambdas)
        import inspect
        func = self.wrapper.func
        try:
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            # If function takes only 1 param, it likely expects a state object
            expects_state_obj = param_count == 1
        except Exception:
            # Can't determine signature, assume standard (data, context)
            expects_state_obj = False

        if self.wrapper.is_async:
            async def async_test(data: Any, context: Dict[str, Any] | None = None):
                if expects_state_obj:
                    # Wrap data for functions expecting state.data pattern
                    from dataknobs_fsm.core.data_wrapper import wrap_for_lambda
                    state_obj = wrap_for_lambda(data)
                    result = await self.wrapper.execute_async(state_obj)
                else:
                    result = await self.wrapper.execute_async(data, context)
                if isinstance(result, tuple):
                    return result
                return (bool(result), None)
            return async_test
        else:
            def sync_test(data: Any, context: Dict[str, Any] | None = None):
                if expects_state_obj:
                    # Wrap data for functions expecting state.data pattern
                    from dataknobs_fsm.core.data_wrapper import wrap_for_lambda
                    state_obj = wrap_for_lambda(data)
                    result = self.wrapper.execute_sync(state_obj)
                else:
                    result = self.wrapper.execute_sync(data, context)
                if isinstance(result, tuple):
                    return result
                return (bool(result), None)
            return sync_test

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the wrapper callable."""
        return self.wrapper(*args, **kwargs)

    @property
    def is_async(self) -> bool:
        """Check if wrapped function is async."""
        return self.wrapper.is_async

    @property
    def __name__(self) -> str:
        """Get function name."""
        return self.wrapper.__name__

    @property
    def _is_async(self) -> bool:
        """Expose _is_async for detection."""
        return self.wrapper.is_async


class FunctionManager:
    """Central manager for all FSM functions.

    This class provides a unified interface for registering, resolving,
    and managing functions across the entire FSM system.
    """

    def __init__(self):
        """Initialize function manager."""
        self._functions: Dict[str, FunctionWrapper] = {}
        self._builtin_functions: Dict[str, FunctionWrapper] = {}
        self._inline_cache: Dict[str, FunctionWrapper] = {}

    def register_function(
        self,
        name: str,
        func: Callable,
        source: FunctionSource = FunctionSource.REGISTERED,
        interface: type | None = None
    ) -> FunctionWrapper:
        """Register a function.

        Args:
            name: Function name
            func: The function to register
            source: Source of the function
            interface: Optional interface to implement

        Returns:
            FunctionWrapper for the registered function
        """
        wrapper = FunctionWrapper(func, name, source, interface)

        if source == FunctionSource.BUILTIN:
            self._builtin_functions[name] = wrapper
        else:
            self._functions[name] = wrapper

        logger.debug(
            f"Registered {'async' if wrapper.is_async else 'sync'} "
            f"function '{name}' from {source.value}"
        )

        return wrapper

    def register_functions(
        self,
        functions: Dict[str, Callable],
        source: FunctionSource = FunctionSource.REGISTERED
    ) -> Dict[str, FunctionWrapper]:
        """Register multiple functions.

        Args:
            functions: Dictionary of name -> function
            source: Source of the functions

        Returns:
            Dictionary of name -> wrapper
        """
        wrappers = {}
        for name, func in functions.items():
            wrappers[name] = self.register_function(name, func, source)
        return wrappers

    def resolve_function(
        self,
        reference: Union[str, Dict[str, Any], Callable],
        interface: type | None = None
    ) -> Union[FunctionWrapper, InterfaceWrapper, None]:
        """Resolve a function reference to a wrapper.

        Args:
            reference: Function reference (name, dict, or callable)
            interface: Optional interface to adapt to

        Returns:
            FunctionWrapper or None if not found
        """
        wrapper = None

        if callable(reference):
            # Direct callable
            wrapper = FunctionWrapper(
                reference,
                getattr(reference, '__name__', 'anonymous'),
                FunctionSource.REGISTERED
            )

        elif isinstance(reference, str):
            # String reference - check registered functions first
            if reference in self._functions:
                wrapper = self._functions[reference]
            elif reference in self._builtin_functions:
                wrapper = self._builtin_functions[reference]
            else:
                # Treat as inline code
                wrapper = self._create_inline_wrapper(reference)

        elif isinstance(reference, dict):
            # Dictionary reference
            ref_type = reference.get('type', 'inline')

            if ref_type == 'registered':
                name = reference.get('name')
                if name:
                    wrapper = self._functions.get(name) or self._builtin_functions.get(name)

            elif ref_type == 'inline':
                code = reference.get('code')
                if code:
                    wrapper = self._create_inline_wrapper(code)

        # Apply interface if needed
        if wrapper and interface:
            return self._adapt_to_interface(wrapper, interface)

        return wrapper

    def _create_inline_wrapper(self, code: str) -> FunctionWrapper:
        """Create a wrapper for inline code.

        Args:
            code: Python code string

        Returns:
            FunctionWrapper for the inline code
        """
        # Check cache first
        if code in self._inline_cache:
            return self._inline_cache[code]

        # Compile and create function
        try:
            # Create a namespace for execution with registered functions
            namespace = {'asyncio': asyncio}

            # Add all registered functions to namespace so inline code can call them
            for name, wrapper in self._functions.items():
                # Add the actual function, not the wrapper
                namespace[name] = wrapper.func if hasattr(wrapper, 'func') else wrapper

            # First try to exec the code directly (might be a full function definition)
            try:
                # Store the initial set of names
                initial_names = set(namespace.keys())

                exec(code, namespace)

                # Find any newly defined function
                func = None
                new_names = set(namespace.keys()) - initial_names

                # Look through newly defined names for a callable
                for name in new_names:
                    if callable(namespace[name]):
                        func = namespace[name]
                        break
            except Exception:
                func = None

            if not func:
                # Check if it's a lambda expression
                if code.strip().startswith('lambda'):
                    # Evaluate lambda directly
                    func = eval(code, namespace)
                else:
                    # Treat as function body - check if it needs to be async
                    if 'await' in code:
                        # Create async wrapper
                        func_def = "async def inline_func(data, context=None):\n"
                    else:
                        # Create sync wrapper
                        func_def = "def inline_func(data, context=None):\n"

                    # Add the code as the function body
                    lines = code.split(';') if ';' in code else [code]

                    # Check if this looks like a simple expression (for conditions)
                    # Common patterns: comparisons, boolean ops, method calls that return bool
                    is_expression = (
                        '==' in code or '!=' in code or '<' in code or '>' in code or
                        ' and ' in code or ' or ' in code or ' not ' in code or
                        code.strip().startswith('not ') or
                        '.get(' in code or
                        'in ' in code or
                        code.strip() in ['True', 'False']
                    )

                    if is_expression and 'return' not in code and len(lines) == 1:
                        # For expressions, return the expression result
                        func_def += f"    return {code.strip()}\n"
                    else:
                        # For statements, add them as-is
                        for line in lines:
                            stmt = line.strip()
                            if stmt:
                                func_def += f"    {stmt}\n"

                        # Ensure we return data if no explicit return (for transforms)
                        if 'return' not in code:
                            func_def += "    return data\n"

                    exec(func_def, namespace)
                    func = namespace.get('inline_func')

            if func is not None and callable(func):
                wrapper = FunctionWrapper(func, f"inline_{id(code)}", FunctionSource.INLINE)
                self._inline_cache[code] = wrapper
                return wrapper
            else:
                # Failed to create function
                raise ValueError(f"Failed to create inline function from code: {code}")

        except Exception as e:
            logger.error(f"Failed to create inline function: {e}")
            # Return a no-op wrapper
            return FunctionWrapper(
                lambda data, context=None: data,  # noqa: ARG005
                f"inline_error_{id(code)}",
                FunctionSource.INLINE
            )

    def _adapt_to_interface(
        self,
        wrapper: FunctionWrapper,
        interface: type
    ) -> Union[InterfaceWrapper, FunctionWrapper]:
        """Adapt a wrapper to implement a specific interface.

        Args:
            wrapper: The function wrapper
            interface: The interface to implement

        Returns:
            InterfaceWrapper that implements the interface
        """
        return InterfaceWrapper(wrapper, interface)

    def get_function(self, name: str) -> FunctionWrapper | None:
        """Get a registered function by name.

        Args:
            name: Function name

        Returns:
            FunctionWrapper or None
        """
        return self._functions.get(name) or self._builtin_functions.get(name)

    def has_function(self, name: str) -> bool:
        """Check if a function is registered.

        Args:
            name: Function name

        Returns:
            True if registered
        """
        return name in self._functions or name in self._builtin_functions

    def list_functions(self) -> Dict[str, Dict[str, Any]]:
        """List all registered functions.

        Returns:
            Dictionary of function info
        """
        result = {}

        for name, wrapper in self._functions.items():
            result[name] = {
                'source': wrapper.source.value,
                'async': wrapper.is_async,
                'type': 'registered'
            }

        for name, wrapper in self._builtin_functions.items():
            result[name] = {
                'source': wrapper.source.value,
                'async': wrapper.is_async,
                'type': 'builtin'
            }

        return result

    def clear(self):
        """Clear all registered functions except builtins."""
        self._functions.clear()
        self._inline_cache.clear()

    def clear_all(self):
        """Clear all functions including builtins."""
        self.clear()
        self._builtin_functions.clear()


# Global function manager instance
_global_manager = FunctionManager()


def get_function_manager() -> FunctionManager:
    """Get the global function manager instance.

    Returns:
        The global FunctionManager
    """
    return _global_manager


def register_function(
    name: str,
    func: Callable,
    source: FunctionSource = FunctionSource.REGISTERED
) -> FunctionWrapper:
    """Register a function with the global manager.

    Args:
        name: Function name
        func: The function
        source: Function source

    Returns:
        FunctionWrapper
    """
    return _global_manager.register_function(name, func, source)


def resolve_function(
    reference: Union[str, Dict[str, Any], Callable],
    interface: type | None = None
) -> Union[FunctionWrapper, InterfaceWrapper, None]:
    """Resolve a function reference.

    Args:
        reference: Function reference
        interface: Optional interface

    Returns:
        FunctionWrapper or None
    """
    return _global_manager.resolve_function(reference, interface)
