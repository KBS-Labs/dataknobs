# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Unified function management for FSM.

This module provides a central, robust system for managing both sync and async functions
across all FSM components. It handles function registration, wrapping, resolution, and execution
in a consistent manner.
"""

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Dict, Tuple, Union, Protocol, cast, runtime_checkable
from enum import Enum
import logging

from dataknobs_common.callbacks import is_async_callable

from dataknobs_fsm.functions.base import (
    IValidationFunction,
    ITransformFunction,
    IStateTestFunction,
    IEndStateTestFunction,
    ExecutionResult,
    INTERFACE_METHODS,
    RegisteredFunction,
    accepts_context,
    interface_method_of,
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

    #: Set by ``FSMBuilder._resolve_function`` on a wrapper it has already
    #: adapted, so the builder's own "wrap if not wrapped" check does not wrap
    #: it a second time. Declared here rather than only assigned there, which
    #: is what made the assignment an ``attr-defined`` finding.
    _is_wrapped: bool = False

    def __init__(
        self,
        func: RegisteredFunction,
        name: str,
        source: FunctionSource = FunctionSource.REGISTERED,
        interface: type | None = None,
    ):
        """Initialize function wrapper.

        Args:
            func: The actual function (sync or async)
            name: Function name for identification
            source: Where the function came from
            interface: Optional interface the function should implement
        """
        # Normalize FSM function-interface *instances* (e.g. a DatabaseUpsert
        # ITransformFunction) to their bound interface method. The instance
        # itself is typically not callable and carries no async signal, so
        # wrapping it directly mis-detects async and cannot be invoked; the
        # real (possibly-async) implementation lives on the interface method.
        self.interface_method: str | None = interface_method_of(func)
        func = self._normalize_interface_callable(func)

        self.func = func
        self.name = name
        self.source = source
        self.interface = interface

        # What shape to call the target with, decided once here rather than at
        # each invocation site. A wrapper is routinely wrapped again --- the
        # config builder resolves a registered name to this object and then
        # adapts *it* to an interface --- and a wrapper's own ``__call__`` is
        # ``(*args, **kwargs)``, so re-reading the signature one layer out
        # answers about the wrapper instead of the implementation. Inheriting
        # keeps the answer attached to the function it describes.
        self.accepts_context: bool
        if isinstance(func, FunctionWrapper):
            self.interface_method = self.interface_method or func.interface_method
            self.accepts_context = func.accepts_context
        else:
            self.accepts_context = accepts_context(func, default=self.interface_method is not None)

        # Determine if function is async
        self._is_async = self._check_async(func)

        # Store original function metadata
        self.__name__ = getattr(func, "__name__", name)
        self.__doc__ = getattr(func, "__doc__", "")

        # A wrapper is not itself an ``async def``, so a caller asking
        # ``asyncio.iscoroutinefunction(wrapper)`` gets the wrong answer unless
        # the wrapper says otherwise --- and the FSM hands wrappers to callers
        # that ask exactly that. ``inspect.markcoroutinefunction`` is the public
        # way to say it (3.12+) and both detectors read it; the private
        # ``asyncio.coroutines._is_coroutine`` sentinel this replaces is
        # undeclared by typeshed and gone in CPython 3.14.
        if self._is_async:
            inspect.markcoroutinefunction(self)

    @staticmethod
    def _normalize_interface_callable(func: RegisteredFunction) -> Callable:
        """Return the bound interface method for an FSM function instance.

        An object implementing one of the FSM function interfaces
        (``ITransformFunction``/``IValidationFunction``/``IStateTestFunction``/
        ``IEndStateTestFunction``) carries its logic on the named interface
        method, not on ``__call__``. Target that bound method so async
        detection and invocation are correct. Plain callables pass through
        unchanged.

        Which method belongs to which interface is
        :data:`~dataknobs_fsm.functions.base.INTERFACE_METHODS`, read here and
        by the config builder's resolved adapter rather than spelled out twice.
        """
        method = interface_method_of(func)
        if method is None:
            return cast("Callable[..., Any]", func)
        return cast("Callable[..., Any]", getattr(func, method))

    def _check_async(self, func: Callable) -> bool:
        """Check if calling ``func`` produces an awaitable.

        Delegates to :func:`~dataknobs_common.callbacks.is_async_callable`
        rather than re-deriving the answer. The version this replaces was that
        function minus two of its cases, which is what an independently
        maintained copy of a shared judgement looks like after a while:

        - It could not unwrap a ``functools.partial`` around a callable
          *object*. ``iscoroutinefunction`` unwraps one around a *function*;
          around an object, ``partial.__call__`` is a C dispatcher, so asking
          about ``__call__`` answers about the wrong object. Binding arguments
          onto a stateful callable is ordinary, and the result was dispatched
          to ``run_in_executor``, where calling it built a coroutine nobody
          awaited.
        - It read a *class* with an ``async def __call__`` as async, because
          ``SomeClass.__call__`` is the plain unbound function. Calling a class
          runs ``type.__call__`` and returns an instance, never an awaitable.

        Args:
            func: Function to check

        Returns:
            True if async, False otherwise
        """
        return is_async_callable(func)

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
            loop = asyncio.get_running_loop()
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

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped function."""
        return getattr(self.func, name)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FunctionWrapper(name={self.name}, async={self._is_async}, source={self.source.value})"
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

    def _setup_interface_methods(self) -> None:
        """Set up methods based on interface."""
        if self.interface == ITransformFunction:
            self.transform = self._create_method("transform")
            self.get_transform_description = lambda: f"Transform: {self.wrapper.name}"

        elif self.interface == IValidationFunction:
            self.validate = self._create_method("validate")
            self.get_validation_rules = lambda: {"name": self.wrapper.name}

        elif self.interface == IStateTestFunction:
            self.test = self._create_test_method()
            self.get_test_description = lambda: f"Test: {self.wrapper.name}"

        elif self.interface == IEndStateTestFunction:
            self.should_end = self._create_test_method()
            self.get_end_condition = lambda: f"End test: {self.wrapper.name}"

    def _call_shape(self, data: Any, context: Any) -> Tuple[Any, ...]:
        """The positional arguments the wrapped function is invoked with.

        Three shapes, and the choice between them is not arity alone:

        * ``(data, context)`` --- the engine's own call shape, for anything
          that can receive the context.
        * ``(data,)`` --- an interface implementation written to the
          one-argument convention. ``ITransformFunction.transform`` declares
          ``data``, not ``state``, so a plain record is what it asked for; this
          is the judgement the config builder's resolved adapter has always
          made, and making it here is what stops the two doors into an FSM
          from disagreeing about the same object.
        * ``(wrap_for_lambda(data),)`` --- a one-argument *plain* callable,
          which is the documented inline form ``lambda state: state.data[...]``
          that :func:`~dataknobs_fsm.core.data_wrapper.wrap_for_lambda` exists
          to serve.

        The question is answered by :class:`FunctionWrapper`, once, against the
        implementation --- not re-introspected here. A registered function
        arrives wrapped twice (the builder resolves the name to a wrapper, then
        adapts *that* to an interface), and a wrapper's ``__call__`` is
        ``(*args, **kwargs)``: read one layer out, every doubly-wrapped
        function looked like it took the context, and a one-argument condition
        was called with two.
        """
        if self.wrapper.accepts_context:
            return (data, context)
        if self.wrapper.interface_method is not None:
            return (data,)

        from dataknobs_fsm.core.data_wrapper import wrap_for_lambda

        return (wrap_for_lambda(data),)

    def _create_method(self, method_name: str) -> Callable[..., Any]:
        """Create an interface method that wraps the function.

        Args:
            method_name: Name of the interface method

        Returns:
            Method that calls the wrapped function
        """
        if self.wrapper.is_async:

            async def async_method(
                data: Any, context: Dict[str, Any] | None = None, **kwargs: Any
            ) -> Any:
                result = await self.wrapper.execute_async(
                    *self._call_shape(data, context), **kwargs
                )
                return self._as_interface_result(method_name, result)

            return async_method
        else:

            def sync_method(data: Any, context: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
                result = self.wrapper.execute_sync(*self._call_shape(data, context), **kwargs)
                return self._as_interface_result(method_name, result)

            return sync_method

    @staticmethod
    def _as_interface_result(method_name: str, result: Any) -> Any:
        """Normalize a wrapped function's answer to what its interface declares.

        Only ``transform`` is wrapped. ``ExecutionResult`` is a member of
        :data:`~dataknobs_fsm.functions.base.TransformOutcome` and both engines
        unwrap one, but it is deliberately *not* a member of
        :data:`~dataknobs_fsm.functions.base.ValidationOutcome` --- no
        validator path unwraps it, so a wrapped ``False`` is neither ``False``
        nor a dict and the gate reads it as a pass. Wrapping a validator's
        answer therefore produced a gate that could not refuse; it stayed
        latent only because the route the engines actually took to a
        pre-validator was ``__call__``, which skipped this.

        ``None`` is passed through rather than wrapped. Both engines read a
        raw ``None`` from a transform as "the record was mutated in place and
        must be preserved" (``BaseExecutionEngine.process_transform_result``
        skips it; ``AsyncExecutionEngine._coalesce_transform_result`` returns
        the current data), whereas an ``ExecutionResult`` carrying ``data=None``
        says the record *is* nothing --- ``ensure_dict(None)`` is ``{}``. So
        wrapping the one produced the other, and a transform that mutated in
        place and returned ``None`` emptied the record it had just edited.

        Written once for both flavours, because the two copies this replaces
        were the same four lines and a fix to one of them would not have
        reached the other.
        """
        if method_name != "transform":
            return result
        if result is None or isinstance(result, ExecutionResult):
            return result
        return ExecutionResult.success_result(result)

    def _create_test_method(self) -> Callable[..., Any]:
        """Create a test method that returns (bool, reason)."""
        if self.wrapper.is_async:

            async def async_test(
                data: Any, context: Dict[str, Any] | None = None, **kwargs: Any
            ) -> Any:
                result = await self.wrapper.execute_async(
                    *self._call_shape(data, context), **kwargs
                )
                if isinstance(result, tuple):
                    return result
                return (bool(result), None)

            return async_test
        else:

            def sync_test(data: Any, context: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
                result = self.wrapper.execute_sync(*self._call_shape(data, context), **kwargs)
                if isinstance(result, tuple):
                    return result
                return (bool(result), None)

            return sync_test

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke through the interface method this wrapper built.

        Not straight to the inner :class:`FunctionWrapper`, which is what this
        did. The interface method is where :meth:`_call_shape` decides how many
        arguments the implementation gets, and the engines reach a resolved
        function *both* ways --- a state's pre-validators are called as
        ``validator(record, context)`` while its transforms go through
        ``.transform`` --- so bypassing it made the two routes to one object
        disagree, and a one-argument validator raised ``TypeError`` on the
        route that skipped the shaping. Mirrors
        ``_ResolvedLibraryFunction.__call__``, which has always dispatched this
        way. An interface this wrapper built no method for falls through to the
        inner wrapper unchanged.
        """
        method = getattr(self, INTERFACE_METHODS.get(self.interface, ""), None)
        if method is None:
            return self.wrapper(*args, **kwargs)
        return method(*args, **kwargs)

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

    def __init__(self) -> None:
        """Initialize function manager."""
        self._functions: Dict[str, FunctionWrapper] = {}
        self._builtin_functions: Dict[str, FunctionWrapper] = {}
        self._inline_cache: Dict[str, FunctionWrapper] = {}

    def register_function(
        self,
        name: str,
        func: RegisteredFunction,
        source: FunctionSource = FunctionSource.REGISTERED,
        interface: type | None = None,
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
        functions: Mapping[str, RegisteredFunction],
        source: FunctionSource = FunctionSource.REGISTERED,
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
        self, reference: Union[str, Dict[str, Any], Callable], interface: type | None = None
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
                reference, getattr(reference, "__name__", "anonymous"), FunctionSource.REGISTERED
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
            ref_type = reference.get("type", "inline")

            if ref_type == "registered":
                name = reference.get("name")
                if name:
                    wrapper = self._functions.get(name) or self._builtin_functions.get(name)

            elif ref_type == "inline":
                code = reference.get("code")
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
            namespace: Dict[str, Any] = {"asyncio": asyncio}

            # Add all registered functions to namespace so inline code can call them
            for name, wrapper in self._functions.items():
                # Add the actual function, not the wrapper
                namespace[name] = wrapper.func if hasattr(wrapper, "func") else wrapper

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
                if code.strip().startswith("lambda"):
                    # Evaluate lambda directly
                    func = eval(code, namespace)
                else:
                    # Treat as function body - check if it needs to be async
                    if "await" in code:
                        # Create async wrapper
                        func_def = "async def inline_func(data, context=None):\n"
                    else:
                        # Create sync wrapper
                        func_def = "def inline_func(data, context=None):\n"

                    # Add the code as the function body
                    lines = code.split(";") if ";" in code else [code]

                    # Check if this looks like a simple expression (for conditions)
                    # Common patterns: comparisons, boolean ops, method calls that return bool
                    is_expression = (
                        "==" in code
                        or "!=" in code
                        or "<" in code
                        or ">" in code
                        or " and " in code
                        or " or " in code
                        or " not " in code
                        or code.strip().startswith("not ")
                        or ".get(" in code
                        or "in " in code
                        or code.strip() in ["True", "False"]
                    )

                    if is_expression and "return" not in code and len(lines) == 1:
                        # For expressions, return the expression result
                        func_def += f"    return {code.strip()}\n"
                    else:
                        # For statements, add them as-is
                        for line in lines:
                            stmt = line.strip()
                            if stmt:
                                func_def += f"    {stmt}\n"

                        # Ensure we return data if no explicit return (for transforms)
                        if "return" not in code:
                            func_def += "    return data\n"

                    exec(func_def, namespace)
                    func = namespace.get("inline_func")

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
                lambda data, context=None: data,
                f"inline_error_{id(code)}",
                FunctionSource.INLINE,
            )

    def _adapt_to_interface(
        self, wrapper: FunctionWrapper, interface: type
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
                "source": wrapper.source.value,
                "async": wrapper.is_async,
                "type": "registered",
            }

        for name, wrapper in self._builtin_functions.items():
            result[name] = {
                "source": wrapper.source.value,
                "async": wrapper.is_async,
                "type": "builtin",
            }

        return result

    def clear(self) -> None:
        """Clear all registered functions except builtins."""
        self._functions.clear()
        self._inline_cache.clear()

    def clear_all(self) -> None:
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
    name: str, func: RegisteredFunction, source: FunctionSource = FunctionSource.REGISTERED
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
    reference: Union[str, Dict[str, Any], Callable], interface: type | None = None
) -> Union[FunctionWrapper, InterfaceWrapper, None]:
    """Resolve a function reference.

    Args:
        reference: Function reference
        interface: Optional interface

    Returns:
        FunctionWrapper or None
    """
    return _global_manager.resolve_function(reference, interface)
