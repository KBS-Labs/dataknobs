# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base interfaces and classes for FSM functions.

This module defines the interfaces for:
- Validation functions (check data validity)
- Transform functions (modify data)
- State test functions (determine next state)
- End state test functions (check if processing should end)
- Resources (external systems and services)
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, TypeAlias, TypeVar

from dataknobs_common.callbacks import is_async_callable
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
        metadata: Dict[str, Any] | None = None,
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
    def success_result(cls, data: Any, metadata: Dict[str, Any] | None = None) -> "ExecutionResult":
        """Create a successful result.

        Args:
            data: The result data.
            metadata: Optional metadata.

        Returns:
            A successful ExecutionResult.
        """
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def failure_result(
        cls, error: str, metadata: Dict[str, Any] | None = None
    ) -> "ExecutionResult":
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
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
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


#: What a validator may hand back. ``False`` fails the record; a dict is merged
#: into the record; ``True`` and ``None`` pass it unchanged. That is the whole
#: of what the engines read --- ``AsyncExecutionEngine._run_pre_validators``
#: tests ``result is False`` and ``isinstance(result, dict)``, and nothing else.
#:
#: ``ExecutionResult`` is deliberately **not** a member, although this method
#: was declared to return one until it was measured: neither validator path
#: unwraps it, so a *failing* ``ExecutionResult`` is neither ``False`` nor a
#: dict and the record passes the gate. An implementation written to the old
#: declaration produced a validator that could never reject.
ValidationOutcome: TypeAlias = bool | Dict[str, Any] | None


class IValidationFunction(ABC):
    """Interface for validation functions."""

    @abstractmethod
    def validate(
        self,
        data: Any,
        context: "FunctionContext | Dict[str, Any] | None" = None,
    ) -> ValidationOutcome | Awaitable[ValidationOutcome]:
        """Validate data according to function logic.

        May be written ``def`` or ``async def``: the engines await an awaitable
        result rather than requiring one flavour.

        Args:
            data: The data to validate.
            context: Optional execution context, of the same shape
                :meth:`ITransformFunction.transform` receives.

        Returns:
            ``False`` to fail the record, a dict to merge into it, or ``True`` /
            ``None`` to pass it unchanged --- see :data:`ValidationOutcome`,
            which records why an ``ExecutionResult`` is not among them.
        """
        pass

    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules this function implements.

        Returns:
            Dictionary describing the validation rules.
        """
        pass


#: What a transform may hand back. The two engines agree on all three members
#: --- ``BaseExecutionEngine.process_transform_result`` and
#: ``AsyncExecutionEngine._coalesce_transform_result`` both unwrap an
#: ``ExecutionResult`` (a failing one is raised as the transform's error),
#: treat ``None`` as "the record was mutated in place", and otherwise take the
#: returned value as the new record.
TransformOutcome: TypeAlias = ExecutionResult | Dict[str, Any] | None


class ITransformFunction(ABC):
    """Interface for transform functions."""

    @abstractmethod
    def transform(
        self,
        data: Any,
        context: "FunctionContext | Dict[str, Any] | None" = None,
    ) -> TransformOutcome | Awaitable[TransformOutcome]:
        """Transform data according to function logic.

        May be written ``def`` or ``async def``. The engines route every
        invocation through ``run_callback_off_loop``, which awaits an async
        implementation and offloads a sync one, so neither flavour is the
        privileged one and an implementation picks whichever its work needs.

        Args:
            data: The data to transform.
            context: Optional execution context. The FSM engines always pass a
                ``FunctionContext`` (carrying injected ``resources`` and the
                ``resource_roles`` map); a plain ``dict`` is accepted for
                lightweight/standalone invocation.

        Returns:
            The transformed record, an :class:`ExecutionResult` wrapping it, or
            ``None`` to mean the record was mutated in place --- see
            :data:`TransformOutcome`.
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
    def should_end(
        self,
        data: Any,
        context: "FunctionContext | Dict[str, Any] | None" = None,
    ) -> Tuple[bool, str | None]:
        """Test if processing should end.

        Args:
            data: The current data.
            context: Optional execution context, of the same shape
                :meth:`ITransformFunction.transform` receives.

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


#: What may be handed to an FSM by name --- the ``custom_functions=`` channel
#: on every engine and façade, and :meth:`FSMBuilder.register_function`.
#:
#: A bare interface *instance* is not a ``Callable``: it carries its logic on
#: ``transform`` / ``validate`` / ``test`` / ``should_end``, which is why the
#: tree has three entry points for finding that method ---
#: ``FunctionWrapper._normalize_interface_callable``,
#: :func:`as_state_test_callable`, and
#: ``AsyncExecutionEngine._is_interface_transform``. Which method belongs to
#: which interface is :data:`INTERFACE_METHODS`, read by all of them and by the
#: config builder's resolved adapter rather than spelled out at each site.
#: Declaring the channel as ``dict[str, Callable]`` excluded the shape it
#: exists to carry; under that annotation mypy read all four ``isinstance``
#: arms of ``_normalize_interface_callable`` as unreachable, which is the type
#: checker saying the same thing.
RegisteredFunction: TypeAlias = (
    Callable[..., Any]
    | ITransformFunction
    | IValidationFunction
    | IStateTestFunction
    | IEndStateTestFunction
)


#: The interface method that carries each interface's logic. One mapping, read
#: by everything that has to find that method on an instance --- the wrapper in
#: :mod:`dataknobs_fsm.functions.manager`, the config builder's resolved
#: adapter, and :func:`as_state_test_callable`.
INTERFACE_METHODS: Dict[type, str] = {
    ITransformFunction: "transform",
    IValidationFunction: "validate",
    IStateTestFunction: "test",
    IEndStateTestFunction: "should_end",
}


def interface_method_of(func: Any) -> str | None:
    """The name of the interface method ``func`` implements, or ``None``.

    ``None`` means the object is not one of the four FSM function interfaces:
    an ordinary callable, an already-normalized bound method, or one of the
    wrappers. Callers use the answer to decide what an object *is*, not merely
    what it looks like --- which is the distinction arity alone cannot draw.
    """
    for interface, method in INTERFACE_METHODS.items():
        if isinstance(func, interface):
            return method
    return None


def accepts_context(func: Callable[..., Any], *, default: bool = True) -> bool:
    """Whether a record callable takes the execution context beyond the record.

    The engines invoke every record step as ``func(record, context)``, and both
    conventions in the tree are live: the FSM function library is a mix of
    ``transform(self, data)`` and ``transform(self, data, context=None)``, and a
    consumer's callable may be either. One reading of that question, because
    the tree previously held three and they did not agree.

    **Positional parameters only.** ``*args`` counts --- such a callable can
    receive the context. ``**kwargs`` does not: the context is passed
    *positionally*, so ``def fn(data, **kwargs)`` raises ``TypeError`` when
    called with two. The copy in the config builder counted ``**kwargs`` and
    produced exactly that crash for a ``(data, **kwargs)`` implementation.

    Args:
        func: The callable to read. A bound method, so ``self`` is already
            excluded; an unintrospectable builtin answers ``default``.
        default: The answer when the signature cannot be read. ``True`` (the
            default) for an interface implementation, whose declaration says it
            takes the context. ``False`` for an arbitrary record callable,
            where the bare ``record -> X`` shape is the common one and passing
            an argument the callable cannot take is the worse failure.

    Returns:
        ``True`` when the callable can receive the context positionally.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return default
    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in positional):
        return True
    return len(positional) >= 2


def accepts_one_argument(func: Any, *, default: bool = True) -> bool:
    """Whether ``func`` can be called with exactly one positional argument.

    The sibling question to :func:`accepts_context`, and a different one. That
    one asks whether a callable *wants* the context; this one asks whether it
    will accept the state object *alone* --- which a ``(record, context=None)``
    callable will and a ``(record, context)`` callable will not. The two
    disagree on exactly that shape, and both answers are needed, because the
    engine's state-step sites and its arc-condition site have opposite
    preferences: a state step is offered the state object first and a
    condition is offered the context first.

    It exists because those sites used to answer this by *calling* the
    function and catching the failure. Argument binding raises ``TypeError``
    before the callable's frame exists, which is the case they were written
    for --- but a ``TypeError`` from inside the body, after the work has been
    done, is indistinguishable from outside, so a transform with an ordinary
    bug in it was run a second time with different arguments. Reading the
    signature answers the same question without running anything.

    **Positional parameters only**, and required ones are what decide it: a
    callable is asked for one argument, so it must have somewhere to put it
    and nothing else it insists on. ``*args`` satisfies both. A *required*
    keyword-only parameter cannot be filled by a positional call at all, so
    such a callable answers ``False`` here and is given ``(record, context)``,
    where a keyword ``context`` at least has a chance of being bound by name
    --- which is more than the one-argument call could offer it.

    Args:
        func: The callable to read. An unintrospectable builtin answers
            ``default``.
        default: The answer when the signature cannot be read. ``True``, which
            is the historical first attempt at every site that asks.

    Returns:
        ``True`` when a one-positional-argument call would bind.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return default
    slots = 0
    required = 0
    for param in params:
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return required <= 1
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            slots += 1
            if param.default is inspect.Parameter.empty:
                required += 1
        elif (
            param.kind is inspect.Parameter.KEYWORD_ONLY
            and param.default is inspect.Parameter.empty
        ):
            return False
    return slots >= 1 and required <= 1


def state_step_args(
    func: Any,
    *,
    state_obj: Any,
    record: Dict[str, Any],
    context: Any,
) -> Tuple[Any, ...]:
    """The positional arguments a state transform or validator should receive.

    One reading for the three sites that dispatch a state step --- the async
    engine's transform and validator loops, and ``AdvancedFSM``'s own copy of
    the transform loop. Each previously carried its own ``try`` / ``except
    (TypeError, AttributeError)`` pair, which is three chances to fix a bug
    in one of them and two places for it to survive.

    The preference is the historical one and deliberately unchanged: a
    callable that will take the state object alone gets the state object, so
    an inline ``lambda state: ...`` keeps reading ``state.data``; anything
    else gets ``(record, context)``. Only the way the question is answered
    changed.

    Args:
        func: The callable about to be invoked.
        state_obj: The ``StateDataWrapper`` for a one-argument callable.
        record: The raw record for a ``(record, context)`` callable.
        context: The ``FunctionContext`` for a ``(record, context)`` callable.

    Returns:
        The positional arguments to splat into the call.
    """
    return (state_obj,) if accepts_one_argument(func) else (record, context)


def normalize_record_callable(
    fn: Callable[..., Any],
    *,
    coerce: Callable[[Any], Any] | None = None,
) -> Callable[..., Any]:
    """Arity- and await-normalize a record callable to ``(record, context)``.

    The FSM engine always invokes a record step as ``fn(record, context)``. A
    supplied callable may be a bare ``record -> X`` callable, a
    ``(record, context) -> X`` callable, or an
    :class:`~dataknobs_fsm.functions.base.ITransformFunction`'s bound method
    (sync ``(data)`` or async ``(data, context)``). The returned callable always
    accepts ``(record, context)``, forwards the right number of arguments, and is
    a coroutine function iff calling ``fn`` produces an awaitable — so the
    caller's ``iscoroutinefunction`` / ``isawaitable`` check routes it correctly.
    Note the second half of that: a callable *object* with an ``async def``
    ``__call__`` is not itself a coroutine function, and normalizing it yields
    one, which is the point. The judgement is
    :func:`~dataknobs_common.callbacks.is_async_callable`'s.

    Arity detection is :func:`~dataknobs_fsm.functions.base.accepts_context`,
    shared with the config builder's resolved adapter and the function wrapper
    so all three doors into an FSM read one signature the same way. It counts
    positional parameters only: a callable declaring two or more positionals (or
    ``*args``) receives ``(record, context)``; otherwise it receives ``record``
    alone. A predicate that declares ``context`` as a *required keyword-only*
    argument (``def fn(record, *, context): ...``) is therefore called with the
    record alone and raises ``TypeError`` at evaluation time — write
    ``(record, context)`` or ``(record, context=None)`` instead.

    Args:
        fn: The user callable to normalize.
        coerce: Optional terminal coercion applied to the callable's result
            (e.g. ``bool`` for a gate). ``None`` (the default) returns the
            result unchanged (the enricher form).

    Returns:
        A ``(record, context)`` callable (a coroutine function when ``fn`` is).
    """
    # Builtins / C-callables with no introspectable signature: be permissive and
    # pass only the record (the common ``record -> X`` shape), which is what
    # ``default=False`` asks of the shared reading.
    wants_context = accepts_context(fn, default=False)

    # `is_async_callable` rather than `inspect.iscoroutinefunction`: the latter
    # answers for functions and reports a callable *object* with an `async def`
    # __call__ as synchronous. Such a callable would take `sync_call` below,
    # where `coerce` is applied to the coroutine rather than to the answer ---
    # and the gate's `coerce` is `bool`, so a predicate that said no becomes a
    # gate that says yes, uniformly, for every record.
    if is_async_callable(fn):

        async def async_call(data: dict, context: Any = None) -> Any:
            out = await (fn(data, context) if wants_context else fn(data))
            return coerce(out) if coerce is not None else out

        return async_call

    def sync_call(data: dict, context: Any = None) -> Any:
        out = fn(data, context) if wants_context else fn(data)
        return coerce(out) if coerce is not None else out

    return sync_call


def as_state_test_callable(func: Any) -> Any:
    """Return the callable form of a resolved arc-condition / pre-test function.

    A bare ``IStateTestFunction`` instance carries its condition logic on
    ``.test(data, context) -> (passed, reason)`` and is not itself callable, so
    the execution engines (which invoke every pre-test uniformly as
    ``func(data, context)``) cannot dispatch it. Return the bound ``.test``
    method for a bare interface instance; every already-callable form — plain
    predicates, :class:`FunctionWrapper`/``InterfaceWrapper``, and the config
    builder's resolved adapters — passes through unchanged.

    This mirrors ``FunctionWrapper._normalize_interface_callable`` for the two
    paths that bypass it, both of which store engine-injected functions raw:
    the async engine's ``custom_functions`` merge
    (``AsyncExecutionEngine._get_merged_functions``), and
    ``AdvancedFSM._resolve_test_function`` — ``AdvancedFSM`` calls
    ``FSMBuilder.build`` itself rather than ``build_fsm``, so nothing
    normalizes what it was handed. It is deliberately scoped to
    ``IStateTestFunction`` only — the transform path has its own deterministic
    ``ITransformFunction`` dispatch
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
    resolved = func.test if isinstance(func, IStateTestFunction) else func
    if not callable(resolved) or accepts_context(resolved, default=True):
        return resolved
    # A one-argument condition, arity-normalized rather than returned raw: both
    # call sites invoke a pre-test as ``func(data, context)``, and
    # ``test(self, data)`` is a shape the tree ships. Calling it with two
    # arguments raises ``TypeError``, and the arc-skipping ``except`` reads
    # that as the condition saying no --- so the arc silently disappears with
    # the step still reporting success. The wrappers are unaffected: their
    # ``__call__`` is ``(*args, **kwargs)``, which accepts the context, so they
    # take the early return above and keep doing their own shaping.
    return normalize_record_callable(resolved)


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
            context={"validation_errors": list(validation_errors)} if validation_errors else None,
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
        super().__init__(message, context={"from_state": from_state, "to_state": to_state})
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
                if self.execution_count > 0
                else 0
            ),
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
    def execute(self, data: Any, context: "FunctionContext") -> Any:
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

    def __init__(self) -> None:
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
        all_names: List[str] = []
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
