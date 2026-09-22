# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Arc implementation for FSM state transitions."""

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, TYPE_CHECKING

from dataknobs_fsm.core.exceptions import FunctionError, ResourceError
from dataknobs_fsm.functions.base import FunctionContext, as_state_test_callable

if TYPE_CHECKING:
    from dataknobs_fsm.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


class DataIsolationMode(Enum):
    """Data isolation modes for push arcs."""

    COPY = "copy"  # Deep copy data when pushing
    REFERENCE = "reference"  # Pass data by reference
    SERIALIZE = "serialize"  # Serialize/deserialize for isolation

    def apply(self, data: Any) -> Any:
        """Produce the sub-network's view of ``data`` under this isolation mode.

        This is the single source of truth for push-arc data isolation, shared
        by every executor that pushes into a sub-network so they cannot drift:

        - ``COPY`` deep-copies the data (full isolation; the default).
        - ``SERIALIZE`` round-trips through the project JSON encoder, which
          handles non-JSON-native types (datetimes, sets, ``FSMData``, …) that
          stdlib ``json`` would reject.
        - ``REFERENCE`` shares the data by reference (no isolation).
        """
        if self is DataIsolationMode.COPY:
            import copy

            return copy.deepcopy(data)
        if self is DataIsolationMode.SERIALIZE:
            from dataknobs_fsm.utils.json_encoder import dumps, loads

            return loads(dumps(data))
        return data


@dataclass
class TransformSpec:
    """A transform function name with optional parameters.

    When transforms are specified with per-invocation configuration
    (e.g., via ``FunctionReference.params`` in FSM config), the params
    are carried here and passed as ``**kwargs`` to the transform function.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """The spec as plain data, for :meth:`ArcDefinition.to_dict`."""
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformSpec":
        """Rebuild a spec from :meth:`to_dict`."""
        return cls(name=data["name"], params=dict(data.get("params") or {}))


def transform_to_data(
    transform: "str | TransformSpec | list[str | TransformSpec] | None",
) -> Any:
    """An arc's ``transform`` field as plain data.

    The field's four shapes are a property of this module, so the reader and
    the writer of them live beside each other --- as
    :func:`transform_function_names` and :meth:`PushArc.parse_target` already
    do. A name stays a string and a :class:`TransformSpec` becomes a mapping,
    which is what tells them apart on the way back in.

    ``StateNetwork.to_dict`` used to write the field whole. That was harmless
    only for as long as a ``TransformSpec`` could not reach the network's arc
    list: the config builder reduced a transform to its name before handing an
    arc over. Now that ``add_arc`` stores the builder's own arc, any config
    with transform ``params`` puts a dataclass instance in a dictionary that
    is supposed to be data, and ``json.dumps`` refuses it.
    """
    if transform is None:
        return None
    if isinstance(transform, list):
        return [transform_to_data(item) for item in transform]
    if isinstance(transform, TransformSpec):
        return transform.to_dict()
    return transform


def _transform_item_from_data(item: Any) -> "str | TransformSpec":
    """One element of a serialized ``transform`` field.

    Split out from :func:`transform_from_data` rather than recursing into it,
    because the field nests exactly one level --- a list of names and specs,
    never a list of lists --- and a function that recursed into itself would
    have to declare a return type saying otherwise.

    Args:
        item: A name, or the mapping :meth:`TransformSpec.to_dict` produces.

    Returns:
        The name or the rebuilt spec.

    Raises:
        ValueError: If the element is neither, which means the payload was not
            written by :func:`transform_to_data`.
    """
    if isinstance(item, dict):
        return TransformSpec.from_dict(item)
    if isinstance(item, str):
        return item
    raise ValueError(f"A transform is a name or a TransformSpec mapping, not {type(item).__name__}")


def transform_from_data(
    data: Any,
) -> "str | TransformSpec | list[str | TransformSpec] | None":
    """The inverse of :func:`transform_to_data`.

    A mapping is a :class:`TransformSpec`; a string is a name; a list is
    handled element by element.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [_transform_item_from_data(item) for item in data]
    return _transform_item_from_data(data)


def transform_function_names(
    transform: "str | TransformSpec | list[str | TransformSpec] | None",
) -> list[str]:
    """Every function name an arc's ``transform`` field refers to.

    The field carries four shapes --- nothing, one name, one
    :class:`TransformSpec` carrying a name plus params, or a list of names and
    specs --- and a caller that wants "which functions does this arc use"
    should not have to know that. ``FSM.get_all_functions`` did: it added
    ``arc.transform`` to a set whole, so a spec went in as an object and a
    *list* went in as an unhashable value, raising ``TypeError`` on any arc
    configured with chained transforms.
    """
    if transform is None:
        return []
    items = transform if isinstance(transform, list) else [transform]
    return [item.name if isinstance(item, TransformSpec) else item for item in items]


@dataclass
class ArcDefinition:
    """Definition of an arc between states.

    This class defines the static properties of an arc,
    including the transition logic and resource requirements.

    There used to be a second arc type. ``StateNetwork`` kept its own ``Arc``
    --- source, target, pre-test, transform, metadata --- in ``_arcs`` and
    ``_arc_index``, while the engines read ``ArcDefinition`` off
    ``StateDefinition.outgoing_arcs``, and the two were populated by different
    writers. The network's ``arcs`` property existed to translate between them
    and rebuilt a *lossy* ``ArcDefinition`` on every call, dropping
    ``priority``, ``definition_order`` and ``required_resources`` --- so the
    accessor that looked like it answered "what arcs are here" answered with
    arcs the engines would have ordered differently. This is now the only arc
    type, and :meth:`StateNetwork.add_arc` stores one object in every index.
    """

    target_state: str
    pre_test: str | None = None
    transform: str | TransformSpec | list[str | TransformSpec] | None = None
    priority: int = 0  # Higher priority arcs are evaluated first
    definition_order: int = 0  # Track definition order for stable sorting
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Resource requirements for this arc
    required_resources: Dict[str, str] = field(default_factory=dict)
    # e.g., {'database': 'main_db', 'llm': 'gpt4'}

    source_state: str = ""
    """The state this arc leaves, stamped by :meth:`StateNetwork.add_arc`.

    Empty on an arc that has not been added to a network yet. It is last in the
    field order, and defaulted, because an arc reached through
    ``state.outgoing_arcs`` already knows its source from the state holding it
    --- the field exists so an arc reached through the *network's* index knows
    it too, without the index having to carry the answer alongside.
    """

    kind: ClassVar[str] = "arc"
    """The tag :meth:`to_dict` writes so :func:`arc_from_dict` can pick a class.

    A serialized arc that does not record which class it was comes back as the
    base one. For a push arc that is silent: it becomes an ordinary transition
    to the state named in ``target``, so the sub-network is never entered and
    nothing raises.
    """

    def to_dict(self) -> Dict[str, Any]:
        """The arc as plain data, tagged with its :attr:`kind`.

        A subclass extends this with its own fields and inherits the tag from
        its own ``kind``; see :meth:`PushArc.to_dict`.
        """
        return {
            "kind": type(self).kind,
            "source": self.source_state,
            "target": self.target_state,
            "pre_test": self.pre_test,
            "transform": transform_to_data(self.transform),
            "metadata": dict(self.metadata),
            "priority": self.priority,
            "definition_order": self.definition_order,
            "required_resources": dict(self.required_resources),
        }

    @classmethod
    def _base_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """The fields every arc carries, read off ``data``.

        Shared with every subclass so the base half of the payload is read in
        one place: a subclass that re-read it would be free to disagree about
        a default, which is the shape of drift this module has already had
        once.
        """
        return {
            "target_state": data["target"],
            "pre_test": data.get("pre_test"),
            "transform": transform_from_data(data.get("transform")),
            "metadata": dict(data.get("metadata") or {}),
            "priority": data.get("priority", 0),
            "definition_order": data.get("definition_order", 0),
            "required_resources": dict(data.get("required_resources") or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArcDefinition":
        """Rebuild an arc of *this* class from :meth:`to_dict`.

        ``source_state`` is deliberately not read back here: it is stamped by
        :meth:`StateNetwork.add_arc`, which is the only writer of it, and a
        caller rebuilding a loose arc has no network to be a source in yet.
        """
        return cls(**cls._base_kwargs(data))

    def __hash__(self) -> int:
        """Make ArcDefinition hashable.

        ``source_state`` participates: two arcs that differ only in where they
        start are different arcs, and a network keyed on the old tuple collided
        them.
        """
        transform_key: tuple[str | None, ...] | str | None
        if isinstance(self.transform, list):
            transform_key = tuple(
                t.name if isinstance(t, TransformSpec) else t for t in self.transform
            )
        elif isinstance(self.transform, TransformSpec):
            transform_key = self.transform.name
        else:
            transform_key = self.transform
        return hash(
            (self.source_state, self.target_state, self.pre_test, transform_key, self.priority)
        )

    @property
    def name(self) -> str:
        """The arc's name: ``metadata['name']``, else ``source->target``.

        Carried over from the retired ``StateNetwork.Arc``, which had it while
        ``ArcDefinition`` did not. The engines filter by it --- ``execute(...,
        arc_name=...)`` reaches ``[arc for arc in state.outgoing_arcs if
        hasattr(arc, "name") and arc.name == arc_name]`` --- and
        ``state.outgoing_arcs`` held the type *without* the property, so that
        filter matched nothing and the guard that should have said so read as
        an ordinary "no arc by that name".
        """
        name = self.metadata.get("name")
        if isinstance(name, str):
            return name
        return f"{self.source_state}->{self.target_state}"


@dataclass
class PushArc(ArcDefinition):
    """Arc that pushes to a sub-network.

    Push arcs allow hierarchical state machine composition
    by pushing execution to a sub-network and returning
    when the sub-network completes.
    """

    kind: ClassVar[str] = "push"

    target_network: str = ""  # Name of the target network
    return_state: str | None = None  # State to return to after sub-network
    isolation_mode: DataIsolationMode = DataIsolationMode.COPY
    pass_context: bool = True  # Whether to pass execution context

    # Mapping of data from parent to child network
    data_mapping: Dict[str, str] = field(default_factory=dict)
    # e.g., {'parent_field': 'child_field'}

    # Mapping of results from child to parent network
    result_mapping: Dict[str, str] = field(default_factory=dict)
    # e.g., {'child_result': 'parent_field'}

    def parse_target(self) -> "tuple[str, str | None]":
        """Split ``target_network`` into ``(network, explicit_initial_state?)``.

        ``target_network`` carries two forms --- ``"validation"`` enters the
        sub-network at its own initial state, ``"validation:deep_check"``
        enters it at a named one --- and the syntax is a property of this
        field, so the one reader of it lives here.

        It did not. The engine split the string and the config builder's
        completeness check compared the whole of it against the known network
        names, which made ``"validation:deep_check"`` --- a documented form ---
        report as a missing network. That never surfaced because the check
        itself could not run: it reached arcs through the network's index,
        which held a different arc type, so ``isinstance(arc, PushArc)`` was
        always false and the branch was dead.
        """
        if ":" in self.target_network:
            network_name, initial_state = self.target_network.split(":", 1)
            return network_name, initial_state.strip()
        return self.target_network, None

    def to_dict(self) -> Dict[str, Any]:
        """The base payload plus the fields that make this a sub-network call."""
        data = super().to_dict()
        data.update(
            {
                "target_network": self.target_network,
                "return_state": self.return_state,
                "isolation_mode": self.isolation_mode.value,
                "pass_context": self.pass_context,
                "data_mapping": dict(self.data_mapping),
                "result_mapping": dict(self.result_mapping),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PushArc":
        """Rebuild a push arc, including where it pushes to."""
        raw_mode = data.get("isolation_mode")
        return cls(
            **cls._base_kwargs(data),
            target_network=data.get("target_network", ""),
            return_state=data.get("return_state"),
            isolation_mode=(
                DataIsolationMode(raw_mode) if raw_mode is not None else DataIsolationMode.COPY
            ),
            pass_context=data.get("pass_context", True),
            data_mapping=dict(data.get("data_mapping") or {}),
            result_mapping=dict(data.get("result_mapping") or {}),
        )


ARC_TYPES: Dict[str, type[ArcDefinition]] = {
    ArcDefinition.kind: ArcDefinition,
    PushArc.kind: PushArc,
}
"""The arc classes :func:`arc_from_dict` can rebuild, by :attr:`ArcDefinition.kind`.

A consumer with its own arc subclass registers it here rather than losing it to
the base class on every round trip --- see :func:`register_arc_type`.
"""


def register_arc_type(arc_type: type[ArcDefinition]) -> None:
    """Make ``arc_type`` rebuildable by :func:`arc_from_dict`.

    Args:
        arc_type: An :class:`ArcDefinition` subclass declaring its own
            :attr:`~ArcDefinition.kind`.

    Raises:
        ValueError: If its ``kind`` is already registered to another class, or
            if it did not declare one of its own (which would silently shadow
            the base entry and send every plain arc through the subclass).
    """
    kind = arc_type.kind
    registered = ARC_TYPES.get(kind)
    if registered is not None and registered is not arc_type:
        raise ValueError(
            f"Arc kind '{kind}' is already registered to {registered.__name__}; "
            f"{arc_type.__name__} needs a kind of its own"
        )
    ARC_TYPES[kind] = arc_type


def arc_from_dict(data: Dict[str, Any]) -> ArcDefinition:
    """Rebuild an arc of whichever class :meth:`ArcDefinition.to_dict` recorded.

    An unrecognised ``kind`` --- a payload written by a consumer whose arc type
    is not registered in *this* process --- raises rather than quietly
    returning a base arc, because quietly returning a base arc is how a
    sub-network call disappears.

    Args:
        data: A mapping in the shape :meth:`ArcDefinition.to_dict` produces.

    Returns:
        The rebuilt arc.

    Raises:
        ValueError: If ``kind`` names a class that is not registered.
    """
    kind = data.get("kind", ArcDefinition.kind)
    arc_type = ARC_TYPES.get(kind)
    if arc_type is None:
        known = ", ".join(sorted(ARC_TYPES))
        raise ValueError(f"Unknown arc kind '{kind}' (known kinds: {known})")
    return arc_type.from_dict(data)


class ArcExecution:
    """Handles the execution of arc transitions.

    This class manages the runtime execution of arcs,
    including resource allocation, streaming support,
    and transaction participation.
    """

    def __init__(self, arc_def: ArcDefinition, source_state: str, function_registry: Any) -> None:
        """Initialize arc execution.

        Args:
            arc_def: Arc definition.
            source_state: Source state name.
            function_registry: Registry of available functions (FunctionRegistry or dict).
        """
        self.arc_def = arc_def
        self.source_state = source_state
        self.function_registry = function_registry

        # Execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0

    def _log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning message to log.
        """
        logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Error message to log.
        """
        logger.error(message)

    async def can_execute_async(self, context: "ExecutionContext", data: Any = None) -> bool:
        """Check if arc can be executed, awaiting async pre-tests.

        Resolves the arc's ``pre_test`` function, normalizes a bare
        ``IStateTestFunction`` instance to its bound ``.test`` method, invokes
        it, and awaits the result when the function is a coroutine.

        Args:
            context: Execution context.
            data: Current data.

        Returns:
            True if arc can be executed.
        """
        if not self.arc_def.pre_test:
            return True

        # Handle both FunctionRegistry and dict for pre-test function lookup
        if hasattr(self.function_registry, "get_function"):
            pre_test_func = self.function_registry.get_function(self.arc_def.pre_test)
        elif isinstance(self.function_registry, dict):
            pre_test_func = self.function_registry.get(self.arc_def.pre_test)
        else:
            pre_test_func = None

        if pre_test_func is None:
            raise FunctionError(
                f"Pre-test function '{self.arc_def.pre_test}' not found",
                from_state=self.source_state,
                to_state=self.arc_def.target_state,
            )

        resources: Dict[str, Any] = {}
        try:
            # Allocate the arc's declared resources (merging state resources) so
            # a resource-bearing pre-test condition can reach them, then build
            # the function context carrying them (+ the role map).
            state_resources = getattr(context, "current_state_resources", None)
            resources = self._allocate_resources(context, state_resources)
            func_context = self._create_function_context(context, resources, apply_factory=False)

            # A bare IStateTestFunction instance is not callable; normalize it to
            # its bound .test method so an interface-instance arc condition is
            # dispatched, not called as func(data, context) (parity with the
            # async engine's _evaluate_arc).
            pre_test_func = as_state_test_callable(pre_test_func)

            # Execute pre-test
            result = pre_test_func(data, func_context)
            if inspect.isawaitable(result):
                result = await result

            # Handle tuple return from InterfaceWrapper (returns (result, error))
            if isinstance(result, tuple) and len(result) == 2:
                return bool(result[0])
            return bool(result)

        except Exception as e:
            raise FunctionError(
                f"Pre-test execution failed: {e}",
                from_state=self.source_state,
                to_state=self.arc_def.target_state,
            ) from e
        finally:
            self._release_resources(context)

    async def execute_async(
        self,
        context: "ExecutionContext",
        data: Any = None,
        stream_enabled: bool = False,
        *,
        arc_resources: Dict[str, Any] | None = None,
    ) -> Any:
        """Execute the arc transition, awaiting async transforms.

        Uses ``_execute_single_transform_async`` so that async transform
        functions are properly awaited.

        Args:
            context: Execution context.
            data: Current data.
            stream_enabled: Whether streaming is enabled.
            arc_resources: Pre-acquired, caller-owned resources. When provided,
                this method neither acquires nor releases resources (the caller
                owns their lifecycle); when ``None`` it allocates and releases
                them itself. Mirrors :meth:`execute`.

        Returns:
            Transformed data.
        """
        import time

        start_time = time.time()

        # When the caller hands in pre-acquired resources it owns their
        # lifecycle; we must not allocate or release them here.
        owns_resources = arc_resources is None

        try:
            # Branching on the value rather than on ``owns_resources`` beside
            # it: the flag and the value say the same thing, and only one of
            # them carries it to the reader.
            if arc_resources is None:
                # Allocate required resources (merging with state resources)
                resources = self._allocate_resources(context, context.current_state_resources)
            else:
                resources = arc_resources

            # Execute transform(s) if defined
            if self.arc_def.transform:
                # Normalize to list for uniform handling
                transform_refs = (
                    self.arc_def.transform
                    if isinstance(self.arc_def.transform, list)
                    else [self.arc_def.transform]
                )

                # Create function context with resources (shared across all transforms)
                func_context = self._create_function_context(context, resources, stream_enabled)

                result = data
                for transform_ref in transform_refs:
                    name = (
                        transform_ref.name
                        if isinstance(transform_ref, TransformSpec)
                        else transform_ref
                    )
                    result = await self._execute_single_transform_async(
                        name,
                        result,
                        func_context,
                        stream_enabled,
                        transform_ref=transform_ref,
                    )
            else:
                # No transform, pass data through
                result = data

            # Update statistics
            self.execution_count += 1
            self.success_count += 1

            return result

        except Exception as e:
            self.execution_count += 1
            self.failure_count += 1

            raise FunctionError(
                f"Arc execution failed: {e}",
                from_state=self.source_state,
                to_state=self.arc_def.target_state,
            ) from e
        finally:
            elapsed = time.time() - start_time
            self.total_execution_time += elapsed

            # Release only resources we allocated; caller-owned (pre-acquired)
            # resources are released by the caller.
            if owns_resources and "resources" in locals():
                self._release_resources(context)

    async def _execute_single_transform_async(
        self,
        transform_name: str,
        data: Any,
        func_context: FunctionContext,
        stream_enabled: bool = False,
        transform_ref: str | TransformSpec | None = None,
    ) -> Any:
        """Execute a single transform function, awaiting if async.

        Awaits the result when the transform function returns a coroutine, so
        both sync and async transform callables are supported.

        Args:
            transform_name: Registered name of the transform function.
            data: Input data to transform.
            func_context: Function context with resources and metadata.
            stream_enabled: Whether streaming is enabled.

        Returns:
            Transformed data.

        Raises:
            FunctionError: If the transform function is not found or fails.
        """
        # Look up the transform function
        if hasattr(self.function_registry, "get_function"):
            transform_func = self.function_registry.get_function(transform_name)
        elif isinstance(self.function_registry, dict):
            transform_func = self.function_registry.get(transform_name)
        else:
            transform_func = None

        if transform_func is None:
            raise FunctionError(
                f"Transform function '{transform_name}' not found",
                from_state=self.source_state,
                to_state=self.arc_def.target_state,
            )

        # Resolve params from TransformSpec if present
        params = (
            transform_ref.params
            if isinstance(transform_ref, TransformSpec) and transform_ref.params
            else {}
        )

        # Handle streaming vs non-streaming execution
        if stream_enabled and hasattr(transform_func, "stream_capable"):
            return self._execute_streaming(transform_func, data, func_context)

        # Call the transform function, passing params as kwargs if present
        if hasattr(transform_func, "transform"):
            result = (
                transform_func.transform(data, func_context, **params)
                if params
                else transform_func.transform(data, func_context)
            )
        elif callable(transform_func):
            result = (
                transform_func(data, func_context, **params)
                if params
                else transform_func(data, func_context)
            )
        else:
            raise ValueError(f"Transform {transform_name} is not callable")

        # Await if the result is a coroutine
        if inspect.isawaitable(result):
            result = await result

        # Handle ExecutionResult objects
        from dataknobs_fsm.functions.base import ExecutionResult

        if isinstance(result, ExecutionResult):
            if result.success:
                return result.data
            raise FunctionError(
                result.error or "Transform failed",
                from_state=self.source_state,
                to_state=self.arc_def.target_state,
            )

        # Transforms that mutate data in-place return None;
        # preserve input data for the next transform in the chain.
        if result is None:
            return data
        return result

    def _create_function_context(
        self,
        exec_context: "ExecutionContext",
        resources: Dict[str, Any] | None = None,
        stream_enabled: bool = False,
        *,
        apply_factory: bool = True,
    ) -> Any:
        """Create function context for execution.

        Builds a :class:`FunctionContext` and, when ``apply_factory`` is True and
        the ``ExecutionContext`` has a ``transform_context_factory``, passes it
        through the factory so that application-level context (e.g.
        ``TransformContext``) can be composed on top of the FSM-level context.

        Arc *condition* (pre-test) paths pass ``apply_factory=False``: the factory's
        documented scope is transforms, so a condition receives the plain
        resource-bearing context — matching the async engine (``apply_factory=False``
        in ``_evaluate_arc``). ``can_execute_async`` (the condition path) builds the
        context with ``apply_factory=False`` while ``execute_async`` (the transform
        path) applies the factory; without this gate the factory would wrap arc
        conditions too.

        Args:
            exec_context: Execution context.
            resources: Allocated resources.
            stream_enabled: Whether streaming is enabled.
            apply_factory: Whether to run ``transform_context_factory`` (True on
                transform paths, False on condition paths).

        Returns:
            ``FunctionContext`` (default) or factory output.
        """
        # Derive a representative function name for the context. A transform
        # is a name, a ``TransformSpec`` carrying that name plus params, or a
        # list of either, so unwrapping the spec is done once here rather than
        # left to whoever reads ``function_name`` and finds an object.
        transform = self.arc_def.transform
        first: str | TransformSpec | None
        if isinstance(transform, list):
            first = transform[0] if transform else None
        else:
            first = transform
        named = first if first is not None else self.arc_def.pre_test
        func_name = named.name if isinstance(named, TransformSpec) else named
        if func_name is None:
            # An arc with neither a transform nor a pre-test still has a name.
            # ``FunctionContext.function_name`` is declared ``str``, so the
            # ``None`` this passed was never a value the contract allowed ---
            # and it reached logs and error messages as one.
            func_name = self.arc_def.name

        func_context = FunctionContext(
            state_name=self.source_state,
            function_name=func_name,
            metadata={
                "source_state": self.source_state,
                "target_state": self.arc_def.target_state,
                "arc_priority": self.arc_def.priority,
                "stream_enabled": stream_enabled,
                # {role: name} map for role-based access via
                # FunctionContext.resource_for_role(role).
                "resource_roles": dict(self.arc_def.required_resources),
            },
            resources=resources or {},
            variables=exec_context.variables,
            network_name=(
                exec_context.network_stack[-1][0] if exec_context.network_stack else None
            ),
        )

        if apply_factory and exec_context.transform_context_factory:
            return exec_context.transform_context_factory(func_context)
        return func_context

    def _allocate_resources(
        self, context: "ExecutionContext", state_resources: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Allocate required resources for arc execution, merging with state resources.

        Args:
            context: Execution context.
            state_resources: Already allocated state resources to merge with.

        Returns:
            Dictionary of merged resources (state + arc-specific).
        """
        # Start with state resources if provided
        resources = dict(state_resources) if state_resources else {}

        # Get resource manager from context
        resource_manager = getattr(context, "resource_manager", None)
        if not resource_manager:
            # No resource manager available - return existing resources
            return resources

        # Generate unique owner ID for this arc execution
        # Create an arc identifier from source and target states
        arc_identifier = f"{self.source_state}_to_{self.arc_def.target_state}"
        owner_id = f"arc_{arc_identifier}_{getattr(context, 'execution_id', 'unknown')}"

        for _resource_role, resource_name in self.arc_def.required_resources.items():
            # Key by resource NAME (not the role/type key) so a function reads
            # context.resources['<name>'] identically on the sync and async
            # engines. The {role: name} map is surfaced separately via
            # FunctionContext.metadata['resource_roles'] for role-based access.
            # Skip if already have this resource from state.
            if resource_name in resources:
                self._log_warning(
                    f"Arc resource '{resource_name}' already allocated by state, skipping"
                )
                continue

            try:
                # Acquire arc-specific resource. No acquire timeout — arc
                # resources are declared by name only (no per-resource
                # timeout_seconds), and acquisition is a cheap in-process
                # bookkeeping call. This matches the async engine
                # (_acquire_named_resources) and the sync condition path
                # (_acquire_arc_resources), which both acquire with timeout=None;
                # a hard-coded value here was the lone cross-engine drift.
                # State-declared resources still honor their own timeout_seconds.
                resource = resource_manager.acquire(
                    name=resource_name,
                    owner_id=owner_id,
                    timeout=None,
                )
                resources[resource_name] = resource

                # Track for cleanup (only arc-specific resources)
                context._arc_acquired_resources[resource_name] = owner_id

            except Exception as e:
                # Resource acquisition failed - clean up only arc-specific resources
                self._release_arc_resources(context, context._arc_acquired_resources)
                # Bounded message AND bounded details: `details` is echoed by
                # generic renderers just as the message is, so relaying the
                # provider's text there would reopen what the message closes.
                raise ResourceError(
                    resource_id=resource_name,
                    message=(f"Failed to acquire arc resource ({type(e).__name__})"),
                    details={
                        "operation": "acquire",
                        "error_type": type(e).__name__,
                    },
                ) from e

        return resources

    def _release_arc_resources(
        self, context: "ExecutionContext", arc_resources: Dict[str, str]
    ) -> None:
        """Release only arc-specific resources, not state resources.

        Args:
            context: Execution context.
            arc_resources: Map of resource_name -> owner_id for arc resources only.
        """
        if not arc_resources:
            return

        resource_manager = getattr(context, "resource_manager", None)
        if not resource_manager:
            return

        for resource_name, owner_id in arc_resources.items():
            try:
                resource_manager.release(resource_name, owner_id)
            except Exception as e:
                self._log_error(f"Failed to release arc resource {resource_name}: {e}")

        # Clear arc resources tracking
        context._arc_acquired_resources = {}

    def _release_resources(
        self,
        context: "ExecutionContext",
    ) -> None:
        """Release the arc-specific resources allocated by this arc.

        Releases only the resources :meth:`_allocate_resources` acquired for
        this arc (tracked in ``context._arc_acquired_resources``) — never the
        state resources merged in, which the engine's state path owns and
        releases. The authoritative release set is the arc-acquired tracking
        map (name → owner_id), not the merged name → handle dict; the latter
        lacks the owner ids needed to release, so it is deliberately not a
        parameter here.

        Args:
            context: Execution context.
        """
        arc_acquired = context._arc_acquired_resources
        if not arc_acquired:
            return
        # _release_arc_resources releases by (name, owner_id) and clears the map.
        self._release_arc_resources(context, dict(arc_acquired))

    def _execute_streaming(self, func: Callable, data: Any, context: FunctionContext) -> Any:
        """Execute function with streaming support.

        Args:
            func: Function to execute.
            data: Input data.
            context: Function context.

        Returns:
            Streamed result.
        """
        # This would integrate with the streaming system
        # For now, we just execute normally
        return func(data, context)

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary of statistics.
        """
        avg_time = 0.0
        if self.execution_count > 0:
            avg_time = self.total_execution_time / self.execution_count

        return {
            "source_state": self.source_state,
            "target_state": self.arc_def.target_state,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_time,
            "success_rate": (
                self.success_count / self.execution_count if self.execution_count > 0 else 0.0
            ),
        }
