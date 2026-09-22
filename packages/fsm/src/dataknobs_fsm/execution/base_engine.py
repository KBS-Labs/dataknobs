# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base execution engine with shared logic for sync and async engines.

This module provides a base class that contains common logic shared between
the synchronous (ExecutionEngine) and asynchronous (AsyncExecutionEngine)
implementations, reducing code duplication and ensuring feature parity.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from dataknobs_fsm.core.data_wrapper import StateDataWrapper, ensure_dict, wrap_for_lambda
from dataknobs_fsm.core.arc import ArcDefinition, PushArc
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import FunctionContext, normalize_record_callable
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelector,
    TransitionSelectionMode,
    TraversalStrategy,
)

logger = logging.getLogger(__name__)


class BaseExecutionEngine(ABC):
    """Base class for execution engines with shared logic.

    This class provides common functionality for both sync and async engines:
    - Initial state finding
    - Network selection
    - State transform preparation
    - Arc evaluation logic
    - Error handling patterns
    - Statistics tracking
    """

    def __init__(
        self,
        fsm: FSM,
        strategy: "TraversalStrategy",
        selection_mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize base execution engine.

        Args:
            fsm: FSM instance to execute.
            strategy: Traversal strategy to use.
            selection_mode: Transition selection mode.
            max_retries: Maximum retry attempts for failures.
            retry_delay: Delay between retries in seconds.
        """
        self.fsm = fsm
        self.strategy = strategy
        self.selection_mode = selection_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize transition selector
        self.transition_selector = TransitionSelector(
            mode=selection_mode, default_strategy=strategy
        )

        # Execution statistics
        self._execution_count = 0
        self._transition_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0

    def find_initial_state_common(self) -> str | None:
        """Find the initial state in the FSM (common logic).

        This method contains the shared logic for finding an initial state,
        used by both sync and async engines.

        Returns:
            Name of initial state or None.
        """
        # ``FSM.main_network`` is a ``StateNetwork | None`` and every value in
        # ``FSM.networks`` is a ``StateNetwork``, so this reads them as such.
        # It used to open with ``getattr(self.fsm, "main_network", None)``,
        # which discards that and returns ``Any`` --- and then branched on
        # ``isinstance(main_network, str)`` and ``hasattr(network,
        # "initial_states")``, neither of which can hold. Those branches were
        # the reason the type was thrown away and the only thing the loss of
        # it bought, so they go together: the ``Any`` they created is where
        # this method's and its callers' type findings came from.
        main_network = self.fsm.main_network
        if main_network is not None and main_network.initial_states:
            return next(iter(main_network.initial_states))

        # Fallback to fsm.name for compatibility
        named = self.fsm.networks.get(self.fsm.name)
        if named is not None and named.initial_states:
            return next(iter(named.initial_states))

        # Last resort: check all networks for any initial state
        for network in self.fsm.networks.values():
            if network.initial_states:
                return next(iter(network.initial_states))

        return None

    def is_final_state_common(self, state_name: str | None) -> bool:
        """Check if state is a final state (common logic).

        Args:
            state_name: Name of state to check.

        Returns:
            True if state is final.
        """
        if not state_name:
            return False

        # Check all networks for this state
        for network in self.fsm.networks.values():
            if hasattr(network, "final_states") and state_name in network.final_states:
                return True
            # Also check states directly
            if hasattr(network, "states") and state_name in network.states:
                state = network.states[state_name]
                if hasattr(state, "type") and state.type == StateType.END:
                    return True

        return False

    def get_current_network_common(self, context: ExecutionContext) -> StateNetwork | None:
        """Get current network using common selection logic.

        Args:
            context: Execution context.

        Returns:
            Current network or None.
        """
        return NetworkSelector.get_current_network(
            self.fsm, context, enable_intelligent_selection=True
        )

    def prepare_state_transform(
        self, state_def: Any, context: ExecutionContext
    ) -> Tuple[List[Any], StateDataWrapper]:
        """Prepare state transform execution (common logic).

        The second element is what ``wrap_for_lambda`` builds --- a
        :class:`~dataknobs_fsm.core.data_wrapper.StateDataWrapper` over the
        record, which is a mapping and reaches the record through ``.data``.
        It was annotated ``SimpleNamespace``, which this has never returned
        and which offers neither of those.

        Args:
            state_def: State definition.
            context: Execution context.

        Returns:
            Tuple of (transform functions, state object for inline lambdas).
        """
        transform_functions = []

        # Check for transform functions on the state
        if hasattr(state_def, "transform_functions") and state_def.transform_functions:
            transform_functions = state_def.transform_functions
        # Also check for single transform function
        elif hasattr(state_def, "transform_function") and state_def.transform_function:
            transform_functions = [state_def.transform_function]

        # Create a wrapper for transforms that expect state.data access pattern
        # This wrapper provides both dict and attribute access
        state_obj = wrap_for_lambda(context.data)

        return transform_functions, state_obj

    def process_transform_result(
        self, result: Any, context: ExecutionContext, state_name: str
    ) -> None:
        """Process transform result (common logic).

        Args:
            result: Result from transform function.
            context: Execution context.
            state_name: Name of current state.
        """
        if result is not None:
            # Handle ExecutionResult objects from unified function manager
            from dataknobs_fsm.functions.base import ExecutionResult

            if isinstance(result, ExecutionResult):
                if result.success:
                    # Ensure we store plain dict data
                    context.data = ensure_dict(result.data)
                else:
                    # Transform failed - handle the error
                    self.handle_transform_error(
                        Exception(result.error or "Transform failed"), context, state_name
                    )
            else:
                # Ensure we always store plain dict data, not wrappers
                context.data = ensure_dict(result)

    def handle_transform_error(
        self, error: Exception, context: ExecutionContext, state_name: str
    ) -> None:
        """Handle transform error (common logic).

        A failing state transform does not halt FSM traversal (the record still
        flows to a final state), but the failure is recorded in
        ``context.failed_states`` so :meth:`finalize_single_result` can surface
        it as a record-level failure rather than silently reporting success.

        ``error`` used to be accepted and dropped. This is the single sink
        every state-transform failure reaches --- the async engine's transform
        loop, ``AdvancedFSM``'s stepped runner, and
        :meth:`process_transform_result` when a transform *returns* a failed
        ``ExecutionResult`` --- so dropping it here meant the reason existed
        nowhere afterwards: not on the record, not in the result, not in a log.
        The caller was told the state and the state alone, which is the one
        part of it the caller already knew.

        The exception is kept on ``context.transform_errors``, keyed by state,
        and read back by :meth:`transform_failure_message`. The **first**
        exception for a state is the one kept: a later transform in a
        ``run_on_failure`` state fails because of the first, not alongside it.

        Args:
            error: Exception that occurred.
            context: Execution context.
            state_name: Name of current state.
        """
        if not hasattr(context, "failed_states"):
            context.failed_states = set()
        if not hasattr(context, "transform_errors"):
            context.transform_errors = {}
        context.failed_states.add(state_name)
        context.transform_errors.setdefault(state_name, error)
        logger.error("State transform failed in '%s': %s", state_name, error, exc_info=error)

    @staticmethod
    def describe_transform_error(error: Exception) -> str:
        """Name an exception the way a caller reading a result needs it.

        The type as well as the message, because a bare message is often
        ambiguous about what raised it and sometimes empty --- ``KeyError`` and
        ``ConnectionError`` are the two shapes this sees most, and one of them
        stringifies to nothing but a quoted key.

        Args:
            error: The exception a transform raised, or the one built from the
                message a transform returned.

        Returns:
            ``"TypeName: message"``, or just ``"TypeName"`` for an exception
            carrying no message.
        """
        message = str(error).strip()
        return f"{type(error).__name__}: {message}" if message else type(error).__name__

    def transform_failure_message(
        self, context: ExecutionContext, states: List[str] | None = None
    ) -> str:
        """The sentence every surface reports a transform failure with.

        One reading, because there were two: this method's caller
        :meth:`finalize_single_result` and ``AdvancedFSM._step_transform_failure``
        each built the string themselves, so a change to what a failure says
        had to be made twice or say two things.

        A state with no recorded reason is named alone, which is what a state
        added to ``failed_states`` by something other than
        :meth:`handle_transform_error` looks like.

        Args:
            context: Execution context for the in-flight record.
            states: The states to describe, defaulting to every state that
                failed for this record. A step passes the single state it
                entered, since that is the failure it is reporting.

        Returns:
            ``"State transform failed in: <state> (<Type>: <message>), ..."``
        """
        names = self.failed_states_sorted(context) if states is None else states
        errors: Dict[str, Exception] = getattr(context, "transform_errors", None) or {}
        described = [
            f"{name} ({self.describe_transform_error(errors[name])})" if name in errors else name
            for name in names
        ]
        return "State transform failed in: " + ", ".join(described)

    def record_has_failed(self, context: ExecutionContext) -> bool:
        """Whether a prior state transform already failed for this record.

        Once a state transform raises, :meth:`handle_transform_error` records
        the offending state in ``context.failed_states`` and the record's data
        is left in an indeterminate (pre-failure) state. Running *further* state
        transforms against it is unsafe — e.g. an ETL ``load`` step would upsert
        the stale, untransformed record into the target even though the run is
        (correctly) reporting the record as a failure.

        While this returns True, the transform guard gates **all** subsequent
        transforms for the record: not only downstream states, but also the
        remaining transforms of the *failing* state (the transform that raised
        flips this to True, so later transforms in the same state are skipped
        too). Only traversal (the record still reaching a final state, for
        accounting) continues, so :meth:`finalize_single_result` reports the
        failure rather than silently persisting corrupt data.

        The skip is overridable per state: a state declared with
        ``run_on_failure=True`` (recovery/compensation/cleanup/dead-letter
        states) still runs its transforms despite a prior failure — see
        :meth:`should_skip_state_transforms`.

        Args:
            context: Execution context for the in-flight record.

        Returns:
            True if any state has recorded a failure for this record.
        """
        return bool(getattr(context, "failed_states", None))

    def failed_states_sorted(self, context: ExecutionContext) -> List[str]:
        """Sorted list of states whose transform failed for this record.

        Centralizes the defensive ``sorted(getattr(...failed_states...))``
        idiom shared by the transform-skip log lines, the batch-result builder,
        and :meth:`finalize_single_result`.

        Args:
            context: Execution context for the in-flight record.

        Returns:
            Sorted list of failed state names (empty if none).
        """
        return sorted(getattr(context, "failed_states", None) or set())

    def should_skip_state_transforms(
        self,
        context: ExecutionContext,
        state_def: Any,
    ) -> bool:
        """Whether to skip a state's transforms because the record already failed.

        Returns True when a prior state transform failed for this record
        (:meth:`record_has_failed`) AND the state is **not** marked
        ``run_on_failure``. A state declared with ``run_on_failure=True`` is a
        recovery/compensation/cleanup/dead-letter state whose transforms must
        run despite the failure, so the guard never trips for it (its transforms
        run even when ``record_has_failed`` is True). For an ordinary state the
        guard trips, skipping its transforms so indeterminate data is not
        mutated or persisted.

        Called once per transform iteration (not once per state) so that, for an
        ordinary state, a transform that raises mid-state still causes the
        *remaining* transforms of that same state to be skipped — while a
        ``run_on_failure`` state runs all of its transforms regardless.

        Args:
            context: Execution context for the in-flight record.
            state_def: The definition of the state whose transforms are running.

        Returns:
            True if the state's (remaining) transforms should be skipped.
        """
        if not self.record_has_failed(context):
            return False
        return not getattr(state_def, "run_on_failure", False)

    def finalize_single_result(self, context: ExecutionContext) -> Tuple[bool, Any]:
        """Build the ``(success, value)`` result for a record at a final state.

        A record reaches a final state even when one of its state transforms
        raised: :meth:`handle_transform_error` records the offending state in
        ``context.failed_states`` but does not stop traversal. That failure
        signal MUST surface in the execution result — otherwise a swallowed
        transform or load error (e.g. a target-write failure in an ETL load
        step) is reported as a successful record, which is silent data loss.

        Returns ``(False, <message>)`` when any state recorded a failure during
        this record's execution, otherwise ``(True, context.data)``.

        Args:
            context: Execution context for the completed record.

        Returns:
            Tuple of (success, result value or failure message).
        """
        failed = self.failed_states_sorted(context)
        if failed:
            return False, self.transform_failure_message(context, failed)
        return True, context.data

    def apply_data_mapping(self, data: Any, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map a parent context's data into a child (sub-network) shape.

        Shared by both engines' push-arc handlers so parent→child field
        mapping cannot drift between the sync and async paths.

        Args:
            data: Source data (parent context data).
            mapping: Dict mapping ``parent_field -> child_field``.

        Returns:
            Mapped data dictionary for the child context. With no mapping the
            data passes through unchanged (wrapped as ``{'value': data}`` only
            when it is not already a dict).
        """
        if not mapping:
            return data if isinstance(data, dict) else {"value": data}

        mapped = {}
        source_data = data if isinstance(data, dict) else {}

        for parent_field, child_field in mapping.items():
            if parent_field in source_data:
                mapped[child_field] = source_data[parent_field]
            elif hasattr(data, parent_field):
                mapped[child_field] = getattr(data, parent_field)

        return mapped

    def apply_result_mapping(
        self, data: Any, mapping: Dict[str, str], parent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map a child (sub-network) result back onto the parent's data.

        Counterpart to :meth:`apply_data_mapping`, shared by both engines'
        subflow-pop handlers.

        Args:
            data: Source data (child context result).
            mapping: Dict mapping ``child_field -> parent_field``.
            parent_data: Parent context data to update.

        Returns:
            Updated parent data with the mapped results. With no mapping the
            child data passes through unchanged (when it is a dict), otherwise
            the parent data is returned untouched.
        """
        if not mapping:
            return data if isinstance(data, dict) else parent_data

        result = dict(parent_data) if parent_data else {}
        source_data = data if isinstance(data, dict) else {}

        for child_field, parent_field in mapping.items():
            if child_field in source_data:
                result[parent_field] = source_data[child_field]
            elif hasattr(data, child_field):
                result[parent_field] = getattr(data, child_field)

        return result

    # ------------------------------------------------------------------
    # Subflow (push-arc) lifecycle — shared, color-free building blocks.
    #
    # Both engines drive an identical push/pop subflow lifecycle; only the
    # state-entry step (sync ``enter_state`` vs the async one) and the hook
    # firing / ``await`` differ by async coloring. Everything that does *not*
    # depend on that coloring lives here, so the two engines' orchestrators are
    # thin and cannot drift on the parsing, depth check, target/initial-state
    # resolution, push commit, rollback, final-state detection, or result
    # mapping. (The orchestrators themselves stay per-engine because a method
    # cannot be both sync and ``async``.)
    # ------------------------------------------------------------------

    def subflow_depth_exceeded(
        self,
        context: ExecutionContext,
        push_arc: Any,
        max_subflow_depth: int,
    ) -> bool:
        """Whether pushing ``push_arc`` would exceed the nesting depth limit."""
        if len(context.network_stack) >= max_subflow_depth:
            logger.error(
                "Maximum subflow depth %d exceeded when pushing to network '%s'",
                max_subflow_depth,
                push_arc.target_network,
            )
            return True
        return False

    def parse_push_target(self, push_arc: PushArc) -> Tuple[str, str | None]:
        """Split a push arc's ``target_network`` into ``(network, initial?)``.

        Supports the ``"network"`` and ``"network:initial_state"`` forms. The
        rule lives on :meth:`PushArc.parse_target`, because the syntax belongs
        to the field; this stays as the engines' door onto it.

        Returns:
            ``(network_name, explicit_initial_state_or_None)``.
        """
        return push_arc.parse_target()

    def resolve_subflow_initial_state(
        self,
        target_network: StateNetwork,
        network_name: str,
        explicit_initial_state: str | None,
    ) -> str | None:
        """Resolve the sub-network's initial state, or ``None`` on a bad target.

        Resolution is done *before* the push is committed so the bad-target
        paths (unknown explicit state, no default initial state) fail cleanly
        without having mutated the context.

        Args:
            target_network: The resolved target network object.
            network_name: Name of the target network (for logging).
            explicit_initial_state: An explicit ``network:state`` override, or
                ``None`` to use the network's default initial state.

        Returns:
            The state name to enter, or ``None`` if it cannot be resolved.
        """
        if explicit_initial_state:
            if explicit_initial_state not in target_network.states:
                logger.error(
                    "Initial state '%s' not found in network '%s'",
                    explicit_initial_state,
                    network_name,
                )
                return None
            return explicit_initial_state
        if target_network.initial_states:
            return next(iter(target_network.initial_states))
        logger.error("No initial state in network '%s'", network_name)
        return None

    def prepare_subflow_input(self, push_arc: Any, data: Any) -> Any:
        """Apply the push arc's parent→child data mapping (pre-isolation).

        The isolation step (``isolation_mode.apply``) is applied by the caller
        so the async engine can offload its (potentially large) deepcopy /
        serialize off the event loop.
        """
        if push_arc.data_mapping:
            return self.apply_data_mapping(data, push_arc.data_mapping)
        return data

    @staticmethod
    def _state_resource_owner_for_name(context: ExecutionContext, state_name: str) -> str:
        """Owner key for a state's resource acquisitions, keyed by state *name*.

        The key is fully determined by the state name + execution id, so the
        release-on-exit path can rebuild it from ``context.current_state``
        without re-resolving the state definition (which may live in a
        sub-network the default ``get_state`` lookup would not find). Shared so
        a state's acquire and its later release cannot drift onto two formats.
        """
        execution_id = getattr(context, "execution_id", "unknown")
        return f"state_{state_name}_{execution_id}"

    def begin_subflow(
        self,
        context: ExecutionContext,
        push_arc: Any,
        network_name: str,
        parent_state_resources: Dict[str, Any],
        isolated_data: Any,
    ) -> None:
        """Commit a push: replace data, push the network, record the frame.

        Captures the parent's pre-push data object and prior
        ``parent_state_resources`` into a :class:`SubflowFrame` *before*
        overwriting them, so :meth:`rollback_push` (failed entry) and the pop
        (result mapping + resource restore) can undo/consume them precisely.
        """
        prev_parent_state_resources = context.parent_state_resources
        # The pushing state's own resources are inherited by the sub-network
        # while it runs (so the push must not release them); they are released
        # for the parent level on pop, when the parent resumes at return_state.
        # Record the pushing state's owner key (only when it owns resources) so
        # the pop can release exactly that state's acquisitions.
        #
        # ``current_state`` is named in the condition rather than assumed by
        # it. Resources are only ever acquired for a state that is current, so
        # a context owning some has one --- but that is an argument about the
        # caller, and the owner key is built *from* the name, so the name is
        # what the guard should be about.
        pushing_state_owner = (
            self._state_resource_owner_for_name(context, context.current_state)
            if context.current_state and context.current_state_owned_resources
            else None
        )
        parent_data = context.data
        context.data = isolated_data
        context.push_network(network_name, push_arc.return_state)
        context.push_subflow_frame(
            push_arc,
            parent_data,
            prev_parent_state_resources,
            pushing_state_owner,
        )
        # The sub-network's states inherit the pushing state's resources.
        context.parent_state_resources = parent_state_resources

    def rollback_push(self, context: ExecutionContext) -> None:
        """Undo a committed push whose initial-state entry failed.

        Pops the frame and network, restores the parent's data object and
        inherited-resource view. No result mapping is applied (the push did not
        complete).
        """
        frame = context.pop_subflow_frame()
        context.pop_network()
        if frame is not None:
            context.data = frame.parent_data
        self.restore_after_pop(context, frame)

    def subflow_at_final_state(self, context: ExecutionContext) -> bool:
        """Whether the current state is a final state of the top-of-stack network.

        ``is_final_state_common`` is global (a name match across *all* networks),
        so it cannot tell "the sub-network finished" from "the whole run
        finished". This network-scoped check is what the pop logic keys on.
        """
        if not context.network_stack:
            return False
        current_network_name = context.network_stack[-1][0]
        current_network = self.fsm.networks.get(current_network_name)
        if not current_network:
            logger.warning(
                "Network '%s' from stack not found in FSM",
                current_network_name,
            )
            return False
        if not context.current_state:
            return False
        return context.current_state in current_network.final_states

    def restore_after_pop(
        self,
        context: ExecutionContext,
        frame: Any,
    ) -> None:
        """Restore the parent level's inherited-resource view after a pop.

        Resets ``parent_state_resources`` to the value captured in the frame
        (the grandparent's resources, or ``{}`` at the top level) so a nested
        pop restores its own parent state rather than a single global slot.
        """
        if frame is not None:
            context.parent_state_resources = frame.prev_parent_state_resources or {}
        elif hasattr(context, "parent_state_resources"):
            context.parent_state_resources = {}

    def apply_subflow_result_mapping(
        self,
        context: ExecutionContext,
        frame: Any,
    ) -> None:
        """Apply the popped frame's push-arc ``result_mapping`` onto the data.

        Maps the completed sub-network's result fields back onto the parent's
        pre-push data (``frame.parent_data``). With no frame or no
        ``result_mapping`` the data is left as-is (the sub-network's result
        flows straight through, the historical default).
        """
        if frame is None:
            return
        push_arc = frame.push_arc
        if push_arc is not None and getattr(push_arc, "result_mapping", None):
            context.data = self.apply_result_mapping(
                context.data,
                push_arc.result_mapping,
                frame.parent_data,
            )

    def evaluate_arc_condition_common(self, arc: ArcDefinition, context: ExecutionContext) -> bool:
        """Evaluate arc condition (common logic).

        Args:
            arc: Arc definition.
            context: Execution context.

        Returns:
            True if arc condition is met.
        """
        # If arc has no condition, it's always valid
        if not hasattr(arc, "condition") or not arc.condition:
            return True

        # Evaluate the condition function
        try:
            # Create function context
            func_context = FunctionContext(
                state_name=context.current_state or "",
                function_name="arc_condition",
                metadata={"arc": arc.name if hasattr(arc, "name") else None},
                resources={},
            )

            # Arity- and await-normalized through the one reading this
            # package has, rather than discovered by calling and catching.
            # This was `try (data, context) / except TypeError: (data)`, and
            # an `except TypeError` cannot tell a failed argument binding
            # from a failed body -- so a condition with a bug in it ran
            # twice, and the enclosing `except Exception` below then turned
            # the second failure into a quiet "no", making a broken condition
            # read as a condition that declined the arc.
            condition = normalize_record_callable(arc.condition, coerce=bool)
            return bool(condition(context.data, func_context))
        except Exception as exc:
            # Condition evaluation failed - arc is not valid. Which reads, from
            # everywhere downstream, as a condition that declined the arc: the
            # record is routed as if the answer were a considered "no" and the
            # exception that produced it is gone. The outcome is unchanged
            # here -- see AsyncExecutionEngine._evaluate_arc_pre_test, which
            # raises instead on the argument that an outage must not be
            # reported as a data-quality drop; the two disagree, and
            # reconciling them changes routing rather than reporting.
            logger.warning(
                "Arc condition raised and the arc is treated as declined (arc %s -> %s): %s",
                getattr(arc, "name", "?"),
                getattr(arc, "target_state", "?"),
                exc,
                exc_info=exc,
            )
            return False

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics (common implementation).

        Returns:
            Dictionary of execution statistics.
        """
        return {
            "execution_count": self._execution_count,
            "transition_count": self._transition_count,
            "error_count": self._error_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0
                else 0
            ),
        }

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000,
        arc_name: str | None = None,
    ) -> Tuple[bool, Any]:
        """Execute the FSM with given context.

        Declared ``async`` because the one engine there is, is. It was
        declared synchronous here while ``AsyncExecutionEngine.execute`` ---
        the only implementation in the package --- is a coroutine function,
        so the base described a signature nothing implemented and the
        subclass was reported as violating it. A synchronous caller reaches
        this through ``FSM.get_sync_bridge()``, not through a second engine.

        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            arc_name: Optional specific arc name to follow.

        Returns:
            Tuple of (success, result).
        """
        pass

    @abstractmethod
    def _execute_single(
        self, context: ExecutionContext, max_transitions: int, arc_name: str | None = None
    ) -> Any:
        """Execute single mode processing.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _execute_batch(self, context: ExecutionContext, max_transitions: int) -> Any:
        """Execute batch mode processing.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _execute_stream(self, context: ExecutionContext, max_transitions: int) -> Any:
        """Execute stream mode processing.

        Must be implemented by subclasses.
        """
        pass
