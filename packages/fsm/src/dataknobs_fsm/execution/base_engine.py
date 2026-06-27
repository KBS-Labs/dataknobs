"""Base execution engine with shared logic for sync and async engines.

This module provides a base class that contains common logic shared between
the synchronous (ExecutionEngine) and asynchronous (AsyncExecutionEngine)
implementations, reducing code duplication and ensuring feature parity.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from types import SimpleNamespace

from dataknobs_fsm.core.data_wrapper import (
    ensure_dict,
    wrap_for_lambda
)
from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import FunctionContext
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelector,
    TransitionSelectionMode
)

if TYPE_CHECKING:
    from dataknobs_fsm.execution.engine import TraversalStrategy

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
        strategy: 'TraversalStrategy',
        selection_mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID,
        max_retries: int = 3,
        retry_delay: float = 1.0
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
            mode=selection_mode,
            default_strategy=strategy
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
        # Try to get main_network attribute
        main_network = getattr(self.fsm, 'main_network', None)

        # Handle string reference to network
        if isinstance(main_network, str):
            if main_network in self.fsm.networks:
                network = self.fsm.networks[main_network]
                if hasattr(network, 'initial_states') and network.initial_states:
                    return next(iter(network.initial_states))
        # Handle direct network object
        elif main_network and hasattr(main_network, 'initial_states'):
            if main_network.initial_states:
                return next(iter(main_network.initial_states))

        # Fallback to fsm.name for compatibility
        if hasattr(self.fsm, 'name') and self.fsm.name in self.fsm.networks:
            network = self.fsm.networks[self.fsm.name]
            if hasattr(network, 'initial_states') and network.initial_states:
                return next(iter(network.initial_states))

        # Last resort: check all networks for any initial state
        for network in self.fsm.networks.values():
            if hasattr(network, 'initial_states') and network.initial_states:
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
            if hasattr(network, 'final_states') and state_name in network.final_states:
                return True
            # Also check states directly
            if hasattr(network, 'states') and state_name in network.states:
                state = network.states[state_name]
                if hasattr(state, 'type') and state.type == StateType.END:
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
            self.fsm,
            context,
            enable_intelligent_selection=True
        )

    def prepare_state_transform(
        self,
        state_def: Any,
        context: ExecutionContext
    ) -> Tuple[List[Any], SimpleNamespace]:
        """Prepare state transform execution (common logic).

        Args:
            state_def: State definition.
            context: Execution context.

        Returns:
            Tuple of (transform functions, state object for inline lambdas).
        """
        transform_functions = []

        # Check for transform functions on the state
        if hasattr(state_def, 'transform_functions') and state_def.transform_functions:
            transform_functions = state_def.transform_functions
        # Also check for single transform function
        elif hasattr(state_def, 'transform_function') and state_def.transform_function:
            transform_functions = [state_def.transform_function]

        # Create a wrapper for transforms that expect state.data access pattern
        # This wrapper provides both dict and attribute access
        state_obj = wrap_for_lambda(context.data)

        return transform_functions, state_obj

    def process_transform_result(
        self,
        result: Any,
        context: ExecutionContext,
        state_name: str
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
                        Exception(result.error or "Transform failed"),
                        context,
                        state_name
                    )
            else:
                # Ensure we always store plain dict data, not wrappers
                context.data = ensure_dict(result)

    def handle_transform_error(
        self,
        error: Exception,
        context: ExecutionContext,
        state_name: str
    ) -> None:
        """Handle transform error (common logic).

        A failing state transform does not halt FSM traversal (the record still
        flows to a final state), but the failure is recorded in
        ``context.failed_states`` so :meth:`finalize_single_result` can surface
        it as a record-level failure rather than silently reporting success.

        Args:
            error: Exception that occurred.
            context: Execution context.
            state_name: Name of current state.
        """
        if not hasattr(context, 'failed_states'):
            context.failed_states = set()
        context.failed_states.add(state_name)

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
        return bool(getattr(context, 'failed_states', None))

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
        return sorted(getattr(context, 'failed_states', None) or set())

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
        return not getattr(state_def, 'run_on_failure', False)

    def finalize_single_result(
        self,
        context: ExecutionContext
    ) -> Tuple[bool, Any]:
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
            return False, (
                "State transform failed in: " + ", ".join(failed)
            )
        return True, context.data

    def apply_data_mapping(
        self,
        data: Any,
        mapping: Dict[str, str]
    ) -> Dict[str, Any]:
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
            return data if isinstance(data, dict) else {'value': data}

        mapped = {}
        source_data = data if isinstance(data, dict) else {}

        for parent_field, child_field in mapping.items():
            if parent_field in source_data:
                mapped[child_field] = source_data[parent_field]
            elif hasattr(data, parent_field):
                mapped[child_field] = getattr(data, parent_field)

        return mapped

    def apply_result_mapping(
        self,
        data: Any,
        mapping: Dict[str, str],
        parent_data: Dict[str, Any]
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

    def parse_push_target(self, push_arc: Any) -> Tuple[str, str | None]:
        """Split a push arc's ``target_network`` into ``(network, initial?)``.

        Supports the ``"network"`` and ``"network:initial_state"`` forms.

        Returns:
            ``(network_name, explicit_initial_state_or_None)``.
        """
        target = push_arc.target_network
        if ':' in target:
            network_name, initial_state = target.split(':', 1)
            return network_name, initial_state.strip()
        return target, None

    def resolve_subflow_initial_state(
        self,
        target_network: Any,
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
        prev_parent_state_resources = getattr(
            context, 'parent_state_resources', {}
        )
        parent_data = context.data
        context.data = isolated_data
        context.push_network(network_name, push_arc.return_state)
        context.push_subflow_frame(
            push_arc, parent_data, prev_parent_state_resources
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
        elif hasattr(context, 'parent_state_resources'):
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
        if push_arc is not None and getattr(push_arc, 'result_mapping', None):
            context.data = self.apply_result_mapping(
                context.data,
                push_arc.result_mapping,
                frame.parent_data,
            )

    def evaluate_arc_condition_common(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Evaluate arc condition (common logic).

        Args:
            arc: Arc definition.
            context: Execution context.

        Returns:
            True if arc condition is met.
        """
        # If arc has no condition, it's always valid
        if not hasattr(arc, 'condition') or not arc.condition:
            return True

        # Evaluate the condition function
        try:
            # Create function context
            func_context = FunctionContext(
                state_name=context.current_state or "",
                function_name="arc_condition",
                metadata={'arc': arc.name if hasattr(arc, 'name') else None},
                resources={}
            )

            # Try different function signatures
            try:
                # Try with data and context
                return bool(arc.condition(context.data, func_context))
            except TypeError:
                # Try with just data
                return bool(arc.condition(context.data))
        except Exception:
            # Condition evaluation failed - arc is not valid
            return False

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics (common implementation).

        Returns:
            Dictionary of execution statistics.
        """
        return {
            'execution_count': self._execution_count,
            'transition_count': self._transition_count,
            'error_count': self._error_count,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0
            )
        }

    @abstractmethod
    def execute(self, context: ExecutionContext, data: Any = None,
                max_transitions: int = 1000, arc_name: str | None = None) -> Tuple[bool, Any]:
        """Execute the FSM with given context.

        This method must be implemented by sync and async engines.

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
    def _execute_single(self, context: ExecutionContext,
                       max_transitions: int, arc_name: str | None = None) -> Any:
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
