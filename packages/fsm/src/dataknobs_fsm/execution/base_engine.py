"""Base execution engine with shared logic for sync and async engines.

This module provides a base class that contains common logic shared between
the synchronous (ExecutionEngine) and asynchronous (AsyncExecutionEngine)
implementations, reducing code duplication and ensuring feature parity.
"""

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

        State transforms failing doesn't stop the FSM, but marks the state as failed.

        Args:
            error: Exception that occurred.
            context: Execution context.
            state_name: Name of current state.
        """
        if not hasattr(context, 'failed_states'):
            context.failed_states = set()
        context.failed_states.add(state_name)

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
