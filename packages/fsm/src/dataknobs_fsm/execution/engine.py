"""Execution engine for FSM state machines."""

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import FunctionContext
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelectionMode
)
from dataknobs_fsm.execution.base_engine import BaseExecutionEngine
from dataknobs_fsm.core.data_wrapper import ensure_dict

logger = logging.getLogger(__name__)


class TraversalStrategy(Enum):
    """Execution traversal strategy."""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    RESOURCE_OPTIMIZED = "resource_optimized"
    STREAM_OPTIMIZED = "stream_optimized"


class ExecutionEngine(BaseExecutionEngine):
    """Execution engine for FSM state machines.
    
    This engine handles:
    - Mode-aware execution (single, batch, stream)
    - Resource-aware scheduling
    - Stream processing support
    - State transitions with validation
    - Branching and parallel paths
    - Network push/pop operations
    - Error handling with retry
    - Execution hooks and callbacks
    """
    
    def __init__(
        self,
        fsm: FSM,
        strategy: TraversalStrategy = TraversalStrategy.DEPTH_FIRST,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_hooks: bool = True,
        selection_mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID
    ):
        """Initialize execution engine.

        Args:
            fsm: FSM instance to execute.
            strategy: Traversal strategy to use.
            max_retries: Maximum retry attempts for failures.
            retry_delay: Delay between retries in seconds.
            enable_hooks: Enable execution hooks.
            selection_mode: Transition selection mode (strategy, scoring, or hybrid).
        """
        # Initialize base class
        super().__init__(fsm, strategy, selection_mode, max_retries, retry_delay)

        self.enable_hooks = enable_hooks

        # Hooks (specific to sync engine)
        self._pre_transition_hooks: List[Callable] = []
        self._post_transition_hooks: List[Callable] = []
        self._error_hooks: List[Callable] = []
    
    def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute the FSM with given context.
        
        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        self._execution_count += 1
        # Only override context.data if data was explicitly provided
        if data is not None:
            context.data = data

        # Ensure context has resource_manager from FSM
        if context.resource_manager is None and self.fsm.resource_manager is not None:
            context.resource_manager = self.fsm.resource_manager

        # Initialize state if needed
        if not context.current_state:
            initial_state = self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"

            # Use the common state entry method
            if not self.enter_state(context, initial_state):
                # Return specific error if available, otherwise generic message
                error_msg = getattr(context, 'last_error', "Failed to enter initial state")
                return False, error_msg
        
        # Execute based on data mode
        if context.data_mode == ProcessingMode.SINGLE:
            return self._execute_single(context, max_transitions, arc_name)
        elif context.data_mode == ProcessingMode.BATCH:
            return self._execute_batch(context, max_transitions)
        elif context.data_mode == ProcessingMode.STREAM:
            return self._execute_stream(context, max_transitions)
        else:
            return False, f"Unknown data mode: {context.data_mode}"
    
    def _execute_single(
        self,
        context: ExecutionContext,
        max_transitions: int,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute in single record mode.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        transitions = 0
        last_state = None
        last_data_hash = None
        stuck_count = 0
        max_stuck_iterations = 3  # Max times we can be in same state without data changes
        
        while transitions < max_transitions:
            # Check if in final state
            if self._is_final_state(context.current_state):
                return True, context.data
            
            # Check for stuck state (infinite loop protection)
            # Only consider it stuck if both state AND data haven't changed
            from dataknobs_fsm.utils.json_encoder import dumps
            current_data_hash = dumps(context.data, sort_keys=True) if context.data else ""
            
            if context.current_state == last_state and current_data_hash == last_data_hash:
                stuck_count += 1
                if stuck_count >= max_stuck_iterations:
                    return False, f"Stuck in state '{context.current_state}' - possible infinite loop"
            else:
                stuck_count = 0
                last_state = context.current_state
                last_data_hash = current_data_hash
            
            # Execute state functions (validators and transforms) before evaluating transitions
            # This ensures that state functions can update the data that arc conditions depend on
            self._execute_state_functions(context, context.current_state)
            
            # Get available transitions
            transitions_available = self._get_available_transitions(context, arc_name)
            
            if not transitions_available:
                # No valid transitions - check if this is a final state
                if self._is_final_state(context.current_state):
                    return True, context.data
                return False, f"No valid transitions from state: {context.current_state}"
            
            # Choose transition based on strategy
            next_transition = self._choose_transition(
                transitions_available,
                context
            )
            
            # Store current state before transition
            state_before = context.current_state
            
            # Execute transition
            success = self._execute_transition(
                context,
                next_transition
            )
            
            if not success:
                return False, f"Transition failed: {next_transition}"
            
            # Only increment if we actually transitioned
            if context.current_state != state_before:
                transitions += 1
        
        return False, f"Maximum transitions ({max_transitions}) exceeded"
    
    def _execute_batch(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in batch mode.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions per item.
            
        Returns:
            Tuple of (success, results).
        """
        if not context.batch_data:
            return False, "No batch data to process"
        
        total_success = True
        
        for i, item in enumerate(context.batch_data):
            # Create child context for this item
            item_context = context.create_child_context(f"batch_{i}")
            item_context.data = item
            
            # Reset to initial state for each item
            initial_state = self._find_initial_state()
            if initial_state:
                # Use the common state entry method
                self.enter_state(item_context, initial_state, run_validators=False)
            
            # Execute for this item
            success, result = self._execute_single(
                item_context,
                max_transitions
            )
            
            if success:
                context.add_batch_result(result)
            else:
                context.add_batch_error(i, Exception(result))
                if context.transaction_mode == TransactionMode.PER_RECORD:
                    # Continue processing other items
                    pass
                elif context.transaction_mode == TransactionMode.PER_BATCH:
                    # Fail entire batch
                    total_success = False
                    break
            
            # Merge child context
            context.merge_child_context(f"batch_{i}")
        
        return total_success, {
            'results': context.batch_results,
            'errors': context.batch_errors
        }
    
    def _execute_stream(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in stream mode.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions per chunk.
            
        Returns:
            Tuple of (success, stream_stats).
        """
        if not context.stream_context:
            return False, "No stream context provided"
        
        chunks_processed = 0
        total_records = 0
        errors = []
        
        # Process each chunk
        while True:
            # Get next chunk from stream
            chunk = context.stream_context.get_next_chunk()
            if not chunk:
                break
            
            context.set_stream_chunk(chunk)
            
            # Process chunk data
            for record in chunk.data:
                record_context = context.create_child_context(
                    f"stream_{chunks_processed}_{total_records}"
                )
                record_context.data = record
                
                # Reset to initial state
                initial_state = self._find_initial_state()
                if initial_state:
                    # Use the common state entry method
                    self.enter_state(record_context, initial_state, run_validators=False)
                
                # Execute for this record
                success, result = self._execute_single(
                    record_context,
                    max_transitions
                )
                
                if not success:
                    errors.append((total_records, result))
                
                # Merge context
                context.merge_child_context(
                    f"stream_{chunks_processed}_{total_records}"
                )
                
                total_records += 1
            
            chunks_processed += 1
            
            # Check if this was the last chunk
            if chunk.is_last:
                break
        
        return len(errors) == 0, {
            'chunks_processed': chunks_processed,
            'records_processed': total_records,
            'errors': errors
        }
    
    def enter_state(
        self,
        context: ExecutionContext,
        state_name: str,
        run_validators: bool = True
    ) -> bool:
        """Public method to handle entering a state with all necessary setup.

        This method handles the complete state entry process including:
        - Setting the current state
        - Allocating resources
        - Running pre-validators (optional)
        - Executing transforms
        - Setting up state tracking

        Args:
            context: Execution context.
            state_name: Name of the state to enter.
            run_validators: Whether to run pre-validators (default True).

        Returns:
            True if state entry was successful, False otherwise.
        """
        # Set the current state
        context.set_state(state_name)

        # Allocate state resources
        state_resources = self._allocate_state_resources(context, state_name)

        # Store in context for cleanup tracking
        context.current_state_resources = state_resources

        # Execute pre-validators if requested
        if run_validators:
            if not self._execute_pre_validators(context, state_name, state_resources):
                # Clean up resources if validation fails
                self._release_state_resources(context, state_name, state_resources)
                # Store error information in context for better error reporting
                if not hasattr(context, 'last_error'):
                    context.last_error = f"Pre-validation failed for state '{state_name}'"
                return False

        # Execute state transforms
        self._execute_state_transforms(context, state_name, state_resources)

        return True

    def exit_state(
        self,
        context: ExecutionContext,
        state_name: str
    ) -> None:
        """Public method to handle exiting a state with cleanup.

        This method handles:
        - Releasing state resources
        - Any other cleanup needed when leaving a state

        Args:
            context: Execution context.
            state_name: Name of the state being exited.
        """
        if hasattr(context, 'current_state_resources') and context.current_state_resources:
            self._release_state_resources(context, state_name, context.current_state_resources)
            context.current_state_resources = {}

    def _execute_transition(
        self,
        context: ExecutionContext,
        arc: ArcDefinition
    ) -> bool:
        """Execute a single transition.
        
        Args:
            context: Execution context.
            arc: Arc to execute.
            
        Returns:
            True if successful.
        """
        # Fire pre-transition hooks
        if self.enable_hooks:
            for hook in self._pre_transition_hooks:
                hook(context, arc)
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # Validate data before processing
                if context.data is None:
                    # Skip processing for None data
                    return False
                
                # Create arc execution (pass current state as source)
                arc_exec = ArcExecution(
                    arc,
                    source_state=context.current_state or "",
                    function_registry=self.fsm.function_registry
                )
                
                # Execute with resource context
                result = arc_exec.execute(context, context.data)

                # If no exception was thrown, the arc execution succeeded
                # Update data with result
                if result is not None:
                    context.data = result
                
                # Use the common state entry method
                if not self.enter_state(context, arc.target_state):
                    return False

                self._transition_count += 1
                
                # Fire post-transition hooks
                if self.enable_hooks:
                    for hook in self._post_transition_hooks:
                        hook(context, arc)
                
                return True
                
            except (TypeError, AttributeError, ValueError, SyntaxError) as e:
                # Deterministic errors (code errors, type errors) - don't retry
                self._error_count += 1
                
                # Fire error hooks
                if self.enable_hooks:
                    for hook in self._error_hooks:
                        hook(context, arc, e)
                
                # Return false immediately for deterministic errors
                return False
                
            except FunctionError as e:
                # Arc transform or pre-test failed - this is a definitive failure
                self._error_count += 1
                
                # Fire error hooks
                if self.enable_hooks:
                    for hook in self._error_hooks:
                        hook(context, arc, e)
                
                # Arc failed, no retry for function errors
                return False
                
            except Exception as e:
                # Other exceptions - may be recoverable (network, resources, etc.)
                self._error_count += 1
                
                # Fire error hooks
                if self.enable_hooks:
                    for hook in self._error_hooks:
                        hook(context, arc, e)
                
                # Only retry for potentially recoverable errors
                retry_count += 1
                if retry_count <= self.max_retries:
                    time.sleep(self.retry_delay * retry_count)
                else:
                    # Don't raise, just return False to allow graceful failure
                    return False
        
        return False
    
    def _execute_pre_validators(
        self,
        context: ExecutionContext,
        state_name: str,
        state_resources: Dict[str, Any] | None = None
    ) -> bool:
        """Execute pre-validation functions when entering a state.

        Args:
            context: Execution context.
            state_name: Name of the state.
            state_resources: Already allocated state resources.

        Returns:
            True if validation passes, False otherwise.
        """
        state_def = self.fsm.get_state(state_name)
        if not state_def:
            return True

        # Use provided resources or empty dict
        resources = state_resources if state_resources is not None else {}

        if hasattr(state_def, 'pre_validation_functions') and state_def.pre_validation_functions:
            for validator_func in state_def.pre_validation_functions:
                try:
                    # Execute validator with state resources
                    func_context = FunctionContext(
                        state_name=state_name,
                        function_name=getattr(validator_func, '__name__', 'validate'),
                        metadata={'state': state_name, 'phase': 'pre_validation'},
                        resources=resources,  # Pass state resources
                        variables=context.variables  # Pass shared variables
                    )
                    result = validator_func(ensure_dict(context.data), func_context)

                    if result is False:
                        return False
                    # Update context.data if result is a dict
                    if isinstance(result, dict):
                        context.data.update(result)
                except Exception:
                    # Log error and fail validation
                    return False
        return True

    def _allocate_state_resources(
        self,
        context: ExecutionContext,
        state_name: str
    ) -> Dict[str, Any]:
        """Allocate resources required by a state.

        Args:
            context: Execution context.
            state_name: Name of the state.

        Returns:
            Dictionary of allocated resources.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"_allocate_state_resources called for state: {state_name}")

        # Start with parent state resources if in subnetwork
        parent_resources = getattr(context, 'parent_state_resources', None)
        if parent_resources:
            resources = dict(parent_resources)  # Copy parent resources
            logger.debug(f"Starting with parent resources: {list(parent_resources.keys())}")
        else:
            resources = {}
            logger.debug("No parent resources")

        state_def = self.fsm.get_state(state_name)
        logger.debug(f"State def found: {state_def is not None}")
        if state_def:
            logger.debug(f"State has resource_requirements: {state_def.resource_requirements if hasattr(state_def, 'resource_requirements') else 'NO ATTRIBUTE'}")

        if not state_def or not state_def.resource_requirements:
            logger.debug("Returning early - no state def or no requirements")
            return resources

        resource_manager = getattr(context, 'resource_manager', None)
        logger.debug(f"Resource manager available: {resource_manager is not None}")
        if not resource_manager:
            logger.debug("No resource manager - returning empty")
            return resources

        # Generate owner ID for state resource allocation
        owner_id = f"state_{state_name}_{getattr(context, 'execution_id', 'unknown')}"

        for resource_config in state_def.resource_requirements:
            # Skip if resource already inherited from parent
            if resource_config.name in resources:
                logger.info(f"Resource '{resource_config.name}' inherited from parent state")
                continue

            try:
                # ResourceConfig from schema has timeout_seconds, not timeout
                timeout = getattr(resource_config, 'timeout_seconds', 30)
                resource = resource_manager.acquire(
                    name=resource_config.name,
                    owner_id=owner_id,
                    timeout=timeout
                )
                resources[resource_config.name] = resource
            except Exception as e:
                # Log error but continue with other resources
                logger.error(f"Failed to acquire resource {resource_config.name}: {e}")

        return resources

    def _release_state_resources(
        self,
        context: ExecutionContext,
        state_name: str,
        resources: Dict[str, Any]
    ) -> None:
        """Release state-allocated resources.

        Args:
            context: Execution context.
            state_name: Name of the state.
            resources: Resources to release.
        """
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return

        # Get parent resources to avoid releasing them
        parent_resources = getattr(context, 'parent_state_resources', {})

        owner_id = f"state_{state_name}_{getattr(context, 'execution_id', 'unknown')}"

        for resource_name in resources.keys():
            # Skip releasing if this is a parent-inherited resource
            if resource_name in parent_resources:
                logger.info(f"Skipping release of inherited resource '{resource_name}'")
                continue

            try:
                resource_manager.release(resource_name, owner_id)
            except Exception as e:
                # Log error but continue
                logger.error(f"Failed to release resource {resource_name}: {e}")

    def _execute_state_transforms(
        self,
        context: ExecutionContext,
        state_name: str,
        state_resources: Dict[str, Any] | None = None
    ) -> None:
        """Execute transform functions when entering a state.

        Args:
            context: Execution context.
            state_name: Name of the state being entered.
            state_resources: Already allocated state resources.
        """
        # Get the state definition
        state_def = self.fsm.get_state(state_name)
        if not state_def:
            return

        # Use provided resources or empty dict
        resources = state_resources if state_resources is not None else {}

        # Use base class logic to prepare and execute transforms
        transform_functions, state_obj = self.prepare_state_transform(state_def, context)

        for transform_func in transform_functions:
            try:
                # Create function context with resources and variables
                func_context = FunctionContext(
                    state_name=state_name,
                    function_name=getattr(transform_func, '__name__', 'transform'),
                    metadata={'state': state_name},
                    resources=resources,  # Pass state resources
                    variables=context.variables  # Pass shared variables
                )

                # Add parent_state_resources to metadata if available
                if hasattr(context, 'parent_state_resources') and context.parent_state_resources:
                    func_context.metadata['parent_state_resources'] = context.parent_state_resources

                # Try calling with state object first (for inline lambdas)
                try:
                    result = transform_func(state_obj)
                except (TypeError, AttributeError):
                    # Fall back to calling with data and context
                    result = transform_func(ensure_dict(context.data), func_context)

                # Process result using base class logic
                self.process_transform_result(result, context, state_name)

            except Exception as e:
                # Handle error using base class logic
                self.handle_transform_error(e, context, state_name)
    
    def _execute_state_functions(
        self,
        context: ExecutionContext,
        state_name: str
    ) -> None:
        """Execute all state functions (validators and transforms) for a state.
        
        This should be called before evaluating arc conditions to ensure
        that state functions can update the data that conditions depend on.
        
        Args:
            context: Execution context.
            state_name: Name of the current state.
        """
        # Get the state definition
        state_def = self.fsm.get_state(state_name)
        if not state_def:
            return
        
        # Execute validation functions first
        if hasattr(state_def, 'validation_functions') and state_def.validation_functions:
            for validator_func in state_def.validation_functions:
                try:
                    # Create function context with variables
                    func_context = FunctionContext(
                        state_name=state_name,
                        function_name=getattr(validator_func, '__name__', 'validate'),
                        metadata={'state': state_name},
                        resources={},
                        variables=context.variables  # Pass shared variables
                    )
                    
                    # Execute the validator
                    # Validators typically return a dict with validation results
                    # Create a wrapper for validators that expect state.data
                    from dataknobs_fsm.core.data_wrapper import wrap_for_lambda
                    state_obj = wrap_for_lambda(context.data)

                    # Try calling with state object first (for inline lambdas)
                    try:
                        result = validator_func(state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to calling with data and context
                        result = validator_func(ensure_dict(context.data), func_context)
                    
                    if result is not None:
                        # Ensure result is a dict and merge into context data
                        result_dict = ensure_dict(result)
                        if isinstance(result_dict, dict):
                            context.data.update(result_dict)
                except Exception:
                    # Log but don't fail - state validators are optional
                    pass
        
        # NOTE: Transform functions are NOT executed here. They are executed
        # by _execute_state_transforms when entering a state after a transition.
        # This method (_execute_state_functions) only executes validators before
        # evaluating transition conditions.
    
    def _get_available_transitions(
        self,
        context: ExecutionContext,
        arc_name: str | None = None
    ) -> List[ArcDefinition]:
        """Get available transitions from current state.
        
        Args:
            context: Execution context.
            arc_name: Optional specific arc name to filter by.
            
        Returns:
            List of available arc definitions.
        """
        if not context.current_state:
            return []
        
        # Get current network
        network = self._get_current_network(context)
        if not network:
            return []
        
        # Get arcs from current state
        available = []
        for arc_id, arc in network.arcs.items():
            # Parse arc_id to get source state
            if ':' in arc_id:
                source_state = arc_id.split(':')[0]
                if source_state == context.current_state:
                    # Filter by arc name if specified
                    if arc_name:
                        arc_actual_name = arc.metadata.get('name')
                        if arc_actual_name != arc_name:
                            continue
                    
                    # Check if pre-test passes
                    if arc.pre_test:
                        if not self._evaluate_pre_test(arc, context):
                            continue
                    available.append(arc)
        
        # Sort by priority
        # Sort by priority (descending) then by definition order (ascending)
        # This ensures stable ordering when priorities are equal
        available.sort(key=lambda x: (-x.priority, x.definition_order))
        
        return available
    
    def _evaluate_pre_test(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Evaluate arc pre-test.
        
        Args:
            arc: Arc with pre-test.
            context: Execution context.
            
        Returns:
            True if pre-test passes.
        """
        if not arc.pre_test:
            return True
        
        # Get pre-test function
        func = self.fsm.function_registry.get_function(arc.pre_test)
        if not func:
            return False
        
        # Create function context
        func_context = FunctionContext(
            state_name=context.current_state or "",
            function_name=arc.pre_test,
            metadata=context.metadata
        )
        
        try:
            # Call the function appropriately based on its interface
            if hasattr(func, 'test'):
                # IStateTestFunction - create state-like object
                from types import SimpleNamespace
                state = SimpleNamespace(data=context.data)
                result = func.test(state)
            elif hasattr(func, 'execute'):
                # Legacy function with execute method
                result = func.execute(context.data, func_context)
            elif callable(func):
                # Direct callable
                result = func(context.data, func_context)
            else:
                return False
            # Handle tuple return from test functions (bool, reason)
            if isinstance(result, tuple):
                return bool(result[0])
            return bool(result)
        except Exception:
            return False
    
    def _choose_transition(
        self,
        transitions: List[ArcDefinition],
        context: ExecutionContext
    ) -> ArcDefinition | None:
        """Choose next transition using common transition selector.
        
        Args:
            transitions: Available transitions.
            context: Execution context.
            
        Returns:
            Chosen arc or None.
        """
        return self.transition_selector.select_transition(
            transitions,
            context,
            strategy=self.strategy
        )
    
    def _find_initial_state(self) -> str | None:
        """Find the initial state in the FSM.

        Returns:
            Name of initial state or None.
        """
        # Use base class implementation
        return self.find_initial_state_common()
    
    def _is_final_state(self, state_name: str | None) -> bool:
        """Check if state is a final state.

        Args:
            state_name: Name of state to check.

        Returns:
            True if final state.
        """
        # Use base class implementation
        return self.is_final_state_common(state_name)

    def _is_final_state_legacy(self, state_name: str | None) -> bool:
        """Legacy implementation kept for reference."""
        if not state_name:
            return False

        # Get the main network - could be a string or object
        main_network_ref = getattr(self.fsm, 'main_network', None)
        
        if main_network_ref is None:
            # If no main network specified, check all networks
            for network in self.fsm.networks.values():
                if state_name in network.states:
                    state = network.states[state_name]
                    if state.is_end_state() if hasattr(state, 'is_end_state') else state.type == StateType.END:
                        return True
            return False
        
        # Handle case where main_network is already a network object (FSM wrapper)
        if hasattr(main_network_ref, 'states'):
            main_network = main_network_ref
        # Handle case where main_network is a string (core FSM)
        elif isinstance(main_network_ref, str) and main_network_ref in self.fsm.networks:
            main_network = self.fsm.networks[main_network_ref]
        else:
            return False
        
        # Check if the state exists and is an end state
        if state_name in main_network.states:
            state = main_network.states[state_name]
            return state.is_end_state() if hasattr(state, 'is_end_state') else state.type == StateType.END
        
        return False
    
    def _get_current_network(
        self,
        context: ExecutionContext
    ) -> StateNetwork | None:
        """Get the current network from context using common network selector.
        
        Args:
            context: Execution context.
            
        Returns:
            Current network or None.
        """
        # Allow intelligent selection to be controlled by selection_mode
        enable_intelligent = self.selection_mode != TransitionSelectionMode.STRATEGY_BASED
        
        return NetworkSelector.get_current_network(
            self.fsm,
            context,
            enable_intelligent_selection=enable_intelligent
        )
    
    def add_pre_transition_hook(self, hook: Callable) -> None:
        """Add a pre-transition hook.
        
        Args:
            hook: Hook function to add.
        """
        self._pre_transition_hooks.append(hook)
    
    def add_post_transition_hook(self, hook: Callable) -> None:
        """Add a post-transition hook.
        
        Args:
            hook: Hook function to add.
        """
        self._post_transition_hooks.append(hook)
    
    def add_error_hook(self, hook: Callable) -> None:
        """Add an error hook.
        
        Args:
            hook: Hook function to add.
        """
        self._error_hooks.append(hook)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Execution statistics.
        """
        return {
            'executions': self._execution_count,
            'transitions': self._transition_count,
            'errors': self._error_count,
            'strategy': self.strategy.value,
            'hooks_enabled': self.enable_hooks
        }
