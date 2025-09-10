"""Execution engine for FSM state machines."""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import State, StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import FunctionContext, StateTransitionError


class TraversalStrategy(Enum):
    """Execution traversal strategy."""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    RESOURCE_OPTIMIZED = "resource_optimized"
    STREAM_OPTIMIZED = "stream_optimized"


class ExecutionEngine:
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
        enable_hooks: bool = True
    ):
        """Initialize execution engine.
        
        Args:
            fsm: FSM instance to execute.
            strategy: Traversal strategy to use.
            max_retries: Maximum retry attempts for failures.
            retry_delay: Delay between retries in seconds.
            enable_hooks: Enable execution hooks.
        """
        self.fsm = fsm
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_hooks = enable_hooks
        
        # Execution tracking
        self._execution_count = 0
        self._transition_count = 0
        self._error_count = 0
        
        # Hooks
        self._pre_transition_hooks: List[callable] = []
        self._post_transition_hooks: List[callable] = []
        self._error_hooks: List[callable] = []
    
    def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000,
        arc_name: Optional[str] = None
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
        
        # Initialize state if needed
        if not context.current_state:
            initial_state = self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"
            context.set_state(initial_state)
        
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
        arc_name: Optional[str] = None
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
            import json
            current_data_hash = json.dumps(context.data, sort_keys=True) if context.data else ""
            
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
                item_context.set_state(initial_state)
            
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
                    record_context.set_state(initial_state)
                
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
                if hasattr(self.fsm, 'function_registry'):
                    if hasattr(self.fsm.function_registry, 'functions'):
                        func_reg = self.fsm.function_registry.functions
                    else:
                        func_reg = {}
                else:
                    func_reg = {}
                    
                arc_exec = ArcExecution(
                    arc,
                    source_state=context.current_state or "",
                    function_registry=func_reg
                )
                
                # Execute with resource context
                result = arc_exec.execute(context, context.data)
                success = result is not None
                
                if success:
                    # Update data with result
                    if result is not None:
                        context.data = result
                    
                    # Update state
                    context.set_state(arc.target_state)
                    self._transition_count += 1
                    
                    # Execute state transforms when entering the new state
                    self._execute_state_transforms(context, arc.target_state)
                    
                    # Fire post-transition hooks
                    if self.enable_hooks:
                        for hook in self._post_transition_hooks:
                            hook(context, arc)
                    
                    return True
                else:
                    # Transition failed but no exception - don't retry
                    return False
                
            except (TypeError, AttributeError, ValueError) as e:
                # Data type errors - don't retry
                self._error_count += 1
                
                # Fire error hooks
                if self.enable_hooks:
                    for hook in self._error_hooks:
                        hook(context, arc, e)
                
                # Return false immediately for data errors
                return False
                
            except Exception as e:
                self._error_count += 1
                
                # Fire error hooks
                if self.enable_hooks:
                    for hook in self._error_hooks:
                        hook(context, arc, e)
                
                retry_count += 1
                if retry_count <= self.max_retries:
                    time.sleep(self.retry_delay * retry_count)
                else:
                    # Don't raise, just return False to allow graceful failure
                    return False
        
        return False
    
    def _execute_state_transforms(
        self,
        context: ExecutionContext,
        state_name: str
    ) -> None:
        """Execute transform functions when entering a state.
        
        Args:
            context: Execution context.
            state_name: Name of the state being entered.
        """
        # Get the state definition
        state_def = self.fsm.get_state(state_name)
        if not state_def:
            return
            
        # Execute any transform functions defined on the state
        if hasattr(state_def, 'transform_functions') and state_def.transform_functions:
            for transform_func in state_def.transform_functions:
                try:
                    # Create function context
                    func_context = FunctionContext(
                        state_name=state_name,
                        function_name=getattr(transform_func, '__name__', 'transform'),
                        metadata={'state': state_name},
                        resources={}
                    )
                    
                    # Execute the transform
                    # Create a mock state object for transforms that expect state.data
                    from types import SimpleNamespace
                    state_obj = SimpleNamespace(data=context.data)
                    
                    # Try calling with state object first (for inline lambdas)
                    try:
                        result = transform_func(state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to calling with data and context
                        result = transform_func(context.data, func_context)
                    
                    if result is not None:
                        context.data = result
                except Exception as e:
                    # Log but don't fail - state transforms are optional
                    pass
    
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
                    # Create function context
                    func_context = FunctionContext(
                        state_name=state_name,
                        function_name=getattr(validator_func, '__name__', 'validate'),
                        metadata={'state': state_name},
                        resources={}
                    )
                    
                    # Execute the validator
                    # Validators typically return a dict with validation results
                    # Create a mock state object for validators that expect state.data
                    from types import SimpleNamespace
                    state_obj = SimpleNamespace(data=context.data)
                    
                    # Try calling with state object first (for inline lambdas)
                    try:
                        result = validator_func(state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to calling with data and context
                        result = validator_func(context.data, func_context)
                    
                    if isinstance(result, dict):
                        # Merge validation results into context data
                        context.data.update(result)
                except Exception as e:
                    # Log but don't fail - state validators are optional
                    pass
        
        # Execute transform functions
        if hasattr(state_def, 'transform_functions') and state_def.transform_functions:
            for transform_func in state_def.transform_functions:
                try:
                    # Create function context
                    func_context = FunctionContext(
                        state_name=state_name,
                        function_name=getattr(transform_func, '__name__', 'transform'),
                        metadata={'state': state_name},
                        resources={}
                    )
                    
                    # Execute the transform
                    # Create a mock state object for transforms that expect state.data
                    from types import SimpleNamespace
                    state_obj = SimpleNamespace(data=context.data)
                    
                    # Try calling with state object first (for inline lambdas)
                    try:
                        result = transform_func(state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to calling with data and context
                        result = transform_func(context.data, func_context)
                    
                    if result is not None:
                        context.data = result
                except Exception as e:
                    # Log but don't fail - state transforms are optional
                    pass
    
    def _get_available_transitions(
        self,
        context: ExecutionContext,
        arc_name: Optional[str] = None
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
        available.sort(key=lambda x: x.priority, reverse=True)
        
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
            return bool(result)
        except Exception:
            return False
    
    def _choose_transition(
        self,
        transitions: List[ArcDefinition],
        context: ExecutionContext
    ) -> Optional[ArcDefinition]:
        """Choose next transition based on strategy.
        
        Args:
            transitions: Available transitions.
            context: Execution context.
            
        Returns:
            Chosen arc or None.
        """
        if not transitions:
            return None
        
        if self.strategy == TraversalStrategy.DEPTH_FIRST:
            # Take first available (highest priority)
            return transitions[0]
        
        elif self.strategy == TraversalStrategy.BREADTH_FIRST:
            # Prefer transitions to unvisited states
            for arc in transitions:
                if arc.target_state not in context.state_history:
                    return arc
            return transitions[0]
        
        elif self.strategy == TraversalStrategy.RESOURCE_OPTIMIZED:
            # Choose transition with least resource requirements
            best_arc = None
            min_resources = float('inf')
            
            for arc in transitions:
                resource_count = len(arc.resource_requirements) if hasattr(arc, 'resource_requirements') else 0
                if resource_count < min_resources:
                    min_resources = resource_count
                    best_arc = arc
            
            return best_arc or transitions[0]
        
        elif self.strategy == TraversalStrategy.STREAM_OPTIMIZED:
            # Prefer transitions that support streaming
            for arc in transitions:
                if hasattr(arc, 'supports_streaming') and arc.supports_streaming:
                    return arc
            return transitions[0]
        
        return transitions[0]
    
    def _find_initial_state(self) -> Optional[str]:
        """Find the initial state in the FSM.
        
        Returns:
            Name of initial state or None.
        """
        # Try to get main_network attribute first (for core FSM)
        main_network_name = getattr(self.fsm, 'main_network', None)
        if main_network_name and main_network_name in self.fsm.networks:
            network = self.fsm.networks[main_network_name]
            if network.initial_states:
                return next(iter(network.initial_states))
        
        # Fallback to fsm.name (for FSMWrapper compatibility)
        if self.fsm.name in self.fsm.networks:
            network = self.fsm.networks[self.fsm.name]
            if network.initial_states:
                return next(iter(network.initial_states))
        
        # Last resort: check all networks for any initial state
        for network in self.fsm.networks.values():
            if network.initial_states:
                return next(iter(network.initial_states))
        
        return None
    
    def _is_final_state(self, state_name: Optional[str]) -> bool:
        """Check if state is a final state.
        
        Args:
            state_name: Name of state to check.
            
        Returns:
            True if final state.
        """
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
    ) -> Optional[StateNetwork]:
        """Get the current network from context.
        
        Args:
            context: Execution context.
            
        Returns:
            Current network or None.
        """
        # Check if we're in a pushed network
        if context.network_stack:
            network_name = context.network_stack[-1][0]
        else:
            # Get the main network name - handle both wrapper and core FSM
            if hasattr(self.fsm, 'core_fsm'):
                # This is a wrapper FSM
                network_name = self.fsm.core_fsm.main_network
            else:
                # This is a core FSM
                network_name = self.fsm.main_network
        
        return self.fsm.networks.get(network_name)
    
    def add_pre_transition_hook(self, hook: callable) -> None:
        """Add a pre-transition hook.
        
        Args:
            hook: Hook function to add.
        """
        self._pre_transition_hooks.append(hook)
    
    def add_post_transition_hook(self, hook: callable) -> None:
        """Add a post-transition hook.
        
        Args:
            hook: Hook function to add.
        """
        self._post_transition_hooks.append(hook)
    
    def add_error_hook(self, hook: callable) -> None:
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