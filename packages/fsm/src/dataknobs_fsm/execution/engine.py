"""Execution engine for FSM state machines."""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dataknobs_fsm.core.arc import ArcDefinition, ArcExecution
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import DataMode, TransactionMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import State
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
    
    async def execute_async(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000
    ) -> Tuple[bool, Any]:
        """Execute the FSM asynchronously with given context.
        
        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            
        Returns:
            Tuple of (success, result).
        """
        # For now, wrap the synchronous execution
        # In the future, this could be made truly async with async state functions
        return self.execute(context, data, max_transitions)
    
    def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000
    ) -> Tuple[bool, Any]:
        """Execute the FSM with given context.
        
        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            
        Returns:
            Tuple of (success, result).
        """
        self._execution_count += 1
        context.data = data
        
        # Initialize state if needed
        if not context.current_state:
            initial_state = self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"
            context.set_state(initial_state)
        
        # Execute based on data mode
        if context.data_mode == DataMode.SINGLE:
            return self._execute_single(context, max_transitions)
        elif context.data_mode == DataMode.BATCH:
            return self._execute_batch(context, max_transitions)
        elif context.data_mode == DataMode.STREAM:
            return self._execute_stream(context, max_transitions)
        else:
            return False, f"Unknown data mode: {context.data_mode}"
    
    def _execute_single(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in single record mode.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions.
            
        Returns:
            Tuple of (success, result).
        """
        transitions = 0
        last_state = None
        stuck_count = 0
        max_stuck_iterations = 3  # Max times we can be in same state
        
        while transitions < max_transitions:
            # Check if in final state
            if self._is_final_state(context.current_state):
                return True, context.data
            
            # Check for stuck state (infinite loop protection)
            if context.current_state == last_state:
                stuck_count += 1
                if stuck_count >= max_stuck_iterations:
                    return False, f"Stuck in state '{context.current_state}' - possible infinite loop"
            else:
                stuck_count = 0
                last_state = context.current_state
            
            # Get available transitions
            transitions_available = self._get_available_transitions(context)
            
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
                arc_exec = ArcExecution(
                    arc,
                    source_state=context.current_state or "",
                    function_registry=self.fsm.function_registry.functions
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
    
    def _get_available_transitions(
        self,
        context: ExecutionContext
    ) -> List[ArcDefinition]:
        """Get available transitions from current state.
        
        Args:
            context: Execution context.
            
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
            result = func.execute(context.data, func_context)
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
        # Get main network
        if self.fsm.name in self.fsm.networks:
            network = self.fsm.networks[self.fsm.name]
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
        
        # Check in main network
        if self.fsm.name in self.fsm.networks:
            network = self.fsm.networks[self.fsm.name]
            return state_name in network.final_states
        
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
            network_name = self.fsm.name
        
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