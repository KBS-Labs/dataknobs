"""Asynchronous execution engine for FSM processing."""

import asyncio
import time
from typing import Any, Dict, List, Tuple

from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import TraversalStrategy
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelectionMode
)
from dataknobs_fsm.execution.base_engine import BaseExecutionEngine
from dataknobs_fsm.functions.base import FunctionContext
from dataknobs_fsm.core.data_wrapper import ensure_dict


class AsyncExecutionEngine(BaseExecutionEngine):
    """Asynchronous execution engine for FSM.
    
    This engine handles:
    - True async execution of state functions
    - Parallel arc evaluation
    - Async resource management
    - Non-blocking state transitions
    """
    
    def __init__(
        self,
        fsm: FSM,
        strategy: TraversalStrategy = TraversalStrategy.DEPTH_FIRST,
        selection_mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID
    ):
        """Initialize async execution engine.

        Args:
            fsm: FSM to execute.
            strategy: Traversal strategy for execution.
            selection_mode: Transition selection mode (strategy, scoring, or hybrid).
        """
        # Initialize base class (no max_retries/retry_delay needed for async)
        super().__init__(fsm, strategy, selection_mode, max_retries=3, retry_delay=1.0)
    
    async def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute the FSM asynchronously with given context.
        
        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Only override context.data if data was explicitly provided
        if data is not None:
            context.data = data
        
        # Initialize state if needed
        if not context.current_state:
            initial_state = await self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"
            context.set_state(initial_state)
            # Execute transforms for the initial state
            await self._execute_state_transforms(context)
        
        try:
            # Execute based on data mode
            if context.data_mode == ProcessingMode.SINGLE:
                result = await self._execute_single(context, max_transitions, arc_name)
            elif context.data_mode == ProcessingMode.BATCH:
                result = await self._execute_batch(context, max_transitions)
            elif context.data_mode == ProcessingMode.STREAM:
                result = await self._execute_stream(context, max_transitions)
            else:
                result = False, f"Unknown data mode: {context.data_mode}"
                
            self._total_execution_time += time.time() - start_time
            return result
            
        except Exception as e:
            self._error_count += 1
            self._total_execution_time += time.time() - start_time
            return False, str(e)
    
    async def _execute_single(
        self,
        context: ExecutionContext,
        max_transitions: int,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute in single record mode asynchronously.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        transitions = 0
        
        while transitions < max_transitions:
            # Check if we're in a final state
            if await self._is_final_state(context.current_state):
                return True, context.data
            
            # Get available transitions
            transitions_available = await self._get_available_transitions(
                context.current_state,
                context,
                arc_name
            )
            
            if not transitions_available:
                # No valid transitions - check if this is a final state
                if await self._is_final_state(context.current_state):
                    return True, context.data
                return False, f"No valid transitions from state: {context.current_state}"
            
            # Choose transition based on strategy
            next_transition = await self._choose_transition(
                transitions_available,
                context
            )
            
            if not next_transition:
                return False, "No transition selected"
            
            # Execute transition
            success = await self._execute_transition(
                next_transition,
                context
            )
            
            if not success:
                return False, f"Transition failed: {next_transition}"
            
            transitions += 1
            self._transition_count += 1
        
        return False, f"Maximum transitions ({max_transitions}) exceeded"
    
    async def _execute_batch(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in batch mode asynchronously.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions per item.
            
        Returns:
            Tuple of (success, results).
        """
        if not context.batch_data:
            return False, "No batch data to process"
        
        # Process items in parallel
        tasks = []
        for i, item in enumerate(context.batch_data):
            # Create child context for this item
            item_context = context.create_child_context(f"batch_{i}")
            item_context.data = item
            
            # Reset to initial state for each item
            initial_state = await self._find_initial_state()
            if initial_state:
                item_context.set_state(initial_state)
            
            # Create task for this item
            task = asyncio.create_task(
                self._execute_single(item_context, max_transitions)
            )
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = []
        batch_errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_errors.append((i, result))
            else:
                # Result is a tuple[bool, Any] at this point
                success, value = result  # type: ignore
                if success:  # success
                    batch_results.append(value)
                else:
                    batch_errors.append((i, Exception(value)))
        
        return len(batch_errors) == 0, {
            'results': batch_results,
            'errors': batch_errors
        }
    
    async def _execute_stream(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in stream mode asynchronously.
        
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
                initial_state = await self._find_initial_state()
                if initial_state:
                    record_context.set_state(initial_state)
                
                # Execute for this record
                success, result = await self._execute_single(
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
    
    async def _get_available_transitions(
        self,
        state_name: str,
        context: ExecutionContext,
        arc_name: str | None = None
    ) -> List[ArcDefinition]:
        """Get available transitions from current state asynchronously.
        
        This evaluates pre-conditions in parallel.
        
        Args:
            state_name: Current state name.
            context: Execution context.
            arc_name: Optional specific arc name to filter by.
            
        Returns:
            List of available arc definitions.
        """
        network = await self._get_current_network(context)
        if not network or state_name not in network.states:
            return []
        
        state = network.states[state_name]
        available = []
        
        # Filter arcs by name if specified
        arcs_to_evaluate = state.outgoing_arcs
        if arc_name:
            arcs_to_evaluate = [arc for arc in state.outgoing_arcs 
                              if hasattr(arc, 'name') and arc.name == arc_name]
            # If no arcs match the specified name, return empty list
            if not arcs_to_evaluate:
                return []
        
        # Evaluate all arc pre-conditions in parallel
        tasks = []
        for arc in arcs_to_evaluate:
            task = asyncio.create_task(self._evaluate_arc(arc, context))
            tasks.append((arc, task))
        
        # Wait for all evaluations
        for arc, task in tasks:
            can_execute = await task
            if can_execute:
                available.append(arc)
        
        # Sort by priority (higher first)
        available.sort(key=lambda a: a.priority, reverse=True)
        
        return available
    
    async def _evaluate_arc(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Evaluate if an arc can be executed.
        
        Args:
            arc: Arc definition.
            context: Execution context.
            
        Returns:
            True if arc can be executed.
        """
        if not arc.pre_test:
            return True
        
        # Get the function registry
        function_registry = getattr(self.fsm, 'function_registry', {})
        if hasattr(function_registry, 'functions'):
            functions = function_registry.functions
        else:
            functions = function_registry
        
        if arc.pre_test not in functions:
            return False
        
        # Execute pre-test function
        pre_test_func = functions[arc.pre_test]

        # Check if it's async
        if asyncio.iscoroutinefunction(pre_test_func):
            result = await pre_test_func(context.data, context)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                pre_test_func,
                context.data,
                context
            )

        # Handle tuple return from test functions (bool, reason)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    
    async def _choose_transition(
        self,
        available: List[ArcDefinition],
        context: ExecutionContext
    ) -> ArcDefinition | None:
        """Choose transition using common transition selector.
        
        Args:
            available: Available transitions.
            context: Execution context.
            
        Returns:
            Selected arc or None.
        """
        return self.transition_selector.select_transition(
            available,
            context,
            strategy=self.strategy
        )
    
    async def _execute_transition(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Execute a state transition asynchronously.
        
        Args:
            arc: Arc to execute.
            context: Execution context.
            
        Returns:
            True if successful.
        """
        try:
            # Execute arc transform if defined
            if arc.transform:
                function_registry = getattr(self.fsm, 'function_registry', {})
                if hasattr(function_registry, 'functions'):
                    functions = function_registry.functions
                else:
                    functions = function_registry
                
                if arc.transform in functions:
                    transform_func = functions[arc.transform]

                    # Check if it's async - check both the function and its __call__ method
                    is_async = asyncio.iscoroutinefunction(transform_func)
                    if not is_async and callable(transform_func) and callable(transform_func):
                        # Check if the __call__ method is async (for wrapped functions)
                        is_async = asyncio.iscoroutinefunction(transform_func.__call__)

                    if is_async:
                        context.data = await transform_func(context.data, context)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        context.data = await loop.run_in_executor(
                            None,
                            transform_func,
                            context.data,
                            context
                        )
            
            # Update state (history is automatically tracked by set_state)
            context.set_state(arc.target_state)

            # Execute state transforms when entering the new state
            await self._execute_state_transforms(context)

            return True
            
        except Exception:
            return False
    
    async def _execute_state_transforms(
        self,
        context: ExecutionContext
    ) -> None:
        """Execute state functions (validators and transforms) when in a state.

        This should be called before evaluating arc conditions to ensure
        that state functions can update the data that conditions depend on.

        Args:
            context: Execution context.
        """
        network = await self._get_current_network(context)
        if not network or context.current_state not in network.states:
            return

        state = network.states[context.current_state]
        state_name = context.current_state

        # Use base class logic to prepare transforms
        transform_functions, state_obj = self.prepare_state_transform(state, context)

        # Execute validation functions first (async-specific)
        if hasattr(state, 'validation_functions') and state.validation_functions:
            for validator in state.validation_functions:
                try:
                    # Handle both async and sync validators
                    if asyncio.iscoroutinefunction(validator.validate):
                        # Try with state object first (for inline lambdas)
                        try:
                            result = await validator.validate(state_obj)
                        except (TypeError, AttributeError):
                            # Fall back to standard signature
                            result = await validator.validate(ensure_dict(context.data), context)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        try:
                            result = await loop.run_in_executor(None, validator.validate, state_obj)
                        except (TypeError, AttributeError):
                            # Fall back to standard signature
                            result = await loop.run_in_executor(None, validator.validate, ensure_dict(context.data), context)

                    if isinstance(result, dict):
                        # Merge validation results into context data
                        context.data.update(result)
                except Exception:
                    # Log but don't fail - validators are optional
                    pass

        # Execute transform functions using base class helpers
        import logging
        logger = logging.getLogger(__name__)
        if transform_functions:
            logger.debug(f"Executing {len(transform_functions)} transform functions for state {state_name}")
        for transform_func in transform_functions:
            try:
                # Create function context
                func_context = FunctionContext(
                    state_name=state_name,
                    function_name=getattr(transform_func, '__name__', 'transform'),
                    metadata={'state': state_name},
                    resources={}
                )

                # Handle both async and sync transforms
                # For InterfaceWrapper objects, use the transform method
                actual_func = transform_func
                if hasattr(transform_func, 'transform'):
                    actual_func = transform_func.transform

                # Check if it's async - check both the function and its __call__ method
                is_async = asyncio.iscoroutinefunction(actual_func)
                if not is_async and callable(actual_func) and callable(actual_func):
                    # Check if the __call__ method is async (for wrapped functions)
                    is_async = asyncio.iscoroutinefunction(actual_func.__call__)

                # Also check for _is_async attribute (for wrapped functions)
                if not is_async and hasattr(transform_func, '_is_async'):
                    is_async = transform_func._is_async

                if is_async:
                    # Try with state object first (for inline lambdas)
                    try:
                        result = await actual_func(state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to standard signature
                        result = await actual_func(ensure_dict(context.data), func_context)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(None, actual_func, state_obj)
                    except (TypeError, AttributeError):
                        # Fall back to standard signature
                        result = await loop.run_in_executor(None, actual_func, ensure_dict(context.data), func_context)

                # Process result using base class logic
                self.process_transform_result(result, context, state_name)

            except Exception as e:
                # Handle error using base class logic
                self.handle_transform_error(e, context, state_name)
    
    async def _find_initial_state(self) -> str | None:
        """Find initial state in FSM.

        Returns:
            Initial state name or None.
        """
        # Use base class implementation (it's synchronous but that's fine)
        return self.find_initial_state_common()
    
    async def _is_final_state(self, state_name: str | None) -> bool:
        """Check if state is a final state.

        Args:
            state_name: Name of state to check.

        Returns:
            True if final state.
        """
        # Use base class implementation
        return self.is_final_state_common(state_name)

    async def _is_final_state_legacy(self, state_name: str | None) -> bool:
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
    
    async def _get_current_network(
        self,
        context: ExecutionContext
    ) -> StateNetwork | None:
        """Get the current network from context using common network selector.
        
        Args:
            context: Execution context.
            
        Returns:
            Current network or None.
        """
        # Use intelligent selection for async engine by default
        return NetworkSelector.get_current_network(
            self.fsm,
            context,
            enable_intelligent_selection=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return {
            'execution_count': self._execution_count,
            'transition_count': self._transition_count,
            'error_count': self._error_count,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0.0
            )
        }
