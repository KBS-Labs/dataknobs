"""Factory for creating and configuring ExecutionContext instances.

This module provides a centralized factory for creating execution contexts,
eliminating duplication between Simple and Advanced APIs.
"""

from typing import Any, Dict, Union
from dataknobs_data import Record

from ..core.fsm import FSM
from ..core.state import StateInstance, StateDefinition, StateType
from ..core.modes import ProcessingMode, TransactionMode
from ..execution.context import ExecutionContext
from ..resources.manager import ResourceManager


class ContextFactory:
    """Factory for creating and configuring ExecutionContext instances."""
    
    @staticmethod
    def create_context(
        fsm: FSM,
        data: Union[Dict[str, Any], Record],
        initial_state: str | None = None,
        data_mode: ProcessingMode = ProcessingMode.SINGLE,
        transaction_mode: TransactionMode = TransactionMode.NONE,
        resources: Dict[str, Any] | None = None,
        resource_manager: ResourceManager | None = None
    ) -> ExecutionContext:
        """Create and configure an ExecutionContext.
        
        Args:
            fsm: The FSM instance
            data: Input data as dict or Record
            initial_state: Optional initial state name
            data_mode: Processing mode (single, batch, stream)
            transaction_mode: Transaction mode
            resources: Resource configuration dict
            resource_manager: Optional resource manager instance
            
        Returns:
            Configured ExecutionContext ready for execution
        """
        # Create base context
        context = ExecutionContext(
            data_mode=data_mode,
            transaction_mode=transaction_mode,
            resources=resources or {}
        )
        
        # Convert and set data
        if isinstance(data, Record):
            context.data = data.to_dict()
        else:
            context.data = data
        
        # Determine initial state
        state_name, state_def = ContextFactory._resolve_initial_state(
            fsm, initial_state
        )
        
        # Set state in context
        context.current_state = state_name
        if state_def:
            context.current_state_instance = StateInstance(
                definition=state_def,
                data=context.data.copy() if isinstance(context.data, dict) else {}
            )
        
        # Add resource manager reference if provided
        if resource_manager:
            context.resource_manager = resource_manager
        
        return context
    
    @staticmethod
    def _resolve_initial_state(
        fsm: FSM, 
        state_name: str | None = None
    ) -> tuple[str, StateDefinition | None]:
        """Resolve the initial state for execution.
        
        Args:
            fsm: The FSM instance
            state_name: Optional requested state name
            
        Returns:
            Tuple of (state_name, state_definition)
        """
        state_def = None
        
        if state_name:
            # Try to get the requested state
            if hasattr(fsm, 'get_state'):
                try:
                    state_def = fsm.get_state(state_name)
                except (KeyError, AttributeError):
                    pass
            
            # If not found, search in networks
            if not state_def and hasattr(fsm, 'networks'):
                for network in fsm.networks.values():
                    if hasattr(network, 'states') and state_name in network.states:
                        state_def = network.states[state_name]
                        break
        else:
            # Find the start state
            if hasattr(fsm, 'get_start_state'):
                state_def = fsm.get_start_state()
                if state_def:
                    state_name = state_def.name
            
            # Fallback: search networks for start state
            if not state_def and hasattr(fsm, 'networks'):
                for network in fsm.networks.values():
                    if hasattr(network, 'states'):
                        for state in network.states.values():
                            if hasattr(state, 'is_start_state') and state.is_start_state():
                                state_def = state
                                state_name = state.name
                                break
                    if state_def:
                        break
            
            # Final fallback
            if not state_name:
                state_name = 'start'
        
        # Create minimal state definition if needed
        if not state_def and state_name:
            state_def = StateDefinition(
                name=state_name,
                type=StateType.START if state_name == 'start' else StateType.NORMAL
            )
        
        return state_name, state_def
    
    @staticmethod
    def create_batch_context(
        fsm: FSM,
        batch_data: list,
        data_mode: ProcessingMode = ProcessingMode.BATCH,
        resources: Dict[str, Any] | None = None
    ) -> ExecutionContext:
        """Create a context for batch processing.
        
        Args:
            fsm: The FSM instance
            batch_data: List of data items to process
            data_mode: Processing mode (defaults to BATCH)
            resources: Resource configuration
            
        Returns:
            ExecutionContext configured for batch processing
        """
        context = ExecutionContext(
            data_mode=data_mode,
            resources=resources or {}
        )
        
        # Set batch data
        context.batch_data = batch_data
        
        # Find start state
        state_name, state_def = ContextFactory._resolve_initial_state(fsm, None)
        context.current_state = state_name
        
        if state_def:
            context.current_state_instance = StateInstance(
                definition=state_def,
                data={}
            )
        
        return context
    
    @staticmethod
    def create_stream_context(
        fsm: FSM,
        stream_context,
        resources: Dict[str, Any] | None = None
    ) -> ExecutionContext:
        """Create a context for stream processing.
        
        Args:
            fsm: The FSM instance
            stream_context: Stream context object
            resources: Resource configuration
            
        Returns:
            ExecutionContext configured for stream processing
        """
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            resources=resources or {},
            stream_context=stream_context
        )
        
        # Find start state
        state_name, state_def = ContextFactory._resolve_initial_state(fsm, None)
        context.current_state = state_name
        
        if state_def:
            context.current_state_instance = StateInstance(
                definition=state_def,
                data={}
            )
        
        return context
