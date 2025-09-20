"""Network executor for running state networks."""

from typing import Any, Dict, List, Tuple

from dataknobs_fsm.core.arc import PushArc, DataIsolationMode
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.functions.base import StateTransitionError


class NetworkExecutor:
    """Executor for running state networks with hierarchical support.
    
    This executor manages:
    - Network execution with context isolation
    - Hierarchical network push/pop operations
    - Data passing between networks
    - Resource management across networks
    - Parallel network execution
    """
    
    def __init__(
        self,
        fsm: FSM,
        enable_parallel: bool = False,
        max_depth: int = 10
    ):
        """Initialize network executor.
        
        Args:
            fsm: FSM containing networks to execute.
            enable_parallel: Enable parallel network execution.
            max_depth: Maximum network push depth.
        """
        self.fsm = fsm
        self.enable_parallel = enable_parallel
        self.max_depth = max_depth
        
        # Create execution engine
        self.engine = ExecutionEngine(fsm)
        
        # Track active networks
        self._active_networks: Dict[str, ExecutionContext] = {}
    
    def execute_network(
        self,
        network_name: str,
        context: ExecutionContext | None = None,
        data: Any = None,
        max_transitions: int = 1000,
        initial_state: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute a specific network.
        
        Args:
            network_name: Name of network to execute.
            context: Execution context (created if None).
            data: Input data for network.
            max_transitions: Maximum transitions allowed.
            
        Returns:
            Tuple of (success, result).
        """
        # Get network
        network = self.fsm.networks.get(network_name)
        if not network:
            return False, f"Network not found: {network_name}"
        
        # Create context if needed
        if context is None:
            context = ExecutionContext()
        
        # Set initial data
        if data is not None:
            context.data = data
        
        # Track this network
        self._active_networks[network_name] = context

        try:
            # For subnetworks, we always need to set the initial state
            # regardless of what was in the parent context
            if initial_state:
                # Use the provided initial state override
                # First verify it exists in the network
                if initial_state not in network.states:
                    return False, f"State '{initial_state}' not found in network '{network_name}'"
                state_to_enter = initial_state
            elif network.initial_states:
                # Use the network's default initial state
                state_to_enter = next(iter(network.initial_states))
            else:
                return False, f"No initial state in network: {network_name}"

            # Clear any previous state and set the subnetwork's initial state
            context.current_state = None  # Clear first
            # Use the engine's public enter_state method for consistent state entry
            if not self.engine.enter_state(context, state_to_enter, run_validators=False):
                return False, f"Failed to enter initial state: {state_to_enter}"

            # Execute the network
            result = self._execute_network_internal(
                network,
                context,
                max_transitions
            )
            
            return result
            
        finally:
            # Clean up tracking
            if network_name in self._active_networks:
                del self._active_networks[network_name]
    
    def _execute_network_internal(
        self,
        network: StateNetwork,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Internal network execution.
        
        Args:
            network: Network to execute.
            context: Execution context.
            max_transitions: Maximum transitions.
            
        Returns:
            Tuple of (success, result).
        """
        transitions = 0
        
        while transitions < max_transitions:
            # Check if in final state
            if context.current_state in network.final_states:
                return True, context.data
            
            # Get available arcs from current state
            available_arcs = self._get_available_arcs(
                network,
                context.current_state
            )
            
            if not available_arcs:
                # No transitions available
                if context.current_state in network.final_states:
                    return True, context.data
                return False, f"No valid transitions from: {context.current_state}"
            
            # Process each arc
            transition_made = False
            for _arc_id, arc in available_arcs:
                # Evaluate arc condition/pre_test first (for all arc types including PushArcs)
                if hasattr(arc, 'pre_test') and arc.pre_test:
                    # Arc has a condition (stored as pre_test)
                    # Need to evaluate it using the function registry
                    from dataknobs_fsm.core.arc import ArcExecution
                    arc_exec = ArcExecution(
                        arc,
                        context.current_state or "",
                        self.fsm.function_registry
                    )
                    if not arc_exec.can_execute(context, context.data):
                        continue  # Skip this arc if condition is not met

                # Check if this is a push arc
                if isinstance(arc, PushArc):
                    # Debug: print push arc detection
                    import logging
                    logging.debug(f"Detected PushArc from {context.current_state} to network {arc.target_network}")
                    success = self._handle_push_arc(
                        arc,
                        context
                    )
                elif hasattr(arc, 'metadata') and 'push_arc' in arc.metadata:
                    # Arc with push_arc in metadata
                    push_arc = arc.metadata['push_arc']
                    if isinstance(push_arc, PushArc):
                        success = self._handle_push_arc(
                            push_arc,
                            context
                        )
                    else:
                        # Regular transition
                        success = self.engine._execute_transition(
                            context,
                            arc
                        )
                else:
                    # Regular transition
                    success = self.engine._execute_transition(
                        context,
                        arc
                    )
                
                if success:
                    transition_made = True
                    transitions += 1
                    break
            
            if not transition_made:
                return False, "No valid transition could be made"
            
            # Check for network pop
            if context.current_state in network.final_states:
                if context.network_stack:
                    self._handle_network_return(context)
        
        return False, f"Maximum transitions ({max_transitions}) exceeded"
    
    def _handle_push_arc(
        self,
        arc: PushArc,
        context: ExecutionContext
    ) -> bool:
        """Handle a push arc to another network.

        Args:
            arc: Push arc to execute.
            context: Execution context.

        Returns:
            True if successful.
        """
        # Check depth limit
        if len(context.network_stack) >= self.max_depth:
            raise StateTransitionError(
                from_state=context.current_state or "unknown",
                to_state=arc.target_network,
                message="Maximum network depth exceeded"
            )

        # Save parent state resources before pushing
        parent_state_resources = getattr(context, 'current_state_resources', None)

        # Parse target network and optional initial state
        # Using Syntax: "network_name" or "network_name:initial_state"
        if ':' in arc.target_network:
            network_name, initial_state = arc.target_network.split(':', 1)
            override_initial_state = initial_state.strip()
        else:
            network_name = arc.target_network
            override_initial_state = None

        # Push current network
        context.push_network(
            network_name,
            arc.return_state
        )

        # Get target network
        target_network = self.fsm.networks.get(network_name)
        if not target_network:
            context.pop_network()
            return False

        # Create isolated context if requested
        if hasattr(arc, 'isolation_mode') and arc.isolation_mode == DataIsolationMode.COPY:
            # Full isolation - new context
            sub_context = ExecutionContext(
                data_mode=context.data_mode,
                transaction_mode=context.transaction_mode,
                resources=context.resource_limits
            )
            sub_context.data = context.data
            sub_context.variables = context.variables  # Share variables for tracking
            # Preserve resource manager in new context
            if hasattr(context, 'resource_manager'):
                sub_context.resource_manager = context.resource_manager
            # Preserve parent state resources in new context
            if parent_state_resources:
                sub_context.parent_state_resources = parent_state_resources
        elif hasattr(arc, 'data_isolation_mode') and arc.data_isolation_mode == 'partial':
            # Partial isolation - clone context
            sub_context = context.clone()
            # Preserve parent state resources - this needs to be accessible to all subnetwork states
            if parent_state_resources:
                sub_context.parent_state_resources = parent_state_resources
                # Also ensure resource_manager is available
                if hasattr(context, 'resource_manager'):
                    sub_context.resource_manager = context.resource_manager
        else:
            # No isolation - use same context
            sub_context = context
            if parent_state_resources:
                context.parent_state_resources = parent_state_resources
            # Ensure resource_manager is available in subcontext
            if hasattr(context, 'resource_manager') and not hasattr(sub_context, 'resource_manager'):
                sub_context.resource_manager = context.resource_manager
        
        # Execute target network (which will handle initial state and transforms)
        import logging
        logging.debug(f"Executing sub-network {network_name} with context type {type(sub_context)}")
        # Pass the override initial state if specified
        # Don't pass data parameter - sub_context already has the correctly transformed data
        success, result = self.execute_network(
            network_name,
            sub_context,
            initial_state=override_initial_state
        )
        logging.debug(f"Sub-network execution result: success={success}, result={result}")

        if success:
            # Update main context with result
            context.data = result

            # Return to specified state and execute its entry logic
            if arc.return_state:
                # Use the engine's public enter_state method to properly enter the return state
                if not self.engine.enter_state(context, arc.return_state, run_validators=False):
                    return False

            return True
        
        return False
    
    def _handle_network_return(
        self,
        context: ExecutionContext
    ) -> None:
        """Handle returning from a pushed network.

        Args:
            context: Execution context.
        """
        if context.network_stack:
            _network_name, return_state = context.pop_network()

            # Clean up parent_state_resources attribute if it was added
            if hasattr(context, 'parent_state_resources'):
                delattr(context, 'parent_state_resources')

            if return_state:
                context.set_state(return_state)
    
    def _get_available_arcs(
        self,
        network: StateNetwork,
        state_name: str | None
    ) -> List[Tuple[str, Any]]:
        """Get available arcs from a state.

        Args:
            network: Network containing arcs.
            state_name: Current state name.

        Returns:
            List of (arc_id, arc) tuples.
        """
        if not state_name:
            return []

        # Get the state definition to access actual arc objects (including PushArcs)
        state_def = network.get_state(state_name)
        if not state_def:
            return []

        available = []
        # Use the state's outgoing_arcs which have the proper arc types
        for i, arc in enumerate(state_def.outgoing_arcs):
            # Create an arc_id for tracking
            arc_id = f"{state_name}:{arc.target_state}:{i}"
            available.append((arc_id, arc))

        return available
    
    def execute_parallel_networks(
        self,
        network_configs: List[Dict[str, Any]],
        base_context: ExecutionContext | None = None
    ) -> List[Tuple[bool, Any]]:
        """Execute multiple networks in parallel.
        
        Args:
            network_configs: List of network configurations.
                Each config should have:
                - 'network_name': Name of network
                - 'data': Input data
                - 'max_transitions': Max transitions (optional)
            base_context: Base context to clone for each network.
            
        Returns:
            List of (success, result) tuples in the same order as configs.
        """
        if not self.enable_parallel:
            # Execute sequentially if parallel disabled
            results = []
            for config in network_configs:
                network_name = config['network_name']
                data = config.get('data')
                max_transitions = config.get('max_transitions', 1000)
                
                # Clone context for each network
                if base_context:
                    context = base_context.clone()
                else:
                    context = ExecutionContext()
                
                success, result = self.execute_network(
                    network_name,
                    context,
                    data,
                    max_transitions
                )
                
                results.append((success, result))
            
            return results
        
        # Parallel execution using asyncio
        import asyncio
        
        async def execute_async(config):
            network_name = config['network_name']
            data = config.get('data')
            max_transitions = config.get('max_transitions', 1000)
            
            # Clone context
            if base_context:
                context = base_context.clone()
            else:
                context = ExecutionContext()
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            success, result = await loop.run_in_executor(
                None,
                self.execute_network,
                network_name,
                context,
                data,
                max_transitions
            )
            
            return (success, result)
        
        # Run all networks in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tasks = [execute_async(config) for config in network_configs]
            results = loop.run_until_complete(
                asyncio.gather(*tasks)
            )
            
            return results
            
        finally:
            loop.close()
    
    def validate_all_networks(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all networks in the FSM.
        
        Returns:
            Dictionary of network_name -> (valid, errors).
        """
        results = {}
        
        for network_name, network in self.fsm.networks.items():
            valid, errors = network.validate()
            results[network_name] = (valid, errors)
        
        return results
    
    def get_network_stats(self, network_name: str) -> Dict[str, Any]:
        """Get statistics for a network.
        
        Args:
            network_name: Name of network.
            
        Returns:
            Network statistics.
        """
        network = self.fsm.networks.get(network_name)
        if not network:
            return {}
        
        # Count various elements
        state_count = len(network.states)
        arc_count = len(network.arcs)
        initial_count = len(network.initial_states)
        final_count = len(network.final_states)
        
        # Check connectivity
        valid, errors = network.validate()
        
        # Get resource requirements
        total_resources = {}
        for resource_type, requirements in network.resource_requirements.items():
            total_resources[resource_type] = len(requirements)
        
        return {
            'states': state_count,
            'arcs': arc_count,
            'initial_states': initial_count,
            'final_states': final_count,
            'is_valid': valid,
            'validation_errors': errors,
            'resource_requirements': total_resources,
            'supports_streaming': network.supports_streaming
        }
    
    def get_active_networks(self) -> List[str]:
        """Get list of currently active networks.
        
        Returns:
            List of active network names.
        """
        return list(self._active_networks.keys())
