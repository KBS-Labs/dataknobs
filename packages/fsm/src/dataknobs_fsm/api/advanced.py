"""Advanced API for full control over FSM execution.

This module provides advanced interfaces for users who need fine-grained
control over FSM execution, resource management, and monitoring.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable, Type
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import asyncio
from contextlib import asynccontextmanager
from dataknobs_data import Record, Query

from ..core.fsm import FSM
from ..core.state import StateInstance, StateDefinition
from ..core.arc import ArcDefinition
from ..core.data_modes import DataHandlingMode, DataHandler
from ..core.modes import ProcessingMode
from ..core.transactions import TransactionStrategy, TransactionManager
from ..execution.engine import ExecutionEngine, TraversalStrategy
from ..execution.async_engine import AsyncExecutionEngine
from ..execution.context import ExecutionContext
from ..execution.network import NetworkExecutor
from ..resources.manager import ResourceManager
from ..resources.base import IResourceProvider
from ..streaming.core import StreamConfig
from ..execution.history import ExecutionHistory
from ..storage.base import IHistoryStorage


class ExecutionMode(Enum):
    """Advanced execution modes."""
    STEP_BY_STEP = "step"  # Execute one transition at a time
    BREAKPOINT = "breakpoint"  # Stop at specific states
    TRACE = "trace"  # Full execution tracing
    PROFILE = "profile"  # Performance profiling
    DEBUG = "debug"  # Debug mode with detailed logging


@dataclass
class ExecutionHook:
    """Hook for monitoring execution events."""
    on_state_enter: Optional[Callable] = None
    on_state_exit: Optional[Callable] = None
    on_arc_execute: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_resource_acquire: Optional[Callable] = None
    on_resource_release: Optional[Callable] = None
    on_transaction_begin: Optional[Callable] = None
    on_transaction_commit: Optional[Callable] = None
    on_transaction_rollback: Optional[Callable] = None


class AdvancedFSM:
    """Advanced FSM interface with full control capabilities."""
    
    def __init__(
        self,
        fsm: FSM,
        execution_mode: ExecutionMode = ExecutionMode.STEP_BY_STEP
    ):
        """Initialize AdvancedFSM.
        
        Args:
            fsm: Core FSM instance
            execution_mode: Execution mode for advanced control
        """
        self.fsm = fsm
        self.execution_mode = execution_mode
        self._engine = ExecutionEngine(fsm)
        self._async_engine = AsyncExecutionEngine(fsm)
        self._resource_manager = ResourceManager()
        self._transaction_manager = None
        self._history = None
        self._storage = None
        self._hooks = ExecutionHook()
        self._breakpoints = set()
        self._trace_buffer = []
        self._profile_data = {}
        
    def set_execution_strategy(self, strategy: TraversalStrategy) -> None:
        """Set custom execution strategy.
        
        Args:
            strategy: Execution strategy to use
        """
        self._engine.strategy = strategy
        
    def set_data_handler(self, handler: DataHandler) -> None:
        """Set custom data handler.
        
        Args:
            handler: Data handler implementation
        """
        self._engine.data_handler = handler
        
    def configure_transactions(
        self,
        strategy: TransactionStrategy,
        **config
    ) -> None:
        """Configure transaction management.
        
        Args:
            strategy: Transaction strategy to use
            **config: Strategy-specific configuration
        """
        self._transaction_manager = TransactionManager.create(strategy, **config)
        
    def register_resource(
        self,
        name: str,
        resource: Union[IResourceProvider, Dict[str, Any]]
    ) -> None:
        """Register a custom resource.
        
        Args:
            name: Resource name
            resource: Resource instance or configuration
        """
        if isinstance(resource, dict):
            # For dict configs, create a simple in-memory provider
            from ..resources.base import IResourceProvider
            
            class SimpleProvider(IResourceProvider):
                def __init__(self, config):
                    self.config = config
                    
                async def get_resource(self):
                    return self.config
                    
                async def close(self):
                    pass
            
            provider = SimpleProvider(resource)
            self._resource_manager.register_provider(name, provider)
        else:
            # Assume it's already a provider
            self._resource_manager.register_provider(name, resource)
            
    def set_hooks(self, hooks: ExecutionHook) -> None:
        """Set execution hooks for monitoring.
        
        Args:
            hooks: Execution hooks configuration
        """
        self._hooks = hooks
        
    def add_breakpoint(self, state_name: str) -> None:
        """Add a breakpoint at a specific state.
        
        Args:
            state_name: Name of state to break at
        """
        self._breakpoints.add(state_name)
        
    def remove_breakpoint(self, state_name: str) -> None:
        """Remove a breakpoint.
        
        Args:
            state_name: Name of state to remove breakpoint from
        """
        self._breakpoints.discard(state_name)
        
    def enable_history(
        self,
        storage: Optional[IHistoryStorage] = None,
        max_depth: int = 100
    ) -> None:
        """Enable execution history tracking.
        
        Args:
            storage: Optional storage backend for history
            max_depth: Maximum history depth to track
        """
        import uuid
        from dataknobs_fsm.core.data_modes import DataHandlingMode
        
        # Get FSM name from the FSM object
        fsm_name = getattr(self.fsm, 'name', 'unnamed_fsm')
        
        # Generate a unique execution ID
        execution_id = str(uuid.uuid4())
        
        self._history = ExecutionHistory(
            fsm_name=fsm_name,
            execution_id=execution_id,
            data_mode=DataHandlingMode.COPY,  # Default data mode
            max_depth=max_depth
        )
        self._storage = storage
        
    @asynccontextmanager
    async def execution_context(
        self,
        data: Union[Dict[str, Any], Record],
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        initial_state: Optional[str] = None
    ):
        """Create an execution context for manual control.
        
        Args:
            data: Initial data
            data_mode: Data handling mode
            initial_state: Starting state name
            
        Yields:
            ExecutionContext for manual execution
        """
        # Convert to Record if needed
        if isinstance(data, dict):
            record = Record(data)
        else:
            record = data
            
        # Create context - use SINGLE processing mode for now
        # The data_mode parameter is for data handling (COPY/REFERENCE/DIRECT)
        # but ExecutionContext expects processing mode (SINGLE/BATCH/STREAM)
        context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE,
            resources={}  # Pass empty dict, not the manager itself
        )
        
        # Set transaction manager if configured
        if self._transaction_manager:
            context.transaction_manager = self._transaction_manager
            
        # Initialize state and data
        context.data = record.to_dict()
        
        # Find the state definition and create a StateInstance
        from dataknobs_fsm.core.state import StateInstance, StateDefinition, StateType
        
        state_def = None
        state_name = initial_state
        
        if not state_name:
            # Find start state from FSM
            if hasattr(self.fsm, 'get_start_state'):
                state_def = self.fsm.get_start_state()
                if state_def:
                    state_name = state_def.name
            elif hasattr(self.fsm, 'core_fsm'):
                # It's an FSM wrapper
                for network in self.fsm.core_fsm.networks.values():
                    for state in network.states.values():
                        if state.is_start_state():
                            state_def = state
                            state_name = state.name
                            break
                    if state_def:
                        break
            
            if not state_name:
                state_name = 'start'  # Default fallback
        
        # Get state definition if we don't have it yet
        if not state_def and hasattr(self.fsm, 'core_fsm'):
            for network in self.fsm.core_fsm.networks.values():
                if state_name in network.states:
                    state_def = network.states[state_name]
                    break
        
        # Create StateInstance if we have a definition, otherwise create a minimal one
        if state_def:
            state_instance = StateInstance(
                definition=state_def,
                data=context.data.copy() if isinstance(context.data, dict) else {}
            )
        else:
            # Create a minimal StateDefinition for backward compatibility
            state_def = StateDefinition(
                name=state_name,
                type=StateType.START if state_name == 'start' else StateType.NORMAL
            )
            state_instance = StateInstance(
                definition=state_def,
                data=context.data.copy() if isinstance(context.data, dict) else {}
            )
        
        # Store both string name and instance for API compatibility
        context.current_state = state_name
        context.current_state_instance = state_instance
        
        # Call hook with StateInstance
        if self._hooks.on_state_enter:
            await self._hooks.on_state_enter(state_instance)
            
        try:
            yield context
        finally:
            # Cleanup
            if self._hooks.on_state_exit:
                await self._hooks.on_state_exit(state_instance)
            await self._resource_manager.cleanup()
            
    async def step(
        self,
        context: ExecutionContext,
        arc_name: Optional[str] = None
    ) -> Optional[StateInstance]:
        """Execute a single transition step.
        
        Args:
            context: Execution context
            arc_name: Optional specific arc to follow
            
        Returns:
            New state instance or None if no transition
        """
        from dataknobs_fsm.core.state import StateInstance, StateDefinition, StateType
        
        # Get the current state instance - now stored separately from state name
        current_state_instance = getattr(context, 'current_state_instance', None)
        if not current_state_instance:
            # Create StateInstance if not present (for backward compatibility)
            # current_state should be a string now
            state_name = context.current_state if isinstance(context.current_state, str) else 'START'
            state_def = StateDefinition(
                name=state_name,
                type=StateType.NORMAL
            )
            current_state_instance = StateInstance(
                definition=state_def,
                data=context.data.copy() if isinstance(context.data, dict) else {}
            )
            # Store it for future use
            context.current_state_instance = current_state_instance
        
        # Find available arcs from the current network
        # Get the main network (simplified - should track network stack in real implementation)
        # Note: fsm.main_network is the actual network object, not a string key
        network = self.fsm.main_network
        if not network:
            return None
            
        # Find arcs from current state
        current_state_name = current_state_instance.definition.name
        arcs = []
        for arc_id, arc in network.arcs.items():
            # Arc IDs are typically formatted as "source:target"
            if ':' in arc_id:
                source_state = arc_id.split(':')[0]
                if source_state == current_state_name:
                    arcs.append(arc)
        
        # Filter to specific arc if requested
        if arc_name:
            arcs = [arc for arc in arcs if arc.metadata.get('name') == arc_name]
            
        # Execute first valid arc
        for arc in arcs:
            # Check pre-test if exists
            if arc.pre_test:
                if not await arc.pre_test.test(current_state_instance):
                    continue
                    
            # Call arc hook
            if self._hooks.on_arc_execute:
                await self._hooks.on_arc_execute(arc)
                
            # Execute transformation
            if arc.transform:
                new_data = await arc.transform.transform(current_state_instance.data)
            else:
                new_data = current_state_instance.data.copy()
                
            # Create new state
            # Get target state definition from network
            target_def = network.states.get(arc.target_state)
            if not target_def:
                continue  # Skip if target state not found
                
            new_state = StateInstance(
                definition=target_def,
                data=new_data
            )
            
            # Update context - store both string name and instance
            context.current_state = new_state.definition.name
            context.current_state_instance = new_state
            context.data = new_data
            
            # Track in history
            if self._history:
                # Add a step for the new state
                step = self._history.add_step(
                    state_name=new_state.definition.name,
                    network_name=getattr(context, 'network_name', 'main'),
                    data=new_data
                )
                # Mark it as completed with the arc taken
                step.complete(arc_taken=arc.metadata.get('name', 'unnamed'))
                
            # Add to trace if tracing
            if self.execution_mode == ExecutionMode.TRACE:
                self._trace_buffer.append({
                    'from': current_state_instance.definition.name,
                    'to': new_state.definition.name,
                    'arc': arc.metadata.get('name', 'unnamed'),
                    'data': new_data
                })
                
            # Call state enter hook
            if self._hooks.on_state_enter:
                await self._hooks.on_state_enter(new_state)
                
            return new_state
            
        return None
        
    async def run_until_breakpoint(
        self,
        context: ExecutionContext
    ) -> StateInstance:
        """Run execution until a breakpoint is hit.
        
        Args:
            context: Execution context
            
        Returns:
            State where execution stopped
        """
        while True:
            # Check if current state is a breakpoint (current_state is now a string)
            if context.current_state in self._breakpoints:
                return context.current_state_instance
                
            # Step to next state
            new_state = await self.step(context)
            
            # Check if we reached an end state
            if not new_state or (hasattr(new_state.definition, 'is_end') and new_state.definition.is_end):
                return context.current_state_instance
                
    async def trace_execution(
        self,
        data: Union[Dict[str, Any], Record],
        initial_state: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute with full tracing enabled.
        
        Args:
            data: Input data
            initial_state: Optional starting state
            
        Returns:
            List of trace entries
        """
        self.execution_mode = ExecutionMode.TRACE
        self._trace_buffer.clear()
        
        async with self.execution_context(data, initial_state=initial_state) as context:
            # Run to completion
            while True:
                new_state = await self.step(context)
                if not new_state or new_state.definition.is_end:
                    break
                    
        return self._trace_buffer
        
    async def profile_execution(
        self,
        data: Union[Dict[str, Any], Record],
        initial_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute with performance profiling.
        
        Args:
            data: Input data
            initial_state: Optional starting state
            
        Returns:
            Profiling data
        """
        import time
        
        self.execution_mode = ExecutionMode.PROFILE
        self._profile_data.clear()
        
        async with self.execution_context(data, initial_state=initial_state) as context:
            start_time = time.time()
            transitions = 0
            
            # Track per-state timing
            state_times = {}
            state_start = time.time()
            
            while True:
                current_state_name = context.current_state.definition.name
                
                # Step
                new_state = await self.step(context)
                
                # Record state timing
                state_duration = time.time() - state_start
                if current_state_name not in state_times:
                    state_times[current_state_name] = []
                state_times[current_state_name].append(state_duration)
                
                if not new_state or new_state.definition.is_end:
                    break
                    
                transitions += 1
                state_start = time.time()
                
        total_time = time.time() - start_time
        
        # Compute statistics
        self._profile_data = {
            'total_time': total_time,
            'transitions': transitions,
            'avg_transition_time': total_time / transitions if transitions > 0 else 0,
            'state_times': {
                state: {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
                for state, times in state_times.items()
            }
        }
        
        return self._profile_data
        
    def get_available_transitions(
        self,
        state_name: str
    ) -> List[Dict[str, Any]]:
        """Get available transitions from a state.
        
        Args:
            state_name: Name of state
            
        Returns:
            List of available transition information
        """
        arcs = self.fsm.get_outgoing_arcs(state_name)
        return [
            {
                'name': arc.name,
                'target': arc.target_state,
                'has_pre_test': arc.pre_test is not None,
                'has_transform': arc.transform is not None
            }
            for arc in arcs
        ]
        
    def inspect_state(self, state_name: str) -> Dict[str, Any]:
        """Inspect a state's configuration.
        
        Args:
            state_name: Name of state to inspect
            
        Returns:
            State configuration details
        """
        state = self.fsm.get_state(state_name)
        return {
            'name': state.name,
            'is_start': state.is_start,
            'is_end': state.is_end,
            'has_schema': state.schema is not None,
            'has_validator': state.validator is not None,
            'resources': list(state.resources) if state.resources else [],
            'metadata': state.metadata
        }
        
    def visualize_fsm(self) -> str:
        """Generate a visual representation of the FSM.
        
        Returns:
            GraphViz DOT format string
        """
        lines = ['digraph FSM {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=circle];')
        
        # Add states
        for state in self.fsm.states.values():
            attrs = []
            if state.is_start:
                attrs.append('style=filled')
                attrs.append('fillcolor=green')
            elif state.is_end:
                attrs.append('shape=doublecircle')
                attrs.append('style=filled')
                attrs.append('fillcolor=red')
                
            if attrs:
                lines.append(f'  {state.name} [{",".join(attrs)}];')
            else:
                lines.append(f'  {state.name};')
                
        # Add arcs
        for state_name, state in self.fsm.states.items():
            for arc in self.fsm.get_outgoing_arcs(state_name):
                label = arc.name if arc.name else ""
                lines.append(f'  {state_name} -> {arc.target_state} [label="{label}"];')
                
        lines.append('}')
        return '\n'.join(lines)
        
    async def validate_network(self) -> Dict[str, Any]:
        """Validate the FSM network for consistency.
        
        Returns:
            Validation results
        """
        issues = []
        
        # Check for unreachable states
        reachable = set()
        to_visit = [s.name for s in self.fsm.states.values() if s.is_start]
        
        while to_visit:
            state = to_visit.pop(0)
            if state in reachable:
                continue
            reachable.add(state)
            
            arcs = self.fsm.get_outgoing_arcs(state)
            for arc in arcs:
                if arc.target_state not in reachable:
                    to_visit.append(arc.target_state)
                    
        unreachable = set(self.fsm.states.keys()) - reachable
        if unreachable:
            issues.append({
                'type': 'unreachable_states',
                'states': list(unreachable)
            })
            
        # Check for dead ends (non-end states with no outgoing arcs)
        for state_name, state in self.fsm.states.items():
            if not state.is_end:
                arcs = self.fsm.get_outgoing_arcs(state_name)
                if not arcs:
                    issues.append({
                        'type': 'dead_end',
                        'state': state_name
                    })
                    
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': {
                'total_states': len(self.fsm.states),
                'reachable_states': len(reachable),
                'unreachable_states': len(unreachable),
                'start_states': sum(1 for s in self.fsm.states.values() if s.is_start),
                'end_states': sum(1 for s in self.fsm.states.values() if s.is_end)
            }
        }
        
    def get_history(self) -> Optional[ExecutionHistory]:
        """Get execution history if enabled.
        
        Returns:
            Execution history or None
        """
        return self._history
        
    async def save_history(self) -> bool:
        """Save execution history to storage.
        
        Returns:
            True if saved successfully
        """
        if self._history and self._storage:
            return await self._storage.save(self._history)
        return False
        
    async def load_history(self, history_id: str) -> bool:
        """Load execution history from storage.
        
        Args:
            history_id: History identifier
            
        Returns:
            True if loaded successfully
        """
        if self._storage:
            history = await self._storage.load(history_id)
            if history:
                self._history = history
                return True
        return False


class FSMDebugger:
    """Interactive debugger for FSM execution."""
    
    def __init__(self, fsm: AdvancedFSM):
        """Initialize debugger.
        
        Args:
            fsm: Advanced FSM instance to debug
        """
        self.fsm = fsm
        self.context = None
        self.watch_vars = {}
        self.command_history = []
        
    async def start(
        self,
        data: Union[Dict[str, Any], Record],
        initial_state: Optional[str] = None
    ):
        """Start debugging session.
        
        Args:
            data: Initial data
            initial_state: Optional starting state
        """
        self.context = await self.fsm.execution_context(
            data,
            initial_state=initial_state
        ).__aenter__()
        
        print(f"Debugger started at state: {self.context.current_state.definition.name}")
        
    async def step(self) -> None:
        """Execute single step."""
        if not self.context:
            print("No active debugging session")
            return
            
        new_state = await self.fsm.step(self.context)
        if new_state:
            print(f"Transitioned to: {new_state.definition.name}")
        else:
            print("No transition available")
            
    async def continue_execution(self) -> None:
        """Continue execution until breakpoint."""
        if not self.context:
            print("No active debugging session")
            return
            
        final_state = await self.fsm.run_until_breakpoint(self.context)
        print(f"Stopped at: {final_state.definition.name}")
        
    def inspect(self, path: str) -> Any:
        """Inspect data at path.
        
        Args:
            path: Dot-separated path to data field
            
        Returns:
            Value at path
        """
        if not self.context:
            return None
            
        data = self.context.current_state.data
        for key in path.split('.'):
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data
        
    def watch(self, name: str, path: str) -> None:
        """Add a watch expression.
        
        Args:
            name: Watch name
            path: Data path to watch
        """
        self.watch_vars[name] = path
        
    def print_watches(self) -> None:
        """Print all watch values."""
        for name, path in self.watch_vars.items():
            value = self.inspect(path)
            print(f"{name}: {value}")
            
    def print_state(self) -> None:
        """Print current state information."""
        if not self.context:
            print("No active debugging session")
            return
            
        state = self.context.current_state
        print(f"State: {state.definition.name}")
        print(f"Data: {state.data}")
        
        # Print available transitions
        transitions = self.fsm.get_available_transitions(state.definition.name)
        if transitions:
            print("Available transitions:")
            for t in transitions:
                print(f"  - {t['name']} -> {t['target']}")
        else:
            print("No available transitions")


def create_advanced_fsm(
    config: Union[str, Path, Dict[str, Any], FSM],
    **kwargs
) -> AdvancedFSM:
    """Factory function to create an AdvancedFSM instance.
    
    Args:
        config: Configuration, FSM instance, or path
        **kwargs: Additional arguments
        
    Returns:
        Configured AdvancedFSM instance
    """
    if isinstance(config, FSM):
        fsm = config
    else:
        from ..config.loader import ConfigLoader
        from ..config.builder import FSMBuilder
        
        if isinstance(config, (str, Path)):
            config_dict = ConfigLoader.load_file(str(config))
        else:
            config_dict = config
            
        builder = FSMBuilder()
        fsm = builder.build_from_config(config_dict)
        
    return AdvancedFSM(fsm, **kwargs)