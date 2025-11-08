r"""Advanced API for debugging and profiling FSM workflows.

This module provides advanced interfaces for users who need fine-grained
control over FSM execution, debugging capabilities, performance profiling,
and detailed execution monitoring. Use this API when building complex
workflows or troubleshooting execution issues.

Architecture:
    The AdvancedFSM API provides three levels of control beyond SimpleFSM/AsyncSimpleFSM:

    **1. Step-by-Step Execution:**
    Execute FSM transitions one at a time with full inspection of state
    between each step. Essential for debugging complex state machines.

    **2. Breakpoint Debugging:**
    Set breakpoints at specific states and inspect execution context when
    reached. Similar to debugger breakpoints in code.

    **3. Performance Profiling:**
    Measure execution time for each state and transition, identify bottlenecks,
    and optimize workflow performance.

    **When to Use AdvancedFSM:**
    - Debugging complex workflows with unexpected behavior
    - Profiling performance to identify slow states
    - Building custom execution strategies
    - Implementing sophisticated error recovery
    - Monitoring production workflows in detail
    - Testing and validating FSM configurations

    **When NOT to Use AdvancedFSM:**
    - Simple production workflows (use AsyncSimpleFSM)
    - High-throughput batch processing (use SimpleFSM.process_batch)
    - Prototyping and scripts (use SimpleFSM)

Execution Modes:
    AdvancedFSM supports five execution modes:

    **STEP_BY_STEP:**
    Execute one transition at a time. After each step, execution pauses and
    returns control to caller with current state and data. Call step() to
    continue to next transition.

    **BREAKPOINT:**
    Set breakpoints at specific states. Execution runs normally until reaching
    a breakpoint state, then pauses. Inspect state and data, then continue.

    **TRACE:**
    Record detailed execution trace including all state transitions, data
    transformations, and timing information. Useful for auditing and debugging.

    **PROFILE:**
    Measure and record performance metrics for each state and transition.
    Identify bottlenecks and optimize slow operations.

    **DEBUG:**
    Enable verbose logging and detailed error messages. All execution events
    are logged with full context for debugging.

Common Debugging Patterns:
    **Step-by-Step Debugging:**
    ```python
    from dataknobs_fsm.api.advanced import AdvancedFSM, ExecutionMode
    from dataknobs_fsm.config.loader import ConfigLoader
    from dataknobs_fsm.config.builder import FSMBuilder

    # Load and build FSM
    config = ConfigLoader().load_from_file('pipeline.yaml')
    fsm = FSMBuilder().build(config)

    # Create advanced FSM in step mode
    advanced = AdvancedFSM(fsm, execution_mode=ExecutionMode.STEP_BY_STEP)

    # Execute one step at a time
    data = {'input': 'test data'}
    while True:
        result = advanced.step(data)

        print(f"Step: {result.from_state} -> {result.to_state}")
        print(f"Data before: {result.data_before}")
        print(f"Data after: {result.data_after}")
        print(f"Duration: {result.duration:.3f}s")

        if result.is_complete:
            break
        if not result.success:
            print(f"Error: {result.error}")
            break

        data = result.data_after
    ```

    **Breakpoint Debugging:**
    ```python
    # Set breakpoints at specific states
    advanced = AdvancedFSM(fsm, execution_mode=ExecutionMode.BREAKPOINT)
    advanced.add_breakpoint('transform_state')
    advanced.add_breakpoint('validate_state')

    # Run until breakpoint
    result = advanced.run_until_breakpoint({'input': 'data'})

    if result.at_breakpoint:
        print(f"Stopped at: {result.to_state}")
        print(f"Current data: {result.data_after}")

        # Inspect, modify data, then continue
        result.data_after['debug_flag'] = True
        final = advanced.run_until_breakpoint(result.data_after)
    ```

    **Performance Profiling:**
    ```python
    # Profile execution to find bottlenecks
    advanced = AdvancedFSM(fsm, execution_mode=ExecutionMode.PROFILE)

    # Execute with profiling
    profile_data = advanced.profile_execution({'input': 'data'})

    # Analyze results
    print("Performance Profile:")
    for state, metrics in profile_data['states'].items():
        print(f"{state}:")
        print(f"  Time: {metrics['duration']:.3f}s")
        print(f"  Calls: {metrics['call_count']}")
        print(f"  Avg: {metrics['avg_duration']:.3f}s")

    # Find slowest state
    slowest = max(profile_data['states'].items(),
                 key=lambda x: x[1]['duration'])
    print(f"\nBottleneck: {slowest[0]} ({slowest[1]['duration']:.2f}s)")
    ```

    **Execution Tracing:**
    ```python
    # Record full execution trace
    advanced = AdvancedFSM(fsm, execution_mode=ExecutionMode.TRACE)
    trace = advanced.trace_execution({'input': 'data'})

    # Analyze trace
    print(f"Total steps: {len(trace['steps'])}")
    print(f"Total time: {trace['total_duration']:.2f}s")
    print(f"\nExecution path:")
    for step in trace['steps']:
        print(f"  {step['timestamp']}: {step['from']} -> {step['to']}")
        if step.get('error'):
            print(f"    ERROR: {step['error']}")
    ```

Execution Hooks:
    Monitor execution events in real-time with hooks:

    ```python
    from dataknobs_fsm.api.advanced import ExecutionHook

    # Define hook functions
    def on_state_enter(state_name, data):
        print(f"Entering: {state_name}")

    def on_state_exit(state_name, data, duration):
        print(f"Exiting: {state_name} ({duration:.3f}s)")

    def on_error(state_name, error):
        print(f"Error in {state_name}: {error}")

    # Create hooks
    hooks = ExecutionHook(
        on_state_enter=on_state_enter,
        on_state_exit=on_state_exit,
        on_error=on_error
    )

    # Set hooks
    advanced = AdvancedFSM(fsm)
    advanced.set_hooks(hooks)

    # Hooks will be called during execution
    result = advanced.execute({'input': 'data'})
    ```

Advanced Use Cases:
    **Custom Execution Strategies:**
    Implement custom traversal strategies for special workflow patterns:

    ```python
    from dataknobs_fsm.execution.engine import TraversalStrategy

    class PriorityTraversal(TraversalStrategy):
        \"\"\"Execute high-priority states first.\"\"\"

        def select_next_arc(self, state, arcs):
            # Custom logic to select next transition
            return max(arcs, key=lambda a: a.priority)

    advanced = AdvancedFSM(fsm)
    advanced.set_execution_strategy(PriorityTraversal())
    ```

    **Transaction Management:**
    Configure transactional execution with rollback:

    ```python
    from dataknobs_fsm.core.transactions import TransactionStrategy

    # Configure transactions
    advanced.configure_transactions(
        strategy=TransactionStrategy.TWO_PHASE_COMMIT,
        isolation_level='READ_COMMITTED'
    )

    # Execution will use transactions
    try:
        result = advanced.execute(data)
    except Exception:
        # Automatic rollback on error
        pass
    ```

    **Resource Monitoring:**
    Monitor resource acquisition and release:

    ```python
    def on_resource_acquire(name, resource):
        print(f"Acquired: {name}")

    def on_resource_release(name, resource):
        print(f"Released: {name}")

    hooks = ExecutionHook(
        on_resource_acquire=on_resource_acquire,
        on_resource_release=on_resource_release
    )
    advanced.set_hooks(hooks)
    ```

Example:
    Complete debugging workflow:

    ```python
    from dataknobs_fsm.api.advanced import (
        AdvancedFSM, ExecutionMode, ExecutionHook
    )
    from dataknobs_fsm.config.loader import ConfigLoader
    from dataknobs_fsm.config.builder import FSMBuilder

    # Load FSM
    config = ConfigLoader().load_from_file('complex_pipeline.yaml')
    fsm = FSMBuilder().build(config)

    # Create advanced FSM with profiling
    advanced = AdvancedFSM(fsm, execution_mode=ExecutionMode.PROFILE)

    # Set up monitoring hooks
    errors = []

    def on_error(state, error):
        errors.append({'state': state, 'error': str(error)})

    hooks = ExecutionHook(on_error=on_error)
    advanced.set_hooks(hooks)

    # Add breakpoints at critical states
    advanced.add_breakpoint('data_validation')
    advanced.add_breakpoint('external_api_call')

    # Execute with monitoring
    try:
        profile_data = advanced.profile_execution({
            'input_file': 'data.json',
            'config': {'mode': 'strict'}
        })

        # Analyze performance
        print("Performance Analysis:")
        for state, metrics in profile_data['states'].items():
            if metrics['duration'] > 1.0:  # Slow states
                print(f"⚠️  {state}: {metrics['duration']:.2f}s")

        # Check for errors
        if errors:
            print("\nErrors encountered:")
            for err in errors:
                print(f"  {err['state']}: {err['error']}")

    except Exception as e:
        print(f"Fatal error: {e}")
        # Get trace for debugging
        trace = advanced.get_trace()
        print(f"Failed at: {trace['steps'][-1]}")
    ```

See Also:
    - :class:`SimpleFSM`: Simple synchronous API
    - :class:`AsyncSimpleFSM`: Async production API
    - :class:`ExecutionMode`: Available execution modes
    - :class:`ExecutionHook`: Hook system for monitoring
    - :mod:`dataknobs_fsm.execution.engine`: Execution engine
    - :mod:`dataknobs_fsm.execution.history`: Execution history tracking
"""

import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dataknobs_data import Record

from ..core.context_factory import ContextFactory
from ..core.data_modes import DataHandler, DataHandlingMode
from ..core.fsm import FSM
from ..core.modes import ProcessingMode
from ..core.state import StateInstance
from ..core.transactions import TransactionManager, TransactionStrategy
from ..execution.async_engine import AsyncExecutionEngine
from ..execution.context import ExecutionContext
from ..execution.engine import ExecutionEngine, TraversalStrategy
from ..execution.history import ExecutionHistory
from ..resources.base import IResourceProvider
from ..resources.manager import ResourceManager
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
    on_state_enter: Callable | None = None
    on_state_exit: Callable | None = None
    on_arc_execute: Callable | None = None
    on_error: Callable | None = None
    on_resource_acquire: Callable | None = None
    on_resource_release: Callable | None = None
    on_transaction_begin: Callable | None = None
    on_transaction_commit: Callable | None = None
    on_transaction_rollback: Callable | None = None


@dataclass
class StepResult:
    """Result from a single step execution."""
    from_state: str
    to_state: str
    transition: str
    data_before: dict[str, Any] = field(default_factory=dict)
    data_after: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    success: bool = True
    error: str | None = None
    at_breakpoint: bool = False
    is_complete: bool = False


class AdvancedFSM:
    """Advanced FSM interface with full control capabilities."""

    def __init__(
        self,
        fsm: FSM,
        execution_mode: ExecutionMode = ExecutionMode.STEP_BY_STEP,
        custom_functions: dict[str, Callable] | None = None
    ):
        """Initialize AdvancedFSM.
        
        Args:
            fsm: Core FSM instance
            execution_mode: Execution mode for advanced control
            custom_functions: Optional custom functions to register
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
        self._custom_functions = custom_functions or {}

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
        **config: Any
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
        resource: IResourceProvider | dict[str, Any]
    ) -> None:
        """Register a custom resource.
        
        Args:
            name: Resource name
            resource: Resource instance or configuration
        """
        if isinstance(resource, dict):
            # Use ResourceManager factory method
            self._resource_manager.register_from_dict(name, resource)
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

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()

    @property
    def breakpoints(self) -> set:
        """Get the current breakpoints."""
        return self._breakpoints.copy()

    @property
    def hooks(self) -> ExecutionHook:
        """Get the current execution hooks."""
        return self._hooks

    @property
    def history_enabled(self) -> bool:
        """Check if history tracking is enabled."""
        return self._history is not None

    @property
    def max_history_depth(self) -> int:
        """Get the maximum history depth."""
        return self._history.max_depth if self._history else 0

    @property
    def execution_history(self) -> list:
        """Get the execution history steps."""
        return self._history.steps if self._history else []

    def enable_history(
        self,
        storage: IHistoryStorage | None = None,
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

    def disable_history(self) -> None:
        """Disable history tracking."""
        self._history = None
        self._storage = None

    def create_context(
        self,
        data: dict[str, Any] | Record,
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        initial_state: str | None = None
    ) -> ExecutionContext:
        """Create an execution context for manual control (synchronous).

        Args:
            data: Initial data
            data_mode: Data handling mode
            initial_state: Starting state name

        Returns:
            ExecutionContext for manual execution
        """
        # Create context with appropriate data handling
        # Use SINGLE processing mode as default
        processing_mode = ProcessingMode.SINGLE

        context = ContextFactory.create_context(
            self.fsm,
            data,
            data_mode=processing_mode
        )

        # Set initial state if provided
        if initial_state:
            context.set_state(initial_state)
        else:
            # Find and set initial state using shared helper
            initial_state = self._find_initial_state()
            if initial_state:
                context.set_state(initial_state)

        # Update state instance using shared helper
        if context.current_state:
            self._update_state_instance(context, context.current_state)

        # Register custom functions if any
        if self._custom_functions:
            if not hasattr(self.fsm, 'function_registry'):
                self.fsm.function_registry = {}
            self.fsm.function_registry.update(self._custom_functions)

        return context

    @asynccontextmanager
    async def execution_context(
        self,
        data: dict[str, Any] | Record,
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        initial_state: str | None = None
    ) -> AsyncGenerator[ExecutionContext, None]:
        """Create an execution context for manual control.

        Args:
            data: Initial data
            data_mode: Data handling mode
            initial_state: Starting state name

        Yields:
            ExecutionContext for manual execution
        """
        # Create context using factory
        context = ContextFactory.create_context(
            fsm=self.fsm,
            data=data,
            initial_state=initial_state,
            data_mode=ProcessingMode.SINGLE,
            resource_manager=self._resource_manager
        )

        # Set transaction manager if configured
        if self._transaction_manager:
            context.transaction_manager = self._transaction_manager  # type: ignore[unreachable]

        # Get the state instance for the hook
        state_instance = context.current_state_instance
        if not state_instance:
            # Create state instance if not set by factory
            state_instance = self.fsm.create_state_instance(
                context.current_state,  # type: ignore
                context.data.copy() if isinstance(context.data, dict) else {}
            )
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
        arc_name: str | None = None
    ) -> StateInstance | None:
        """Execute a single transition step.

        Args:
            context: Execution context
            arc_name: Optional specific arc to follow

        Returns:
            New state instance or None if no transition
        """
        # Store the current state before the transition
        state_before = context.current_state

        # Use the async execution engine to execute one step
        # This ensures consistent execution logic across all FSM types
        _success, _result = await self._async_engine.execute(
            context=context,
            data=None,  # Don't override context data
            max_transitions=1,  # Execute exactly one transition
            arc_name=arc_name  # Pass arc_name for filtering
        )

        # Check if we actually transitioned to a new state
        if context.current_state != state_before and context.current_state is not None:
            # Update state instance using shared helper
            self._update_state_instance(context, context.current_state)
            new_state = context.current_state_instance

            # Track in history using shared helper
            if context.current_state is not None:
                self._record_history_step(context.current_state, arc_name, context)

            # Add to trace using shared helper (we need to adjust the helper slightly)
            if self.execution_mode == ExecutionMode.TRACE:
                self._trace_buffer.append({
                    'from': state_before,
                    'to': context.current_state,
                    'arc': arc_name or 'transition',
                    'data': context.data
                })

            # Call state enter hook (async version)
            if self._hooks.on_state_enter:
                await self._hooks.on_state_enter(new_state)

            return new_state

        # No transition occurred
        return None

    async def run_until_breakpoint(
        self,
        context: ExecutionContext,
        max_steps: int = 1000
    ) -> StateInstance | None:
        """Run execution until a breakpoint is hit.

        Args:
            context: Execution context
            max_steps: Maximum steps to execute (safety limit)

        Returns:
            State instance where execution stopped
        """
        for _ in range(max_steps):
            # Check if current state is a breakpoint
            if context.current_state in self._breakpoints:
                return context.current_state_instance

            # Step to next state
            new_state = await self.step(context)

            # Check if we reached an end state or no transition occurred
            if not new_state or self._is_at_end_state(context):
                return context.current_state_instance

        # Hit max steps limit
        return context.current_state_instance

    async def trace_execution(
        self,
        data: dict[str, Any] | Record,
        initial_state: str | None = None
    ) -> list[dict[str, Any]]:
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
        data: dict[str, Any] | Record,
        initial_state: str | None = None
    ) -> dict[str, Any]:
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
                # Get current state name
                if isinstance(context.current_state, str):
                    current_state_name = context.current_state
                else:
                    current_state_name = context.current_state if context.current_state else "unknown"

                # Step
                new_state = await self.step(context)

                # Record state timing
                state_duration = time.time() - state_start
                if current_state_name not in state_times:
                    state_times[current_state_name] = []
                state_times[current_state_name].append(state_duration)

                if not new_state or (hasattr(new_state, 'definition') and new_state.definition.is_end):
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
    ) -> list[dict[str, Any]]:
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

    def inspect_state(self, state_name: str) -> dict[str, Any]:
        """Inspect a state's configuration.
        
        Args:
            state_name: Name of state to inspect
            
        Returns:
            State configuration details
        """
        state = self.fsm.get_state(state_name)
        if not state:
            return {'error': f'State {state_name} not found'}

        return {
            'name': state.name,
            'is_start': self.fsm.is_start_state(state_name),
            'is_end': self.fsm.is_end_state(state_name),
            'has_transform': len(state.transform_functions) > 0,
            'has_validator': len(state.validation_functions) > 0,
            'resources': [r.name for r in state.resource_requirements] if state.resource_requirements else [],
            'metadata': state.metadata,
            'arcs': state.arcs
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
        for state_name in self.fsm.states:
            for arc in self.fsm.get_outgoing_arcs(state_name):
                label = arc.name if arc.name else ""
                lines.append(f'  {state_name} -> {arc.target_state} [label="{label}"];')

        lines.append('}')
        return '\n'.join(lines)

    async def validate_network(self) -> dict[str, Any]:
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
                'start_states': sum(1 for s in self.fsm.states.values() if s.is_start),  # type: ignore
                'end_states': sum(1 for s in self.fsm.states.values() if s.is_end)  # type: ignore
            }
        }

    def get_history(self) -> ExecutionHistory | None:
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
        if self._history and self._storage:  # type: ignore[unreachable]
            return await self._storage.save(self._history)  # type: ignore[unreachable]
        return False

    async def load_history(self, history_id: str) -> bool:
        """Load execution history from storage.
        
        Args:
            history_id: History identifier
            
        Returns:
            True if loaded successfully
        """
        if self._storage:
            history = await self._storage.load(history_id)  # type: ignore[unreachable]
            if history:
                self._history = history
                return True
        return False

    # ========== Shared Helper Methods ==========
    # These methods contain logic shared between sync and async implementations

    def _get_available_transitions(
        self,
        context: ExecutionContext,
        arc_name: str | None = None
    ) -> list:
        """Get available transitions from current state (shared logic).

        Args:
            context: Execution context
            arc_name: Optional specific arc to filter for

        Returns:
            List of available arcs
        """
        transitions = []
        if not context.current_state:
            return transitions

        # Get arcs from current state
        arcs = self.fsm.get_outgoing_arcs(context.current_state)

        for arc in arcs:
            # Filter by arc name if specified
            if arc_name and arc.name != arc_name:
                continue

            # Check arc condition
            if arc.pre_test:
                # Get function registry
                registry = getattr(self.fsm, 'function_registry', {})
                if hasattr(registry, 'functions'):
                    functions = registry.functions
                else:
                    functions = registry

                # Check in registry and custom functions
                test_func = functions.get(arc.pre_test) or self._custom_functions.get(arc.pre_test)
                if test_func:
                    try:
                        if test_func(context.data, context):
                            transitions.append(arc)
                    except Exception:
                        pass
            else:
                # No condition, arc is always available
                transitions.append(arc)

        return transitions

    def _execute_arc_transform(
        self,
        arc,
        context: ExecutionContext
    ) -> tuple[bool, Any]:
        """Execute arc transform function (shared logic).

        Args:
            arc: Arc with potential transform
            context: Execution context

        Returns:
            Tuple of (success, result_or_error)
        """
        if not arc.transform:
            return True, context.data

        # Get function registry
        registry = getattr(self.fsm, 'function_registry', {})
        if hasattr(registry, 'functions'):
            functions = registry.functions
        else:
            functions = registry

        # Look for transform in registry or custom functions
        transform_func = functions.get(arc.transform) or self._custom_functions.get(arc.transform)

        if transform_func:
            try:
                result = transform_func(context.data, context)
                return True, result
            except Exception as e:
                return False, str(e)

        return True, context.data

    def _update_state_instance(
        self,
        context: ExecutionContext,
        state_name: str
    ) -> None:
        """Update the current state instance in context (shared logic).

        Args:
            context: Execution context
            state_name: Name of the new state
        """
        state_def = self.fsm.states.get(state_name)
        if state_def:
            context.current_state_instance = StateInstance(
                definition=state_def,
                data=context.data
            )
            # Mark if it's an end state
            context.metadata['is_end_state'] = state_def.is_end

    def _is_at_end_state(self, context: ExecutionContext) -> bool:
        """Check if context is at an end state (shared logic).

        Args:
            context: Execution context

        Returns:
            True if at an end state
        """
        if not context.current_state:
            return False

        state = self.fsm.states.get(context.current_state)
        if state:
            return state.is_end

        return context.metadata.get('is_end_state', False)

    def _record_trace_entry(
        self,
        from_state: str,
        to_state: str,
        arc_name: str | None,
        context: ExecutionContext
    ) -> None:
        """Record a trace entry if in trace mode (shared logic).

        Args:
            from_state: State transitioning from
            to_state: State transitioning to
            arc_name: Name of arc taken
            context: Execution context
        """
        if self.execution_mode == ExecutionMode.TRACE:
            self._trace_buffer.append({
                'from_state': from_state,
                'to_state': to_state,
                'transition': arc_name or f"{from_state}->{to_state}",
                'data': context.get_data_snapshot(),
                'timestamp': time.time()
            })

    def _record_history_step(
        self,
        state_name: str,
        arc_name: str | None,
        context: ExecutionContext
    ) -> None:
        """Record a history step if history is enabled (shared logic).

        Args:
            state_name: Current state name
            arc_name: Arc taken
            context: Execution context
        """
        if self._history:
            step = self._history.add_step(  # type: ignore[unreachable]
                state_name=state_name,
                network_name=getattr(context, 'network_name', 'main'),
                data=context.data
            )
            step.complete(arc_taken=arc_name or 'transition')

    def _call_hook_sync(
        self,
        hook_name: str,
        *args: Any
    ) -> None:
        """Call a hook synchronously if it exists (shared logic).

        Args:
            hook_name: Name of hook attribute
            args: Arguments to pass to hook
        """
        hook = getattr(self._hooks, hook_name, None)
        if hook:
            try:
                hook(*args)
            except Exception:
                pass  # Silently ignore hook errors

    def _find_initial_state(self) -> str | None:
        """Find the initial state in the FSM (shared logic).

        Returns:
            Name of initial state or None
        """
        for state_name, state in self.fsm.states.items():
            if state.is_start:
                return state_name
        return None

    # ========== Synchronous Execution Methods ==========

    def execute_step_sync(
        self,
        context: ExecutionContext,
        arc_name: str | None = None
    ) -> StepResult:
        """Execute a single transition step synchronously.

        Args:
            context: Execution context
            arc_name: Optional specific arc to follow

        Returns:
            StepResult with transition details
        """
        start_time = time.time()
        from_state = context.current_state or "initial"
        data_before = context.get_data_snapshot()

        try:
            # Initialize state if needed
            if not context.current_state:
                initial_state = self._find_initial_state()
                if initial_state:
                    context.set_state(initial_state)
                    self._update_state_instance(context, initial_state)
                    # Execute transforms for initial state
                    if hasattr(self._engine, '_execute_state_transforms'):
                        self._engine._execute_state_transforms(context, initial_state)
                else:
                    return StepResult(
                        from_state=from_state,
                        to_state=from_state,
                        transition="error",
                        data_before=data_before,
                        data_after=context.get_data_snapshot(),
                        duration=time.time() - start_time,
                        success=False,
                        error="No initial state found"
                    )

            # Use shared logic to get transitions
            transitions = self._get_available_transitions(context, arc_name)

            if not transitions:
                # No transitions available
                return StepResult(
                    from_state=from_state,
                    to_state=from_state,
                    transition="none",
                    data_before=data_before,
                    data_after=context.get_data_snapshot(),
                    duration=time.time() - start_time,
                    success=True,
                    is_complete=self._is_at_end_state(context)
                )

            # Take first valid transition (could be enhanced with strategy selection)
            arc = transitions[0]

            # Execute arc transform using shared logic
            success, result = self._execute_arc_transform(arc, context)
            if success:
                context.data = result
            else:
                return StepResult(
                    from_state=from_state,
                    to_state=from_state,
                    transition=arc.name or "error",
                    data_before=data_before,
                    data_after=context.get_data_snapshot(),
                    duration=time.time() - start_time,
                    success=False,
                    error=result
                )

            # Update state
            context.set_state(arc.target_state)
            self._update_state_instance(context, arc.target_state)

            # Execute state transforms when entering the new state
            # This is critical for sync execution to match async behavior
            if hasattr(self._engine, '_execute_state_transforms'):
                self._engine._execute_state_transforms(context, arc.target_state)

            # Check if we hit a breakpoint
            at_breakpoint = arc.target_state in self._breakpoints

            # Record in trace buffer if in trace mode
            self._record_trace_entry(from_state, arc.target_state, arc.name, context)

            # Record in history if enabled
            self._record_history_step(arc.target_state, arc.name, context)

            # Call hooks if configured
            self._call_hook_sync('on_state_exit', from_state)
            self._call_hook_sync('on_state_enter', arc.target_state)

            return StepResult(
                from_state=from_state,
                to_state=arc.target_state,
                transition=arc.name or f"{from_state}->{arc.target_state}",
                data_before=data_before,
                data_after=context.get_data_snapshot(),
                duration=time.time() - start_time,
                success=True,
                at_breakpoint=at_breakpoint,
                is_complete=self._is_at_end_state(context)
            )

        except Exception as e:
            self._call_hook_sync('on_error', e)

            return StepResult(
                from_state=from_state,
                to_state=from_state,
                transition="error",
                data_before=data_before,
                data_after=context.get_data_snapshot(),
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def run_until_breakpoint_sync(
        self,
        context: ExecutionContext,
        max_steps: int = 1000
    ) -> StateInstance | None:
        """Run execution until a breakpoint is hit (synchronous).

        Args:
            context: Execution context
            max_steps: Maximum steps to execute

        Returns:
            State instance where execution stopped
        """
        for _ in range(max_steps):
            # Check if at breakpoint
            if context.current_state in self._breakpoints:
                return context.current_state_instance

            # Execute step
            result = self.execute_step_sync(context)

            # Check for completion or error
            if not result.success or result.is_complete:
                return context.current_state_instance

            # Check if stuck
            if result.from_state == result.to_state and result.transition == "none":
                return context.current_state_instance

        return context.current_state_instance

    def trace_execution_sync(
        self,
        data: dict[str, Any] | Record,
        initial_state: str | None = None,
        max_steps: int = 1000
    ) -> list[dict[str, Any]]:
        """Execute with full tracing enabled (synchronous).

        Args:
            data: Input data
            initial_state: Optional starting state
            max_steps: Maximum steps to execute

        Returns:
            List of trace entries
        """
        self.execution_mode = ExecutionMode.TRACE
        self._trace_buffer.clear()

        context = self.create_context(data, initial_state=initial_state)

        for _ in range(max_steps):
            # Execute step (trace recording happens inside execute_step_sync)
            result = self.execute_step_sync(context)

            # Check termination conditions
            if not result.success or result.is_complete:
                break

            if result.from_state == result.to_state and result.transition == "none":
                break

        return self._trace_buffer

    def profile_execution_sync(
        self,
        data: dict[str, Any] | Record,
        initial_state: str | None = None,
        max_steps: int = 1000
    ) -> dict[str, Any]:
        """Execute with performance profiling (synchronous).

        Args:
            data: Input data
            initial_state: Optional starting state
            max_steps: Maximum steps to execute

        Returns:
            Profiling data
        """
        self.execution_mode = ExecutionMode.PROFILE
        self._profile_data.clear()

        context = self.create_context(data, initial_state=initial_state)

        start_time = time.time()
        transitions = 0
        state_times = {}
        transition_times = []

        for _ in range(max_steps):
            state_start = time.time()
            current_state = context.current_state

            # Execute step
            result = self.execute_step_sync(context)

            # Record timings
            if current_state:
                if current_state not in state_times:
                    state_times[current_state] = []
                state_times[current_state].append(time.time() - state_start)

            if result.success and result.from_state != result.to_state:
                transition_times.append(result.duration)
                transitions += 1

            # Check termination
            if not result.success or result.is_complete:
                break

            if result.from_state == result.to_state and result.transition == "none":
                break

        # Calculate statistics
        self._profile_data = {
            'total_time': time.time() - start_time,
            'transitions': transitions,
            'states_visited': len(state_times),
            'avg_transition_time': sum(transition_times) / len(transition_times) if transition_times else 0,
            'state_times': {
                state: {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
                for state, times in state_times.items()
            },
            'final_state': context.current_state,
            'final_data': context.get_data_snapshot()
        }

        return self._profile_data


class FSMDebugger:
    """Interactive debugger for FSM execution (fully synchronous)."""

    def __init__(self, fsm: AdvancedFSM):
        """Initialize debugger.

        Args:
            fsm: Advanced FSM instance to debug
        """
        self.fsm = fsm
        self.context: ExecutionContext | None = None
        self.watch_vars: dict[str, Any] = {}
        self.command_history: list[str] = []
        self.step_count: int = 0
        self.execution_history: list[StepResult] = []

    @property
    def current_state(self) -> str | None:
        """Get the current state name."""
        if not self.context:
            return None
        return self.context.get_current_state()

    @property
    def watches(self) -> dict[str, Any]:
        """Get current watch variable values."""
        return self.watch_vars.copy()

    def start(
        self,
        data: dict[str, Any] | Record,
        initial_state: str | None = None
    ) -> None:
        """Start debugging session (synchronous).

        Args:
            data: Initial data
            initial_state: Optional starting state
        """
        self.context = self.fsm.create_context(data, initial_state=initial_state)
        self.step_count = 0
        self.execution_history.clear()

        print(f"Debugger started at state: {self.context.current_state or 'initial'}")
        print(f"Data: {self.context.get_data_snapshot()}")

    def step(self) -> StepResult:
        """Execute single step and return detailed result.

        Returns:
            StepResult with transition details
        """
        if not self.context:
            print("No active debugging session. Call start() first.")
            return StepResult(
                from_state="none",
                to_state="none",
                transition="error",
                success=False,
                error="No active debugging session"
            )

        result = self.fsm.execute_step_sync(self.context)
        self.step_count += 1
        self.execution_history.append(result)

        # Print step information
        if result.success:
            if result.from_state == result.to_state and result.transition == "none":
                print(f"Step {self.step_count}: No transition available from '{result.from_state}'")
            else:
                print(f"Step {self.step_count}: {result.from_state} -> {result.to_state} via '{result.transition}'")

            if result.at_breakpoint:
                print("*** Hit breakpoint ***")

            if result.is_complete:
                print("*** Reached end state ***")
        else:
            print(f"Step {self.step_count}: Error - {result.error}")

        # Check watches
        self._check_watches()

        return result

    def continue_to_breakpoint(self) -> StateInstance | None:
        """Continue execution until a breakpoint is hit.

        Returns:
            State instance where execution stopped
        """
        if not self.context:
            print("No active debugging session")
            return None

        print(f"Continuing from state: {self.context.current_state}")
        final_state = self.fsm.run_until_breakpoint_sync(self.context)

        if final_state:
            print(f"Stopped at: {self.context.current_state}")
            if self.context.current_state in self.fsm._breakpoints:
                print("*** At breakpoint ***")
            if self.context.is_complete():
                print("*** Execution complete ***")

        return final_state

    def inspect(self, path: str = "") -> Any:
        """Inspect data at path.

        Args:
            path: Dot-separated path to data field (empty for all data)

        Returns:
            Value at path
        """
        if not self.context:
            print("No active debugging session")
            return None

        data = self.context.data

        if not path:
            return data

        # Navigate path
        for key in path.split('.'):
            if isinstance(data, dict):
                data = data.get(key)
            elif hasattr(data, key):
                data = getattr(data, key)
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
        value = self.inspect(path)
        print(f"Watch '{name}' added: {path} = {value}")

    def unwatch(self, name: str) -> None:
        """Remove a watch expression.

        Args:
            name: Watch name to remove
        """
        if name in self.watch_vars:
            del self.watch_vars[name]
            print(f"Watch '{name}' removed")

    def _check_watches(self) -> None:
        """Check and print changed watch values."""
        if not self.watch_vars:
            return

        for name, path in self.watch_vars.items():
            value = self.inspect(path)
            print(f"  Watch '{name}': {path} = {value}")

    def print_watches(self) -> None:
        """Print all watch values."""
        if not self.watch_vars:
            print("No watches set")
            return

        for name, path in self.watch_vars.items():
            value = self.inspect(path)
            print(f"{name}: {path} = {value}")

    def print_state(self) -> None:
        """Print current state information."""
        if not self.context:
            print("No active debugging session")
            return

        print("\n=== State Information ===")
        print(f"Current State: {self.context.current_state}")
        print(f"Previous State: {self.context.previous_state}")
        print(f"Is Complete: {self.context.is_complete()}")
        print("\nData:")
        data = self.context.get_data_snapshot()
        for key, value in data.items():
            print(f"  {key}: {value}")

        # Print available transitions
        transitions = self.fsm._get_available_transitions(self.context)
        if transitions:
            print("\nAvailable Transitions:")
            for arc in transitions:
                print(f"  - {arc.name or 'unnamed'} -> {arc.target_state}")
        else:
            if self.context.is_complete():
                print("\nNo transitions (end state)")
            else:
                print("\nNo available transitions")

    def inspect_current_state(self) -> dict[str, Any]:
        """Get detailed information about current state.

        Returns:
            Dictionary with state details
        """
        if not self.context:
            return {"error": "No active debugging session"}

        return {
            'state': self.context.current_state,
            'previous_state': self.context.previous_state,
            'data': self.context.get_data_snapshot(),
            'is_complete': self.context.is_complete(),
            'step_count': self.step_count,
            'at_breakpoint': self.context.current_state in self.fsm._breakpoints,
            'available_transitions': [
                {'name': arc.name, 'target': arc.target_state}
                for arc in self.fsm._get_available_transitions(self.context)
            ]
        }

    def get_history(self, limit: int = 10) -> list[StepResult]:
        """Get recent execution history.

        Args:
            limit: Maximum number of steps to return

        Returns:
            List of recent step results
        """
        return self.execution_history[-limit:]

    def reset(self, data: dict[str, Any] | Record | None = None) -> None:
        """Reset debugger with new data.

        Args:
            data: New data (uses current data if None)
        """
        if data is None and self.context:
            data = self.context.data

        if data is None:
            print("No data available for reset")
            return

        self.start(data)


def create_advanced_fsm(
    config: str | Path | dict[str, Any] | FSM,
    custom_functions: dict[str, Callable] | None = None,
    **kwargs: Any
) -> AdvancedFSM:
    """Factory function to create an AdvancedFSM instance.
    
    Args:
        config: Configuration, FSM instance, or path
        custom_functions: Optional custom functions to register
        **kwargs: Additional arguments
        
    Returns:
        Configured AdvancedFSM instance
    """
    if isinstance(config, FSM):
        fsm = config
    else:
        from ..config.builder import FSMBuilder
        from ..config.loader import ConfigLoader

        loader = ConfigLoader()

        if isinstance(config, (str, Path)):
            config_obj = loader.load_from_file(str(config))
        else:
            # Load from dict
            config_obj = loader.load_from_dict(config)

        builder = FSMBuilder()

        # Register custom functions if provided
        if custom_functions:
            for name, func in custom_functions.items():
                builder.register_function(name, func)

        fsm = builder.build(config_obj)

    return AdvancedFSM(fsm, **kwargs)
