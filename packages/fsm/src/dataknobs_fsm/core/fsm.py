"""Core FSM class for managing state machines.

This module provides the core FSM class that serves as the foundation for all
dataknobs-fsm functionality. The FSM class manages state networks, function
registries, data modes, transactions, and resources.

Architecture:
    The FSM class is the central orchestrator in the dataknobs-fsm architecture:

    **Responsibilities:**
    - Network Management: Manages one or more state networks with transitions
    - Function Registry: Maintains registered functions for state operations
    - Data Mode Control: Configures how data flows through states (COPY/REFERENCE/DIRECT)
    - Transaction Management: Coordinates transactional state changes
    - Resource Management: Tracks and manages resource requirements
    - Execution Engines: Creates and manages sync/async execution engines

    **Design Philosophy:**
    - Separation of Concerns: FSM defines structure, engines handle execution
    - Multiple Networks: Support complex workflows with multiple state graphs
    - Configurable Processing: Data modes adapt to different use cases
    - Resource Awareness: Track dependencies before execution

    **Layer Architecture:**
    ```
    API Layer (simple/async_simple/advanced)
            ↓
    Core FSM (this module) ← defines structure
            ↓
    Execution Engines ← handle runtime
            ↓
    State Network ← graph of states/transitions
    ```

Processing Modes:
    The FSM supports three data processing modes:

    **SINGLE Mode (ProcessingMode.SINGLE):**
    - Process one data item at a time
    - Simplest execution model
    - Data flows through state network sequentially
    - Best for: Scripts, simple pipelines, prototypes

    **BATCH Mode (ProcessingMode.BATCH):**
    - Process multiple items as a batch
    - All items go through same state sequence
    - Can parallelize within batch
    - Best for: High-throughput processing, bulk operations

    **STREAM Mode (ProcessingMode.STREAM):**
    - Process data as streaming chunks
    - Memory-efficient for large datasets
    - Supports backpressure and flow control
    - Best for: Large files, real-time data, memory-constrained environments

Transaction Modes:
    The FSM supports configurable transaction handling:

    **NONE (TransactionMode.NONE):**
    - No transactional guarantees
    - State changes are immediate and permanent
    - Fastest performance
    - Best for: Read-only operations, idempotent workflows

    **OPTIMISTIC (TransactionMode.OPTIMISTIC):**
    - Changes committed at workflow end
    - Rollback on failure
    - Moderate performance overhead
    - Best for: Most production workflows

    **PESSIMISTIC (TransactionMode.PESSIMISTIC):**
    - Changes committed at each state
    - Highest consistency guarantees
    - Higher performance overhead
    - Best for: Critical workflows, strict consistency requirements

State Networks:
    An FSM can contain multiple state networks:

    **Main Network:**
    - Primary execution path
    - Default for most operations
    - Set via is_main parameter or first network added

    **Sub-Networks:**
    - Additional state graphs for modular workflows
    - Can represent sub-processes or alternate paths
    - Accessed by name

    **Network Operations:**
    - add_network(): Add a network to the FSM
    - remove_network(): Remove a network by name
    - get_network(): Get network by name (None = main network)

Function Registry:
    The FSM maintains a registry of functions used in state operations:

    **Function Types:**
    - Validation Functions: Check data validity before/after state entry
    - Transform Functions: Modify data during state processing
    - Test Functions: Determine state transitions
    - Pre/Post Hooks: Execute before/after state operations

    **Registration:**
    - Functions registered via function_registry.register()
    - Referenced by name in state/arc definitions
    - Validated during FSM.validate()

Resource Management:
    The FSM tracks resource requirements across all networks:

    **Resource Types:**
    - Database connections
    - API clients
    - File handles
    - External service connections
    - Memory allocations

    **Resource Tracking:**
    - Aggregated from all networks
    - Available via resource_requirements dict
    - Summary via get_resource_summary()
    - Managers set via resource_manager/transaction_manager

Execution Engines:
    The FSM creates execution engines on demand:

    **Synchronous Engine (ExecutionEngine):**
    - Created via get_engine()
    - Used by SimpleFSM API
    - Supports multiple traversal strategies:
      - DEPTH_FIRST: Deep exploration before backtracking
      - BREADTH_FIRST: Explore all neighbors before going deeper
      - RESOURCE_OPTIMIZED: Minimize resource acquisition/release
      - STREAM_OPTIMIZED: Optimize for streaming workflows

    **Asynchronous Engine (AsyncExecutionEngine):**
    - Created via get_async_engine()
    - Used by AsyncSimpleFSM API
    - Supports concurrent state execution
    - Efficient I/O handling

Validation:
    The FSM can validate its structure before execution:

    **Checks Performed:**
    - At least one network exists
    - Main network exists and is valid
    - All networks are internally valid
    - All referenced functions are registered
    - No dangling references

    **Usage:**
    ```python
    valid, errors = fsm.validate()
    if not valid:
        for error in errors:
            print(f"Validation error: {error}")
    ```

Note:
    This is an internal API primarily used by the builder and API layers.
    Users typically interact with SimpleFSM, AsyncSimpleFSM, or AdvancedFSM
    rather than instantiating FSM directly.

    The FSM class is designed to be serializable (to_dict/from_dict) for
    persistence and transmission.

See Also:
    - :class:`~dataknobs_fsm.api.simple.SimpleFSM`: Synchronous API wrapper
    - :class:`~dataknobs_fsm.api.async_simple.AsyncSimpleFSM`: Async API wrapper
    - :class:`~dataknobs_fsm.api.advanced.AdvancedFSM`: Advanced debugging API
    - :class:`~dataknobs_fsm.core.network.StateNetwork`: State graph container
    - :class:`~dataknobs_fsm.execution.engine.ExecutionEngine`: Sync execution engine
    - :class:`~dataknobs_fsm.execution.async_engine.AsyncExecutionEngine`: Async execution engine
"""

from typing import Any, Dict, List, Set, Tuple, Optional, TYPE_CHECKING

from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode

if TYPE_CHECKING:
    from dataknobs_fsm.execution.engine import ExecutionEngine
    from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateInstance, StateType
from dataknobs_fsm.functions.base import FunctionRegistry


class FSM:
    """Finite State Machine core class.

    This is the foundational class that defines the structure and configuration
    of a finite state machine. It serves as the container for state networks,
    functions, and execution configuration.

    Architecture:
        The FSM class uses a builder pattern approach where:
        1. FSM is constructed with configuration
        2. Networks are added with states and transitions
        3. Functions are registered for state operations
        4. FSM is validated before execution
        5. Execution engines are created on-demand

    Key Components:
        **State Networks:**
        - One or more state graphs (nodes=states, edges=transitions)
        - Main network for primary execution path
        - Sub-networks for modular workflows
        - Accessed via networks dict or get_network()

        **Function Registry:**
        - Central registry of all functions used in workflows
        - Validation, transform, and test functions
        - Referenced by name in state/arc definitions
        - Ensures all function references are valid

        **Processing Configuration:**
        - data_mode: How data flows (SINGLE/BATCH/STREAM)
        - transaction_mode: Transaction guarantees (NONE/OPTIMISTIC/PESSIMISTIC)
        - resource_manager: Manages external resources
        - transaction_manager: Coordinates transactions

        **Execution Engines:**
        - Created lazily via get_engine() / get_async_engine()
        - Sync engine for SimpleFSM
        - Async engine for AsyncSimpleFSM
        - Cached for reuse

    Attributes:
        name (str): Unique identifier for this FSM
        data_mode (ProcessingMode): Data processing mode (SINGLE/BATCH/STREAM)
        transaction_mode (TransactionMode): Transaction handling (NONE/OPTIMISTIC/PESSIMISTIC)
        description (str | None): Optional FSM description
        networks (Dict[str, StateNetwork]): State networks by name
        main_network_name (str | None): Name of the main network
        function_registry (FunctionRegistry): Registry of all functions
        resource_requirements (Dict[str, Any]): Aggregated resource requirements
        config (Dict[str, Any]): Additional configuration
        metadata (Dict[str, Any]): FSM metadata
        version (str): FSM version for compatibility
        created_at (float | None): Creation timestamp
        updated_at (float | None): Last update timestamp
        resource_manager (Any | None): External resource manager
        transaction_manager (Any | None): Transaction coordinator

    Design Patterns:
        **Separation of Concerns:**
        - FSM defines structure (what to execute)
        - Engines handle execution (how to execute)
        - Clear boundary between definition and runtime

        **Multiple Networks:**
        - FSMs can contain multiple state graphs
        - Enables modular workflow composition
        - Sub-networks can be reused across FSMs

        **Lazy Initialization:**
        - Engines created on first use
        - Reduces overhead for validation-only use cases
        - Cached for subsequent use

        **Validation Before Execution:**
        - validate() checks structure before execution
        - Catches configuration errors early
        - Provides detailed error messages

    Resource Management:
        The FSM aggregates resource requirements from all networks:
        - Database connections needed
        - API clients required
        - File handles expected
        - Memory requirements

        This enables:
        - Pre-execution resource checks
        - Resource pooling optimization
        - Clear dependency documentation

    Serialization:
        FSM supports serialization for persistence and transmission:
        - to_dict(): Convert to dictionary
        - from_dict(): Reconstruct from dictionary
        - Note: Function registry not serialized (functions must be re-registered)

    Thread Safety:
        The FSM structure itself is read-only after construction and validation.
        However, execution engines may or may not be thread-safe depending on
        data_mode:
        - SINGLE mode: Not thread-safe (by design)
        - BATCH mode: Thread-safe if using DataHandlingMode.COPY
        - STREAM mode: Thread-safe with proper streaming context

    Note:
        This is an internal API. Users should use the higher-level APIs:
        - SimpleFSM for synchronous workflows
        - AsyncSimpleFSM for async/production workflows
        - AdvancedFSM for debugging and profiling

        Direct FSM instantiation is primarily for builders and internal use.

    See Also:
        - :class:`~dataknobs_fsm.api.simple.SimpleFSM`: Synchronous API
        - :class:`~dataknobs_fsm.api.async_simple.AsyncSimpleFSM`: Async API
        - :class:`~dataknobs_fsm.core.network.StateNetwork`: State graph
        - :class:`~dataknobs_fsm.functions.base.FunctionRegistry`: Function registry
    """
    
    def __init__(
        self,
        name: str,
        data_mode: ProcessingMode = ProcessingMode.SINGLE,
        transaction_mode: TransactionMode = TransactionMode.NONE,
        description: str | None = None,
        resource_manager: Any | None = None,
        transaction_manager: Any | None = None
    ):
        """Initialize FSM.
        
        Args:
            name: Name of the FSM.
            data_mode: Data processing mode.
            transaction_mode: Transaction handling mode.
            description: Optional FSM description.
        """
        self.name = name
        self.data_mode = data_mode
        self.transaction_mode = transaction_mode
        self.description = description
        
        # Networks
        self.networks: Dict[str, StateNetwork] = {}
        self.main_network_name: str | None = None
        
        # Function registry
        self.function_registry = FunctionRegistry()
        
        # Resource requirements
        self.resource_requirements: Dict[str, Any] = {}
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.version: str = "1.0.0"
        self.created_at: float | None = None
        self.updated_at: float | None = None
        
        # Execution support (from builder FSM wrapper)
        self.resource_manager = resource_manager
        self.transaction_manager = transaction_manager
        self._engine: Any | None = None  # ExecutionEngine
        self._async_engine: Any | None = None  # AsyncExecutionEngine
    
    def add_network(
        self,
        network: StateNetwork,
        is_main: bool = False
    ) -> None:
        """Add a network to the FSM.
        
        Args:
            network: Network to add.
            is_main: Whether this is the main network.
        """
        self.networks[network.name] = network
        
        if is_main or self.main_network_name is None:
            self.main_network_name = network.name
        
        # Aggregate resource requirements
        for resource_type, requirements in network.resource_requirements.items():
            if resource_type not in self.resource_requirements:
                self.resource_requirements[resource_type] = set()
            self.resource_requirements[resource_type].update(requirements)
    
    def remove_network(self, network_name: str) -> bool:
        """Remove a network from the FSM.
        
        Args:
            network_name: Name of network to remove.
            
        Returns:
            True if removed successfully.
        """
        if network_name in self.networks:
            del self.networks[network_name]
            
            # Update main network if needed
            if self.main_network_name == network_name:
                if self.networks:
                    self.main_network_name = next(iter(self.networks.keys()))
                else:
                    self.main_network_name = None
            
            return True
        return False
    
    def get_network(self, network_name: str | None = None) -> StateNetwork | None:
        """Get a network by name.
        
        Args:
            network_name: Name of network (None for main network).
            
        Returns:
            Network or None if not found.
        """
        if network_name is None:
            network_name = self.main_network_name
        
        if network_name:
            return self.networks.get(network_name)
        return None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the FSM.
        
        Returns:
            Tuple of (valid, list of errors).
        """
        errors = []
        
        # Check for at least one network
        if not self.networks:
            errors.append("FSM has no networks")
        
        # Check main network exists
        if self.main_network_name and self.main_network_name not in self.networks:
            errors.append(f"Main network '{self.main_network_name}' not found")
        
        # Validate each network
        for network_name, network in self.networks.items():
            valid, network_errors = network.validate()
            if not valid:
                for error in network_errors:
                    errors.append(f"Network '{network_name}': {error}")
        
        # Check function references
        all_functions = self._get_all_function_references()
        for func_name in all_functions:
            if not self.function_registry.get_function(func_name):
                errors.append(f"Function '{func_name}' not registered")
        
        return len(errors) == 0, errors
    
    def _get_all_function_references(self) -> Set[str]:
        """Get all function references from all networks.
        
        Returns:
            Set of function names referenced.
        """
        functions = set()
        
        for network in self.networks.values():
            for arc in network.arcs.values():
                if hasattr(arc, 'pre_test') and arc.pre_test:
                    functions.add(arc.pre_test)
                if hasattr(arc, 'transform') and arc.transform:
                    functions.add(arc.transform)
        
        return functions
    
    def get_all_states(self) -> Dict[str, List[str]]:
        """Get all states from all networks.
        
        Returns:
            Dictionary of network_name -> list of state names.
        """
        all_states = {}
        
        for network_name, network in self.networks.items():
            all_states[network_name] = list(network.states.keys())
        
        return all_states
    
    def get_all_arcs(self) -> Dict[str, List[str]]:
        """Get all arcs from all networks.
        
        Returns:
            Dictionary of network_name -> list of arc IDs.
        """
        all_arcs = {}
        
        for network_name, network in self.networks.items():
            all_arcs[network_name] = list(network.arcs.keys())
        
        return all_arcs
    
    def supports_streaming(self) -> bool:
        """Check if FSM supports streaming.
        
        Returns:
            True if any network supports streaming.
        """
        return any(network.supports_streaming for network in self.networks.values())
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource requirements summary.
        
        Returns:
            Resource requirements summary.
        """
        summary = {
            'total_networks': len(self.networks),
            'total_states': sum(len(n.states) for n in self.networks.values()),
            'total_arcs': sum(len(n.arcs) for n in self.networks.values()),
            'resource_types': list(self.resource_requirements.keys()),
            'supports_streaming': self.supports_streaming(),
            'data_mode': self.data_mode.value,
            'transaction_mode': self.transaction_mode.value
        }
        
        # Add resource counts
        for resource_type, requirements in self.resource_requirements.items():
            summary[f'{resource_type}_count'] = len(requirements)
        
        return summary
    
    def clone(self) -> 'FSM':
        """Create a clone of this FSM.
        
        Returns:
            Cloned FSM.
        """
        clone = FSM(
            name=f"{self.name}_clone",
            data_mode=self.data_mode,
            transaction_mode=self.transaction_mode,
            description=self.description
        )
        
        # Clone networks
        for network_name, network in self.networks.items():
            # Note: This is a shallow copy - for deep clone would need to implement network.clone()
            clone.networks[network_name] = network
        
        clone.main_network_name = self.main_network_name
        clone.function_registry = self.function_registry
        clone.resource_requirements = self.resource_requirements.copy()
        clone.config = self.config.copy()
        clone.metadata = self.metadata.copy()
        clone.version = self.version
        
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FSM to dictionary representation.
        
        Returns:
            Dictionary representation.
        """
        return {
            'name': self.name,
            'description': self.description,
            'data_mode': self.data_mode.value,
            'transaction_mode': self.transaction_mode.value,
            'main_network': self.main_network_name,
            'networks': list(self.networks.keys()),
            'resource_requirements': {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.resource_requirements.items()
            },
            'config': self.config,
            'metadata': self.metadata,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FSM':
        """Create FSM from dictionary representation.
        
        Args:
            data: Dictionary with FSM data.
            
        Returns:
            New FSM instance.
        """
        fsm = cls(
            name=data['name'],
            data_mode=ProcessingMode(data.get('data_mode', 'single')),
            transaction_mode=TransactionMode(data.get('transaction_mode', 'none')),
            description=data.get('description')
        )
        
        fsm.main_network_name = data.get('main_network')
        fsm.config = data.get('config', {})
        fsm.metadata = data.get('metadata', {})
        fsm.version = data.get('version', '1.0.0')
        
        # Resource requirements
        for resource_type, requirements in data.get('resource_requirements', {}).items():
            fsm.resource_requirements[resource_type] = set(requirements)
        
        return fsm
    
    def find_state_definition(self, state_name: str, network_name: str | None = None) -> StateDefinition | None:
        """Find a state definition by name.
        
        Args:
            state_name: Name of the state to find
            network_name: Optional specific network to search in
            
        Returns:
            StateDefinition if found, None otherwise
        """
        if network_name:
            # Search specific network
            network = self.networks.get(network_name)
            if network and hasattr(network, 'states'):
                return network.states.get(state_name)
        else:
            # Search all networks
            for network in self.networks.values():
                if hasattr(network, 'states') and state_name in network.states:
                    return network.states[state_name]
        
        return None
    
    def create_state_instance(self, state_name: str, data: Dict[str, Any] | None = None, network_name: str | None = None) -> StateInstance:
        """Create a state instance from a state name.
        
        Args:
            state_name: Name of the state
            data: Optional initial data for the state
            network_name: Optional specific network to search in
            
        Returns:
            StateInstance object
        """
        # Try to find existing state definition
        state_def = self.find_state_definition(state_name, network_name)
        
        if not state_def:
            # Create minimal state definition
            state_def = StateDefinition(
                name=state_name,
                type=StateType.START if state_name in ['start', 'Start', 'START'] else StateType.NORMAL
            )
        
        # Create and return state instance
        return StateInstance(
            definition=state_def,
            data=data or {}
        )
    
    def get_state(self, state_name: str, network_name: str | None = None) -> StateDefinition | None:
        """Get a state definition by name.
        
        This is an alias for find_state_definition for compatibility.
        
        Args:
            state_name: Name of the state
            network_name: Optional specific network to search in
            
        Returns:
            StateDefinition if found, None otherwise
        """
        return self.find_state_definition(state_name, network_name)
    
    def is_start_state(self, state_name: str, network_name: str | None = None) -> bool:
        """Check if a state is a start state.
        
        Args:
            state_name: Name of the state
            network_name: Optional specific network to check in (defaults to main network)
            
        Returns:
            True if the state is a start state
        """
        network_name = network_name or self.main_network_name
        if network_name:
            network = self.networks.get(network_name)
            if network:
                return network.is_initial_state(state_name)
        return False
    
    def is_end_state(self, state_name: str, network_name: str | None = None) -> bool:
        """Check if a state is an end state.
        
        Args:
            state_name: Name of the state
            network_name: Optional specific network to check in (defaults to main network)
            
        Returns:
            True if the state is an end state
        """
        network_name = network_name or self.main_network_name
        if network_name:
            network = self.networks.get(network_name)
            if network:
                return network.is_final_state(state_name)
        return False
    
    def get_start_state(self, network_name: str | None = None) -> StateDefinition | None:
        """Get the start state definition.
        
        Args:
            network_name: Optional specific network to search in
            
        Returns:
            Start state definition if found, None otherwise
        """
        # If network specified, search that network
        if network_name:
            network = self.networks.get(network_name)
            if network and hasattr(network, 'states'):
                for state in network.states.values():
                    if (hasattr(state, 'is_start_state') and state.is_start_state()) or (hasattr(state, 'type') and state.type == StateType.START):
                        return state
        else:
            # Search main network first
            if self.main_network_name:
                start_state = self.get_start_state(self.main_network_name)
                if start_state:
                    return start_state
            
            # Search all networks
            for network in self.networks.values():
                if hasattr(network, 'states'):
                    for state in network.states.values():
                        if (hasattr(state, 'is_start_state') and state.is_start_state()) or (hasattr(state, 'type') and state.type == StateType.START):
                            return state
        
        # Fallback: look for state named 'start'
        return self.find_state_definition('start', network_name)
    
    @property
    def main_network(self) -> Optional['StateNetwork']:
        """Get the main network object.
        
        Returns:
            The main StateNetwork object or None if not set
        """
        if self.main_network_name:
            return self.networks.get(self.main_network_name)
        return None
    
    @property
    def states(self) -> Dict[str, StateDefinition]:
        """Get all states from the main network.
        
        Returns:
            Dictionary of state_name -> state_definition for the main network
        """
        if not self.main_network_name:
            return {}
        
        network = self.get_network(self.main_network_name)
        if network and hasattr(network, 'states'):
            return network.states
        return {}
    
    def get_all_states_dict(self) -> Dict[str, Dict[str, StateDefinition]]:
        """Get all states from all networks.
        
        Returns:
            Dictionary of network_name -> {state_name -> state_definition}
        """
        all_states = {}
        for network_name, network in self.networks.items():
            if hasattr(network, 'states'):
                all_states[network_name] = network.states
        return all_states
    
    def get_outgoing_arcs(self, state_name: str, network_name: str | None = None) -> List[Any]:
        """Get outgoing arcs from a state.
        
        Args:
            state_name: Name of the state
            network_name: Optional network name (uses main network if None)
            
        Returns:
            List of outgoing arcs from the state
        """
        network_name = network_name or self.main_network_name
        if not network_name:
            return []
        
        network = self.get_network(network_name)
        if network:
            return network.get_arcs_from_state(state_name)
        return []
    
    def get_engine(self, strategy: str | None = None) -> "ExecutionEngine":
        """Get or create the execution engine.

        Args:
            strategy: Optional execution strategy override

        Returns:
            ExecutionEngine instance.
        """
        if self._engine is None:
            from dataknobs_fsm.execution.engine import ExecutionEngine, TraversalStrategy
            
            # Map strategy strings to enum
            strategy_map = {
                "depth_first": TraversalStrategy.DEPTH_FIRST,
                "breadth_first": TraversalStrategy.BREADTH_FIRST,
                "resource_optimized": TraversalStrategy.RESOURCE_OPTIMIZED,
                "stream_optimized": TraversalStrategy.STREAM_OPTIMIZED,
            }
            
            strat = TraversalStrategy.DEPTH_FIRST  # Default
            if strategy and strategy in strategy_map:
                strat = strategy_map[strategy]
            
            self._engine = ExecutionEngine(
                fsm=self,
                strategy=strat,
            )
        
        return self._engine
    
    def get_async_engine(self, strategy: str | None = None) -> "AsyncExecutionEngine":
        """Get or create the async execution engine.

        Args:
            strategy: Optional execution strategy override

        Returns:
            AsyncExecutionEngine instance.
        """
        if self._async_engine is None:
            from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
            
            self._async_engine = AsyncExecutionEngine(fsm=self)
        
        return self._async_engine
    
    def _prepare_execution_context(self, initial_data: Dict[str, Any] | None = None):
        """Prepare execution context for FSM execution.
        
        Args:
            initial_data: Initial data for execution.
            
        Returns:
            Configured ExecutionContext instance.
        """
        from dataknobs_fsm.execution.context import ExecutionContext
        from dataknobs_fsm.streaming.core import StreamContext, StreamConfig
        
        # Create execution context
        context = ExecutionContext(
            data_mode=self.data_mode,
            transaction_mode=self.transaction_mode
        )
        
        # Set resource and transaction managers if available
        if self.resource_manager:
            context.resource_manager = self.resource_manager
        if self.transaction_manager:
            context.transaction_manager = self.transaction_manager
        
        # Set up context based on data mode
        if self.data_mode == ProcessingMode.BATCH:
            # For batch mode, treat input as batch data
            if initial_data is not None:
                # If it's not already a list, make it one
                if not isinstance(initial_data, list):  # type: ignore[unreachable]
                    context.batch_data = [initial_data]
                else:
                    context.batch_data = initial_data  # type: ignore[unreachable]
            else:
                context.batch_data = []
        elif self.data_mode == ProcessingMode.STREAM:
            # For stream mode, create a stream context
            stream_config = StreamConfig()
            context.stream_context = StreamContext(config=stream_config)
            
            # Add initial data as a chunk if provided
            if initial_data is not None:
                # Add the data as a single chunk to the stream
                context.stream_context.add_data(initial_data, is_last=True)
                # Also set context.data for compatibility
                context.data = initial_data
        else:
            # Single mode - data passed normally
            pass
        
        return context
    
    def _format_execution_result(self, success: bool, result: Any, context: Any, 
                                 duration: float, initial_data: Any = None,
                                 error: str | None = None) -> Dict[str, Any]:
        """Format the execution result in a standard format.
        
        Args:
            success: Whether execution succeeded.
            result: The execution result data.
            context: The execution context.
            duration: Time taken for execution.
            initial_data: Original input data.
            error: Error message if execution failed.
            
        Returns:
            Formatted result dictionary.
        """
        if error:
            return {
                "status": "error",
                "error": error,
                "data": initial_data,
                "execution_id": None,
                "transitions": 0,
                "duration": None
            }
        
        return {
            "status": "completed" if success else "failed",
            "data": result,
            "execution_id": getattr(context, 'execution_id', None),
            "transitions": getattr(context, 'transition_count', 0),
            "duration": duration
        }
    
    async def execute_async(self, initial_data: Dict[str, Any] | None = None) -> Any:
        """Execute the FSM asynchronously with initial data.
        
        Args:
            initial_data: Initial data for execution.
            
        Returns:
            Execution result.
        """
        import time
        
        try:
            # Get the async execution engine
            engine = self.get_async_engine()
            
            # Prepare execution context
            context = self._prepare_execution_context(initial_data)
            
            # Track execution time
            start_time = time.time()
            
            # Execute the FSM
            success, result = await engine.execute(
                context, 
                initial_data if self.data_mode == ProcessingMode.SINGLE else None
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            return self._format_execution_result(success, result, context, duration)
            
        except Exception as e:
            # Handle any exception that occurs during execution
            return self._format_execution_result(
                False, None, None, 0.0, initial_data, str(e)
            )
    
    def execute(self, initial_data: Dict[str, Any] | None = None) -> Any:
        """Execute the FSM synchronously with initial data.
        
        This is a simplified API for running the FSM.
        
        Args:
            initial_data: Initial data for execution.
            
        Returns:
            Execution result.
        """
        import time
        
        try:
            # Get the execution engine
            engine = self.get_engine()
            
            # Prepare execution context
            context = self._prepare_execution_context(initial_data)
            
            # Track execution time
            start_time = time.time()
            
            # Execute the FSM
            success, result = engine.execute(
                context, 
                initial_data if self.data_mode == ProcessingMode.SINGLE else None
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            return self._format_execution_result(success, result, context, duration)
            
        except Exception as e:
            # Handle any exception that occurs during execution
            return self._format_execution_result(
                False, None, None, 0.0, initial_data, str(e)
            )
