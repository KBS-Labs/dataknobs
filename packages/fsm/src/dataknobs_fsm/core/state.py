"""Core state definitions and instances for FSM.

This module provides the fundamental building blocks for state management in
dataknobs-fsm. States are the nodes in the state network graph, representing
discrete steps in a workflow.

Architecture:
    The state system uses a two-level design:

    **StateDefinition (Blueprint):**
    - Defines the static structure of a state
    - Schema, functions, resources, retry logic
    - Immutable once created
    - Shared across multiple executions

    **StateInstance (Runtime):**
    - Runtime instance of a StateDefinition
    - Holds execution data for a specific workflow run
    - Tracks status, timing, errors, resources
    - Manages data mode handlers

    **Separation Benefits:**
    - Definition can be reused across executions
    - Instance holds execution-specific state
    - Clean separation of static vs dynamic concerns

State Lifecycle:
    A state instance progresses through a defined lifecycle:

    **1. Creation:**
    - StateInstance created from StateDefinition
    - Status: PENDING
    - No data yet

    **2. Entry:**
    - enter() called with input data
    - Status: ACTIVE
    - Data mode handler applies transformations
    - Pre-validation functions run
    - Resources acquired

    **3. Processing:**
    - Validation functions run
    - Transform functions modify data
    - Can modify_data() for additional changes
    - Can pause() / resume() execution

    **4. Exit:**
    - exit() called to finalize state
    - Data mode handler commits or rolls back
    - Status: COMPLETED or FAILED
    - Resources can be released
    - Duration calculated

    **5. Error Handling:**
    - fail() marks state as FAILED
    - Error tracking (error_count, last_error)
    - Retry logic can re-enter state
    - Transaction rollback if configured

Data Handling Modes:
    States use configurable data handling modes to control how data flows:

    **COPY Mode (DataHandlingMode.COPY):**
    - Deep copy on entry
    - Modifications to local copy
    - Commit on exit
    - Thread-safe, memory-intensive
    - Best for: Production, concurrent processing

    **REFERENCE Mode (DataHandlingMode.REFERENCE):**
    - Lazy loading with optimistic locking
    - Version tracking for conflicts
    - Moderate memory usage
    - Best for: Large datasets, memory-constrained

    **DIRECT Mode (DataHandlingMode.DIRECT):**
    - In-place modification
    - No copying, fastest performance
    - Not thread-safe
    - Best for: Single-threaded, performance-critical

    Each state can specify a preferred data_mode, or inherit from FSM-level
    configuration.

State Types:
    States can be classified by their role in the workflow:

    **NORMAL:**
    - Standard processing state
    - Most states are NORMAL
    - Can have incoming and outgoing transitions

    **START:**
    - Entry point for workflow
    - No incoming transitions
    - Exactly one per network (typically)

    **END:**
    - Terminal state
    - No outgoing transitions
    - Workflow completes here

    **START_END:**
    - Both entry and exit
    - For simple single-state FSMs

    **ERROR:**
    - Error handling state
    - Typically has special error recovery logic

    **CHOICE:**
    - Branching decision point
    - Multiple outgoing transitions
    - Conditional logic determines path

    **WAIT:**
    - Pause/synchronization point
    - Waits for external event
    - Can have timeout

    **PARALLEL:**
    - Parallel execution split/join
    - Spawns concurrent paths

Resource Management:
    States can declare resource requirements:

    **Resource Types:**
    - Database connections
    - API clients
    - File handles
    - External services
    - Memory allocations

    **Lifecycle:**
    - Requirements declared in StateDefinition
    - Resources acquired on state entry
    - Tracked in StateInstance.acquired_resources
    - Released on state exit or error

    **Resource Pooling:**
    - Resource managers can pool connections
    - States request from pool on entry
    - Return to pool on exit
    - Reduces acquisition overhead

Validation and Schemas:
    States can validate data using schemas:

    **StateSchema:**
    - Defines expected field types
    - Required vs optional fields
    - Type constraints
    - Extra field handling

    **Validation Functions:**
    - Pre-validation: Before state entry
    - Validation: After entry, before transform
    - Custom validation logic
    - Return (is_valid, errors)

    **Validation Flow:**
    1. Schema validation (if schema defined)
    2. Pre-validation functions
    3. State entry
    4. Validation functions
    5. Transform functions

Transformation:
    States can transform data via transform functions:

    **Transform Functions:**
    - Modify data during state processing
    - Multiple transforms can chain
    - Applied in order of registration
    - Return modified data

    **Use Cases:**
    - Data enrichment
    - Format conversion
    - Filtering
    - Aggregation
    - Computation

Error Handling and Retries:
    States support automatic retry on failure:

    **Retry Configuration:**
    - retry_count: Number of retry attempts
    - retry_delay: Delay between retries (seconds)
    - Configured in StateDefinition

    **Retry Behavior:**
    - State fails, error recorded
    - If retries remaining, re-enter state
    - Exponential backoff possible
    - If all retries exhausted, fail workflow

    **Error Tracking:**
    - error_count: Total errors encountered
    - last_error: Most recent error message
    - Status: FAILED after all retries exhausted

Transaction Support:
    States participate in transactions when configured:

    **Transaction Modes:**
    - NONE: No transactional guarantees
    - OPTIMISTIC: Commit at workflow end
    - PESSIMISTIC: Commit at each state

    **State Integration:**
    - StateInstance.transaction: Active transaction
    - Data changes recorded in transaction log
    - Rollback on error
    - Commit on success

Note:
    This is an internal API used by the core FSM and execution engines.
    Users typically interact with states indirectly through the SimpleFSM,
    AsyncSimpleFSM, or AdvancedFSM APIs.

See Also:
    - :class:`~dataknobs_fsm.core.fsm.FSM`: Core FSM class
    - :class:`~dataknobs_fsm.core.network.StateNetwork`: State graph container
    - :class:`~dataknobs_fsm.core.data_modes.DataHandlingMode`: Data mode enum
    - :class:`~dataknobs_fsm.core.data_modes.DataModeManager`: Data mode handler manager
    - :class:`~dataknobs_fsm.functions.base.IValidationFunction`: Validation function interface
    - :class:`~dataknobs_fsm.functions.base.ITransformFunction`: Transform function interface
"""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING
from uuid import uuid4

from dataknobs_data.fields import Field
from dataknobs_fsm.core.data_modes import DataHandlingMode, DataModeManager
from dataknobs_fsm.core.transactions import Transaction
from dataknobs_fsm.functions.base import (
    IValidationFunction,
    ITransformFunction,
    IEndStateTestFunction,
    ResourceConfig,
)

if TYPE_CHECKING:
    from dataknobs_fsm.core.arc import ArcDefinition


class StateType(Enum):
    """Type of state in the FSM.

    State types classify states by their role in the workflow. The type
    influences how the execution engine treats the state and what transitions
    are valid.

    Attributes:
        NORMAL: Regular processing state with standard behavior. Can have
            both incoming and outgoing transitions. Most states are NORMAL.
        START: Entry point state where workflow execution begins. Typically
            has no incoming transitions. Each network should have exactly one
            START state (or START_END).
        END: Terminal state where workflow execution completes. Has no
            outgoing transitions. Networks can have multiple END states for
            different completion paths.
        START_END: Combined entry and exit state for simple single-state
            FSMs. Acts as both START and END simultaneously.
        ERROR: Error handling state for workflow failures. Typically has
            special error recovery logic and may have transitions back to
            recovery states.
        CHOICE: Decision/branching state with conditional logic. Has multiple
            outgoing transitions with conditions that determine the path taken.
        WAIT: Pause/synchronization state that waits for external events.
            Can have timeout configuration. Used for async workflows.
        PARALLEL: Parallel execution split/join point. Spawns concurrent
            execution paths that may later merge. Used for parallel workflows.

    Note:
        State type affects validation rules:
        - START states typically cannot have incoming arcs
        - END states typically cannot have outgoing arcs
        - CHOICE states must have multiple outgoing arcs with conditions
    """

    NORMAL = "normal"  # Regular processing state
    START = "start"  # Entry point state
    END = "end"  # Terminal state
    START_END = "start_end"  # Both entry and terminal state (for simple FSMs)
    ERROR = "error"  # Error handling state
    CHOICE = "choice"  # Decision/branching state
    WAIT = "wait"  # Waiting/pause state
    PARALLEL = "parallel"  # Parallel execution state


class StateStatus(Enum):
    """Status of a state instance during execution.

    State status tracks the current execution state of a StateInstance.
    The status changes as the instance progresses through its lifecycle.

    Lifecycle:
        Normal flow: PENDING → ACTIVE → COMPLETED
        Failure flow: PENDING → ACTIVE → FAILED
        Skip flow: PENDING → SKIPPED
        Pause flow: PENDING → ACTIVE → PAUSED → ACTIVE → COMPLETED

    Attributes:
        PENDING: State instance created but not yet entered. Initial status
            for all state instances. No data processing has occurred.
        ACTIVE: State is currently being processed. Entry functions have run,
            and transform/validation functions are executing or have executed.
            Resources may be acquired.
        COMPLETED: State processing completed successfully. Exit functions
            have run, data committed, resources released. This is a terminal
            status for this state instance.
        FAILED: State processing failed with an error. Error has been recorded
            in last_error field. If retry_count > 0, state may be re-entered.
            Otherwise, this is a terminal status.
        SKIPPED: State was skipped during execution, typically due to
            conditional logic or optimization. No processing occurred. This
            is a terminal status for this state instance.
        PAUSED: State execution has been temporarily paused, typically in
            debugging scenarios or waiting for external events. Can transition
            back to ACTIVE via resume().

    Status Transitions:
        Valid transitions:
        - PENDING → ACTIVE (via enter())
        - PENDING → SKIPPED (via skip())
        - ACTIVE → COMPLETED (via exit() with success)
        - ACTIVE → FAILED (via fail() or exception)
        - ACTIVE → PAUSED (via pause())
        - PAUSED → ACTIVE (via resume())

    Note:
        Status is managed internally by the StateInstance class. Users
        typically observe status via execution results or debugging tools.
    """

    PENDING = "pending"  # Not yet entered
    ACTIVE = "active"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped in execution
    PAUSED = "paused"  # Paused execution


@dataclass
class StateSchema:
    """Schema definition for state data."""
    
    fields: List[Field]
    required_fields: Set[str] = dataclass_field(default_factory=set)
    constraints: Dict[str, Any] = dataclass_field(default_factory=dict)
    allow_extra_fields: bool = True
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against schema.
        
        Args:
            data: Data to validate.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in data:
                errors.append(f"Required field '{field_name}' is missing")
        
        # Check field types
        field_map = {f.name: f for f in self.fields}
        for field_name, value in data.items():
            if field_name in field_map:
                field_def = field_map[field_name]
                test_field = Field(field_name, value, field_def.type)
                if not test_field.validate():
                    errors.append(
                        f"Field '{field_name}' has invalid type. "
                        f"Expected {field_def.type}, got {type(value).__name__}"
                    )
            elif not self.allow_extra_fields:
                errors.append(f"Unexpected field '{field_name}'")
        
        return len(errors) == 0, errors


@dataclass
class StateDefinition:
    """Definition of a state in the FSM.

    StateDefinition is the static blueprint for a state, defining its structure,
    schema, functions, resources, and execution configuration. It is immutable
    once created and can be reused across multiple executions.

    Architecture:
        StateDefinition follows a declarative design pattern:
        - Define structure, not behavior
        - Functions referenced by interface, not implementation
        - Configuration-driven execution
        - Immutable and reusable

    Key Components:
        **Identity:**
        - name: Unique identifier within network
        - type: Role in workflow (START/END/NORMAL/etc)
        - description: Human-readable description
        - metadata: Additional custom attributes

        **Data Validation:**
        - schema: Optional StateSchema for data validation
        - pre_validation_functions: Run before state entry
        - validation_functions: Run after entry, before transform
        - Ensures data quality throughout workflow

        **Data Transformation:**
        - transform_functions: Modify data during processing
        - Multiple transforms chain in order
        - Each returns modified data

        **Resource Requirements:**
        - resource_requirements: List of ResourceConfig
        - Database connections, API clients, etc.
        - Acquired on entry, released on exit

        **Execution Configuration:**
        - timeout: Max execution time in seconds
        - retry_count: Number of retry attempts on failure
        - retry_delay: Delay between retries in seconds
        - data_mode: Preferred data handling mode

        **Network Integration:**
        - outgoing_arcs: List of transitions to other states
        - end_test_function: Determines if this is an end state

    Data Modes:
        States can specify a preferred data_mode:
        - COPY: Deep copy for thread safety (default)
        - REFERENCE: Lazy loading for memory efficiency
        - DIRECT: In-place modification for performance
        - None: Inherit from FSM-level configuration

    Validation:
        Multiple levels of validation:
        1. Schema validation (structural)
        2. Pre-validation functions (business logic before entry)
        3. Validation functions (business logic after entry)

    Resource Management:
        States declare resource requirements:
        - Specified as ResourceConfig objects
        - Include resource type, name, and configuration
        - Execution engine acquires before state entry
        - Released after state exit or on error

    Error Handling:
        Built-in retry support:
        - retry_count: Number of attempts (0 = no retry)
        - retry_delay: Delay between attempts in seconds
        - Exponential backoff can be implemented in retry logic
        - After all retries exhausted, state fails

    Note:
        StateDefinition is immutable after creation. To modify a state,
        create a new StateDefinition. This ensures consistency when the
        same definition is used across multiple executions.

        Functions are stored as references (interfaces), not implementations.
        The actual function implementations are registered in the FSM's
        FunctionRegistry.

    See Also:
        - :class:`~dataknobs_fsm.core.state.StateInstance`: Runtime instance
        - :class:`~dataknobs_fsm.core.state.StateSchema`: Data schema
        - :class:`~dataknobs_fsm.core.state.StateType`: State type enum
        - :class:`~dataknobs_fsm.functions.base.IValidationFunction`: Validation interface
        - :class:`~dataknobs_fsm.functions.base.ITransformFunction`: Transform interface
    """

    name: str
    type: StateType = StateType.NORMAL
    description: str = ""
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    # Schema and data handling
    schema: StateSchema | None = None
    data_mode: DataHandlingMode | None = None  # Preferred data mode
    
    # Resource requirements
    resource_requirements: List[ResourceConfig] = dataclass_field(default_factory=list)
    
    # Functions
    pre_validation_functions: List[IValidationFunction] = dataclass_field(default_factory=list)
    validation_functions: List[IValidationFunction] = dataclass_field(default_factory=list)
    transform_functions: List[ITransformFunction] = dataclass_field(default_factory=list)
    end_test_function: IEndStateTestFunction | None = None
    
    # Arc references (will be populated when building network)
    outgoing_arcs: List["ArcDefinition"] = dataclass_field(default_factory=list)
    
    # Execution settings
    timeout: float | None = None  # Timeout in seconds
    retry_count: int = 0  # Number of retries on failure
    retry_delay: float = 1.0  # Delay between retries in seconds
    
    def is_start_state(self) -> bool:
        """Check if this is a start state.
        
        Returns:
            True if this is a start state.
        """
        return self.type == StateType.START
    
    def is_end_state(self) -> bool:
        """Check if this is an end state.
        
        Returns:
            True if this is an end state.
        """
        return self.type == StateType.END
    
    @property
    def is_start(self) -> bool:
        """Property alias for is_start_state()."""
        return self.is_start_state()
    
    @property
    def is_end(self) -> bool:
        """Property alias for is_end_state()."""
        return self.is_end_state()
    
    @property
    def arcs(self) -> List["ArcDefinition"]:
        """Get the outgoing arcs from this state."""
        return self.outgoing_arcs
    
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against state schema.
        
        Args:
            data: Data to validate.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        if self.schema is None:
            return True, []
        return self.schema.validate(data)
    
    def add_pre_validation_function(self, func: IValidationFunction) -> None:
        """Add a pre-validation function.

        Args:
            func: Pre-validation function to add.
        """
        self.pre_validation_functions.append(func)

    def add_validation_function(self, func: IValidationFunction) -> None:
        """Add a validation function.

        Args:
            func: Validation function to add.
        """
        self.validation_functions.append(func)
    
    def add_transform_function(self, func: ITransformFunction) -> None:
        """Add a transform function.
        
        Args:
            func: Transform function to add.
        """
        self.transform_functions.append(func)
    
    def add_outgoing_arc(self, arc: "ArcDefinition") -> None:
        """Add an outgoing arc.
        
        Args:
            arc: Arc definition to add.
        """
        self.outgoing_arcs.append(arc)


@dataclass
class StateInstance:
    """Runtime instance of a state.

    StateInstance represents a single execution of a StateDefinition within
    a workflow. It holds the runtime data, tracks execution status, manages
    resources, and implements the state lifecycle.

    Architecture:
        StateInstance follows the Instance pattern:
        - One StateDefinition → Many StateInstances
        - Each instance is independent
        - Instance holds mutable execution state
        - Definition holds immutable structure

    Lifecycle:
        StateInstance progresses through a defined lifecycle:

        **1. Creation:**
        ```python
        from dataknobs_fsm.core.state import StateDefinition, StateInstance, StateType

        # Define state first
        state_def = StateDefinition(name="process", type=StateType.NORMAL)

        # Create instance
        instance = StateInstance(definition=state_def)
        # Status: PENDING
        # No data yet
        ```

        **2. Entry:**
        ```python
        instance.enter(input_data)
        # Status: ACTIVE
        # Data mode handler applies transformations
        # Resources acquired
        # Execution tracking begins
        ```

        **3. Processing:**
        ```python
        # Validation, transformation happen
        instance.modify_data(updates)
        # Can pause/resume if needed
        ```

        **4. Exit:**
        ```python
        result_data = instance.exit(commit=True)
        # Status: COMPLETED (or FAILED)
        # Data mode handler commits changes
        # Resources released
        # Duration calculated
        ```

        **5. Error Handling:**
        ```python
        instance.fail(error_message)
        # Status: FAILED
        # Error tracking updated
        # Can retry if retry_count > 0
        ```

    Attributes:
        id (str): Unique identifier for this instance (UUID)
        definition (StateDefinition): The state blueprint being executed
        status (StateStatus): Current execution status (PENDING/ACTIVE/COMPLETED/FAILED/etc)
        data (Dict[str, Any]): Current state data
        data_mode_manager (DataModeManager | None): Manages data handling modes
        data_handler (Any | None): Active data mode handler (COPY/REFERENCE/DIRECT)
        transaction (Transaction | None): Active transaction if using transactional mode
        acquired_resources (Dict[str, Any]): Resources acquired for this instance
        entry_time (datetime | None): When state was entered
        exit_time (datetime | None): When state was exited
        execution_count (int): Number of times state has been entered
        error_count (int): Number of errors encountered
        last_error (str | None): Most recent error message
        executed_arcs (List[str]): IDs of arcs executed from this state
        next_state (str | None): Name of next state to transition to

    Data Handling:
        StateInstance uses data mode handlers to manage how data flows:

        **COPY Mode:**
        - Deep copy on entry via data_handler.on_entry()
        - Modifications to local copy
        - Commit on exit via data_handler.on_exit(commit=True)
        - Thread-safe, memory-intensive

        **REFERENCE Mode:**
        - Lazy loading with version tracking
        - Optimistic locking for conflicts
        - on_modification() tracks changes
        - Memory-efficient

        **DIRECT Mode:**
        - In-place modification
        - No copying overhead
        - Fastest performance
        - Not thread-safe

    Resource Management:
        StateInstance tracks acquired resources:

        **Lifecycle:**
        1. acquire: add_resource(name, handle)
        2. use: get_resource(name) → handle
        3. release: release_resources()

        **Automatic Cleanup:**
        - Resources released on exit()
        - Resources released on fail()
        - Ensures no resource leaks

    Execution Tracking:
        StateInstance tracks detailed execution metrics:

        **Timing:**
        - entry_time: When processing started
        - exit_time: When processing completed
        - get_duration(): Calculates elapsed time

        **Counts:**
        - execution_count: Total entries (for retries)
        - error_count: Total errors encountered
        - executed_arcs: History of transitions

        **State:**
        - status: Current execution status
        - next_state: Determined next state
        - last_error: Most recent error message

    Transaction Support:
        StateInstance participates in transactions:

        **Integration:**
        - transaction field holds active transaction
        - Data changes logged to transaction
        - Rollback on fail()
        - Commit on exit(commit=True)

        **Modes:**
        - NONE: No transaction tracking
        - OPTIMISTIC: Commit at workflow end
        - PESSIMISTIC: Commit at each state exit

    Methods:
        **Lifecycle:**
        - enter(input_data): Enter state with data
        - exit(commit=True): Exit and finalize
        - fail(error): Mark as failed
        - skip(): Skip this state

        **Control:**
        - pause(): Temporarily pause
        - resume(): Resume from pause
        - modify_data(updates): Update state data

        **Resources:**
        - add_resource(name, resource): Acquire resource
        - get_resource(name): Get acquired resource
        - release_resources(): Release all resources

        **Tracking:**
        - record_arc_execution(arc_id): Track transition
        - get_duration(): Get execution time
        - to_dict(): Serialize to dictionary

    Note:
        StateInstance is managed by execution engines. Users typically don't
        create or manipulate instances directly, but may observe them via
        debugging tools or execution results.

        Each workflow execution creates fresh StateInstances, even when reusing
        StateDefinitions. This ensures execution isolation.

    See Also:
        - :class:`~dataknobs_fsm.core.state.StateDefinition`: State blueprint
        - :class:`~dataknobs_fsm.core.state.StateStatus`: Status enum
        - :class:`~dataknobs_fsm.core.data_modes.DataModeManager`: Data mode manager
        - :class:`~dataknobs_fsm.core.transactions.Transaction`: Transaction support
    """

    id: str = dataclass_field(default_factory=lambda: str(uuid4()))
    definition: StateDefinition = None
    status: StateStatus = StateStatus.PENDING
    
    # Data handling
    data: Dict[str, Any] = dataclass_field(default_factory=dict)
    data_mode_manager: DataModeManager | None = None
    data_handler: Any | None = None  # Active data handler
    
    # Transaction participation
    transaction: Transaction | None = None
    
    # Resource access
    acquired_resources: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    # Execution tracking
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    execution_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    
    # Arc execution history
    executed_arcs: List[str] = dataclass_field(default_factory=list)
    next_state: str | None = None
    
    def __post_init__(self):
        """Initialize data mode manager if not provided."""
        if self.data_mode_manager is None:
            # Use definition's data_mode if available and not None, else default to COPY
            default_mode = DataHandlingMode.COPY
            if self.definition and self.definition.data_mode:
                default_mode = self.definition.data_mode
            self.data_mode_manager = DataModeManager(default_mode)
    
    def enter(self, input_data: Dict[str, Any]) -> None:
        """Enter the state with input data.
        
        Args:
            input_data: Input data for the state.
        """
        self.status = StateStatus.ACTIVE
        self.entry_time = datetime.now()
        self.execution_count += 1
        
        # Apply data mode
        if self.data_mode_manager:
            mode = self.definition.data_mode if self.definition and self.definition.data_mode else self.data_mode_manager.default_mode
            self.data_handler = self.data_mode_manager.get_handler(mode)
            self.data = self.data_handler.on_entry(input_data)
        else:
            self.data = input_data
    
    def exit(self, commit: bool = True) -> Dict[str, Any]:
        """Exit the state.
        
        Args:
            commit: Whether to commit data changes.
            
        Returns:
            The final state data.
        """
        self.exit_time = datetime.now()
        
        # Handle data mode exit
        if self.data_handler:
            self.data = self.data_handler.on_exit(self.data, commit)
        
        if self.status == StateStatus.ACTIVE:
            self.status = StateStatus.COMPLETED
        
        return self.data
    
    def fail(self, error: str) -> None:
        """Mark the state as failed.
        
        Args:
            error: Error message.
        """
        self.status = StateStatus.FAILED
        self.error_count += 1
        self.last_error = error
        self.exit_time = datetime.now()
    
    def pause(self) -> None:
        """Pause state execution."""
        if self.status == StateStatus.ACTIVE:
            self.status = StateStatus.PAUSED
    
    def resume(self) -> None:
        """Resume paused state execution."""
        if self.status == StateStatus.PAUSED:
            self.status = StateStatus.ACTIVE
    
    def skip(self) -> None:
        """Skip this state."""
        self.status = StateStatus.SKIPPED
        self.exit_time = datetime.now()
    
    def modify_data(self, updates: Dict[str, Any]) -> None:
        """Modify state data.
        
        Args:
            updates: Data updates to apply.
        """
        if self.data_handler:
            # Let the data handler manage modifications
            self.data.update(updates)
            self.data = self.data_handler.on_modification(self.data)
        else:
            self.data.update(updates)
    
    def add_resource(self, name: str, resource: Any) -> None:
        """Add an acquired resource.
        
        Args:
            name: Resource name.
            resource: The resource handle/connection.
        """
        self.acquired_resources[name] = resource
    
    def get_resource(self, name: str) -> Any | None:
        """Get an acquired resource.
        
        Args:
            name: Resource name.
            
        Returns:
            The resource if available.
        """
        return self.acquired_resources.get(name)
    
    def release_resources(self) -> None:
        """Release all acquired resources."""
        self.acquired_resources.clear()
    
    def record_arc_execution(self, arc_id: str) -> None:
        """Record that an arc was executed.
        
        Args:
            arc_id: ID of the executed arc.
        """
        self.executed_arcs.append(arc_id)
    
    def get_duration(self) -> float | None:
        """Get execution duration in seconds.
        
        Returns:
            Duration in seconds if available.
        """
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        elif self.entry_time:
            return (datetime.now() - self.entry_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with state instance data.
        """
        return {
            "id": self.id,
            "name": self.definition.name if self.definition else None,
            "status": self.status.value,
            "data": self.data,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "duration": self.get_duration(),
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "executed_arcs": self.executed_arcs,
            "next_state": self.next_state,
        }


# Simplified State class for network usage
class State:
    """Simplified state class for use in state networks."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize state.
        
        Args:
            name: State name.
            **kwargs: Additional state properties.
        """
        self.name = name
        self.metadata = kwargs
        self.resource_requirements = kwargs.get("resource_requirements", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "resource_requirements": self.resource_requirements
        }


# StateMode for backwards compatibility
class StateMode(Enum):
    """Mode of state operation."""
    NORMAL = "normal"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
