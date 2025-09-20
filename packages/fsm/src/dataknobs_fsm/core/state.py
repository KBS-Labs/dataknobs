"""Core state definitions and instances for FSM.

This module provides:
- StateDefinition: Blueprint for states with schema, functions, etc.
- StateInstance: Runtime instance of a state with data
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
    """Type of state in the FSM."""

    NORMAL = "normal"  # Regular processing state
    START = "start"  # Entry point state
    END = "end"  # Terminal state
    START_END = "start_end"  # Both entry and terminal state (for simple FSMs)
    ERROR = "error"  # Error handling state
    CHOICE = "choice"  # Decision/branching state
    WAIT = "wait"  # Waiting/pause state
    PARALLEL = "parallel"  # Parallel execution state


class StateStatus(Enum):
    """Status of a state instance."""
    
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
    """Definition of a state in the FSM."""
    
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
    """Runtime instance of a state."""
    
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
