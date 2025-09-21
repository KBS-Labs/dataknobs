"""Arc implementation for FSM state transitions."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, TYPE_CHECKING

from dataknobs_fsm.core.exceptions import FunctionError, ResourceError
from dataknobs_fsm.functions.base import FunctionContext

if TYPE_CHECKING:
    from dataknobs_fsm.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


class DataIsolationMode(Enum):
    """Data isolation modes for push arcs."""
    COPY = "copy"  # Deep copy data when pushing
    REFERENCE = "reference"  # Pass data by reference
    SERIALIZE = "serialize"  # Serialize/deserialize for isolation


@dataclass
class ArcDefinition:
    """Definition of an arc between states.

    This class defines the static properties of an arc,
    including the transition logic and resource requirements.
    """

    target_state: str
    pre_test: str | None = None
    transform: str | None = None
    priority: int = 0  # Higher priority arcs are evaluated first
    definition_order: int = 0  # Track definition order for stable sorting
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Resource requirements for this arc
    required_resources: Dict[str, str] = field(default_factory=dict)
    # e.g., {'database': 'main_db', 'llm': 'gpt4'}
    
    def __hash__(self) -> int:
        """Make ArcDefinition hashable."""
        return hash((
            self.target_state,
            self.pre_test,
            self.transform,
            self.priority
        ))


@dataclass
class PushArc(ArcDefinition):
    """Arc that pushes to a sub-network.
    
    Push arcs allow hierarchical state machine composition
    by pushing execution to a sub-network and returning
    when the sub-network completes.
    """
    
    target_network: str = ""  # Name of the target network
    return_state: str | None = None  # State to return to after sub-network
    isolation_mode: DataIsolationMode = DataIsolationMode.COPY
    pass_context: bool = True  # Whether to pass execution context
    
    # Mapping of data from parent to child network
    data_mapping: Dict[str, str] = field(default_factory=dict)
    # e.g., {'parent_field': 'child_field'}
    
    # Mapping of results from child to parent network
    result_mapping: Dict[str, str] = field(default_factory=dict)
    # e.g., {'child_result': 'parent_field'}


class ArcExecution:
    """Handles the execution of arc transitions.
    
    This class manages the runtime execution of arcs,
    including resource allocation, streaming support,
    and transaction participation.
    """
    
    def __init__(
        self,
        arc_def: ArcDefinition,
        source_state: str,
        function_registry
    ):
        """Initialize arc execution.

        Args:
            arc_def: Arc definition.
            source_state: Source state name.
            function_registry: Registry of available functions (FunctionRegistry or dict).
        """
        self.arc_def = arc_def
        self.source_state = source_state
        self.function_registry = function_registry

        # Execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0

    def _log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning message to log.
        """
        logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Error message to log.
        """
        logger.error(message)
    
    def can_execute(
        self,
        context: "ExecutionContext",
        data: Any = None
    ) -> bool:
        """Check if arc can be executed.
        
        This runs the pre-test function if defined.
        
        Args:
            context: Execution context.
            data: Current data.
            
        Returns:
            True if arc can be executed.
        """
        if not self.arc_def.pre_test:
            return True
        
        # Handle both FunctionRegistry and dict for pre-test function lookup
        if hasattr(self.function_registry, 'get_function'):
            # FunctionRegistry object
            pre_test_func = self.function_registry.get_function(self.arc_def.pre_test)
        elif isinstance(self.function_registry, dict):
            # Plain dictionary
            pre_test_func = self.function_registry.get(self.arc_def.pre_test)
        else:
            pre_test_func = None
        
        if pre_test_func is None:
            raise FunctionError(
                f"Pre-test function '{self.arc_def.pre_test}' not found",
                from_state=self.source_state,
                to_state=self.arc_def.target_state
            )
        
        try:

            # Create function context with resources
            func_context = self._create_function_context(context)

            # Execute pre-test
            result = pre_test_func(data, func_context)

            # Handle tuple return from InterfaceWrapper (returns (result, error))
            if isinstance(result, tuple) and len(result) == 2:
                return bool(result[0])
            return bool(result)
            
        except Exception as e:
            raise FunctionError(
                f"Pre-test execution failed: {e}",
                from_state=self.source_state,
                to_state=self.arc_def.target_state
            ) from e
    
    def execute(
        self,
        context: "ExecutionContext",
        data: Any = None,
        stream_enabled: bool = False
    ) -> Any:
        """Execute the arc transition.
        
        This runs the transform function if defined and
        manages resource allocation.
        
        Args:
            context: Execution context.
            data: Current data.
            stream_enabled: Whether streaming is enabled.
            
        Returns:
            Transformed data.
        """
        import time
        start_time = time.time()
        
        try:
            # Get state resources from context if available
            state_resources = getattr(context, 'current_state_resources', None)

            # Allocate required resources (merging with state resources)
            resources = self._allocate_resources(context, state_resources)
            
            # Execute transform if defined
            if self.arc_def.transform:
                # Handle both FunctionRegistry and dict
                if hasattr(self.function_registry, 'get_function'):
                    # FunctionRegistry object
                    transform_func = self.function_registry.get_function(self.arc_def.transform)
                elif isinstance(self.function_registry, dict):
                    # Plain dictionary
                    transform_func = self.function_registry.get(self.arc_def.transform)
                else:
                    transform_func = None
                
                if transform_func is None:
                    raise FunctionError(
                        f"Transform function '{self.arc_def.transform}' not found",
                        from_state=self.source_state,
                        to_state=self.arc_def.target_state
                    )
                
                # Create function context with resources
                func_context = self._create_function_context(
                    context,
                    resources,
                    stream_enabled
                )
                
                # Handle streaming vs non-streaming execution
                if stream_enabled and hasattr(transform_func, 'stream_capable'):
                    result = self._execute_streaming(
                        transform_func,
                        data,
                        func_context
                    )
                else:
                    # Call the transform function properly
                    # Check if it has a transform method (wrapped function)
                    if hasattr(transform_func, 'transform'):
                        result = transform_func.transform(data, func_context)
                    elif callable(transform_func):
                        result = transform_func(data, func_context)
                    else:
                        raise ValueError(f"Transform {self.arc_def.transform} is not callable")

                    # Handle ExecutionResult objects
                    from dataknobs_fsm.functions.base import ExecutionResult
                    if isinstance(result, ExecutionResult):
                        if result.success:
                            result = result.data
                        else:
                            raise FunctionError(
                                result.error or "Transform failed",
                                from_state=self.source_state,
                                to_state=self.arc_def.target_state
                            )
            else:
                # No transform, pass data through
                result = data

            # Update statistics
            self.execution_count += 1
            self.success_count += 1

            return result
            
        except Exception as e:
            self.execution_count += 1
            self.failure_count += 1
            
            raise FunctionError(
                f"Arc execution failed: {e}",
                from_state=self.source_state,
                to_state=self.arc_def.target_state
            ) from e
        finally:
            elapsed = time.time() - start_time
            self.total_execution_time += elapsed
            
            # Release resources
            if 'resources' in locals():
                self._release_resources(context, resources)
    
    def execute_with_transaction(
        self,
        context: "ExecutionContext",
        data: Any = None,
        transaction_id: str | None = None
    ) -> Any:
        """Execute arc within a transaction context.
        
        Args:
            context: Execution context.
            data: Current data.
            transaction_id: Transaction identifier.
            
        Returns:
            Transformed data.
        """
        # Get or create transaction
        if transaction_id is None:
            import uuid
            transaction_id = str(uuid.uuid4())
        
        try:
            # Begin transaction on required resources
            self._begin_transaction(context, transaction_id)
            
            # Execute arc
            result = self.execute(context, data)
            
            # Commit transaction
            self._commit_transaction(context, transaction_id)
            
            return result
            
        except Exception:
            # Rollback transaction
            self._rollback_transaction(context, transaction_id)
            raise
    
    def execute_push(
        self,
        push_arc: PushArc,
        context: "ExecutionContext",
        data: Any = None
    ) -> Any:
        """Execute a push arc to a sub-network.
        
        Args:
            push_arc: Push arc definition.
            context: Execution context.
            data: Current data.
            
        Returns:
            Result from sub-network execution.
        """
        # Prepare data for sub-network based on isolation mode
        if push_arc.isolation_mode == DataIsolationMode.COPY:
            import copy
            sub_data = copy.deepcopy(data)
        elif push_arc.isolation_mode == DataIsolationMode.SERIALIZE:
            import json
            serialized = json.dumps(data)
            sub_data = json.loads(serialized)
        else:
            sub_data = data
        
        # Apply data mapping
        if push_arc.data_mapping:
            mapped_data = {}
            for parent_field, child_field in push_arc.data_mapping.items():
                if hasattr(data, parent_field):
                    mapped_data[child_field] = getattr(data, parent_field)
                elif isinstance(data, dict) and parent_field in data:
                    mapped_data[child_field] = data[parent_field]
            sub_data = mapped_data
        
        # Push context to sub-network
        context.push_network(push_arc.target_network, push_arc.return_state)
        
        # Execute sub-network (this would be handled by execution engine)
        # For now, we just return the data
        result = sub_data
        
        # Apply result mapping
        if push_arc.result_mapping:
            for child_field, parent_field in push_arc.result_mapping.items():
                if isinstance(result, dict) and child_field in result:
                    if isinstance(data, dict):
                        data[parent_field] = result[child_field]
                    elif hasattr(data, parent_field):
                        setattr(data, parent_field, result[child_field])
        
        return result
    
    def _create_function_context(
        self,
        exec_context: "ExecutionContext",
        resources: Dict[str, Any] | None = None,
        stream_enabled: bool = False
    ) -> FunctionContext:
        """Create function context for execution.
        
        Args:
            exec_context: Execution context.
            resources: Allocated resources.
            stream_enabled: Whether streaming is enabled.
            
        Returns:
            Function context.
        """
        return FunctionContext(
            state_name=self.source_state,
            function_name=self.arc_def.transform or self.arc_def.pre_test,
            metadata={
                'source_state': self.source_state,
                'target_state': self.arc_def.target_state,
                'arc_priority': self.arc_def.priority,
                'stream_enabled': stream_enabled
            },
            resources=resources or {}
        )
    
    def _allocate_resources(
        self,
        context: "ExecutionContext",
        state_resources: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Allocate required resources for arc execution, merging with state resources.

        Args:
            context: Execution context.
            state_resources: Already allocated state resources to merge with.

        Returns:
            Dictionary of merged resources (state + arc-specific).
        """
        # Start with state resources if provided
        resources = dict(state_resources) if state_resources else {}

        # Get resource manager from context
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            # No resource manager available - return existing resources
            return resources

        # Generate unique owner ID for this arc execution
        # Create an arc identifier from source and target states
        arc_identifier = f"{self.source_state}_to_{self.arc_def.target_state}"
        owner_id = f"arc_{arc_identifier}_{getattr(context, 'execution_id', 'unknown')}"

        for resource_type, resource_name in self.arc_def.required_resources.items():
            # Skip if already have this resource from state
            if resource_type in resources:
                self._log_warning(
                    f"Arc resource '{resource_type}' already allocated by state, skipping"
                )
                continue

            try:
                # Acquire arc-specific resource
                resource = resource_manager.acquire(
                    name=resource_name,
                    owner_id=owner_id,
                    timeout=30.0  # 30 second timeout
                )
                resources[resource_type] = resource

                # Track for cleanup (only arc-specific resources)
                if not hasattr(context, '_arc_acquired_resources'):
                    context._arc_acquired_resources = {}
                context._arc_acquired_resources[resource_name] = owner_id

            except Exception as e:
                # Resource acquisition failed - clean up only arc-specific resources
                self._release_arc_resources(context, getattr(context, '_arc_acquired_resources', {}))
                raise ResourceError(
                    resource_id=resource_name,
                    message=f"Failed to acquire arc resource: {e}",
                    details={"operation": "acquire", "error": str(e)}
                ) from e

        return resources

    def _release_arc_resources(
        self,
        context: "ExecutionContext",
        arc_resources: Dict[str, str]
    ) -> None:
        """Release only arc-specific resources, not state resources.

        Args:
            context: Execution context.
            arc_resources: Map of resource_name -> owner_id for arc resources only.
        """
        if not arc_resources:
            return

        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return

        for resource_name, owner_id in arc_resources.items():
            try:
                resource_manager.release(resource_name, owner_id)
            except Exception as e:
                self._log_error(f"Failed to release arc resource {resource_name}: {e}")

        # Clear arc resources tracking
        if hasattr(context, '_arc_acquired_resources'):
            context._arc_acquired_resources = {}
    
    def _release_resources(
        self,
        context: "ExecutionContext",
        resources: Dict[str, Any]
    ) -> None:
        """Release allocated resources.
        
        Args:
            context: Execution context.
            resources: Resources to release.
        """
        # Get resource manager from context
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return
        
        # Get acquired resources from context if available
        acquired_resources = getattr(context, '_acquired_resources', {})
        
        # Release each resource
        for resource_type in resources.keys():
            # Find the resource name for this resource type
            resource_name = None
            for rtype, rname in self.arc_def.required_resources.items():
                if rtype == resource_type:
                    resource_name = rname
                    break
            
            if resource_name and resource_name in acquired_resources:
                owner_id = acquired_resources[resource_name]
                try:
                    resource_manager.release(resource_name, owner_id)
                    # Remove from tracking
                    del acquired_resources[resource_name]
                except Exception:
                    # Best effort cleanup - don't propagate release errors
                    pass
    
    def _execute_streaming(
        self,
        func: Callable,
        data: Any,
        context: FunctionContext
    ) -> Any:
        """Execute function with streaming support.
        
        Args:
            func: Function to execute.
            data: Input data.
            context: Function context.
            
        Returns:
            Streamed result.
        """
        # This would integrate with the streaming system
        # For now, we just execute normally
        return func(data, context)
    
    def _begin_transaction(
        self,
        context: "ExecutionContext",
        transaction_id: str
    ) -> None:
        """Begin transaction on required resources.
        
        Args:
            context: Execution context.
            transaction_id: Transaction ID.
        """
        # This would interface with transactional resources
        pass
    
    def _commit_transaction(
        self,
        context: "ExecutionContext",
        transaction_id: str
    ) -> None:
        """Commit transaction on resources.
        
        Args:
            context: Execution context.
            transaction_id: Transaction ID.
        """
        # This would interface with transactional resources
        pass
    
    def _rollback_transaction(
        self,
        context: "ExecutionContext",
        transaction_id: str
    ) -> None:
        """Rollback transaction on resources.
        
        Args:
            context: Execution context.
            transaction_id: Transaction ID.
        """
        # This would interface with transactional resources
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary of statistics.
        """
        avg_time = 0.0
        if self.execution_count > 0:
            avg_time = self.total_execution_time / self.execution_count
        
        return {
            'source_state': self.source_state,
            'target_state': self.arc_def.target_state,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_time,
            'success_rate': (
                self.success_count / self.execution_count
                if self.execution_count > 0 else 0.0
            )
        }
