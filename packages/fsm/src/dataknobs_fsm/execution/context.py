"""Execution context for FSM state machines."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any, Dict, List, Tuple, Union

from dataknobs_data.database import AsyncDatabase, SyncDatabase

from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.streaming.core import StreamChunk, StreamContext

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Status of a resource in the execution context."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ResourceAllocation:
    """Resource allocation information."""
    resource_type: str
    resource_id: str
    status: ResourceStatus
    allocated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionInfo:
    """Transaction tracking information."""
    transaction_id: str
    mode: TransactionMode
    started_at: float
    operations: List[Dict[str, Any]] = field(default_factory=list)
    is_committed: bool = False
    is_rolled_back: bool = False


@dataclass
class SubflowFrame:
    """Per-push bookkeeping for a sub-network entered via a ``PushArc``.

    ``network_stack`` carries only ``(network_name, return_state)`` and that
    two-tuple shape is consumed across both execution engines,
    ``NetworkExecutor``, and the per-call function contexts, so it cannot also
    carry the originating arc or the parent's pre-push state. This parallel
    frame does:

    - ``push_arc`` — the originating arc, so the pop can apply its
      ``result_mapping`` (which the pop site does not otherwise have).
    - ``parent_data`` — the parent context's data object *as it was before* the
      push replaced it with the sub-network's isolated view, so result mapping
      has a base to overlay onto and a failed push can restore it.
    - ``prev_parent_state_resources`` — the value of
      ``context.parent_state_resources`` before this push overwrote it, so the
      pop restores the correct inherited-resource view for the parent level
      (correct across nesting, not a single global slot).

    The frames are a stack (one per live push), so nested subflows each restore
    their own parent state on pop.
    """

    push_arc: Any
    parent_data: Any
    prev_parent_state_resources: Any


class ExecutionContext:
    """Execution context for FSM processing with full mode support.
    
    This context manages:
    - Data mode configuration (single, batch, stream)
    - Transaction management
    - Resource allocation and tracking
    - Stream coordination
    - State tracking and network stack
    - Parallel execution paths
    """
    
    def __init__(
        self,
        data_mode: ProcessingMode = ProcessingMode.SINGLE,
        transaction_mode: TransactionMode = TransactionMode.NONE,
        resources: Dict[str, Any] | None = None,
        database: Union[SyncDatabase, AsyncDatabase] | None = None,
        stream_context: StreamContext | None = None
    ):
        """Initialize execution context.
        
        Args:
            data_mode: Data processing mode.
            transaction_mode: Transaction handling mode.
            resources: Initial resource configurations.
            database: Database connection for transactions.
            stream_context: Stream context for stream mode.
        """
        # Mode configuration
        self.data_mode = data_mode
        self.transaction_mode = transaction_mode
        
        # State tracking
        self.current_state: str | None = None
        self.previous_state: str | None = None
        self.network_stack: List[Tuple[str, str | None]] = []
        # Parallel to ``network_stack``: one ``SubflowFrame`` per live push,
        # carrying the originating ``PushArc`` and the parent's pre-push
        # data/resource state for result mapping and rollback. ``network_stack``
        # stays a ``(name, return_state)`` tuple list (its shape is consumed
        # across the engines, ``NetworkExecutor``, and function contexts); this
        # is the per-push bookkeeping the tuple cannot carry. Like
        # ``network_stack`` it starts empty and is not copied into child/cloned
        # contexts (a fresh sub-path begins with no pushes of its own).
        self.subflow_frames: List[SubflowFrame] = []
        self.state_history: List[str] = []
        
        # Data management
        self.data: Any = None
        self.metadata: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        
        # Resource management
        self.resources: Dict[str, ResourceAllocation] = {}
        self.resource_limits: Dict[str, Any] = resources or {}
        self.resource_manager: Any = None  # ResourceManager instance
        self.current_state_resources: Dict[str, Any] = {}  # Resources allocated to current state
        self.parent_state_resources: Dict[str, Any] = {}  # Resources from parent state (in subnetworks)
        
        # Transaction management
        self.database = database
        self.current_transaction: TransactionInfo | None = None
        self.transaction_history: List[TransactionInfo] = []
        
        # Stream coordination
        self.stream_context = stream_context
        self.current_chunk: StreamChunk | None = None
        self.processed_chunks: int = 0
        
        # Batch processing
        self.batch_data: List[Any] = []
        self.batch_results: List[Any] = []
        self.batch_errors: List[Tuple[int, Exception]] = []
        
        # Parallel execution
        self.parallel_paths: Dict[str, ExecutionContext] = {}
        self.is_child_context: bool = False
        self.parent_context: ExecutionContext | None = None
        
        # Performance tracking
        self.start_time: float = time.time()
        self.state_timings: Dict[str, float] = {}
        self.function_call_count: Dict[str, int] = {}

        # State instance tracking for debugging
        self.current_state_instance: Any = None

        # Context factory — optional callable that transforms FunctionContext
        # into an application-specific context object for transform functions.
        # When set, _create_function_context() calls this with the built
        # FunctionContext and returns the factory's result instead.
        self.transform_context_factory: Callable[..., Any] | None = None

        # Idempotency flag — tracks whether initial-state transforms have
        # already been executed, preventing duplicate execution when the
        # AdvancedFSM sync/async entry points call _execute_state_transforms
        # directly (bypassing the engine's own enter_state logic).
        self._initial_transforms_executed: bool = False

        # State transforms that raised for this record. Populated by
        # BaseExecutionEngine.handle_transform_error; gates downstream transform
        # execution (record_has_failed / should_skip_state_transforms) and the
        # record-level success of finalize_single_result. This is per-record
        # data-integrity state, NOT shared across records: clone() and
        # create_child_context() deliberately do NOT copy it (each record /
        # parallel sub-path starts clean), while merge_child_context() unions a
        # child's failures back into the parent.
        self.failed_states: set[str] = set()
    
    def push_network(self, network_name: str, return_state: str | None = None) -> None:
        """Push a network onto the execution stack.
        
        Args:
            network_name: Name of network to push.
            return_state: State to return to after network completes.
        """
        self.network_stack.append((network_name, return_state))
    
    def pop_network(self) -> Tuple[str, str | None]:
        """Pop a network from the execution stack.

        Returns:
            Tuple of (network_name, return_state).
        """
        if self.network_stack:
            return self.network_stack.pop()
        return ("", None)

    def push_subflow_frame(
        self,
        push_arc: Any,
        parent_data: Any,
        prev_parent_state_resources: Any,
    ) -> None:
        """Record the per-push state for a sub-network entered via a push arc.

        Pushed alongside :meth:`push_network` when an engine commits a push, and
        consumed by :meth:`pop_subflow_frame` on completion (for result mapping)
        or on a failed entry (for rollback). See :class:`SubflowFrame`.

        Args:
            push_arc: The originating ``PushArc``.
            parent_data: The parent context's data object before the push
                replaced it with the sub-network's isolated view.
            prev_parent_state_resources: ``parent_state_resources`` before this
                push overwrote it, restored for the parent level on pop.
        """
        self.subflow_frames.append(
            SubflowFrame(push_arc, parent_data, prev_parent_state_resources)
        )

    def pop_subflow_frame(self) -> SubflowFrame | None:
        """Pop the most recent subflow frame, or ``None`` if there is none.

        Returns:
            The popped :class:`SubflowFrame`, or ``None`` when no push frame is
            recorded (kept tolerant so a pop can never raise on an unexpectedly
            empty stack).
        """
        if self.subflow_frames:
            return self.subflow_frames.pop()
        return None
    
    def set_state(self, state_name: str) -> None:
        """Set the current state.
        
        Args:
            state_name: Name of the new state.
        """
        if self.current_state:
            self.previous_state = self.current_state
            self.state_history.append(self.current_state)
        self.current_state = state_name
        self.state_timings[state_name] = time.time()
    
    def allocate_resource(
        self,
        resource_type: str,
        resource_id: str,
        metadata: Dict[str, Any] | None = None
    ) -> bool:
        """Allocate a resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Unique resource identifier.
            metadata: Optional resource metadata.
            
        Returns:
            True if allocation successful.
        """
        key = f"{resource_type}:{resource_id}"
        
        if key in self.resources:
            if self.resources[key].status != ResourceStatus.AVAILABLE:
                return False
        
        self.resources[key] = ResourceAllocation(
            resource_type=resource_type,
            resource_id=resource_id,
            status=ResourceStatus.ALLOCATED,
            allocated_at=time.time(),
            metadata=metadata or {}
        )
        return True
    
    def release_resource(self, resource_type: str, resource_id: str) -> bool:
        """Release an allocated resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource identifier.
            
        Returns:
            True if release successful.
        """
        key = f"{resource_type}:{resource_id}"
        
        if key in self.resources:
            self.resources[key].status = ResourceStatus.AVAILABLE
            return True
        return False
    
    def start_transaction(self, transaction_id: str | None = None) -> bool:
        """Start a new transaction.
        
        Args:
            transaction_id: Optional transaction ID.
            
        Returns:
            True if transaction started.
        """
        if self.transaction_mode == TransactionMode.NONE:
            return False
        
        if self.current_transaction and not self.current_transaction.is_committed:
            return False
        
        self.current_transaction = TransactionInfo(
            transaction_id=transaction_id or str(time.time()),
            mode=self.transaction_mode,
            started_at=time.time()
        )

        self._note_db_transaction_decoupled("start_transaction")
        return True

    def _note_db_transaction_decoupled(self, action: str) -> None:
        """DEBUG-log that DB atomicity is not driven from this sync context.

        ``start/commit/rollback_transaction`` maintain in-memory *logical*
        transaction bookkeeping only. Real database-level atomicity is driven
        through the async :meth:`AsyncDatabase.transaction` primitive (and the
        ``DatabaseTransaction`` FSM function) — a synchronous context cannot
        drive the async transaction context manager.

        This replaces the former ``hasattr``-guarded ``self.database.<action>()``
        calls, which silently no-op'd on backends without the method and, once
        the async transaction primitive landed, would have invoked an
        un-awaited coroutine on an ``AsyncDatabase`` (``begin_transaction`` is
        now async) — a silent miss, not a real commit. No dataknobs database
        exposes a bare sync ``commit``/``rollback``; atomicity lives on the
        transaction handle.
        """
        if self.database is None:
            return
        supports = getattr(self.database, "supports_transactions", None)
        supported = bool(supports()) if callable(supports) else False
        logger.debug(
            "ExecutionContext.%s: logical transaction tracked in-memory only; "
            "DB-level atomicity (%s on %s) is driven via the async "
            "db.transaction() primitive / DatabaseTransaction function, not this "
            "synchronous context.",
            action,
            "supported" if supported else "unsupported",
            type(self.database).__name__,
        )

    def commit_transaction(self) -> bool:
        """Commit the current transaction.

        Returns:
            True if commit successful.
        """
        if not self.current_transaction:
            return False

        self._note_db_transaction_decoupled("commit_transaction")
        self.current_transaction.is_committed = True
        self.transaction_history.append(self.current_transaction)
        self.current_transaction = None

        return True

    def rollback_transaction(self) -> bool:
        """Rollback the current transaction.

        Returns:
            True if rollback successful.
        """
        if not self.current_transaction:
            return False

        self._note_db_transaction_decoupled("rollback_transaction")
        self.current_transaction.is_rolled_back = True
        self.transaction_history.append(self.current_transaction)
        self.current_transaction = None

        return True
    
    def log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log an operation in the current transaction.
        
        Args:
            operation: Operation name.
            details: Operation details.
        """
        if self.current_transaction:
            self.current_transaction.operations.append({
                'operation': operation,
                'details': details,
                'timestamp': time.time()
            })
    
    def set_stream_chunk(self, chunk: StreamChunk) -> None:
        """Set the current stream chunk for processing.
        
        Args:
            chunk: Stream chunk to process.
        """
        self.current_chunk = chunk
        self.processed_chunks += 1
    
    def add_batch_item(self, item: Any) -> None:
        """Add an item to the batch.
        
        Args:
            item: Item to add to batch.
        """
        if self.data_mode == ProcessingMode.BATCH:
            self.batch_data.append(item)
    
    def add_batch_result(self, result: Any) -> None:
        """Add a result to batch results.
        
        Args:
            result: Processing result.
        """
        if self.data_mode == ProcessingMode.BATCH:
            self.batch_results.append(result)
    
    def add_batch_error(self, index: int, error: Exception) -> None:
        """Add an error to batch errors.
        
        Args:
            index: Batch item index.
            error: Error that occurred.
        """
        if self.data_mode == ProcessingMode.BATCH:
            self.batch_errors.append((index, error))
    
    def create_child_context(self, path_id: str) -> 'ExecutionContext':
        """Create a child context for parallel execution.
        
        Args:
            path_id: Unique identifier for the execution path.
            
        Returns:
            New child execution context.
        """
        child = ExecutionContext(
            data_mode=self.data_mode,
            transaction_mode=self.transaction_mode,
            resources=self.resource_limits.copy(),
            database=self.database,
            stream_context=self.stream_context
        )
        
        child.is_child_context = True
        child.parent_context = self
        child.variables = self.variables.copy()
        # failed_states deliberately NOT copied — a parallel sub-path starts
        # clean; its failures are unioned back via merge_child_context().

        self.parallel_paths[path_id] = child
        return child
    
    def merge_child_context(self, path_id: str) -> bool:
        """Merge a child context back into parent.
        
        Args:
            path_id: Path identifier to merge.
            
        Returns:
            True if merge successful.
        """
        if path_id not in self.parallel_paths:
            return False
        
        child = self.parallel_paths[path_id]
        
        # Merge results
        if self.data_mode == ProcessingMode.BATCH:
            self.batch_results.extend(child.batch_results)
            self.batch_errors.extend(child.batch_errors)

        # Propagate transform failures from the child sub-path. failed_states is
        # load-bearing for data integrity (it gates persistence and the parent's
        # finalize_single_result), so a failure on a parallel/batch path must
        # surface in the parent's result rather than being lost on merge.
        # (Sub-network/push-arc failures use a separate isolation boundary and
        # are merged back in NetworkExecutor._handle_push_arc, not here.)
        self.failed_states |= getattr(child, 'failed_states', set())

        # Merge metadata
        self.metadata.update(child.metadata)
        
        # Update function call counts
        for func, count in child.function_call_count.items():
            self.function_call_count[func] = self.function_call_count.get(func, 0) + count
        
        del self.parallel_paths[path_id]
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics.
        
        Returns:
            Resource usage information.
        """
        allocated = sum(1 for r in self.resources.values() 
                       if r.status == ResourceStatus.ALLOCATED)
        busy = sum(1 for r in self.resources.values() 
                  if r.status == ResourceStatus.BUSY)
        
        return {
            'total_resources': len(self.resources),
            'allocated': allocated,
            'busy': busy,
            'available': len(self.resources) - allocated - busy,
            'by_type': self._group_resources_by_type()
        }
    
    def _group_resources_by_type(self) -> Dict[str, int]:
        """Group resources by type.
        
        Returns:
            Resource counts by type.
        """
        by_type: Dict[str, int] = {}
        for resource in self.resources.values():
            by_type[resource.resource_type] = by_type.get(resource.resource_type, 0) + 1
        return by_type
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics.
        """
        elapsed_time = time.time() - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'states_visited': len(self.state_history),
            'current_state': self.current_state,
            'transactions': len(self.transaction_history),
            'chunks_processed': self.processed_chunks,
            'batch_items': len(self.batch_data),
            'batch_results': len(self.batch_results),
            'batch_errors': len(self.batch_errors),
            'function_calls': dict(self.function_call_count),
            'parallel_paths': len(self.parallel_paths)
        }
    
    def get_complete_path(self) -> List[str]:
        """Get the complete state traversal path including current state.
        
        Returns:
            List of state names in traversal order.
        """
        path = self.state_history.copy() if self.state_history else []
        
        # Add current state if not already in path and if it exists
        if self.current_state and (not path or path[-1] != self.current_state):
            path.append(self.current_state)
        
        return path
    
    def clone(self) -> 'ExecutionContext':
        """Create a clone of this context.
        
        Returns:
            Cloned execution context.
        """
        clone = ExecutionContext(
            data_mode=self.data_mode,
            transaction_mode=self.transaction_mode,
            resources=self.resource_limits.copy(),
            database=self.database,
            stream_context=self.stream_context
        )
        
        clone.current_state = self.current_state
        clone.previous_state = self.previous_state
        clone.state_history = self.state_history.copy()
        clone.data = self.data
        clone.metadata = self.metadata.copy()
        clone.variables = self.variables.copy()
        clone._initial_transforms_executed = self._initial_transforms_executed
        # Preserve the resource manager so cloned contexts (batch items,
        # COPY-mode per-record children) can still acquire state resources.
        clone.resource_manager = self.resource_manager
        # failed_states deliberately NOT copied — a clone (batch item /
        # per-record child) starts clean so one record's transform failure does
        # not taint the next record's persistence decision.

        return clone
    
    def is_complete(self) -> bool:
        """Check if the FSM execution has reached an end state.
        
        Returns:
            True if in an end state or no current state, False otherwise.
        """
        if not self.current_state:
            return True
        
        # Check if current state is marked as ended
        return self.metadata.get('is_end_state', False)
    
    def get_current_state(self) -> str | None:
        """Get the name of the current state.
        
        Returns:
            Current state name or None if not set.
        """
        return self.current_state
    
    def get_data_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current data.
        
        Returns:
            Copy of the current data dictionary.
        """
        if isinstance(self.data, dict):
            return self.data.copy()
        elif hasattr(self.data, '__dict__'):
            return vars(self.data).copy()
        else:
            return {'value': self.data}
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution metrics.
        """
        return {
            'states_visited': len(self.state_history),
            'current_state': self.current_state,
            'previous_state': self.previous_state,
            'transition_count': self.transition_count,
            'execution_id': self.execution_id,
            'data_mode': self.data_mode.value if self.data_mode else None,
            'transaction_mode': self.transaction_mode.value if self.transaction_mode else None
        }

    def get_current_state_instance(self) -> Any:
        """Get the current state instance object.

        Returns:
            The StateInstance object for the current state, or None if not set.
        """
        return self.current_state_instance
