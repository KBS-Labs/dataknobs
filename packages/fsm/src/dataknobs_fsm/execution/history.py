"""Execution history tracking for FSM state machines."""

import time
from enum import Enum
from typing import Any, Dict, List

from dataknobs_structures import Tree

from dataknobs_fsm.core.data_modes import DataHandlingMode


class ExecutionStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep:
    """Represents a single step in the execution history."""
    
    def __init__(
        self,
        step_id: str,
        state_name: str,
        network_name: str,
        timestamp: float,
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        status: ExecutionStatus = ExecutionStatus.PENDING
    ):
        """Initialize execution step.
        
        Args:
            step_id: Unique identifier for this step.
            state_name: Name of the state.
            network_name: Name of the network.
            timestamp: Unix timestamp when step was created.
            data_mode: Data mode used for this step.
            status: Current status of the step.
        """
        self.step_id = step_id
        self.state_name = state_name
        self.network_name = network_name
        self.timestamp = timestamp
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.data_mode = data_mode
        self.status = status
        
        # Execution details
        self.arc_taken: str | None = None
        self.data_snapshot: Any | None = None
        self.error: Exception | None = None
        self.metrics: Dict[str, Any] = {}
        self.resource_usage: Dict[str, Any] = {}
        
        # Stream tracking
        self.stream_progress: Dict[str, Any] | None = None
        self.chunks_processed: int = 0
        self.records_processed: int = 0
    
    def start(self) -> None:
        """Mark step as started."""
        self.start_time = time.time()
        self.status = ExecutionStatus.IN_PROGRESS
    
    def complete(self, arc_taken: str | None = None) -> None:
        """Mark step as completed.
        
        Args:
            arc_taken: The arc that was taken from this state.
        """
        self.end_time = time.time()
        self.status = ExecutionStatus.COMPLETED
        self.arc_taken = arc_taken
    
    def fail(self, error: Exception) -> None:
        """Mark step as failed.
        
        Args:
            error: The exception that caused the failure.
        """
        self.end_time = time.time()
        self.status = ExecutionStatus.FAILED
        self.error = error
    
    def skip(self, reason: str) -> None:
        """Mark step as skipped.
        
        Args:
            reason: Reason for skipping.
        """
        self.end_time = time.time()
        self.status = ExecutionStatus.SKIPPED
        self.metrics['skip_reason'] = reason
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add a metric to this step.
        
        Args:
            key: Metric key.
            value: Metric value.
        """
        self.metrics[key] = value
    
    def add_resource_usage(self, resource_type: str, usage: Dict[str, Any]) -> None:
        """Track resource usage for this step.
        
        Args:
            resource_type: Type of resource (e.g., "database", "llm").
            usage: Usage metrics.
        """
        self.resource_usage[resource_type] = usage
    
    def update_stream_progress(
        self,
        chunks: int,
        records: int,
        current_position: int | None = None
    ) -> None:
        """Update streaming progress.
        
        Args:
            chunks: Number of chunks processed.
            records: Number of records processed.
            current_position: Current position in stream.
        """
        self.chunks_processed = chunks
        self.records_processed = records
        if current_position is not None:
            if self.stream_progress is None:
                self.stream_progress = {}
            self.stream_progress['position'] = current_position
    
    @property
    def duration(self) -> float | None:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'step_id': self.step_id,
            'state_name': self.state_name,
            'network_name': self.network_name,
            'timestamp': self.timestamp,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'data_mode': self.data_mode.value,
            'status': self.status.value,
            'arc_taken': self.arc_taken,
            'error': str(self.error) if self.error else None,
            'metrics': self.metrics,
            'resource_usage': self.resource_usage,
            'stream_progress': self.stream_progress,
            'chunks_processed': self.chunks_processed,
            'records_processed': self.records_processed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionStep':
        """Create ExecutionStep from dictionary representation.
        
        Args:
            data: Dictionary with step data.
            
        Returns:
            ExecutionStep instance.
        """
        step = cls(
            step_id=data['step_id'],
            state_name=data['state_name'],
            network_name=data['network_name'],
            timestamp=data['timestamp'],
            data_mode=DataHandlingMode(data['data_mode']),
            status=ExecutionStatus(data['status'])
        )
        
        # Restore timing info
        step.start_time = data.get('start_time')
        step.end_time = data.get('end_time')
        
        # Restore execution details
        step.arc_taken = data.get('arc_taken')
        
        # Restore error (as string - can't fully reconstruct Exception)
        error_str = data.get('error')
        if error_str:
            step.error = Exception(error_str)
        
        # Restore metrics and usage
        step.metrics = data.get('metrics', {})
        step.resource_usage = data.get('resource_usage', {})
        
        # Restore stream progress
        step.stream_progress = data.get('stream_progress')
        step.chunks_processed = data.get('chunks_processed', 0)
        step.records_processed = data.get('records_processed', 0)
        
        return step


class ExecutionHistory:
    """Tracks execution history using a tree structure.
    
    This class manages the execution history of an FSM, tracking:
    - State transitions
    - Data modifications based on mode
    - Resource usage
    - Stream progress
    - Execution metrics
    """
    
    def __init__(
        self,
        fsm_name: str,
        execution_id: str,
        data_mode: DataHandlingMode = DataHandlingMode.COPY,
        max_depth: int | None = None,
        enable_data_snapshots: bool = False
    ):
        """Initialize execution history.
        
        Args:
            fsm_name: Name of the FSM.
            execution_id: Unique execution identifier.
            data_mode: Default data mode for execution.
            max_depth: Maximum tree depth (for pruning).
            enable_data_snapshots: Whether to store data snapshots.
        """
        self.fsm_name = fsm_name
        self.execution_id = execution_id
        self.data_mode = data_mode
        self.max_depth = max_depth
        self.enable_data_snapshots = enable_data_snapshots
        
        # Create tree structure - Tree stores root nodes
        self.tree_roots: List[Tree] = []
        self.current_node: Tree | None = None
        
        # Tracking
        self.start_time = time.time()
        self.end_time: float | None = None
        self.total_steps = 0
        self.failed_steps = 0
        self.skipped_steps = 0
        
        # Mode-specific storage
        self._mode_storage: Dict[DataHandlingMode, List[ExecutionStep]] = {
            DataHandlingMode.COPY: [],
            DataHandlingMode.REFERENCE: [],
            DataHandlingMode.DIRECT: []
        }
        
        # Resource tracking
        self.resource_summary: Dict[str, Dict[str, Any]] = {}
        
        # Stream tracking
        self.stream_summary: Dict[str, Any] = {
            'total_chunks': 0,
            'total_records': 0,
            'streams_processed': 0
        }
    
    def add_step(
        self,
        state_name: str,
        network_name: str,
        data: Any | None = None,
        parent_step_id: str | None = None
    ) -> ExecutionStep:
        """Add a new execution step.
        
        Args:
            state_name: Name of the state.
            network_name: Name of the network.
            data: Optional data snapshot.
            parent_step_id: Parent step ID for branching.
            
        Returns:
            The created ExecutionStep.
        """
        import uuid
        
        step_id = str(uuid.uuid4())
        step = ExecutionStep(
            step_id=step_id,
            state_name=state_name,
            network_name=network_name,
            timestamp=time.time(),
            data_mode=self.data_mode
        )
        
        # Store data snapshot if enabled
        if self.enable_data_snapshots and data is not None:
            step.data_snapshot = self._snapshot_data(data)
        
        # Add to tree
        if parent_step_id:
            parent_node = self._find_node_by_step_id(parent_step_id)
            if parent_node:
                self.current_node = Tree(step, parent=parent_node)
        elif self.current_node:
            self.current_node = Tree(step, parent=self.current_node)
        else:
            # Create root node
            self.current_node = Tree(step)
            self.tree_roots.append(self.current_node)
        
        # Track in mode-specific storage
        self._mode_storage[self.data_mode].append(step)
        
        self.total_steps += 1
        
        # Prune if needed
        if self.max_depth and self._get_max_depth() > self.max_depth:
            self._prune_old_branches()
        
        return step
    
    def update_step(
        self,
        step_id: str,
        status: ExecutionStatus | None = None,
        arc_taken: str | None = None,
        error: Exception | None = None,
        metrics: Dict[str, Any] | None = None
    ) -> bool:
        """Update an existing step.
        
        Args:
            step_id: Step ID to update.
            status: New status.
            arc_taken: Arc taken from this step.
            error: Error if failed.
            metrics: Additional metrics.
            
        Returns:
            True if step was found and updated.
        """
        node = self._find_node_by_step_id(step_id)
        if not node:
            return False
        
        step = node.data
        
        if status == ExecutionStatus.COMPLETED:
            step.complete(arc_taken)
        elif status == ExecutionStatus.FAILED:
            if error:
                step.fail(error)
                self.failed_steps += 1
        elif status == ExecutionStatus.SKIPPED:
            step.skip(metrics.get('reason', 'Unknown') if metrics else 'Unknown')
            self.skipped_steps += 1
        elif status:
            step.status = status
        
        if metrics:
            for key, value in metrics.items():
                step.add_metric(key, value)
        
        return True
    
    def get_path_to_current(self) -> List[ExecutionStep]:
        """Get the path from root to current step.
        
        Returns:
            List of steps from root to current.
        """
        if not self.current_node:
            return []
        
        path = []
        node = self.current_node
        while node:
            path.insert(0, node.data)
            node = node.parent
        
        return path
    
    def get_all_paths(self) -> List[List[ExecutionStep]]:
        """Get all execution paths in the tree.
        
        Returns:
            List of all paths from root to leaves.
        """
        paths = []
        
        def collect_paths(node: Tree, current_path: List[ExecutionStep]):
            current_path.append(node.data)
            
            if not node.children:
                # Leaf node - save path
                paths.append(current_path.copy())
            else:
                # Continue down each branch
                for child in node.children:
                    collect_paths(child, current_path.copy())
        
        for root in self.tree_roots:
            collect_paths(root, [])
        
        return paths
    
    @property
    def steps(self) -> List[ExecutionStep]:
        """Get all execution steps from the history tree.
        
        Returns:
            List of all execution steps in order.
        """
        all_steps = []
        
        def collect_steps(node: Tree):
            all_steps.append(node.data)
            if node.children:
                for child in node.children:
                    collect_steps(child)
        
        for root in self.tree_roots:
            collect_steps(root)
        
        return all_steps
    
    def get_steps_by_state(self, state_name: str) -> List[ExecutionStep]:
        """Get all steps for a specific state.
        
        Args:
            state_name: Name of the state.
            
        Returns:
            List of steps for that state.
        """
        steps = []
        
        def collect_steps(node: Tree):
            if node.data.state_name == state_name:
                steps.append(node.data)
            if node.children:
                for child in node.children:
                    collect_steps(child)
        
        for root in self.tree_roots:
            collect_steps(root)
        
        return steps
    
    def get_steps_by_mode(self, mode: DataHandlingMode) -> List[ExecutionStep]:
        """Get all steps executed in a specific data mode.
        
        Args:
            mode: Data mode to filter by.
            
        Returns:
            List of steps in that mode.
        """
        return self._mode_storage.get(mode, []).copy()
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get aggregated resource usage.
        
        Returns:
            Resource usage summary.
        """
        usage = {}
        
        def aggregate_usage(node: Tree):
            step = node.data
            for resource_type, metrics in step.resource_usage.items():
                if resource_type not in usage:
                    usage[resource_type] = {
                        'total_calls': 0,
                        'total_duration': 0,
                        'steps': []
                    }
                
                usage[resource_type]['total_calls'] += 1  # type: ignore
                if 'duration' in metrics:
                    usage[resource_type]['total_duration'] += metrics['duration']
                usage[resource_type]['steps'].append(step.step_id)
            
            if node.children:
                for child in node.children:
                    aggregate_usage(child)
        
        for root in self.tree_roots:
            aggregate_usage(root)
        
        return usage
    
    def get_stream_progress(self) -> Dict[str, Any]:
        """Get streaming progress summary.
        
        Returns:
            Stream progress information.
        """
        total_chunks = 0
        total_records = 0
        
        def aggregate_stream(node: Tree):
            nonlocal total_chunks, total_records
            step = node.data
            total_chunks += step.chunks_processed
            total_records += step.records_processed
            
            if node.children:
                for child in node.children:
                    aggregate_stream(child)
        
        for root in self.tree_roots:
            aggregate_stream(root)
        
        return {
            'total_chunks': total_chunks,
            'total_records': total_records,
            'streams_processed': self.stream_summary['streams_processed']
        }
    
    def finalize(self) -> None:
        """Mark execution as complete."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.
        
        Returns:
            Summary of the execution.
        """
        return {
            'fsm_name': self.fsm_name,
            'execution_id': self.execution_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time if self.end_time else None,
            'total_steps': self.total_steps,
            'failed_steps': self.failed_steps,
            'skipped_steps': self.skipped_steps,
            'completed_steps': self.total_steps - self.failed_steps - self.skipped_steps,
            'data_mode': self.data_mode.value,
            'tree_depth': self._get_max_depth(),
            'total_paths': len(self.get_all_paths()),
            'resource_usage': self.get_resource_usage(),
            'stream_progress': self.get_stream_progress()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of history.
        """
        paths = []
        for path in self.get_all_paths():
            paths.append([step.to_dict() for step in path])
        
        return {
            'summary': self.get_summary(),
            'paths': paths,
            'mode_storage': {
                mode.value: [s.to_dict() for s in steps]
                for mode, steps in self._mode_storage.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionHistory':
        """Create ExecutionHistory from dictionary representation.
        
        Args:
            data: Dictionary with history data.
            
        Returns:
            ExecutionHistory instance.
        """
        summary = data['summary']
        
        # Create history instance
        history = cls(
            fsm_name=summary['fsm_name'],
            execution_id=summary['execution_id'],
            data_mode=DataHandlingMode(summary['data_mode']),
            max_depth=None,  # Will be inferred
            enable_data_snapshots=False  # Will be inferred from data
        )
        
        # Restore properties from summary
        history.start_time = summary['start_time']
        history.end_time = summary.get('end_time')
        history.total_steps = summary['total_steps']
        history.failed_steps = summary['failed_steps']
        history.skipped_steps = summary['skipped_steps']
        
        # Rebuild tree structure from paths
        for path_data in data.get('paths', []):
            parent_node = None
            for step_dict in path_data:
                step = ExecutionStep.from_dict(step_dict)
                
                # Create tree node
                if parent_node is None:
                    # This is a root step
                    node = Tree(step)
                    history.tree_roots.append(node)
                else:
                    # Child step
                    node = Tree(step, parent=parent_node)
                
                parent_node = node
                
                # Track in mode-specific storage
                history._mode_storage[step.data_mode].append(step)
            
            # Set current node to the last node in this path
            if parent_node:
                history.current_node = parent_node
        
        return history
    
    def _find_node_by_step_id(self, step_id: str) -> Tree | None:
        """Find a node by step ID.
        
        Args:
            step_id: Step ID to find.
            
        Returns:
            Tree node if found, None otherwise.
        """
        def search_node(node: Tree) -> Tree | None:
            if node.data.step_id == step_id:
                return node
            if node.children:
                for child in node.children:
                    result = search_node(child)
                    if result:
                        return result
            return None
        
        for root in self.tree_roots:
            result = search_node(root)
            if result:
                return result
        
        return None
    
    def _snapshot_data(self, data: Any) -> Any:
        """Create a snapshot of data based on mode.
        
        Args:
            data: Data to snapshot.
            
        Returns:
            Snapshot of the data.
        """
        import copy
        
        if self.data_mode == DataHandlingMode.COPY:
            # Deep copy for COPY mode
            return copy.deepcopy(data)
        elif self.data_mode == DataHandlingMode.REFERENCE:
            # Store reference info only
            return {
                'type': type(data).__name__,
                'id': id(data),
                'size': len(data) if hasattr(data, '__len__') else None
            }
        else:  # DIRECT mode
            # Store minimal info
            return {'type': type(data).__name__}
    
    def _get_max_depth(self) -> int:
        """Get the maximum depth of the tree.
        
        Returns:
            Maximum depth across all trees.
        """
        max_depth = 0
        
        def get_depth(node: Tree, depth: int = 0):
            nonlocal max_depth
            depth += 1
            max_depth = max(max_depth, depth)
            if node.children:
                for child in node.children:
                    get_depth(child, depth)
        
        for root in self.tree_roots:
            get_depth(root)
        
        return max_depth
    
    def _prune_old_branches(self) -> None:
        """Prune old branches based on mode-specific strategy."""
        if self.data_mode == DataHandlingMode.COPY:
            # Keep all branches for COPY mode (full history)
            pass
        elif self.data_mode == DataHandlingMode.REFERENCE:
            # Prune branches not on critical path
            # Keep only paths that lead to current node
            pass
        else:  # DIRECT mode
            # Aggressive pruning - keep only recent history
            # Remove all but the current path
            current_path = self.get_path_to_current()
            if current_path and len(current_path) > self.max_depth:  # type: ignore
                # Keep only last max_depth steps
                steps_to_keep = current_path[-self.max_depth:]  # type: ignore
                # Rebuild tree with only these steps
                self.tree_roots = []
                self.current_node = None
                for step in steps_to_keep:
                    if not self.current_node:
                        self.current_node = Tree(step)
                        self.tree_roots.append(self.current_node)
                    else:
                        self.current_node = Tree(step, parent=self.current_node)
