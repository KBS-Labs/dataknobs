"""Wizard state transition observability and task tracking.

This module provides data structures for recording wizard state transitions
and tracking granular tasks within wizard flows, enabling observability,
debugging, and auditing.

The types here are wizard-specific extensions of the generic FSM observability
types from dataknobs_fsm. Key differences from FSM types:
- Uses "stage" terminology instead of "state" (wizard domain)
- Includes `user_input` field for capturing user responses
- TransitionStats includes wizard-specific metrics (backtrack_count, restart_count)
- WizardStateSnapshot provides complete wizard state capture
- WizardTask and WizardTaskList enable granular task tracking within stages

Conversion utilities are provided to convert between wizard and FSM types.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dataknobs_fsm.observability import (
    ExecutionHistoryQuery,
    ExecutionRecord,
    ExecutionStats,
    ExecutionTracker,
)


@dataclass
class TransitionRecord:
    """Record of a single wizard state transition.

    Captures all relevant information about a state transition including
    timing, stages, trigger, and data snapshots.

    Attributes:
        from_stage: Stage name before the transition
        to_stage: Stage name after the transition
        timestamp: Unix timestamp when transition occurred
        trigger: What triggered the transition (user_input, navigation, skip, restart)
        duration_in_stage_ms: Time spent in the from_stage in milliseconds
        data_snapshot: Optional snapshot of wizard data at transition time
        user_input: Optional user input that triggered the transition
        condition_evaluated: The condition expression that was evaluated (if any)
        condition_result: Result of the condition evaluation (True/False)
        error: Error message if transition failed

    Example:
        ```python
        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=time.time(),
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"intent": "create"},
            condition_evaluated="data.get('intent')",
            condition_result=True,
        )
        ```
    """

    from_stage: str
    to_stage: str
    timestamp: float
    trigger: str  # "user_input", "navigation_back", "navigation_skip", "restart", "auto"
    duration_in_stage_ms: float = 0.0
    data_snapshot: dict[str, Any] | None = None
    user_input: str | None = None
    condition_evaluated: str | None = None
    condition_result: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation of the record
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary containing record fields

        Returns:
            TransitionRecord instance
        """
        return cls(**data)


@dataclass
class TransitionHistoryQuery:
    """Query parameters for filtering transition history.

    Attributes:
        from_stage: Filter by source stage
        to_stage: Filter by target stage
        trigger: Filter by trigger type
        since: Filter to records after this timestamp
        until: Filter to records before this timestamp
        limit: Maximum number of records to return
    """

    from_stage: str | None = None
    to_stage: str | None = None
    trigger: str | None = None
    since: float | None = None
    until: float | None = None
    limit: int | None = None


@dataclass
class TransitionStats:
    """Aggregated statistics for wizard transitions.

    Attributes:
        total_transitions: Total number of transitions
        unique_paths: Number of unique stage-to-stage paths taken
        avg_duration_per_stage_ms: Average time spent in each stage
        most_common_trigger: Most frequent transition trigger
        backtrack_count: Number of backward navigation transitions
        restart_count: Number of restart transitions
        first_transition: Timestamp of first transition
        last_transition: Timestamp of last transition
    """

    total_transitions: int = 0
    unique_paths: int = 0
    avg_duration_per_stage_ms: float = 0.0
    most_common_trigger: str | None = None
    backtrack_count: int = 0
    restart_count: int = 0
    first_transition: float | None = None
    last_transition: float | None = None

    @property
    def has_backtracks(self) -> bool:
        """Check if there were any backtrack navigations.

        Returns:
            True if backtrack_count > 0
        """
        return self.backtrack_count > 0


# =============================================================================
# Task Tracking
# =============================================================================

# Type alias for task status
TaskStatus = Literal["pending", "in_progress", "completed", "skipped"]

# Type alias for task completion trigger
TaskCompletionTrigger = Literal["field_extraction", "tool_result", "stage_exit", "manual"]


@dataclass
class WizardTask:
    """A trackable task within the wizard flow.

    Tasks provide granular progress tracking within stages. A stage may have
    multiple tasks (e.g., collect_bot_name, collect_description), and global
    tasks can span stages (e.g., validate_config, save_config).

    Attributes:
        id: Unique task identifier (e.g., "collect_bot_name")
        description: Human-readable description
        status: Current status (pending, in_progress, completed, skipped)
        stage: Stage this task belongs to, or None for global tasks
        required: Whether this task is required for wizard completion
        completed_at: Timestamp when task was completed
        depends_on: List of task IDs that must complete first
        completed_by: What triggers task completion
        field_name: For field_extraction: which field completes this task
        tool_name: For tool_result: which tool completes this task

    Example:
        ```python
        task = WizardTask(
            id="collect_bot_name",
            description="Collect bot name",
            status="pending",
            stage="configure_identity",
            required=True,
            completed_by="field_extraction",
            field_name="bot_name",
        )
        ```
    """

    id: str
    description: str
    status: TaskStatus = "pending"
    stage: str | None = None  # None = global task
    required: bool = True
    completed_at: float | None = None
    depends_on: list[str] = field(default_factory=list)
    completed_by: TaskCompletionTrigger | None = None
    field_name: str | None = None  # For field_extraction
    tool_name: str | None = None  # For tool_result

    @property
    def is_complete(self) -> bool:
        """Check if task is completed."""
        return self.status == "completed"

    @property
    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == "pending"

    @property
    def is_global(self) -> bool:
        """Check if this is a global (stage-independent) task."""
        return self.stage is None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary.

        Returns:
            Dictionary representation of the task
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WizardTask":
        """Create task from dictionary.

        Args:
            data: Dictionary containing task fields

        Returns:
            WizardTask instance
        """
        return cls(**data)


@dataclass
class WizardTaskList:
    """Manages wizard tasks with dependency tracking.

    Provides methods for querying, completing, and calculating progress
    across a collection of tasks.

    Attributes:
        tasks: List of WizardTask instances

    Example:
        ```python
        task_list = WizardTaskList(tasks=[
            WizardTask(id="validate", description="Validate config"),
            WizardTask(id="save", description="Save config", depends_on=["validate"]),
        ])

        # Check available tasks
        available = task_list.get_available_tasks()  # Only "validate"

        # Complete a task
        task_list.complete_task("validate")

        # Now "save" is available
        available = task_list.get_available_tasks()  # ["validate", "save"]
        ```
    """

    tasks: list[WizardTask] = field(default_factory=list)

    def get_task(self, task_id: str) -> WizardTask | None:
        """Get a task by ID.

        Args:
            task_id: The task ID to find

        Returns:
            WizardTask if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.

        Only completes if all dependencies are met.

        Args:
            task_id: The task ID to complete

        Returns:
            True if task was completed, False otherwise
        """
        task = self.get_task(task_id)
        if task and self._dependencies_met(task):
            task.status = "completed"
            task.completed_at = time.time()
            return True
        return False

    def skip_task(self, task_id: str) -> bool:
        """Mark a task as skipped.

        Args:
            task_id: The task ID to skip

        Returns:
            True if task was skipped, False otherwise
        """
        task = self.get_task(task_id)
        if task:
            task.status = "skipped"
            return True
        return False

    def _dependencies_met(self, task: WizardTask) -> bool:
        """Check if all dependencies are completed.

        Args:
            task: The task to check dependencies for

        Returns:
            True if all dependencies are completed or skipped
        """
        for dep_id in task.depends_on:
            dep = self.get_task(dep_id)
            if not dep or dep.status not in ("completed", "skipped"):
                return False
        return True

    def get_pending_tasks(self) -> list[WizardTask]:
        """Get all pending tasks.

        Returns:
            List of tasks with status "pending"
        """
        return [t for t in self.tasks if t.status == "pending"]

    def get_completed_tasks(self) -> list[WizardTask]:
        """Get all completed tasks.

        Returns:
            List of tasks with status "completed"
        """
        return [t for t in self.tasks if t.status == "completed"]

    def get_available_tasks(self) -> list[WizardTask]:
        """Get tasks that are pending and have met dependencies.

        Returns:
            List of tasks ready to be worked on
        """
        return [
            t for t in self.tasks
            if t.status == "pending" and self._dependencies_met(t)
        ]

    def get_tasks_for_stage(self, stage: str) -> list[WizardTask]:
        """Get all tasks for a specific stage.

        Args:
            stage: Stage name to filter by

        Returns:
            List of tasks belonging to the stage
        """
        return [t for t in self.tasks if t.stage == stage]

    def get_global_tasks(self) -> list[WizardTask]:
        """Get all global (stage-independent) tasks.

        Returns:
            List of tasks with no stage association
        """
        return [t for t in self.tasks if t.stage is None]

    def calculate_progress(self) -> float:
        """Calculate progress based on required tasks.

        Returns:
            Progress as percentage (0.0 to 100.0)
        """
        required = [t for t in self.tasks if t.required]
        if not required:
            return 100.0
        completed = sum(1 for t in required if t.status in ("completed", "skipped"))
        return (completed / len(required)) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert task list to dictionary.

        Returns:
            Dictionary with serialized tasks
        """
        return {"tasks": [t.to_dict() for t in self.tasks]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WizardTaskList":
        """Create task list from dictionary.

        Args:
            data: Dictionary containing tasks list

        Returns:
            WizardTaskList instance
        """
        tasks = [WizardTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(tasks=tasks)

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)


# =============================================================================
# State Snapshots
# =============================================================================


@dataclass
class WizardStateSnapshot:
    """Complete snapshot of wizard state for auditing and UI display.

    Provides a complete picture of wizard state at a point in time,
    useful for debugging, audit trails, and driving UI components.

    Attributes:
        current_stage: Current stage name
        data: Collected wizard data
        history: List of visited stages
        transitions: List of all transitions
        completed: Whether wizard is complete
        snapshot_timestamp: When this snapshot was taken
        clarification_attempts: Current clarification attempt count
        tasks: List of all tasks (serialized)
        pending_tasks: Count of pending tasks
        completed_tasks: Count of completed tasks
        total_tasks: Total number of tasks
        available_task_ids: IDs of tasks ready to execute
        task_progress_percent: Progress based on tasks (if tasks defined)
        stage_index: Index of current stage
        total_stages: Total number of stages
        can_skip: Whether current stage can be skipped
        can_go_back: Whether back navigation is allowed
        suggestions: Quick-reply suggestions for current stage
    """

    current_stage: str
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    completed: bool = False
    snapshot_timestamp: float = field(default_factory=time.time)
    clarification_attempts: int = 0
    # Task tracking fields
    tasks: list[dict[str, Any]] = field(default_factory=list)
    pending_tasks: int = 0
    completed_tasks: int = 0
    total_tasks: int = 0
    available_task_ids: list[str] = field(default_factory=list)
    task_progress_percent: float = 0.0
    # Stage context fields
    stage_index: int = 0
    total_stages: int = 0
    can_skip: bool = False
    can_go_back: bool = True
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation with serialized transitions and tasks
        """
        return {
            "current_stage": self.current_stage,
            "data": self.data,
            "history": self.history,
            "transitions": [t.to_dict() for t in self.transitions],
            "completed": self.completed,
            "snapshot_timestamp": self.snapshot_timestamp,
            "clarification_attempts": self.clarification_attempts,
            "tasks": self.tasks,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "available_task_ids": self.available_task_ids,
            "task_progress_percent": self.task_progress_percent,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "can_skip": self.can_skip,
            "can_go_back": self.can_go_back,
            "suggestions": self.suggestions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WizardStateSnapshot":
        """Create snapshot from dictionary.

        Args:
            data: Dictionary containing snapshot fields

        Returns:
            WizardStateSnapshot instance
        """
        transitions = [
            TransitionRecord.from_dict(t) for t in data.get("transitions", [])
        ]
        return cls(
            current_stage=data["current_stage"],
            data=data.get("data", {}),
            history=data.get("history", []),
            transitions=transitions,
            completed=data.get("completed", False),
            snapshot_timestamp=data.get("snapshot_timestamp", time.time()),
            clarification_attempts=data.get("clarification_attempts", 0),
            tasks=data.get("tasks", []),
            pending_tasks=data.get("pending_tasks", 0),
            completed_tasks=data.get("completed_tasks", 0),
            total_tasks=data.get("total_tasks", 0),
            available_task_ids=data.get("available_task_ids", []),
            task_progress_percent=data.get("task_progress_percent", 0.0),
            stage_index=data.get("stage_index", 0),
            total_stages=data.get("total_stages", 0),
            can_skip=data.get("can_skip", False),
            can_go_back=data.get("can_go_back", True),
            suggestions=data.get("suggestions", []),
        )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get a specific task by ID.

        Args:
            task_id: The task ID to find

        Returns:
            Task dict if found, None otherwise
        """
        for task in self.tasks:
            if task.get("id") == task_id:
                return task
        return None

    def get_tasks_for_stage(self, stage: str) -> list[dict[str, Any]]:
        """Get all tasks for a specific stage.

        Args:
            stage: Stage name to filter by

        Returns:
            List of task dicts belonging to the stage
        """
        return [t for t in self.tasks if t.get("stage") == stage]

    def get_global_tasks(self) -> list[dict[str, Any]]:
        """Get all global (stage-independent) tasks.

        Returns:
            List of task dicts with no stage association
        """
        return [t for t in self.tasks if t.get("stage") is None]

    def is_task_available(self, task_id: str) -> bool:
        """Check if a task is available to execute.

        Args:
            task_id: The task ID to check

        Returns:
            True if task is in available_task_ids
        """
        return task_id in self.available_task_ids

    def get_latest_transition(self) -> dict[str, Any] | None:
        """Get the most recent transition record.

        Returns:
            Last transition as dict, or None if no transitions
        """
        if self.transitions:
            return self.transitions[-1].to_dict()
        return None


class TransitionTracker:
    """Tracks wizard transition history with query capabilities.

    Manages a bounded history of wizard state transitions and provides
    methods for querying and aggregating transition data.

    Attributes:
        max_history: Maximum number of records to retain

    Example:
        ```python
        tracker = TransitionTracker(max_history=100)

        # Record a transition
        tracker.record(TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=time.time(),
            trigger="user_input",
            duration_in_stage_ms=5000.0,
        ))

        # Query history
        recent = tracker.query(TransitionHistoryQuery(
            trigger="user_input",
            since=time.time() - 3600,
        ))

        # Get stats
        stats = tracker.get_stats()
        ```
    """

    def __init__(self, max_history: int = 100):
        """Initialize tracker.

        Args:
            max_history: Maximum records to retain (default 100)
        """
        self._history: list[TransitionRecord] = []
        self._max_history = max_history

    def record(self, transition: TransitionRecord) -> None:
        """Record a transition.

        Args:
            transition: The transition record to store
        """
        self._history.append(transition)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def query(
        self, query: TransitionHistoryQuery | None = None
    ) -> list[TransitionRecord]:
        """Query transition history.

        Args:
            query: Query parameters, or None for all records

        Returns:
            List of matching transition records
        """
        if query is None:
            return list(self._history)

        results = self._history

        if query.from_stage:
            results = [r for r in results if r.from_stage == query.from_stage]

        if query.to_stage:
            results = [r for r in results if r.to_stage == query.to_stage]

        if query.trigger:
            results = [r for r in results if r.trigger == query.trigger]

        if query.since:
            results = [r for r in results if r.timestamp >= query.since]

        if query.until:
            results = [r for r in results if r.timestamp <= query.until]

        if query.limit:
            results = results[-query.limit:]

        return results

    def get_stats(self) -> TransitionStats:
        """Get aggregated transition statistics.

        Returns:
            TransitionStats with aggregated metrics
        """
        if not self._history:
            return TransitionStats()

        # Count triggers
        trigger_counts: dict[str, int] = {}
        unique_paths: set[tuple[str, str]] = set()
        total_duration = 0.0
        backtrack_count = 0
        restart_count = 0

        for record in self._history:
            # Count trigger types
            trigger_counts[record.trigger] = trigger_counts.get(record.trigger, 0) + 1

            # Track unique paths
            unique_paths.add((record.from_stage, record.to_stage))

            # Sum durations
            total_duration += record.duration_in_stage_ms

            # Count special navigations
            if record.trigger == "navigation_back":
                backtrack_count += 1
            elif record.trigger == "restart":
                restart_count += 1

        # Find most common trigger
        most_common_trigger = max(trigger_counts, key=trigger_counts.get) if trigger_counts else None

        return TransitionStats(
            total_transitions=len(self._history),
            unique_paths=len(unique_paths),
            avg_duration_per_stage_ms=total_duration / len(self._history),
            most_common_trigger=most_common_trigger,
            backtrack_count=backtrack_count,
            restart_count=restart_count,
            first_transition=self._history[0].timestamp,
            last_transition=self._history[-1].timestamp,
        )

    def clear(self) -> None:
        """Clear all transition history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return number of records in history."""
        return len(self._history)


def create_transition_record(
    from_stage: str,
    to_stage: str,
    trigger: str,
    duration_in_stage_ms: float = 0.0,
    data_snapshot: dict[str, Any] | None = None,
    user_input: str | None = None,
    condition_evaluated: str | None = None,
    condition_result: bool | None = None,
    error: str | None = None,
) -> TransitionRecord:
    """Factory function to create a transition record.

    Convenience function that automatically sets the timestamp.

    Args:
        from_stage: Stage name before transition
        to_stage: Stage name after transition
        trigger: What triggered the transition
        duration_in_stage_ms: Time spent in from_stage
        data_snapshot: Optional data snapshot
        user_input: Optional user input
        condition_evaluated: The condition expression that was evaluated
        condition_result: Result of the condition evaluation
        error: Error message if transition failed

    Returns:
        TransitionRecord with current timestamp
    """
    return TransitionRecord(
        from_stage=from_stage,
        to_stage=to_stage,
        timestamp=time.time(),
        trigger=trigger,
        duration_in_stage_ms=duration_in_stage_ms,
        data_snapshot=data_snapshot,
        user_input=user_input,
        condition_evaluated=condition_evaluated,
        condition_result=condition_result,
        error=error,
    )


# =============================================================================
# Conversion utilities between wizard and FSM observability types
# =============================================================================


def transition_record_to_execution_record(
    record: TransitionRecord,
) -> ExecutionRecord:
    """Convert a wizard TransitionRecord to a generic FSM ExecutionRecord.

    Maps wizard-specific fields to FSM equivalents:
    - from_stage -> from_state
    - to_stage -> to_state
    - data_snapshot -> data_after (data_before is None)
    - error presence determines success

    Args:
        record: Wizard TransitionRecord to convert

    Returns:
        Equivalent FSM ExecutionRecord
    """
    return ExecutionRecord(
        from_state=record.from_stage,
        to_state=record.to_stage,
        timestamp=record.timestamp,
        trigger=record.trigger,
        transition_name=None,  # Wizard doesn't track transition names
        duration_in_state_ms=record.duration_in_stage_ms,
        data_before=None,
        data_after=record.data_snapshot,
        condition_evaluated=record.condition_evaluated,
        condition_result=record.condition_result,
        success=record.error is None,
        error=record.error,
    )


def execution_record_to_transition_record(
    record: ExecutionRecord,
    user_input: str | None = None,
) -> TransitionRecord:
    """Convert a generic FSM ExecutionRecord to a wizard TransitionRecord.

    Maps FSM fields to wizard equivalents:
    - from_state -> from_stage
    - to_state -> to_stage
    - data_after -> data_snapshot (data_before is ignored)

    Args:
        record: FSM ExecutionRecord to convert
        user_input: Optional user input to attach (not present in FSM record)

    Returns:
        Equivalent wizard TransitionRecord
    """
    return TransitionRecord(
        from_stage=record.from_state,
        to_stage=record.to_state,
        timestamp=record.timestamp,
        trigger=record.trigger,
        duration_in_stage_ms=record.duration_in_state_ms,
        data_snapshot=record.data_after,
        user_input=user_input,
        condition_evaluated=record.condition_evaluated,
        condition_result=record.condition_result,
        error=record.error,
    )


def transition_stats_to_execution_stats(
    stats: TransitionStats,
) -> ExecutionStats:
    """Convert wizard TransitionStats to generic FSM ExecutionStats.

    Note: Wizard-specific metrics (backtrack_count, restart_count) are not
    carried over as FSM ExecutionStats doesn't have these fields.

    Args:
        stats: Wizard TransitionStats to convert

    Returns:
        Equivalent FSM ExecutionStats (without wizard-specific metrics)
    """
    return ExecutionStats(
        total_transitions=stats.total_transitions,
        successful_transitions=stats.total_transitions,  # Wizard doesn't track failures
        failed_transitions=0,
        unique_paths=stats.unique_paths,
        avg_duration_per_state_ms=stats.avg_duration_per_stage_ms,
        most_visited_state=None,  # Wizard doesn't track this
        most_common_trigger=stats.most_common_trigger,
        first_transition=stats.first_transition,
        last_transition=stats.last_transition,
    )


# Re-export FSM types for convenience
__all__ = [
    # Task tracking types
    "WizardTask",
    "WizardTaskList",
    "TaskStatus",
    "TaskCompletionTrigger",
    # Wizard-specific types
    "TransitionRecord",
    "TransitionHistoryQuery",
    "TransitionStats",
    "TransitionTracker",
    "WizardStateSnapshot",
    "create_transition_record",
    # Conversion utilities
    "transition_record_to_execution_record",
    "execution_record_to_transition_record",
    "transition_stats_to_execution_stats",
    # Re-exported FSM types
    "ExecutionRecord",
    "ExecutionHistoryQuery",
    "ExecutionStats",
    "ExecutionTracker",
]
