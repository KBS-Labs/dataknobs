"""Wizard state transition observability.

This module provides data structures for recording wizard state transitions,
enabling observability, debugging, and auditing of wizard flows.

The types here are wizard-specific extensions of the generic FSM observability
types from dataknobs_fsm. Key differences from FSM types:
- Uses "stage" terminology instead of "state" (wizard domain)
- Includes `user_input` field for capturing user responses
- TransitionStats includes wizard-specific metrics (backtrack_count, restart_count)
- WizardStateSnapshot provides complete wizard state capture

Conversion utilities are provided to convert between wizard and FSM types.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any

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


@dataclass
class WizardStateSnapshot:
    """Complete snapshot of wizard state for auditing.

    Provides a complete picture of wizard state at a point in time,
    useful for debugging and audit trails.

    Attributes:
        current_stage: Current stage name
        data: Collected wizard data
        history: List of visited stages
        transitions: List of all transitions
        completed: Whether wizard is complete
        snapshot_timestamp: When this snapshot was taken
        clarification_attempts: Current clarification attempt count
    """

    current_stage: str
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    completed: bool = False
    snapshot_timestamp: float = field(default_factory=time.time)
    clarification_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation with serialized transitions
        """
        return {
            "current_stage": self.current_stage,
            "data": self.data,
            "history": self.history,
            "transitions": [t.to_dict() for t in self.transitions],
            "completed": self.completed,
            "snapshot_timestamp": self.snapshot_timestamp,
            "clarification_attempts": self.clarification_attempts,
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
        )


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
