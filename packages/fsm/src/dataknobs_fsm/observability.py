"""FSM execution observability for tracking state transitions.

This module provides data structures for recording FSM execution history,
enabling observability, debugging, and auditing of state machine flows.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExecutionRecord:
    """Record of a single FSM state transition.

    Captures all relevant information about a state transition including
    timing, states, trigger, and data snapshots.

    Attributes:
        from_state: State name before the transition
        to_state: State name after the transition
        timestamp: Unix timestamp when transition occurred
        trigger: What triggered the transition (e.g., "step", "auto", "external")
        transition_name: Name of the transition/arc that was taken
        duration_in_state_ms: Time spent in the from_state in milliseconds
        data_before: Data state before the transition
        data_after: Data state after the transition
        condition_evaluated: The condition expression that was evaluated (if any)
        condition_result: Result of the condition evaluation (True/False)
        success: Whether the transition completed successfully
        error: Error message if transition failed

    Example:
        ```python
        record = ExecutionRecord(
            from_state="processing",
            to_state="complete",
            timestamp=time.time(),
            trigger="step",
            duration_in_state_ms=5000.0,
            data_before={"items": []},
            data_after={"items": ["processed"]},
            success=True,
        )
        ```
    """

    from_state: str
    to_state: str
    timestamp: float
    trigger: str = "step"
    transition_name: str | None = None
    duration_in_state_ms: float = 0.0
    data_before: dict[str, Any] | None = None
    data_after: dict[str, Any] | None = None
    condition_evaluated: str | None = None
    condition_result: bool | None = None
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation of the record
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary containing record fields

        Returns:
            ExecutionRecord instance
        """
        return cls(**data)

    @classmethod
    def from_step_result(
        cls,
        step_result: Any,
        trigger: str = "step",
        state_entry_time: float | None = None,
        condition_evaluated: str | None = None,
    ) -> "ExecutionRecord":
        """Create record from a StepResult.

        Args:
            step_result: StepResult from AdvancedFSM execution
            trigger: What triggered this transition
            state_entry_time: When the from_state was entered (for duration calc)
            condition_evaluated: The condition expression if known

        Returns:
            ExecutionRecord instance
        """
        now = time.time()
        duration_ms = (now - state_entry_time) * 1000 if state_entry_time else 0.0

        return cls(
            from_state=step_result.from_state,
            to_state=step_result.to_state,
            timestamp=now,
            trigger=trigger,
            transition_name=step_result.transition,
            duration_in_state_ms=duration_ms,
            data_before=step_result.data_before,
            data_after=step_result.data_after,
            condition_evaluated=condition_evaluated,
            condition_result=True if condition_evaluated else None,
            success=step_result.success,
            error=step_result.error,
        )


@dataclass
class ExecutionHistoryQuery:
    """Query parameters for filtering execution history.

    Attributes:
        from_state: Filter by source state
        to_state: Filter by target state
        trigger: Filter by trigger type
        transition_name: Filter by transition name
        since: Filter to records after this timestamp
        until: Filter to records before this timestamp
        success_only: Only include successful transitions
        failed_only: Only include failed transitions
        limit: Maximum number of records to return
    """

    from_state: str | None = None
    to_state: str | None = None
    trigger: str | None = None
    transition_name: str | None = None
    since: float | None = None
    until: float | None = None
    success_only: bool = False
    failed_only: bool = False
    limit: int | None = None


@dataclass
class ExecutionStats:
    """Aggregated statistics for FSM executions.

    Attributes:
        total_transitions: Total number of transitions
        successful_transitions: Number of successful transitions
        failed_transitions: Number of failed transitions
        unique_paths: Number of unique state-to-state paths taken
        avg_duration_per_state_ms: Average time spent in each state
        most_visited_state: State that was visited most often
        most_common_trigger: Most frequent transition trigger
        first_transition: Timestamp of first transition
        last_transition: Timestamp of last transition
    """

    total_transitions: int = 0
    successful_transitions: int = 0
    failed_transitions: int = 0
    unique_paths: int = 0
    avg_duration_per_state_ms: float = 0.0
    most_visited_state: str | None = None
    most_common_trigger: str | None = None
    first_transition: float | None = None
    last_transition: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate between 0.0 and 100.0
        """
        if self.total_transitions == 0:
            return 0.0
        return (self.successful_transitions / self.total_transitions) * 100.0

    @property
    def has_failures(self) -> bool:
        """Check if there were any failed transitions.

        Returns:
            True if failed_transitions > 0
        """
        return self.failed_transitions > 0


class ExecutionTracker:
    """Tracks FSM execution history with query capabilities.

    Manages a bounded history of state transitions and provides
    methods for querying and aggregating execution data.

    Attributes:
        max_history: Maximum number of records to retain

    Example:
        ```python
        tracker = ExecutionTracker(max_history=1000)

        # Record from StepResult
        tracker.record_from_step_result(step_result, trigger="step")

        # Or record directly
        tracker.record(ExecutionRecord(
            from_state="start",
            to_state="processing",
            timestamp=time.time(),
            trigger="step",
            success=True,
        ))

        # Query history
        recent = tracker.query(ExecutionHistoryQuery(
            from_state="processing",
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
        self._history: list[ExecutionRecord] = []
        self._max_history = max_history
        self._state_entry_time: float | None = None

    def record(self, execution: ExecutionRecord) -> None:
        """Record an execution.

        Args:
            execution: The execution record to store
        """
        self._history.append(execution)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        # Update state entry time for next transition
        self._state_entry_time = time.time()

    def record_from_step_result(
        self,
        step_result: Any,
        trigger: str = "step",
        condition_evaluated: str | None = None,
    ) -> ExecutionRecord:
        """Record execution from a StepResult.

        Convenience method that creates an ExecutionRecord from StepResult
        and records it.

        Args:
            step_result: StepResult from AdvancedFSM execution
            trigger: What triggered this transition
            condition_evaluated: The condition expression if known

        Returns:
            The created ExecutionRecord
        """
        record = ExecutionRecord.from_step_result(
            step_result=step_result,
            trigger=trigger,
            state_entry_time=self._state_entry_time,
            condition_evaluated=condition_evaluated,
        )
        self.record(record)
        return record

    def mark_state_entry(self) -> None:
        """Mark the current time as state entry time.

        Call this when entering a new state to track duration.
        """
        self._state_entry_time = time.time()

    def query(
        self, query: ExecutionHistoryQuery | None = None
    ) -> list[ExecutionRecord]:
        """Query execution history.

        Args:
            query: Query parameters, or None for all records

        Returns:
            List of matching execution records
        """
        if query is None:
            return list(self._history)

        results = self._history

        if query.from_state:
            results = [r for r in results if r.from_state == query.from_state]

        if query.to_state:
            results = [r for r in results if r.to_state == query.to_state]

        if query.trigger:
            results = [r for r in results if r.trigger == query.trigger]

        if query.transition_name:
            results = [r for r in results if r.transition_name == query.transition_name]

        if query.since:
            results = [r for r in results if r.timestamp >= query.since]

        if query.until:
            results = [r for r in results if r.timestamp <= query.until]

        if query.success_only:
            results = [r for r in results if r.success]

        if query.failed_only:
            results = [r for r in results if not r.success]

        if query.limit:
            results = results[-query.limit:]

        return results

    def get_stats(self) -> ExecutionStats:
        """Get aggregated execution statistics.

        Returns:
            ExecutionStats with aggregated metrics
        """
        if not self._history:
            return ExecutionStats()

        # Count triggers and states
        trigger_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        unique_paths: set[tuple[str, str]] = set()
        total_duration = 0.0
        successful = 0
        failed = 0

        for record in self._history:
            # Count trigger types
            trigger_counts[record.trigger] = trigger_counts.get(record.trigger, 0) + 1

            # Count state visits (to_state)
            state_counts[record.to_state] = state_counts.get(record.to_state, 0) + 1

            # Track unique paths
            unique_paths.add((record.from_state, record.to_state))

            # Sum durations
            total_duration += record.duration_in_state_ms

            # Count success/failure
            if record.success:
                successful += 1
            else:
                failed += 1

        # Find most common trigger
        most_common_trigger = (
            max(trigger_counts, key=trigger_counts.get) if trigger_counts else None
        )

        # Find most visited state
        most_visited_state = (
            max(state_counts, key=state_counts.get) if state_counts else None
        )

        return ExecutionStats(
            total_transitions=len(self._history),
            successful_transitions=successful,
            failed_transitions=failed,
            unique_paths=len(unique_paths),
            avg_duration_per_state_ms=(
                total_duration / len(self._history) if self._history else 0.0
            ),
            most_visited_state=most_visited_state,
            most_common_trigger=most_common_trigger,
            first_transition=self._history[0].timestamp,
            last_transition=self._history[-1].timestamp,
        )

    def get_state_flow(self) -> list[str]:
        """Get the sequence of states visited.

        Returns:
            List of state names in order visited
        """
        if not self._history:
            return []

        states = [self._history[0].from_state]
        for record in self._history:
            states.append(record.to_state)
        return states

    def clear(self) -> None:
        """Clear all execution history."""
        self._history.clear()
        self._state_entry_time = None

    def __len__(self) -> int:
        """Return number of records in history."""
        return len(self._history)


def create_execution_record(
    from_state: str,
    to_state: str,
    trigger: str = "step",
    transition_name: str | None = None,
    duration_in_state_ms: float = 0.0,
    data_before: dict[str, Any] | None = None,
    data_after: dict[str, Any] | None = None,
    condition_evaluated: str | None = None,
    condition_result: bool | None = None,
    success: bool = True,
    error: str | None = None,
) -> ExecutionRecord:
    """Factory function to create an execution record.

    Convenience function that automatically sets the timestamp.

    Args:
        from_state: State name before transition
        to_state: State name after transition
        trigger: What triggered the transition
        transition_name: Name of the transition/arc
        duration_in_state_ms: Time spent in from_state
        data_before: Data state before transition
        data_after: Data state after transition
        condition_evaluated: The condition expression that was evaluated
        condition_result: Result of the condition evaluation
        success: Whether transition succeeded
        error: Error message if transition failed

    Returns:
        ExecutionRecord with current timestamp
    """
    return ExecutionRecord(
        from_state=from_state,
        to_state=to_state,
        timestamp=time.time(),
        trigger=trigger,
        transition_name=transition_name,
        duration_in_state_ms=duration_in_state_ms,
        data_before=data_before,
        data_after=data_after,
        condition_evaluated=condition_evaluated,
        condition_result=condition_result,
        success=success,
        error=error,
    )
