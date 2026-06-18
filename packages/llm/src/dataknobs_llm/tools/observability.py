"""Tool execution observability for tracking tool invocations.

This module provides data structures for recording tool executions,
enabling observability, debugging, and auditing of tool usage.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.capabilities import (
    Capability,
    CapabilityLike,
    CapabilityMixin,
)

EXECUTION_RECORD_TOPIC = "execution:record"
"""Topic fired on :meth:`ExecutionTracker.record`.

Payload keys: ``tool_name``, ``success`` (bool), ``duration_ms``
(float), ``error`` (the error message string, or ``None`` on success).
"""


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution.

    Captures all relevant information about a tool invocation including
    timing, parameters, results, and any errors that occurred.

    Attributes:
        tool_name: Name of the executed tool
        timestamp: Unix timestamp when execution started
        parameters: Parameters passed to the tool (sanitized)
        result: Result returned by the tool (None if failed)
        duration_ms: Execution duration in milliseconds
        success: Whether execution completed without error
        error: Error message if execution failed
        context_id: Optional conversation/request ID for correlation

    Example:
        ```python
        record = ToolExecutionRecord(
            tool_name="calculator",
            timestamp=time.time(),
            parameters={"operation": "add", "a": 5, "b": 3},
            result=8,
            duration_ms=1.5,
            success=True,
        )
        ```
    """

    tool_name: str
    timestamp: float
    parameters: dict[str, Any]
    result: Any
    duration_ms: float
    success: bool
    error: str | None = None
    context_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation of the record
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary containing record fields

        Returns:
            ToolExecutionRecord instance
        """
        return cls(**data)


@dataclass
class ExecutionHistoryQuery:
    """Query parameters for filtering execution history.

    Attributes:
        tool_name: Filter by tool name
        context_id: Filter by context/conversation ID
        since: Filter to records after this timestamp
        until: Filter to records before this timestamp
        success_only: Only include successful executions
        failed_only: Only include failed executions
        limit: Maximum number of records to return
    """

    tool_name: str | None = None
    context_id: str | None = None
    since: float | None = None
    until: float | None = None
    success_only: bool = False
    failed_only: bool = False
    limit: int | None = None


@dataclass
class ExecutionStats:
    """Aggregated statistics for tool executions.

    Attributes:
        tool_name: Name of the tool (or None for aggregate)
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        avg_duration_ms: Average execution duration
        min_duration_ms: Minimum execution duration
        max_duration_ms: Maximum execution duration
        first_execution: Timestamp of first execution
        last_execution: Timestamp of last execution
    """

    tool_name: str | None = None
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    first_execution: float | None = None
    last_execution: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate between 0.0 and 100.0
        """
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100.0


class ExecutionTracker(CapabilityMixin):
    """Tracks tool execution history with query capabilities.

    Manages a bounded history of tool executions and provides
    methods for querying and aggregating execution data.

    Composes an in-process callback registry: every :meth:`record`
    fires the ``execution:record`` topic on :attr:`execution_callbacks`
    so consumers can observe tool executions live (guard callbacks,
    metrics, audit) without polling the history. The registry is opt-in
    — it is constructed only on first access, so the existing
    ``record / query / get_stats / clear / __len__`` surface is
    byte-identical for consumers that never touch it.

    Attributes:
        max_history: Maximum number of records to retain

    Example:
        ```python
        tracker = ExecutionTracker(max_history=1000)

        # Record an execution
        tracker.record(ToolExecutionRecord(
            tool_name="search",
            timestamp=time.time(),
            parameters={"query": "test"},
            result=["result1"],
            duration_ms=150.0,
            success=True,
        ))

        # Query history
        recent = tracker.query(ExecutionHistoryQuery(
            tool_name="search",
            since=time.time() - 3600,
        ))

        # Get stats
        stats = tracker.get_stats("search")
        ```
    """

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset({
        Capability.CALLBACK_REGISTRY,
        Capability.EXECUTION_TRACKING,
    })

    def __init__(self, max_history: int = 100):
        """Initialize tracker.

        Args:
            max_history: Maximum records to retain (default 100)
        """
        self._history: list[ToolExecutionRecord] = []
        self._max_history = max_history
        self._execution_callbacks: CallbackRegistry | None = None

    @property
    def execution_callbacks(self) -> CallbackRegistry:
        """In-process registry receiving tool-execution events.

        Fires the ``execution:record`` topic
        (:data:`EXECUTION_RECORD_TOPIC`) after every :meth:`record`
        call. Payload: ``{tool_name, success, duration_ms, error}``.

        Consumers register callbacks for in-process observability (a
        ``priority=-100`` guard that raises under
        :attr:`~dataknobs_common.callbacks.ErrorPolicy.RAISE` aborts the
        recording path; metrics callbacks aggregate). Compose with
        :meth:`~dataknobs_common.callbacks.CallbackRegistry.also_publish_to`
        for cross-replica fan-out when :meth:`record` is driven outside
        a running event loop. Lazily constructed on first access, so the
        recording path stays allocation-free for consumers that never
        register.
        """
        if self._execution_callbacks is None:
            self._execution_callbacks = CallbackRegistry()
        return self._execution_callbacks

    def record(self, execution: ToolExecutionRecord) -> None:
        """Record a tool execution.

        Args:
            execution: The execution record to store
        """
        self._history.append(execution)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        if self._execution_callbacks is not None:
            self._execution_callbacks.fire(
                EXECUTION_RECORD_TOPIC,
                {
                    "tool_name": execution.tool_name,
                    "success": execution.success,
                    "duration_ms": execution.duration_ms,
                    "error": execution.error,
                },
            )

    def query(
        self, query: ExecutionHistoryQuery | None = None
    ) -> list[ToolExecutionRecord]:
        """Query execution history.

        Args:
            query: Query parameters, or None for all records

        Returns:
            List of matching execution records
        """
        if query is None:
            return list(self._history)

        results = self._history

        if query.tool_name:
            results = [r for r in results if r.tool_name == query.tool_name]

        if query.context_id:
            results = [r for r in results if r.context_id == query.context_id]

        if query.since:
            results = [r for r in results if r.timestamp >= query.since]

        if query.until:
            results = [r for r in results if r.timestamp <= query.until]

        if query.success_only:
            results = [r for r in results if r.success]

        if query.failed_only:
            results = [r for r in results if not r.success]

        if query.limit:
            results = results[-query.limit :]

        return results

    def get_stats(self, tool_name: str | None = None) -> ExecutionStats:
        """Get aggregated statistics.

        Args:
            tool_name: Filter to specific tool, or None for all tools

        Returns:
            ExecutionStats with aggregated metrics
        """
        records = self._history
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]

        if not records:
            return ExecutionStats(tool_name=tool_name)

        durations = [r.duration_ms for r in records]
        successful = [r for r in records if r.success]
        failed = [r for r in records if not r.success]

        return ExecutionStats(
            tool_name=tool_name,
            total_executions=len(records),
            successful_executions=len(successful),
            failed_executions=len(failed),
            avg_duration_ms=sum(durations) / len(durations),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            first_execution=records[0].timestamp,
            last_execution=records[-1].timestamp,
        )

    def clear(self) -> None:
        """Clear all execution history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return number of records in history."""
        return len(self._history)


def create_execution_record(
    tool_name: str,
    parameters: dict[str, Any],
    result: Any,
    duration_ms: float,
    success: bool,
    error: str | None = None,
    context_id: str | None = None,
) -> ToolExecutionRecord:
    """Factory function to create an execution record.

    Convenience function that automatically sets the timestamp.

    Args:
        tool_name: Name of the executed tool
        parameters: Parameters passed to the tool
        result: Result from the tool (None if failed)
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
        error: Error message if failed
        context_id: Optional context/conversation ID

    Returns:
        ToolExecutionRecord with current timestamp
    """
    return ToolExecutionRecord(
        tool_name=tool_name,
        timestamp=time.time(),
        parameters=parameters,
        result=result,
        duration_ms=duration_ms,
        success=success,
        error=error,
        context_id=context_id,
    )
