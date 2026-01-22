"""Tests for tool execution observability."""

import time

import pytest

from dataknobs_llm.tools.observability import (
    ExecutionHistoryQuery,
    ExecutionStats,
    ExecutionTracker,
    ToolExecutionRecord,
    create_execution_record,
)


class TestToolExecutionRecord:
    """Tests for ToolExecutionRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating an execution record."""
        timestamp = time.time()
        record = ToolExecutionRecord(
            tool_name="calculator",
            timestamp=timestamp,
            parameters={"operation": "add", "a": 5, "b": 3},
            result=8,
            duration_ms=1.5,
            success=True,
        )

        assert record.tool_name == "calculator"
        assert record.timestamp == timestamp
        assert record.parameters == {"operation": "add", "a": 5, "b": 3}
        assert record.result == 8
        assert record.duration_ms == 1.5
        assert record.success is True
        assert record.error is None
        assert record.context_id is None

    def test_create_failed_record(self) -> None:
        """Test creating a failed execution record."""
        record = ToolExecutionRecord(
            tool_name="calculator",
            timestamp=time.time(),
            parameters={"operation": "divide", "a": 10, "b": 0},
            result=None,
            duration_ms=0.5,
            success=False,
            error="Division by zero",
        )

        assert record.success is False
        assert record.error == "Division by zero"
        assert record.result is None

    def test_record_with_context_id(self) -> None:
        """Test creating record with context ID."""
        record = ToolExecutionRecord(
            tool_name="search",
            timestamp=time.time(),
            parameters={"query": "test"},
            result=["result1"],
            duration_ms=100.0,
            success=True,
            context_id="conv-123",
        )

        assert record.context_id == "conv-123"

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        timestamp = time.time()
        record = ToolExecutionRecord(
            tool_name="calculator",
            timestamp=timestamp,
            parameters={"a": 1, "b": 2},
            result=3,
            duration_ms=1.0,
            success=True,
            context_id="ctx-1",
        )

        data = record.to_dict()

        assert data["tool_name"] == "calculator"
        assert data["timestamp"] == timestamp
        assert data["parameters"] == {"a": 1, "b": 2}
        assert data["result"] == 3
        assert data["duration_ms"] == 1.0
        assert data["success"] is True
        assert data["error"] is None
        assert data["context_id"] == "ctx-1"

    def test_from_dict(self) -> None:
        """Test creating record from dictionary."""
        data = {
            "tool_name": "search",
            "timestamp": 1234567890.0,
            "parameters": {"query": "hello"},
            "result": ["world"],
            "duration_ms": 50.0,
            "success": True,
            "error": None,
            "context_id": "ctx-2",
        }

        record = ToolExecutionRecord.from_dict(data)

        assert record.tool_name == "search"
        assert record.timestamp == 1234567890.0
        assert record.parameters == {"query": "hello"}
        assert record.result == ["world"]
        assert record.duration_ms == 50.0
        assert record.success is True
        assert record.context_id == "ctx-2"


class TestCreateExecutionRecord:
    """Tests for the create_execution_record factory function."""

    def test_create_with_factory(self) -> None:
        """Test the factory function sets timestamp automatically."""
        before = time.time()
        record = create_execution_record(
            tool_name="test_tool",
            parameters={"key": "value"},
            result="result",
            duration_ms=10.0,
            success=True,
        )
        after = time.time()

        assert record.tool_name == "test_tool"
        assert before <= record.timestamp <= after
        assert record.parameters == {"key": "value"}
        assert record.result == "result"
        assert record.duration_ms == 10.0
        assert record.success is True

    def test_create_failed_with_factory(self) -> None:
        """Test creating failed record with factory."""
        record = create_execution_record(
            tool_name="failing_tool",
            parameters={},
            result=None,
            duration_ms=5.0,
            success=False,
            error="Something went wrong",
            context_id="ctx-fail",
        )

        assert record.success is False
        assert record.error == "Something went wrong"
        assert record.context_id == "ctx-fail"


class TestExecutionHistoryQuery:
    """Tests for ExecutionHistoryQuery dataclass."""

    def test_default_query(self) -> None:
        """Test query with default values."""
        query = ExecutionHistoryQuery()

        assert query.tool_name is None
        assert query.context_id is None
        assert query.since is None
        assert query.until is None
        assert query.success_only is False
        assert query.failed_only is False
        assert query.limit is None

    def test_custom_query(self) -> None:
        """Test query with custom values."""
        query = ExecutionHistoryQuery(
            tool_name="calculator",
            context_id="ctx-123",
            since=1000.0,
            until=2000.0,
            success_only=True,
            limit=10,
        )

        assert query.tool_name == "calculator"
        assert query.context_id == "ctx-123"
        assert query.since == 1000.0
        assert query.until == 2000.0
        assert query.success_only is True
        assert query.limit == 10


class TestExecutionStats:
    """Tests for ExecutionStats dataclass."""

    def test_default_stats(self) -> None:
        """Test stats with default values."""
        stats = ExecutionStats()

        assert stats.tool_name is None
        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.avg_duration_ms == 0.0
        assert stats.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        stats = ExecutionStats(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
        )

        assert stats.success_rate == 80.0

    def test_success_rate_zero_executions(self) -> None:
        """Test success rate with zero executions."""
        stats = ExecutionStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_successful(self) -> None:
        """Test success rate with all successful."""
        stats = ExecutionStats(
            total_executions=5,
            successful_executions=5,
            failed_executions=0,
        )

        assert stats.success_rate == 100.0


class TestExecutionTracker:
    """Tests for ExecutionTracker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        tracker = ExecutionTracker()
        assert len(tracker) == 0

    def test_init_with_max_history(self) -> None:
        """Test initialization with custom max history."""
        tracker = ExecutionTracker(max_history=50)
        assert len(tracker) == 0

    def test_record_execution(self) -> None:
        """Test recording an execution."""
        tracker = ExecutionTracker()

        record = ToolExecutionRecord(
            tool_name="test",
            timestamp=time.time(),
            parameters={},
            result="ok",
            duration_ms=1.0,
            success=True,
        )

        tracker.record(record)
        assert len(tracker) == 1

    def test_max_history_enforcement(self) -> None:
        """Test that max history is enforced."""
        tracker = ExecutionTracker(max_history=5)

        for i in range(10):
            tracker.record(
                ToolExecutionRecord(
                    tool_name=f"tool_{i}",
                    timestamp=time.time() + i,
                    parameters={},
                    result=i,
                    duration_ms=1.0,
                    success=True,
                )
            )

        assert len(tracker) == 5
        # Should keep the most recent 5
        results = tracker.query()
        tool_names = [r.tool_name for r in results]
        assert "tool_5" in tool_names
        assert "tool_9" in tool_names
        assert "tool_0" not in tool_names

    def test_query_all(self) -> None:
        """Test querying all records."""
        tracker = ExecutionTracker()

        for i in range(3):
            tracker.record(
                ToolExecutionRecord(
                    tool_name="test",
                    timestamp=time.time(),
                    parameters={},
                    result=i,
                    duration_ms=1.0,
                    success=True,
                )
            )

        results = tracker.query()
        assert len(results) == 3

    def test_query_by_tool_name(self) -> None:
        """Test filtering by tool name."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="calculator",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=1.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="search",
                timestamp=time.time(),
                parameters={},
                result=2,
                duration_ms=1.0,
                success=True,
            )
        )

        query = ExecutionHistoryQuery(tool_name="calculator")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].tool_name == "calculator"

    def test_query_by_context_id(self) -> None:
        """Test filtering by context ID."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=1.0,
                success=True,
                context_id="ctx-1",
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=2,
                duration_ms=1.0,
                success=True,
                context_id="ctx-2",
            )
        )

        query = ExecutionHistoryQuery(context_id="ctx-1")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].context_id == "ctx-1"

    def test_query_by_time_range(self) -> None:
        """Test filtering by time range."""
        tracker = ExecutionTracker()

        now = time.time()
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=now - 100,
                parameters={},
                result=1,
                duration_ms=1.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=now,
                parameters={},
                result=2,
                duration_ms=1.0,
                success=True,
            )
        )

        # Query for recent records only
        query = ExecutionHistoryQuery(since=now - 50)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].result == 2

    def test_query_success_only(self) -> None:
        """Test filtering for successful executions only."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=1.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=None,
                duration_ms=1.0,
                success=False,
                error="Failed",
            )
        )

        query = ExecutionHistoryQuery(success_only=True)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].success is True

    def test_query_failed_only(self) -> None:
        """Test filtering for failed executions only."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=1.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool",
                timestamp=time.time(),
                parameters={},
                result=None,
                duration_ms=1.0,
                success=False,
                error="Failed",
            )
        )

        query = ExecutionHistoryQuery(failed_only=True)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].success is False

    def test_query_with_limit(self) -> None:
        """Test limiting query results."""
        tracker = ExecutionTracker()

        for i in range(10):
            tracker.record(
                ToolExecutionRecord(
                    tool_name="tool",
                    timestamp=time.time() + i,
                    parameters={},
                    result=i,
                    duration_ms=1.0,
                    success=True,
                )
            )

        query = ExecutionHistoryQuery(limit=3)
        results = tracker.query(query)

        assert len(results) == 3
        # Should return last 3 (most recent)
        assert [r.result for r in results] == [7, 8, 9]

    def test_get_stats_empty(self) -> None:
        """Test getting stats with no records."""
        tracker = ExecutionTracker()
        stats = tracker.get_stats()

        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0

    def test_get_stats_all_tools(self) -> None:
        """Test getting stats for all tools."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="tool1",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=10.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool2",
                timestamp=time.time(),
                parameters={},
                result=2,
                duration_ms=20.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool1",
                timestamp=time.time(),
                parameters={},
                result=None,
                duration_ms=5.0,
                success=False,
                error="Error",
            )
        )

        stats = tracker.get_stats()

        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert stats.failed_executions == 1
        assert stats.avg_duration_ms == pytest.approx(35.0 / 3)
        assert stats.min_duration_ms == 5.0
        assert stats.max_duration_ms == 20.0

    def test_get_stats_specific_tool(self) -> None:
        """Test getting stats for a specific tool."""
        tracker = ExecutionTracker()

        tracker.record(
            ToolExecutionRecord(
                tool_name="tool1",
                timestamp=time.time(),
                parameters={},
                result=1,
                duration_ms=10.0,
                success=True,
            )
        )
        tracker.record(
            ToolExecutionRecord(
                tool_name="tool2",
                timestamp=time.time(),
                parameters={},
                result=2,
                duration_ms=20.0,
                success=True,
            )
        )

        stats = tracker.get_stats("tool1")

        assert stats.tool_name == "tool1"
        assert stats.total_executions == 1
        assert stats.avg_duration_ms == 10.0

    def test_clear(self) -> None:
        """Test clearing execution history."""
        tracker = ExecutionTracker()

        for i in range(5):
            tracker.record(
                ToolExecutionRecord(
                    tool_name="tool",
                    timestamp=time.time(),
                    parameters={},
                    result=i,
                    duration_ms=1.0,
                    success=True,
                )
            )

        assert len(tracker) == 5

        tracker.clear()

        assert len(tracker) == 0
        assert tracker.query() == []
