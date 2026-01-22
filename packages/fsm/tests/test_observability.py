"""Tests for FSM execution observability module."""

import time
from dataclasses import dataclass

import pytest

from dataknobs_fsm.observability import (
    ExecutionHistoryQuery,
    ExecutionRecord,
    ExecutionStats,
    ExecutionTracker,
    create_execution_record,
)


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic execution record."""
        record = ExecutionRecord(
            from_state="start",
            to_state="processing",
            timestamp=1000.0,
            trigger="step",
            success=True,
        )
        assert record.from_state == "start"
        assert record.to_state == "processing"
        assert record.timestamp == 1000.0
        assert record.trigger == "step"
        assert record.success is True
        assert record.error is None

    def test_full_creation(self) -> None:
        """Test creating a record with all fields."""
        record = ExecutionRecord(
            from_state="processing",
            to_state="complete",
            timestamp=2000.0,
            trigger="auto",
            transition_name="finish_processing",
            duration_in_state_ms=5000.0,
            data_before={"count": 0},
            data_after={"count": 10},
            condition_evaluated="data.count >= 10",
            condition_result=True,
            success=True,
            error=None,
        )
        assert record.transition_name == "finish_processing"
        assert record.duration_in_state_ms == 5000.0
        assert record.data_before == {"count": 0}
        assert record.data_after == {"count": 10}
        assert record.condition_evaluated == "data.count >= 10"
        assert record.condition_result is True

    def test_failed_transition(self) -> None:
        """Test creating a failed transition record."""
        record = ExecutionRecord(
            from_state="processing",
            to_state="error",
            timestamp=3000.0,
            trigger="step",
            success=False,
            error="Processing failed: timeout",
        )
        assert record.success is False
        assert record.error == "Processing failed: timeout"

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = ExecutionRecord(
            from_state="start",
            to_state="end",
            timestamp=1000.0,
            trigger="step",
            transition_name="complete",
            duration_in_state_ms=100.0,
            data_before={"x": 1},
            data_after={"x": 2},
            condition_evaluated="x > 0",
            condition_result=True,
            success=True,
            error=None,
        )
        data = record.to_dict()
        assert data["from_state"] == "start"
        assert data["to_state"] == "end"
        assert data["timestamp"] == 1000.0
        assert data["trigger"] == "step"
        assert data["transition_name"] == "complete"
        assert data["duration_in_state_ms"] == 100.0
        assert data["data_before"] == {"x": 1}
        assert data["data_after"] == {"x": 2}
        assert data["condition_evaluated"] == "x > 0"
        assert data["condition_result"] is True
        assert data["success"] is True
        assert data["error"] is None

    def test_from_dict(self) -> None:
        """Test creating record from dictionary."""
        data = {
            "from_state": "a",
            "to_state": "b",
            "timestamp": 500.0,
            "trigger": "external",
            "transition_name": "move",
            "duration_in_state_ms": 50.0,
            "data_before": None,
            "data_after": {"done": True},
            "condition_evaluated": None,
            "condition_result": None,
            "success": True,
            "error": None,
        }
        record = ExecutionRecord.from_dict(data)
        assert record.from_state == "a"
        assert record.to_state == "b"
        assert record.timestamp == 500.0
        assert record.trigger == "external"
        assert record.transition_name == "move"
        assert record.data_after == {"done": True}

    def test_round_trip_dict(self) -> None:
        """Test to_dict and from_dict round trip."""
        original = ExecutionRecord(
            from_state="init",
            to_state="ready",
            timestamp=100.0,
            trigger="auto",
            transition_name="initialize",
            duration_in_state_ms=10.0,
            data_before={"status": "new"},
            data_after={"status": "initialized"},
            condition_evaluated="status == 'new'",
            condition_result=True,
            success=True,
            error=None,
        )
        restored = ExecutionRecord.from_dict(original.to_dict())
        assert restored == original

    def test_from_step_result(self) -> None:
        """Test creating record from a StepResult-like object."""

        @dataclass
        class MockStepResult:
            from_state: str
            to_state: str
            transition: str | None
            data_before: dict | None
            data_after: dict | None
            success: bool
            error: str | None

        step_result = MockStepResult(
            from_state="processing",
            to_state="complete",
            transition="finish",
            data_before={"items": []},
            data_after={"items": ["done"]},
            success=True,
            error=None,
        )
        state_entry_time = time.time() - 5.0  # 5 seconds ago

        record = ExecutionRecord.from_step_result(
            step_result=step_result,
            trigger="auto",
            state_entry_time=state_entry_time,
            condition_evaluated="items.length == 0",
        )

        assert record.from_state == "processing"
        assert record.to_state == "complete"
        assert record.transition_name == "finish"
        assert record.trigger == "auto"
        assert record.data_before == {"items": []}
        assert record.data_after == {"items": ["done"]}
        assert record.condition_evaluated == "items.length == 0"
        assert record.condition_result is True  # Set when condition_evaluated is given
        assert record.success is True
        assert record.duration_in_state_ms >= 4900.0  # ~5 seconds

    def test_from_step_result_no_entry_time(self) -> None:
        """Test creating record without state entry time."""

        @dataclass
        class MockStepResult:
            from_state: str
            to_state: str
            transition: str | None
            data_before: dict | None
            data_after: dict | None
            success: bool
            error: str | None

        step_result = MockStepResult(
            from_state="a",
            to_state="b",
            transition=None,
            data_before=None,
            data_after=None,
            success=True,
            error=None,
        )

        record = ExecutionRecord.from_step_result(step_result)
        assert record.duration_in_state_ms == 0.0
        assert record.condition_result is None  # No condition evaluated


class TestExecutionHistoryQuery:
    """Tests for ExecutionHistoryQuery dataclass."""

    def test_default_query(self) -> None:
        """Test creating query with defaults."""
        query = ExecutionHistoryQuery()
        assert query.from_state is None
        assert query.to_state is None
        assert query.trigger is None
        assert query.transition_name is None
        assert query.since is None
        assert query.until is None
        assert query.success_only is False
        assert query.failed_only is False
        assert query.limit is None

    def test_filtered_query(self) -> None:
        """Test creating query with filters."""
        query = ExecutionHistoryQuery(
            from_state="processing",
            to_state="complete",
            trigger="auto",
            since=1000.0,
            until=2000.0,
            success_only=True,
            limit=10,
        )
        assert query.from_state == "processing"
        assert query.to_state == "complete"
        assert query.trigger == "auto"
        assert query.since == 1000.0
        assert query.until == 2000.0
        assert query.success_only is True
        assert query.limit == 10


class TestExecutionStats:
    """Tests for ExecutionStats dataclass."""

    def test_default_stats(self) -> None:
        """Test creating stats with defaults."""
        stats = ExecutionStats()
        assert stats.total_transitions == 0
        assert stats.successful_transitions == 0
        assert stats.failed_transitions == 0
        assert stats.unique_paths == 0
        assert stats.avg_duration_per_state_ms == 0.0
        assert stats.most_visited_state is None
        assert stats.most_common_trigger is None
        assert stats.first_transition is None
        assert stats.last_transition is None

    def test_success_rate_empty(self) -> None:
        """Test success rate with no transitions."""
        stats = ExecutionStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_successful(self) -> None:
        """Test success rate with all successful transitions."""
        stats = ExecutionStats(
            total_transitions=10,
            successful_transitions=10,
            failed_transitions=0,
        )
        assert stats.success_rate == 100.0

    def test_success_rate_mixed(self) -> None:
        """Test success rate with mixed results."""
        stats = ExecutionStats(
            total_transitions=10,
            successful_transitions=7,
            failed_transitions=3,
        )
        assert stats.success_rate == 70.0

    def test_has_failures_false(self) -> None:
        """Test has_failures when no failures."""
        stats = ExecutionStats(failed_transitions=0)
        assert stats.has_failures is False

    def test_has_failures_true(self) -> None:
        """Test has_failures when there are failures."""
        stats = ExecutionStats(failed_transitions=1)
        assert stats.has_failures is True


class TestExecutionTracker:
    """Tests for ExecutionTracker class."""

    def test_empty_tracker(self) -> None:
        """Test empty tracker state."""
        tracker = ExecutionTracker()
        assert len(tracker) == 0
        assert tracker.query() == []
        assert tracker.get_state_flow() == []

    def test_record_execution(self) -> None:
        """Test recording an execution."""
        tracker = ExecutionTracker()
        record = ExecutionRecord(
            from_state="start",
            to_state="end",
            timestamp=1000.0,
            trigger="step",
            success=True,
        )
        tracker.record(record)
        assert len(tracker) == 1
        assert tracker.query() == [record]

    def test_bounded_history(self) -> None:
        """Test that history is bounded by max_history."""
        tracker = ExecutionTracker(max_history=3)

        for i in range(5):
            tracker.record(
                ExecutionRecord(
                    from_state=f"s{i}",
                    to_state=f"s{i+1}",
                    timestamp=float(i),
                    trigger="step",
                    success=True,
                )
            )

        assert len(tracker) == 3
        # Should have the last 3 records
        records = tracker.query()
        assert records[0].from_state == "s2"
        assert records[1].from_state == "s3"
        assert records[2].from_state == "s4"

    def test_query_by_from_state(self) -> None:
        """Test querying by from_state."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b", to_state="c", timestamp=2.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="d", timestamp=3.0, trigger="step", success=True
            )
        )

        results = tracker.query(ExecutionHistoryQuery(from_state="a"))
        assert len(results) == 2
        assert all(r.from_state == "a" for r in results)

    def test_query_by_to_state(self) -> None:
        """Test querying by to_state."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="x", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b", to_state="y", timestamp=2.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="c", to_state="x", timestamp=3.0, trigger="step", success=True
            )
        )

        results = tracker.query(ExecutionHistoryQuery(to_state="x"))
        assert len(results) == 2
        assert all(r.to_state == "x" for r in results)

    def test_query_by_trigger(self) -> None:
        """Test querying by trigger type."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="auto", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b", to_state="c", timestamp=2.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="c",
                to_state="d",
                timestamp=3.0,
                trigger="external",
                success=True,
            )
        )

        results = tracker.query(ExecutionHistoryQuery(trigger="auto"))
        assert len(results) == 1
        assert results[0].trigger == "auto"

    def test_query_by_transition_name(self) -> None:
        """Test querying by transition name."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a",
                to_state="b",
                timestamp=1.0,
                trigger="step",
                transition_name="move",
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b",
                to_state="c",
                timestamp=2.0,
                trigger="step",
                transition_name="jump",
                success=True,
            )
        )

        results = tracker.query(ExecutionHistoryQuery(transition_name="move"))
        assert len(results) == 1
        assert results[0].transition_name == "move"

    def test_query_by_time_range(self) -> None:
        """Test querying by time range."""
        tracker = ExecutionTracker()
        for i in range(5):
            tracker.record(
                ExecutionRecord(
                    from_state=f"s{i}",
                    to_state=f"s{i+1}",
                    timestamp=float(i * 100),
                    trigger="step",
                    success=True,
                )
            )

        # Query records between 100 and 300
        results = tracker.query(ExecutionHistoryQuery(since=100.0, until=300.0))
        assert len(results) == 3
        assert results[0].timestamp == 100.0
        assert results[1].timestamp == 200.0
        assert results[2].timestamp == 300.0

    def test_query_success_only(self) -> None:
        """Test querying only successful transitions."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b",
                to_state="c",
                timestamp=2.0,
                trigger="step",
                success=False,
                error="failed",
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="c", to_state="d", timestamp=3.0, trigger="step", success=True
            )
        )

        results = tracker.query(ExecutionHistoryQuery(success_only=True))
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_query_failed_only(self) -> None:
        """Test querying only failed transitions."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="b",
                to_state="c",
                timestamp=2.0,
                trigger="step",
                success=False,
                error="failed",
            )
        )

        results = tracker.query(ExecutionHistoryQuery(failed_only=True))
        assert len(results) == 1
        assert results[0].success is False

    def test_query_with_limit(self) -> None:
        """Test querying with limit."""
        tracker = ExecutionTracker()
        for i in range(10):
            tracker.record(
                ExecutionRecord(
                    from_state=f"s{i}",
                    to_state=f"s{i+1}",
                    timestamp=float(i),
                    trigger="step",
                    success=True,
                )
            )

        results = tracker.query(ExecutionHistoryQuery(limit=3))
        assert len(results) == 3
        # Should return the last 3
        assert results[0].from_state == "s7"
        assert results[1].from_state == "s8"
        assert results[2].from_state == "s9"

    def test_query_combined_filters(self) -> None:
        """Test querying with multiple filters."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="a",
                to_state="b",
                timestamp=2.0,
                trigger="step",
                success=False,
                error="fail",
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="c", timestamp=3.0, trigger="step", success=True
            )
        )

        results = tracker.query(
            ExecutionHistoryQuery(from_state="a", to_state="b", success_only=True)
        )
        assert len(results) == 1
        assert results[0].timestamp == 1.0

    def test_get_stats_empty(self) -> None:
        """Test getting stats from empty tracker."""
        tracker = ExecutionTracker()
        stats = tracker.get_stats()
        assert stats.total_transitions == 0
        assert stats.success_rate == 0.0

    def test_get_stats(self) -> None:
        """Test getting aggregated stats."""
        tracker = ExecutionTracker()

        # Add various records
        tracker.record(
            ExecutionRecord(
                from_state="start",
                to_state="processing",
                timestamp=100.0,
                trigger="step",
                duration_in_state_ms=50.0,
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="processing",
                to_state="complete",
                timestamp=200.0,
                trigger="auto",
                duration_in_state_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="start",
                to_state="error",
                timestamp=300.0,
                trigger="step",
                duration_in_state_ms=25.0,
                success=False,
                error="oops",
            )
        )

        stats = tracker.get_stats()
        assert stats.total_transitions == 3
        assert stats.successful_transitions == 2
        assert stats.failed_transitions == 1
        assert stats.unique_paths == 3  # start->processing, processing->complete, start->error
        assert stats.avg_duration_per_state_ms == pytest.approx(58.33, rel=0.01)
        assert stats.first_transition == 100.0
        assert stats.last_transition == 300.0
        assert stats.most_common_trigger == "step"  # 2 step vs 1 auto

    def test_get_state_flow(self) -> None:
        """Test getting the state flow sequence."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="start",
                to_state="processing",
                timestamp=1.0,
                trigger="step",
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="processing",
                to_state="validation",
                timestamp=2.0,
                trigger="step",
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="validation",
                to_state="complete",
                timestamp=3.0,
                trigger="step",
                success=True,
            )
        )

        flow = tracker.get_state_flow()
        assert flow == ["start", "processing", "validation", "complete"]

    def test_clear(self) -> None:
        """Test clearing the tracker."""
        tracker = ExecutionTracker()
        tracker.record(
            ExecutionRecord(
                from_state="a", to_state="b", timestamp=1.0, trigger="step", success=True
            )
        )
        tracker.mark_state_entry()

        assert len(tracker) == 1
        tracker.clear()
        assert len(tracker) == 0
        assert tracker.query() == []

    def test_mark_state_entry(self) -> None:
        """Test marking state entry time."""
        tracker = ExecutionTracker()
        tracker.mark_state_entry()

        # The internal state_entry_time should be set
        # We can verify this indirectly through record_from_step_result
        @dataclass
        class MockStepResult:
            from_state: str
            to_state: str
            transition: str | None
            data_before: dict | None
            data_after: dict | None
            success: bool
            error: str | None

        time.sleep(0.01)  # Small delay to ensure measurable duration

        step_result = MockStepResult(
            from_state="a",
            to_state="b",
            transition=None,
            data_before=None,
            data_after=None,
            success=True,
            error=None,
        )
        record = tracker.record_from_step_result(step_result)

        # Duration should be > 0 since we marked state entry
        assert record.duration_in_state_ms > 0

    def test_record_from_step_result(self) -> None:
        """Test recording directly from step result."""
        tracker = ExecutionTracker()
        tracker.mark_state_entry()

        @dataclass
        class MockStepResult:
            from_state: str
            to_state: str
            transition: str | None
            data_before: dict | None
            data_after: dict | None
            success: bool
            error: str | None

        step_result = MockStepResult(
            from_state="processing",
            to_state="complete",
            transition="finish",
            data_before={"x": 1},
            data_after={"x": 2},
            success=True,
            error=None,
        )

        record = tracker.record_from_step_result(
            step_result=step_result,
            trigger="auto",
            condition_evaluated="x > 0",
        )

        assert len(tracker) == 1
        assert record.from_state == "processing"
        assert record.to_state == "complete"
        assert record.trigger == "auto"
        assert record.condition_evaluated == "x > 0"

    def test_most_visited_state(self) -> None:
        """Test that stats correctly identify most visited state."""
        tracker = ExecutionTracker()

        # Visit "processing" twice, others once
        tracker.record(
            ExecutionRecord(
                from_state="start",
                to_state="processing",
                timestamp=1.0,
                trigger="step",
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="processing",
                to_state="validation",
                timestamp=2.0,
                trigger="step",
                success=True,
            )
        )
        tracker.record(
            ExecutionRecord(
                from_state="validation",
                to_state="processing",
                timestamp=3.0,
                trigger="step",
                success=True,
            )
        )

        stats = tracker.get_stats()
        assert stats.most_visited_state == "processing"  # visited as to_state twice


class TestCreateExecutionRecord:
    """Tests for create_execution_record factory function."""

    def test_basic_creation(self) -> None:
        """Test creating a basic record."""
        record = create_execution_record(
            from_state="start",
            to_state="end",
        )
        assert record.from_state == "start"
        assert record.to_state == "end"
        assert record.trigger == "step"  # default
        assert record.success is True  # default
        assert record.timestamp > 0  # auto-set

    def test_full_creation(self) -> None:
        """Test creating a record with all parameters."""
        record = create_execution_record(
            from_state="processing",
            to_state="complete",
            trigger="auto",
            transition_name="finish",
            duration_in_state_ms=500.0,
            data_before={"status": "running"},
            data_after={"status": "done"},
            condition_evaluated="status == 'running'",
            condition_result=True,
            success=True,
            error=None,
        )
        assert record.from_state == "processing"
        assert record.to_state == "complete"
        assert record.trigger == "auto"
        assert record.transition_name == "finish"
        assert record.duration_in_state_ms == 500.0
        assert record.data_before == {"status": "running"}
        assert record.data_after == {"status": "done"}
        assert record.condition_evaluated == "status == 'running'"
        assert record.condition_result is True
        assert record.success is True

    def test_error_creation(self) -> None:
        """Test creating a failed record with error."""
        record = create_execution_record(
            from_state="processing",
            to_state="error",
            success=False,
            error="Something went wrong",
        )
        assert record.success is False
        assert record.error == "Something went wrong"

    def test_timestamp_is_current(self) -> None:
        """Test that timestamp is set to current time."""
        before = time.time()
        record = create_execution_record(from_state="a", to_state="b")
        after = time.time()

        assert before <= record.timestamp <= after
