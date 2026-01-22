"""Tests for wizard transition observability."""

import time

import pytest

from dataknobs_bots.reasoning.observability import (
    TransitionHistoryQuery,
    TransitionRecord,
    TransitionStats,
    TransitionTracker,
    WizardStateSnapshot,
    create_transition_record,
    # Conversion utilities
    execution_record_to_transition_record,
    transition_record_to_execution_record,
    transition_stats_to_execution_stats,
    # FSM types
    ExecutionRecord,
    ExecutionStats,
)


class TestTransitionRecord:
    """Tests for TransitionRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a transition record."""
        timestamp = time.time()
        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=timestamp,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
        )

        assert record.from_stage == "welcome"
        assert record.to_stage == "configure"
        assert record.timestamp == timestamp
        assert record.trigger == "user_input"
        assert record.duration_in_stage_ms == 5000.0
        assert record.data_snapshot is None
        assert record.user_input is None
        assert record.condition_evaluated is None
        assert record.condition_result is None
        assert record.error is None

    def test_create_record_with_all_fields(self) -> None:
        """Test creating a record with all fields."""
        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=time.time(),
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"intent": "create"},
            user_input="I want to create something",
            condition_evaluated="data.get('intent')",
            condition_result=True,
            error=None,
        )

        assert record.data_snapshot == {"intent": "create"}
        assert record.user_input == "I want to create something"
        assert record.condition_evaluated == "data.get('intent')"
        assert record.condition_result is True

    def test_create_record_with_condition(self) -> None:
        """Test creating a record with condition evaluation info."""
        record = TransitionRecord(
            from_stage="start",
            to_stage="path_a",
            timestamp=time.time(),
            trigger="user_input",
            condition_evaluated="data.get('choice') == 'a'",
            condition_result=True,
        )

        assert record.condition_evaluated == "data.get('choice') == 'a'"
        assert record.condition_result is True

    def test_create_record_with_failed_condition(self) -> None:
        """Test creating a record where condition was evaluated but returned False."""
        # This could happen if we track all evaluated conditions, not just the successful one
        record = TransitionRecord(
            from_stage="start",
            to_stage="default",
            timestamp=time.time(),
            trigger="user_input",
            condition_evaluated=None,  # Default path, no condition
            condition_result=None,
        )

        assert record.condition_evaluated is None
        assert record.condition_result is None

    def test_create_error_record(self) -> None:
        """Test creating a record with error."""
        record = TransitionRecord(
            from_stage="configure",
            to_stage="configure",  # Same stage - transition failed
            timestamp=time.time(),
            trigger="user_input",
            error="Validation failed",
        )

        assert record.error == "Validation failed"

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        timestamp = time.time()
        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=timestamp,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"key": "value"},
            condition_evaluated="data.get('intent')",
            condition_result=True,
        )

        data = record.to_dict()

        assert data["from_stage"] == "welcome"
        assert data["to_stage"] == "configure"
        assert data["timestamp"] == timestamp
        assert data["trigger"] == "user_input"
        assert data["duration_in_stage_ms"] == 5000.0
        assert data["data_snapshot"] == {"key": "value"}
        assert data["condition_evaluated"] == "data.get('intent')"
        assert data["condition_result"] is True

    def test_from_dict(self) -> None:
        """Test creating record from dictionary."""
        data = {
            "from_stage": "welcome",
            "to_stage": "configure",
            "timestamp": 1234567890.0,
            "trigger": "navigation_back",
            "duration_in_stage_ms": 3000.0,
            "data_snapshot": None,
            "user_input": "back",
            "condition_evaluated": "data.get('choice') == 'a'",
            "condition_result": True,
            "error": None,
        }

        record = TransitionRecord.from_dict(data)

        assert record.from_stage == "welcome"
        assert record.to_stage == "configure"
        assert record.timestamp == 1234567890.0
        assert record.trigger == "navigation_back"
        assert record.user_input == "back"
        assert record.condition_evaluated == "data.get('choice') == 'a'"
        assert record.condition_result is True


class TestCreateTransitionRecord:
    """Tests for the create_transition_record factory function."""

    def test_create_with_factory(self) -> None:
        """Test the factory function sets timestamp automatically."""
        before = time.time()
        record = create_transition_record(
            from_stage="welcome",
            to_stage="configure",
            trigger="user_input",
            duration_in_stage_ms=5000.0,
        )
        after = time.time()

        assert record.from_stage == "welcome"
        assert record.to_stage == "configure"
        assert before <= record.timestamp <= after
        assert record.trigger == "user_input"
        assert record.duration_in_stage_ms == 5000.0

    def test_create_with_all_fields(self) -> None:
        """Test factory with all optional fields."""
        record = create_transition_record(
            from_stage="configure",
            to_stage="complete",
            trigger="user_input",
            duration_in_stage_ms=10000.0,
            data_snapshot={"config": "done"},
            user_input="finish",
            condition_evaluated="data.get('config')",
            condition_result=True,
            error=None,
        )

        assert record.data_snapshot == {"config": "done"}
        assert record.user_input == "finish"
        assert record.condition_evaluated == "data.get('config')"
        assert record.condition_result is True


class TestTransitionHistoryQuery:
    """Tests for TransitionHistoryQuery dataclass."""

    def test_default_query(self) -> None:
        """Test query with default values."""
        query = TransitionHistoryQuery()

        assert query.from_stage is None
        assert query.to_stage is None
        assert query.trigger is None
        assert query.since is None
        assert query.until is None
        assert query.limit is None

    def test_custom_query(self) -> None:
        """Test query with custom values."""
        query = TransitionHistoryQuery(
            from_stage="welcome",
            to_stage="configure",
            trigger="user_input",
            since=1000.0,
            until=2000.0,
            limit=10,
        )

        assert query.from_stage == "welcome"
        assert query.to_stage == "configure"
        assert query.trigger == "user_input"
        assert query.since == 1000.0
        assert query.until == 2000.0
        assert query.limit == 10


class TestTransitionStats:
    """Tests for TransitionStats dataclass."""

    def test_default_stats(self) -> None:
        """Test stats with default values."""
        stats = TransitionStats()

        assert stats.total_transitions == 0
        assert stats.unique_paths == 0
        assert stats.avg_duration_per_stage_ms == 0.0
        assert stats.most_common_trigger is None
        assert stats.backtrack_count == 0
        assert stats.restart_count == 0
        assert stats.has_backtracks is False

    def test_has_backtracks_property(self) -> None:
        """Test has_backtracks property."""
        stats = TransitionStats(backtrack_count=0)
        assert stats.has_backtracks is False

        stats = TransitionStats(backtrack_count=2)
        assert stats.has_backtracks is True


class TestTransitionTracker:
    """Tests for TransitionTracker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        tracker = TransitionTracker()
        assert len(tracker) == 0

    def test_init_with_max_history(self) -> None:
        """Test initialization with custom max history."""
        tracker = TransitionTracker(max_history=50)
        assert len(tracker) == 0

    def test_record_transition(self) -> None:
        """Test recording a transition."""
        tracker = TransitionTracker()

        record = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=time.time(),
            trigger="user_input",
        )

        tracker.record(record)
        assert len(tracker) == 1

    def test_max_history_enforcement(self) -> None:
        """Test that max history is enforced."""
        tracker = TransitionTracker(max_history=5)

        for i in range(10):
            tracker.record(
                TransitionRecord(
                    from_stage=f"stage_{i}",
                    to_stage=f"stage_{i+1}",
                    timestamp=time.time() + i,
                    trigger="user_input",
                )
            )

        assert len(tracker) == 5
        # Should keep the most recent 5
        results = tracker.query()
        assert results[0].from_stage == "stage_5"
        assert results[-1].from_stage == "stage_9"

    def test_query_all(self) -> None:
        """Test querying all records."""
        tracker = TransitionTracker()

        for i in range(3):
            tracker.record(
                TransitionRecord(
                    from_stage=f"stage_{i}",
                    to_stage=f"stage_{i+1}",
                    timestamp=time.time(),
                    trigger="user_input",
                )
            )

        results = tracker.query()
        assert len(results) == 3

    def test_query_by_from_stage(self) -> None:
        """Test filtering by from_stage."""
        tracker = TransitionTracker()

        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=time.time(),
                trigger="user_input",
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="complete",
                timestamp=time.time(),
                trigger="user_input",
            )
        )

        query = TransitionHistoryQuery(from_stage="welcome")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].from_stage == "welcome"

    def test_query_by_to_stage(self) -> None:
        """Test filtering by to_stage."""
        tracker = TransitionTracker()

        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=time.time(),
                trigger="user_input",
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="complete",
                timestamp=time.time(),
                trigger="user_input",
            )
        )

        query = TransitionHistoryQuery(to_stage="complete")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].to_stage == "complete"

    def test_query_by_trigger(self) -> None:
        """Test filtering by trigger."""
        tracker = TransitionTracker()

        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=time.time(),
                trigger="user_input",
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="welcome",
                timestamp=time.time(),
                trigger="navigation_back",
            )
        )

        query = TransitionHistoryQuery(trigger="navigation_back")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].trigger == "navigation_back"

    def test_query_by_time_range(self) -> None:
        """Test filtering by time range."""
        tracker = TransitionTracker()

        now = time.time()
        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=now - 100,
                trigger="user_input",
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="complete",
                timestamp=now,
                trigger="user_input",
            )
        )

        # Query for recent records only
        query = TransitionHistoryQuery(since=now - 50)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].from_stage == "configure"

    def test_query_with_limit(self) -> None:
        """Test limiting query results."""
        tracker = TransitionTracker()

        for i in range(10):
            tracker.record(
                TransitionRecord(
                    from_stage=f"stage_{i}",
                    to_stage=f"stage_{i+1}",
                    timestamp=time.time() + i,
                    trigger="user_input",
                )
            )

        query = TransitionHistoryQuery(limit=3)
        results = tracker.query(query)

        assert len(results) == 3
        # Should return last 3 (most recent)
        assert results[0].from_stage == "stage_7"
        assert results[2].from_stage == "stage_9"

    def test_get_stats_empty(self) -> None:
        """Test getting stats with no records."""
        tracker = TransitionTracker()
        stats = tracker.get_stats()

        assert stats.total_transitions == 0
        assert stats.unique_paths == 0

    def test_get_stats_with_data(self) -> None:
        """Test getting stats with records."""
        tracker = TransitionTracker()

        # Add some transitions with different triggers
        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=time.time(),
                trigger="user_input",
                duration_in_stage_ms=5000.0,
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="welcome",
                timestamp=time.time(),
                trigger="navigation_back",
                duration_in_stage_ms=3000.0,
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="welcome",
                to_stage="configure",
                timestamp=time.time(),
                trigger="user_input",
                duration_in_stage_ms=4000.0,
            )
        )
        tracker.record(
            TransitionRecord(
                from_stage="configure",
                to_stage="welcome",
                timestamp=time.time(),
                trigger="restart",
                duration_in_stage_ms=2000.0,
            )
        )

        stats = tracker.get_stats()

        assert stats.total_transitions == 4
        assert stats.unique_paths == 2  # welcome->configure, configure->welcome
        assert stats.avg_duration_per_stage_ms == pytest.approx(3500.0)
        assert stats.most_common_trigger == "user_input"
        assert stats.backtrack_count == 1
        assert stats.restart_count == 1
        assert stats.has_backtracks is True

    def test_clear(self) -> None:
        """Test clearing transition history."""
        tracker = TransitionTracker()

        for i in range(5):
            tracker.record(
                TransitionRecord(
                    from_stage=f"stage_{i}",
                    to_stage=f"stage_{i+1}",
                    timestamp=time.time(),
                    trigger="user_input",
                )
            )

        assert len(tracker) == 5

        tracker.clear()

        assert len(tracker) == 0
        assert tracker.query() == []


class TestWizardStateSnapshot:
    """Tests for WizardStateSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Test creating a wizard state snapshot."""
        timestamp = time.time()
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            data={"intent": "create"},
            history=["welcome", "configure"],
            transitions=[],
            completed=False,
            snapshot_timestamp=timestamp,
            clarification_attempts=0,
        )

        assert snapshot.current_stage == "configure"
        assert snapshot.data == {"intent": "create"}
        assert snapshot.history == ["welcome", "configure"]
        assert snapshot.transitions == []
        assert snapshot.completed is False
        assert snapshot.snapshot_timestamp == timestamp
        assert snapshot.clarification_attempts == 0

    def test_to_dict(self) -> None:
        """Test converting snapshot to dictionary."""
        transition = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=1234567890.0,
            trigger="user_input",
        )
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            data={"key": "value"},
            history=["welcome", "configure"],
            transitions=[transition],
        )

        data = snapshot.to_dict()

        assert data["current_stage"] == "configure"
        assert data["data"] == {"key": "value"}
        assert data["history"] == ["welcome", "configure"]
        assert len(data["transitions"]) == 1
        assert data["transitions"][0]["from_stage"] == "welcome"

    def test_from_dict(self) -> None:
        """Test creating snapshot from dictionary."""
        data = {
            "current_stage": "complete",
            "data": {"result": "done"},
            "history": ["welcome", "configure", "complete"],
            "transitions": [
                {
                    "from_stage": "welcome",
                    "to_stage": "configure",
                    "timestamp": 1234567890.0,
                    "trigger": "user_input",
                    "duration_in_stage_ms": 5000.0,
                    "data_snapshot": None,
                    "user_input": None,
                    "condition_evaluated": "data.get('intent')",
                    "condition_result": True,
                    "error": None,
                }
            ],
            "completed": True,
            "snapshot_timestamp": 1234567899.0,
            "clarification_attempts": 0,
        }

        snapshot = WizardStateSnapshot.from_dict(data)

        assert snapshot.current_stage == "complete"
        assert snapshot.data == {"result": "done"}
        assert len(snapshot.history) == 3
        assert len(snapshot.transitions) == 1
        assert snapshot.transitions[0].from_stage == "welcome"
        assert snapshot.transitions[0].condition_evaluated == "data.get('intent')"
        assert snapshot.transitions[0].condition_result is True
        assert snapshot.completed is True


class TestConversionUtilities:
    """Tests for conversion utilities between wizard and FSM types."""

    def test_transition_record_to_execution_record(self) -> None:
        """Test converting TransitionRecord to ExecutionRecord."""
        transition = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=1000.0,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"intent": "create"},
            user_input="I want to create",
            condition_evaluated="data.get('intent')",
            condition_result=True,
            error=None,
        )

        execution = transition_record_to_execution_record(transition)

        assert execution.from_state == "welcome"
        assert execution.to_state == "configure"
        assert execution.timestamp == 1000.0
        assert execution.trigger == "user_input"
        assert execution.duration_in_state_ms == 5000.0
        assert execution.data_before is None
        assert execution.data_after == {"intent": "create"}
        assert execution.condition_evaluated == "data.get('intent')"
        assert execution.condition_result is True
        assert execution.success is True
        assert execution.error is None

    def test_transition_record_to_execution_record_with_error(self) -> None:
        """Test that error presence sets success=False."""
        transition = TransitionRecord(
            from_stage="configure",
            to_stage="configure",
            timestamp=1000.0,
            trigger="user_input",
            error="Validation failed",
        )

        execution = transition_record_to_execution_record(transition)

        assert execution.success is False
        assert execution.error == "Validation failed"

    def test_execution_record_to_transition_record(self) -> None:
        """Test converting ExecutionRecord to TransitionRecord."""
        execution = ExecutionRecord(
            from_state="start",
            to_state="processing",
            timestamp=2000.0,
            trigger="step",
            transition_name="begin_processing",
            duration_in_state_ms=100.0,
            data_before={"count": 0},
            data_after={"count": 5},
            condition_evaluated="count < 10",
            condition_result=True,
            success=True,
            error=None,
        )

        transition = execution_record_to_transition_record(
            execution, user_input="process"
        )

        assert transition.from_stage == "start"
        assert transition.to_stage == "processing"
        assert transition.timestamp == 2000.0
        assert transition.trigger == "step"
        assert transition.duration_in_stage_ms == 100.0
        assert transition.data_snapshot == {"count": 5}  # Uses data_after
        assert transition.user_input == "process"
        assert transition.condition_evaluated == "count < 10"
        assert transition.condition_result is True
        assert transition.error is None

    def test_execution_record_to_transition_record_without_user_input(self) -> None:
        """Test conversion without specifying user_input."""
        execution = ExecutionRecord(
            from_state="a",
            to_state="b",
            timestamp=1000.0,
            trigger="auto",
            success=True,
        )

        transition = execution_record_to_transition_record(execution)

        assert transition.user_input is None

    def test_transition_stats_to_execution_stats(self) -> None:
        """Test converting TransitionStats to ExecutionStats."""
        transition_stats = TransitionStats(
            total_transitions=10,
            unique_paths=4,
            avg_duration_per_stage_ms=500.0,
            most_common_trigger="user_input",
            backtrack_count=2,
            restart_count=1,
            first_transition=1000.0,
            last_transition=5000.0,
        )

        execution_stats = transition_stats_to_execution_stats(transition_stats)

        assert execution_stats.total_transitions == 10
        assert execution_stats.unique_paths == 4
        assert execution_stats.avg_duration_per_state_ms == 500.0
        assert execution_stats.most_common_trigger == "user_input"
        assert execution_stats.first_transition == 1000.0
        assert execution_stats.last_transition == 5000.0
        # Wizard doesn't track success/failure the same way
        assert execution_stats.successful_transitions == 10
        assert execution_stats.failed_transitions == 0
        # Wizard doesn't track most_visited_state
        assert execution_stats.most_visited_state is None

    def test_round_trip_conversion(self) -> None:
        """Test that round-trip conversion preserves key data."""
        original = TransitionRecord(
            from_stage="welcome",
            to_stage="configure",
            timestamp=1000.0,
            trigger="user_input",
            duration_in_stage_ms=5000.0,
            data_snapshot={"key": "value"},
            user_input=None,
            condition_evaluated="data.get('key')",
            condition_result=True,
            error=None,
        )

        # Convert to FSM and back
        execution = transition_record_to_execution_record(original)
        restored = execution_record_to_transition_record(execution)

        assert restored.from_stage == original.from_stage
        assert restored.to_stage == original.to_stage
        assert restored.timestamp == original.timestamp
        assert restored.trigger == original.trigger
        assert restored.duration_in_stage_ms == original.duration_in_stage_ms
        assert restored.data_snapshot == original.data_snapshot
        assert restored.condition_evaluated == original.condition_evaluated
        assert restored.condition_result == original.condition_result
        # user_input is lost in round-trip (FSM doesn't have it)
        assert restored.user_input is None
