"""Tests for wizard transition observability and task tracking."""

import time

import pytest

from dataknobs_bots.reasoning.observability import (
    TransitionHistoryQuery,
    TransitionRecord,
    TransitionStats,
    TransitionTracker,
    WizardStateSnapshot,
    WizardTask,
    WizardTaskList,
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


class TestWizardTask:
    """Tests for WizardTask dataclass."""

    def test_create_task(self) -> None:
        """Test creating a task with default values."""
        task = WizardTask(
            id="collect_name",
            description="Collect user name",
        )

        assert task.id == "collect_name"
        assert task.description == "Collect user name"
        assert task.status == "pending"
        assert task.stage is None
        assert task.required is True
        assert task.completed_at is None
        assert task.depends_on == []
        assert task.completed_by is None
        assert task.field_name is None
        assert task.tool_name is None

    def test_create_task_with_all_fields(self) -> None:
        """Test creating a task with all fields specified."""
        task = WizardTask(
            id="collect_bot_name",
            description="Collect bot name",
            status="pending",
            stage="configure_identity",
            required=True,
            depends_on=["validate_config"],
            completed_by="field_extraction",
            field_name="bot_name",
        )

        assert task.id == "collect_bot_name"
        assert task.stage == "configure_identity"
        assert task.completed_by == "field_extraction"
        assert task.field_name == "bot_name"
        assert task.depends_on == ["validate_config"]

    def test_create_tool_task(self) -> None:
        """Test creating a task triggered by tool result."""
        task = WizardTask(
            id="run_validate",
            description="Run validation tool",
            completed_by="tool_result",
            tool_name="validate_config",
            required=True,
        )

        assert task.completed_by == "tool_result"
        assert task.tool_name == "validate_config"

    def test_is_complete_property(self) -> None:
        """Test is_complete property."""
        task = WizardTask(id="t1", description="Task 1")
        assert task.is_complete is False

        task.status = "completed"
        assert task.is_complete is True

        task.status = "skipped"
        assert task.is_complete is False

    def test_is_pending_property(self) -> None:
        """Test is_pending property."""
        task = WizardTask(id="t1", description="Task 1")
        assert task.is_pending is True

        task.status = "in_progress"
        assert task.is_pending is False

        task.status = "completed"
        assert task.is_pending is False

    def test_is_global_property(self) -> None:
        """Test is_global property."""
        # Global task (no stage)
        global_task = WizardTask(id="save", description="Save")
        assert global_task.is_global is True

        # Stage-specific task
        stage_task = WizardTask(id="collect", description="Collect", stage="welcome")
        assert stage_task.is_global is False

    def test_to_dict(self) -> None:
        """Test converting task to dictionary."""
        task = WizardTask(
            id="test_task",
            description="Test task",
            status="completed",
            stage="configure",
            required=True,
            completed_at=1234567890.0,
            depends_on=["other_task"],
            completed_by="field_extraction",
            field_name="test_field",
        )

        data = task.to_dict()

        assert data["id"] == "test_task"
        assert data["description"] == "Test task"
        assert data["status"] == "completed"
        assert data["stage"] == "configure"
        assert data["required"] is True
        assert data["completed_at"] == 1234567890.0
        assert data["depends_on"] == ["other_task"]
        assert data["completed_by"] == "field_extraction"
        assert data["field_name"] == "test_field"

    def test_from_dict(self) -> None:
        """Test creating task from dictionary."""
        data = {
            "id": "my_task",
            "description": "My task",
            "status": "in_progress",
            "stage": "step1",
            "required": False,
            "completed_at": None,
            "depends_on": ["dep1", "dep2"],
            "completed_by": "tool_result",
            "field_name": None,
            "tool_name": "my_tool",
        }

        task = WizardTask.from_dict(data)

        assert task.id == "my_task"
        assert task.status == "in_progress"
        assert task.stage == "step1"
        assert task.required is False
        assert task.depends_on == ["dep1", "dep2"]
        assert task.completed_by == "tool_result"
        assert task.tool_name == "my_tool"


class TestWizardTaskList:
    """Tests for WizardTaskList class."""

    def test_empty_task_list(self) -> None:
        """Test empty task list."""
        task_list = WizardTaskList()

        assert len(task_list) == 0
        assert task_list.get_pending_tasks() == []
        assert task_list.get_completed_tasks() == []
        assert task_list.calculate_progress() == 100.0  # No required tasks

    def test_get_task(self) -> None:
        """Test getting a task by ID."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
            WizardTask(id="t2", description="Task 2"),
        ])

        assert task_list.get_task("t1") is not None
        assert task_list.get_task("t1").description == "Task 1"
        assert task_list.get_task("t2") is not None
        assert task_list.get_task("nonexistent") is None

    def test_complete_task(self) -> None:
        """Test completing a task."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
        ])

        before = time.time()
        result = task_list.complete_task("t1")
        after = time.time()

        assert result is True
        task = task_list.get_task("t1")
        assert task.status == "completed"
        assert task.completed_at is not None
        assert before <= task.completed_at <= after

    def test_complete_nonexistent_task(self) -> None:
        """Test completing a task that doesn't exist."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
        ])

        result = task_list.complete_task("nonexistent")
        assert result is False

    def test_complete_task_with_dependencies(self) -> None:
        """Test that tasks with unmet dependencies cannot be completed."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="validate", description="Validate"),
            WizardTask(id="save", description="Save", depends_on=["validate"]),
        ])

        # Cannot complete "save" before "validate"
        result = task_list.complete_task("save")
        assert result is False
        assert task_list.get_task("save").status == "pending"

        # Complete "validate" first
        task_list.complete_task("validate")

        # Now "save" can be completed
        result = task_list.complete_task("save")
        assert result is True
        assert task_list.get_task("save").status == "completed"

    def test_skip_task(self) -> None:
        """Test skipping a task."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
        ])

        result = task_list.skip_task("t1")

        assert result is True
        assert task_list.get_task("t1").status == "skipped"

    def test_skip_nonexistent_task(self) -> None:
        """Test skipping a task that doesn't exist."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
        ])

        result = task_list.skip_task("nonexistent")
        assert result is False

    def test_skipped_dependency_allows_completion(self) -> None:
        """Test that skipped dependencies count as met."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="validate", description="Validate"),
            WizardTask(id="save", description="Save", depends_on=["validate"]),
        ])

        # Skip "validate"
        task_list.skip_task("validate")

        # "save" should now be completable
        result = task_list.complete_task("save")
        assert result is True

    def test_get_pending_tasks(self) -> None:
        """Test getting pending tasks."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", status="pending"),
            WizardTask(id="t2", description="Task 2", status="completed"),
            WizardTask(id="t3", description="Task 3", status="pending"),
        ])

        pending = task_list.get_pending_tasks()

        assert len(pending) == 2
        assert all(t.status == "pending" for t in pending)
        assert {t.id for t in pending} == {"t1", "t3"}

    def test_get_completed_tasks(self) -> None:
        """Test getting completed tasks."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", status="completed"),
            WizardTask(id="t2", description="Task 2", status="pending"),
            WizardTask(id="t3", description="Task 3", status="completed"),
        ])

        completed = task_list.get_completed_tasks()

        assert len(completed) == 2
        assert all(t.status == "completed" for t in completed)
        assert {t.id for t in completed} == {"t1", "t3"}

    def test_get_available_tasks(self) -> None:
        """Test getting tasks that are ready to execute."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="validate", description="Validate"),
            WizardTask(id="save", description="Save", depends_on=["validate"]),
            WizardTask(id="preview", description="Preview"),
        ])

        # Initially only "validate" and "preview" are available
        available = task_list.get_available_tasks()
        assert len(available) == 2
        assert {t.id for t in available} == {"validate", "preview"}

        # Complete "validate"
        task_list.complete_task("validate")

        # Now all pending tasks are available
        available = task_list.get_available_tasks()
        assert len(available) == 2
        assert {t.id for t in available} == {"save", "preview"}

    def test_get_tasks_for_stage(self) -> None:
        """Test getting tasks for a specific stage."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", stage="configure"),
            WizardTask(id="t2", description="Task 2", stage="complete"),
            WizardTask(id="t3", description="Task 3", stage="configure"),
            WizardTask(id="global", description="Global task"),  # No stage
        ])

        configure_tasks = task_list.get_tasks_for_stage("configure")

        assert len(configure_tasks) == 2
        assert all(t.stage == "configure" for t in configure_tasks)
        assert {t.id for t in configure_tasks} == {"t1", "t3"}

    def test_get_global_tasks(self) -> None:
        """Test getting global (stage-independent) tasks."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", stage="configure"),
            WizardTask(id="validate", description="Validate"),  # Global
            WizardTask(id="save", description="Save"),  # Global
        ])

        global_tasks = task_list.get_global_tasks()

        assert len(global_tasks) == 2
        assert all(t.stage is None for t in global_tasks)
        assert {t.id for t in global_tasks} == {"validate", "save"}

    def test_calculate_progress_all_pending(self) -> None:
        """Test progress calculation with all tasks pending."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", required=True),
            WizardTask(id="t2", description="Task 2", required=True),
        ])

        assert task_list.calculate_progress() == 0.0

    def test_calculate_progress_partial(self) -> None:
        """Test progress calculation with some tasks completed."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", required=True, status="completed"),
            WizardTask(id="t2", description="Task 2", required=True, status="pending"),
        ])

        assert task_list.calculate_progress() == 50.0

    def test_calculate_progress_all_completed(self) -> None:
        """Test progress calculation with all tasks completed."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", required=True, status="completed"),
            WizardTask(id="t2", description="Task 2", required=True, status="completed"),
        ])

        assert task_list.calculate_progress() == 100.0

    def test_calculate_progress_optional_ignored(self) -> None:
        """Test that optional tasks don't affect progress."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", required=True, status="completed"),
            WizardTask(id="t2", description="Task 2", required=True, status="pending"),
            WizardTask(id="opt", description="Optional", required=False, status="pending"),
        ])

        # Progress is 1/2 required = 50%
        assert task_list.calculate_progress() == 50.0

    def test_calculate_progress_skipped_counts(self) -> None:
        """Test that skipped tasks count toward progress."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1", required=True, status="skipped"),
            WizardTask(id="t2", description="Task 2", required=True, status="pending"),
        ])

        # 1/2 tasks skipped = 50%
        assert task_list.calculate_progress() == 50.0

    def test_to_dict(self) -> None:
        """Test converting task list to dictionary."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
            WizardTask(id="t2", description="Task 2"),
        ])

        data = task_list.to_dict()

        assert "tasks" in data
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["id"] == "t1"
        assert data["tasks"][1]["id"] == "t2"

    def test_from_dict(self) -> None:
        """Test creating task list from dictionary."""
        data = {
            "tasks": [
                {"id": "t1", "description": "Task 1", "status": "pending",
                 "stage": None, "required": True, "completed_at": None,
                 "depends_on": [], "completed_by": None, "field_name": None,
                 "tool_name": None},
                {"id": "t2", "description": "Task 2", "status": "completed",
                 "stage": "step1", "required": True, "completed_at": 1234567890.0,
                 "depends_on": ["t1"], "completed_by": None, "field_name": None,
                 "tool_name": None},
            ]
        }

        task_list = WizardTaskList.from_dict(data)

        assert len(task_list) == 2
        assert task_list.get_task("t1").status == "pending"
        assert task_list.get_task("t2").status == "completed"
        assert task_list.get_task("t2").stage == "step1"

    def test_from_dict_empty(self) -> None:
        """Test creating task list from empty dictionary."""
        task_list = WizardTaskList.from_dict({})
        assert len(task_list) == 0

    def test_len(self) -> None:
        """Test __len__ method."""
        task_list = WizardTaskList(tasks=[
            WizardTask(id="t1", description="Task 1"),
            WizardTask(id="t2", description="Task 2"),
            WizardTask(id="t3", description="Task 3"),
        ])

        assert len(task_list) == 3


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

    def test_create_snapshot_with_tasks(self) -> None:
        """Test creating a snapshot with task data."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            tasks=[
                {"id": "t1", "description": "Task 1", "status": "completed"},
                {"id": "t2", "description": "Task 2", "status": "pending"},
            ],
            pending_tasks=1,
            completed_tasks=1,
            total_tasks=2,
            available_task_ids=["t2"],
            task_progress_percent=50.0,
        )

        assert len(snapshot.tasks) == 2
        assert snapshot.pending_tasks == 1
        assert snapshot.completed_tasks == 1
        assert snapshot.total_tasks == 2
        assert snapshot.available_task_ids == ["t2"]
        assert snapshot.task_progress_percent == 50.0

    def test_create_snapshot_with_stage_context(self) -> None:
        """Test creating a snapshot with stage context fields."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            stage_index=2,
            total_stages=5,
            can_skip=True,
            can_go_back=False,
            suggestions=["Option A", "Option B"],
        )

        assert snapshot.stage_index == 2
        assert snapshot.total_stages == 5
        assert snapshot.can_skip is True
        assert snapshot.can_go_back is False
        assert snapshot.suggestions == ["Option A", "Option B"]

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
            tasks=[{"id": "t1", "status": "pending"}],
            pending_tasks=1,
            completed_tasks=0,
            total_tasks=1,
        )

        data = snapshot.to_dict()

        assert data["current_stage"] == "configure"
        assert data["data"] == {"key": "value"}
        assert data["history"] == ["welcome", "configure"]
        assert len(data["transitions"]) == 1
        assert data["transitions"][0]["from_stage"] == "welcome"
        assert data["tasks"] == [{"id": "t1", "status": "pending"}]
        assert data["pending_tasks"] == 1
        assert data["completed_tasks"] == 0
        assert data["total_tasks"] == 1

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
            "tasks": [{"id": "t1", "description": "Task 1", "status": "completed"}],
            "pending_tasks": 0,
            "completed_tasks": 1,
            "total_tasks": 1,
            "available_task_ids": [],
            "task_progress_percent": 100.0,
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
        assert len(snapshot.tasks) == 1
        assert snapshot.completed_tasks == 1
        assert snapshot.task_progress_percent == 100.0

    def test_get_task(self) -> None:
        """Test getting a task by ID from snapshot."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            tasks=[
                {"id": "t1", "description": "Task 1", "status": "completed"},
                {"id": "t2", "description": "Task 2", "status": "pending"},
            ],
        )

        task = snapshot.get_task("t1")
        assert task is not None
        assert task["id"] == "t1"
        assert task["status"] == "completed"

        assert snapshot.get_task("nonexistent") is None

    def test_get_tasks_for_stage(self) -> None:
        """Test getting tasks for a specific stage."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            tasks=[
                {"id": "t1", "stage": "configure"},
                {"id": "t2", "stage": "complete"},
                {"id": "t3", "stage": "configure"},
            ],
        )

        configure_tasks = snapshot.get_tasks_for_stage("configure")

        assert len(configure_tasks) == 2
        assert all(t["stage"] == "configure" for t in configure_tasks)

    def test_get_global_tasks(self) -> None:
        """Test getting global tasks from snapshot."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            tasks=[
                {"id": "t1", "stage": "configure"},
                {"id": "validate", "stage": None},
                {"id": "save", "stage": None},
            ],
        )

        global_tasks = snapshot.get_global_tasks()

        assert len(global_tasks) == 2
        assert all(t["stage"] is None for t in global_tasks)

    def test_is_task_available(self) -> None:
        """Test checking if task is available."""
        snapshot = WizardStateSnapshot(
            current_stage="configure",
            available_task_ids=["t1", "t3"],
        )

        assert snapshot.is_task_available("t1") is True
        assert snapshot.is_task_available("t2") is False
        assert snapshot.is_task_available("t3") is True

    def test_get_latest_transition(self) -> None:
        """Test getting the most recent transition."""
        snapshot = WizardStateSnapshot(
            current_stage="complete",
            transitions=[
                TransitionRecord(
                    from_stage="welcome",
                    to_stage="configure",
                    timestamp=1000.0,
                    trigger="user_input",
                ),
                TransitionRecord(
                    from_stage="configure",
                    to_stage="complete",
                    timestamp=2000.0,
                    trigger="user_input",
                ),
            ],
        )

        latest = snapshot.get_latest_transition()

        assert latest is not None
        assert latest["from_stage"] == "configure"
        assert latest["to_stage"] == "complete"

    def test_get_latest_transition_empty(self) -> None:
        """Test getting latest transition when no transitions exist."""
        snapshot = WizardStateSnapshot(current_stage="welcome")

        assert snapshot.get_latest_transition() is None


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
