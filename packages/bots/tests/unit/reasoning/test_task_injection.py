"""Tests for task injection system."""

import pytest

from dataknobs_bots.reasoning.observability import WizardTask
from dataknobs_bots.reasoning.task_injection import (
    TaskInjectionContext,
    TaskInjectionResult,
    TaskInjector,
)


class TestTaskInjectionContext:
    """Tests for TaskInjectionContext."""

    def test_basic_creation(self) -> None:
        """Test basic context creation."""
        ctx = TaskInjectionContext(
            event="artifact_created",
            current_stage="build_questions",
            wizard_data={"subject": "math"},
        )

        assert ctx.event == "artifact_created"
        assert ctx.current_stage == "build_questions"
        assert ctx.wizard_data["subject"] == "math"

    def test_for_artifact_created(self) -> None:
        """Test artifact_created context factory."""
        artifact = type("Artifact", (), {"id": "art-123"})()

        ctx = TaskInjectionContext.for_artifact_created(
            artifact=artifact,
            current_stage="build",
            wizard_data={"v": 1},
        )

        assert ctx.event == "artifact_created"
        assert ctx.artifact == artifact
        assert ctx.current_stage == "build"
        assert ctx.wizard_data["v"] == 1

    def test_for_artifact_reviewed(self) -> None:
        """Test artifact_reviewed context factory."""
        artifact = type("Artifact", (), {"id": "art-123"})()
        review = type("Review", (), {"passed": True, "score": 0.9})()

        ctx = TaskInjectionContext.for_artifact_reviewed(
            artifact=artifact,
            review=review,
            current_stage="review",
            wizard_data={},
        )

        assert ctx.event == "artifact_reviewed"
        assert ctx.artifact == artifact
        assert ctx.review == review

    def test_for_stage_entered(self) -> None:
        """Test stage_entered context factory."""
        ctx = TaskInjectionContext.for_stage_entered(
            stage="configure",
            from_stage="welcome",
            wizard_data={"name": "Bot"},
        )

        assert ctx.event == "stage_entered"
        assert ctx.current_stage == "configure"
        assert ctx.stage_from == "welcome"
        assert ctx.stage_to == "configure"

    def test_for_stage_exited(self) -> None:
        """Test stage_exited context factory."""
        ctx = TaskInjectionContext.for_stage_exited(
            stage="welcome",
            to_stage="configure",
            wizard_data={},
        )

        assert ctx.event == "stage_exited"
        assert ctx.current_stage == "welcome"
        assert ctx.stage_from == "welcome"
        assert ctx.stage_to == "configure"

    def test_for_review_failed(self) -> None:
        """Test review_failed context factory."""
        artifact = type("Artifact", (), {"id": "art-123"})()
        review = type("Review", (), {"passed": False})()

        ctx = TaskInjectionContext.for_review_failed(
            artifact=artifact,
            review=review,
            current_stage="review",
            wizard_data={},
        )

        assert ctx.event == "review_failed"
        assert ctx.artifact == artifact
        assert ctx.review == review


class TestTaskInjectionResult:
    """Tests for TaskInjectionResult."""

    def test_empty_result(self) -> None:
        """Test empty result."""
        result = TaskInjectionResult()

        assert result.tasks_to_add == []
        assert result.tasks_to_complete == []
        assert result.tasks_to_skip == []
        assert result.messages == []
        assert result.block_transition is False
        assert result.has_changes is False

    def test_result_with_tasks(self) -> None:
        """Test result with tasks."""
        task = WizardTask(
            id="review_task",
            description="Review artifact",
            status="pending",
        )
        result = TaskInjectionResult(tasks_to_add=[task])

        assert len(result.tasks_to_add) == 1
        assert result.has_changes is True

    def test_merge_results(self) -> None:
        """Test merging two results."""
        task1 = WizardTask(id="t1", description="Task 1")
        task2 = WizardTask(id="t2", description="Task 2")

        result1 = TaskInjectionResult(
            tasks_to_add=[task1],
            tasks_to_complete=["old1"],
            messages=["msg1"],
        )
        result2 = TaskInjectionResult(
            tasks_to_add=[task2],
            tasks_to_complete=["old2"],
            messages=["msg2"],
            block_transition=True,
            block_reason="Failed review",
        )

        merged = result1.merge(result2)

        assert len(merged.tasks_to_add) == 2
        assert len(merged.tasks_to_complete) == 2
        assert len(merged.messages) == 2
        assert merged.block_transition is True
        assert merged.block_reason == "Failed review"


class TestTaskInjector:
    """Tests for TaskInjector."""

    def test_init(self) -> None:
        """Test injector initialization."""
        injector = TaskInjector()

        # Should have all supported events
        assert "artifact_created" in injector.EVENTS
        assert "artifact_reviewed" in injector.EVENTS
        assert "review_failed" in injector.EVENTS
        assert "stage_entered" in injector.EVENTS
        assert "stage_exited" in injector.EVENTS
        assert "wizard_completed" in injector.EVENTS

    def test_register_hook(self) -> None:
        """Test registering a hook."""
        injector = TaskInjector()

        def my_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult()

        injector.register("artifact_created", my_hook)

        assert injector.has_hooks("artifact_created")

    def test_register_invalid_event(self) -> None:
        """Test registering for invalid event raises error."""
        injector = TaskInjector()

        def my_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult()

        with pytest.raises(ValueError, match="Unknown event"):
            injector.register("invalid_event", my_hook)

    def test_decorator_registration(self) -> None:
        """Test decorator-based registration."""
        injector = TaskInjector()

        @injector.on("stage_entered")
        def my_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(messages=["Entered stage"])

        assert injector.has_hooks("stage_entered")

        # Trigger and verify
        ctx = TaskInjectionContext(event="stage_entered")
        result = injector.trigger("stage_entered", ctx)

        assert "Entered stage" in result.messages

    def test_trigger_hooks(self) -> None:
        """Test triggering hooks."""
        injector = TaskInjector()

        def add_review_task(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(
                tasks_to_add=[
                    WizardTask(
                        id="review_task",
                        description="Review the artifact",
                    )
                ]
            )

        injector.register("artifact_created", add_review_task)

        ctx = TaskInjectionContext.for_artifact_created(
            artifact=type("Artifact", (), {"id": "art-1"})(),
            current_stage="build",
            wizard_data={},
        )
        result = injector.trigger("artifact_created", ctx)

        assert len(result.tasks_to_add) == 1
        assert result.tasks_to_add[0].id == "review_task"

    def test_trigger_multiple_hooks(self) -> None:
        """Test triggering multiple hooks for same event."""
        injector = TaskInjector()

        def hook1(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(messages=["hook1"])

        def hook2(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(messages=["hook2"])

        injector.register("stage_entered", hook1)
        injector.register("stage_entered", hook2)

        ctx = TaskInjectionContext(event="stage_entered")
        result = injector.trigger("stage_entered", ctx)

        assert "hook1" in result.messages
        assert "hook2" in result.messages

    def test_trigger_no_hooks(self) -> None:
        """Test triggering event with no hooks."""
        injector = TaskInjector()

        ctx = TaskInjectionContext(event="artifact_created")
        result = injector.trigger("artifact_created", ctx)

        assert result.has_changes is False

    def test_hook_exception_handling(self) -> None:
        """Test that hook exceptions are logged and skipped."""
        injector = TaskInjector()

        def bad_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            raise ValueError("Hook failed")

        def good_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(messages=["good"])

        injector.register("artifact_created", bad_hook)
        injector.register("artifact_created", good_hook)

        ctx = TaskInjectionContext(event="artifact_created")
        result = injector.trigger("artifact_created", ctx)

        # Good hook should still execute
        assert "good" in result.messages

    def test_unregister_hook(self) -> None:
        """Test unregistering a hook."""
        injector = TaskInjector()

        def my_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult()

        injector.register("artifact_created", my_hook)
        assert injector.has_hooks("artifact_created")

        success = injector.unregister("artifact_created", my_hook)
        assert success is True
        assert injector.has_hooks("artifact_created") is False

    def test_clear_hooks(self) -> None:
        """Test clearing hooks."""
        injector = TaskInjector()

        def my_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult()

        injector.register("artifact_created", my_hook)
        injector.register("stage_entered", my_hook)

        # Clear specific event
        injector.clear("artifact_created")
        assert injector.has_hooks("artifact_created") is False
        assert injector.has_hooks("stage_entered") is True

        # Clear all
        injector.clear()
        assert injector.has_hooks("stage_entered") is False

    def test_from_config_with_custom_functions(self) -> None:
        """Test creating injector from config with custom functions."""
        def custom_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
            return TaskInjectionResult(messages=["custom"])

        config = {
            "hooks": {
                "artifact_created": [
                    {"function": "my_custom_hook"},
                ],
            },
        }

        injector = TaskInjector.from_config(
            config,
            custom_functions={"my_custom_hook": custom_hook},
        )

        assert injector.has_hooks("artifact_created")

        ctx = TaskInjectionContext(event="artifact_created")
        result = injector.trigger("artifact_created", ctx)
        assert "custom" in result.messages
