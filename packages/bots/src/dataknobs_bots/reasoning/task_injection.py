"""Task injection for dynamic wizard task management.

This module provides a system for dynamically injecting tasks into wizard flows
based on events like artifact creation, stage transitions, and reviews.

The task injection system enables:
- Event-driven task creation (e.g., add review tasks when artifacts are created)
- Config-driven hook registration
- Conditional task injection based on context

Example:
    ```python
    injector = TaskInjector()

    @injector.on("artifact_created")
    def add_review_task(ctx: TaskInjectionContext) -> TaskInjectionResult:
        if ctx.artifact and ctx.artifact.definition_id == "assessment_questions":
            return TaskInjectionResult(
                tasks_to_add=[
                    WizardTask(
                        id=f"review_{ctx.artifact.id}",
                        description="Review assessment questions",
                        stage=ctx.current_stage,
                    )
                ]
            )
        return TaskInjectionResult()

    # In wizard flow
    result = injector.trigger("artifact_created", context)
    for task in result.tasks_to_add:
        wizard_state.tasks.tasks.append(task)
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .observability import WizardTask

logger = logging.getLogger(__name__)


@dataclass
class TaskInjectionContext:
    """Context provided to task injection hooks.

    Contains all information needed to decide whether and what tasks to inject.

    Attributes:
        event: The event that triggered injection (e.g., "artifact_created")
        current_stage: Current wizard stage name
        wizard_data: Current wizard collected data
        artifact: Optional artifact that triggered the event
        review: Optional review result that triggered the event
        stage_from: For transitions, the stage being left
        stage_to: For transitions, the stage being entered
        extra: Additional context data
    """

    event: str
    current_stage: str | None = None
    wizard_data: dict[str, Any] = field(default_factory=dict)
    artifact: Any | None = None
    review: Any | None = None
    stage_from: str | None = None
    stage_to: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_artifact_created(
        cls,
        artifact: Any,
        current_stage: str,
        wizard_data: dict[str, Any],
    ) -> TaskInjectionContext:
        """Create context for artifact_created event.

        Args:
            artifact: The created artifact
            current_stage: Current wizard stage
            wizard_data: Current wizard data

        Returns:
            TaskInjectionContext for artifact creation
        """
        return cls(
            event="artifact_created",
            current_stage=current_stage,
            wizard_data=wizard_data,
            artifact=artifact,
        )

    @classmethod
    def for_artifact_reviewed(
        cls,
        artifact: Any,
        review: Any,
        current_stage: str,
        wizard_data: dict[str, Any],
    ) -> TaskInjectionContext:
        """Create context for artifact_reviewed event.

        Args:
            artifact: The reviewed artifact
            review: The review result
            current_stage: Current wizard stage
            wizard_data: Current wizard data

        Returns:
            TaskInjectionContext for artifact review
        """
        return cls(
            event="artifact_reviewed",
            current_stage=current_stage,
            wizard_data=wizard_data,
            artifact=artifact,
            review=review,
        )

    @classmethod
    def for_stage_entered(
        cls,
        stage: str,
        from_stage: str | None,
        wizard_data: dict[str, Any],
    ) -> TaskInjectionContext:
        """Create context for stage_entered event.

        Args:
            stage: The stage being entered
            from_stage: The stage being left (None if first stage)
            wizard_data: Current wizard data

        Returns:
            TaskInjectionContext for stage entry
        """
        return cls(
            event="stage_entered",
            current_stage=stage,
            wizard_data=wizard_data,
            stage_from=from_stage,
            stage_to=stage,
        )

    @classmethod
    def for_stage_exited(
        cls,
        stage: str,
        to_stage: str,
        wizard_data: dict[str, Any],
    ) -> TaskInjectionContext:
        """Create context for stage_exited event.

        Args:
            stage: The stage being exited
            to_stage: The stage being entered
            wizard_data: Current wizard data

        Returns:
            TaskInjectionContext for stage exit
        """
        return cls(
            event="stage_exited",
            current_stage=stage,
            wizard_data=wizard_data,
            stage_from=stage,
            stage_to=to_stage,
        )

    @classmethod
    def for_review_failed(
        cls,
        artifact: Any,
        review: Any,
        current_stage: str,
        wizard_data: dict[str, Any],
    ) -> TaskInjectionContext:
        """Create context for review_failed event.

        Args:
            artifact: The artifact that failed review
            review: The failed review result
            current_stage: Current wizard stage
            wizard_data: Current wizard data

        Returns:
            TaskInjectionContext for review failure
        """
        return cls(
            event="review_failed",
            current_stage=current_stage,
            wizard_data=wizard_data,
            artifact=artifact,
            review=review,
        )


@dataclass
class TaskInjectionResult:
    """Result of a task injection hook.

    Contains tasks to add and optionally tasks to modify or remove.

    Attributes:
        tasks_to_add: New tasks to add to the wizard
        tasks_to_complete: Task IDs to mark as completed
        tasks_to_skip: Task IDs to mark as skipped
        messages: Optional messages to include in response
        block_transition: If True, block stage transition
        block_reason: Reason for blocking transition
    """

    tasks_to_add: list[WizardTask] = field(default_factory=list)
    tasks_to_complete: list[str] = field(default_factory=list)
    tasks_to_skip: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    block_transition: bool = False
    block_reason: str | None = None

    def merge(self, other: TaskInjectionResult) -> TaskInjectionResult:
        """Merge another result into this one.

        Args:
            other: Result to merge

        Returns:
            New merged result
        """
        return TaskInjectionResult(
            tasks_to_add=self.tasks_to_add + other.tasks_to_add,
            tasks_to_complete=self.tasks_to_complete + other.tasks_to_complete,
            tasks_to_skip=self.tasks_to_skip + other.tasks_to_skip,
            messages=self.messages + other.messages,
            block_transition=self.block_transition or other.block_transition,
            block_reason=self.block_reason or other.block_reason,
        )

    @property
    def has_changes(self) -> bool:
        """Check if this result has any changes."""
        return bool(
            self.tasks_to_add
            or self.tasks_to_complete
            or self.tasks_to_skip
            or self.messages
            or self.block_transition
        )


# Type alias for injection hook functions
InjectionHook = Callable[[TaskInjectionContext], TaskInjectionResult]


class TaskInjector:
    """Manages task injection hooks for wizard flows.

    Provides event-driven task injection with support for:
    - Decorator-based hook registration
    - Config-driven hook loading
    - Multiple hooks per event

    Supported events:
    - artifact_created: When a new artifact is created
    - artifact_reviewed: When an artifact review completes
    - review_failed: When an artifact fails review
    - stage_entered: When entering a wizard stage
    - stage_exited: When exiting a wizard stage
    - wizard_completed: When the wizard completes

    Attributes:
        _hooks: Dict mapping event names to lists of hook functions

    Example:
        ```python
        injector = TaskInjector()

        # Register via decorator
        @injector.on("artifact_created")
        def my_hook(ctx):
            return TaskInjectionResult(...)

        # Register directly
        injector.register("stage_entered", another_hook)

        # Trigger hooks
        result = injector.trigger("artifact_created", context)
        ```
    """

    # Supported event types
    EVENTS = frozenset({
        "artifact_created",
        "artifact_reviewed",
        "review_failed",
        "stage_entered",
        "stage_exited",
        "wizard_completed",
    })

    def __init__(self) -> None:
        """Initialize TaskInjector with empty hook registry."""
        self._hooks: dict[str, list[InjectionHook]] = {
            event: [] for event in self.EVENTS
        }

    def on(self, event: str) -> Callable[[InjectionHook], InjectionHook]:
        """Decorator for registering a hook for an event.

        Args:
            event: Event name to register for

        Returns:
            Decorator function

        Raises:
            ValueError: If event is not a supported event type

        Example:
            ```python
            @injector.on("artifact_created")
            def add_review_task(ctx):
                return TaskInjectionResult(...)
            ```
        """
        if event not in self.EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Supported: {sorted(self.EVENTS)}"
            )

        def decorator(func: InjectionHook) -> InjectionHook:
            self._hooks[event].append(func)
            logger.debug("Registered hook %s for event %s", func.__name__, event)
            return func

        return decorator

    def register(self, event: str, hook: InjectionHook) -> None:
        """Register a hook function for an event.

        Args:
            event: Event name to register for
            hook: Hook function to register

        Raises:
            ValueError: If event is not a supported event type
        """
        if event not in self.EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Supported: {sorted(self.EVENTS)}"
            )
        self._hooks[event].append(hook)
        logger.debug("Registered hook %s for event %s", hook.__name__, event)

    def unregister(self, event: str, hook: InjectionHook) -> bool:
        """Unregister a hook function from an event.

        Args:
            event: Event name
            hook: Hook function to unregister

        Returns:
            True if hook was found and removed, False otherwise
        """
        if event not in self._hooks:
            return False
        try:
            self._hooks[event].remove(hook)
            logger.debug("Unregistered hook %s from event %s", hook.__name__, event)
            return True
        except ValueError:
            return False

    def trigger(self, event: str, context: TaskInjectionContext) -> TaskInjectionResult:
        """Trigger all hooks for an event.

        Executes all registered hooks for the event and merges their results.

        Args:
            event: Event name to trigger
            context: Context to pass to hooks

        Returns:
            Merged TaskInjectionResult from all hooks

        Note:
            If a hook raises an exception, it is logged and skipped.
            Other hooks continue to execute.
        """
        result = TaskInjectionResult()

        if event not in self._hooks:
            logger.warning("Attempted to trigger unknown event: %s", event)
            return result

        hooks = self._hooks[event]
        if not hooks:
            return result

        logger.debug("Triggering %d hooks for event %s", len(hooks), event)

        for hook in hooks:
            try:
                hook_result = hook(context)
                if hook_result and hook_result.has_changes:
                    result = result.merge(hook_result)
                    logger.debug(
                        "Hook %s returned %d tasks to add",
                        hook.__name__,
                        len(hook_result.tasks_to_add),
                    )
            except Exception as e:
                logger.error(
                    "Hook %s failed for event %s: %s",
                    hook.__name__,
                    event,
                    e,
                    exc_info=True,
                )

        return result

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event.

        Args:
            event: Event name to check

        Returns:
            True if hooks are registered
        """
        return bool(self._hooks.get(event))

    def clear(self, event: str | None = None) -> None:
        """Clear registered hooks.

        Args:
            event: Specific event to clear, or None to clear all
        """
        if event is not None:
            if event in self._hooks:
                self._hooks[event] = []
        else:
            for evt in self._hooks:
                self._hooks[evt] = []

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any]] | None = None,
    ) -> TaskInjector:
        """Create TaskInjector from configuration.

        Config format:
        ```yaml
        task_injection:
          hooks:
            artifact_created:
              - function: "myapp.hooks:add_review_task"
              - function: "myapp.hooks:notify_created"
            stage_entered:
              - function: "myapp.hooks:stage_entry_hook"
        ```

        Args:
            config: Configuration dict with task_injection section
            custom_functions: Optional dict of pre-loaded functions

        Returns:
            Configured TaskInjector
        """
        injector = cls()
        custom_functions = custom_functions or {}

        hooks_config = config.get("hooks", {})
        for event, hook_list in hooks_config.items():
            if event not in cls.EVENTS:
                logger.warning(
                    "Ignoring unknown event '%s' in task injection config", event
                )
                continue

            for hook_def in hook_list:
                func_ref = hook_def.get("function") if isinstance(hook_def, dict) else hook_def

                if not func_ref:
                    continue

                # Try custom functions first
                if func_ref in custom_functions:
                    injector.register(event, custom_functions[func_ref])
                    continue

                # Try to import the function
                try:
                    hook_func = _import_function(func_ref)
                    injector.register(event, hook_func)
                except Exception as e:
                    logger.warning(
                        "Failed to load hook function '%s' for event '%s': %s",
                        func_ref,
                        event,
                        e,
                    )

        return injector


def _import_function(func_ref: str) -> Callable[..., Any]:
    """Import a function from a module:function reference.

    Args:
        func_ref: Function reference in format "module.path:function_name"

    Returns:
        The imported function

    Raises:
        ValueError: If func_ref format is invalid
        ImportError: If module cannot be imported
        AttributeError: If function not found in module
    """
    if ":" not in func_ref:
        raise ValueError(
            f"Invalid function reference '{func_ref}'. "
            "Expected format: 'module.path:function_name'"
        )

    module_path, func_name = func_ref.rsplit(":", 1)

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# =============================================================================
# Built-in injection hooks
# =============================================================================


def create_review_task_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
    """Built-in hook that creates review tasks when artifacts are created.

    This hook checks if the artifact has a definition with configured reviews,
    and creates tasks for each review protocol.

    Args:
        ctx: Injection context with artifact information

    Returns:
        TaskInjectionResult with review tasks
    """
    from .observability import WizardTask

    if not ctx.artifact:
        return TaskInjectionResult()

    # Check if artifact has reviews configured
    definition_id = getattr(ctx.artifact, "definition_id", None)
    if not definition_id:
        return TaskInjectionResult()

    # Get registry from extra context if available
    registry = ctx.extra.get("artifact_registry")
    if not registry:
        return TaskInjectionResult()

    # Get definition
    definition = registry.get_definition(definition_id)
    if not definition or not definition.reviews:
        return TaskInjectionResult()

    # Create review tasks
    tasks: list[WizardTask] = []
    artifact_id = ctx.artifact.id

    for review_id in definition.reviews:
        task = WizardTask(
            id=f"review_{artifact_id}_{review_id}",
            description=f"Run {review_id} review on artifact",
            status="pending",
            stage=ctx.current_stage,
            required=True,
            completed_by="tool_result",
            tool_name="review_artifact",
        )
        tasks.append(task)

    return TaskInjectionResult(tasks_to_add=tasks)


def block_on_failed_review_hook(ctx: TaskInjectionContext) -> TaskInjectionResult:
    """Built-in hook that blocks transition when a required review fails.

    This hook checks the review result and blocks stage transition if
    the review failed and the artifact definition requires passing reviews.

    Args:
        ctx: Injection context with review information

    Returns:
        TaskInjectionResult with block_transition if review failed
    """
    if not ctx.review or not ctx.artifact:
        return TaskInjectionResult()

    # Check if review passed
    passed = getattr(ctx.review, "passed", True)
    if passed:
        return TaskInjectionResult()

    # Check if definition requires passing reviews
    definition_id = getattr(ctx.artifact, "definition_id", None)
    if not definition_id:
        return TaskInjectionResult()

    registry = ctx.extra.get("artifact_registry")
    if not registry:
        return TaskInjectionResult()

    definition = registry.get_definition(definition_id)
    if not definition:
        return TaskInjectionResult()

    # If required_status_for_stage_completion is "approved", block on failure
    if definition.required_status_for_stage_completion == "approved":
        return TaskInjectionResult(
            block_transition=True,
            block_reason=f"Artifact {ctx.artifact.id} failed review and requires approval",
            messages=[
                f"Review failed for artifact: {getattr(ctx.review, 'feedback', 'No feedback')}"
            ],
        )

    return TaskInjectionResult()
