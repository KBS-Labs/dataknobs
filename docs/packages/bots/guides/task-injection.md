# Task Injection

The task injection system enables dynamic task creation in wizard flows based on events like artifact creation, stage transitions, and review results.

## Overview

Task injection allows wizards to respond dynamically to workflow events:

- **Artifact created** → Add review tasks
- **Review failed** → Add revision tasks
- **Stage entered** → Add stage-specific tasks
- **Stage exited** → Clean up or add follow-up tasks

This makes wizard workflows adaptive rather than static.

## Quick Start

```python
from dataknobs_bots.reasoning import TaskInjector, TaskInjectionContext, TaskInjectionResult
from dataknobs_bots.reasoning.observability import WizardTask

# Create injector
injector = TaskInjector()

# Register a hook
@injector.on("artifact_created")
def add_review_task(ctx: TaskInjectionContext) -> TaskInjectionResult:
    if ctx.artifact and ctx.artifact.definition_id == "assessment_questions":
        return TaskInjectionResult(
            tasks_to_add=[
                WizardTask(
                    id=f"review_{ctx.artifact.id}",
                    description="Review assessment questions",
                    stage=ctx.current_stage,
                    required=True,
                )
            ]
        )
    return TaskInjectionResult()

# Trigger on artifact creation
context = TaskInjectionContext.for_artifact_created(
    artifact=my_artifact,
    current_stage="build_questions",
    wizard_data={"domain": "education"},
)
result = injector.trigger("artifact_created", context)

# Apply injected tasks to wizard
for task in result.tasks_to_add:
    wizard_state.tasks.tasks.append(task)
```

## Supported Events

| Event | When Triggered | Context |
|-------|----------------|---------|
| `artifact_created` | New artifact created | artifact |
| `artifact_reviewed` | Review completed | artifact, review |
| `review_failed` | Review failed | artifact, review |
| `stage_entered` | Entering a stage | stage_to |
| `stage_exited` | Exiting a stage | stage_from, stage_to |
| `wizard_completed` | Wizard finished | - |

## TaskInjectionContext

Context provided to hooks with all relevant information:

```python
from dataknobs_bots.reasoning import TaskInjectionContext

# Factory methods for common events
context = TaskInjectionContext.for_artifact_created(
    artifact=artifact,
    current_stage="build",
    wizard_data={"key": "value"},
)

context = TaskInjectionContext.for_artifact_reviewed(
    artifact=artifact,
    review=review_result,
    current_stage="review",
    wizard_data={},
)

context = TaskInjectionContext.for_stage_entered(
    stage="configuration",
    from_stage="requirements",
    wizard_data={},
)

context = TaskInjectionContext.for_stage_exited(
    stage="requirements",
    to_stage="configuration",
    wizard_data={},
)

context = TaskInjectionContext.for_review_failed(
    artifact=artifact,
    review=failed_review,
    current_stage="review",
    wizard_data={},
)
```

### Context Fields

| Field | Description |
|-------|-------------|
| `event` | Event name that triggered injection |
| `current_stage` | Current wizard stage |
| `wizard_data` | Collected wizard data |
| `artifact` | Artifact (for artifact events) |
| `review` | Review result (for review events) |
| `stage_from` | Stage being left (for transitions) |
| `stage_to` | Stage being entered (for transitions) |
| `extra` | Additional context data |

## TaskInjectionResult

Result returned from hooks:

```python
from dataknobs_bots.reasoning import TaskInjectionResult
from dataknobs_bots.reasoning.observability import WizardTask

result = TaskInjectionResult(
    # Tasks to add
    tasks_to_add=[
        WizardTask(
            id="review_artifact",
            description="Review the generated artifact",
            stage="current_stage",
            required=True,
        ),
    ],

    # Tasks to mark complete
    tasks_to_complete=["task_id_1", "task_id_2"],

    # Tasks to skip
    tasks_to_skip=["optional_task"],

    # Messages to include in response
    messages=["Artifact created successfully. Review tasks added."],

    # Block stage transition
    block_transition=False,
    block_reason=None,
)

# Check if result has changes
if result.has_changes:
    # Apply changes
    pass

# Merge multiple results
combined = result1.merge(result2)
```

## TaskInjector

The main class for managing hooks:

### Registering Hooks

```python
from dataknobs_bots.reasoning import TaskInjector

injector = TaskInjector()

# Decorator registration
@injector.on("artifact_created")
def my_hook(ctx):
    return TaskInjectionResult(...)

# Direct registration
injector.register("stage_entered", another_hook)

# Unregister
injector.unregister("stage_entered", another_hook)
```

### Triggering Events

```python
# Trigger hooks for an event
result = injector.trigger("artifact_created", context)

# Check if hooks exist
if injector.has_hooks("stage_entered"):
    # Trigger
    pass

# Clear hooks
injector.clear("artifact_created")  # Clear specific event
injector.clear()  # Clear all
```

### Configuration-Based Loading

```python
# From configuration
injector = TaskInjector.from_config(
    config={
        "hooks": {
            "artifact_created": [
                {"function": "myapp.hooks:add_review_task"},
                {"function": "myapp.hooks:notify_created"},
            ],
            "stage_entered": [
                {"function": "myapp.hooks:stage_entry_hook"},
            ],
        }
    },
    custom_functions={
        "inline_hook": my_inline_function,
    },
)
```

## Built-in Hooks

### Review Task Hook

Automatically adds review tasks when artifacts with configured reviews are created:

```python
from dataknobs_bots.reasoning.task_injection import create_review_task_hook

injector.register("artifact_created", create_review_task_hook)
```

### Block on Failed Review Hook

Blocks stage transitions when required reviews fail:

```python
from dataknobs_bots.reasoning.task_injection import block_on_failed_review_hook

injector.register("review_failed", block_on_failed_review_hook)
```

## Common Hook Patterns

### Add Review Tasks on Artifact Creation

```python
@injector.on("artifact_created")
def add_review_tasks(ctx: TaskInjectionContext) -> TaskInjectionResult:
    artifact = ctx.artifact
    if not artifact:
        return TaskInjectionResult()

    # Check artifact definition for required reviews
    registry = ctx.extra.get("artifact_registry")
    if not registry:
        return TaskInjectionResult()

    definition = registry.get_definition(artifact.definition_id)
    if not definition or not definition.reviews:
        return TaskInjectionResult()

    # Create review tasks
    tasks = []
    for review_id in definition.reviews:
        tasks.append(WizardTask(
            id=f"review_{artifact.id}_{review_id}",
            description=f"Run {review_id} review",
            stage=ctx.current_stage,
            required=True,
            tool_name="review_artifact",
        ))

    return TaskInjectionResult(
        tasks_to_add=tasks,
        messages=[f"Added {len(tasks)} review tasks"],
    )
```

### Add Revision Task on Failed Review

```python
@injector.on("review_failed")
def add_revision_task(ctx: TaskInjectionContext) -> TaskInjectionResult:
    artifact = ctx.artifact
    review = ctx.review

    if not artifact or not review:
        return TaskInjectionResult()

    return TaskInjectionResult(
        tasks_to_add=[
            WizardTask(
                id=f"revise_{artifact.id}",
                description=f"Revise artifact based on {review.reviewer} feedback",
                stage=ctx.current_stage,
                required=True,
            )
        ],
        messages=[f"Review failed: {review.feedback}. Revision task added."],
    )
```

### Block Transition Until Approved

```python
@injector.on("stage_exited")
def check_approvals(ctx: TaskInjectionContext) -> TaskInjectionResult:
    registry = ctx.extra.get("artifact_registry")
    if not registry:
        return TaskInjectionResult()

    # Check for unapproved artifacts in current stage
    stage_artifacts = registry.get_by_stage(ctx.stage_from)
    unapproved = [a for a in stage_artifacts if a.status != "approved"]

    if unapproved:
        return TaskInjectionResult(
            block_transition=True,
            block_reason=f"Cannot leave {ctx.stage_from}: {len(unapproved)} artifacts not approved",
        )

    return TaskInjectionResult()
```

### Stage-Specific Setup

```python
@injector.on("stage_entered")
def setup_stage_tasks(ctx: TaskInjectionContext) -> TaskInjectionResult:
    stage = ctx.stage_to

    if stage == "collect_requirements":
        return TaskInjectionResult(
            tasks_to_add=[
                WizardTask(
                    id="intro_user",
                    description="Introduce the bot and explain the process",
                    stage=stage,
                    required=True,
                ),
                WizardTask(
                    id="gather_basic_info",
                    description="Collect basic requirements",
                    stage=stage,
                    required=True,
                ),
            ]
        )

    elif stage == "review":
        return TaskInjectionResult(
            tasks_to_add=[
                WizardTask(
                    id="summarize_config",
                    description="Summarize the configuration for user review",
                    stage=stage,
                    required=True,
                ),
            ]
        )

    return TaskInjectionResult()
```

## Configuration

Configure task injection in bot configuration:

```yaml
# bot_config.yaml
task_injection:
  hooks:
    artifact_created:
      - function: "dataknobs_bots.reasoning.task_injection:create_review_task_hook"
      - function: "myapp.hooks:log_artifact_creation"

    review_failed:
      - function: "dataknobs_bots.reasoning.task_injection:block_on_failed_review_hook"
      - function: "myapp.hooks:notify_review_failure"

    stage_entered:
      - function: "myapp.hooks:setup_stage_tasks"

    stage_exited:
      - function: "myapp.hooks:cleanup_stage"

    wizard_completed:
      - function: "myapp.hooks:finalize_wizard"
```

## Integration with WizardReasoning

Task injection integrates with WizardReasoning:

```python
from dataknobs_bots.reasoning import WizardReasoning, TaskInjector

# Create injector with hooks
injector = TaskInjector()

@injector.on("artifact_created")
def my_hook(ctx):
    return TaskInjectionResult(...)

# Pass to wizard
wizard = WizardReasoning(
    wizard_fsm=fsm,
    artifact_registry=registry,
    task_injector=injector,
)

# Or from config with custom functions
wizard = WizardReasoning.from_config(
    config,
    custom_injection_hooks={"my_hook": my_hook},
)
```

## Error Handling

Hooks that raise exceptions are logged and skipped:

```python
@injector.on("artifact_created")
def risky_hook(ctx):
    # If this raises, it's logged and other hooks continue
    raise ValueError("Something went wrong")

# Other hooks still execute
result = injector.trigger("artifact_created", context)
```

## Best Practices

1. **Keep hooks focused** - Each hook should do one thing
2. **Return empty results** - Return `TaskInjectionResult()` when no action needed
3. **Use factory methods** - Use `TaskInjectionContext.for_*()` for type safety
4. **Handle missing data** - Check for None artifacts, reviews, etc.
5. **Merge results carefully** - Consider blocking conflicts when merging
6. **Test hooks independently** - Unit test hooks with mock contexts

## Related Documentation

- [Artifact System](artifacts.md) - Managing artifacts
- [Review System](reviews.md) - Validating artifacts
- [Wizard Observability](observability.md) - Task tracking
