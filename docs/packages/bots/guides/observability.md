# Wizard Observability

This guide covers the observability features for wizard flows, including task tracking, state snapshots, and transition auditing.

## Overview

Wizard observability provides:

- **Task Tracking** - Granular progress tracking within and across wizard stages
- **Transition Audit Trail** - Complete history of state transitions with timing and context
- **State Snapshots** - Read-only snapshots for UI display and debugging
- **Statistics** - Aggregated metrics for monitoring and analysis

All observability types are in `dataknobs_bots.reasoning.observability`.

## Task Tracking

Tasks provide granular progress tracking beyond stage-level progress. A stage may have multiple tasks (e.g., collect bot name, collect description), and global tasks can span stages (e.g., validate config, save config).

### WizardTask

A `WizardTask` represents a single trackable action within the wizard:

```python
from dataknobs_bots.reasoning.observability import WizardTask

task = WizardTask(
    id="collect_bot_name",
    description="Collect bot name",
    status="pending",           # pending, in_progress, completed, skipped
    stage="configure_identity", # None for global tasks
    required=True,
    depends_on=[],              # Task IDs that must complete first
    completed_by="field_extraction",  # What triggers completion
    field_name="bot_name",      # For field_extraction trigger
)

# Check task state
if task.is_pending:
    print(f"Task {task.id} is waiting")
if task.is_complete:
    print(f"Completed at {task.completed_at}")
if task.is_global:
    print("This is a global task")
```

#### Task Completion Triggers

Tasks can be completed by different triggers:

| Trigger | Description | Required Field |
|---------|-------------|----------------|
| `field_extraction` | Completed when a field is extracted | `field_name` |
| `tool_result` | Completed when a tool succeeds | `tool_name` |
| `stage_exit` | Completed when leaving a stage | - |
| `manual` | Completed programmatically | - |

### WizardTaskList

`WizardTaskList` manages a collection of tasks with dependency tracking:

```python
from dataknobs_bots.reasoning.observability import WizardTask, WizardTaskList

# Create task list with dependencies
task_list = WizardTaskList(tasks=[
    WizardTask(id="validate", description="Validate config", depends_on=[]),
    WizardTask(id="save", description="Save config", depends_on=["validate"]),
])

# Query tasks
pending = task_list.get_pending_tasks()
completed = task_list.get_completed_tasks()
available = task_list.get_available_tasks()  # Pending with deps met

# Get tasks by scope
stage_tasks = task_list.get_tasks_for_stage("configure_identity")
global_tasks = task_list.get_global_tasks()

# Complete a task (checks dependencies)
if task_list.complete_task("validate"):
    print("Validate completed!")
    # Now "save" is available
    print(task_list.get_available_tasks())  # [save]

# Skip a task
task_list.skip_task("validate")

# Calculate progress (based on required tasks)
progress = task_list.calculate_progress()  # 0.0 to 100.0
print(f"Progress: {progress}%")
```

### Defining Tasks in Configuration

Tasks can be defined in wizard YAML configuration:

```yaml
stages:
  configure_identity:
    prompt: "Let's set up your bot's identity..."
    schema:
      type: object
      properties:
        bot_name: { type: string }
        description: { type: string }
    # Task definitions for this stage
    tasks:
      - id: collect_bot_name
        description: "Collect bot name"
        completed_by: field_extraction
        field_name: bot_name
        required: true
      - id: collect_description
        description: "Collect bot description"
        completed_by: field_extraction
        field_name: description
        required: false

# Global tasks (not tied to a specific stage)
global_tasks:
  - id: preview_config
    description: "Preview the configuration"
    completed_by: tool_result
    tool_name: preview_config
    required: false
  - id: validate_config
    description: "Validate the configuration"
    completed_by: tool_result
    tool_name: validate_config
    required: true
  - id: save_config
    description: "Save the configuration"
    completed_by: tool_result
    tool_name: save_config
    required: true
    depends_on: [validate_config]  # Must validate first
```

## Transition Tracking

### TransitionRecord

A `TransitionRecord` captures a single state transition with full context:

```python
from dataknobs_bots.reasoning.observability import TransitionRecord, create_transition_record

# Create using factory function (auto-sets timestamp)
record = create_transition_record(
    from_stage="welcome",
    to_stage="configure",
    trigger="user_input",
    duration_in_stage_ms=5000.0,
    data_snapshot={"intent": "create"},
    user_input="I want to create a math tutor bot",
    condition_evaluated="data.get('intent')",
    condition_result=True,
)

# Access fields
print(f"Transition: {record.from_stage} -> {record.to_stage}")
print(f"Trigger: {record.trigger}")
print(f"Duration: {record.duration_in_stage_ms}ms")
```

#### Trigger Types

| Trigger | Description |
|---------|-------------|
| `user_input` | User message triggered the transition |
| `navigation_back` | User navigated backward |
| `navigation_skip` | User skipped the stage |
| `restart` | Wizard was restarted |
| `auto` | Automatic transition (e.g., condition-based) |

### TransitionTracker

`TransitionTracker` maintains a bounded history of transitions with query and statistics capabilities:

```python
from dataknobs_bots.reasoning.observability import (
    TransitionTracker,
    TransitionHistoryQuery,
    create_transition_record,
)

# Create tracker with max history
tracker = TransitionTracker(max_history=100)

# Record transitions
tracker.record(create_transition_record(
    from_stage="welcome",
    to_stage="configure",
    trigger="user_input",
    duration_in_stage_ms=3000.0,
))

tracker.record(create_transition_record(
    from_stage="configure",
    to_stage="review",
    trigger="user_input",
    duration_in_stage_ms=15000.0,
))

# Query history
all_transitions = tracker.query()  # All records

# Query with filters
query = TransitionHistoryQuery(
    trigger="user_input",
    since=time.time() - 3600,  # Last hour
    limit=10,
)
recent_user_transitions = tracker.query(query)

# Get statistics
stats = tracker.get_stats()
print(f"Total transitions: {stats.total_transitions}")
print(f"Unique paths: {stats.unique_paths}")
print(f"Avg duration: {stats.avg_duration_per_stage_ms}ms")
print(f"Backtracks: {stats.backtrack_count}")
print(f"Restarts: {stats.restart_count}")
print(f"Most common trigger: {stats.most_common_trigger}")
```

### TransitionHistoryQuery

Filter transitions with flexible query parameters:

```python
from dataknobs_bots.reasoning.observability import TransitionHistoryQuery

query = TransitionHistoryQuery(
    from_stage="welcome",     # Filter by source stage
    to_stage="configure",     # Filter by target stage
    trigger="user_input",     # Filter by trigger type
    since=1700000000.0,       # After this timestamp
    until=1700100000.0,       # Before this timestamp
    limit=50,                 # Max records to return
)
```

## State Snapshots

### WizardStateSnapshot

`WizardStateSnapshot` provides a complete read-only view of wizard state, useful for UI rendering and debugging:

```python
from dataknobs_bots.reasoning.observability import WizardStateSnapshot

# Snapshots are typically created by WizardReasoning.get_state_snapshot()
# but can be created directly or deserialized:
snapshot = WizardStateSnapshot(
    current_stage="configure_identity",
    data={"bot_name": "MathHelper"},
    history=["welcome", "configure_identity"],
    transitions=[transition_record],
    completed=False,
    # Task tracking
    tasks=[{"id": "collect_name", "status": "completed"}, ...],
    pending_tasks=3,
    completed_tasks=2,
    total_tasks=5,
    available_task_ids=["collect_description"],
    task_progress_percent=40.0,
    # Stage context
    stage_index=1,
    total_stages=4,
    can_skip=True,
    can_go_back=True,
    suggestions=["Create a math tutor", "Build a quiz bot"],
)

# Query tasks from snapshot
task = snapshot.get_task("collect_name")
stage_tasks = snapshot.get_tasks_for_stage("configure_identity")
global_tasks = snapshot.get_global_tasks()

if snapshot.is_task_available("save_config"):
    print("Save is available!")

# Get latest transition
latest = snapshot.get_latest_transition()
if latest:
    print(f"Last: {latest['from_stage']} -> {latest['to_stage']}")

# Serialize for storage or API response
data = snapshot.to_dict()

# Deserialize
restored = WizardStateSnapshot.from_dict(data)
```

### Using Snapshots for UI

Snapshots are designed for driving UI components:

```python
# In your web application
snapshot = reasoning.get_state_snapshot(manager)

# Display stage progress
progress_bar.set_value(snapshot.stage_index / snapshot.total_stages)
stage_label.set_text(f"Stage {snapshot.stage_index + 1} of {snapshot.total_stages}")

# Display task checklist
for task in snapshot.tasks:
    status_icon = "✅" if task["status"] == "completed" else "⬜"
    if task["status"] == "skipped":
        status_icon = "⏭️"
    task_list.add_item(f"{status_icon} {task['description']}")

# Show available actions
if snapshot.can_go_back:
    show_back_button()
if snapshot.can_skip:
    show_skip_button()

# Enable/disable action buttons based on task availability
save_button.enabled = snapshot.is_task_available("save_config")

# Display suggestions
for suggestion in snapshot.suggestions:
    quick_reply_buttons.add(suggestion)
```

## Integration with WizardReasoning

The observability features integrate with `WizardReasoning`:

```python
from dataknobs_bots.reasoning import WizardReasoning

# Get current state snapshot
snapshot = reasoning.get_state_snapshot(manager)

# Access from conversation metadata (static method)
snapshot = WizardReasoning.snapshot_from_metadata(
    manager.metadata,
    stage_definitions=wizard_config.get("stages"),
)

# Tasks are automatically completed when:
# - Fields are extracted (completed_by: field_extraction)
# - Tools succeed (completed_by: tool_result)
# - Stages are exited (completed_by: stage_exit)
```

## Conversion Utilities

Convert between wizard and FSM observability types:

```python
from dataknobs_bots.reasoning.observability import (
    TransitionRecord,
    transition_record_to_execution_record,
    execution_record_to_transition_record,
    transition_stats_to_execution_stats,
)
from dataknobs_fsm.observability import ExecutionRecord

# Convert wizard record to FSM record
wizard_record = TransitionRecord(...)
fsm_record = transition_record_to_execution_record(wizard_record)

# Convert FSM record to wizard record
wizard_record = execution_record_to_transition_record(
    fsm_record,
    user_input="optional user message",
)

# Convert stats
fsm_stats = transition_stats_to_execution_stats(wizard_stats)
```

## Best Practices

### 1. Use Tasks for Granular Progress

Define tasks for each meaningful action in your wizard:

```yaml
# Instead of relying only on stage progress:
stages:
  configure:
    tasks:
      - id: collect_name
        completed_by: field_extraction
        field_name: name
      - id: collect_email
        completed_by: field_extraction
        field_name: email
      - id: verify_email
        completed_by: tool_result
        tool_name: send_verification
```

### 2. Use Dependencies for Workflow Control

Express task ordering through dependencies:

```yaml
global_tasks:
  - id: validate
    required: true
  - id: preview
    depends_on: [validate]
    required: false
  - id: save
    depends_on: [validate]  # Not preview - it's optional
    required: true
```

### 3. Query Statistics for Monitoring

Use `TransitionStats` for monitoring wizard performance:

```python
stats = tracker.get_stats()

# Monitor backtracking (may indicate confusing UX)
if stats.backtrack_count > stats.total_transitions * 0.3:
    log.warning("High backtrack rate - review wizard flow")

# Monitor average time per stage
if stats.avg_duration_per_stage_ms > 60000:  # 1 minute
    log.info("Users spending significant time on stages")
```

### 4. Serialize Snapshots for APIs

Expose snapshots through your API for frontend state:

```python
@app.get("/wizard/state")
async def get_wizard_state(conversation_id: str):
    manager = await get_manager(conversation_id)
    snapshot = reasoning.get_state_snapshot(manager)
    return snapshot.to_dict()
```

## API Reference

### Type Aliases

```python
TaskStatus = Literal["pending", "in_progress", "completed", "skipped"]
TaskCompletionTrigger = Literal["field_extraction", "tool_result", "stage_exit", "manual"]
```

### Classes

| Class | Description |
|-------|-------------|
| `WizardTask` | Single trackable task within wizard flow |
| `WizardTaskList` | Collection of tasks with dependency tracking |
| `TransitionRecord` | Record of a single state transition |
| `TransitionHistoryQuery` | Query parameters for filtering transitions |
| `TransitionStats` | Aggregated transition statistics |
| `TransitionTracker` | Manages transition history with queries |
| `WizardStateSnapshot` | Complete read-only wizard state snapshot |

### Factory Functions

| Function | Description |
|----------|-------------|
| `create_transition_record()` | Create TransitionRecord with auto-timestamp |

### Conversion Utilities

| Function | Description |
|----------|-------------|
| `transition_record_to_execution_record()` | Convert wizard to FSM record |
| `execution_record_to_transition_record()` | Convert FSM to wizard record |
| `transition_stats_to_execution_stats()` | Convert wizard to FSM stats |

## See Also

- [Configuration Reference](configuration.md) - Task definition in wizard config
- [User Guide](user-guide.md) - Wizard reasoning tutorial
- [Architecture](architecture.md) - System design overview
