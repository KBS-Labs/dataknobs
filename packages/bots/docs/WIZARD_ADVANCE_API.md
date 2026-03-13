# Wizard Advance API

Non-conversational API for advancing wizards without DynaBot, LLM, or ConversationManager infrastructure.

## Overview

`WizardReasoning.advance()` operates the same FSM lifecycle as `generate()` but accepts structured data directly and returns a result object instead of an LLM response. This enables:

- Structured API endpoints for step-by-step data collection
- Custom UIs that manage their own state persistence
- Server-side wizard orchestration without LLM overhead

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `generate()` | Conversational | User message via ConversationManager | LLM response with metadata |
| `advance()` | Non-conversational | Structured `dict` + `WizardState` | `WizardAdvanceResult` |

## Quick Start

```python
from dataknobs_bots.reasoning import WizardReasoning, WizardState
from dataknobs_bots.reasoning import WizardConfigLoader

# Load wizard
loader = WizardConfigLoader()
wizard_fsm = loader.load("wizards/onboarding.yaml")
reasoning = WizardReasoning(wizard_fsm=wizard_fsm)

# Create initial state
state = WizardState(current_stage=reasoning.initial_stage)

# Advance with structured data
result = await reasoning.advance(
    user_input={"name": "Alice"},
    state=state,
)

print(result.stage_name)    # Next stage name
print(result.stage_prompt)  # Prompt to show user
print(result.completed)     # Whether wizard is done
```

## API Reference

### `WizardReasoning.advance()`

```python
async def advance(
    self,
    user_input: dict[str, Any],
    state: WizardState,
    *,
    navigation: str | None = None,
) -> WizardAdvanceResult:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_input` | `dict[str, Any]` | Structured data for the current stage. Merged into `state.data` before evaluating transitions. |
| `state` | `WizardState` | Current wizard state. Mutated in place and returned in the result. |
| `navigation` | `str \| None` | Optional navigation command: `"back"`, `"skip"`, or `"restart"`. |

**Returns:** `WizardAdvanceResult`

### `WizardAdvanceResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `state` | `WizardState` | Updated wizard state (caller should persist this). |
| `stage_name` | `str` | Name of the current stage after advance. |
| `stage_prompt` | `str` | Prompt text for the current stage. |
| `stage_schema` | `dict \| None` | JSON Schema for the current stage (if any). |
| `suggestions` | `list[str]` | Quick-reply suggestions for the current stage. |
| `can_skip` | `bool` | Whether the current stage can be skipped. |
| `can_go_back` | `bool` | Whether back navigation is allowed. |
| `completed` | `bool` | Whether the wizard has reached its end state. |
| `transitioned` | `bool` | Whether a stage transition occurred. |
| `from_stage` | `str \| None` | Stage before the advance (None if no transition). |
| `auto_advance_messages` | `list[str]` | Rendered templates from auto-advanced intermediate stages. Empty when no auto-advance occurred. |
| `metadata` | `dict` | Full wizard metadata dict for UI rendering. |

### `WizardReasoning.initial_stage`

Property returning the name of the wizard's start stage.

### `WizardReasoning.get_wizard_metadata(state)`

Build wizard metadata from state without advancing. Useful for initial page renders or status checks.

## Navigation

```python
# Go back to previous stage
result = await reasoning.advance({}, state, navigation="back")

# Skip current stage (must be skippable)
result = await reasoning.advance({}, state, navigation="skip")

# Restart wizard from beginning
result = await reasoning.advance({}, state, navigation="restart")
```

## State Persistence

The caller is responsible for persisting `WizardState` between calls. Use `to_dict()` / `from_dict()` for safe round-trip serialization:

```python
import json

# Save
state_dict = state.to_dict()
json_str = json.dumps(state_dict)

# Restore
state = WizardState.from_dict(json.loads(json_str))
```

## Hooks

Lifecycle hooks (`WizardHooks`) fire during `advance()`. The hooks that fire depend on the type of advance:

```python
from dataknobs_bots.reasoning import WizardHooks

hooks = WizardHooks()
hooks.on_exit(lambda stage, data: print(f"Left {stage}"))
hooks.on_enter(lambda stage, data: print(f"Entered {stage}"))
hooks.on_complete(lambda data: print("Wizard complete"))
hooks.on_restart(lambda data: print("Wizard restarted"))

reasoning = WizardReasoning(wizard_fsm=wizard_fsm, hooks=hooks)
```

### Hooks by navigation type

| Hook | Forward | Back | Skip | Restart |
|------|---------|------|------|---------|
| Exit | Yes | No | No | No |
| Enter | Yes (if transitioned) | Yes\* | Yes\* | No |
| Complete | Yes (if end stage) | No | Yes\* (if end stage) | No |
| Auto-advance | Yes (if transitioned) | No | Yes\* | No |
| Subflow pop | Yes (if transitioned) | No | Yes\* | No |
| Restart | No | No | No | Yes |

\* Requires `consistent_navigation_lifecycle=True` (the default).

**Design rationale:**

- **Forward** fires exit before attempting the transition and full post-transition lifecycle (enter, auto-advance, subflow pop) when a transition occurs. No hooks fire when the FSM stays at the same stage.
- **Back** fires only the enter hook — you are returning to a known previous stage, not completing the current one, so exit hooks do not apply. Auto-advance and subflow pop are not run because back navigation targets an explicit history entry.
- **Skip** runs the full post-transition lifecycle (matching forward) because skipping moves forward through the wizard, just without user-provided data.
- **Restart** fires only the restart hook via `_restart_cleanup()`. Enter/exit hooks do not fire because restart is a full state reset, not a stage-to-stage transition.

## Navigation Lifecycle Flag

The `consistent_navigation_lifecycle` parameter controls whether back and skip navigation fire lifecycle hooks (marked with \* in the table above).

| Value | Back behavior | Skip behavior |
|-------|--------------|---------------|
| `True` (default) | Fires enter hook on destination stage | Runs full post-transition lifecycle (subflow pop, auto-advance, enter/complete hooks) |
| `False` | No hooks (original behavior) | FSM step only, no lifecycle hooks (original behavior) |

```python
# New behavior (default): back/skip fire lifecycle hooks
reasoning = WizardReasoning(wizard_fsm=wizard_fsm, hooks=hooks)

# Original behavior: back/skip only perform FSM operation
reasoning = WizardReasoning(
    wizard_fsm=wizard_fsm,
    hooks=hooks,
    consistent_navigation_lifecycle=False,
)
```

Via configuration:

```yaml
reasoning:
  strategy: wizard
  wizard_config: wizards/onboarding.yaml
  consistent_navigation_lifecycle: false  # restore original behavior
```

## Example: REST API Endpoint

```python
from fastapi import FastAPI
from dataknobs_bots.reasoning import WizardReasoning, WizardState

app = FastAPI()
reasoning: WizardReasoning  # initialized at startup

@app.post("/wizard/advance")
async def advance_wizard(
    user_input: dict,
    state: dict,
    navigation: str | None = None,
):
    wizard_state = WizardState.from_dict(state)

    result = await reasoning.advance(
        user_input=user_input,
        state=wizard_state,
        navigation=navigation,
    )

    return {
        "stage_name": result.stage_name,
        "stage_prompt": result.stage_prompt,
        "stage_schema": result.stage_schema,
        "suggestions": result.suggestions,
        "can_skip": result.can_skip,
        "can_go_back": result.can_go_back,
        "completed": result.completed,
        "state": result.state.to_dict(),
    }
```
