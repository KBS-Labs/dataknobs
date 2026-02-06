# Wizard Subflows

Reusable, nestable wizard flows that can be invoked from within a parent wizard.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
  - [Transition Syntax](#transition-syntax)
  - [Subflow Block Fields](#subflow-block-fields)
- [Data Flow](#data-flow)
  - [data_mapping (Parent to Child)](#data_mapping-parent-to-child)
  - [result_mapping (Child to Parent)](#result_mapping-child-to-parent)
- [SubflowContext](#subflowcontext)
- [Subflow Definitions](#subflow-definitions)
  - [Inline Definitions](#inline-definitions)
  - [File-Based Definitions](#file-based-definitions)
- [Nested Subflows](#nested-subflows)
- [Example: Knowledge Base Acquisition Subflow](#example-knowledge-base-acquisition-subflow)

---

## Overview

Subflows allow a wizard to delegate a portion of its conversation to an independent wizard network and then resume where it left off. This is useful when:

- A multi-step data collection sequence is shared across several wizards (e.g., collecting user credentials, gathering document metadata).
- A wizard stage needs to branch into a detailed sub-conversation that has its own stages, validation, and transitions.
- You want to keep wizard configurations modular and maintainable.

When a subflow is triggered, the parent wizard's state is saved onto a stack, the subflow wizard takes over, and once the subflow reaches its end stage, control returns to the parent with results mapped back into the parent's data.

Because the state is maintained on a stack (`WizardState.subflow_stack`), subflows can be nested -- a subflow can itself push another subflow.

## Configuration

### Transition Syntax

A subflow is triggered by a transition whose `target` is the sentinel value `"_subflow"` (defined as `SUBFLOW_TARGET` in `wizard_loader.py`). The transition includes a `subflow:` block that specifies which subflow network to invoke and how data flows between parent and child.

```yaml
stages:
  - name: collect_info
    prompt: "Let me gather some details about your knowledge base."
    transitions:
      - target: "_subflow"
        condition: "data.get('needs_kb_details')"
        subflow:
          network: kb_acquisition
          return_stage: review
          data_mapping:
            project_name: kb_name
            user_id: owner_id
          result_mapping:
            kb_url: knowledge_base_url
            kb_status: ingestion_status
```

When the condition evaluates to true, the wizard pushes the `kb_acquisition` subflow. Once the subflow completes, the parent resumes at the `review` stage with the mapped results.

### Subflow Block Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `network` | `str` | Yes | Name of the subflow network to load |
| `return_stage` | `str` | No | Stage to transition to when the subflow completes. Defaults to the stage that pushed the subflow. |
| `data_mapping` | `dict[str, str]` | No | Maps parent field names to subflow field names (parent -> child). |
| `result_mapping` | `dict[str, str]` | No | Maps subflow field names back to parent field names (child -> parent). |

## Data Flow

Data mapping is directional and explicit. Only the fields you list in the mapping dictionaries are transferred; the rest stay isolated between parent and child.

### data_mapping (Parent to Child)

When a subflow is pushed, `data_mapping` controls which fields from the parent's `wizard_state.data` are copied into the subflow's initial data. The keys are parent field names; the values are the field names they become in the subflow.

```yaml
data_mapping:
  project_name: kb_name      # parent's "project_name" -> subflow's "kb_name"
  user_id: owner_id           # parent's "user_id" -> subflow's "owner_id"
```

Internally, `WizardReasoning._apply_data_mapping()` iterates the mapping and copies each matched field:

```python
def _apply_data_mapping(
    self,
    source_data: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    if not mapping:
        return {}
    result: dict[str, Any] = {}
    for parent_field, subflow_field in mapping.items():
        if parent_field in source_data:
            result[subflow_field] = source_data[parent_field]
    return result
```

If `data_mapping` is empty or omitted, the subflow starts with an empty data dict.

### result_mapping (Child to Parent)

When a subflow reaches its end stage and is popped, `result_mapping` controls which fields from the subflow's data are merged back into the parent's data. The keys are subflow field names; the values are the parent field names they map to.

```yaml
result_mapping:
  kb_url: knowledge_base_url  # subflow's "kb_url" -> parent's "knowledge_base_url"
  kb_status: ingestion_status # subflow's "kb_status" -> parent's "ingestion_status"
```

The parent's original data (captured at push time in `SubflowContext.parent_data`) is restored first, then the mapped results are merged on top via `dict.update()`. This means the subflow cannot accidentally overwrite parent fields that are not listed in `result_mapping`.

```python
# In _handle_subflow_pop:
parent_data = dict(subflow_context.parent_data)
result_data = self._apply_result_mapping(
    wizard_state.data, subflow_context.result_mapping
)
parent_data.update(result_data)
```

## SubflowContext

When a subflow is pushed, a `SubflowContext` dataclass is created and appended to `WizardState.subflow_stack`. This captures everything needed to restore the parent's state when the subflow completes.

```python
@dataclass
class SubflowContext:
    parent_stage: str              # Stage in parent flow before push
    parent_data: dict[str, Any]    # Copy of wizard data at push time
    parent_history: list[str]      # Copy of stage history at push time
    return_stage: str              # Stage to transition to on pop
    result_mapping: dict[str, str] # Subflow field -> parent field mapping
    subflow_network: str           # Name of the subflow network
    push_timestamp: float          # When the subflow was pushed (time.time())
```

`SubflowContext` supports serialization via `to_dict()` and `from_dict()` class methods, so the full subflow stack can be persisted and restored across conversation turns.

`WizardState` exposes three convenience properties for inspecting subflow status:

| Property | Return Type | Description |
|----------|-------------|-------------|
| `is_in_subflow` | `bool` | `True` if `subflow_stack` is not empty |
| `subflow_depth` | `int` | Number of subflows on the stack (0 = main flow) |
| `current_subflow` | `SubflowContext \| None` | Top of the stack, or `None` if in main flow |

## Subflow Definitions

Subflows are self-contained wizard configurations. They have their own `stages`, `transitions`, start stage, and end stage. The `WizardConfigLoader` supports three locations for subflow definitions.

### Inline Definitions

Define the subflow directly in the parent wizard config under a top-level `subflows:` key:

```yaml
name: onboarding-wizard
version: "1.0"

stages:
  - name: welcome
    is_start: true
    prompt: "Welcome! Let's set up your project."
    transitions:
      - target: "_subflow"
        condition: "data.get('intent') == 'import'"
        subflow:
          network: kb_acquisition
          return_stage: review
          result_mapping:
            collected_url: kb_url

  - name: review
    prompt: "Great, your KB is at {{ data.kb_url }}. Ready to continue?"
    transitions:
      - target: complete
        condition: "data.get('confirmed')"

  - name: complete
    is_end: true
    prompt: "All set!"

subflows:
  kb_acquisition:
    name: kb-acquisition
    stages:
      - name: ask_url
        is_start: true
        prompt: "What is the URL of the knowledge base?"
        schema:
          type: object
          properties:
            collected_url:
              type: string
              format: uri
        transitions:
          - target: confirm_url
            condition: "data.get('collected_url')"

      - name: confirm_url
        is_end: true
        prompt: "Got it: {{ data.collected_url }}"
```

Inline definitions are loaded by `_load_single_subflow()` when it finds the subflow name as a key in `wizard_config["subflows"]`.

### File-Based Definitions

For larger subflows, define them in separate YAML files. The loader searches two paths relative to the parent config file:

1. **Adjacent file**: `<name>.yaml` next to the main config file.
2. **Subdirectory**: `subflows/<name>.yaml` under the main config directory.

```
project/
  wizard.yaml               # Main wizard config
  kb_acquisition.yaml        # Option 1: adjacent file
  subflows/
    kb_acquisition.yaml      # Option 2: subdirectory
```

The loader tries each location in order and uses the first match. If the subflow is defined both inline and as a file, the inline definition takes precedence.

File-based subflows use the exact same YAML structure as any wizard config:

```yaml
# kb_acquisition.yaml
name: kb-acquisition
version: "1.0"

stages:
  - name: ask_url
    is_start: true
    prompt: "What is the URL of the knowledge base?"
    schema:
      type: object
      properties:
        collected_url:
          type: string
          format: uri
    transitions:
      - target: validate_url
        condition: "data.get('collected_url')"

  - name: validate_url
    prompt: "Checking access to {{ data.collected_url }}..."
    tools:
      - url_validator
    transitions:
      - target: confirm
        condition: "data.get('url_valid')"
      - target: ask_url

  - name: confirm
    is_end: true
    prompt: "Knowledge base verified and ready."
```

All loaded subflows are stored in `WizardFSM._subflow_registry`, a dict mapping subflow names to `WizardFSM` instances.

## Nested Subflows

Because `WizardState.subflow_stack` is a list, subflows can trigger other subflows. When a nested subflow completes, control returns to its immediate parent (not the root wizard).

```yaml
# Main wizard pushes "setup_project" subflow
# "setup_project" subflow pushes "collect_credentials" subflow

# Stack during deepest nesting:
# [0] SubflowContext(subflow_network="setup_project", ...)
# [1] SubflowContext(subflow_network="collect_credentials", ...)
#
# subflow_depth = 2
```

On pop, the `_handle_subflow_pop()` method checks whether additional subflows remain on the stack. If so, it restores the next subflow's FSM as the active FSM rather than the main flow:

```python
# Switch back to parent FSM (or next subflow if nested)
if wizard_state.subflow_stack:
    parent_subflow = wizard_state.subflow_stack[-1].subflow_network
    self._active_subflow_fsm = self._fsm.get_subflow(parent_subflow)
else:
    self._active_subflow_fsm = None
```

The `_get_active_fsm()` method always returns the correct FSM for the current nesting level:

```python
def _get_active_fsm(self) -> WizardFSM:
    return self._active_subflow_fsm if self._active_subflow_fsm else self._fsm
```

## Example: Knowledge Base Acquisition Subflow

This end-to-end example demonstrates a bot-building wizard that delegates knowledge base setup to a reusable subflow.

### Parent Wizard (`bot_builder.yaml`)

```yaml
name: bot-builder
version: "1.0"

stages:
  - name: welcome
    is_start: true
    prompt: "What kind of bot would you like to build?"
    schema:
      type: object
      properties:
        bot_type:
          type: string
          enum: [qa, tutor, companion]
    transitions:
      - target: "_subflow"
        condition: "data.get('bot_type') == 'qa'"
        subflow:
          network: kb_acquisition
          return_stage: configure_personality
          data_mapping:
            bot_type: source_type
          result_mapping:
            kb_url: knowledge_base_url
            document_count: kb_doc_count
      - target: configure_personality
        condition: "data.get('bot_type')"

  - name: configure_personality
    prompt: "How should your bot communicate?"
    schema:
      type: object
      properties:
        tone:
          type: string
          enum: [formal, casual, friendly]
    transitions:
      - target: complete
        condition: "data.get('tone')"

  - name: complete
    is_end: true
    prompt: >
      Your {{ data.bot_type }} bot is ready!
      {% if data.knowledge_base_url %}
      Knowledge base: {{ data.knowledge_base_url }}
      ({{ data.kb_doc_count }} documents indexed)
      {% endif %}
      Tone: {{ data.tone }}
```

### Subflow (`subflows/kb_acquisition.yaml`)

```yaml
name: kb-acquisition
version: "1.0"

stages:
  - name: ask_source
    is_start: true
    prompt: "Where is your knowledge base? Provide a URL or upload path."
    schema:
      type: object
      properties:
        kb_url:
          type: string
    transitions:
      - target: ingest
        condition: "data.get('kb_url')"

  - name: ingest
    prompt: "Indexing {{ data.kb_url }}... This may take a moment."
    tools:
      - kb_indexer
    schema:
      type: object
      properties:
        document_count:
          type: integer
    transitions:
      - target: done
        condition: "data.get('document_count', 0) > 0"
      - target: ask_source

  - name: done
    is_end: true
    prompt: "Indexed {{ data.document_count }} documents from {{ data.kb_url }}."
```

### Conversation Flow

1. User says "qa" at `welcome` stage.
2. Condition matches the subflow transition. `SubflowContext` is created and pushed:
   - `parent_stage = "welcome"`
   - `parent_data = {"bot_type": "qa"}`
   - `return_stage = "configure_personality"`
   - `result_mapping = {"kb_url": "knowledge_base_url", "document_count": "kb_doc_count"}`
3. `data_mapping` copies `bot_type` as `source_type` into the subflow's initial data.
4. Subflow runs through `ask_source` -> `ingest` -> `done`.
5. At `done` (end stage), `_should_pop_subflow()` returns `True`.
6. `_handle_subflow_pop()` restores parent data and applies `result_mapping`:
   - Parent data `{"bot_type": "qa"}` is restored.
   - Subflow's `kb_url` is mapped to `knowledge_base_url`.
   - Subflow's `document_count` is mapped to `kb_doc_count`.
7. Parent wizard resumes at `configure_personality` with the enriched data.
