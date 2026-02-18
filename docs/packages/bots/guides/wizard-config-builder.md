# Wizard Config Builder

The `WizardConfigBuilder` provides a fluent, programmatic API for creating wizard configurations. It complements the YAML-based approach by enabling code-driven wizard generation with built-in validation, serialization, and full roundtrip compatibility with `WizardConfigLoader`.

## Overview

| Component | Class | Purpose |
|-----------|-------|---------|
| **Builder** | `WizardConfigBuilder` | Fluent API for constructing wizard configs |
| **Config** | `WizardConfig` | Immutable, validated wizard configuration |
| **Stage** | `StageConfig` | Configuration for a single wizard stage |
| **Transition** | `TransitionConfig` | Configuration for a stage-to-stage transition |
| **Intent Detection** | `IntentDetectionConfig` | Intent detection settings for conversation stages |
| **Context Generation** | `ContextGenerationConfig` | LLM-generated context variable settings |

## Quick Start

### Minimal Wizard

```python
from dataknobs_bots.config import WizardConfigBuilder

wizard = (
    WizardConfigBuilder("onboarding")
    .add_structured_stage("welcome", "What is your name?", is_start=True)
    .add_end_stage("done", "All set!")
    .add_transition("welcome", "done")
    .build()
)

wizard.to_file("configs/wizards/onboarding.yaml")
```

### Conversation-Start Pattern

For the common single-conversation-stage pattern:

```python
from dataknobs_bots.config import WizardConfigBuilder

wizard = (
    WizardConfigBuilder.conversation_start(
        name="assistant",
        prompt="You are a helpful assistant.",
        tools=["knowledge_search", "web_search"],
        tool_reasoning="react",
        max_tool_iterations=5,
    )
    .build()
)
```

### Multi-Stage Wizard

```python
from dataknobs_bots.config import WizardConfigBuilder

wizard = (
    WizardConfigBuilder("survey")
    .set_version("2.0.0")
    .set_description("Customer feedback survey")
    .set_settings(
        tool_reasoning="react",
        max_tool_iterations=3,
        auto_advance_filled_stages=True,
    )
    .add_conversation_stage(
        name="chat",
        prompt="Welcome! Let's collect your feedback.",
        tools=["knowledge_search"],
        is_start=True,
        suggestions=["Tell me about your experience", "I have a complaint"],
        intent_detection={
            "method": "keyword",
            "intents": [
                {"id": "start_survey", "keywords": ["survey", "feedback"]},
            ],
        },
    )
    .add_structured_stage(
        name="rating",
        prompt="How would you rate our service? (1-5)",
        schema={"type": "object", "properties": {"score": {"type": "integer"}}},
        help_text="Enter a number from 1 to 5.",
        suggestions=["1", "2", "3", "4", "5"],
    )
    .add_end_stage("thanks", "Thank you for your feedback!")
    .add_transition("chat", "rating", condition="data.get('_intent') == 'start_survey'")
    .add_transition("rating", "thanks")
    .build()
)
```

## Builder API

### Constructor

```python
WizardConfigBuilder(name: str)
```

Creates a builder with defaults: version `"1.0"`, empty description, no settings.

### Metadata Methods

| Method | Description |
|--------|-------------|
| `set_version(version)` | Set the wizard version string |
| `set_description(description)` | Set a human-readable description |

### Settings

```python
builder.set_settings(
    tool_reasoning="react",
    max_tool_iterations=3,
    auto_advance_filled_stages=True,
    extraction_scope="global",
    conflict_strategy="latest",
    timeout_seconds=300,
)
```

Settings are passed through to the wizard runtime. Common keys include `tool_reasoning`, `max_tool_iterations`, `auto_advance_filled_stages`, `extraction_scope`, `conflict_strategy`, and `timeout_seconds`.

### Stage Methods

#### `add_conversation_stage()`

Adds a `mode: conversation` stage for free-form chat with optional tool access and intent detection.

```python
builder.add_conversation_stage(
    name="chat",
    prompt="Let's discuss your project.",
    tools=["knowledge_search"],
    is_start=True,
    suggestions=["Tell me more", "What options do I have?"],
    intent_detection={
        "method": "keyword",
        "intents": [
            {"id": "switch_topic", "keywords": ["change", "different"]},
        ],
    },
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Stage identifier |
| `prompt` | `str` | User-facing prompt |
| `tools` | `list[str] \| None` | Available tool names |
| `is_start` | `bool` | Whether this is the start stage |
| `suggestions` | `list[str] \| None` | Quick-reply suggestions |
| `intent_detection` | `dict \| None` | Intent detection config (method + intents) |
| `**kwargs` | `Any` | Additional `StageConfig` fields |

#### `add_structured_stage()`

Adds a data-collection stage with optional JSON Schema validation.

```python
builder.add_structured_stage(
    name="collect_name",
    prompt="What is your full name?",
    schema={"type": "object", "properties": {"name": {"type": "string"}}},
    is_start=True,
    can_skip=True,
    skip_default="Anonymous",
    reasoning="react",
    max_iterations=5,
    response_template="Got it, {name}!",
    help_text="Enter your first and last name.",
    context_generation={"variables": {"greeting": "Generate a greeting for {name}"}},
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Stage identifier |
| `prompt` | `str` | User-facing prompt |
| `schema` | `dict \| None` | JSON Schema for input validation |
| `tools` | `list[str] \| None` | Available tool names |
| `is_start` | `bool` | Whether this is the start stage |
| `is_end` | `bool` | Whether this is an end stage |
| `can_skip` | `bool` | Whether the user can skip |
| `skip_default` | `Any` | Default value when skipped |
| `suggestions` | `list[str] \| None` | Quick-reply suggestions |
| `response_template` | `str \| None` | Template-driven response (bypasses LLM) |
| `help_text` | `str \| None` | Help message |
| `reasoning` | `str \| None` | Reasoning mode: `"single"` or `"react"` |
| `max_iterations` | `int \| None` | Max iterations for ReAct reasoning |
| `context_generation` | `dict \| None` | LLM context generation config |
| `**kwargs` | `Any` | Additional `StageConfig` fields |

#### `add_end_stage()`

Adds a terminal stage (sets `is_end=True` automatically).

```python
builder.add_end_stage("done", "Thank you! Your session is complete.")
```

#### `add_stage()`

Adds a pre-built `StageConfig` directly for advanced use cases.

```python
from dataknobs_bots.config import StageConfig

stage = StageConfig(
    name="custom",
    prompt="Custom stage with full control.",
    is_start=True,
    llm_assist=True,
    llm_assist_prompt="Help the user with their query.",
)
builder.add_stage(stage)
```

### Transitions

```python
builder.add_transition(
    from_stage="chat",
    to_stage="quiz",
    condition="data.get('_intent') == 'quiz'",
    transform="transform_chat_to_quiz",
    priority=10,
    derive={"quiz_topic": {"from_field": "selected_topic"}},
    metadata={"description": "User wants to start a quiz"},
)
```

Transitions are stored separately and attached to their source stage at build time. This means stages and transitions can be added in any order.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `from_stage` | `str` | Source stage name |
| `to_stage` | `str` | Target stage name |
| `condition` | `str \| None` | Python expression evaluated with `data` in scope |
| `transform` | `str \| list[str] \| None` | Transform function name(s) |
| `priority` | `int \| None` | Evaluation priority |
| `derive` | `dict \| None` | Data derivation rules |
| `metadata` | `dict \| None` | Custom metadata |

### Intent Detection

Intent detection can be configured inline on conversation stages or added separately:

```python
# Method 1: Inline (via add_conversation_stage)
builder.add_conversation_stage(
    name="chat",
    prompt="...",
    intent_detection={
        "method": "keyword",
        "intents": [{"id": "quiz", "keywords": ["quiz", "test"]}],
    },
)

# Method 2: Separate (via add_intent_detection)
builder.add_intent_detection(
    stage="chat",
    method="llm",
    intents=[{"id": "quiz", "description": "User wants to take a quiz"}],
)
```

### Global Tasks

```python
builder.add_global_task(
    task_id="collect_name",
    description="Get the user's name",
    required=True,
    completed_by="field",
    field_name="user_name",
)

builder.add_global_task(
    task_id="search_kb",
    description="Search the knowledge base",
    required=False,
    depends_on=["collect_name"],
    completed_by="tool",
    tool_name="knowledge_search",
)
```

## Validation

The builder validates the configuration at build time and provides both errors and warnings.

### Errors (block build)

| Condition | Error |
|-----------|-------|
| No stages | "Wizard must have at least one stage" |
| No start stage | "Wizard must have exactly one start stage" |
| Multiple start stages | "Wizard has multiple start stages: [...]" |
| Duplicate stage names | "Duplicate stage name: '...'" |
| Transition to unknown stage | "Stage '...' has transition to unknown stage '...'" |
| Transition from unknown stage | "Transition from unknown stage '...' to '...'" |
| Intent detection on unknown stage | "Intent detection references unknown stage '...'" |
| Invalid reasoning value | "Stage '...' has invalid reasoning value '...'" |
| Invalid mode value | "Stage '...' has invalid mode value '...'" |

### Warnings (logged, don't block build)

| Condition | Warning |
|-----------|---------|
| `max_iterations` without `reasoning` | "Stage '...' sets max_iterations but has no reasoning mode" |
| End stage with transitions | "End stage '...' has transitions that will never be followed" |
| Unreachable stages | "Stage '...' is not reachable from the start stage" |

### Pre-Build Validation

Use `validate()` to check without building:

```python
result = builder.validate()
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
for warning in result.warnings:
    print(f"Warning: {warning}")
```

## Serialization

### To Dict

```python
config = builder.build()
config_dict = config.to_dict()
# config_dict is compatible with WizardConfigLoader.load_from_dict()
```

### To YAML

```python
yaml_str = config.to_yaml()
```

### To File

```python
config.to_file("configs/wizards/my-wizard.yaml")
# Creates parent directories if they don't exist
```

## Roundtrip

Existing wizard YAML files can be loaded into a builder, modified, and re-exported:

```python
# Load from file
builder = WizardConfigBuilder.from_file("configs/wizards/existing.yaml")

# Modify
builder.add_structured_stage("new_stage", "New question?")
builder.add_transition("existing_stage", "new_stage")

# Rebuild
wizard = builder.build()
wizard.to_file("configs/wizards/updated.yaml")
```

Or from a dict:

```python
builder = WizardConfigBuilder.from_dict(existing_config_dict)
```

## Integration with DynaBotConfigBuilder

Use `set_reasoning_wizard()` on `DynaBotConfigBuilder` to combine bot and wizard configs:

```python
from dataknobs_bots.config import DynaBotConfigBuilder, WizardConfigBuilder

# Build the wizard config
wizard = (
    WizardConfigBuilder("assistant")
    .add_conversation_stage("chat", "How can I help?", is_start=True)
    .build()
)

# Write it to disk
wizard.to_file("configs/wizards/assistant.yaml")

# Reference it from the bot config
bot_config = (
    DynaBotConfigBuilder()
    .set_llm("ollama", model="llama3.2")
    .set_conversation_storage("memory")
    .set_reasoning_wizard(wizard)  # Uses wizard.name as the config path
    .build()
)
```

You can also pass a file path string directly:

```python
bot_config = (
    DynaBotConfigBuilder()
    .set_llm("ollama", model="llama3.2")
    .set_conversation_storage("memory")
    .set_reasoning_wizard("configs/wizards/assistant.yaml")
    .build()
)
```

## StageConfig Reference

All fields available on `StageConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | (required) | Stage identifier |
| `prompt` | `str` | (required) | User-facing prompt |
| `is_start` | `bool` | `False` | Whether this is the start stage |
| `is_end` | `bool` | `False` | Whether this is an end stage |
| `can_skip` | `bool` | `False` | Whether the user can skip this stage |
| `skip_default` | `Any` | `None` | Default value if skipped |
| `can_go_back` | `bool` | `True` | Whether the user can go back |
| `auto_advance` | `bool` | `False` | Auto-advance when data is collected |
| `label` | `str \| None` | `None` | Display label |
| `suggestions` | `tuple[str, ...]` | `()` | Quick-reply suggestions |
| `help_text` | `str \| None` | `None` | Help message |
| `schema` | `dict \| None` | `None` | JSON Schema for validation |
| `transitions` | `tuple[TransitionConfig, ...]` | `()` | Stage transitions |
| `tools` | `tuple[str, ...]` | `()` | Available tool names |
| `reasoning` | `str \| None` | `None` | `"single"` or `"react"` |
| `max_iterations` | `int \| None` | `None` | Max ReAct iterations |
| `extraction_model` | `str \| None` | `None` | Model for extraction |
| `response_template` | `str \| None` | `None` | Template-driven response |
| `llm_assist` | `bool` | `False` | Enable LLM-assisted responses |
| `llm_assist_prompt` | `str \| None` | `None` | Custom LLM assist prompt |
| `context_generation` | `ContextGenerationConfig \| None` | `None` | LLM context generation |
| `mode` | `str \| None` | `None` | `"conversation"` for chat stages |
| `intent_detection` | `IntentDetectionConfig \| None` | `None` | Intent detection settings |
| `tasks` | `tuple[dict, ...]` | `()` | Stage-level tasks |

## TransitionConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target` | `str` | (required) | Target stage name |
| `condition` | `str \| None` | `None` | Python expression with `data` in scope |
| `transform` | `str \| list[str] \| None` | `None` | Transform function name(s) |
| `priority` | `int \| None` | `None` | Evaluation priority |
| `derive` | `dict \| None` | `None` | Data derivation rules |
| `metadata` | `dict \| None` | `None` | Custom metadata |
| `subflow` | `dict \| None` | `None` | Subflow configuration |
