# ConfigBot Toolkit

The ConfigBot toolkit provides reusable infrastructure for building wizard-driven bot configuration experiences. It extracts the generic DynaBot config-building logic into composable components that any DynaBot consumer can assemble into their own ConfigBot.

## Overview

The toolkit provides five layers:

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Schema** | `DynaBotConfigSchema`, `ComponentSchema` | Queryable registry of valid config options |
| **Validation** | `ConfigValidator`, `ValidationResult` | Pluggable validation pipeline |
| **Templates** | `ConfigTemplate`, `ConfigTemplateRegistry`, `TemplateVariable` | Template loading, variable substitution, tag-based filtering |
| **Builder** | `DynaBotConfigBuilder` | Fluent builder for DynaBot configs |
| **Drafts** | `ConfigDraftManager`, `DraftMetadata` | File-based draft lifecycle management |
| **Tools** | `ListTemplatesTool`, `GetTemplateDetailsTool`, `PreviewConfigTool`, `ValidateConfigTool`, `SaveConfigTool` | LLM-callable tools for wizard flows |

## Quick Start

### Building a Config from Scratch

```python
from dataknobs_bots.config import DynaBotConfigBuilder

config = (
    DynaBotConfigBuilder()
    .set_llm("ollama", model="llama3.2", temperature=0.7)
    .set_conversation_storage("memory")
    .set_system_prompt(content="You are a helpful assistant.")
    .set_memory("buffer", max_messages=50)
    .build()
)
# config is compatible with DynaBot.from_config()
```

### Using Templates

```python
from pathlib import Path
from dataknobs_bots.config import ConfigTemplateRegistry, DynaBotConfigBuilder

registry = ConfigTemplateRegistry()
registry.load_from_directory(Path("configs/templates"))

# Apply template with variables
builder = DynaBotConfigBuilder().from_template(
    registry.get("basic_assistant"),
    {"bot_name": "Helper", "temperature": 0.5},
)
config = builder.build()
```

### Portable Configs with $resource References

```python
builder = (
    DynaBotConfigBuilder()
    .set_llm_resource("default")  # $resource reference
    .set_conversation_storage_resource("conversations")
    .set_custom_section("domain", {"id": "my-bot"})
)

# Flat format for DynaBot.from_config()
flat = builder.build()

# Portable format with bot wrapper
portable = builder.build_portable()
```

## Components

### Schema

`DynaBotConfigSchema` is a queryable registry of valid DynaBot config options. It auto-registers the 8 default components (llm, conversation_storage, memory, reasoning, knowledge_base, tools, middleware, system_prompt) and supports consumer extensions.

```python
from dataknobs_bots.config import DynaBotConfigSchema

schema = DynaBotConfigSchema()

# Query available options
providers = schema.get_valid_options("llm", "provider")
backends = schema.get_valid_options("conversation_storage", "backend")

# Register consumer extension
schema.register_extension("educational", {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["quiz", "tutor"]},
    }
})

# Generate LLM-friendly description
description = schema.to_description()
```

### Validation

`ConfigValidator` runs completeness checks, schema validation, and custom validators.

```python
from dataknobs_bots.config import ConfigValidator, ValidationResult

validator = ConfigValidator(schema=schema)
result = validator.validate(config)

# Register custom validators
validator.register_validator("my_check", my_validator_fn)
```

### Templates

Templates use `{{variable}}` placeholders and `$resource` references for portability.

Three built-in templates are included:

- **basic_assistant** — Simple chatbot
- **rag_assistant** — Bot with knowledge base
- **tool_user** — Bot with ReAct reasoning and tools

### Builder

`DynaBotConfigBuilder` provides fluent methods for all DynaBot components. Consumer extension via `set_custom_section()` — no subclassing needed.

Two output formats:

- `build()` — flat format compatible with `DynaBot.from_config()`
- `build_portable()` — environment-aware format with `$resource` refs and `bot` wrapper

### Draft Management

`ConfigDraftManager` provides file-based draft persistence for interactive config creation with automatic cleanup of stale drafts.

### Tools

Five `ContextAwareTool` implementations for wizard-driven config flows. Consumer extension via `builder_factory` and `on_save` callbacks.

## Consumer Extension Pattern

The toolkit uses composition, not subclassing. Consumers provide:

1. **`builder_factory`** callback — builds domain-specific config from wizard data
2. **`on_save`** callback — performs post-save actions
3. **`register_extension()`** — adds domain-specific schema sections
4. **`set_custom_section()`** — adds domain-specific config sections

```python
# Example: EduBot setup
schema = DynaBotConfigSchema()
schema.register_extension("educational", edu_schema)

def edu_builder_factory(wizard_data):
    builder = DynaBotConfigBuilder(schema=schema)
    # ... build generic sections ...
    builder.set_custom_section("educational", {
        "mode": wizard_data.get("mode", "tutor"),
        "enable_hints": wizard_data.get("hints_enabled", True),
    })
    return builder

preview_tool = PreviewConfigTool(builder_factory=edu_builder_factory)
save_tool = SaveConfigTool(
    draft_manager=manager,
    on_save=register_with_bot_manager,
)
```
