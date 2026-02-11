# ConfigBot Toolkit

The ConfigBot toolkit provides reusable infrastructure for building wizard-driven bot configuration experiences. It extracts the generic DynaBot config-building logic into composable components that any DynaBot consumer can assemble into their own ConfigBot.

## Overview

The toolkit provides the following layers:

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Schema** | `DynaBotConfigSchema`, `ComponentSchema` | Queryable registry of valid config options |
| **Validation** | `ConfigValidator`, `ValidationResult` | Pluggable validation pipeline |
| **Templates** | `ConfigTemplate`, `ConfigTemplateRegistry`, `TemplateVariable` | Template loading, variable substitution, tag-based filtering |
| **Builder** | `DynaBotConfigBuilder` | Fluent builder for DynaBot configs |
| **Drafts** | `ConfigDraftManager`, `DraftMetadata` | File-based draft lifecycle management |
| **Tools** | `ListTemplatesTool`, `GetTemplateDetailsTool`, `PreviewConfigTool`, `ValidateConfigTool`, `SaveConfigTool`, `ListAvailableToolsTool` | LLM-callable tools for wizard flows |
| **KB Tools** | `CheckKnowledgeSourceTool`, `ListKBResourcesTool`, `AddKBResourceTool`, `RemoveKBResourceTool`, `IngestKnowledgeBaseTool` | RAG resource management during wizard flows |

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
# {"bot": {"llm": {"$resource": "default", ...}}, "domain": {"id": "my-bot"}}
```

## Schema

`DynaBotConfigSchema` is a queryable registry of valid DynaBot config options. It auto-registers the 8 default components and supports consumer extensions.

```python
from dataknobs_bots.config import DynaBotConfigSchema

schema = DynaBotConfigSchema()

# Query available options
providers = schema.get_valid_options("llm", "provider")
# ["ollama", "openai", "anthropic", "huggingface", "echo"]

backends = schema.get_valid_options("conversation_storage", "backend")
# ["memory", "sqlite", "postgres", ...]

# Register consumer extension
schema.register_extension("educational", {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["quiz", "tutor"]},
    }
})

# Generate description for LLM system prompts
description = schema.to_description()
```

## Validation

`ConfigValidator` runs a pipeline of validators and returns a unified `ValidationResult`.

```python
from dataknobs_bots.config import ConfigValidator, ValidationResult

validator = ConfigValidator(schema=schema)

# Validate a config
result = validator.validate(config)
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
for warning in result.warnings:
    print(f"Warning: {warning}")

# Register custom validators
def check_domain_id(config):
    domain = config.get("domain", {})
    if isinstance(domain, dict) and " " in domain.get("id", ""):
        return ValidationResult.error("domain.id must not contain spaces")
    return ValidationResult.ok()

validator.register_validator("domain_id", check_domain_id)
```

## Templates

Templates define config structures with `{{variable}}` placeholders.

### Template YAML Format

```yaml
name: my_template
description: A bot template
version: "1.0.0"
tags: [assistant, rag]

variables:
  - name: bot_name
    type: string
    required: true
  - name: temperature
    type: number
    default: 0.7

structure:
  bot:
    llm:
      $resource: default
      type: llm_providers
      temperature: "{{temperature}}"
    conversation_storage:
      $resource: conversations
      type: databases
    system_prompt: "I am {{bot_name}}, here to help."
```

### Built-in Templates

Three built-in templates are included:

- **basic_assistant** — Simple chatbot with LLM + system prompt + storage
- **rag_assistant** — Bot with knowledge base, vector store, embedding
- **tool_user** — Bot with ReAct reasoning and tool definitions

## Builder

`DynaBotConfigBuilder` provides fluent methods for all DynaBot components plus an extension point for domain-specific sections.

### Extension Point: `set_custom_section()`

```python
builder = (
    DynaBotConfigBuilder()
    .set_llm("ollama")
    .set_conversation_storage("memory")
    .set_custom_section("educational", {
        "mode": "tutor",
        "enable_hints": True,
    })
    .set_custom_section("domain", {
        "id": "bio-tutor",
        "name": "Biology Tutor",
    })
)
```

### Template + Override Pattern

```python
builder = (
    DynaBotConfigBuilder()
    .from_template(template, variables)
    .merge_overrides({"llm": {"temperature": 0.3}})
    .add_tool("my_module.ExtraTool")
)
```

## Draft Management

`ConfigDraftManager` provides file-based draft persistence for wizard flows.

```python
from pathlib import Path
from dataknobs_bots.config import ConfigDraftManager

manager = ConfigDraftManager(output_dir=Path("/data/configs"))

# Create and update drafts
draft_id = manager.create_draft(config, stage="configure_llm")
manager.update_draft(draft_id, updated_config, stage="review", config_name="my-bot")

# Finalize
final = manager.finalize(draft_id, final_name="my-bot")

# Cleanup stale drafts
cleaned = manager.cleanup_stale()
```

## Tools

Six ContextAwareTool implementations for wizard-driven config flows:

| Tool | Purpose | Key Dependency |
|------|---------|---------------|
| `ListTemplatesTool` | List available templates | `ConfigTemplateRegistry` |
| `GetTemplateDetailsTool` | Get template details | `ConfigTemplateRegistry` |
| `PreviewConfigTool` | Preview config being built | `builder_factory` callback |
| `ValidateConfigTool` | Validate current config | `ConfigValidator` |
| `SaveConfigTool` | Save/finalize config | `ConfigDraftManager` + `on_save` + `portable` |
| `ListAvailableToolsTool` | List tools for bot config | `available_tools` catalog |

### Consumer Extension Points

- **`builder_factory`**: `PreviewConfigTool` and `ValidateConfigTool` accept a `builder_factory: Callable[[dict], DynaBotConfigBuilder]` that encapsulates domain-specific config building logic.
- **`on_save`**: `SaveConfigTool` accepts an `on_save: Callable[[str, dict], Any]` callback for post-save actions (e.g., registering the bot with a manager).
- **`portable`**: `SaveConfigTool` accepts `portable: bool = False`. When `True`, uses `build_portable()` to produce configs with a `bot` wrapper key.
- **`available_tools`**: `ListAvailableToolsTool` accepts a list of tool descriptors (consumer-specific catalog).

```python
from dataknobs_bots.tools import (
    ListTemplatesTool, PreviewConfigTool, SaveConfigTool,
    ListAvailableToolsTool,
)

# Consumer provides domain-specific builder factory
def my_builder_factory(wizard_data):
    builder = (
        DynaBotConfigBuilder()
        .set_llm(wizard_data.get("provider", "ollama"))
        .set_conversation_storage("memory")
    )
    # Add domain-specific sections
    builder.set_custom_section("domain", {
        "id": wizard_data.get("domain_id"),
    })
    return builder

list_tool = ListTemplatesTool(template_registry=registry)
preview_tool = PreviewConfigTool(builder_factory=my_builder_factory)
save_tool = SaveConfigTool(
    draft_manager=manager,
    on_save=lambda name, config: register_bot(name, config),
    portable=True,  # Use build_portable() for bot-wrapped output
)

# Consumer provides tool catalog
tools_tool = ListAvailableToolsTool(available_tools=[
    {"name": "search", "description": "Web search", "category": "info"},
    {"name": "calculator", "description": "Math operations", "category": "math"},
])
```

## KB Tools

Five ContextAwareTool implementations for managing RAG knowledge base resources during wizard flows. These tools operate on wizard collected data to track, add, remove, and ingest knowledge sources.

| Tool | Purpose | Constructor Params |
|------|---------|-------------------|
| `CheckKnowledgeSourceTool` | Verify a knowledge source directory | (none) |
| `ListKBResourcesTool` | List tracked KB resources | (none) |
| `AddKBResourceTool` | Add a resource to the KB list | `knowledge_dir: Path \| None` |
| `RemoveKBResourceTool` | Remove a resource from the KB list | (none) |
| `IngestKnowledgeBaseTool` | Write manifest and finalize KB config | `knowledge_dir: Path \| None` |

### Knowledge Directory Resolution

Tools that write files (`AddKBResourceTool`, `IngestKnowledgeBaseTool`) resolve the knowledge directory from:

1. **Constructor param** (`knowledge_dir`) — takes priority
2. **Wizard data** (`_knowledge_dir` key) — fallback

Consumers pass the directory at construction time (e.g., resolving from an environment variable) or let users set it during the wizard flow.

### Wizard Data Keys

KB tools read and write specific keys in wizard collected data:

| Key | Written By | Read By | Description |
|-----|-----------|---------|-------------|
| `source_verified` | Check | — | Whether source directory was found |
| `files_found` | Check | Ingest | Auto-discovered file names |
| `_source_path_resolved` | Check | List, Ingest | Resolved source path |
| `_kb_resources` | Check, Add, Remove | List, Add, Remove, Ingest | Resource list |
| `kb_config` | Ingest | — | Final KB configuration for bot config |
| `kb_resources` | Ingest | — | Finalized resource list (public key) |
| `ingestion_complete` | Ingest | — | Whether ingestion manifest was written |

### Example

```python
from pathlib import Path
from dataknobs_bots.tools import (
    CheckKnowledgeSourceTool, AddKBResourceTool,
    IngestKnowledgeBaseTool,
)

knowledge_dir = Path("/data/knowledge")

check_tool = CheckKnowledgeSourceTool()
add_tool = AddKBResourceTool(knowledge_dir=knowledge_dir)
ingest_tool = IngestKnowledgeBaseTool(knowledge_dir=knowledge_dir)
```
