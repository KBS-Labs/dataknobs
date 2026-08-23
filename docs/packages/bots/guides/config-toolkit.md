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
| **Tool Catalog** | `ToolCatalog`, `ToolEntry`, `CatalogDescribable` | Tool name → class path registry with metadata, tags, and dependency tracking |
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

--8<-- "packages/bots/docs/CONFIG_TOOLKIT.md:combining-results"

### Templates

Templates use `{{variable}}` placeholders and `$resource` references for portability.

Three built-in templates are included:

- **basic_assistant** — Simple chatbot
- **rag_assistant** — Bot with knowledge base
- **tool_user** — Bot with ReAct reasoning and tools

### Builder

`DynaBotConfigBuilder` provides fluent methods for all DynaBot components. Consumer extension via `set_custom_section()` — no subclassing needed.

Three output methods:

- `build()` — flat format compatible with `DynaBot.from_config()`
- `build_portable()` — environment-aware format with `$resource` refs and `bot` wrapper
- `build_unvalidated()` — the flat dict **without** validating, for callers that
  report a `ValidationResult` rather than raise on one. Pair it with `validate()`;
  never use it on a path that writes or deploys

#### Custom Storage Classes

Use `set_conversation_storage_class()` to configure a custom `ConversationStorage`
implementation instead of the default `DataknobsConversationStorage`:

```python
config = (
    DynaBotConfigBuilder()
    .set_llm("ollama", model="llama3.2")
    .set_conversation_storage_class(
        "myapp.storage:AcmeConversationStorage",
        db_url="postgres://...",
        tenant_id="acme-corp",
    )
    .build()
)
```

The import path supports both `"module.path:ClassName"` (recommended) and
`"module.path.ClassName"` formats. The class must implement `ConversationStorage`
and provide an async `create(config: dict)` classmethod. When `storage_class` is
set, the `backend` key is ignored and the class handles its own initialization.

### Draft Management

`ConfigDraftManager` provides file-based draft persistence for interactive config creation with automatic cleanup of stale drafts.

### Tool Catalog

`ToolCatalog` maps tool names to fully-qualified class paths and default configuration. Built on `Registry[ToolEntry]` for thread safety, metrics, and consistent error handling.

The `default_catalog` singleton is pre-populated with all 12 built-in tools. Use `create_default_catalog()` for an extensible copy.

```python
from dataknobs_bots.config import (
    DynaBotConfigBuilder, default_catalog, create_default_catalog,
)

# Add tools to builder by name
builder = DynaBotConfigBuilder()
builder.add_tool_by_name(default_catalog, "knowledge_search", k=10)
builder.add_tools_by_name(default_catalog, ["list_templates", "preview_config"])

# Extend with custom tools
catalog = create_default_catalog()
catalog.register_tool(
    name="calculator",
    class_path="myapp.tools.CalculatorTool",
    description="Perform math calculations.",
    tags=("educational",),
)

# Generate bot config entries
config = catalog.to_bot_config("knowledge_search", k=10)
# {"class": "dataknobs_bots.tools.knowledge_search.KnowledgeSearchTool",
#  "params": {"k": 10}}
```

Tool classes can self-describe via `catalog_metadata()` classmethod (the `CatalogDescribable` protocol). `WizardConfigBuilder` validates stage tool names against the catalog when one is provided via `set_tool_catalog()`.

A parameter a tool declares under `requires` has two spellings — the live object, or the YAML value the tool builds one from (`template_registry` or `template_dir`, `draft_manager` or `config_dir`, a `builder_factory` callable or a dotted path to it). Both arrive in the same params dict, and the tool prefers the live object. Writing your own tool with a `requires` entry? `injected_dependency(config, key, expected)` is the one line that tells the two apart; pass `InjectedCallable` as `expected` where the YAML form is a dotted path. See [CONFIG_TOOLKIT.md](https://github.com/KBS-Labs/dataknobs/blob/main/packages/bots/docs/CONFIG_TOOLKIT.md) for the full table.

The `default_catalog` contains all 12 built-in tools with tag-based filtering, dependency validation, serialization, and self-describing tool support via `catalog_metadata()`.

### Tools

Six `ContextAwareTool` implementations for wizard-driven config flows:

| Tool | Purpose | Key Dependency |
|------|---------|---------------|
| `ListTemplatesTool` | List available templates | `ConfigTemplateRegistry` |
| `GetTemplateDetailsTool` | Get template details | `ConfigTemplateRegistry` |
| `PreviewConfigTool` | Preview config being built | `builder_factory` callback |
| `ValidateConfigTool` | Validate current config | `builder_factory` — the builder's validator decides |
| `SaveConfigTool` | Save/finalize config | `ConfigDraftManager` + `on_save` + `portable` |
| `ListAvailableToolsTool` | List tools for bot config | `available_tools` catalog |

Consumer extension via `builder_factory`, `on_save`, `portable`, and `available_tools`.

`ValidateConfigTool` and `SaveConfigTool` wired to the **same** `builder_factory`
reach one verdict: the builder's own validator decides both, at either setting of
`portable`. Two caveats, both by construction — each tool resolves its own factory
from its own config block, so nothing checks that the two name the same callable;
and a `ConfigValidator` passed to `ValidateConfigTool` is optional and runs *in
addition*, so it can refuse what save accepts. That asymmetry is deliberate: an
extra error stops an author, a missing one misleads them.

### KB Tools

Five `ContextAwareTool` implementations for RAG resource management during wizard flows:

| Tool | Purpose | Constructor Params |
|------|---------|-------------------|
| `CheckKnowledgeSourceTool` | Verify a knowledge source directory | (none) |
| `ListKBResourcesTool` | List tracked KB resources | (none) |
| `AddKBResourceTool` | Add a resource to the KB list | `knowledge_dir: Path \| None` |
| `RemoveKBResourceTool` | Remove a resource from the KB list | (none) |
| `IngestKnowledgeBaseTool` | Write manifest and finalize KB config | `knowledge_dir: Path \| None` |

KB tools operate on wizard collected data to track knowledge sources, supporting both file references and inline content. The `knowledge_dir` parameter is resolved from the constructor or wizard data `_knowledge_dir` key.

## Consumer Extension Pattern

The toolkit uses composition, not subclassing. Consumers provide:

1. **`builder_factory`** callback — builds domain-specific config from wizard data
2. **`on_save`** callback — performs post-save actions
3. **`portable`** flag — use `build_portable()` for bot-wrapped output instead of
   the flat `build()`. Both validate and refuse an invalid config
4. **`available_tools`** catalog — consumer-specific tool list
5. **`ToolCatalog`** — tool name → class path registry, extensible via `create_default_catalog()`
6. **`knowledge_dir`** path — base directory for KB files
7. **`register_extension()`** — adds domain-specific schema sections
8. **`set_custom_section()`** — adds domain-specific config sections

```python
# Example: EduBot setup
from pathlib import Path
from dataknobs_bots.tools import (
    PreviewConfigTool, SaveConfigTool, ListAvailableToolsTool,
    CheckKnowledgeSourceTool, AddKBResourceTool, IngestKnowledgeBaseTool,
)

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
    portable=True,
)
tools_tool = ListAvailableToolsTool(available_tools=MY_TOOL_CATALOG)

# KB tools with consumer-resolved knowledge directory
kb_dir = Path(os.environ.get("KNOWLEDGE_DIR", "data/knowledge"))
check_tool = CheckKnowledgeSourceTool()
add_tool = AddKBResourceTool(knowledge_dir=kb_dir)
ingest_tool = IngestKnowledgeBaseTool(knowledge_dir=kb_dir)
```
