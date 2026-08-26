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
# {"bot": {"llm": {"$resource": "default", ...}}, "domain": {"id": "my-bot"}}

# Flat format WITHOUT validating -- for callers that report a
# ValidationResult rather than raise on one. Pair it with validate().
draft = builder.build_unvalidated()
result = builder.validate()
```

### Adding Tools by Name

```python
from dataknobs_bots.config import DynaBotConfigBuilder, default_catalog

config = (
    DynaBotConfigBuilder()
    .set_llm("ollama", model="llama3.2")
    .set_conversation_storage("memory")
    .set_reasoning("react")
    .add_tool_by_name(default_catalog, "knowledge_search", k=10)
    .build()
)
# tool entry: {"class": "dataknobs_bots.tools.knowledge_search.KnowledgeSearchTool",
#              "params": {"k": 10}}
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

<!-- --8<-- [start:marker-rule] -->
### The `$resource` marker rule

A config section may be a `$resource` reference rather than a literal config.
Its marker vocabulary is closed — `$resource`, `type`, `$requires`, `$required`
— and anything else `$`-prefixed inside a reference is a malformed reference,
not an inline default. A stranded `$required` or `$requires` on a block with no
`$resource` is the same defect from the other side: it says the selector key
itself is the misspelled one.

Validation enforces that rule **at every depth**, on every section, whether or
not a schema is registered for it — `$requred: true` reads as *not required*,
and catching it at config-lint time is the difference between one confusing
message and a factory called with a keyword argument it did not expect.

```python
result = validator.validate(config)
# -> valid=False, errors=[
#      "Unknown marker key(s) ['$requred'] in the $resource reference for
#       'vectors' at config path 'knowledge_base.vector_store'. ..."
#    ]

# Or on one section, rooted so the path locates something:
result = validator.validate_component("knowledge_base", section)
```

The messages are `dataknobs-config`'s own — the same sentences resolution
raises — because one defect described two ways is two defects to the reader.
The rule itself is `collect_marker_violations()`, exported from
`dataknobs_config`; `marker_violations_result()` wraps it in a
`ValidationResult` for a pipeline composing its own validators.
<!-- --8<-- [end:marker-rule] -->

<!-- --8<-- [start:combining-results] -->
### Combining results

`ValidationResult` offers two ways to combine, and the difference matters when
two validators cover overlapping ground:

| Method | Repeated message | Use when |
|---|---|---|
| `merge` | kept | accumulating findings from one validator |
| `merge_unique` | reported once | composing two validators over the same config |

`merge` concatenates and keeps every message, because whether a repeat is one
finding or two is a property of the composition, which the method cannot see.
Most call sites accumulate findings from a single pass, where dropping a repeat
would drop a real finding.

`merge_unique` is for the other case. `validate_completeness` runs inside every
`ConfigValidator`, so running two of them over one config finds the same missing
key twice — an artefact of running two validators, not a second defect. Distinct
messages are never collapsed and order is preserved either way.

```python
builder_result = builder.validate()
combined = builder_result.merge_unique(my_validator.validate(config))
```
<!-- --8<-- [end:combining-results] -->

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

### Custom Storage Classes

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
`"module.path.ClassName"` formats. The async `create(config: dict)` classmethod is
required and checked when the config loads; implementing `ConversationStorage` is
expected but not gated, so a duck-typed class is accepted. See
[CONFIGURATION.md](CONFIGURATION.md#custom-storage-class) for details.

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

### Names stay inside the output directory

Every path the manager composes — the final config name, the alias name
passed as `config_name`, and the `draft_id` — is checked to land inside
`output_dir` before anything is written or unlinked. A name that walks
out with `..`, or one that is absolute and so discards the directory
entirely, raises `PathEscapeError` — a `ValueError` subclass, so an
existing `except ValueError` still catches it, and code that needs to
tell a refused name from any other bad value can now do so:

```python
from dataknobs_common import PathEscapeError

manager.finalize(draft_id, final_name="../../etc/cron.d/job")  # PathEscapeError
manager.config_path("reports/quarterly")                       # fine
```

A name addressing a subdirectory is written there whether or not the
subdirectory exists yet — the manager creates it.

`SaveConfigTool` catches this and returns its ordinary
`{"success": False, "error": ...}` rather than letting it raise, so a
model that supplies an escaping name (or an escaping `_draft_id` out of
wizard data) gets something it can correct on its next turn.

This matters because the name is not always the caller's. `finalize()`
with no `final_name` reads it back out of the draft file's own metadata,
and `SaveConfigTool` supplies it from LLM tool arguments and wizard data
— so the check is at the point the path is composed rather than at any
one entry point. `config_path(name)` is public for callers that need the
resolved path themselves; use it instead of joining onto `output_dir`.

Nesting is still legal — `team/alpha` resolves to
`{output_dir}/team/alpha.yaml`, and the parent directory is created. Note
that `SaveConfigTool` applies a stricter *naming policy* on top of this,
rejecting any separator in a config name and returning a structured tool
error rather than raising, so an LLM can correct it on the next turn.

## Tool Catalog

`ToolCatalog` maps tool names to fully-qualified class paths and default configuration. It serves as a single source of truth for tools available to config builders and wizard flows.

Built on `Registry[ToolEntry]` from `dataknobs-common` for thread safety, metrics, and consistent error handling.

### Built-in Tools

The `default_catalog` singleton is pre-populated with all 21 built-in tools:

| Name | Class | Tags | Requires |
|------|-------|------|----------|
| `knowledge_search` | `KnowledgeSearchTool` | general, rag | knowledge_base |
| `list_templates` | `ListTemplatesTool` | configbot | template_registry |
| `get_template_details` | `GetTemplateDetailsTool` | configbot | template_registry |
| `preview_config` | `PreviewConfigTool` | configbot | builder_factory |
| `validate_config` | `ValidateConfigTool` | configbot | builder_factory |
| `save_config` | `SaveConfigTool` | configbot | draft_manager |
| `list_available_tools` | `ListAvailableToolsTool` | configbot | — |
| `check_knowledge_source` | `CheckKnowledgeSourceTool` | configbot, kb | — |
| `list_kb_resources` | `ListKBResourcesTool` | configbot, kb | — |
| `add_kb_resource` | `AddKBResourceTool` | configbot, kb | — |
| `remove_kb_resource` | `RemoveKBResourceTool` | configbot, kb | — |
| `ingest_knowledge_base` | `IngestKnowledgeBaseTool` | configbot, kb | — |
| `list_bank_records` | `ListBankRecordsTool` | wizard, bank | — |
| `add_bank_record` | `AddBankRecordTool` | wizard, bank | — |
| `update_bank_record` | `UpdateBankRecordTool` | wizard, bank | — |
| `remove_bank_record` | `RemoveBankRecordTool` | wizard, bank | — |
| `finalize_bank` | `FinalizeBankTool` | wizard, bank | — |
| `compile_artifact` | `CompileArtifactTool` | wizard, bank, artifact | — |
| `finalize_artifact` | `FinalizeArtifactTool` | wizard, bank, artifact | — |
| `complete_wizard` | `CompleteWizardTool` | wizard | — |
| `restart_wizard` | `RestartWizardTool` | wizard | — |

An empty **Requires** cell says the tool declares nothing to be handed at
construction — not that it needs nothing. The nine wizard tools at the foot of
the table take whichever of `banks`, `catalog` and `artifact` they use as a
constructor override, and otherwise read it from `context.extra` on the turn
that calls them. That is a third channel alongside the two the next section
describes, and it is why they declare no `requires`: by the time the value
exists there is no constructor left to inject it into. They are registered here
so a wizard stage can name them, and documented in
[TOOLS.md](TOOLS.md#data-collection-tools-reference) — parameters, effects, and
the two-layer stage wiring — rather than repeated here.

### Supplying a declared dependency

The **Requires** column names what a tool needs handed to it, and each of
those parameters has two spellings: the live object, or the YAML value the
tool builds one from. A tool takes whichever it is given, preferring the
live object.

```python
# Live: the catalog passes keywords straight into the tool's params.
tool = default_catalog.instantiate_tool("list_templates", template_registry=registry)
registry_of_tools = default_catalog.create_tool_registry(
    ["save_config"], overrides={"save_config": {"draft_manager": manager}}
)

# YAML: the same parameters, spelled as config data.
tool = default_catalog.instantiate_tool("list_templates", template_dir="configs/templates")
```

The two spellings, per parameter:

| Parameter | Live form | YAML form |
|---|---|---|
| `template_registry` | `ConfigTemplateRegistry` | `template_dir` — a directory to load |
| `draft_manager` | `ConfigDraftManager` | `config_dir` — an output directory |
| `builder_factory` | the callable | a dotted import path to it |
| `on_save` | the callable | a dotted import path to it |

`DynaBot._resolve_tool` fills the same dict from a `dependencies` map,
matching on the names the entry's `requires` declares, before handing it to
`from_config`. **Its supply side is not yet open**: `DynaBot.from_config()`
builds that map itself and puts exactly one thing in it, the configured
`knowledge_base`. So a bot built from config today reaches the four
parameters above through their YAML spelling; the live spelling is for code
that constructs tools itself, through the catalog or directly.

Writing a tool with a `requires` entry of your own? `from_config` has to
tell the two channels apart, and
`dataknobs_bots.config.injected_dependency` is the one line that does it:

```python
from dataknobs_bots.config import InjectedCallable, injected_dependency

@classmethod
def from_config(cls, config: dict[str, Any]) -> MyTool:
    store = injected_dependency(config, "vector_store", VectorStore)
    if store is None:
        store = build_store_from(config["store_path"])
    # `InjectedCallable` for a key whose YAML form is a dotted path:
    # a live callable satisfies it, a string does not.
    hook = injected_dependency(config, "on_event", InjectedCallable)
    return cls(vector_store=store, on_event=hook)
```

### Usage with Builders

```python
from dataknobs_bots.config import DynaBotConfigBuilder, default_catalog

# Single tool by name
builder = DynaBotConfigBuilder()
builder.add_tool_by_name(default_catalog, "knowledge_search", k=10)

# Multiple tools by name with per-tool overrides
builder.add_tools_by_name(
    default_catalog,
    ["list_templates", "preview_config"],
    overrides={"list_templates": {"template_dir": "custom/templates"}},
)
```

### Extending the Catalog

Use `create_default_catalog()` for a fresh copy that can be extended without affecting the singleton:

```python
from dataknobs_bots.config import create_default_catalog

catalog = create_default_catalog()
catalog.register_tool(
    name="calculator",
    class_path="myapp.tools.CalculatorTool",
    description="Perform math calculations.",
    tags=("educational",),
)
```

### Self-Describing Tools

Tool classes can declare their own catalog metadata via the `CatalogDescribable` protocol:

```python
from dataknobs_llm.tools import ContextAwareTool

class MyTool(ContextAwareTool):
    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        return {
            "name": "my_tool",
            "description": "Does something useful.",
            "tags": ("general",),
            "requires": ("knowledge_base",),
        }
    # ... rest of tool implementation

# Register from the class — class_path is computed automatically
catalog.register_from_class(MyTool)
```

### Config Generation

```python
# Single tool config dict
config = catalog.to_bot_config("knowledge_search", k=10)
# {"class": "dataknobs_bots.tools.knowledge_search.KnowledgeSearchTool",
#  "params": {"k": 10}}

# Multiple tool configs
configs = catalog.to_bot_configs(
    ["knowledge_search", "list_templates"],
    overrides={"knowledge_search": {"k": 5}},
)
```

### Dependency Validation

```python
# Check that tool requirements are satisfied by a config
warnings = catalog.check_requirements(
    ["knowledge_search", "list_templates"],
    {"knowledge_base": {...}, "template_registry": {...}},
)
# warnings is empty — both requirements met
```

### Wizard Builder Integration

When a `ToolCatalog` is provided to `WizardConfigBuilder`, stage tool names are validated against the catalog during `validate()`:

```python
from dataknobs_bots.config.wizard_builder import WizardConfigBuilder
from dataknobs_bots.config import default_catalog

builder = (
    WizardConfigBuilder("my-wizard")
    .set_tool_catalog(default_catalog)
    .add_conversation_stage(
        name="search",
        tools=["knowledge_search", "nonexistent_tool"],
    )
)

result = builder.validate()
# Error: Stage 'search' references unknown tool 'nonexistent_tool'
#        (not in the tool catalog)
```

### Serialization

Catalogs serialize to/from dicts (suitable for YAML storage):

```python
# Serialize
data = catalog.to_dict()
# {"tools": [{"name": "knowledge_search", "class_path": "...", ...}, ...]}

# Deserialize
restored = ToolCatalog.from_dict(data)
```

## Tools

Six ContextAwareTool implementations for wizard-driven config flows:

| Tool | Purpose | Key Dependency |
|------|---------|---------------|
| `ListTemplatesTool` | List available templates | `ConfigTemplateRegistry` |
| `GetTemplateDetailsTool` | Get template details | `ConfigTemplateRegistry` |
| `PreviewConfigTool` | Preview config being built | `builder_factory` callback |
| `ValidateConfigTool` | Validate current config | the `builder_factory`'s builder — its validator decides |
| `SaveConfigTool` | Save/finalize config | `ConfigDraftManager` + `on_save` + `portable` |
| `ListAvailableToolsTool` | List tools for bot config | `available_tools` catalog |

### Consumer Extension Points

- **`builder_factory`**: `PreviewConfigTool` and `ValidateConfigTool` accept a `builder_factory: Callable[[dict], DynaBotConfigBuilder]` that encapsulates domain-specific config building logic.
  When `ValidateConfigTool` has one, the builder's own validator decides the
  verdict — the same validator `build()` and `build_portable()` run — so wiring
  `ValidateConfigTool` and `SaveConfigTool` to the **same** factory makes the
  save outcome predictable from the validate outcome, at either setting of
  `portable`.

  Two things fall outside that, both by construction. Each tool resolves its own
  `builder_factory` from its own config block, and nothing checks that the two
  name the same callable. And a `ConfigValidator` passed to `ValidateConfigTool`
  is optional and runs *in addition* to the builder's — merged with
  `merge_unique`, so overlapping validators do not report a shared failure twice
  — which means it can refuse what save would accept. That direction is
  deliberate: an extra error stops an author, a missing one misleads them.
- **`on_save`**: `SaveConfigTool` accepts an `on_save: Callable[[str, dict], Any]` callback for post-save actions (e.g., registering the bot with a manager).
- **`portable`**: `SaveConfigTool` accepts `portable: bool = False`. When `True`, uses `build_portable()` to produce configs with a `bot` wrapper key; when `False`, `build()` produces the flat format. The flag selects the output shape, not whether the config is validated — both refuse an invalid config rather than writing it.
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

These are writes to the wizard's own collected data, not to a copy of it,
and the distinction has two halves that are worth separating:

- **Within a turn**, one tool's write is visible to the next tool called on
  the same turn — that is how Add and Remove see the `_kb_resources` list
  Check created.
- **Across turns**, the write survives being saved and reloaded, so a
  resource added three turns ago is still in the list.

Both halves require a wizard: the tools reach this data through
`ToolExecutionContext.wizard_data()`, and outside a wizard conversation it
returns `None` and the tool reports an error rather than writing somewhere
nothing will read.

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
