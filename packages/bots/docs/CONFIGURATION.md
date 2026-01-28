# Configuration Reference

Complete reference for configuring DynaBot instances.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Environment-Aware Configuration](#environment-aware-configuration)
- [LLM Configuration](#llm-configuration)
- [Conversation Storage](#conversation-storage)
- [Memory Configuration](#memory-configuration)
- [Knowledge Base Configuration](#knowledge-base-configuration)
- [Reasoning Configuration](#reasoning-configuration)
- [Tools Configuration](#tools-configuration)
- [Prompts Configuration](#prompts-configuration)
- [Middleware Configuration](#middleware-configuration)
- [Resource Resolution](#resource-resolution)
- [Environment Variables](#environment-variables)
- [Complete Examples](#complete-examples)

---

## Overview

DynaBot uses a configuration-first approach where all bot behavior is defined through configuration files (YAML/JSON) or dictionaries. This allows for:

- Easy bot customization without code changes
- Configuration version control
- Environment-specific configurations
- Dynamic bot creation

### Configuration Formats

DynaBot supports multiple configuration formats:

**Python Dictionary:**
```python
config = {
    "llm": {"provider": "ollama", "model": "gemma3:1b"},
    "conversation_storage": {"backend": "memory"}
}
bot = await DynaBot.from_config(config)
```

**YAML File:**
```yaml
# bot_config.yaml
llm:
  provider: ollama
  model: gemma3:1b

conversation_storage:
  backend: memory
```

```python
import yaml

with open("bot_config.yaml") as f:
    config = yaml.safe_load(f)

bot = await DynaBot.from_config(config)
```

**JSON File:**
```json
{
  "llm": {
    "provider": "ollama",
    "model": "gemma3:1b"
  },
  "conversation_storage": {
    "backend": "memory"
  }
}
```

---

## Configuration Structure

### Minimal Configuration

The minimal configuration requires only LLM and conversation storage:

```yaml
llm:
  provider: ollama
  model: gemma3:1b

conversation_storage:
  backend: memory
```

### Full Configuration Schema

```yaml
# Required: LLM Configuration
llm:
  provider: string
  model: string
  temperature: float (optional, default: 0.7)
  max_tokens: int (optional, default: 1000)
  # ... provider-specific options

# Required: Conversation Storage
conversation_storage:
  backend: string
  # ... backend-specific options

# Optional: Memory
memory:
  type: string
  # ... memory-type-specific options

# Optional: Knowledge Base (RAG)
knowledge_base:
  enabled: boolean
  # ... knowledge base options

# Optional: Reasoning Strategy
reasoning:
  strategy: string
  # ... strategy-specific options

# Optional: Tools
tools:
  - class: string
    params: dict
  # or
  - xref:tools[tool_name]

# Optional: Tool Definitions
tool_definitions:
  tool_name:
    class: string
    params: dict

# Optional: Prompts Library
prompts:
  prompt_name: string
  # or
  prompt_name:
    template: string
    type: string

# Optional: System Prompt (smart detection)
system_prompt:
  name: string            # Explicit template reference
  strict: boolean         # If true, error if template not found
  # or
  content: string         # Inline content
  rag_configs: list       # RAG configs for inline content
  # or just
system_prompt: string     # Smart detection: template if exists in library, else inline

# Optional: Middleware
middleware:
  - class: string
    params: dict
```

---

## Environment-Aware Configuration

DynaBot supports environment-aware configuration for deploying the same bot across different environments (development, staging, production) where infrastructure differs. This is the **recommended approach** for production deployments.

### The Problem

Without environment-aware configuration, bot configs contain environment-specific details:

```yaml
# PROBLEMATIC: This config is not portable
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434  # Local only!

conversation_storage:
  backend: sqlite
  path: ~/.local/share/myapp/conversations.db  # Local path!
```

When stored in a shared registry or database, this config fails in production because:
- The Ollama URL doesn't exist in production
- The local path doesn't exist in containers

### The Solution: Resource References

Use **logical resource references** (`$resource`) to separate bot behavior from infrastructure:

```yaml
# PORTABLE: This config works in any environment
bot:
  llm:
    $resource: default        # Logical name
    type: llm_providers       # Resource type
    temperature: 0.7          # Behavioral setting (portable)

  conversation_storage:
    $resource: conversations
    type: databases
```

The logical names (`default`, `conversations`) are resolved at **instantiation time** against environment-specific bindings.

### Environment Configuration Files

Environment configs define concrete implementations for logical names:

**Development** (`config/environments/development.yaml`):
```yaml
name: development
resources:
  llm_providers:
    default:
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434

  databases:
    conversations:
      backend: memory
```

**Production** (`config/environments/production.yaml`):
```yaml
name: production
resources:
  llm_providers:
    default:
      provider: openai
      model: gpt-4
      api_key: ${OPENAI_API_KEY}

  databases:
    conversations:
      backend: postgres
      connection_string: ${DATABASE_URL}
```

### Using Environment-Aware Configuration

#### Method 1: BotResourceResolver (Recommended)

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import BotResourceResolver

# Load environment (auto-detects from DATAKNOBS_ENVIRONMENT)
env = EnvironmentConfig.load()

# Create resolver with all DynaBot factories registered
resolver = BotResourceResolver(env)

# Get initialized resources
llm = await resolver.get_llm("default")
db = await resolver.get_database("conversations")
vs = await resolver.get_vector_store("knowledge")
embedder = await resolver.get_embedding_provider("default")
```

#### Method 2: Lower-Level Resolution

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import create_bot_resolver

# Load environment
env = EnvironmentConfig.load("production", config_dir="config/environments")

# Create resolver
resolver = create_bot_resolver(env)

# Resolve resources manually
llm = resolver.resolve("llm_providers", "default")
await llm.initialize()

db = resolver.resolve("databases", "conversations")
await db.connect()
```

### Environment Detection

The environment is determined in this order:

1. **Explicit**: `DATAKNOBS_ENVIRONMENT=production`
2. **Cloud indicators**: AWS Lambda, ECS, Kubernetes, Google Cloud Run, Azure Functions
3. **Default**: `development`

```bash
# Set environment explicitly
export DATAKNOBS_ENVIRONMENT=production

# Or auto-detect based on cloud environment
# (AWS_EXECUTION_ENV, KUBERNETES_SERVICE_HOST, etc.)
```

### Resource Reference Syntax

```yaml
# Full syntax with type
llm:
  $resource: default
  type: llm_providers
  temperature: 0.7  # Override/default value

# Supported resource types
# - llm_providers
# - databases
# - vector_stores
# - embedding_providers
```

Additional fields in a resource reference are merged with the resolved config:

```yaml
# In bot config
llm:
  $resource: default
  type: llm_providers
  temperature: 0.9  # Override the environment's default

# If environment defines:
# llm_providers:
#   default:
#     provider: openai
#     model: gpt-4
#     temperature: 0.7

# Resolved config:
# provider: openai
# model: gpt-4
# temperature: 0.9  # Overridden!
```

### Best Practices

1. **Store portable configs**: Only store configs with `$resource` references in databases
2. **Resolve at instantiation**: Call `resolve_for_build()` immediately before creating objects
3. **Use environment variables for secrets**: Never hardcode API keys or credentials
4. **Define all environments**: Create config files for development, staging, and production
5. **Use logical names consistently**: Use the same logical names across all environment configs

---

## LLM Configuration

Configure the Large Language Model provider.

### Common Options

```yaml
llm:
  provider: string      # Required: LLM provider name
  model: string         # Required: Model identifier
  temperature: float    # Optional: Randomness (0.0-1.0), default: 0.7
  max_tokens: int       # Optional: Max response tokens, default: 1000
  api_key: string       # Optional: API key (use env var reference)
  base_url: string      # Optional: Custom API endpoint
```

### Provider-Specific Configurations

#### Ollama (Local)

```yaml
llm:
  provider: ollama
  model: gemma3:1b
  base_url: http://localhost:11434  # Optional, default
  temperature: 0.7
  max_tokens: 1000
```

**Supported Models:**
- `gemma3:1b` - Small, fast model
- `gemma3:7b` - Larger, better quality
- `llama3.1:8b` - Advanced reasoning
- `phi3:mini` - Compact model
- `mistral:7b` - General purpose

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull gemma3:1b
```

#### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}  # Reference environment variable
  temperature: 0.7
  max_tokens: 2000
  organization: ${OPENAI_ORG_ID}  # Optional
```

**Supported Models:**
- `gpt-4` - Most capable
- `gpt-4-turbo` - Faster, cheaper
- `gpt-3.5-turbo` - Fast, economical

**Environment Variables:**
```bash
export OPENAI_API_KEY=sk-...
export OPENAI_ORG_ID=org-...  # Optional
```

#### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.7
  max_tokens: 4096
```

**Supported Models:**
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fast, economical

**Environment Variables:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

#### Azure OpenAI

```yaml
llm:
  provider: azure_openai
  model: gpt-4
  api_key: ${AZURE_OPENAI_KEY}
  api_base: ${AZURE_OPENAI_ENDPOINT}
  api_version: "2023-05-15"
  deployment_name: my-gpt4-deployment
```

---

## Conversation Storage

Configure where conversation history is stored.

### Memory Backend (Development Only)

In-memory storage, not persistent:

```yaml
conversation_storage:
  backend: memory
```

**Use Cases:**
- Development and testing
- Demos and prototyping
- Ephemeral conversations

**Limitations:**
- Data lost on restart
- Not suitable for production
- No horizontal scaling

### PostgreSQL Backend (Production)

Persistent database storage:

```yaml
conversation_storage:
  backend: postgres
  host: localhost
  port: 5432
  database: myapp_db
  user: postgres
  password: ${DB_PASSWORD}
  pool_size: 20          # Optional, default: 10
  max_overflow: 10       # Optional, default: 5
  pool_timeout: 30       # Optional, default: 30 seconds
```

**Environment Variables:**
```bash
export DB_PASSWORD=your-secure-password
```

**Docker Setup:**
```bash
docker run -d \
  --name postgres-bots \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=myapp_db \
  -p 5432:5432 \
  postgres:14
```

**Connection Options:**
- `host`: Database host
- `port`: Database port (default: 5432)
- `database`: Database name
- `user`: Database user
- `password`: Database password
- `pool_size`: Connection pool size
- `max_overflow`: Extra connections beyond pool_size
- `pool_timeout`: Connection timeout in seconds

---

## Memory Configuration

Configure conversation context memory.

### Buffer Memory

Simple sliding window of recent messages:

```yaml
memory:
  type: buffer
  max_messages: 10  # Number of recent messages to keep
```

**Characteristics:**
- Fast and simple
- Low memory usage
- No semantic understanding
- Perfect for short conversations

**Recommended Settings:**
- Short conversations: `max_messages: 5-10`
- Medium conversations: `max_messages: 15-20`
- Long conversations: `max_messages: 30-50`

### Vector Memory

Semantic search over conversation history:

```yaml
memory:
  type: vector
  max_messages: 100
  embedding_provider: ollama
  embedding_model: nomic-embed-text
  backend: faiss
  dimension: 384      # Must match embedding model dimension
  metric: cosine      # Optional: cosine, l2, ip
```

**Embedding Models:**

| Provider | Model | Dimension | Use Case |
|----------|-------|-----------|----------|
| Ollama | nomic-embed-text | 384 | General purpose, fast |
| OpenAI | text-embedding-3-small | 1536 | High quality |
| OpenAI | text-embedding-3-large | 3072 | Best quality |
| OpenAI | text-embedding-ada-002 | 1536 | Legacy |

**Characteristics:**
- Semantic understanding
- Finds relevant context regardless of recency
- Higher memory usage
- Slightly slower than buffer memory

**When to Use:**
- Long conversations with topic changes
- Need to recall specific information
- Complex context requirements

---

## Knowledge Base Configuration

Enable Retrieval Augmented Generation (RAG).

### Basic Configuration

```yaml
knowledge_base:
  enabled: true
  documents_path: ./docs
  vector_store:
    backend: faiss
    dimension: 384
    collection: knowledge
  embedding_provider: ollama
  embedding_model: nomic-embed-text
```

### Advanced Configuration

```yaml
knowledge_base:
  enabled: true
  documents_path: ./docs

  # Vector store configuration
  vector_store:
    backend: faiss          # faiss, chroma, pinecone, weaviate
    dimension: 384          # Must match embedding dimension
    collection: knowledge   # Collection/index name
    metric: cosine         # Similarity metric

  # Embedding configuration
  embedding_provider: ollama
  embedding_model: nomic-embed-text

  # Document chunking
  chunking:
    max_chunk_size: 500    # Max characters per chunk
    chunk_overlap: 50      # Overlap between chunks
    separator: "\n\n"      # Chunk separator

  # File processing
  file_types:
    - txt
    - md
    - pdf                  # Requires pdfplumber
    - docx                 # Requires python-docx

  # Metadata
  metadata_fields:
    - filename
    - created_at
    - source
```

### Vector Store Backends

#### FAISS (Local)

```yaml
vector_store:
  backend: faiss
  dimension: 384
  index_type: IVF        # Optional: Flat, IVF, HNSW
  nlist: 100            # Optional: For IVF index
```

**Characteristics:**
- Fast local search
- No external dependencies
- Good for small to medium datasets
- Not distributed

#### Chroma (Local/Hosted)

```yaml
vector_store:
  backend: chroma
  dimension: 384
  collection: knowledge
  persist_directory: ./chroma_db  # Optional
```

**Characteristics:**
- Easy to use
- Local or hosted
- Good developer experience
- Persistent storage

#### Pinecone (Cloud)

```yaml
vector_store:
  backend: pinecone
  api_key: ${PINECONE_API_KEY}
  environment: us-west1-gcp
  index_name: knowledge
  dimension: 384
```

**Characteristics:**
- Fully managed
- High scalability
- Low latency
- Paid service

### Document Processing

```yaml
knowledge_base:
  enabled: true
  documents_path: ./docs

  # Process on startup
  auto_index: true

  # File filtering
  include_patterns:
    - "**/*.md"
    - "**/*.txt"
    - "docs/**/*.pdf"

  exclude_patterns:
    - "**/draft/*"
    - "**/_archive/*"
    - "**/README.md"

  # Chunking strategy
  chunking:
    strategy: recursive    # recursive, character, token
    max_chunk_size: 500
    chunk_overlap: 50
```

---

## Reasoning Configuration

Configure multi-step reasoning strategies.

### Simple Reasoning

Direct LLM response without reasoning steps:

```yaml
reasoning:
  strategy: simple
```

**Use Cases:**
- Simple Q&A
- Chatbots without tools
- Fast responses

### ReAct Reasoning

Reasoning + Acting with tools:

```yaml
reasoning:
  strategy: react
  max_iterations: 5       # Max reasoning steps
  verbose: true           # Log reasoning steps
  store_trace: true       # Store reasoning trace
  early_stopping: true    # Stop when answer found
```

**Configuration Options:**
- `max_iterations` (int): Maximum reasoning loops (default: 5)
- `verbose` (bool): Print reasoning steps to console (default: false)
- `store_trace` (bool): Store trace in memory for debugging (default: false)
- `early_stopping` (bool): Stop when final answer is reached (default: true)

**Use Cases:**
- Tool-using agents
- Multi-step problem solving
- Research and analysis tasks

**Example Trace:**
```
Iteration 1:
  Thought: I need to calculate 15 * 7
  Action: calculator(operation=multiply, a=15, b=7)
  Observation: 105

Iteration 2:
  Thought: I have the answer
  Final Answer: 15 multiplied by 7 is 105
```

### Wizard Reasoning

Guided conversational flows with FSM-backed state management:

```yaml
reasoning:
  strategy: wizard
  wizard_config: path/to/wizard.yaml  # Required: wizard definition file
  strict_validation: true              # Enforce JSON Schema validation
  extraction_config:                   # Optional: LLM for data extraction
    provider: ollama
    model: qwen3:1b
  hooks:                               # Optional: lifecycle callbacks
    on_enter:
      - "myapp.hooks:log_stage_entry"
    on_complete:
      - "myapp.hooks:save_wizard_results"
```

**Configuration Options:**
- `wizard_config` (string, required): Path to wizard YAML configuration file
- `strict_validation` (bool): Enforce JSON Schema validation per stage (default: true)
- `extraction_config` (dict): LLM configuration for extracting structured data from user input
- `custom_functions` (dict): Custom Python functions for transition conditions
- `hooks` (dict): Lifecycle hooks for stage transitions and completion

**Use Cases:**
- Multi-step data collection (onboarding, forms)
- Guided workflows with validation
- Branching conversational flows
- Complex wizards with conditional logic

**Wizard Configuration File Format:**

```yaml
# wizard.yaml
name: onboarding-wizard
version: "1.0"
description: User onboarding flow

stages:
  - name: welcome
    is_start: true
    prompt: "What kind of bot would you like to create?"
    schema:
      type: object
      properties:
        intent:
          type: string
          enum: [tutor, quiz, companion]
    suggestions: ["Create a tutor", "Build a quiz", "Make a companion"]
    transitions:
      - target: select_template
        condition: "data.get('intent')"

  - name: select_template
    prompt: "Would you like to start from a template?"
    can_skip: true
    help_text: "Templates provide pre-configured settings for common use cases."
    schema:
      type: object
      properties:
        use_template:
          type: boolean
    transitions:
      - target: configure
        condition: "data.get('use_template') == True"
      - target: configure

  - name: configure
    prompt: "Enter the bot name:"
    schema:
      type: object
      properties:
        bot_name:
          type: string
          minLength: 3
      required: ["bot_name"]
    transitions:
      - target: complete

  - name: complete
    is_end: true
    prompt: "Your bot '{{bot_name}}' is ready!"
```

**Stage Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Unique stage identifier (required) |
| `prompt` | string | User-facing message for this stage (required) |
| `is_start` | bool | Mark as wizard entry point |
| `is_end` | bool | Mark as wizard completion point |
| `schema` | object | JSON Schema for data validation |
| `suggestions` | list | Quick-reply buttons for users |
| `help_text` | string | Additional help shown on request |
| `can_skip` | bool | Allow users to skip this stage |
| `skip_default` | object | Default values to apply when user skips this stage |
| `can_go_back` | bool | Allow back navigation (default: true) |
| `tools` | list | Tool names available in this stage (must be explicit; omitting means no tools) |
| `reasoning` | string | Tool reasoning mode: "single" (default) or "react" for multi-tool loops |
| `max_iterations` | int | Max ReAct iterations for this stage (overrides wizard default) |
| `transitions` | list | Rules for transitioning to next stage |
| `tasks` | list | Task definitions for granular progress tracking |

**Task Configuration:**

Tasks enable granular progress tracking within and across wizard stages. Define per-stage tasks or global tasks that span stages.

```yaml
stages:
  configure_identity:
    prompt: "Let's set up your bot's identity..."
    schema:
      type: object
      properties:
        bot_name: { type: string }
        description: { type: string }
    # Per-stage task definitions
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

**Task Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique task identifier (required) |
| `description` | string | Human-readable description (required) |
| `completed_by` | string | Completion trigger: `field_extraction`, `tool_result`, `stage_exit`, `manual` |
| `field_name` | string | For `field_extraction`: which field triggers completion |
| `tool_name` | string | For `tool_result`: which tool triggers completion |
| `required` | bool | Whether task is required for wizard completion (default: true) |
| `depends_on` | list | List of task IDs that must complete first |

**Task Completion Triggers:**
| Trigger | Description |
|---------|-------------|
| `field_extraction` | Completed when specified field is extracted from user input |
| `tool_result` | Completed when specified tool executes successfully |
| `stage_exit` | Completed when user leaves the associated stage |
| `manual` | Completed programmatically via code |

For full task tracking API, see the Wizard Observability guide in the documentation.

**Navigation Commands:**
Users can navigate the wizard with natural language:
- "back" / "go back" / "previous" - Return to previous stage
- "skip" / "skip this" - Skip current stage (if `can_skip: true`)
- "use default" / "use defaults" - Skip and apply `skip_default` values (if `can_skip: true`)
- "restart" / "start over" - Restart from beginning

**Lifecycle Hooks:**
```yaml
hooks:
  on_enter:                    # Called when entering any stage
    - "myapp.hooks:log_entry"
  on_exit:                     # Called when leaving any stage
    - "myapp.hooks:validate_exit"
  on_complete:                 # Called when wizard finishes
    - "myapp.hooks:submit_results"
  on_restart:                  # Called when wizard is restarted
    - "myapp.hooks:log_restart"
  on_error:                    # Called on processing errors
    - "myapp.hooks:handle_error"
```

**Function Reference Syntax:**

Hook functions and custom transition functions are specified as string references.
Two formats are supported:

| Format | Example | Description |
|--------|---------|-------------|
| Colon (preferred) | `myapp.hooks:log_entry` | Explicit separator between module path and function |
| Dot (accepted) | `myapp.hooks.log_entry` | Last segment is treated as function name |

The colon format is preferred as it's unambiguous. The dot format is accepted for
convenience but requires the function to be the final segment.

**Error Messages:**

Invalid function references produce helpful error messages:

```
ImportError: Cannot import module 'myapp.hooks' from reference 'myapp.hooks:missing_func':
No module named 'myapp'. Ensure the module is installed and the path is correct.

AttributeError: Function 'missing_func' not found in module 'myapp.hooks'.
Available functions: log_entry, validate_data, submit_results
```

Hooks that fail to load are skipped with a warning, allowing the wizard to continue
operating even if some hooks are misconfigured.

**Tool Availability:**

Tools are only available to stages that explicitly list them via the `tools` property.
Stages without a `tools` key receive **no tools** by default. This prevents accidental
tool calls during data collection stages that could produce blank responses.

```yaml
stages:
  # Data collection stage - no tools available
  - name: configure_identity
    prompt: "What should we call your bot?"
    schema:
      type: object
      properties:
        bot_name: { type: string }
    # Note: no 'tools' key = no tools available

  # Tool-using stage - explicit tool list
  - name: review
    prompt: "Let's review your configuration"
    tools: [preview_config, validate_config]  # Only these tools available
    transitions:
      - target: save
```

**Skipping with Defaults:**

When users skip a stage, you can apply default values using `skip_default`:

```yaml
stages:
  - name: configure_llm
    prompt: "Which AI provider should power your bot?"
    can_skip: true
    skip_default:
      llm_provider: anthropic
      llm_model: claude-3-sonnet
    schema:
      type: object
      properties:
        llm_provider:
          type: string
          enum: [anthropic, openai, ollama]
        llm_model:
          type: string
    transitions:
      - target: next_stage
        condition: "data.get('llm_provider')"
```

This gives users three paths:
1. **Explicit choice**: User says "Use OpenAI GPT-4" → extraction captures their choice
2. **Accept defaults**: User says "skip" or "use defaults" → `skip_default` values applied
3. **Guided help**: User says "I'm not sure" → wizard explains options and re-prompts

> **Note:** Schema `default` values are stripped before extraction to prevent the LLM from
> auto-filling them. Use `skip_default` for user-facing defaults instead of schema defaults.

**Auto-Advance:**

When pre-populating wizard data (e.g., from templates or previous sessions), stages can
automatically advance if all required fields are already filled. Enable globally or per-stage:

```yaml
# wizard.yaml
name: configbot
settings:
  auto_advance_filled_stages: true  # Global setting

stages:
  - name: configure_identity
    prompt: "What's your bot's name?"
    schema:
      type: object
      properties:
        bot_name: { type: string }
        description: { type: string }
      required: [bot_name, description]
    # If both bot_name and description exist in wizard data,
    # this stage will auto-advance to the next stage
    transitions:
      - target: configure_llm
        condition: "data.get('bot_name')"

  - name: configure_llm
    auto_advance: true  # Per-stage override (works even if global is false)
    prompt: "Which LLM provider?"
    schema:
      type: object
      properties:
        llm_provider: { type: string }
      required: [llm_provider]
    transitions:
      - target: done
```

Auto-advance conditions:
- Global `auto_advance_filled_stages: true` in settings, OR stage has `auto_advance: true`
- Stage has a schema with `required` fields (or all `properties` if no `required` list)
- All required fields have non-empty values in wizard data
- Stage is not an end stage (`is_end: false`)
- At least one transition condition is satisfied

This enables "template-first" workflows where users select a template that pre-fills most
fields, and the wizard skips to the first stage needing user input.

**Post-Completion Amendments:**

Allow users to make changes after the wizard has completed. When enabled, the wizard
detects edit requests and re-opens at the relevant stage:

```yaml
# wizard.yaml
name: configbot
settings:
  allow_post_completion_edits: true
  section_to_stage_mapping:    # Optional: custom section-to-stage mapping
    model: configure_llm
    ai: configure_llm
    bot: configure_identity

stages:
  - name: configure_llm
    prompt: "Which LLM provider?"
    # ...
  - name: configure_identity
    prompt: "What's your bot's name?"
    # ...
  - name: save
    is_end: true
    prompt: "Configuration saved!"
```

After completing the wizard, if the user says "change the LLM to ollama", the wizard:
1. Detects the edit intent using extraction
2. Maps "llm" to `configure_llm` stage
3. Re-opens the wizard at that stage
4. Normal wizard flow resumes (user makes change, wizard advances through review/save)

Default section-to-stage mappings:
| Section | Stage |
|---------|-------|
| llm, model, ai | configure_llm |
| identity, name | configure_identity |
| knowledge, kb, rag | configure_knowledge |
| tools | configure_tools |
| behavior | configure_behavior |
| template | select_template |
| config | review |

Custom mappings in `section_to_stage_mapping` override defaults. Only stages that exist
in your wizard configuration are valid targets.

Requirements for amendment detection:
- `allow_post_completion_edits: true` in settings
- An `extraction_config` must be specified (extractor is used to detect edit intent)
- The target stage must exist in the wizard

**Context Template:**

Customize how stage context is formatted in the system prompt using Jinja2 templates:

```yaml
# wizard.yaml
name: configbot
settings:
  context_template: |
    ## Wizard Stage: {{stage_name}}

    **Goal**: {{stage_prompt}}

    ((Additional help: {{help_text}}))

    {% if collected_data %}
    ### Already Collected (DO NOT ASK AGAIN)
    {% for key, value in collected_data.items() %}
    - **{{key}}**: {{value}}
    {% endfor %}
    {% endif %}

    {% if not completed %}
    Navigation: {% if can_skip %}Can skip{% endif %}{% if can_go_back %}, Can go back{% endif %}
    {% endif %}

    {% if suggestions %}
    Suggestions: {{ suggestions | join(', ') }}
    {% endif %}
```

Available template variables:
| Variable | Type | Description |
|----------|------|-------------|
| `stage_name` | string | Current stage name |
| `stage_prompt` | string | Stage's goal/prompt text |
| `help_text` | string | Additional help text (empty string if none) |
| `suggestions` | list | Quick-reply suggestions |
| `collected_data` | dict | Data collected so far (excludes `_` prefixed keys) |
| `raw_data` | dict | All wizard data including internal keys |
| `completed` | bool | Whether wizard is complete |
| `history` | list | List of visited stage names |
| `can_skip` | bool | Whether current stage can be skipped |
| `can_go_back` | bool | Whether back navigation is allowed |

Special syntax:
- `((content))` - Conditional section, removed if any variable inside is empty/falsy
- Standard Jinja2: `{% if %}`, `{% for %}`, `{{ var | filter }}`

If no `context_template` is specified, the wizard uses a default format that includes
stage info, collected data, and navigation hints.

**Wizard State API:**

Access wizard state programmatically from your application code:

```python
# Get current wizard state for a conversation
state = bot.get_wizard_state("conversation-123")

if state:
    print(f"Stage: {state['current_stage']} ({state['stage_index'] + 1}/{state['total_stages']})")
    print(f"Progress: {state['progress'] * 100:.0f}%")
    print(f"Collected: {state['data']}")

    if state['can_skip']:
        print("User can skip this stage")
    if not state['completed']:
        print(f"Suggestions: {state['suggestions']}")
```

The `get_wizard_state()` method returns a normalized dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `current_stage` | string | Name of the current stage |
| `stage_index` | int | Zero-based index of current stage |
| `total_stages` | int | Total number of stages in wizard |
| `progress` | float | Completion progress (0.0 to 1.0) |
| `completed` | bool | Whether wizard has finished |
| `data` | dict | All collected data |
| `can_skip` | bool | Whether current stage can be skipped |
| `can_go_back` | bool | Whether back navigation is allowed |
| `suggestions` | list | Quick-reply suggestions for current stage |
| `history` | list | List of visited stage names |

Returns `None` if the conversation doesn't exist or has no active wizard.

**WizardFSM Introspection:**

When working directly with WizardFSM instances (e.g., in custom reasoning strategies),
you can introspect the wizard structure:

```python
from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

loader = WizardConfigLoader()
wizard_fsm = loader.load("path/to/wizard.yaml")

# Get all stage names in order
print(wizard_fsm.stage_names)  # ['welcome', 'configure', 'review', 'complete']

# Get total stage count
print(wizard_fsm.stage_count)  # 4

# Get full stage metadata
for name, meta in wizard_fsm.stages.items():
    print(f"{name}: {meta.get('prompt', 'No prompt')}")
    if meta.get('can_skip'):
        print(f"  - Can be skipped")
```

| Property | Type | Description |
|----------|------|-------------|
| `stages` | dict | All stage metadata (returns a copy) |
| `stage_names` | list | Ordered list of stage names |
| `stage_count` | int | Total number of stages |
| `current_stage` | string | Current stage name |
| `current_metadata` | dict | Metadata for current stage |

**ReAct-Style Tool Reasoning:**

Enable multi-tool ReAct loops within wizard stages. When a stage has tools and is
configured for ReAct reasoning, the LLM can make multiple sequential tool calls within
a single wizard turn, reasoning about results before responding.

```yaml
# wizard.yaml
name: configbot
settings:
  tool_reasoning: single       # Default for all stages: "single" or "react"
  max_tool_iterations: 3       # Default max tool-calling iterations

stages:
  - name: review
    prompt: "Let's review your configuration"
    reasoning: react           # Override: use ReAct loop for this stage
    max_iterations: 5          # Override: allow up to 5 tool calls
    tools: [preview_config, validate_config]
    transitions:
      - target: save

  - name: configure_llm
    prompt: "Which LLM provider?"
    # No reasoning specified = uses wizard-level default (single)
    # No tools specified = no tools available
    transitions:
      - target: review
```

**ReAct Behavior:**

With `reasoning: react`, the wizard:
1. Calls the LLM with available tools
2. If LLM requests a tool call, executes it
3. Adds tool result to conversation
4. Repeats from step 1 (up to `max_iterations`)
5. When LLM responds without tool calls, returns that as the final response

This enables complex multi-tool interactions in a single wizard turn:

```
User: "Show me the config and validate it"
LLM calls: preview_config
Tool returns: {preview: {...}}
LLM calls: validate_config
Tool returns: {valid: true, errors: []}
LLM responds: "Here's your config preview: ... Validation passed!"
```

**Configuration Options:**

| Setting | Level | Description |
|---------|-------|-------------|
| `tool_reasoning` | wizard settings | Default reasoning mode: "single" (one LLM call) or "react" (loop) |
| `max_tool_iterations` | wizard settings | Default max iterations for react mode |
| `reasoning` | stage | Per-stage override: "single" or "react" |
| `max_iterations` | stage | Per-stage max iterations override |

**When to Use ReAct:**

Use `reasoning: react` for stages where:
- Multiple tools may need to be called together
- Tool results inform subsequent tool calls
- You want the LLM to reason about tool outputs before responding

Keep `reasoning: single` (default) for:
- Data collection stages without tools
- Simple single-tool stages
- Stages where you want predictable single-call behavior

---

## Tools Configuration

Configure tools that extend bot capabilities.

### Tool Loading Methods

#### Direct Class Instantiation

```yaml
tools:
  - class: my_module.CalculatorTool
    params:
      precision: 2

  - class: my_module.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}
      default_location: "New York"
```

#### XRef to Predefined Tools

```yaml
# Define reusable tool configurations
tool_definitions:
  calculator_2dp:
    class: my_module.CalculatorTool
    params:
      precision: 2

  calculator_5dp:
    class: my_module.CalculatorTool
    params:
      precision: 5

  weather:
    class: my_module.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}

# Reference tools by name
tools:
  - xref:tools[calculator_2dp]
  - xref:tools[weather]
```

#### Mixed Approach

```yaml
tool_definitions:
  weather:
    class: my_module.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}

tools:
  # Direct instantiation
  - class: my_module.CalculatorTool
    params:
      precision: 3

  # XRef reference
  - xref:tools[weather]
```

### Built-in Tools

#### Knowledge Search Tool

Automatically available when knowledge base is enabled:

```yaml
knowledge_base:
  enabled: true
  # ... knowledge base config

tools:
  - class: dataknobs_bots.tools.KnowledgeSearchTool
    params:
      k: 5  # Number of results to return
```

### Custom Tool Structure

Custom tools must inherit from `dataknobs_llm.tools.Tool`:

```python
# my_tools.py
from dataknobs_llm.tools import Tool
from typing import Dict, Any

class CalculatorTool(Tool):
    def __init__(self, precision: int = 2):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic"
        )
        self.precision = precision

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(self, operation: str, a: float, b: float) -> float:
        # Implementation
        pass
```

**Configuration:**
```yaml
tools:
  - class: my_tools.CalculatorTool
    params:
      precision: 3
```

See [TOOLS.md](tools.md) for detailed tool development guide.

---

## Prompts Configuration

Configure custom prompts for the bot.

### Simple String Prompts

```yaml
prompts:
  helpful_assistant: "You are a helpful AI assistant."
  technical_support: "You are a technical support specialist."
  creative_writer: "You are a creative writing assistant."

system_prompt:
  name: helpful_assistant
```

### Structured Prompts

```yaml
prompts:
  customer_support:
    type: system
    template: |
      You are a customer support agent for {company_name}.

      Your role:
      - Be helpful and friendly
      - Answer questions about {product}
      - Escalate complex issues

      Guidelines:
      - Always greet customers
      - Use simple language
      - Stay professional

system_prompt:
  name: customer_support
```

### Prompt Variables

Use variables in prompts:

```yaml
prompts:
  personalized:
    template: |
      You are an AI assistant helping {user_name}.
      Context: {user_context}

system_prompt:
  name: personalized
```

Provide variables at runtime:

```python
# Pass variables in BotContext
context = BotContext(
    conversation_id="conv-123",
    client_id="client-456",
    session_metadata={
        "user_name": "Alice",
        "user_context": "Premium customer since 2020"
    }
)
```

### System Prompt Configuration

The `system_prompt` field supports multiple formats for flexibility with **smart detection**.

#### Smart Detection (Recommended)

When you provide a string, DynaBot uses **smart detection** to determine if it's a template name or inline content:

- If the string **exists in the prompt library** → treated as template name
- If the string **does not exist in the library** → treated as inline content

```yaml
# Example 1: String exists in prompts - used as template name
prompts:
  helpful_assistant: "You are a helpful AI assistant."

system_prompt: helpful_assistant  # Found in prompts → template reference
```

```yaml
# Example 2: String does NOT exist in prompts - used as inline content
prompts: {}  # Empty or no prompts section

system_prompt: "You are a helpful AI assistant."  # Not found → inline content
```

This means you can write short, simple prompts directly without needing to define them in the prompts library first.

#### 1. Dict with Template Name (Explicit)

Explicitly reference a prompt defined in the `prompts` section:

```yaml
prompts:
  helpful_assistant: "You are a helpful AI assistant."

system_prompt:
  name: helpful_assistant
```

#### 2. Dict with Strict Mode

Use `strict: true` to raise an error if the template doesn't exist:

```yaml
system_prompt:
  name: my_template
  strict: true  # Raises ValueError if my_template doesn't exist
```

#### 3. Dict with Inline Content

Provide the prompt content directly without defining it in `prompts`:

```yaml
system_prompt:
  content: "You are a helpful AI assistant specialized in customer support."
```

#### 4. Dict with Inline Content + RAG

Inline content can also include RAG configurations for context injection:

```yaml
system_prompt:
  content: |
    You are a helpful assistant.

    Use the following context to answer questions:
    {{CONTEXT}}
  rag_configs:
    - adapter_name: docs
      query: "assistant guidelines"
      placeholder: CONTEXT
      k: 3
```

#### 5. Multi-line String as Inline Content

Multi-line strings are common for inline prompts in YAML:

```yaml
system_prompt: |
  You are a helpful AI assistant specialized in customer support.

  Key responsibilities:
  - Answer questions accurately and helpfully
  - Be polite and professional at all times
  - Escalate complex issues to human agents when necessary

  Remember to always verify customer identity before sharing sensitive information.
```

This format is ideal when:
- Writing prompts directly in YAML without a separate prompts library
- The prompt is specific to this configuration and won't be reused
- You want to keep the entire configuration self-contained

#### Best Practices

**Use template names when:**
- The same prompt is reused across multiple configurations
- You want centralized prompt management
- Prompts need variables/templating
- You want to version control prompts separately

**Use inline content when:**
- The prompt is specific to one configuration
- You want a self-contained YAML file
- Quick prototyping or testing

**Use strict mode when:**
- You want to catch configuration errors early
- The template MUST exist (e.g., production configs)

---

## Middleware Configuration

Add request/response processing middleware for logging, cost tracking, and more.

### Built-in Middleware

DataKnobs Bots provides two built-in middleware classes:

**CostTrackingMiddleware** - Tracks LLM costs and token usage:

```yaml
middleware:
  - class: dataknobs_bots.middleware.CostTrackingMiddleware
    params:
      track_tokens: true
      cost_rates:  # Optional: override default rates
        openai:
          gpt-4o:
            input: 0.0025
            output: 0.01
```

**LoggingMiddleware** - Logs all interactions:

```yaml
middleware:
  - class: dataknobs_bots.middleware.LoggingMiddleware
    params:
      log_level: INFO
      include_metadata: true
      json_format: false  # Set true for log aggregation
```

### Custom Middleware

```yaml
middleware:
  - class: my_middleware.RateLimitMiddleware
    params:
      max_requests: 100
      window_seconds: 60

  - class: my_middleware.AuthMiddleware
    params:
      api_key: ${API_KEY}
```

### Middleware Interface

Custom middleware should extend the `Middleware` base class:

```python
from dataknobs_bots.middleware import Middleware
from dataknobs_bots import BotContext
from typing import Any

class MyMiddleware(Middleware):
    def __init__(self, **params):
        # Initialize with params
        pass

    async def before_message(self, message: str, context: BotContext) -> None:
        # Pre-processing
        pass

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        # Post-processing (kwargs includes tokens_used, provider, model)
        pass

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        # Error handling
        pass
```

For comprehensive middleware documentation, see the [Middleware Guide](middleware.md).

---

## Resource Resolution

The `dataknobs_bots.config` module provides utilities for resolving resources from environment configurations.

### BotResourceResolver

High-level resolver that automatically initializes resources:

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import BotResourceResolver

# Load environment
env = EnvironmentConfig.load()  # Auto-detects environment

# Create resolver
resolver = BotResourceResolver(env)

# Get initialized LLM (calls initialize() automatically)
llm = await resolver.get_llm("default")

# Get connected database (calls connect() automatically)
db = await resolver.get_database("conversations")

# Get initialized vector store (calls initialize() automatically)
vs = await resolver.get_vector_store("knowledge")

# Get initialized embedding provider
embedder = await resolver.get_embedding_provider("default")

# With config overrides
llm = await resolver.get_llm("default", temperature=0.9)

# Without caching (create new instance)
llm = await resolver.get_llm("default", use_cache=False)

# Clear cache
resolver.clear_cache()  # All resources
resolver.clear_cache("llm_providers")  # Specific type
```

### create_bot_resolver

Lower-level function for creating a ConfigBindingResolver with DynaBot factories:

```python
from dataknobs_config import EnvironmentConfig
from dataknobs_bots.config import create_bot_resolver

env = EnvironmentConfig.load("production")
resolver = create_bot_resolver(env)

# Resolve without auto-initialization
llm = resolver.resolve("llm_providers", "default")
await llm.initialize()  # Manual initialization

# Check registered factories
resolver.has_factory("llm_providers")  # True
resolver.get_registered_types()  # ['llm_providers', 'databases', ...]
```

### Individual Factory Registration

Register factories individually for custom resolvers:

```python
from dataknobs_config import ConfigBindingResolver, EnvironmentConfig
from dataknobs_bots.config import (
    register_llm_factory,
    register_database_factory,
    register_vector_store_factory,
    register_embedding_factory,
)

env = EnvironmentConfig.load()
resolver = ConfigBindingResolver(env)

# Register only what you need
register_llm_factory(resolver)
register_database_factory(resolver)

# Or use create_bot_resolver with register_defaults=False
from dataknobs_bots.config import create_bot_resolver
resolver = create_bot_resolver(env, register_defaults=False)
register_llm_factory(resolver)  # Register only LLM factory
```

### Factory Overview

| Resource Type | Factory | Creates |
|---------------|---------|---------|
| `llm_providers` | `LLMProviderFactory(is_async=True)` | Async LLM providers |
| `databases` | `AsyncDatabaseFactory()` | Database backends |
| `vector_stores` | `VectorStoreFactory()` | Vector store backends |
| `embedding_providers` | `LLMProviderFactory(is_async=True)` | Embedding providers |

---

## Environment Variables

### Using Environment Variables

Reference environment variables in configuration:

```yaml
llm:
  provider: openai
  api_key: ${OPENAI_API_KEY}

conversation_storage:
  backend: postgres
  password: ${DB_PASSWORD}

tools:
  - class: my_tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}
```

### Setting Environment Variables

**Shell:**
```bash
export OPENAI_API_KEY=sk-...
export DB_PASSWORD=secure-password
export WEATHER_API_KEY=abc123
```

**.env File:**
```bash
# .env
OPENAI_API_KEY=sk-...
DB_PASSWORD=secure-password
WEATHER_API_KEY=abc123
```

**Load with python-dotenv:**
```python
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create bot
bot = await DynaBot.from_config(config)
```

---

## Complete Examples

### Production Configuration

```yaml
# production_config.yaml

# LLM
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000

# Storage
conversation_storage:
  backend: postgres
  host: ${DB_HOST}
  port: 5432
  database: ${DB_NAME}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  pool_size: 20

# Memory
memory:
  type: buffer
  max_messages: 20

# Knowledge Base
knowledge_base:
  enabled: true
  documents_path: /app/docs
  vector_store:
    backend: pinecone
    api_key: ${PINECONE_API_KEY}
    environment: us-west1-gcp
    index_name: production-kb
    dimension: 1536
  embedding_provider: openai
  embedding_model: text-embedding-3-small
  chunking:
    max_chunk_size: 500
    chunk_overlap: 50

# Reasoning
reasoning:
  strategy: react
  max_iterations: 5
  verbose: false
  store_trace: false

# Tools
tool_definitions:
  weather:
    class: tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}

  calendar:
    class: tools.CalendarTool
    params:
      api_key: ${CALENDAR_API_KEY}

tools:
  - xref:tools[weather]
  - xref:tools[calendar]

# Prompts
prompts:
  customer_support: |
    You are a customer support AI assistant.
    Be helpful, friendly, and professional.
    Use the knowledge base to answer questions accurately.

system_prompt:
  name: customer_support

# Middleware
middleware:
  - class: middleware.LoggingMiddleware
    params:
      log_level: INFO

  - class: middleware.MetricsMiddleware
    params:
      export_endpoint: ${METRICS_ENDPOINT}
```

### Development Configuration

```yaml
# development_config.yaml

llm:
  provider: ollama
  model: gemma3:1b
  temperature: 0.7

conversation_storage:
  backend: memory

memory:
  type: buffer
  max_messages: 10

reasoning:
  strategy: react
  max_iterations: 5
  verbose: true
  store_trace: true

tools:
  - class: tools.CalculatorTool
    params:
      precision: 2

prompts:
  dev_assistant: "You are a development assistant. Be concise."

system_prompt:
  name: dev_assistant
```

---

## Configuration Validation

### Validation Best Practices

1. **Required Fields**: Ensure all required fields are present
2. **Type Checking**: Validate field types match expected types
3. **Value Ranges**: Check numeric values are within valid ranges
4. **Dependencies**: Verify dependent configurations are present

### Example Validation

```python
def validate_config(config: dict) -> None:
    """Validate configuration."""
    # Check required fields
    assert "llm" in config, "LLM configuration required"
    assert "conversation_storage" in config, "Storage configuration required"

    # Validate LLM
    llm = config["llm"]
    assert "provider" in llm, "LLM provider required"
    assert "model" in llm, "LLM model required"

    # Validate temperature range
    if "temperature" in llm:
        temp = llm["temperature"]
        assert 0.0 <= temp <= 1.0, "Temperature must be between 0.0 and 1.0"

    # Validate knowledge base dependencies
    if config.get("knowledge_base", {}).get("enabled"):
        kb = config["knowledge_base"]
        assert "vector_store" in kb, "Vector store required for knowledge base"
        assert "embedding_provider" in kb, "Embedding provider required for knowledge base"
        assert "embedding_model" in kb, "Embedding model required for knowledge base"
```

---

## See Also

- [Migration Guide](migration.md) - Migrate existing configs to environment-aware pattern
- [API Reference](../api/reference.md) - Complete API documentation
- [User Guide](user-guide.md) - Tutorials and how-to guides
- [Tools Development](tools.md) - Creating custom tools
- [Architecture](architecture.md) - System design
