# Configuration Reference

Complete reference for configuring DynaBot instances.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [LLM Configuration](#llm-configuration)
- [Conversation Storage](#conversation-storage)
- [Memory Configuration](#memory-configuration)
- [Knowledge Base Configuration](#knowledge-base-configuration)
- [Reasoning Configuration](#reasoning-configuration)
- [Tools Configuration](#tools-configuration)
- [Prompts Configuration](#prompts-configuration)
- [Middleware Configuration](#middleware-configuration)
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

- [API Reference](../api/reference.md) - Complete API documentation
- [User Guide](user-guide.md) - Tutorials and how-to guides
- [Tools Development](tools.md) - Creating custom tools
- [Architecture](architecture.md) - System design
- [Examples](../examples/index.md) - Working configurations
