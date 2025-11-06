# Getting Started with Dataknobs

This guide will help you get up and running with Dataknobs based on your use case.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

### Choose Based on Your Use Case

=== "AI Applications"

    ```bash
    # For building chatbots and AI agents
    pip install dataknobs-bots dataknobs-llm

    # Add data persistence if needed
    pip install dataknobs-data dataknobs-config
    ```

=== "Data Processing"

    ```bash
    # For ETL and workflow orchestration
    pip install dataknobs-fsm dataknobs-data dataknobs-config
    ```

=== "General Development"

    ```bash
    # Core utilities and data structures
    pip install dataknobs-structures dataknobs-utils dataknobs-xization
    ```

=== "Everything"

    ```bash
    # Install all packages
    pip install dataknobs-config dataknobs-data dataknobs-fsm \
                dataknobs-llm dataknobs-bots \
                dataknobs-structures dataknobs-utils dataknobs-xization
    ```

### Installing with uv

If you're using the `uv` package manager:

```bash
uv pip install dataknobs-bots dataknobs-llm  # Or any combination
```

### Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/kbs-labs/dataknobs.git
cd dataknobs
uv sync --all-packages
```

## Quick Start by Use Case

### Building AI Chatbots

Create an intelligent chatbot with memory and tools:

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Configure bot from dictionary or YAML
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        },
        "conversation_storage": {"backend": "memory"},
        "memory": {
            "type": "buffer",
            "max_messages": 10
        },
        "prompts": {
            "support_assistant": "You are a helpful customer support assistant."
        },
        "system_prompt": {"name": "support_assistant"}
    }

    # Create bot
    bot = await DynaBot.from_config(config)

    # Create context for conversation
    context = BotContext(
        conversation_id="support-001",
        client_id="my-company",
        user_id="customer123"
    )

    # Chat with context retention
    response = await bot.chat("I need help with my order", context)
    print(response)

asyncio.run(main())
```

[Learn more about Bots →](packages/bots/quickstart.md)

### Processing Data with Workflows

Build robust ETL pipelines with finite state machines:

```python
from dataknobs_fsm import SimpleFSM, DataHandlingMode

# Define a data processing pipeline
config = {
    "name": "user_import",
    "states": [
        {"name": "load", "is_start": True},
        {"name": "validate"},
        {"name": "transform"},
        {"name": "save", "is_end": True}
    ],
    "arcs": [
        {"from": "load", "to": "validate"},
        {"from": "validate", "to": "transform"},
        {"from": "transform", "to": "save"}
    ]
}

fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"users": [{"name": "Alice", "age": 30}]})
```

[Learn more about FSM →](packages/fsm/quickstart.md)

### Working with Multiple Data Backends

Use a unified interface across different storage systems:

```python
from dataknobs_config import Config
from dataknobs_data import database_factory, Record, Query

# Load configuration (supports environment variables)
config = Config("config.yaml")  # or dict
config.register_factory("database", database_factory)

# Get database instance - backend determined by config
# Supports: Memory, File, PostgreSQL, Elasticsearch, S3
db = config.get_instance("databases", "primary")

# Unified API regardless of backend
record = Record({"name": "Alice", "email": "alice@example.com"})
record_id = db.create(record)

# Query with same API across all backends
results = db.search(Query().filter("name", "=", "Alice"))
```

[Learn more about Data →](packages/data/index.md) | [Learn more about Config →](packages/config/index.md)

### Integrating LLMs

Manage prompts and multi-provider LLM access:

```python
from dataknobs_llm import create_llm_provider, LLMMessage

# Create LLM provider
llm = create_llm_provider({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-key"
})

# Generate completion
messages = [
    LLMMessage(role="user", content="What's the capital of France?")
]
response = await llm.generate(messages)
print(response.content)

# Continue conversation
messages.append(LLMMessage(role="assistant", content=response.content))
messages.append(LLMMessage(role="user", content="What's its population?"))
response = await llm.generate(messages)  # Maintains context
```

[Learn more about LLM →](packages/llm/quickstart.md)

### Working with Data Structures

Use trees, documents, and utilities for common tasks:

```python
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Hierarchical data
tree = Tree("root")
chapter1 = tree.add_child("Chapter 1")
chapter1.add_child("Section 1.1")
chapter1.add_child("Section 1.2")

# Navigate tree
for node in tree.traverse():
    print(f"{'  ' * node.level}{node.value}")

# JSON utilities
data = {"users": {"alice": {"age": 30, "city": "Paris"}}}
age = json_utils.get_value(data, "users.alice.age")  # 30

# Text normalization
text = "  Hello   WORLD!!!  "
normalized = normalize.basic_normalization_fn(text)  # "hello world!"
```

[Learn more about Structures →](packages/structures/index.md) | [Learn more about Utils →](packages/utils/index.md)

## Understanding the Package Ecosystem

Dataknobs packages are organized by capability:

**Configuration & Data Layer** (Foundation for other packages):
- `dataknobs-config`: Environment-aware configuration management
- `dataknobs-data`: Unified data access across multiple backends

**AI & LLM Capabilities** (Building intelligent applications):
- `dataknobs-llm`: LLM integration with prompt management
- `dataknobs-bots`: Pre-built AI agents with memory and tools

**Workflow & Processing** (Orchestrating complex operations):
- `dataknobs-fsm`: Finite state machines for robust pipelines

**Core Utilities** (Building blocks):
- `dataknobs-structures`: Trees, documents, record stores
- `dataknobs-utils`: JSON, file operations, integrations
- `dataknobs-xization`: Text normalization and tokenization
- `dataknobs-common`: Shared base classes

## Next Steps

Based on what you want to build:

**For AI/ML Projects:**
1. Start with [Bots Quickstart](packages/bots/quickstart.md) for chatbots
2. Or [LLM Quickstart](packages/llm/quickstart.md) for custom LLM integration
3. Add [Data package](packages/data/index.md) for persistence
4. Use [FSM](packages/fsm/quickstart.md) for complex AI workflows

**For Data Engineering:**
1. Begin with [FSM Quickstart](packages/fsm/quickstart.md)
2. Add [Data package](packages/data/index.md) for backend abstraction
3. Use [Config](packages/config/index.md) for environment management
4. Explore [Data Examples](examples/index.md#data-backend-examples)

**For General Development:**
1. Check [Basic Usage Guide](user-guide/basic-usage.md) for structures and utilities
2. Explore [Advanced Usage](user-guide/advanced-usage.md) for patterns
3. Browse [Examples](examples/index.md) for real-world use cases

## Getting Help

- **Documentation**: Comprehensive guides in the [User Guide](user-guide/index.md)
- **API Reference**: Detailed [API documentation](api/index.md) for each package
- **Examples**: Real-world [usage examples](examples/index.md)
- **GitHub Issues**: Report bugs or request features at [GitHub](https://github.com/kbs-labs/dataknobs/issues)

## Common Patterns

### Combining Packages

Packages work seamlessly together:

```python
# FSM + Data + Config: Robust data pipeline
from dataknobs_fsm import SimpleFSM
from dataknobs_data import database_factory
from dataknobs_config import Config

config = Config("pipeline.yaml")
config.register_factory("database", database_factory)

# FSM can access database through config
fsm = SimpleFSM(pipeline_config)
fsm.context["db"] = config.get_instance("databases", "primary")

# Bots + LLM + Data: Chatbot with persistence
from dataknobs_bots import BotRegistry
from dataknobs_data import MemoryDatabase

registry = BotRegistry()
db = MemoryDatabase()  # For conversation history

bot = registry.create_bot("assistant", {
    "llm": {"provider": "openai"},
    "memory": {"type": "buffer"}
})
```

### Environment-Based Configuration

All packages support environment variables through Config:

```yaml
# config.yaml
databases:
  primary:
    backend: ${DB_BACKEND:memory}  # Default to memory
    connection: ${DB_CONNECTION:}

llm:
  provider: ${LLM_PROVIDER:openai}
  api_key: ${OPENAI_API_KEY}
```

```python
from dataknobs_config import Config

# Reads from environment or uses defaults
config = Config("config.yaml")
```

## Troubleshooting

### Import Errors

Use the new package names with underscores:

```python
# ✅ Correct
from dataknobs_structures import Tree
from dataknobs_bots import BotRegistry

# ❌ Old style (deprecated)
from dataknobs.structures import Tree
```

### Missing Dependencies

Some packages require additional dependencies:

```bash
# For PostgreSQL support
pip install psycopg2-binary

# For Elasticsearch
pip install elasticsearch

# For S3 support
pip install boto3

# For LLM providers
pip install openai anthropic
```

See the [Installation Guide](installation.md) for complete dependency information.
