# Dataknobs Documentation

Welcome to **Dataknobs** - simple, standardized tools for working productively with knowledge and data.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Get up and running with Dataknobs in minutes

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

-   :material-package-variant:{ .lg .middle } __Modular Packages__

    ---

    Explore our modular architecture with specialized packages

    [:octicons-arrow-right-24: Package Overview](packages/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation with examples

    [:octicons-arrow-right-24: API Docs](api/index.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Learn best practices and advanced usage patterns

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

</div>

## What is Dataknobs?

Dataknobs provides simple, standardized implementations and interfaces for data structures, tools, and processes that enable effective and productive work with knowledge. Whether you're building AI applications, processing large datasets, orchestrating complex workflows, or creating intelligent chatbots, Dataknobs gives you the building blocks to work efficiently and responsibly.

**Core Capabilities:**

- **Configuration Management**: Flexible, environment-aware configuration with factory patterns
- **Data Abstraction**: Unified interface across Memory, File, PostgreSQL, Elasticsearch, and S3 backends
- **Workflow Orchestration**: Finite State Machines for building robust data processing pipelines
- **LLM Integration**: Prompt management, conversations, versioning, and tool calling
- **AI Agents**: Configuration-driven chatbots with memory, RAG, and reasoning capabilities
- **Data Structures**: Trees, documents, and record stores for organizing information
- **Text Processing**: Tokenization, normalization, and text analysis
- **Utilities**: JSON processing, file handling, and integration tools

**Our Mission:**

Dataknobs is open-source because we believe in **democratizing access to data** through useful tools that can be employed toward productive ends. We're committed to promoting **responsible and ethical use of technology and AI** by engineering safeguards into processes that work with data of all quantities and sizes.

## Package Overview

| Package | Description | Version |
|---------|-------------|---------|
| [dataknobs-bots](packages/bots/index.md) | Configuration-driven AI agents with RAG, memory, and reasoning strategies | 0.4.0 |
| [dataknobs-common](packages/common/index.md) | Foundation library with exceptions, registries, serialization, and event bus | 1.2.1 |
| [dataknobs-config](packages/config/index.md) | Modular configuration system with environment variable overrides and factories | 0.3.4 |
| [dataknobs-data](packages/data/index.md) | Unified data abstraction layer with multiple backends | 0.4.5 |
| [dataknobs-fsm](packages/fsm/index.md) | Finite State Machine framework for workflows with data modes and resource management | 0.1.6 |
| [dataknobs-llm](packages/llm/index.md) | Unified LLM abstraction with prompt management and conversations | 0.3.2 |
| [dataknobs-structures](packages/structures/index.md) | Core data structures for AI knowledge bases and document processing | 1.0.5 |
| [dataknobs-utils](packages/utils/index.md) | Utilities for file I/O, JSON processing, HTTP requests, and integrations | 1.2.3 |
| [dataknobs-xization](packages/xization/index.md) | Text normalization, tokenization, annotation, and markdown chunking library | 1.2.4 |
| [dataknobs](packages/legacy/index.md) | Legacy compatibility package (deprecated) | 0.1.1 |

## Quick Installation

=== "Core Packages"

    ```bash
    # Configuration and data abstraction
    pip install dataknobs-config dataknobs-data

    # AI and LLM capabilities
    pip install dataknobs-llm dataknobs-bots

    # Workflow orchestration
    pip install dataknobs-fsm
    ```

=== "Foundation Packages"

    ```bash
    # Data structures and utilities
    pip install dataknobs-structures dataknobs-utils dataknobs-xization
    ```

=== "Everything"

    ```bash
    # Install all packages
    pip install dataknobs-config dataknobs-data dataknobs-fsm \
                dataknobs-llm dataknobs-bots \
                dataknobs-structures dataknobs-utils dataknobs-xization
    ```

## Quick Examples

### Configuration-Driven Database

```python
from dataknobs_config import Config
from dataknobs_data import database_factory, Record, Query

# Load configuration with environment variables
config = Config("config.yaml")
config.register_factory("database", database_factory)

# Create database from config - supports PostgreSQL, Elasticsearch, S3, etc.
db = config.get_instance("databases", "primary")

# Unified API across all backends
record = Record({"name": "Alice", "role": "engineer"})
record_id = db.create(record)
results = db.search(Query().filter("role", "=", "engineer"))
```

### AI Chatbot with Memory

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Configure bot with memory and tools (from YAML/dict)
    bot_config = {
        "llm": {"provider": "openai", "model": "gpt-4"},
        "conversation_storage": {"backend": "memory"},
        "memory": {"type": "buffer", "max_messages": 10}
    }

    # Create bot
    bot = await DynaBot.from_config(bot_config)

    # Create context for conversation
    context = BotContext(
        conversation_id="conv-001",
        client_id="my-app",
        user_id="user123"
    )

    # Multi-turn conversation with context
    response1 = await bot.chat("What's the weather in Paris?", context)
    response2 = await bot.chat("How about tomorrow?", context)  # Remembers context

asyncio.run(main())
```

### Data Processing Workflow

```python
from dataknobs_fsm import SimpleFSM, DataHandlingMode

# Define workflow with inline transformations
pipeline_config = {
    "name": "etl_pipeline",
    "states": [
        {"name": "extract", "is_start": True},
        {"name": "transform"},
        {"name": "load", "is_end": True}
    ],
    "arcs": [
        {
            "from": "extract",
            "to": "transform",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: {'records': data['raw_data']}"
            }
        },
        {
            "from": "transform",
            "to": "load",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: [r.upper() for r in data['records']]"
            }
        }
    ]
}

fsm = SimpleFSM(pipeline_config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"raw_data": ["item1", "item2"]})
```

### Simple Data Structures

```python
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Hierarchical data organization
tree = Tree("root")
child = tree.add_child("child1")
child.add_child("grandchild")

# JSON navigation
data = {"users": {"alice": {"age": 30}}}
age = json_utils.get_value(data, "users.alice.age")

# Text normalization
text = "Hello WORLD!"
normalized = normalize.basic_normalization_fn(text)  # "hello world!"
```

## Use Cases

**For Data Engineers**: Build robust ETL pipelines with FSM, unified data access with the Data package, and flexible configuration management.

**For AI/ML Developers**: Integrate LLMs with prompt management, create intelligent chatbots with memory and RAG, and orchestrate complex AI workflows.

**For Application Developers**: Use simple data structures, text processing utilities, and standardized interfaces to build applications faster.

**For Researchers**: Access democratized tools for working with knowledge bases, experiment with different storage backends, and build reproducible workflows.

## Migration from Legacy

If you're using the old `dataknobs` package, see our [Migration Guide](migration-guide.md) for upgrading to the new modular structure.

## Contributing

We welcome contributions! Dataknobs is open-source to democratize access to productive data tools. See our [Contributing Guide](development/contributing.md) for details.

## License

Dataknobs is released under the MIT License. See [License](license.md) for details.
