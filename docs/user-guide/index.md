# User Guide

Welcome to the Dataknobs User Guide. This section provides detailed information on using Dataknobs effectively for building knowledge-centric applications.

## Getting Started
- [Quick Start](quickstart.md) - Get up and running quickly
- [Basic Usage](basic-usage.md) - Common patterns and examples
- [Advanced Usage](advanced-usage.md) - Advanced features and techniques
- [Best Practices](best-practices.md) - Recommended patterns

## Core Capabilities

### AI Agents & Chatbots
**Package**: [dataknobs-bots](../packages/bots/index.md)

Configuration-driven AI agents with memory, RAG, reasoning strategies, and multi-tenancy.

- [User Guide](../packages/bots/guides/user-guide.md) - Comprehensive tutorials
- [Configuration](../packages/bots/guides/configuration.md) - Bot configuration options
- [Architecture](../packages/bots/guides/architecture.md) - Understanding bot components
- [Tools](../packages/bots/guides/tools.md) - Built-in and custom tools

**Use Cases**: Customer support bots, virtual assistants, knowledge base Q&A, multi-user chat systems

### Configuration Management
**Package**: [dataknobs-config](../packages/config/index.md)

Flexible configuration with environment variable support, factory patterns, and cross-references.

- [Configuration System](../packages/config/configuration-system.md) - Understanding the configuration architecture
- [Environment Variables](../packages/config/environment-variables.md) - Using environment-based configuration
- [Factory Registration](../packages/config/factory-registration.md) - Dynamic object creation

**Use Cases**: Multi-environment deployments, dynamic backend selection, application configuration

### Data Abstraction
**Package**: [dataknobs-data](../packages/data/index.md)

Unified interface across Memory, File, PostgreSQL, Elasticsearch, and S3 backends with transactions and migrations.

- [Record Model](../packages/data/record-model.md) - Working with records
- [Query Interface](../packages/data/query.md) - Building queries
- [Backends](../packages/data/backends.md) - Choosing and configuring backends
- [Async Pooling](../packages/data/async-pooling.md) - High-performance async operations
- [Pandas Integration](../packages/data/pandas-integration.md) - DataFrame workflows

**Use Cases**: Backend-agnostic data access, ETL pipelines, multi-backend applications, data migration

### Workflow Orchestration
**Package**: [dataknobs-fsm](../packages/fsm/index.md)

Finite State Machine framework for building robust, testable data processing pipelines.

- [FSM Basics](../packages/fsm/quickstart.md) - Introduction to the FSM framework
- [Data Handling Modes](../packages/fsm/guides/data-modes.md) - Understanding COPY, REFERENCE, and DIRECT modes
- [Resources](../packages/fsm/guides/resources.md) - Built-in resource managers (DB, HTTP, LLM, files)
- [Streaming Workflows](../packages/fsm/guides/streaming.md) - Building streaming data pipelines
- [Configuration](../packages/fsm/guides/configuration.md) - YAML/JSON-based workflow definitions
- [Debugging FSMs](../packages/fsm/guides/cli.md) - Using AdvancedFSM for debugging

**Use Cases**: ETL pipelines, data validation, multi-step processing, workflow automation

### LLM Integration
**Package**: [dataknobs-llm](../packages/llm/index.md)

Multi-provider LLM integration with prompt management, conversations, versioning, and tool calling.

- [Prompts](../packages/llm/guides/prompts.md) - Prompt template management and versioning
- [Conversations](../packages/llm/guides/conversations.md) - Multi-turn conversation handling
- [Flows](../packages/llm/guides/flows.md) - Complex LLM workflows
- [Tools and Enhancements](../packages/llm/guides/tools-and-enhancements.md) - Function calling and tool use
- [Performance](../packages/llm/guides/performance.md) - Optimization and cost tracking

**Use Cases**: Chatbots, content generation, code analysis, document summarization, Q&A systems

### Data Structures
**Package**: [dataknobs-structures](../packages/structures/index.md)

Core data structures for organizing knowledge: trees, documents, record stores, conditional dictionaries.

- [Tree Structures](../packages/structures/tree.md) - Hierarchical data organization
- [Documents](../packages/structures/document.md) - Text and metadata handling
- [Record Stores](../packages/structures/record-store.md) - Simple key-value storage
- [Conditional Dictionaries](../packages/structures/conditional-dict.md) - Filtered dictionaries

**Use Cases**: Hierarchical data, document management, knowledge graphs, data organization

### Utilities
**Package**: [dataknobs-utils](../packages/utils/index.md)

Utility functions for JSON manipulation, file operations, HTTP requests, and more.

- [JSON Utils](../packages/utils/json-utils.md) - JSON navigation and manipulation
- [File Utils](../packages/utils/file-utils.md) - File I/O operations
- [Elasticsearch](../packages/utils/elasticsearch.md) - Elasticsearch helpers
- [LLM Utils](../packages/utils/llm-utils.md) - LLM-related utilities

**Use Cases**: JSON processing, file handling, search integration, API interactions

### Text Processing
**Package**: [dataknobs-xization](../packages/xization/index.md)

Text normalization, tokenization, masking, and lexical analysis for NLP and data processing.

- [Tokenization](../packages/xization/tokenization.md) - Text tokenization strategies
- [Normalization](../packages/xization/normalization.md) - Text normalization functions
- [Masking](../packages/xization/masking.md) - PII and sensitive data masking

**Use Cases**: Data anonymization, text preprocessing, NLP pipelines, search indexing

## Common Workflows

### Building a Data Pipeline

Combine FSM, Data, and Config packages:

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_data import database_factory
from dataknobs_config import Config

# Load configuration
config = Config("pipeline.yaml")
config.register_factory("database", database_factory)

# Access database through config
source_db = config.get_instance("databases", "source")
target_db = config.get_instance("databases", "target")

# Define FSM workflow
fsm = SimpleFSM({
    "states": [...],
    "arcs": [...]
})

# Process with database access
fsm.context["source"] = source_db
fsm.context["target"] = target_db
result = fsm.process(data)
```

### Building an AI Chatbot

Combine Bots, LLM, and Data packages:

```python
from dataknobs_bots import BotRegistry
from dataknobs_data import PostgresDatabase

# Persistent storage for conversations
db = PostgresDatabase(connection_string="...")

# Configure bot with memory and RAG
bot_config = {
    "llm": {"provider": "openai", "model": "gpt-4"},
    "memory": {"type": "vector", "db": db},
    "knowledge_base": {"type": "elasticsearch", "index": "docs"}
}

registry = BotRegistry()
bot = registry.create_bot("support", bot_config)

# Multi-session conversations with persistence
response = bot.chat("How do I reset my password?", session_id="user123")
```

### Processing Text at Scale

Combine FSM, Data, and Xization packages:

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_data import S3Database
from dataknobs_xization import normalize

# Read from S3, process, write back
s3_db = S3Database(bucket="documents")

fsm_config = {
    "name": "text_processor",
    "states": [
        {"name": "load", "is_start": True},
        {"name": "normalize"},
        {"name": "save", "is_end": True}
    ],
    "arcs": [
        {
            "from": "load",
            "to": "normalize",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: normalize.basic_normalization_fn(data['text'])"
            }
        }
    ]
}

fsm = SimpleFSM(fsm_config)
```

## Learning Path

**Beginners** - Start Here:
1. [Quick Start](quickstart.md) - Get familiar with basic concepts
2. [Basic Usage](basic-usage.md) - Learn core data structures and utilities
3. [Examples](../examples/index.md) - See practical applications

**Intermediate** - Build Applications:
1. [Configuration System](../packages/config/index.md) - Environment management
2. [Data Abstraction](../packages/data/index.md) - Backend-agnostic data access
3. [FSM Workflows](../packages/fsm/quickstart.md) - Build robust pipelines
4. [Advanced Usage](advanced-usage.md) - Advanced patterns

**Advanced** - AI & Complex Systems:
1. [LLM Integration](../packages/llm/quickstart.md) - Integrate language models
2. [AI Agents](../packages/bots/quickstart.md) - Build intelligent chatbots
3. [Streaming Workflows](../packages/fsm/guides/streaming.md) - Real-time processing
4. [Production Best Practices](best-practices.md) - Deploy at scale

## Package Integration

Dataknobs packages are designed to work together seamlessly:

- **Config** → **Data**: Dynamic backend configuration
- **Data** → **FSM**: Database access in workflows
- **LLM** → **Bots**: LLM integration in AI agents
- **Bots** → **Data**: Persistent conversation memory
- **FSM** → **LLM**: LLM calls in workflow states
- **Utils** → **Everything**: Common utilities across all packages

## Additional Resources

- [API Reference](../api/index.md) - Complete API documentation
- [Examples](../examples/index.md) - Real-world usage examples
- [Development Guide](../development/index.md) - Contributing and extending
- [Migration Guide](../migration-guide.md) - Upgrading from legacy versions
