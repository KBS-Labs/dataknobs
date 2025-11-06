# Examples

Practical examples of using Dataknobs packages across different use cases.

## AI & LLM Examples

### Chatbots and AI Agents
- [Simple Chatbot](../packages/bots/examples/simple-chatbot.md) - Basic conversational bot
- [Memory-Enhanced Chatbot](../packages/bots/examples/memory-chatbot.md) - Bot with conversation memory
- [RAG Chatbot](../packages/bots/examples/rag-chatbot.md) - Retrieval-augmented generation bot
- [ReAct Agent](../packages/bots/examples/react-agent.md) - Reasoning and acting agent with tools
- [Multi-Tenant Bots](../packages/bots/examples/multi-tenant.md) - Managing multiple bot instances
- [Custom Tools](../packages/bots/examples/custom-tools.md) - Building custom tool integrations

### LLM Integration
- [Basic LLM Usage](../packages/llm/examples/basic-usage.md) - Getting started with LLMs
- [Advanced Prompting](../packages/llm/examples/advanced-prompting.md) - Prompt engineering patterns
- [Conversation Flows](../packages/llm/examples/conversation-flows.md) - Multi-turn conversations
- [FSM Conversation Flow](../packages/llm/examples/fsm-conversation-flow.md) - LLM workflows with FSM
- [A/B Testing Prompts](../packages/llm/examples/ab-testing.md) - Prompt version testing

## Workflow & Data Processing

### FSM Examples
- [File Processor](../packages/fsm/examples/file-processor.md) - Simple file processing pipeline
- [Data Pipeline](../packages/fsm/examples/data-pipeline.md) - ETL data processing workflow
- [Database ETL](../packages/fsm/examples/database-etl.md) - Database extract-transform-load
- [End-to-End Streaming](../packages/fsm/examples/end-to-end-streaming.md) - Streaming data processing
- [API Workflow Orchestration](../packages/fsm/examples/api-workflow.md) - API integration workflow
- [LLM Conversation Flow](../packages/fsm/examples/llm-conversation.md) - LLM-powered workflows
- [LLM Chain Processing](../packages/fsm/examples/llm-chain.md) - Chaining LLM operations
- [Regex Transformations](../packages/fsm/examples/regex-transformations.md) - Text transformation pipelines

### Data Backend Examples
- [Data Backends Overview](data-backends.md) - Using different storage backends
- [S3 Storage](s3-storage.md) - Working with S3 for data storage
- [Data Migration](data-migration.md) - Migrating between backends
- [Data Validation](data-validation.md) - Validating data with constraints
- [Pandas Workflow](pandas-workflow.md) - Integrating with pandas DataFrames
- [Sensor Dashboard](sensor-dashboard.md) - Real-time sensor data processing

## Configuration Examples
- [Database Configuration](database-config.md) - Multi-environment database setup
- [Service Configuration](service-config.md) - Configuring application services
- [Multi-Environment](multi-environment.md) - Managing dev/staging/prod configs
- [Configuration Patterns](configuration.md) - Advanced configuration patterns
- [Factory Pattern](factory-pattern.md) - Dynamic object creation

## Data Structures Examples
- [Basic Tree Operations](basic-tree.md) - Working with tree structures
- [Document Processing](document-processing.md) - Text and metadata handling

## Text Processing Examples
- [Text Normalization](text-normalization.md) - Normalizing and cleaning text
- [Markdown Chunking](markdown-chunking.md) - Chunking markdown documents

## Integration Examples
- [Elasticsearch Integration](elasticsearch-integration.md) - Working with Elasticsearch

## Complete Application Examples

### RAG Application with Multi-Backend Storage

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext
from dataknobs_data import database_factory
from dataknobs_config import Config

async def main():
    # Configuration for multi-backend setup
    config = Config({
        "databases": {
            "conversations": {
                "backend": "postgres",
                "connection": "postgresql://..."
            },
            "knowledge": {
                "backend": "elasticsearch",
                "host": "localhost:9200",
                "index": "documentation"
            }
        }
    })

    config.register_factory("database", database_factory)

    # Create databases
    conversations_db = config.get_instance("databases", "conversations")
    knowledge_db = config.get_instance("databases", "knowledge")

    # Configure RAG bot with multi-backend storage
    bot_config = {
        "llm": {"provider": "openai", "model": "gpt-4"},
        "conversation_storage": {"backend": "postgres", "connection": "postgresql://..."},
        "memory": {
            "type": "buffer",
            "max_messages": 20
        },
        "rag": {
            "enabled": True,
            "knowledge_base": knowledge_db,  # Elasticsearch for search
            "top_k": 5
        },
        "system_prompt": "Answer questions using documentation when available."
    }

    bot = await DynaBot.from_config(bot_config)

    # Use bot with persistent memory and knowledge retrieval
    context = BotContext(
        conversation_id="docs-session",
        client_id="my-app",
        user_id="user123"
    )
    response = await bot.chat(
        "How do I configure the database backend?",
        context
    )
    print(response)

asyncio.run(main())
```

### Data Pipeline with FSM and Multiple Backends

```python
from dataknobs_fsm import SimpleFSM, DataHandlingMode
from dataknobs_data import database_factory
from dataknobs_config import Config

# Multi-backend configuration
config = Config({
    "databases": {
        "source": {"backend": "postgres", "connection": "..."},
        "staging": {"backend": "memory"},
        "target": {"backend": "elasticsearch", "host": "..."}
    }
})

config.register_factory("database", database_factory)

# Create databases
source_db = config.get_instance("databases", "source")
staging_db = config.get_instance("databases", "staging")
target_db = config.get_instance("databases", "target")

# FSM pipeline configuration
fsm_config = {
    "name": "multi_backend_pipeline",
    "states": [
        {"name": "extract", "is_start": True},
        {"name": "stage"},
        {"name": "transform"},
        {"name": "load", "is_end": True}
    ],
    "arcs": [
        {
            "from": "extract",
            "to": "stage",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: ctx.resources['source'].search(...)"
            }
        },
        {
            "from": "stage",
            "to": "transform",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: ctx.resources['staging'].create(...)"
            }
        },
        {
            "from": "transform",
            "to": "load",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: ctx.resources['target'].create(...)"
            }
        }
    ]
}

fsm = SimpleFSM(fsm_config, data_mode=DataHandlingMode.COPY)
fsm.context["resources"] = {
    "source": source_db,
    "staging": staging_db,
    "target": target_db
}

result = fsm.process({"query": "SELECT * FROM users"})
```

### LLM-Powered Content Processing

```python
import asyncio
from dataknobs_fsm import SimpleFSM
from dataknobs_llm import create_llm_provider, LLMMessage
from dataknobs_data import database_factory, Record
from dataknobs_xization import normalize

async def main():
    # Initialize components
    llm = create_llm_provider({"provider": "openai", "model": "gpt-4"})
    s3_storage = database_factory.create({
        "backend": "s3",
        "bucket": "processed-content"
    })

    # Content processing pipeline
    pipeline_config = {
        "name": "content_processor",
        "states": [
            {"name": "load", "is_start": True},
            {"name": "normalize"},
            {"name": "summarize"},
            {"name": "tag"},
            {"name": "store", "is_end": True}
        ],
        "arcs": [
            {
                "from": "load",
                "to": "normalize",
                "transform": {
                    "type": "inline",
                    "code": "lambda data, ctx: {'text': normalize.basic_normalization_fn(data['text'])}"
                }
            },
            {
                "from": "normalize",
                "to": "summarize",
                "transform": {
                    "type": "inline",
                    "code": "lambda data, ctx: {'summary': 'Summary of text'}  # Async LLM call needed"
                }
            },
            {
                "from": "summarize",
                "to": "tag",
                "transform": {
                    "type": "inline",
                    "code": "lambda data, ctx: {'tags': ['tag1', 'tag2']}  # Async LLM call needed"
                }
            },
            {
                "from": "tag",
                "to": "store",
                "transform": {
                    "type": "inline",
                    "code": "lambda data, ctx: ctx['storage'].create(Record(data))"
                }
            }
        ]
    }

    fsm = SimpleFSM(pipeline_config)
    fsm.context["llm"] = llm
    fsm.context["storage"] = s3_storage

    # Process content
    result = fsm.process({"text": "Long article content..."})
    print(result)

asyncio.run(main())
```

## Example Categories

### By Use Case

**AI & Machine Learning:**
- [Bots Examples](../packages/bots/examples/index.md) - Chatbots and AI agents
- [FSM LLM Examples](../packages/fsm/examples/llm-conversation.md) - LLM workflows

**Data Engineering:**
- [FSM Data Pipelines](../packages/fsm/examples/data-pipeline.md) - ETL workflows
- [Database Examples](database-config.md) - Multi-backend data access
- [Streaming Examples](../packages/fsm/examples/end-to-end-streaming.md) - Real-time processing

**Application Development:**
- [Configuration Examples](configuration.md) - App configuration patterns
- [Data Structure Examples](basic-tree.md) - Trees and documents
- [Text Processing Examples](text-normalization.md) - Text utilities

### By Package

- **[Bots Examples](../packages/bots/examples/index.md)** - AI agents and chatbots
- **[FSM Examples](../packages/fsm/examples/index.md)** - Workflow orchestration
- **[Data Examples](#data-backend-examples)** - Backend abstraction
- **[Config Examples](#configuration-examples)** - Configuration management
- **[Structures Examples](#data-structures-examples)** - Trees and documents
- **[Xization Examples](#text-processing-examples)** - Text processing

## Next Steps

- Explore [Package Documentation](../packages/index.md) for detailed guides
- Read the [User Guide](../user-guide/index.md) for comprehensive tutorials
- Check [API Reference](../api/index.md) for complete API documentation
