# DataKnobs Bots

**Configuration-driven AI agents and chatbots for the DataKnobs ecosystem**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

DynaBot is a flexible, configuration-driven framework for building AI agents and chatbots. It provides a complete solution for multi-tenant AI deployments with features like memory management, knowledge retrieval (RAG), tool integration, and advanced reasoning strategies.

### Key Features

- **Configuration-First Design** - Define bot behavior entirely through YAML/JSON configuration
- **Multi-Tenant Architecture** - Single bot instance serves multiple clients with isolated conversations
- **Flexible Memory Systems** - Buffer, summary, and vector memory implementations
- **RAG Support** - Built-in knowledge base with document chunking and vector search
- **Tool Integration** - Load and configure tools from configuration without code changes
- **Reasoning Strategies** - Simple, Chain-of-Thought, and ReAct reasoning
- **Ecosystem Integration** - Seamlessly integrates with dataknobs-config, dataknobs-llm, dataknobs-data, and dataknobs-xization
- **Stateless Design** - Perfect for horizontal scaling in containerized environments
- **Production Ready** - PostgreSQL storage, error handling, and logging

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Configuration](#configuration)
- [Examples](#examples)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Installation

Install using pip or uv:

```bash
# Using pip
pip install dataknobs-bots

# Using uv
uv pip install dataknobs-bots
```

### Optional Dependencies

For specific features, install optional dependencies:

```bash
# PostgreSQL storage
pip install dataknobs-bots[postgres]

# Vector memory with FAISS
pip install dataknobs-bots[faiss]

# All optional dependencies
pip install dataknobs-bots[all]
```

## Quick Start

### Simple Chatbot

Create a basic chatbot with memory:

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Configuration
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "conversation_storage": {
            "backend": "memory"
        },
        "memory": {
            "type": "buffer",
            "max_messages": 10
        }
    }

    # Create bot from configuration
    bot = await DynaBot.from_config(config)

    # Create conversation context
    context = BotContext(
        conversation_id="conv-001",
        client_id="demo-client",
        user_id="user-123"
    )

    # Chat with the bot
    response = await bot.chat("Hello! What can you help me with?", context)
    print(f"Bot: {response}")

    response = await bot.chat("Tell me about yourself", context)
    print(f"Bot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### RAG Chatbot with Knowledge Base

Create a bot with knowledge retrieval:

```python
config = {
    "llm": {
        "provider": "ollama",
        "model": "gemma3:1b"
    },
    "conversation_storage": {
        "backend": "memory"
    },
    "knowledge_base": {
        "enabled": True,
        "documents_path": "./docs",
        "vector_store": {
            "backend": "faiss",
            "dimension": 384
        },
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text"
    }
}

bot = await DynaBot.from_config(config)
```

### ReAct Agent with Tools

Create an agent that can use tools:

```python
config = {
    "llm": {
        "provider": "ollama",
        "model": "phi3:mini"
    },
    "conversation_storage": {
        "backend": "memory"
    },
    "reasoning": {
        "strategy": "react",
        "max_iterations": 5,
        "verbose": True
    },
    "tools": [
        {
            "class": "my_tools.CalculatorTool",
            "params": {"precision": 2}
        }
    ]
}

bot = await DynaBot.from_config(config)
```

## Core Concepts

### DynaBot

The main bot class that orchestrates all components. Created from configuration and handles:
- Message processing
- Conversation management
- Memory integration
- Knowledge retrieval
- Tool execution
- Reasoning strategies

### BotContext

Encapsulates execution context for each bot interaction:
- `conversation_id` - Unique ID for the conversation
- `client_id` - Tenant/client identifier
- `user_id` - User identifier
- `session_metadata` - Additional metadata

### Memory Systems

Three types of memory for context management:

1. **Buffer Memory** - Simple sliding window of recent messages
2. **Summary Memory** - Compressed summaries of conversation history
3. **Vector Memory** - Semantic search over conversation history

### Knowledge Base (RAG)

Retrieval Augmented Generation support with:
- Document ingestion and chunking
- Vector embeddings
- Semantic search
- Context injection

### Reasoning Strategies

1. **Simple** - Direct LLM response
2. **Chain-of-Thought** - Step-by-step reasoning
3. **ReAct** - Reasoning + Acting with tools

### Tools

Tools extend bot capabilities with external functions. Loaded from configuration:

```python
"tools": [
    # Direct instantiation
    {
        "class": "my_tools.CalculatorTool",
        "params": {"precision": 3}
    },
    # XRef to predefined tool
    "xref:tools[my_calculator]"
]
```

## Configuration

DynaBot uses a configuration-first approach. All bot behavior is defined through configuration.

### Basic Configuration Structure

```yaml
# LLM Configuration
llm:
  provider: ollama
  model: gemma3:1b
  temperature: 0.7
  max_tokens: 1000

# Conversation Storage
conversation_storage:
  backend: memory  # or postgres

# Optional: Memory
memory:
  type: buffer
  max_messages: 10

# Optional: Knowledge Base
knowledge_base:
  enabled: true
  documents_path: ./docs
  vector_store:
    backend: faiss
    dimension: 384
  embedding_provider: ollama
  embedding_model: nomic-embed-text

# Optional: Reasoning
reasoning:
  strategy: react
  max_iterations: 5
  verbose: true

# Optional: Tools
tools:
  - class: my_tools.CalculatorTool
    params:
      precision: 2

# Optional: System Prompt
prompts:
  helpful_assistant: "You are a helpful AI assistant."

system_prompt:
  name: helpful_assistant
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete configuration reference.

## Examples

The `examples/` directory contains working examples demonstrating various features:

1. **Simple Chatbot** (`01_simple_chatbot.py`) - Basic conversational bot
2. **Chatbot with Memory** (`02_chatbot_with_memory.py`) - Buffer memory for context
3. **RAG Chatbot** (`03_rag_chatbot.py`) - Knowledge base integration
4. **ReAct Agent** (`04_react_agent.py`) - Tool-using agent with reasoning
5. **Multi-Tenant Bot** (`05_multi_tenant.py`) - Multiple clients, isolated conversations
6. **Config-Based Tools** (`06_config_based_tools.py`) - Configuration-driven tool loading

### Running Examples

All examples use Ollama for local LLM inference:

```bash
# Install Ollama: https://ollama.ai/

# Pull required models
ollama pull gemma3:1b
ollama pull phi3:mini
ollama pull nomic-embed-text

# Run an example
python examples/01_simple_chatbot.py
```

See [examples/README.md](examples/README.md) for detailed information on each example.

## Documentation

### User Documentation

- [User Guide](docs/USER_GUIDE.md) - Tutorials and how-to guides
- [Configuration Reference](docs/CONFIGURATION.md) - Complete configuration options
- [Tools Development](docs/TOOLS.md) - Creating and configuring tools

### Developer Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Examples](examples/README.md) - Working code examples

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/kbs-labs/dataknobs.git
cd dataknobs/packages/bots

# Install dependencies with development extras
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dataknobs_bots --cov-report=html

# Run specific test file
pytest tests/unit/test_dynabot.py

# Run integration tests (requires Ollama)
TEST_OLLAMA=true pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
packages/bots/
├── src/dataknobs_bots/
│   ├── bot/              # Core bot implementation
│   │   ├── base.py       # DynaBot class
│   │   ├── context.py    # BotContext class
│   │   └── registry.py   # Bot registry
│   ├── memory/           # Memory implementations
│   │   ├── base.py       # Memory interface
│   │   ├── buffer.py     # Buffer memory
│   │   └── vector.py     # Vector memory
│   ├── knowledge/        # RAG implementation
│   │   └── rag.py        # Knowledge base
│   ├── reasoning/        # Reasoning strategies
│   │   ├── base.py       # Reasoning interface
│   │   ├── simple.py     # Simple reasoning
│   │   └── react.py      # ReAct reasoning
│   ├── tools/            # Built-in tools
│   │   └── knowledge_search.py
│   └── utils/            # Utilities
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
├── examples/             # Working examples
├── docs/                 # Documentation
└── README.md             # This file
```

## Use Cases

### Customer Support Bot

Multi-tenant bot with knowledge base for customer support:

```python
config = {
    "llm": {"provider": "openai", "model": "gpt-4"},
    "conversation_storage": {"backend": "postgres"},
    "memory": {"type": "buffer", "max_messages": 20},
    "knowledge_base": {
        "enabled": True,
        "documents_path": "./support_docs"
    }
}
```

### Personal Assistant

Agent with tools for task automation:

```python
config = {
    "llm": {"provider": "anthropic", "model": "claude-3-sonnet"},
    "reasoning": {"strategy": "react"},
    "tools": [
        {"class": "tools.CalendarTool", "params": {}},
        {"class": "tools.EmailTool", "params": {}},
        {"class": "tools.WeatherTool", "params": {}}
    ]
}
```

### Document Q&A

RAG-powered document question answering:

```python
config = {
    "llm": {"provider": "ollama", "model": "llama3.1:8b"},
    "knowledge_base": {
        "enabled": True,
        "documents_path": "./company_docs",
        "chunking": {"max_chunk_size": 500, "chunk_overlap": 50}
    }
}
```

## Performance Considerations

### Scaling

- **Stateless Design** - Each request is independent, enabling horizontal scaling
- **Connection Pooling** - PostgreSQL connection pooling for high concurrency
- **Caching** - In-memory conversation caching reduces database queries
- **Async/Await** - Fully asynchronous for high throughput

### Optimization Tips

1. **Use Buffer Memory** - Faster than vector memory for most use cases
2. **Limit Memory Window** - Keep `max_messages` reasonable (10-20)
3. **PostgreSQL for Production** - Don't use in-memory storage in production
4. **Batch Knowledge Base Indexing** - Index documents offline
5. **Use Local LLMs** - Ollama for reduced latency and cost

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/kbs-labs/dataknobs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kbs-labs/dataknobs/discussions)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

## Acknowledgments

Built on the DataKnobs ecosystem:
- [dataknobs-config](../config/) - Configuration management
- [dataknobs-llm](../llm/) - LLM providers and tools
- [dataknobs-data](../data/) - Data storage backends
- [dataknobs-xization](../xization/) - Configuration resolution

## Roadmap

- [ ] Streaming responses
- [ ] Multi-modal support (images, audio)
- [ ] Advanced memory strategies (hybrid, hierarchical)
- [ ] Tool marketplace
- [ ] Web UI for bot management
- [ ] Performance monitoring and analytics
- [ ] A/B testing framework
- [ ] Voice interface support

---

Made with ❤️ by the DataKnobs team
