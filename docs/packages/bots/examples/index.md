# Examples

Complete working examples demonstrating dataknobs-bots features.

## Available Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [Simple Chatbot](simple-chatbot.md) | Basic conversational bot | LLM integration, basic configuration |
| [Memory Chatbot](memory-chatbot.md) | Bot with conversation memory | Buffer memory, context management |
| [RAG Chatbot](rag-chatbot.md) | Knowledge base integration | Document loading, vector search, RAG |
| [ReAct Agent](react-agent.md) | Tool-using agent | Custom tools, ReAct reasoning |
| [Wizard Bot](wizard-bot.md) | Guided conversational wizard | FSM stages, validation, data collection |
| [Multi-Tenant Bot](multi-tenant.md) | Multiple clients setup | Bot registry, client isolation |
| [Config-Based Tools](custom-tools.md) | Configuration-driven tools | Tool configuration, XRef system |

## Running Examples

All examples use Ollama for local LLM inference. Follow these steps:

### 1. Install Ollama

Visit [https://ollama.ai/](https://ollama.ai/) and install Ollama for your platform.

### 2. Pull Required Models

```bash
# For basic chatbot examples
ollama pull gemma3:1b

# For agent examples
ollama pull phi3:mini

# For RAG examples (embedding model)
ollama pull nomic-embed-text
```

### 3. Install Dependencies

```bash
# Basic installation
pip install dataknobs-bots

# With FAISS for RAG examples
pip install dataknobs-bots[faiss]
```

### 4. Run an Example

```bash
# Clone the repository
git clone https://github.com/kbs-labs/dataknobs.git
cd dataknobs/packages/bots

# Run an example
python examples/01_simple_chatbot.py
```

## Example Categories

### Getting Started

Perfect for beginners learning dataknobs-bots:

- [Simple Chatbot](simple-chatbot.md) - Your first bot
- [Memory Chatbot](memory-chatbot.md) - Adding memory

### Advanced Features

For users building production bots:

- [RAG Chatbot](rag-chatbot.md) - Knowledge retrieval
- [ReAct Agent](react-agent.md) - Tool integration
- [Wizard Bot](wizard-bot.md) - Guided conversational flows
- [Multi-Tenant Bot](multi-tenant.md) - Multi-tenancy

### Configuration Patterns

Learn configuration best practices:

- [Config-Based Tools](custom-tools.md) - Tool configuration

## Prerequisites

### Required

- Python 3.10+
- dataknobs-bots package
- Ollama (for local LLM)

### Optional

- FAISS (`pip install faiss-cpu`) - For RAG examples
- PostgreSQL - For production storage examples

## Common Patterns

### Basic Bot Setup

```python
from dataknobs_bots import DynaBot, BotContext

config = {
    "llm": {"provider": "ollama", "model": "gemma3:1b"},
    "conversation_storage": {"backend": "memory"}
}

bot = await DynaBot.from_config(config)
context = BotContext(
    conversation_id="example-001",
    client_id="demo"
)

response = await bot.chat("Hello!", context)
```

### Adding Memory

```python
config = {
    "llm": {"provider": "ollama", "model": "gemma3:1b"},
    "conversation_storage": {"backend": "memory"},
    "memory": {
        "type": "buffer",
        "max_messages": 10
    }
}
```

### Enabling RAG

```python
config = {
    "llm": {"provider": "ollama", "model": "gemma3:1b"},
    "conversation_storage": {"backend": "memory"},
    "knowledge_base": {
        "enabled": True,
        "documents_path": "./docs",
        "vector_store": {"backend": "faiss", "dimension": 384}
    }
}
```

### Adding Tools

```python
config = {
    "llm": {"provider": "ollama", "model": "phi3:mini"},
    "conversation_storage": {"backend": "memory"},
    "reasoning": {"strategy": "react"},
    "tools": [
        {"class": "my_tools.CalculatorTool", "params": {}}
    ]
}
```

## Tips

1. **Start with Simple Chatbot** - Build foundation understanding
2. **Use Ollama** - No API keys needed for development
3. **Enable Verbose Mode** - Set `verbose: True` in reasoning config
4. **Check Logs** - Enable logging for debugging
5. **Read the Code** - All examples are well-commented

## Troubleshooting

### Model Not Found

```
Error: model 'gemma3:1b' not found
```

**Solution**: Pull the model first:
```bash
ollama pull gemma3:1b
```

### Connection Error

```
Error: Failed to connect to Ollama
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Import Error

```
ModuleNotFoundError: No module named 'dataknobs_bots'
```

**Solution**: Install the package:
```bash
pip install dataknobs-bots
```

## Next Steps

After running the examples:

1. Read the [User Guide](../guides/user-guide.md) for in-depth tutorials
2. Explore [Configuration Reference](../guides/configuration.md) for all options
3. Learn [Tool Development](../guides/tools.md) to create custom tools
4. Check [Architecture](../guides/architecture.md) for system design

## Source Code

All example source code is available in the GitHub repository:

[https://github.com/kbs-labs/dataknobs/tree/main/packages/bots/examples](https://github.com/kbs-labs/dataknobs/tree/main/packages/bots/examples)
