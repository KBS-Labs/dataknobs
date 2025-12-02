# DynaBot Examples

This directory contains working examples demonstrating various features of the DynaBot framework.

## Prerequisites

All examples use [Ollama](https://ollama.ai/) for local LLM inference. You'll need to:

1. **Install Ollama**: Download from https://ollama.ai/
2. **Pull required models**: Run the ollama pull commands below for each example you want to try

## Examples

### 1. Simple Chatbot (`01_simple_chatbot.py`)

A basic conversational bot with no memory or advanced features.

**Required Ollama Model:**
```bash
ollama pull gemma3:1b
```

**Features Demonstrated:**
- Basic bot configuration
- Echo LLM provider for testing
- In-memory conversation storage
- Simple message exchange

**Run:**
```bash
python examples/01_simple_chatbot.py
```

### 2. Chatbot with Memory (`02_chatbot_with_memory.py`)

A chatbot that remembers previous messages using buffer memory.

**Required Ollama Model:**
```bash
ollama pull gemma3:1b
```

**Features Demonstrated:**
- Buffer memory configuration
- Context retention across messages
- Memory limits and management

**Run:**
```bash
python examples/02_chatbot_with_memory.py
```

### 3. RAG Chatbot (`03_rag_chatbot.py`)

A chatbot with Retrieval Augmented Generation using a knowledge base.

**Required Ollama Model:**
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text  # For embeddings
```

**Features Demonstrated:**
- Knowledge base integration
- Document chunking and indexing
- Vector search and retrieval
- Context-aware responses
- Swapping storage backends (memory → postgres)

**Run:**
```bash
# With in-memory storage (default)
python examples/03_rag_chatbot.py

# With PostgreSQL storage (requires postgres)
STORAGE_BACKEND=postgres python examples/03_rag_chatbot.py
```

### 4. ReAct Agent (`04_react_agent.py`)

An agent that uses ReAct reasoning with tools to solve multi-step problems.

**Required Ollama Model:**
```bash
ollama pull gemma3:1b
```

**Features Demonstrated:**
- ReAct reasoning strategy
- Tool definition and registration
- Multi-step problem solving
- Reasoning trace storage
- Verbose logging

**Run:**
```bash
python examples/04_react_agent.py
```

### 5. Multi-Tenant Bot (`05_multi_tenant.py`)

Demonstrates how a single bot instance can serve multiple clients with isolated conversations.

**Required Ollama Model:**
```bash
ollama pull gemma3:1b
```

**Features Demonstrated:**
- Multi-tenant architecture
- Conversation isolation per client
- Shared bot configuration
- Concurrent conversation handling

**Run:**
```bash
python examples/05_multi_tenant.py
```

### 6. Config-Based Tool Loading (`06_config_based_tools.py`)

Demonstrates loading tools directly from configuration without manual instantiation.

**Required Ollama Model:**
```bash
ollama pull phi3:mini
```

**Features Demonstrated:**
- Loading tools via configuration
- Direct class instantiation with parameters
- XRef-based tool definitions
- Tool parameter customization
- Reusable tool configurations
- No manual tool registration needed

**Run:**
```bash
python examples/06_config_based_tools.py
```

**Key Concepts:**

This example shows two ways to configure tools:

1. **Direct Class Instantiation:**
```python
"tools": [
    {
        "class": "my_module.CalculatorTool",
        "params": {"precision": 3}
    }
]
```

2. **XRef to Predefined Tools:**
```python
"tool_definitions": {
    "my_calculator": {
        "class": "my_module.CalculatorTool",
        "params": {"precision": 5}
    }
},
"tools": [
    "xref:tools[my_calculator]"
]
```

Benefits:
- Tools are configuration-driven, not hardcoded
- Easy to swap tools without code changes
- Parameters can be customized per instance
- XRef allows tool definition reuse
- Supports nested XRef references

## Storage Backends

Most examples default to in-memory storage for simplicity. To use PostgreSQL:

1. **Start PostgreSQL** (using Docker):
```bash
docker run -d \
  --name postgres-dynabot \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=dynabot_examples \
  -p 5432:5432 \
  postgres:14
```

2. **Set environment variable**:
```bash
export STORAGE_BACKEND=postgres
```

3. **Run any example** - it will automatically use PostgreSQL

## Integration Tests

The examples are also tested via integration tests. To run them:

```bash
# Ensure Ollama is running with required models
ollama pull gemma3:1b
ollama pull nomic-embed-text

# Run integration tests
TEST_OLLAMA=true pytest tests/integration/
```

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:
- Ensure Ollama is running: `ollama list`
- Check the default port (11434) is accessible
- Try pulling the model again: `ollama pull gemma3:1b`

### PostgreSQL Connection Issues

If PostgreSQL examples fail:
- Check PostgreSQL is running: `docker ps | grep postgres`
- Verify connection: `psql -h localhost -U postgres -d dynabot_examples`
- Check environment variables: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

### Memory Issues

If you run out of memory:
- Use smaller Ollama models (gemma3:1b instead of larger models)
- Reduce `max_messages` in memory configuration
- Reduce `max_tokens` in LLM configuration

## Model Recommendations

| Model | Size | Use Case |
|-------|------|----------|
| gemma3:1b | ~1 GB | General chatbots, examples, testing |
| gemma3:4b | ~4 GB | Better quality responses |
| llama3.1:8b | 8 GB | Advanced reasoning, tool use |
| nomic-embed-text | ~270 MB | Embeddings for RAG |

## Next Steps

After trying the examples:
1. Modify configurations to experiment
2. Create your own tools for the ReAct agent
3. Add your own documents to the knowledge base
4. Implement custom memory strategies
5. Build your own middleware for logging/monitoring
