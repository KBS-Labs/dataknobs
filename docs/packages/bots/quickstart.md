# Quick Start

This guide will help you get started with dataknobs-bots quickly by building your first AI chatbot.

## Installation

Install dataknobs-bots using pip:

```bash
pip install dataknobs-bots
```

For this quickstart, we'll use Ollama for local LLM inference:

```bash
# Install Ollama: https://ollama.ai/

# Pull the model
ollama pull gemma3:3b
```

## Your First Chatbot

Let's create a simple chatbot with memory:

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Step 1: Define configuration
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:3b",
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

    # Step 2: Create bot from configuration
    bot = await DynaBot.from_config(config)

    # Step 3: Create conversation context
    context = BotContext(
        conversation_id="quickstart-001",
        client_id="demo-client",
        user_id="user-123"
    )

    # Step 4: Start chatting
    messages = [
        "Hello! What can you help me with?",
        "My name is Alice. Can you remember it?",
        "What's my name?"
    ]

    for message in messages:
        print(f"User: {message}")
        response = await bot.chat(message, context)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `quickstart.py` and run it:

```bash
python quickstart.py
```

### What's Happening?

1. **Configuration** - We define the bot's behavior using a dictionary:
   - `llm`: Specifies the LLM provider (Ollama) and model (gemma3:3b)
   - `conversation_storage`: Where conversations are stored (memory)
   - `memory`: How the bot remembers context (buffer with 10 messages)

2. **Bot Creation** - `DynaBot.from_config(config)` creates a bot from the configuration

3. **Conversation Context** - `BotContext` encapsulates the conversation:
   - `conversation_id`: Unique identifier for this conversation
   - `client_id`: Tenant/application identifier
   - `user_id`: User identifier

4. **Chatting** - `bot.chat(message, context)` sends a message and gets a response

## Adding a Knowledge Base (RAG)

Let's enhance our bot with RAG (Retrieval Augmented Generation):

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext
from pathlib import Path

async def main():
    # Create a simple knowledge base
    docs_dir = Path("./my_docs")
    docs_dir.mkdir(exist_ok=True)

    # Create a sample document
    (docs_dir / "product_info.md").write_text("""
    # Product Information

    ## Features
    - Fast processing
    - Easy to use
    - Scalable architecture

    ## Pricing
    - Basic: $10/month
    - Pro: $50/month
    - Enterprise: Contact sales
    """)

    # Configuration with knowledge base
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:3b"
        },
        "conversation_storage": {
            "backend": "memory"
        },
        "knowledge_base": {
            "enabled": True,
            "documents_path": str(docs_dir),
            "vector_store": {
                "backend": "faiss",
                "dimension": 384
            },
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text"
        }
    }

    # Create bot
    bot = await DynaBot.from_config(config)

    # Create context
    context = BotContext(
        conversation_id="rag-quickstart",
        client_id="demo-client"
    )

    # Ask questions about the documents
    questions = [
        "What are the product features?",
        "How much does the Pro plan cost?"
    ]

    for question in questions:
        print(f"User: {question}")
        response = await bot.chat(question, context)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Before running, pull the embedding model:

```bash
ollama pull nomic-embed-text
pip install faiss-cpu  # Install FAISS
python rag_quickstart.py
```

### What Changed?

We added a `knowledge_base` section to the configuration:

- `enabled: True` - Enables RAG
- `documents_path` - Directory containing markdown documents
- `vector_store` - Vector database configuration (FAISS)
- `embedding` - Embedding model configuration

The bot now:
1. Loads and chunks markdown documents
2. Creates embeddings for each chunk
3. Stores embeddings in FAISS
4. Retrieves relevant chunks for each question
5. Injects retrieved context into prompts

## Creating a Tool-Using Agent

Let's create an agent that can use tools:

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext
from dataknobs_llm.tools import Tool
from typing import Dict, Any

# Step 1: Define a custom tool
class CalculatorTool(Tool):
    def __init__(self, precision: int = 2):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations"
        )
        self.precision = precision

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(
        self,
        operation: str,
        a: float,
        b: float,
        **kwargs
    ) -> float:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return round(result, self.precision)

# Step 2: Configure bot with tools and ReAct reasoning
async def main():
    config = {
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini"  # Better for tool use
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
                "class": "__main__.CalculatorTool",
                "params": {"precision": 2}
            }
        ]
    }

    # Create bot
    bot = await DynaBot.from_config(config)

    # Create context
    context = BotContext(
        conversation_id="agent-quickstart",
        client_id="demo-client"
    )

    # Ask questions that require calculation
    questions = [
        "What is 15 multiplied by 23?",
        "If I have $100 and spend $37.50, how much do I have left?"
    ]

    for question in questions:
        print(f"User: {question}")
        response = await bot.chat(question, context)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Before running, pull the model:

```bash
ollama pull phi3:mini
python agent_quickstart.py
```

### What's New?

We added:

1. **Custom Tool** - `CalculatorTool` implements the `Tool` interface:
   - `name` and `description` - Identifies the tool
   - `schema` - JSON schema defining parameters
   - `execute()` - Performs the actual calculation

2. **ReAct Reasoning** - Enables the Reasoning + Acting pattern:
   - `strategy: "react"` - Uses ReAct reasoning
   - `max_iterations: 5` - Maximum reasoning steps
   - `verbose: True` - Shows reasoning trace

3. **Tools Configuration** - Loads the tool from configuration:
   - `class` - Fully qualified class name
   - `params` - Constructor parameters

The agent now:
1. Receives a question
2. Reasons about how to answer it
3. Decides to use the calculator tool
4. Executes the tool
5. Uses the result to formulate an answer

## Multi-Tenant Setup

Here's how to set up a multi-tenant bot serving multiple clients:

```python
import asyncio
from dataknobs_bots import BotRegistry, BotContext

async def main():
    # Create bot registry with base configuration
    base_config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:3b"
        },
        "conversation_storage": {
            "backend": "memory"
        }
    }

    registry = BotRegistry(
        config=base_config,
        cache_ttl=300,  # Cache bots for 5 minutes
        max_cache_size=1000
    )

    # Register clients with custom configurations
    await registry.register_client(
        "client-a",
        {
            "memory": {"type": "buffer", "max_messages": 10},
            "prompts": {
                "system": "You are a helpful customer support assistant."
            }
        }
    )

    await registry.register_client(
        "client-b",
        {
            "memory": {"type": "buffer", "max_messages": 20},
            "prompts": {
                "system": "You are a technical expert."
            }
        }
    )

    # Get bots for different clients
    bot_a = await registry.get_bot("client-a")
    bot_b = await registry.get_bot("client-b")

    # Chat with client A's bot
    context_a = BotContext(
        conversation_id="conv-a-001",
        client_id="client-a"
    )
    response_a = await bot_a.chat("How can I reset my password?", context_a)
    print(f"Client A: {response_a}\n")

    # Chat with client B's bot
    context_b = BotContext(
        conversation_id="conv-b-001",
        client_id="client-b"
    )
    response_b = await bot_b.chat("Explain async/await in Python", context_b)
    print(f"Client B: {response_b}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Tenancy Benefits

- **Client Isolation** - Each client has separate conversations and configuration
- **Efficient Caching** - Bot instances are cached and reused
- **Centralized Management** - Single registry manages all bots
- **Horizontal Scaling** - Stateless design enables scaling across multiple servers

## Next Steps

Now that you've built your first bots, explore more advanced features:

- **[User Guide](guides/user-guide.md)** - Comprehensive tutorials and patterns
- **[Configuration Reference](guides/configuration.md)** - All configuration options
- **[Tools Development](guides/tools.md)** - Create custom tools
- **[Architecture](guides/architecture.md)** - System design and scaling
- **[Examples](examples/simple-chatbot.md)** - More complete examples

## Common Issues

### Model Not Found

```
Error: model 'gemma3:3b' not found
```

**Solution**: Pull the model first:
```bash
ollama pull gemma3:3b
```

### Import Error

```
ModuleNotFoundError: No module named 'dataknobs_bots'
```

**Solution**: Install the package:
```bash
pip install dataknobs-bots
```

### FAISS Not Found

```
ModuleNotFoundError: No module named 'faiss'
```

**Solution**: Install FAISS:
```bash
pip install faiss-cpu  # For CPU
pip install faiss-gpu  # For GPU
```

## Tips

1. **Start Simple** - Begin with a basic chatbot, then add features incrementally
2. **Use Ollama** - Great for development and testing without API costs
3. **Enable Verbose** - Set `verbose: True` in reasoning config to see what's happening
4. **Check Logs** - Enable logging to debug issues
5. **Read Examples** - The examples directory has complete working code

## Summary

You've learned how to:

- ✅ Create a basic chatbot with memory
- ✅ Add a knowledge base (RAG)
- ✅ Build a tool-using agent
- ✅ Set up multi-tenant bots

You're now ready to build sophisticated AI agents with dataknobs-bots!
