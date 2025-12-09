# User Guide

Complete guide to using DataKnobs Bots with tutorials and how-to guides.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Tutorials](#basic-tutorials)
  - [Tutorial 1: Your First Chatbot](#tutorial-1-your-first-chatbot)
  - [Tutorial 2: Adding Memory](#tutorial-2-adding-memory)
  - [Tutorial 3: Building a RAG Chatbot](#tutorial-3-building-a-rag-chatbot)
  - [Tutorial 4: Creating Tool-Using Agents](#tutorial-4-creating-tool-using-agents)
- [Advanced Topics](#advanced-topics)
  - [Multi-Tenant Deployment](#multi-tenant-deployment)
  - [Custom Tools Development](#custom-tools-development)
  - [Production Deployment](#production-deployment)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Basic understanding of async/await in Python
- (Optional) Ollama installed for local LLM testing

### Installation

```bash
# Basic installation
pip install dataknobs-bots

# With PostgreSQL support
pip install dataknobs-bots[postgres]

# With all optional dependencies
pip install dataknobs-bots[all]
```

### Install Ollama (Optional, for Local Testing)

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull gemma3:1b
```

---

## Basic Tutorials

### Tutorial 1: Your First Chatbot

Build a simple conversational chatbot in 5 minutes.

#### Step 1: Create the Bot Configuration

```python
# first_bot.py
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Configuration
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "conversation_storage": {
            "backend": "memory"
        }
    }

    # Create bot
    bot = await DynaBot.from_config(config)
    print("✓ Bot created successfully!")

    # Create context
    context = BotContext(
        conversation_id="tutorial-1",
        client_id="my-app"
    )

    # Chat loop
    print("\nChat with the bot (type 'quit' to exit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = await bot.chat(user_input, context)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 2: Run the Bot

```bash
python first_bot.py
```

#### Step 3: Try It Out

```
You: Hello!
Bot: Hi there! How can I help you today?

You: What can you do?
Bot: I'm a conversational AI assistant. I can chat with you about various topics, answer questions, and help with tasks.

You: quit
```

#### What's Happening?

1. **Configuration**: Defines LLM (Ollama) and storage (in-memory)
2. **Bot Creation**: `from_config()` creates a configured bot
3. **Context**: Identifies the conversation
4. **Chat**: `bot.chat()` processes messages and returns responses

#### Adding a System Prompt

You can add a system prompt to customize the bot's behavior:

```python
config = {
    "llm": {
        "provider": "ollama",
        "model": "gemma3:1b",
    },
    "conversation_storage": {
        "backend": "memory"
    },
    # Add a system prompt (smart detection: if not in prompts library,
    # treated as inline content)
    "system_prompt": "You are a helpful coding assistant. Be concise and technical."
}
```

DynaBot uses **smart detection** for system prompts:
- If the string exists in the `prompts` library → used as a template reference
- If not → treated as inline content

#### Next Steps

- Try different models: `llama3.1:8b`, `phi3:mini`
- Adjust temperature (0.0 = focused, 1.0 = creative)
- Change max_tokens for longer/shorter responses
- See [CONFIGURATION.md](configuration.md) for all system prompt options

---

### Tutorial 2: Adding Memory

Add conversation memory so the bot remembers previous messages.

#### Step 1: Add Memory Configuration

```python
# memory_bot.py
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
        },
        "conversation_storage": {
            "backend": "memory"
        },
        # Add memory configuration
        "memory": {
            "type": "buffer",
            "max_messages": 10  # Remember last 10 messages
        }
    }

    bot = await DynaBot.from_config(config)

    context = BotContext(
        conversation_id="tutorial-2",
        client_id="my-app",
        user_id="user-123"
    )

    # Test memory
    print("Testing conversation memory:\n")

    response1 = await bot.chat("My name is Alice", context)
    print(f"Bot: {response1}\n")

    response2 = await bot.chat("What is my name?", context)
    print(f"Bot: {response2}\n")

    response3 = await bot.chat("Tell me about yourself", context)
    print(f"Bot: {response3}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 2: Run and Observe

```bash
python memory_bot.py
```

**Output:**
```
Testing conversation memory:

Bot: Nice to meet you, Alice! How can I help you today?

Bot: Your name is Alice!

Bot: I'm an AI assistant designed to have helpful, harmless conversations...
```

#### Understanding Memory Types

**Buffer Memory** (What we used):
- Keeps last N messages
- Fast and simple
- Good for most use cases

**Vector Memory** (For advanced use):
```python
"memory": {
    "type": "vector",
    "max_messages": 100,
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "backend": "faiss",
    "dimension": 384
}
```

---

### Tutorial 3: Building a RAG Chatbot

Create a chatbot that answers questions using your documents.

#### Step 1: Prepare Documents

```bash
# Create a docs directory
mkdir my_docs

# Add some documents
echo "Our company was founded in 2020 by Alice and Bob." > my_docs/about.txt
echo "We offer Premium ($99/month) and Enterprise ($299/month) plans." > my_docs/pricing.txt
echo "Email support@company.com for help or call 555-0123." > my_docs/contact.txt
```

#### Step 2: Create RAG Bot

```python
# rag_bot.py
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
        },
        "conversation_storage": {
            "backend": "memory"
        },
        # Enable knowledge base
        "knowledge_base": {
            "enabled": True,
            "documents_path": "./my_docs",
            "vector_store": {
                "backend": "faiss",
                "dimension": 384,
                "collection": "my_knowledge"
            },
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "chunking": {
                "max_chunk_size": 500,
                "chunk_overlap": 50
            }
        }
    }

    print("Creating RAG bot and indexing documents...")
    bot = await DynaBot.from_config(config)
    print("✓ Bot ready!\n")

    context = BotContext(
        conversation_id="tutorial-3",
        client_id="my-app"
    )

    # Ask questions about documents
    questions = [
        "When was the company founded?",
        "What are the pricing plans?",
        "How can I contact support?",
    ]

    for question in questions:
        print(f"Question: {question}")
        response = await bot.chat(question, context)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 3: Pull Required Model

```bash
ollama pull nomic-embed-text
```

#### Step 4: Run the RAG Bot

```bash
python rag_bot.py
```

**Output:**
```
Creating RAG bot and indexing documents...
✓ Bot ready!

Question: When was the company founded?
Answer: According to the documents, the company was founded in 2020 by Alice and Bob.

Question: What are the pricing plans?
Answer: We offer two pricing plans: Premium at $99/month and Enterprise at $299/month.

Question: How can I contact support?
Answer: You can email support@company.com or call 555-0123 for help.
```

#### How RAG Works

```
User Question
    ↓
1. Convert to embedding
    ↓
2. Search knowledge base
    ↓
3. Retrieve relevant chunks
    ↓
4. Add chunks to LLM context
    ↓
5. Generate answer with context
```

---

### Tutorial 4: Creating Tool-Using Agents

Build an agent that can use tools to perform actions.

#### Step 1: Define Custom Tools

```python
# tools.py
from dataknobs_llm.tools import Tool
from typing import Dict, Any

class CalculatorTool(Tool):
    """Tool for arithmetic operations."""

    def __init__(self, precision: int = 2):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic: add, subtract, multiply, divide"
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
                    "description": "Operation to perform"
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(self, operation: str, a: float, b: float) -> float:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float('inf')
        }
        result = operations[operation](a, b)
        return round(result, self.precision)


class WeatherTool(Tool):
    """Mock weather tool (in real use, call weather API)."""

    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a location"
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }

    async def execute(self, location: str) -> str:
        # Mock weather data (in real use, call API)
        weather_data = {
            "new york": "Sunny, 72°F",
            "london": "Cloudy, 15°C",
            "tokyo": "Rainy, 20°C"
        }
        location_lower = location.lower()
        return weather_data.get(location_lower, f"Weather data not available for {location}")
```

#### Step 2: Create Agent with Tools

```python
# agent.py
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def main():
    config = {
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",  # phi3 is good with tools
        },
        "conversation_storage": {
            "backend": "memory"
        },
        # Enable ReAct reasoning
        "reasoning": {
            "strategy": "react",
            "max_iterations": 5,
            "verbose": True,  # See reasoning steps
            "store_trace": True
        },
        # Configure tools
        "tools": [
            {
                "class": "tools.CalculatorTool",
                "params": {"precision": 2}
            },
            {
                "class": "tools.WeatherTool",
                "params": {}
            }
        ]
    }

    print("Creating agent with tools...\n")
    bot = await DynaBot.from_config(config)
    print("✓ Agent ready!\n")

    context = BotContext(
        conversation_id="tutorial-4",
        client_id="my-app"
    )

    # Tasks requiring tools
    tasks = [
        "What is 15 multiplied by 7?",
        "What's the weather in Tokyo?",
        "Calculate 100 divided by 4, then add 10 to that result"
    ]

    for task in tasks:
        print(f"Task: {task}\n")
        response = await bot.chat(task, context)
        print(f"Agent: {response}\n")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 3: Run the Agent

```bash
python agent.py
```

**Output:**
```
Creating agent with tools...
✓ Agent ready!

Task: What is 15 multiplied by 7?

[Iteration 1]
Thought: I need to multiply 15 by 7
Action: calculator
Action Input: {"operation": "multiply", "a": 15, "b": 7}
Observation: 105

[Iteration 2]
Thought: I have the answer
Agent: 15 multiplied by 7 is 105.

------------------------------------------------------------

Task: What's the weather in Tokyo?

[Iteration 1]
Thought: I need to check the weather
Action: get_weather
Action Input: {"location": "Tokyo"}
Observation: Rainy, 20°C

[Iteration 2]
Thought: I have the weather information
Agent: The weather in Tokyo is rainy with a temperature of 20°C.
```

#### Understanding ReAct

ReAct = **Rea**soning + **Act**ing

Each iteration:
1. **Thought**: What should I do?
2. **Action**: Which tool to use?
3. **Observation**: What did the tool return?
4. **Repeat or Answer**: Continue or provide final answer

---

## Advanced Topics

### Multi-Tenant Deployment

Deploy a single bot instance serving multiple clients.

```python
# multi_tenant_bot.py
import asyncio
from dataknobs_bots import DynaBot, BotContext

async def handle_client_request(
    bot: DynaBot,
    client_id: str,
    user_id: str,
    message: str
):
    """Handle request from a specific client."""
    context = BotContext(
        conversation_id=f"{client_id}-{user_id}",
        client_id=client_id,
        user_id=user_id,
        session_metadata={
            "client_name": f"Client {client_id}",
            "subscription": "premium"
        }
    )

    response = await bot.chat(message, context)
    return response


async def main():
    # Shared bot configuration
    config = {
        "llm": {"provider": "ollama", "model": "gemma3:1b"},
        "conversation_storage": {
            "backend": "postgres",  # Shared storage
            "host": "localhost",
            "database": "multi_tenant_db"
        },
        "memory": {"type": "buffer", "max_messages": 10}
    }

    bot = await DynaBot.from_config(config)

    # Simulate multiple clients
    tasks = [
        handle_client_request(bot, "client-A", "user-1", "Hello from Client A"),
        handle_client_request(bot, "client-B", "user-2", "Hello from Client B"),
        handle_client_request(bot, "client-A", "user-3", "Another user from A"),
    ]

    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

**Key Points**:
- Single bot instance
- Separate `client_id` for each tenant
- Conversations isolated by ID
- Shared storage with tenant partitioning

---

### Custom Tools Development

See [TOOLS.md](tools.md) for comprehensive guide.

**Quick Example**:

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any
import httpx

class StockPriceTool(Tool):
    """Get current stock price."""

    def __init__(self, api_key: str):
        super().__init__(
            name="get_stock_price",
            description="Get current stock price for a ticker symbol"
        )
        self.api_key = api_key

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                }
            },
            "required": ["ticker"]
        }

    async def execute(self, ticker: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.example.com/stock/{ticker}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            data = response.json()
            return {
                "ticker": ticker,
                "price": data["price"],
                "change": data["change"]
            }
```

**Usage**:
```python
config = {
    # ... other config
    "tools": [
        {
            "class": "my_tools.StockPriceTool",
            "params": {"api_key": "${STOCK_API_KEY}"}
        }
    ]
}
```

---

### Production Deployment

#### Configuration for Production

```yaml
# production.yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000

conversation_storage:
  backend: postgres
  host: ${DB_HOST}
  port: 5432
  database: ${DB_NAME}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  pool_size: 20
  max_overflow: 10

memory:
  type: buffer
  max_messages: 20

reasoning:
  strategy: react
  max_iterations: 5
  verbose: false
  store_trace: false

# Logging middleware
middleware:
  - class: middleware.RequestLoggingMiddleware
    params:
      log_level: INFO
  - class: middleware.MetricsMiddleware
    params:
      statsd_host: ${STATSD_HOST}
```

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: botdb
      POSTGRES_USER: botuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  bot:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DB_HOST=postgres
      - DB_NAME=botdb
      - DB_USER=botuser
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      - postgres
    ports:
      - "8000:8000"
    deploy:
      replicas: 3

volumes:
  postgres_data:
```

#### Health Checks

```python
# app.py
from fastapi import FastAPI
from dataknobs_bots import DynaBot

app = FastAPI()
bot = None

@app.on_event("startup")
async def startup():
    global bot
    bot = await DynaBot.from_config(config)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "bot_ready": bot is not None}

@app.post("/chat")
async def chat(request: ChatRequest):
    context = BotContext(
        conversation_id=request.conversation_id,
        client_id=request.client_id,
        user_id=request.user_id
    )
    response = await bot.chat(request.message, context)
    return {"response": response}
```

---

## Common Patterns

### Pattern 1: Configuration per Environment

```python
import os
import yaml

def load_config():
    env = os.getenv("ENV", "development")
    config_file = f"config/{env}.yaml"

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config

config = load_config()
bot = await DynaBot.from_config(config)
```

### Pattern 2: Dynamic Tool Loading

```python
config = {
    # ... base config
    "tool_definitions": {
        "calculator": {
            "class": "tools.CalculatorTool",
            "params": {"precision": 2}
        },
        "weather": {
            "class": "tools.WeatherTool",
            "params": {}
        }
    },
    "tools": []  # Empty initially
}

# Load tools based on user subscription
if user.has_feature("calculator"):
    config["tools"].append("xref:tools[calculator]")

if user.has_feature("weather"):
    config["tools"].append("xref:tools[weather]")

bot = await DynaBot.from_config(config)
```

### Pattern 3: Conversation Handoff

```python
async def escalate_to_human(conversation_id: str):
    """Transfer conversation to human agent."""
    # Get conversation history
    history = await storage.load_conversation(conversation_id)

    # Send to human agent system
    await human_agent_system.create_ticket(
        conversation_id=conversation_id,
        history=history,
        priority="high"
    )

    # Update conversation metadata
    await storage.update_metadata(
        conversation_id,
        {"status": "escalated", "escalated_at": datetime.now()}
    )
```

---

## Troubleshooting

### Issue: Bot responses are too slow

**Possible Causes**:
- Using a large LLM model
- Knowledge base search is slow
- Network latency to LLM API

**Solutions**:
```python
# Use a faster model
config["llm"]["model"] = "gemma3:1b"  # Instead of "llama3.1:70b"

# Reduce max_tokens
config["llm"]["max_tokens"] = 500  # Instead of 2000

# Use local LLM (Ollama)
config["llm"]["provider"] = "ollama"

# Optimize knowledge base
config["knowledge_base"]["chunking"]["max_chunk_size"] = 300  # Smaller chunks
```

### Issue: Out of memory errors

**Possible Causes**:
- Too many cached conversations
- Vector memory using too much RAM
- Large knowledge base in memory

**Solutions**:
```python
# Use buffer memory instead of vector
config["memory"] = {"type": "buffer", "max_messages": 10}

# Use external vector store
config["knowledge_base"]["vector_store"]["backend"] = "pinecone"

# Implement conversation cache eviction
# (Future feature)
```

### Issue: Knowledge base doesn't find relevant docs

**Possible Causes**:
- Poor chunking strategy
- Embeddings don't match query semantics
- Wrong similarity threshold

**Solutions**:
```python
# Adjust chunking
config["knowledge_base"]["chunking"] = {
    "max_chunk_size": 500,  # Larger chunks
    "chunk_overlap": 100    # More overlap
}

# Try different embedding model
config["knowledge_base"]["embedding_model"] = "text-embedding-3-large"

# Return more results
# In query: kb.query(query, k=10)  # Instead of k=3
```

### Issue: Tools not being called

**Possible Causes**:
- Tool description not clear
- Model not good at tool use
- Max iterations too low

**Solutions**:
```python
# Use a model better at tool use
config["llm"]["model"] = "phi3:mini"  # Or "gpt-4"

# Increase max iterations
config["reasoning"]["max_iterations"] = 10

# Improve tool descriptions
class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="VERY CLEAR description of what this tool does, when to use it, and what it returns"  # Be explicit!
        )
```

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

config["reasoning"]["verbose"] = True
config["reasoning"]["store_trace"] = True
```

---

## Next Steps

- **Explore Examples**: Check out [examples/](../examples/index.md) for more patterns
- **Read API Docs**: See [API.md](../api/reference.md) for complete API reference
- **Configuration Deep Dive**: Read [CONFIGURATION.md](configuration.md)
- **Build Custom Tools**: Follow [TOOLS.md](tools.md) guide
- **Understand Architecture**: Study [ARCHITECTURE.md](architecture.md)

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/kbs-labs/dataknobs/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/kbs-labs/dataknobs/discussions)
- **Documentation**: You're reading it!
- **Examples**: [Working code examples](../examples/index.md)
