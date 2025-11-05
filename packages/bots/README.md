# DataKnobs Bots

Configuration-driven AI agents for the DataKnobs ecosystem.

## Overview

DynaBot is a dynamic, configuration-driven chatbot/agent framework that enables multi-tenant AI agent deployment. A single stateless bot instance can serve as any agent based on runtime configuration.

## Features

- **Configuration-First**: All bot behavior defined in YAML/JSON configuration
- **Multi-Tenant Ready**: Isolated execution per client with secure partitioning
- **Ecosystem Integration**: Leverages dataknobs-config, dataknobs-llm, dataknobs-data, dataknobs-xization, and dataknobs-fsm
- **Flexible Memory**: Buffer, Summary, and Vector memory implementations
- **RAG Support**: Built-in knowledge base with document chunking and vector search
- **Reasoning Strategies**: Simple, Chain-of-Thought, and ReAct reasoning
- **Stateless Design**: Horizontal scaling in containerized environments

## Installation

```bash
uv pip install dataknobs-bots
```

## Quick Start

```python
import asyncio
from dataknobs_config import Config
from dataknobs_bots import DynaBot, BotContext

async def main():
    # Load configuration
    config = Config("config/bots.yaml")

    # Create bot from config
    bot = await DynaBot.from_config(config.get("bots", "customer_support"))

    # Create context
    context = BotContext(
        conversation_id="conv-001",
        client_id="demo-client",
        user_id="user-123"
    )

    # Chat
    response = await bot.chat(
        message="Hello, how can I help you?",
        context=context
    )

    print(f"Bot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

See the [implementation guide](../../tmp/active/DYNABOT_IMPLEMENTATION_GUIDE.md) for detailed information.
