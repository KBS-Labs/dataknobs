# Multi-Tenant Bot Management

Guide to deploying and managing multiple bot instances with BotManager.

## Table of Contents

- [Overview](#overview)
- [BotManager](#botmanager)
  - [Basic Usage](#basic-usage)
  - [Configuration Loaders](#configuration-loaders)
  - [Bot Lifecycle](#bot-lifecycle)
- [FastAPI Integration](#fastapi-integration)
  - [Dependency Injection](#dependency-injection)
  - [Exception Handling](#exception-handling)
  - [Complete Example](#complete-example)
- [Patterns](#patterns)
- [Best Practices](#best-practices)

---

## Overview

Multi-tenant bot deployment allows a single application to serve multiple clients, each with their own bot configuration and isolated conversations.

**Key Concepts:**

- **BotManager**: Manages bot instances with caching and lifecycle control
- **Bot ID**: Unique identifier for each bot configuration (e.g., "support-bot", "sales-bot")
- **Client ID**: Tenant identifier within BotContext
- **Conversation ID**: Unique conversation identifier

```
                    BotManager
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   support-bot      sales-bot      custom-bot
        │               │               │
    ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
    │       │       │       │       │       │
 client-A client-B client-C client-D client-E client-F
```

---

## BotManager

### Basic Usage

```python
from dataknobs_bots import BotManager, BotContext

# Create manager
manager = BotManager()

# Create or get a bot with inline configuration
bot = await manager.get_or_create("support-bot", config={
    "llm": {"provider": "openai", "model": "gpt-4o"},
    "conversation_storage": {"backend": "memory"},
    "system_prompt": "You are a helpful customer support assistant."
})

# Use the bot
context = BotContext(
    conversation_id="conv-123",
    client_id="client-A",
    user_id="user-456"
)
response = await bot.chat("Hello, I need help", context)

# Get the same bot instance (cached)
same_bot = await manager.get_or_create("support-bot")
assert same_bot is bot  # Same instance
```

### Configuration Loaders

BotManager supports pluggable configuration loading for dynamic bot creation.

#### Function-based Loader

```python
import yaml

def load_bot_config(bot_id: str) -> dict:
    """Load bot configuration from YAML files."""
    with open(f"configs/{bot_id}.yaml") as f:
        return yaml.safe_load(f)

# Create manager with loader
manager = BotManager(config_loader=load_bot_config)

# Bot will be created using loaded config
bot = await manager.get_or_create("support-bot")
```

#### Async Function Loader

```python
async def load_config_from_db(bot_id: str) -> dict:
    """Load configuration from database."""
    async with db.acquire() as conn:
        row = await conn.fetchone(
            "SELECT config FROM bot_configs WHERE bot_id = $1",
            bot_id
        )
        return row["config"]

manager = BotManager(config_loader=load_config_from_db)
```

#### Class-based Loader

```python
class ConfigLoader:
    """Configuration loader with caching."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self._cache = {}

    def load(self, bot_id: str) -> dict:
        if bot_id not in self._cache:
            with open(f"{self.config_dir}/{bot_id}.yaml") as f:
                self._cache[bot_id] = yaml.safe_load(f)
        return self._cache[bot_id]

loader = ConfigLoader("./configs")
manager = BotManager(config_loader=loader)
```

### Bot Lifecycle

```python
# List active bots
active_bots = manager.list_bots()
print(f"Active bots: {active_bots}")

# Get bot count
count = manager.get_bot_count()
print(f"Total bots: {count}")

# Get bot without creating
bot = await manager.get("support-bot")
if bot is None:
    print("Bot not yet created")

# Remove a bot
removed = await manager.remove("support-bot")
print(f"Bot removed: {removed}")

# Reload bot with fresh config (requires config_loader)
bot = await manager.reload("support-bot")

# Clear all bots
await manager.clear_all()
```

---

## FastAPI Integration

### Dependency Injection

The `api` module provides FastAPI integration with singleton management.

```python
from fastapi import FastAPI
from dataknobs_bots.api import (
    init_bot_manager,
    get_bot_manager,
    BotManagerDep,
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Initialize the singleton with a config loader
    init_bot_manager(config_loader=load_config)

@app.post("/chat/{bot_id}")
async def chat(
    bot_id: str,
    message: str,
    manager: BotManagerDep,  # Injected dependency
):
    bot = await manager.get_or_create(bot_id)
    context = BotContext(
        conversation_id="...",
        client_id="..."
    )
    return await bot.chat(message, context)
```

### Exception Handling

Built-in exceptions provide consistent API error responses.

```python
from dataknobs_bots.api import (
    APIError,
    BotNotFoundError,
    BotCreationError,
    ConversationNotFoundError,
    ValidationError,
    ConfigurationError,
    RateLimitError,
    register_exception_handlers,
)

app = FastAPI()

# Register all exception handlers
register_exception_handlers(app)

@app.get("/bots/{bot_id}")
async def get_bot(bot_id: str, manager: BotManagerDep):
    bot = await manager.get(bot_id)
    if not bot:
        raise BotNotFoundError(bot_id)
    return {"bot_id": bot_id, "status": "active"}

@app.post("/bots/{bot_id}")
async def create_bot(bot_id: str, config: dict, manager: BotManagerDep):
    try:
        bot = await manager.get_or_create(bot_id, config=config)
        return {"bot_id": bot_id, "created": True}
    except Exception as e:
        raise BotCreationError(bot_id, str(e))
```

**Error Response Format:**

```json
{
    "error": "BotNotFoundError",
    "message": "Bot with ID 'unknown-bot' not found",
    "detail": {"bot_id": "unknown-bot"},
    "timestamp": "2024-12-08T10:30:00+00:00"
}
```

### Complete Example

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from dataknobs_bots import BotContext
from dataknobs_bots.api import (
    init_bot_manager,
    reset_bot_manager,
    BotManagerDep,
    BotNotFoundError,
    register_exception_handlers,
)

app = FastAPI(title="Multi-Tenant Bot API")

# Register exception handlers
register_exception_handlers(app)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    client_id: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


def load_config(bot_id: str) -> dict:
    """Load bot configuration."""
    configs = {
        "support": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": "You are a customer support assistant.",
        },
        "sales": {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": "You are a sales assistant.",
        },
    }
    if bot_id not in configs:
        raise ValueError(f"Unknown bot: {bot_id}")
    return configs[bot_id]


@app.on_event("startup")
async def startup():
    init_bot_manager(config_loader=load_config)


@app.on_event("shutdown")
async def shutdown():
    reset_bot_manager()


@app.post("/bots/{bot_id}/chat", response_model=ChatResponse)
async def chat(
    bot_id: str,
    request: ChatRequest,
    manager: BotManagerDep,
):
    """Chat with a bot."""
    bot = await manager.get_or_create(bot_id)

    context = BotContext(
        conversation_id=request.conversation_id,
        client_id=request.client_id,
        user_id=request.user_id,
    )

    response = await bot.chat(request.message, context)

    return ChatResponse(
        response=response,
        conversation_id=request.conversation_id,
    )


@app.get("/bots")
async def list_bots(manager: BotManagerDep):
    """List active bots."""
    return {
        "bots": manager.list_bots(),
        "count": manager.get_bot_count(),
    }


@app.delete("/bots/{bot_id}")
async def remove_bot(bot_id: str, manager: BotManagerDep):
    """Remove a bot instance."""
    removed = await manager.remove(bot_id)
    if not removed:
        raise BotNotFoundError(bot_id)
    return {"removed": True, "bot_id": bot_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Patterns

### Pattern 1: Per-Client Bot Configuration

```python
async def get_bot_for_client(manager: BotManager, client_id: str) -> DynaBot:
    """Each client gets their own bot configuration."""
    bot_id = f"bot-{client_id}"

    # Check if bot exists
    bot = await manager.get(bot_id)
    if bot:
        return bot

    # Load client-specific config
    config = await load_client_config(client_id)
    return await manager.get_or_create(bot_id, config=config)
```

### Pattern 2: Shared Bot with Client Context

```python
async def chat_with_shared_bot(
    manager: BotManager,
    client_id: str,
    user_id: str,
    message: str,
) -> str:
    """All clients share the same bot but have isolated conversations."""
    # Single shared bot
    bot = await manager.get_or_create("shared-bot", config=shared_config)

    # Client isolation through context
    context = BotContext(
        conversation_id=f"{client_id}-{user_id}",
        client_id=client_id,
        user_id=user_id,
    )

    return await bot.chat(message, context)
```

### Pattern 3: Bot Pools by Type

```python
class BotPool:
    """Manage pools of bots by type."""

    def __init__(self):
        self.managers = {
            "support": BotManager(config_loader=support_loader),
            "sales": BotManager(config_loader=sales_loader),
            "general": BotManager(config_loader=general_loader),
        }

    async def get_bot(self, bot_type: str, bot_id: str) -> DynaBot:
        manager = self.managers.get(bot_type)
        if not manager:
            raise ValueError(f"Unknown bot type: {bot_type}")
        return await manager.get_or_create(bot_id)
```

---

## Best Practices

### 1. Configuration Management

- Store configurations in version control
- Use environment variables for secrets
- Validate configurations at startup

### 2. Resource Management

- Set reasonable cache limits for bot instances
- Implement bot eviction for unused instances
- Monitor memory usage

### 3. Error Handling

- Use specific exception types
- Log errors with context
- Return consistent error responses

### 4. Security

- Validate client IDs
- Implement authentication
- Rate limit requests

### 5. Monitoring

- Track bot creation/destruction
- Monitor conversation counts
- Alert on errors

---

## See Also

- [MIDDLEWARE.md](MIDDLEWARE.md) - Request/response middleware
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [USER_GUIDE.md](USER_GUIDE.md) - Getting started
