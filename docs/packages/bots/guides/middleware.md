# Middleware Guide

Middleware components for request/response lifecycle processing in DataKnobs Bots.

## Overview

Middleware provides hooks into the bot request/response lifecycle, enabling:

- **Logging**: Track all interactions for debugging and analytics
- **Cost Tracking**: Monitor LLM usage and costs
- **Metrics**: Export performance data to monitoring systems
- **Rate Limiting**: Control request rates
- **Authentication**: Validate requests before processing

### Lifecycle Hooks

The middleware lifecycle differs slightly between non-streaming (`chat()`) and streaming (`stream_chat()`) responses:

**Non-Streaming Flow (`chat()`)**:
```
User Message
    │
    ▼
┌─────────────────────┐
│  before_message()   │  ← Pre-processing
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Bot Processing    │
└─────────────────────┘
    │
    ▼ (success)          ▼ (error)
┌─────────────────────┐  ┌─────────────────────┐
│  after_message()    │  │     on_error()      │
└─────────────────────┘  └─────────────────────┘
    │                        │
    ▼                        ▼
Response                Error Response
```

**Streaming Flow (`stream_chat()`)**:
```
User Message
    │
    ▼
┌─────────────────────┐
│  before_message()   │  ← Pre-processing
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Stream Response   │ ──► chunks yielded to caller
└─────────────────────┘
    │
    ▼ (stream complete)  ▼ (error)
┌─────────────────────┐  ┌─────────────────────┐
│   post_stream()     │  │     on_error()      │
└─────────────────────┘  └─────────────────────┘
    │                        │
    ▼                        ▼
Complete               Error Response
```

## Built-in Middleware

DataKnobs Bots provides two built-in middleware classes.

### CostTrackingMiddleware

Tracks LLM API costs and token usage across different providers.

#### Features

- Token tracking per request
- Cost calculation with configurable rates
- Statistics by client and provider
- Export to JSON/CSV

#### Basic Usage

```python
from dataknobs_bots.middleware import CostTrackingMiddleware

# Create middleware with default rates
cost_tracker = CostTrackingMiddleware()

# Or with custom rates
cost_tracker = CostTrackingMiddleware(
    cost_rates={
        "openai": {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        },
        "anthropic": {
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        }
    }
)
```

#### Retrieving Statistics

```python
# Get stats for a specific client
stats = cost_tracker.get_client_stats("my-client")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Total requests: {stats['total_requests']}")

# Get total cost across all clients
total = cost_tracker.get_total_cost()
print(f"Total cost: ${total:.4f}")

# Get token counts
tokens = cost_tracker.get_total_tokens()
print(f"Input tokens: {tokens['input']}")
print(f"Output tokens: {tokens['output']}")

# Export as JSON
json_data = cost_tracker.export_stats_json()

# Export as CSV
csv_data = cost_tracker.export_stats_csv()
```

#### Default Cost Rates

The middleware includes current rates (as of Dec 2024) for:

| Provider | Model | Input (per 1K) | Output (per 1K) |
|----------|-------|----------------|-----------------|
| OpenAI | gpt-4o | $0.0025 | $0.01 |
| OpenAI | gpt-4o-mini | $0.00015 | $0.0006 |
| OpenAI | gpt-4-turbo | $0.01 | $0.03 |
| Anthropic | claude-3-5-sonnet | $0.003 | $0.015 |
| Anthropic | claude-3-5-haiku | $0.0008 | $0.004 |
| Google | gemini-1.5-pro | $0.00125 | $0.005 |
| Google | gemini-2.0-flash | $0.0001 | $0.0004 |
| Ollama | * | $0.00 | $0.00 |

### LoggingMiddleware

Logs all user messages and bot responses for monitoring and debugging.

#### Features

- Structured logging with timestamps
- Configurable log levels
- Optional JSON format for log aggregation
- Full metadata capture

#### Basic Usage

```python
from dataknobs_bots.middleware import LoggingMiddleware

# Basic logging
logger = LoggingMiddleware()

# With JSON format (for log aggregation systems)
logger = LoggingMiddleware(
    log_level="INFO",
    include_metadata=True,
    json_format=True
)

# Debug logging without metadata
logger = LoggingMiddleware(
    log_level="DEBUG",
    include_metadata=False
)
```

#### Log Output Examples

**Standard Format:**
```
INFO:dataknobs_bots.middleware.logging.ConversationLogger:User message: {'timestamp': '2024-12-08T10:30:00+00:00', 'event': 'user_message', 'client_id': 'my-app', 'user_id': 'user-123', 'conversation_id': 'conv-1', 'message_length': 45}
```

**JSON Format:**
```json
{"timestamp": "2024-12-08T10:30:00+00:00", "event": "user_message", "client_id": "my-app", "user_id": "user-123", "conversation_id": "conv-1", "message_length": 45}
```

## Creating Custom Middleware

### Basic Template

```python
from dataknobs_bots.middleware import Middleware
from dataknobs_bots import BotContext
from typing import Any


class MyMiddleware(Middleware):
    """Custom middleware description."""

    def __init__(self, option1: str = "default", option2: int = 10):
        self.option1 = option1
        self.option2 = option2

    async def before_message(self, message: str, context: BotContext) -> None:
        """Called before processing user message."""
        print(f"Processing message from {context.client_id}")

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response (non-streaming)."""
        tokens = kwargs.get("tokens_used", {})
        print(f"Response generated, tokens used: {tokens}")

    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        """Called after streaming response completes."""
        print(f"Streamed response to '{message[:30]}...': {len(response)} chars")

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when an error occurs."""
        print(f"Error: {error} for message: {message[:50]}...")
```

### Example: Rate Limiting Middleware

```python
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from dataknobs_bots.middleware import Middleware
from dataknobs_bots import BotContext


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: dict[str, list[datetime]] = defaultdict(list)

    async def before_message(self, message: str, context: BotContext) -> None:
        client_id = context.client_id
        now = datetime.now()

        # Clean old requests
        cutoff = now - self.window
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > cutoff
        ]

        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            raise Exception(f"Rate limit exceeded for {client_id}")

        # Record request
        self.requests[client_id].append(now)

    async def after_message(
        self, response: str, context: BotContext, **kwargs
    ) -> None:
        pass

    async def post_stream(
        self, message: str, response: str, context: BotContext
    ) -> None:
        pass  # Rate limiting handled in before_message

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        pass
```

## Configuration

### YAML Configuration

```yaml
middleware:
  # Built-in cost tracking
  - class: dataknobs_bots.middleware.CostTrackingMiddleware
    params:
      track_tokens: true
      cost_rates:
        openai:
          gpt-4o:
            input: 0.0025
            output: 0.01

  # Built-in logging
  - class: dataknobs_bots.middleware.LoggingMiddleware
    params:
      log_level: INFO
      include_metadata: true
      json_format: false

  # Custom middleware
  - class: my_middleware.RateLimitMiddleware
    params:
      max_requests: 100
      window_seconds: 60
```

### Python Configuration

```python
from dataknobs_bots import DynaBot
from dataknobs_bots.middleware import CostTrackingMiddleware, LoggingMiddleware

config = {
    "llm": {"provider": "openai", "model": "gpt-4o"},
    "conversation_storage": {"backend": "memory"},
    "middleware": [
        {
            "class": "dataknobs_bots.middleware.CostTrackingMiddleware",
            "params": {"track_tokens": True}
        },
        {
            "class": "dataknobs_bots.middleware.LoggingMiddleware",
            "params": {"log_level": "INFO", "json_format": True}
        }
    ]
}

bot = await DynaBot.from_config(config)
```

## Best Practices

1. **Order Matters**: Middleware executes in order. Put logging first to capture all requests.

2. **Error Handling**: Always implement `on_error` to handle failures gracefully.

3. **Performance**: Keep middleware lightweight. Offload heavy processing to background tasks.

4. **Testing**: Test middleware independently before integration.

5. **Stateless Design**: Prefer stateless middleware when possible for scalability.

## Related Documentation

- [Configuration Reference](configuration.md) - Full configuration options
- [User Guide](user-guide.md) - Getting started tutorials
- [Bot Manager Guide](bot-manager.md) - Multi-tenant deployment
