# Middleware Guide

Middleware components for request/response lifecycle processing in DataKnobs Bots.

## Table of Contents

- [Overview](#overview)
- [Built-in Middleware](#built-in-middleware)
  - [CostTrackingMiddleware](#costtrackingmiddleware)
  - [LoggingMiddleware](#loggingmiddleware)
- [Creating Custom Middleware](#creating-custom-middleware)
- [Middleware Interface](#middleware-interface)
- [Configuration](#configuration)

---

## Overview

Middleware provides hooks into the bot request/response lifecycle, enabling:

- **Logging**: Track all interactions for debugging and analytics
- **Cost Tracking**: Monitor LLM usage and costs
- **Metrics**: Export performance data to monitoring systems
- **Rate Limiting**: Control request rates
- **Authentication**: Validate requests before processing

### Lifecycle Hooks

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

---

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

---

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
{"timestamp": "2024-12-08T10:30:00+00:00", "event": "user_message", "client_id": "my-app", "user_id": "user-123", "conversation_id": "conv-1", "message_length": 45, "session_metadata": {}, "request_metadata": {}}
```

---

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
        # Pre-processing logic
        print(f"Processing message from {context.client_id}")

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response."""
        # Post-processing logic
        tokens = kwargs.get("tokens_used", {})
        print(f"Response generated, tokens used: {tokens}")

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

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        pass
```

### Example: Metrics Middleware

```python
import time
from dataknobs_bots.middleware import Middleware
from dataknobs_bots import BotContext


class MetricsMiddleware(Middleware):
    """Export metrics to monitoring system."""

    def __init__(self, statsd_host: str = "localhost", statsd_port: int = 8125):
        self.statsd_host = statsd_host
        self.statsd_port = statsd_port
        self._start_times: dict[str, float] = {}

    async def before_message(self, message: str, context: BotContext) -> None:
        # Record start time
        key = f"{context.client_id}:{context.conversation_id}"
        self._start_times[key] = time.time()

    async def after_message(
        self, response: str, context: BotContext, **kwargs
    ) -> None:
        key = f"{context.client_id}:{context.conversation_id}"
        start = self._start_times.pop(key, None)

        if start:
            duration_ms = (time.time() - start) * 1000
            # Send to metrics system
            await self._send_metric("bot.response_time", duration_ms)
            await self._send_metric("bot.response_length", len(response))

            if "tokens_used" in kwargs:
                tokens = kwargs["tokens_used"]
                await self._send_metric("bot.tokens_input", tokens.get("input", 0))
                await self._send_metric("bot.tokens_output", tokens.get("output", 0))

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        await self._send_metric("bot.errors", 1)

    async def _send_metric(self, name: str, value: float) -> None:
        # Implementation depends on your metrics system
        # Example: StatsD, Prometheus, CloudWatch, etc.
        pass
```

---

## Middleware Interface

All middleware must implement the `Middleware` abstract base class:

```python
from abc import ABC, abstractmethod
from typing import Any
from dataknobs_bots import BotContext


class Middleware(ABC):
    """Abstract base class for bot middleware."""

    @abstractmethod
    async def before_message(self, message: str, context: BotContext) -> None:
        """Called before processing user message.

        Args:
            message: User's input message
            context: Bot context with conversation and user info
        """
        ...

    @abstractmethod
    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Called after generating bot response.

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Additional data including:
                - tokens_used: Dict with 'input' and 'output' counts
                - response_time_ms: Response generation time
                - provider: LLM provider name
                - model: Model identifier
        """
        ...

    @abstractmethod
    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when an error occurs during message processing.

        Args:
            error: The exception that occurred
            message: User message that caused the error
            context: Bot context
        """
        ...
```

---

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
  - class: my_app.middleware.RateLimitMiddleware
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

### Programmatic Middleware Registration

```python
# Create middleware instances
cost_tracker = CostTrackingMiddleware()
logger = LoggingMiddleware(json_format=True)

# Access middleware later
stats = cost_tracker.get_all_stats()
```

---

## Best Practices

1. **Order Matters**: Middleware executes in order. Put logging first to capture all requests.

2. **Error Handling**: Always implement `on_error` to handle failures gracefully.

3. **Performance**: Keep middleware lightweight. Offload heavy processing to background tasks.

4. **Testing**: Test middleware independently before integration.

5. **Stateless Design**: Prefer stateless middleware when possible for scalability.

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Full configuration reference
- [USER_GUIDE.md](USER_GUIDE.md) - Getting started tutorials
- [API.md](API.md) - API reference
