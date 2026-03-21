# Middleware Guide

Middleware components for request/response lifecycle processing in DataKnobs Bots.

## Table of Contents

- [Overview](#overview)
- [Built-in Middleware](#built-in-middleware)
  - [CostTrackingMiddleware](#costtrackingmiddleware)
  - [LoggingMiddleware](#loggingmiddleware)
- [Creating Custom Middleware](#creating-custom-middleware)
- [TurnState Reference](#turnstate-reference)
- [Middleware Interface](#middleware-interface)
- [Legacy Hooks](#legacy-hooks)
- [Configuration](#configuration)

---

## Overview

Middleware provides hooks into the bot request/response lifecycle, enabling:

- **Logging**: Track all interactions for debugging and analytics
- **Cost Tracking**: Monitor LLM usage and costs per turn
- **Tool Observability**: React to tool executions (audit, cost, analytics)
- **Dependency Injection**: Inject per-request resources (DB sessions, auth tokens)
- **Message Transforms**: Strip PII, sanitize input, detect attacks
- **Metrics**: Export performance data to monitoring systems
- **Rate Limiting**: Control request rates
- **Authentication**: Validate requests before processing

### Lifecycle Hooks

The middleware pipeline uses a unified lifecycle based on `TurnState`. All turn
types (`chat()`, `stream_chat()`, `greet()`) flow through the same hooks:

```
User Message (or greet)
    │
    ▼
┌─────────────────────────┐
│    on_turn_start(turn)   │  ← Pre-processing, plugin_data, message transforms
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    Bot Processing        │  ← Generation + tool execution loop
│    (tool_calls? →        │
│     _execute_tools →     │
│     re-generate)         │
└─────────────────────────┘
    │
    ▼ (success)               ▼ (error)
┌─────────────────────────┐  ┌─────────────────────────┐
│  on_tool_executed(exec)  │  │     on_error()          │
│  (once per tool call)    │  └─────────────────────────┘
├─────────────────────────┤       │
│  after_turn(turn)        │       ▼
└─────────────────────────┘  Error re-raised
    │
    ▼
Response (or stream complete)
```

**Hook failure handling (`on_hook_error`)**:

If any middleware hook itself raises (e.g., a logging sink is down during
`after_turn`), the exception is caught, logged, and all middleware are
notified via `on_hook_error(hook_name, error, context)`.  This is separate
from `on_error`, which fires for request-level failures.

### Error Semantics

| Hook | Fires when | Request succeeded? |
|------|-----------|-------------------|
| `on_error` | Request preparation or generation fails | No |
| `on_hook_error` | A middleware hook itself raises | Yes (response already delivered) |

This distinction lets middleware differentiate "the request failed" from
"observability/post-processing broke."  Error-tracking middleware can count
request failures via `on_error` and infrastructure failures via
`on_hook_error` independently.

### Middleware Base Class

`Middleware` is a concrete class with all hooks as no-ops. Subclasses override
only the hooks they need — no need to implement every method:

```python
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.bot.turn import TurnState

class MyCostTracker(Middleware):
    # Only override what you need — everything else is a no-op
    async def after_turn(self, turn: TurnState) -> None:
        if turn.usage:
            await save_usage(turn.usage, turn.context.client_id)
```

---

## Built-in Middleware

DataKnobs Bots provides two built-in middleware classes. Both are fully migrated
to the unified `TurnState` hooks (`on_turn_start`, `after_turn`).

### CostTrackingMiddleware

Tracks LLM API costs and token usage across different providers.

#### Features

- Real token usage from provider responses (via `after_turn`)
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

### Basic Template — Unified Hooks (Preferred)

Override only the hooks you need. All others are no-ops.

```python
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.bot.turn import TurnState, ToolExecution
from dataknobs_bots.bot.context import BotContext


class MyMiddleware(Middleware):
    """Custom middleware — override only what you need."""

    async def on_turn_start(self, turn: TurnState) -> str | None:
        """Pre-processing before LLM generation."""
        # Write shared data for downstream pipeline participants
        turn.plugin_data["request_id"] = generate_id()
        # Optionally return a transformed message
        return None  # or return sanitized_message

    async def after_turn(self, turn: TurnState) -> None:
        """Post-processing after any turn completes."""
        if turn.usage:
            log.info(
                "Turn complete: %s tokens",
                turn.usage.get("input", 0) + turn.usage.get("output", 0),
            )

    async def on_tool_executed(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        """Called after each tool execution."""
        log.info("Tool %s: %s", execution.tool_name, execution.result)

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Called when a request-level error occurs."""
        log.error("Request failed: %s", error)
```

### Example: Per-Request Dependency Injection

Use `on_turn_start` to inject per-request resources and `after_turn` to clean up.
Resources are available to tools via `ToolExecutionContext.extra["turn_data"]`.

```python
class SessionMiddleware(Middleware):
    """Inject a database session for each turn."""

    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def on_turn_start(self, turn: TurnState) -> str | None:
        turn.plugin_data["db.session"] = await self._db_factory()
        return None

    async def after_turn(self, turn: TurnState) -> None:
        session = turn.plugin_data.get("db.session")
        if session:
            await session.close()
```

Tools access the session via the context bridge:

```python
class MyTool(ContextAwareTool):
    async def execute(self, **kwargs):
        session = self.context.extra["turn_data"]["db.session"]
        return await session.execute(...)
```

### Example: PII Stripping with Restoration

Use `on_turn_start` to strip PII and `after_turn` to restore it.

**Note:** For `chat()` and `greet()`, mutations to `turn.response_content`
in `after_turn` propagate to the caller. For `stream_chat()`, chunks were
already yielded before `after_turn` runs — mutations update
`turn.response_content` (available for logging/storage) but do not affect
the content the streaming consumer already received.

The transformed message also replaces the original in conversation history
and memory. For PII stripping this is typically desired (the original PII
is not persisted).

```python
class PIIMiddleware(Middleware):
    async def on_turn_start(self, turn: TurnState) -> str | None:
        stripped, mappings = strip_pii(turn.message)
        turn.plugin_data["pii.mappings"] = mappings
        return stripped  # Transformed message sent to LLM

    async def after_turn(self, turn: TurnState) -> None:
        # For chat/greet this updates the returned response.
        # For streaming, this updates turn.response_content for
        # logging/storage but does not affect already-yielded chunks.
        mappings = turn.plugin_data.get("pii.mappings", {})
        if mappings:
            turn.response_content = restore_pii(
                turn.response_content, mappings
            )
```

### Example: Rate Limiting Middleware

Use `InMemoryRateLimiter` from `dataknobs-common` for the rate limiting backend:

```python
from dataknobs_common.ratelimit import (
    InMemoryRateLimiter, RateLimit, RateLimiterConfig,
)
from dataknobs_common.exceptions import RateLimitError
from dataknobs_bots.middleware.base import Middleware
from dataknobs_bots.bot.turn import TurnState


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware backed by InMemoryRateLimiter."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=max_requests, interval=window_seconds)],
        )
        self._limiter = InMemoryRateLimiter(config)

    async def on_turn_start(self, turn: TurnState) -> str | None:
        client_id = turn.context.client_id
        if not await self._limiter.try_acquire(client_id):
            status = await self._limiter.get_status(client_id)
            raise RateLimitError(
                f"Rate limit exceeded for {client_id}",
                retry_after=status.reset_after,
            )
        return None
```

See the [Rate Limiting guide](../../packages/common/ratelimit.md) for the full `InMemoryRateLimiter` API, including per-category rates, weighted operations, and distributed backends.

### Example: Tool Execution Auditing

```python
class ToolAuditor(Middleware):
    """Log tool executions with timing data."""

    async def on_tool_executed(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        if execution.error:
            log.warning(
                "Tool %s failed: %s", execution.tool_name, execution.error,
            )
        else:
            log.info(
                "Tool %s completed in %.1fms",
                execution.tool_name,
                execution.duration_ms,
            )

    async def after_turn(self, turn: TurnState) -> None:
        """Batch-process all tool executions at turn end."""
        for exec in turn.tool_executions:
            await save_tool_audit_record(exec, turn.context)
```

---

## TurnState Reference

`TurnState` (`dataknobs_bots.bot.turn`) is the per-turn state carrier. Created at
the start of each `chat()`, `stream_chat()`, or `greet()` call. Available to
middleware via `on_turn_start` and `after_turn`.

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `TurnMode` | How the turn was initiated: `CHAT`, `STREAM`, `GREET` |
| `message` | `str` | User message (`""` for greet) |
| `context` | `BotContext` | Bot context (client_id, conversation_id, user_id, etc.) |
| `response_content` | `str` | Final response text (populated after generation) |
| `usage` | `dict[str, int] \| None` | Token usage: `{"input": N, "output": M}` |
| `model` | `str \| None` | Model that generated the response |
| `provider_name` | `str \| None` | Provider name (e.g., `"OpenAIProvider"`) |
| `tool_executions` | `list[ToolExecution]` | Tool executions recorded during the turn |
| `plugin_data` | `dict[str, Any]` | Cross-middleware communication dict |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_streaming` | `bool` | Whether this is a streaming turn |
| `is_greet` | `bool` | Whether this is a greet turn |

### ToolExecution

`ToolExecution` (`dataknobs_bots.bot.turn`) records a single tool execution:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool |
| `parameters` | `dict[str, Any]` | Parameters passed to the tool |
| `result` | `Any` | Tool return value (if successful) |
| `error` | `str \| None` | Error message (if failed) |
| `duration_ms` | `float \| None` | Execution time in milliseconds |

### plugin_data

`plugin_data` is a per-turn writable dict that bridges across the entire pipeline:

```
on_turn_start(turn)          ← write plugin_data
  → ConversationMiddleware   ← reads/writes via state.turn_data (same dict)
    → Tool execution         ← reads via context.extra["turn_data"]
  → after_turn(turn)         ← reads final plugin_data
```

Namespace by convention with dotted keys: `"pii.mappings"`, `"session.db"`,
`"auth.claims"`.

---

## Middleware Interface

All hooks are concrete no-ops on the `Middleware` base class. Override only
what you need.

### Preferred Hooks

| Hook | Signature | When |
|------|-----------|------|
| `on_turn_start` | `(turn: TurnState) -> str \| None` | Before processing; can transform message and write plugin_data |
| `after_turn` | `(turn: TurnState) -> None` | After any turn completes (chat, stream, greet) |
| `on_tool_executed` | `(execution: ToolExecution, context: BotContext) -> None` | After each tool call (post-turn, not real-time) |

### Error Hooks

| Hook | Signature | When |
|------|-----------|------|
| `on_error` | `(error: Exception, message: str, context: BotContext) -> None` | Request failed |
| `on_hook_error` | `(hook_name: str, error: Exception, context: BotContext) -> None` | A middleware hook itself failed |

### `on_tool_executed` Timing

`on_tool_executed` fires **post-turn** during `_finalize_turn()`, not in real-time
as tools execute. This hook is for auditing and logging, not for aborting or
rate-limiting mid-turn. Tool executions are also available as
`turn.tool_executions` in the `after_turn` hook for batch processing.

---

## Legacy Hooks

The following hooks are kept for backward compatibility but are deprecated.
Existing middleware using these hooks will continue to work. Migrate to the
unified hooks at your convenience.

| Legacy Hook | Replacement | Notes |
|-------------|-------------|-------|
| `before_message(message, context)` | `on_turn_start(turn)` | `on_turn_start` provides full TurnState + plugin_data + message transforms |
| `after_message(response, context, **kwargs)` | `after_turn(turn)` | `after_turn` fires for all turn types with real usage data |
| `post_stream(message, response, context)` | `after_turn(turn)` | `after_turn` eliminates the chat-vs-stream split |

Both legacy and unified hooks fire on every turn — you can migrate incrementally.

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

### Programmatic Middleware via `from_config()`

Use the `middleware=` keyword argument to inject middleware programmatically,
bypassing config-driven middleware construction:

```python
from dataknobs_bots import DynaBot
from dataknobs_bots.middleware import CostTrackingMiddleware, LoggingMiddleware

cost_tracker = CostTrackingMiddleware()
logger_mw = LoggingMiddleware(json_format=True)

bot = await DynaBot.from_config(
    config,
    middleware=[cost_tracker, logger_mw],  # Overrides config middleware
)
```

When `middleware=` is passed, it completely replaces any middleware defined in the
config dict. Pass `middleware=[]` to explicitly disable all middleware.

---

## Best Practices

1. **Use Unified Hooks**: Prefer `on_turn_start` and `after_turn` over legacy hooks.
   They provide the full `TurnState` and work uniformly across all turn types.

2. **Order Matters**: Middleware executes in list order. `on_turn_start` message
   transforms chain: each middleware receives the message as modified by the
   previous one.

3. **Namespace plugin_data**: Use dotted keys (`"pii.mappings"`, `"session.db"`)
   to avoid collisions between middleware.

4. **Error Handling**: Implement `on_error` for request failures and `on_hook_error`
   for middleware infrastructure failures.

5. **Performance**: Keep middleware lightweight. Offload heavy processing to
   background tasks.

6. **Testing**: Use `BotTestHarness` with `middleware=[...]` to test middleware
   in integration.

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Full configuration reference
- [USER_GUIDE.md](USER_GUIDE.md) - Getting started tutorials
- [API.md](API.md) - API reference
