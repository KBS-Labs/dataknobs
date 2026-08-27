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
| Bedrock | claude-3-5-sonnet | $0.003 | $0.015 |
| Bedrock | claude-3-5-haiku | $0.0008 | $0.004 |
| Google | gemini-1.5-pro | $0.00125 | $0.005 |
| Google | gemini-2.0-flash | $0.0001 | $0.0004 |
| Ollama | * | $0.00 | $0.00 |
| Echo | * | $0.00 | $0.00 |

Bedrock is listed separately from Anthropic even though it resells the same
models, so the two can diverge when Bedrock's pricing does. Its fully-qualified
model IDs (`anthropic.claude-3-5-sonnet-20241022-v2:0`) resolve through the
partial-match fallback.

#### The rate table is keyed by provider family

The first key is the **canonical family key** — the same value
`LLMProvider.provider_name` reports, which is what `TurnState.provider_name`
carries into the middleware. It is lower-cased, so `provider: OpenAI` and
`provider: openai` both price against the `openai` row.

Do not key rates on a provider *class* name. `OpenAIProvider` resembles its
family key by naming convention only, and the resemblance breaks for wrapped
providers — a rate table keyed that way matches nothing and silently prices
every request at $0.00.

#### Unpriced traffic warns

When no rate entry matches, the middleware records $0.00 **and logs a
WARNING**, once per `(provider, model)` pair:

```
Cost tracking: no rate entry for unknown provider family
(provider='huggingface', model='meta-llama/Llama-3-8b'); this traffic is
being recorded at $0.00. Supply rates via
CostTrackingMiddleware(cost_rates=...) to price it.
```

Cost tracking is opt-in, so an operator who enabled it asked for real numbers
and needs to know when they are not. Providers with a genuine zero price
(`ollama`, `echo`) do not warn — a real zero is not a miss. `ollama` qualifies
because it is self-hosted, so there is no per-token charge to record; the
infrastructure cost is real but is not a function of tokens.

`huggingface` is deliberately **not** given a default rate: it covers both free
local inference and the paid Inference API, so a zero entry would assert "this
traffic is free" for the paid case. Supply rates explicitly:

```python
cost_tracker = CostTrackingMiddleware(
    cost_rates={
        "huggingface": {"input": 0.0006, "output": 0.0006},
    }
)
```

Both the defaults and the dict you pass are deep-copied, so one middleware
instance's overrides never affect another's — and a shared module-level rate
constant handed to several per-tenant instances is neither aliased between
them nor mutated in place.

#### Where a rate comes from

Rates resolve in this order, most authoritative first:

1. **A rate you supplied** via `cost_rates=`. You have stated the price you
   are billed; nothing derived outranks it.
2. **The provider's own pricing**, resolved through `dataknobs-llm`'s model
   profiles — dated catalogs, overridable per config, isolated per provider.
   `TurnState` captures it while the provider is still in hand and hands it to
   the middleware on the turn.
3. **The middleware's built-in table**, for a family or model the catalogs do
   not cover: a consumer's out-of-tree provider, a self-hosted gateway.
4. Otherwise $0.00, with the warning above.

The built-in table sits *below* the provider's catalog deliberately. It is a
hand-maintained copy of the same numbers and has drifted from them, so it is a
fallback rather than a source of truth. Within it, a model id that is not an
exact key matches the **longest** table key it contains — `gpt-4o` is a prefix
of `gpt-4o-mini-2024-07-18`, and taking the first match instead bills the mini
model at the full model's rate.

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

**Turn completion (JSON):**
```json
{"timestamp": "2024-12-08T10:30:02+00:00", "event": "turn_complete", "mode": "chat", "client_id": "my-app", "user_id": "user-123", "conversation_id": "conv-1", "response_length": 25, "tokens_used": {"input": 120, "output": 45}, "provider": "openai", "provider_impl": "OpenAIProvider", "model": "gpt-4o-mini", "session_metadata": {}, "request_metadata": {}}
```

The payload carries **two** provider fields, and they answer different
questions:

| Field | Value | Meaning |
|---|---|---|
| `provider` | `"openai"` | Canonical family key — the stable label to group, filter, and aggregate on |
| `provider_impl` | `"OpenAIProvider"` | Concrete class that served the call, including wrappers — diagnostic |

For a wrapped provider these diverge (`provider: "openai"`,
`provider_impl: "CachingEmbedProvider"`), which is what makes the second field
worth having: it is the only place the wrapper is visible.

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

See the [Rate Limiting guide](../../common/docs/guides/ratelimit.md) for the full `InMemoryRateLimiter` API, including per-category rates, weighted operations, and distributed backends.

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
| `provider_name` | `str \| None` | Canonical provider **family** key (e.g. `"openai"`) — key rate tables, metrics labels, and log fields on this. `None` when the object served the turn but declared no family |
| `provider_impl` | `str \| None` | Concrete provider **class** (e.g. `"CachingEmbedProvider"`) — diagnostics only, never a lookup key |
| `pricing` | `ModelPricing \| None` | Per-model USD rates the provider resolved for this turn's model, or `None` when it sources none |
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

### Platform (additive) middleware

The `middleware=` / `conversation_middleware=` kwargs **replace** config
middleware — the right shape for "run exactly this, ignore config" (e.g.
single-middleware testing). When you instead need to install an **always-on,
cross-cutting** middleware on **every** bot a platform builds — *without*
dropping each bot's own config-declared middleware — use the additive
`platform_middleware=` / `platform_conversation_middleware=` kwargs. These
**append** to whatever the resolve produced (the config path **or** the
`middleware=` replace-override path):

```python
# A shared state-writer holding a live per-deployment collaborator.
state_writer = StudentStateMiddleware(user_state_store)  # one shared instance

bot = await DynaBot.from_config(
    config,                            # config may declare its own middleware:
    platform_middleware=[state_writer],  # appended, never substituted
)
# bot.middleware == [<config middleware...>, state_writer]
```

This channel exists specifically for a **live shared collaborator** (an object
that cannot be expressed as a config `{class, params}` spec) that must not
clobber the bot's own middleware. Omitting the platform kwargs is
byte-identical to today.

**Ordering.** Appended platform middleware runs **after** config middleware:

- **Bot-turn `platform_middleware`** — dispatched by simple forward iteration,
  so the platform middleware runs **last** on every hook. Its `on_turn_start`
  transform sees the message as modified by all config middleware; its
  `after_turn` observer sees the fully-processed turn. This is exactly what a
  platform observer / state-writer wants.
- **LLM-call `platform_conversation_middleware`** — forwarded to every
  `ConversationManager`, which runs middleware **onion-style**
  (`process_request` forward, `process_response` reversed). Appending therefore
  positions the platform middleware **innermost on the request** (last before
  the LLM call) and **outermost on the response** (first after the LLM
  returns).

To exercise the additive channel in tests, route it through `from_config` via
`BotTestHarness.create(..., platform_middleware=[...])` — distinct from the
harness's `middleware=` param, which post-appends to the built bot.

---

## Building middleware from specs

`DynaBot` resolves its own configured `middleware:` and
`conversation_middleware:` blocks through two free-standing factories.
They are exported so anything that assembles middleware *declaratively* —
a composed behavior pack, a deployment's policy bundle, a test fixture —
can turn specs into live instances without reaching into bot internals or
reimplementing the resolution rules.

```python
from dataknobs_bots import build_conversation_middleware, build_middleware

turn_specs = [
    {"class": "acme.mw.AuditMiddleware", "params": {"level": "info"}},
    {"class": "acme.mw.OptionalTracer", "optional": True},
]
call_specs = [{"class": "acme.mw.PromptRedactor"}]

bot = await DynaBot.from_config(
    config,
    platform_middleware=build_middleware(turn_specs),
    platform_conversation_middleware=build_conversation_middleware(call_specs),
)
```

Both take an **iterable** of specs and return a **list** of live instances —
the shape both install channels want. Each is consumed once, so a
one-shot generator is fine.

| Function | Builds | For |
|---|---|---|
| `build_middleware` | `Middleware` | Bot-turn lifecycle hooks (`on_turn_start` / `after_turn` / ...) |
| `build_conversation_middleware` | `ConversationMiddleware` | LLM-call wraps (`process_request` / `process_response`) |
| `resolve_middleware_from_spec` | either | One spec at a time — see [Resolving a single spec](#resolving-a-single-spec) |

Both wrappers delegate to `resolve_middleware_from_spec`, so there is
exactly one resolution body and the two flavors cannot drift.

> **Middleware specs are trusted configuration.** A spec's `class` is a
> dotted path that gets imported and instantiated, so resolving one executes
> whatever that module and constructor do — import *is* execution, and there
> is no allow-list or sandbox here.
>
> Specs must come from the same trust domain as the application's own code:
> a config file, a deployment's policy bundle, a pack a platform team
> authored. Never build one from end-user input, a request body, or a
> per-tenant blob the tenant supplies.
>
> The same boundary applies to every config key taking a dotted path — hook
> paths, custom transforms, merge filters, tool classes — and is stated once
> for all of them in the
> [dotted-path guide](https://kbs-labs.github.io/dataknobs/packages/common/dotted-paths/),
> along with why returning the class rather than an instance is only a
> *partial* mitigation.

### Spec shape

```python
{
    "class": "my_pkg.mw.AuditMiddleware",   # dotted import path (required)
    "params": {"level": "info"},            # constructor kwargs (optional)
    "optional": False,                      # skip on failure (optional)
}
```

### `optional` covers environment failures, not layout mistakes

`optional: true` skips a spec whose **module or class cannot be resolved**,
or whose **constructor fails** — a genuinely absent integration in this
environment. A warning is logged and the spec is left out.

It does **not** cover a class-shape mismatch. A class listed under the
wrong field — a turn-lifecycle `Middleware` under
`conversation_middleware:`, or the reverse — always raises
`ConfigurationError`, regardless of `optional`. That is a programmer error
in the config layout, and the only safe response is to surface it at
config-load rather than silently run a bot missing behavior it declared.

The shape check runs **before instantiation**, so a wrong-shape spec never
executes its constructor — no network read, file open, or log write from a
misplaced spec's initializer.

### Skipped specs are absent, not `None`

A skipped `optional` spec is **removed** from the result, so the returned
list is directly usable as `platform_middleware`. Positional
correspondence with the input specs is deliberately not preserved:

```python
specs = [good_spec, broken_optional_spec, another_good_spec]
build_middleware(specs)     # -> [<good>, <another_good>]   (length 2)
```

If you need to know *which* spec was skipped, resolve them one at a time.

### Resolving a single spec

`resolve_middleware_from_spec` is the shared resolution body, exposed for
per-spec handling and for middleware families neither wrapper covers:

```python
from dataknobs_bots import resolve_middleware_from_spec
from dataknobs_bots.middleware import Middleware

for spec in specs:
    mw = resolve_middleware_from_spec(spec, Middleware, label="middleware")
    if mw is None:
        record_missing_integration(spec["class"])
    else:
        installed.append(mw)
```

For the two built-in flavors prefer the wrappers — they supply the correct
`expected_base` and the `label` that appears in error messages.

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

- [configuration.md](configuration.md) - Full configuration reference
- [user-guide.md](user-guide.md) - Getting started tutorials
- [api.md](api.md) - API reference
