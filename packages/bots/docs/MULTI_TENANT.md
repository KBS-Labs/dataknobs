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
        # Keep `reason` authored, and let `from e` carry the underlying
        # error to the logs. Building it from `str(e)` puts whatever a tool
        # or middleware constructor raised — a driver error naming its
        # connection URL, for instance — into an exception the caller may
        # see. `BotCreationError` is masked in responses for exactly that
        # reason, so the detail below is logged, not returned.
        raise BotCreationError(bot_id, "configuration could not be loaded") from e
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

<!-- --8<-- [start:catching-api-errors] -->
#### Catching these errors

**Raise the `dataknobs_bots.api` class; catch the `dataknobs_common.exceptions`
one.** Every API exception with a common counterpart subclasses it, so the
common name catches both the API variant and the one DataKnobs itself raises:

```python
from dataknobs_common.exceptions import RateLimitError

try:
    reply = await bot.chat(message, context)
except RateLimitError as exc:
    # Catches both dataknobs_bots.api.RateLimitError and the one
    # dataknobs_llm.conversations.middleware.RateLimitMiddleware raises.
    retry_after = exc.retry_after
```

Catching the `dataknobs_bots.api` name instead narrows you to errors your own
handlers raised, and silently misses the ones raised inside DataKnobs. Three of
the API classes are also *same-named* as their common counterpart
(`ValidationError`, `ConfigurationError`, `RateLimitError`), so which one an
`except` clause binds depends only on which module it was imported from — an
easy thing to get wrong by accident and a hard one to notice. Importing the
common name avoids the ambiguity entirely.

The pairing is:

| `dataknobs_bots.api` | also catchable as |
|---|---|
| `BotNotFoundError`, `ConversationNotFoundError` | `NotFoundError` |
| `ValidationError` | `ValidationError` |
| `ConfigurationError` | `ConfigurationError` |
| `RateLimitError` | `RateLimitError` (and `OperationError`) |
| `BotCreationError` | `OperationError` |
| `APIError` | `DataknobsError` only — it is the API layer's own base |

#### Handler coverage

`register_exception_handlers` registers handlers for `APIError`, for
`DataknobsError`, for FastAPI's `HTTPException`, and a catch-all for
`Exception`. The `DataknobsError` handler is what gives DataKnobs' own errors a
status matching the failure — a `ConfigurationError` from config validation, a
`RateLimitError` from `RateLimitMiddleware`, a `RecordNotFoundError` from a
database backend. You do not need to catch these and re-raise them as API
variants to get a useful status.

Resolution is by MRO, so a subclass inherits the nearest listed ancestor's row:
`RecordNotFoundError` returns 404 without appearing in the table at all.

The prerequisite is that the type is a `DataknobsError` at all — the table is
keyed on that hierarchy, and Starlette picks a handler the same way. An error
rooted at a plain `Exception` never reaches `dataknobs_error_handler`; it lands
on the `Exception` catch-all and comes back as a generic 500, whatever
happened. So when you define your own error type, subclass the common type that
describes the condition rather than `Exception`, and you inherit a sensible
status without writing a row. That choice also governs how *libraries* treat
it: retry logic keyed on `OperationError` or `ConcurrencyError` reads the same
base, so pick it for what happened, and use a row here only when the right
library base and the right status disagree.

**These cover your routes, not your whole ASGI stack.** Starlette builds
`ServerErrorMiddleware` → your middleware → `ExceptionMiddleware` → router, and
only `ExceptionMiddleware` consults the per-type handlers `register_exception_handlers`
adds. An error raised in an `app.add_middleware` layer is above that: it reaches
`ServerErrorMiddleware`, which holds the `Exception` catch-all alone, and comes
back as a generic 500. A tenant-resolving middleware raising `BotNotFoundError`
gets 500, not 404 — and this applies to `APIError` too, not just the
`DataknobsError` handler. Middleware that wants a status should return the
response instead of raising:

```python
class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            tenant = await resolve_tenant(request)
        except APIError as exc:
            return await api_error_handler(request, exc)
        return await call_next(request)
```

A route dependency is the other option — those run inside the router, so an
error raised there is handled normally.

Message and `detail` are decided separately, because the types disagree about
which half is safe — and in both directions. `NotFoundError`'s message is the
caller's own key echoed back while its `context` enumerates a registry's whole
keyspace; `ValidationError`'s `context` is the caller's own field names and
values while its message can be a database driver's.

| DataKnobs error | Status | Message disclosed? | `detail` disclosed? |
|---|---|---|---|
| `ValidationError` | 422 | yes | yes |
| `ConsentRequiredError` | 403 | yes | yes |
| `ConcurrencyError` | 409 | yes | yes |
| `InvalidTransitionError` | 409 | yes | yes |
| `RateLimitError` | 429 | yes | yes |
| `NotFoundError` | 404 | yes | no — the context lists a registry's keys |
| `TimeoutError` | 504 | yes | no — the context can carry a query |
| `ConfigurationError` | 500 | no — masked, see below | no |
| `DottedPathError` | 500 | no — masked | no |
| `DottedPathTypeError` | 500 | no — masked | no |
| `ResourceError` | 503 | no — masked | no |
| `SerializationError` | 500 | no — masked | no |
| `OperationError` | 500 | no — masked | no |
| `DataknobsError` | 500 | no — masked (terminal fallback) | no |

A withheld message is replaced by `"An unexpected error occurred"`, and a
withheld `detail` by `{}`. Both halves are logged in full whichever way the row
falls, so a diagnostic is relocated rather than lost.

**`ConfigurationError` is masked by default, and many deployments will want it
disclosed.** Most config diagnostics are authored — a key name, a sorted list
of the valid ones — and are exactly what you want back from a failing config
route. But that type is also where the funnels that wrap a third-party
*constructor* or *module import* land, and their text is unbounded: a database
or cache client raises with its connection URL, credentials included. DataKnobs
bounds its own funnels (they name the class path and the exception type, and
let `raise ... from e` carry the rest to the logs), but it cannot audit yours,
and bots are built lazily on the request path. So the default is closed.

Turn it on with one line when the route is not public — an admin API, an
internal control plane:

```python
from dataknobs_bots.api import ErrorPolicy, register_exception_handlers
from dataknobs_common.exceptions import ConfigurationError

register_exception_handlers(
    app, error_policy={ConfigurationError: ErrorPolicy(500, True, True)}
)
```

Masked or not, the diagnostic is logged in full, with its `context` — so it is
relocated rather than lost, and a config typo is diagnosable from the server
logs without disclosing anything. The line also carries `__cause__` when the
error was raised `from` another one. That matters most for the rows that are
*disclosed*: a library wrapping a failure it must not repeat leaves a
deliberately thin message and puts the real one on `__cause__` — a provider
translating a vendor error raises `ValidationError("openai API error (HTTP
400)")` with the vendor's response body chained beneath it. Logging only the
outer message would make every such failure read alike.

The same parameter gives your own `DataknobsError` subclasses a policy —
`error_policy={TenantQuotaError: ErrorPolicy(402, True, True)}` — and is merged
over the defaults, so rows you do not mention keep working. The third argument
defaults to `False`, so a row written as `ErrorPolicy(402, True)` discloses the
message and withholds the `context`: forgetting to think about `context` fails
closed.

**Adding a row means reading what that type's raise sites put in `context`, not
only what its message says.** They are separate arguments precisely because the
answer differs — a type whose message is authored for the caller may still
carry a query string or a credential in its `context`, and a type whose
`context` is the caller's own input may still relay a third party's text in its
message.

A `context` value the JSON encoder cannot represent — a `Path`, an object, a
`float("inf")` — is rendered with `str()` rather than expanded into its
attributes. It neither breaks the response (an encoder error inside the handler
would surface as the generic 500 the handler exists to replace) nor discloses
more than the value's own `__str__` says.

**The table does not govern the `dataknobs_bots.api` classes.** `APIError`
precedes every common base in their MROs, so they reach `api_error_handler`
instead — deliberately, since they are the one family authored *for* the HTTP
boundary, carrying a per-instance `status_code` and a public, overridable
`to_dict()`. Their equivalent is a `client_safe` class attribute, `True` on
`APIError` so a subclass of your own is disclosed without opting in:

```python
class TenantQuotaExceeded(APIError):
    """Disclosed, like every APIError subclass, unless it says otherwise."""
```

`BotCreationError` is the one that says otherwise. Its whole payload is a
free-text `reason`, and bots are built lazily on the request path, so a tool or
middleware constructor that fails puts *its* error text into that field. It is
masked; the reason is logged. Set `client_safe = True` on a subclass if you
author the `reason` yourself and want it returned.

This one stays a single bit where the table's is two, on purpose. A subclass
writes its message and its `detail` in the same constructor for the same
audience, so both halves have one author — unlike a table row, which governs a
type raised across several packages by people not thinking about HTTP. And
`to_dict()` is yours to override and returns whatever you put in it, so
disclosing *part* of it could only mean allow-listing keys, which would
silently drop one you added. If you want a message without a `detail`, author
the message and leave `detail` empty.

A 429 carries a `Retry-After` header — integer seconds, rounded up — whenever
the exception was given a `retry_after`, from either the API or the common
`RateLimitError`. Both also report it as `detail.retry_after`.

**One monitoring consequence.** Starlette re-raises after the `Exception`
catch-all so the ASGI server sees the failure, but not after a handler
registered for a narrower type. DataKnobs errors now take the narrower path, so
they no longer reach the server as unhandled exceptions. A deployment alerting
on that signal will see it drop; the handlers log every error they handle
instead — a 4xx at `info`, a 5xx at `error` with the traceback. The level
follows the status, not whether the error was masked: a 404 is a routine
outcome of serving traffic even when a policy override hides its message, and
a 504 is a server-side fault even though it is disclosed. Alert on the 5xx
lines.

**Registering your own handler runs the other way round.** The advice above is
for `except` clauses; handler *registration* resolves by a different rule.
Starlette walks `type(exc).__mro__` and takes the first registered match, and
`APIError` precedes the common base there — so this is shadowed for the API
variant and fires only for middleware-raised errors:

```python
app.add_exception_handler(CommonRateLimitError, my_handler)  # API variant → api_error_handler
```

Register against `APIError` (or a specific API class) to handle the API side.
The same rule is what keeps the built-in `DataknobsError` handler from taking
API traffic, and it means your own registration for a narrower type wins over
it.
<!-- --8<-- [end:catching-api-errors] -->

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
