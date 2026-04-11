# Custom Reasoning Strategies

DynaBot's reasoning strategies are modular and extensible. You can implement, register, and select custom strategies entirely through configuration — no modifications to core DynaBot code required.

> **Migration note:** The strategy registry now uses `PluginRegistry` from
> `dataknobs-common`. Exception types have changed:
>
> - **Duplicate registration** now raises `OperationError` (was `ValueError`)
> - **Unknown strategy** now raises `NotFoundError` (was `ValueError`)
>
> Both are subclasses of `DataknobsError` from `dataknobs_common.exceptions`.
> Code that catches `ValueError` from `register_strategy()` or
> `create_reasoning_from_config()` should be updated.

## Built-in Strategies

| Strategy | Key | Use Case |
|----------|-----|----------|
| **Simple** | `simple` | Direct LLM call — fast, no tools |
| **ReAct** | `react` | Reason + Act loop with tool calls |
| **Wizard** | `wizard` | FSM-driven guided data collection |
| **Grounded** | `grounded` | Deterministic multi-source KB retrieval |
| **Hybrid** | `hybrid` | Grounded retrieval + ReAct tool use |

## Creating a Custom Strategy

### Step 1: Subclass `ReasoningStrategy`

```python
from dataknobs_bots.reasoning import ReasoningStrategy, StrategyCapabilities


class SummarizeReasoning(ReasoningStrategy):
    """Strategy that always summarizes the conversation so far."""

    def __init__(
        self,
        *,
        greeting_template: str | None = None,
        max_summary_tokens: int = 200,
    ) -> None:
        super().__init__(greeting_template=greeting_template)
        self.max_summary_tokens = max_summary_tokens

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Inject a summarization instruction into the system prompt
        messages = manager.get_messages()
        instruction = (
            f"Summarize the conversation so far in at most "
            f"{self.max_summary_tokens} tokens."
        )
        return await manager.complete(
            system_prompt_override=instruction,
            tools=tools,
            **kwargs,
        )
```

The only required method is `generate()`. It receives:

- **`manager`** — a `ReasoningManagerProtocol` (conversation history, `add_message()`, `complete()`, `stream_complete()`)
- **`llm`** — the bot's main LLM provider instance
- **`tools`** — list of registered tools (may be `None`)
- **`**kwargs`** — generation parameters (`temperature`, `max_tokens`, etc.)

**Tool execution:** Pass `tools=tools` to `manager.complete()` so the
LLM can see available tools. If the LLM returns `tool_calls` in its
response, DynaBot automatically executes them and re-calls the LLM —
your strategy does not need to handle tool execution itself. This
fallback loop runs after `generate()` returns, with configurable
iteration limits and timeouts (`max_tool_iterations`,
`tool_loop_timeout`).

Strategies that handle tool execution internally (like ReAct, which
runs its own reason-act loop) should consume the `tool_calls` before
returning, so DynaBot's fallback loop sees no pending calls and
becomes a no-op. Record any tool executions via
`self._tool_executions.append(ToolExecution(...))` so DynaBot can
fire `on_tool_executed` middleware hooks.

### Step 2: Override `from_config`

The `from_config` classmethod lets DynaBot create your strategy from a YAML/JSON config dict:

```python
from typing import Any


class SummarizeReasoning(ReasoningStrategy):
    # ... __init__ and generate as above ...

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> "SummarizeReasoning":
        return cls(
            greeting_template=config.get("greeting_template"),
            max_summary_tokens=config.get("max_summary_tokens", 200),
        )
```

The `**kwargs` may contain context from DynaBot (e.g. `knowledge_base`). Ignore what you don't need.

### Step 3: Register the Strategy

```python
from dataknobs_bots.reasoning import register_strategy

register_strategy("summarize", SummarizeReasoning)
```

Registration should happen at application startup, before any bot configs reference the strategy.

### Step 4: Use It in Configuration

As the bot's primary reasoning strategy:

```yaml
llm:
  provider: ollama
  model: llama3.2

reasoning:
  strategy: summarize
  max_summary_tokens: 150
  greeting_template: "Hello! I'll summarize our conversation as we go."

conversation_storage:
  backend: memory
```

Or in specific wizard stages via per-state strategy injection:

```yaml
reasoning:
  strategy: wizard
  wizard_config:
    settings:
      tool_reasoning: single
    stages:
      - name: research
        reasoning: summarize          # Use custom strategy for this stage
        reasoning_config:             # Strategy-specific config
          max_summary_tokens: 100
        tools: [search_docs]
        transitions:
          - target: review
```

Any registered strategy can be referenced by name in a wizard stage's `reasoning`
field. The optional `reasoning_config` dict is forwarded to the strategy's
`from_config()`.

```python
bot = await DynaBot.from_config(config)
response = await bot.chat("Tell me about the project status", context)
```

## Optional Overrides

### `capabilities()` — Declare Autonomous Behavior

Override `capabilities()` to tell DynaBot what your strategy manages. This controls which orchestration steps DynaBot performs on your behalf.

```python
@classmethod
def capabilities(cls) -> StrategyCapabilities:
    return StrategyCapabilities(manages_sources=True, manages_tools=True)
```

| Field | Default | Effect When `True` |
|-------|---------|-------------------|
| `manages_sources` | `False` | DynaBot performs config-driven source construction via `add_source()`, and disables redundant `auto_context` on the knowledge base. |
| `manages_tools` | `False` | Strategy runs its own tool execution loop. In wizard stages, collected artifact context is pre-injected into the system prompt so the LLM can see collected data without making tool calls. |

Set `manages_sources=True` if your strategy uses retrieval sources (like grounded/hybrid). Set `manages_tools=True` if your strategy manages its own tool execution loop (like ReAct/hybrid). Most custom strategies leave both at the default.

### `get_source_configs()` — Custom Source Config Layout

If your strategy declares `manages_sources=True`, DynaBot reads source definitions from the config and calls `add_source()` for each one. By default, sources are read from a top-level `"sources"` key:

```yaml
reasoning:
  strategy: my_strategy
  sources:
    - name: kb
      source_type: vector_kb
```

If your strategy nests sources differently, override `get_source_configs()`:

```python
@classmethod
def get_source_configs(cls, config: dict[str, Any]) -> list[dict[str, Any]]:
    # Sources live under "retrieval.sources" in our config layout
    return config.get("retrieval", {}).get("sources", [])
```

### `add_source()` — Receive Constructed Sources

Required if `manages_sources=True`. DynaBot calls this for each source it constructs from config:

```python
def add_source(self, source) -> None:
    self._sources.append(source)
```

### `stream_generate()` — True Token-Level Streaming

The default `stream_generate()` wraps `generate()` and yields the complete response as a single chunk. Override for true streaming:

```python
async def stream_generate(
    self,
    manager: Any,
    llm: Any,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    async for chunk in manager.stream_complete(tools=tools, **kwargs):
        yield chunk
```

### `greet()` — Custom Greeting Behavior

The default renders `greeting_template` via Jinja2. Override for dynamic greetings:

```python
async def greet(
    self,
    manager: Any,
    llm: Any,
    *,
    initial_context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any | None:
    # Generate a greeting using the LLM
    await manager.add_message(role="user", content="Greet the user briefly.")
    return await manager.complete(**kwargs)
```

### `providers()` and `set_provider()` — Internal LLM Providers

If your strategy creates its own LLM providers (e.g. a separate model for query generation), expose them so DynaBot can register them in the provider catalog:

```python
def providers(self) -> dict[str, Any]:
    return {"query_generation": self._query_provider}

def set_provider(self, role: str, provider: Any) -> bool:
    if role == "query_generation":
        self._query_provider = provider
        return True
    return False
```

### `close()` — Resource Cleanup

Override if your strategy holds resources (connections, providers, background tasks):

```python
async def close(self) -> None:
    if self._query_provider:
        await self._query_provider.close()
```

### Phased Turn Execution (`PhasedReasoningProtocol`)

By default, DynaBot calls `generate()` as a single opaque call, then runs
its own tool execution loop on any `tool_calls` in the response.  For
strategies that need DynaBot to interleave tool execution *within* the
generation lifecycle (e.g. to update wizard state from tool results before
saving), the `PhasedReasoningProtocol` splits the turn into three phases:

```
DynaBot.chat()
  → strategy.begin_turn(manager, llm, tools, **kwargs)
      ← TurnHandle (or early_response for navigation/amendments)
  → strategy.process_input(handle)
      ← ProcessResult (or early_response for clarification/validation)
  → [DynaBot tool execution — when ProcessResult.needs_tool_execution]
  → strategy.finalize_turn(handle, tool_results)
      ← LLM response
```

DynaBot detects phased support via `isinstance(strategy,
PhasedReasoningProtocol)` and uses the phased path automatically.
Non-phased strategies continue using the single `generate()` call.

When `process_input` sets `needs_tool_execution=True` and populates
`pending_tool_calls` with `ToolCallSpec` objects, DynaBot executes those
tools between `process_input` and `finalize_turn`. The tool results
(as `list[ToolExecution]`) are passed to `finalize_turn`. This is used
by wizard stages with `tool_result_mapping` to populate state from tool
results before FSM transition evaluation.

**When to implement phased execution:**

- Your strategy has complex internal state (like an FSM) that needs to
  reflect tool results before being saved
- You need tool execution to happen at a specific point in the generation
  lifecycle, not after the entire response is produced
- Your strategy has multiple early-return paths (navigation, validation,
  clarification) that should bypass tool execution

**When NOT to implement it:**

- Simple strategies that just call `manager.complete()` — use `generate()`
- Strategies with internal tool loops (like ReAct) — manage tools yourself
  via `self._tool_executions`
- Most custom strategies — the single `generate()` call is sufficient

**Implementation:**

```python
from dataknobs_bots.reasoning.base import (
    PhasedReasoningProtocol,
    ProcessResult,
    ToolCallSpec,
    TurnHandle,
)


class MyPhasedStrategy(ReasoningStrategy):

    async def begin_turn(self, manager, llm, tools=None, **kwargs):
        handle = TurnHandle(manager=manager, llm=llm, tools=tools, kwargs=kwargs)
        # Restore state, handle navigation...
        # Set handle.early_response to short-circuit
        return handle

    async def process_input(self, handle):
        result = ProcessResult()
        # Extract data, validate...
        # Set result.early_response for clarification/errors
        # To request tool execution before finalize_turn:
        #   result.pending_tool_calls = [ToolCallSpec(tool_name="...", parameters={...})]
        #   result.needs_tool_execution = True
        return result

    async def finalize_turn(self, handle, tool_results=None):
        # tool_results is a list[ToolExecution] when tools ran, None otherwise
        # Process tool_results, transition state, generate response, save
        return response

    async def generate(self, manager, llm, tools=None, **kwargs):
        """Backward-compatible wrapper."""
        handle = await self.begin_turn(manager, llm, tools, **kwargs)
        if handle.early_response:
            return handle.early_response
        result = await self.process_input(handle)
        if result.early_response:
            return result.early_response
        return await self.finalize_turn(handle)
```

`WizardReasoning` is currently the only built-in strategy that implements
the phased protocol. Subclass `TurnHandle` to carry strategy-specific
state between phases (see `WizardTurnHandle` for an example).

#### Streaming Phased Execution (`StreamingPhasedProtocol`)

Strategies that implement `StreamingPhasedProtocol` (extends
`PhasedReasoningProtocol`) add `stream_finalize_turn()`, which yields
`LLMStreamResponse` chunks instead of returning a complete response.
`DynaBot.stream_chat()` detects this and streams the finalize phase
token-by-token while keeping `begin_turn` and `process_input` blocking.

```python
from collections.abc import AsyncIterator
from dataknobs_llm import LLMStreamResponse

class MyStreamingPhasedStrategy(ReasoningStrategy):

    # begin_turn and process_input are the same as above

    async def stream_finalize_turn(self, handle, tool_results=None):
        manager = handle.manager
        # Pre-stream work (transitions, state updates)
        # ...
        # Yield streaming chunks
        async for chunk in manager.stream_complete(...):
            yield chunk
        # Post-stream work (save state — only if fully consumed)
```

`WizardReasoning` implements `StreamingPhasedProtocol`.

## Registry API

```python
from dataknobs_bots.reasoning import (
    register_strategy,       # Register a strategy class or factory
    list_strategies,         # List all registered strategy names
    get_strategy_factory,    # Get the factory for a strategy name
    is_strategy_registered,  # Check if a name is registered
    get_registry,            # Access the PluginRegistry singleton
)

# Register (raises OperationError if already registered)
register_strategy("my_strategy", MyStrategy)

# Register with override
register_strategy("simple", MyBetterSimple, override=True)

# Factory functions work too
def my_factory(config, **kwargs):
    return MyStrategy(param=config["param"])

register_strategy("my_factory_strategy", my_factory)

# Introspection
list_strategies()         # ['grounded', 'hybrid', 'my_strategy', 'react', ...]
is_strategy_registered("my_strategy")  # True
get_strategy_factory("simple")         # <class 'SimpleReasoning'>
```

## Testing Custom Strategies

Use `BotTestHarness` for end-to-end testing:

```python
import pytest
from dataknobs_bots.reasoning import register_strategy, get_registry
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.testing import text_response


@pytest.fixture(autouse=True)
def _register(monkeypatch):
    """Register custom strategy in an isolated registry."""
    import dataknobs_bots.reasoning.registry as reg_module
    from dataknobs_common.registry import PluginRegistry
    from dataknobs_bots.reasoning import ReasoningStrategy
    from dataknobs_bots.reasoning.registry import _register_builtins

    fresh = PluginRegistry[ReasoningStrategy](
        "test_strategies",
        validate_type=ReasoningStrategy,
        canonicalize_keys=True,
        config_key="strategy",
        config_key_default="simple",
        on_first_access=_register_builtins,
    )
    fresh.register("summarize", SummarizeReasoning)
    monkeypatch.setattr(reg_module, "_registry", fresh)


@pytest.mark.asyncio()
async def test_summarize_strategy():
    async with await BotTestHarness.create(
        bot_config={
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "summarize",
                "max_summary_tokens": 100,
            },
        },
        main_responses=[text_response("Here is a summary...")],
    ) as harness:
        result = await harness.chat("What happened so far?")
        assert result.response == "Here is a summary..."
```

Use `monkeypatch` to isolate the registry — this prevents test pollution of the global singleton and is safe for parallel test execution.

## Complete Example

```python
"""Custom sentiment-aware reasoning strategy."""

from typing import Any

from dataknobs_bots.reasoning import (
    ReasoningStrategy,
    StrategyCapabilities,
    register_strategy,
)


class SentimentReasoning(ReasoningStrategy):
    """Adjusts system prompt based on detected user sentiment."""

    def __init__(
        self,
        *,
        greeting_template: str | None = None,
        positive_prompt: str = "The user seems happy. Be enthusiastic.",
        negative_prompt: str = "The user seems frustrated. Be empathetic.",
        neutral_prompt: str = "Respond naturally.",
    ) -> None:
        super().__init__(greeting_template=greeting_template)
        self._prompts = {
            "positive": positive_prompt,
            "negative": negative_prompt,
            "neutral": neutral_prompt,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> "SentimentReasoning":
        return cls(
            greeting_template=config.get("greeting_template"),
            positive_prompt=config.get("positive_prompt", "Be enthusiastic."),
            negative_prompt=config.get("negative_prompt", "Be empathetic."),
            neutral_prompt=config.get("neutral_prompt", "Respond naturally."),
        )

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = manager.get_messages()
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )

        # Simple keyword-based sentiment (real implementation would use LLM)
        sentiment = "neutral"
        if any(w in last_user_msg.lower() for w in ("thanks", "great", "love")):
            sentiment = "positive"
        elif any(w in last_user_msg.lower() for w in ("frustrated", "broken", "hate")):
            sentiment = "negative"

        return await manager.complete(
            system_prompt_override=self._prompts[sentiment],
            tools=tools,
            **kwargs,
        )


# Register at application startup
register_strategy("sentiment", SentimentReasoning)
```

```yaml
# config.yaml
llm:
  provider: ollama
  model: llama3.2

reasoning:
  strategy: sentiment
  positive_prompt: "The user is happy! Match their energy."
  negative_prompt: "The user needs support. Be patient and helpful."

conversation_storage:
  backend: memory
```
