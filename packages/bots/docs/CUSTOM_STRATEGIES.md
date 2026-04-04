# Custom Reasoning Strategies

DynaBot's reasoning strategies are modular and extensible. You can implement, register, and select custom strategies entirely through configuration — no modifications to core DynaBot code required.

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

    async def generate(self, manager, llm, tools=None, **kwargs):
        # Inject a summarization instruction into the system prompt
        messages = manager.get_messages()
        instruction = (
            f"Summarize the conversation so far in at most "
            f"{self.max_summary_tokens} tokens."
        )
        return await manager.complete(
            system_prompt_override=instruction,
            **kwargs,
        )
```

The only required method is `generate()`. It receives:

- **`manager`** — a `ReasoningManagerProtocol` (conversation history, `add_message()`, `complete()`, `stream_complete()`)
- **`llm`** — the bot's main LLM provider instance
- **`tools`** — list of registered tools (may be `None`)
- **`**kwargs`** — generation parameters (`temperature`, `max_tokens`, etc.)

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
    return StrategyCapabilities(manages_sources=True)
```

| Field | Default | Effect When `True` |
|-------|---------|-------------------|
| `manages_sources` | `False` | DynaBot performs config-driven source construction via `add_source()`, and disables redundant `auto_context` on the knowledge base. |

Only set `manages_sources=True` if your strategy uses retrieval sources (like grounded/hybrid). Most custom strategies leave this at the default.

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
async def stream_generate(self, manager, llm, tools=None, **kwargs):
    async for chunk in manager.stream_complete(**kwargs):
        yield chunk
```

### `greet()` — Custom Greeting Behavior

The default renders `greeting_template` via Jinja2. Override for dynamic greetings:

```python
async def greet(self, manager, llm, *, initial_context=None, **kwargs):
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

## Registry API

```python
from dataknobs_bots.reasoning import (
    register_strategy,       # Register a strategy class or factory
    list_strategies,         # List all registered strategy names
    get_strategy_factory,    # Get the factory for a strategy name
    is_strategy_registered,  # Check if a name is registered
    get_registry,            # Access the StrategyRegistry singleton
)

# Register (raises ValueError if already registered)
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
    from dataknobs_bots.reasoning.registry import StrategyRegistry

    fresh = StrategyRegistry()
    fresh._ensure_builtins()
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

    async def generate(self, manager, llm, tools=None, **kwargs):
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
