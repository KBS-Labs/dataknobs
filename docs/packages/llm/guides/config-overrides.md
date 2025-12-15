# Per-Request Config Overrides

Override LLM configuration on a per-request basis without modifying the provider's base config.

## Overview

Config overrides allow you to dynamically adjust LLM parameters for individual requests. This enables:

- **A/B Testing**: Test different models or parameters per request
- **Fallback Routing**: Switch to alternative models when needed
- **Cost Optimization**: Use cheaper models for simple tasks
- **Dynamic Tuning**: Adjust creativity/precision based on context

## Basic Usage

### Provider-Level Overrides

```python
from dataknobs_llm.llm import OpenAIProvider, LLMConfig

# Create provider with default config
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.7
)
llm = OpenAIProvider(config)

# Override config for a specific request
response = await llm.complete(
    "Write a creative story",
    config_overrides={
        "model": "gpt-4-turbo",  # Use different model
        "temperature": 1.2,      # More creative
        "max_tokens": 2000
    }
)

# Original config is unchanged
print(llm.config.model)  # Still "gpt-4"
print(llm.config.temperature)  # Still 0.7
```

### Streaming with Overrides

```python
async for chunk in llm.stream_complete(
    "Explain quantum physics",
    config_overrides={"model": "gpt-3.5-turbo", "temperature": 0.3}
):
    print(chunk.delta, end="", flush=True)
```

### ConversationManager Integration

```python
from dataknobs_llm.conversations import ConversationManager

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

await manager.add_message(role="user", content="Hello")

# Complete with overrides
response = await manager.complete(
    llm_config_overrides={"model": "gpt-4-turbo", "temperature": 0.9}
)

# Stream with overrides
async for chunk in manager.stream_complete(
    llm_config_overrides={"model": "gpt-3.5-turbo"}
):
    print(chunk.delta, end="")
```

## Supported Override Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Switch models per-request |
| `temperature` | `float` | Adjust creativity (0.0-2.0) |
| `max_tokens` | `int` | Control response length |
| `top_p` | `float` | Nucleus sampling parameter |
| `stop_sequences` | `List[str]` | Custom stop tokens |
| `seed` | `int` | Reproducibility seed |
| `presence_penalty` | `float` | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | `float` | Frequency penalty (-2.0 to 2.0) |
| `logit_bias` | `Dict[str, float]` | Token biases |
| `response_format` | `str` | Output format ("text" or "json") |
| `functions` | `List[Dict]` | Dynamic function definitions |
| `function_call` | `str` or `Dict` | Function calling mode |
| `options` | `Dict[str, Any]` | Provider-specific options (merged) |

## Override Presets

Register named presets for common override combinations.

### Registering Presets

```python
from dataknobs_llm.llm import AsyncLLMProvider

# Register presets (class-level, shared across all providers)
AsyncLLMProvider.register_preset("creative", {
    "temperature": 1.2,
    "top_p": 0.95,
    "presence_penalty": 0.5
})

AsyncLLMProvider.register_preset("precise", {
    "temperature": 0.1,
    "top_p": 0.9
})

AsyncLLMProvider.register_preset("fast", {
    "model": "gpt-3.5-turbo",
    "max_tokens": 500
})

AsyncLLMProvider.register_preset("json_mode", {
    "response_format": "json",
    "temperature": 0.2
})
```

### Using Presets

```python
# Use a preset
response = await llm.complete(
    "Write a poem",
    config_overrides={"preset": "creative"}
)

# Override preset values
response = await llm.complete(
    "Write a short poem",
    config_overrides={
        "preset": "creative",
        "max_tokens": 100  # Add to preset config
    }
)
```

### Managing Presets

```python
# List available presets
presets = AsyncLLMProvider.list_presets()
print(presets)  # ["creative", "precise", "fast", "json_mode"]

# Get preset config
config = AsyncLLMProvider.get_preset("creative")
print(config)  # {"temperature": 1.2, "top_p": 0.95, "presence_penalty": 0.5}
```

## Override Callbacks

Register callbacks to track override usage for logging and metrics.

### Registering Callbacks

```python
from dataknobs_llm.llm import AsyncLLMProvider, LLMConfig

def log_overrides(provider, overrides: dict, runtime_config: LLMConfig):
    """Log when overrides are applied."""
    print(f"Provider: {provider.__class__.__name__}")
    print(f"Overrides: {overrides}")
    print(f"Runtime model: {runtime_config.model}")
    print(f"Runtime temperature: {runtime_config.temperature}")

# Register callback (class-level)
AsyncLLMProvider.on_override_applied(log_overrides)
```

### Metrics Collection Example

```python
import time
from collections import defaultdict

class OverrideMetrics:
    def __init__(self):
        self.model_usage = defaultdict(int)
        self.override_counts = defaultdict(int)
        self.start_times = {}

    def track_override(self, provider, overrides: dict, runtime_config: LLMConfig):
        """Track override usage metrics."""
        self.model_usage[runtime_config.model] += 1
        for key in overrides:
            self.override_counts[key] += 1

    def get_stats(self):
        return {
            "model_usage": dict(self.model_usage),
            "override_counts": dict(self.override_counts)
        }

# Use the metrics collector
metrics = OverrideMetrics()
AsyncLLMProvider.on_override_applied(metrics.track_override)

# Make some requests...
await llm.complete("Hello", config_overrides={"model": "gpt-4-turbo"})
await llm.complete("Hi", config_overrides={"temperature": 0.5})

# Get stats
print(metrics.get_stats())
# {"model_usage": {"gpt-4-turbo": 1, "gpt-4": 1}, "override_counts": {"model": 1, "temperature": 1}}
```

### Clearing Callbacks

```python
# Clear all callbacks (important for testing)
AsyncLLMProvider.clear_override_callbacks()
```

## Options Dict Merging

The `options` field is shallowly merged with the base config's options:

```python
from dataknobs_llm.llm.providers import EchoProvider

# Base config with options
config = LLMConfig(
    provider="echo",
    model="echo-model",
    options={
        "echo_prefix": "Response: ",
        "mock_tokens": True,
        "embedding_dim": 768
    }
)
llm = EchoProvider(config)

# Override merges with base options
response = await llm.complete(
    "Hello",
    config_overrides={
        "options": {"echo_prefix": "Custom: "}  # Override one option
    }
)

# Effective options: {"echo_prefix": "Custom: ", "mock_tokens": True, "embedding_dim": 768}
```

## Error Handling

### Invalid Override Fields

```python
try:
    response = await llm.complete(
        "Hello",
        config_overrides={"invalid_field": "value"}
    )
except ValueError as e:
    print(e)
    # "Unsupported config overrides: {'invalid_field'}.
    #  Allowed fields: {'model', 'temperature', ...}"
```

### Unknown Preset

```python
try:
    response = await llm.complete(
        "Hello",
        config_overrides={"preset": "nonexistent"}
    )
except ValueError as e:
    print(e)
    # "Unknown preset: 'nonexistent'. Available presets: ['creative', 'precise']"
```

## Use Cases

### A/B Testing Models

```python
import random

async def ab_test_completion(prompt: str, user_id: str):
    """A/B test between GPT-4 and GPT-4-Turbo."""
    # Deterministic assignment based on user
    use_turbo = hash(user_id) % 100 < 50

    model = "gpt-4-turbo" if use_turbo else "gpt-4"

    response = await llm.complete(
        prompt,
        config_overrides={"model": model}
    )

    # Log for analysis
    log_ab_result(user_id, model, response)
    return response
```

### Fallback Routing

```python
async def complete_with_fallback(prompt: str) -> str:
    """Try GPT-4, fallback to GPT-3.5 on error."""
    try:
        response = await llm.complete(prompt)
        return response.content
    except Exception as e:
        # Fallback to cheaper model
        response = await llm.complete(
            prompt,
            config_overrides={"model": "gpt-3.5-turbo"}
        )
        return response.content
```

### Task-Specific Configuration

```python
async def process_task(task_type: str, content: str):
    """Use different configs based on task type."""
    presets = {
        "creative_writing": "creative",
        "code_review": "precise",
        "quick_answer": "fast",
        "data_extraction": "json_mode"
    }

    preset = presets.get(task_type, "precise")

    response = await llm.complete(
        content,
        config_overrides={"preset": preset}
    )
    return response.content
```

### Cost Optimization

```python
async def smart_complete(prompt: str, complexity: str = "low"):
    """Use appropriate model based on task complexity."""
    if complexity == "high":
        overrides = {"model": "gpt-4", "max_tokens": 2000}
    elif complexity == "medium":
        overrides = {"model": "gpt-4-turbo", "max_tokens": 1000}
    else:
        overrides = {"model": "gpt-3.5-turbo", "max_tokens": 500}

    return await llm.complete(prompt, config_overrides=overrides)
```

## Best Practices

1. **Use Presets for Common Configurations**: Define presets for frequently used override combinations to ensure consistency and reduce errors.

2. **Track Override Usage**: Register callbacks to monitor which overrides are being used and their impact on performance/cost.

3. **Clear Callbacks in Tests**: Always clear callbacks in test fixtures to prevent interference between tests.

4. **Validate Override Fields**: The system validates override fields automatically, but document which overrides your application uses.

5. **Don't Modify Base Config**: Overrides create a temporary config; the original is preserved. Don't try to use overrides as a way to permanently change configuration.

## API Reference

### ConfigOverrideMixin

Both `AsyncLLMProvider` and `SyncLLMProvider` inherit from `ConfigOverrideMixin`:

```python
class ConfigOverrideMixin:
    ALLOWED_CONFIG_OVERRIDES: Set[str]

    @classmethod
    def register_preset(cls, name: str, overrides: Dict[str, Any]) -> None: ...

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any] | None: ...

    @classmethod
    def list_presets(cls) -> List[str]: ...

    @classmethod
    def on_override_applied(
        cls,
        callback: Callable[[Any, Dict[str, Any], LLMConfig], None]
    ) -> None: ...

    @classmethod
    def clear_override_callbacks(cls) -> None: ...
```

## See Also

- [Conversation Management](conversations.md) - Using overrides with ConversationManager
- [Performance Guide](performance.md) - Optimizing LLM usage
- [Versioning & A/B Testing](versioning.md) - Prompt-level A/B testing
