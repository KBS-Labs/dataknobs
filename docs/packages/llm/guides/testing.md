# Testing Utilities

The `dataknobs_llm.testing` module provides convenience builders for creating test LLM
responses, designed for use with `EchoProvider` in unit and integration tests.

## Overview

Testing LLM-dependent code requires scripting specific responses, including tool calls.
While `EchoProvider` supports this via `set_responses()`, creating `LLMResponse` objects
manually is verbose. The testing utilities provide a cleaner API.

**Before (verbose):**
```python
from dataknobs_llm.llm.base import LLMResponse, ToolCall

provider.set_responses([
    LLMResponse(
        content="",
        model="echo",
        finish_reason="tool_calls",
        tool_calls=[ToolCall(name="preview_config", parameters={}, id="tc-1")]
    ),
    LLMResponse(
        content="Preview complete!",
        model="echo",
        finish_reason="stop"
    )
])
```

**After (clean):**
```python
from dataknobs_llm.testing import tool_call_response, text_response

provider.set_responses([
    tool_call_response("preview_config", {}),
    text_response("Preview complete!"),
])
```

## Quick Start

```python
from dataknobs_llm.llm.providers import EchoProvider
from dataknobs_llm.testing import tool_call_response, text_response

# Create test provider
provider = EchoProvider({"provider": "echo", "model": "test"})

# Script responses
provider.set_responses([
    tool_call_response("get_weather", {"city": "NYC"}),
    text_response("The weather in NYC is sunny!"),
])

# Use in tests
response = await provider.complete("What's the weather?")
assert response.tool_calls[0].name == "get_weather"
```

## Response Builders

### text_response

Create a simple text response:

```python
from dataknobs_llm.testing import text_response

response = text_response("Hello, world!")
# response.content == "Hello, world!"
# response.finish_reason == "stop"

# With options
response = text_response(
    "Hello!",
    model="custom-model",
    finish_reason="length",
    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    metadata={"request_id": "abc123"},
)
```

### tool_call_response

Create a response with tool call(s):

```python
from dataknobs_llm.testing import tool_call_response

# Single tool call
response = tool_call_response("preview_config", {"format": "yaml"})
# response.tool_calls[0].name == "preview_config"
# response.tool_calls[0].parameters == {"format": "yaml"}
# response.finish_reason == "tool_calls"

# With custom tool ID
response = tool_call_response("test", {}, tool_id="my-custom-id")

# With accompanying text content
response = tool_call_response(
    "preview_config",
    {},
    content="Let me preview that for you.",
)

# Multiple tool calls
response = tool_call_response(
    "first_tool", {"a": 1},
    additional_tools=[
        ("second_tool", {"b": 2}),
        ("third_tool", {}),
    ]
)
# len(response.tool_calls) == 3
```

### multi_tool_response

Create a response with multiple tool calls (cleaner than `additional_tools`):

```python
from dataknobs_llm.testing import multi_tool_response

response = multi_tool_response([
    ("preview_config", {}),
    ("validate_config", {"strict": True}),
])
# len(response.tool_calls) == 2
```

### extraction_response

Create a response for schema extraction (JSON content):

```python
from dataknobs_llm.testing import extraction_response
import json

response = extraction_response({"name": "Math Tutor", "level": 5})
data = json.loads(response.content)
# data == {"name": "Math Tutor", "level": 5}
```

## ResponseSequenceBuilder

For complex multi-turn sequences, use the fluent builder API:

```python
from dataknobs_llm.testing import ResponseSequenceBuilder

responses = (
    ResponseSequenceBuilder()
    .add_tool_call("list_templates", {})
    .add_text("I found these templates: quiz, tutor, assistant")
    .add_tool_call("get_template", {"name": "quiz"})
    .add_text("The quiz template is perfect for assessments!")
    .build()
)

provider.set_responses(responses)
```

### Builder Methods

| Method | Description |
|--------|-------------|
| `add_text(content)` | Add a text response |
| `add_tool_call(name, args)` | Add a single tool call response |
| `add_multi_tool(tools)` | Add a multi-tool response |
| `add_extraction(data)` | Add a JSON extraction response |
| `add(response)` | Add a custom `LLMResponse` |
| `build()` | Return the list of responses |
| `configure(provider)` | Set responses on an EchoProvider |

### Configure Provider Directly

```python
from dataknobs_llm.llm.providers import EchoProvider
from dataknobs_llm.testing import ResponseSequenceBuilder

provider = EchoProvider({"provider": "echo", "model": "test"})

(
    ResponseSequenceBuilder()
    .add_extraction({"stage": "identity", "domain_id": "math-tutor"})
    .add_tool_call("preview_config", {"format": "yaml"})
    .add_text("Configuration complete!")
    .configure(provider)  # Sets responses directly
)
```

### Custom Model Name

```python
responses = (
    ResponseSequenceBuilder(model="gpt-4-test")
    .add_text("Hello")
    .add_text("World")
    .build()
)
# All responses have model="gpt-4-test"
```

## Testing Patterns

### Wizard Flow Testing

```python
from dataknobs_llm.llm.providers import EchoProvider
from dataknobs_llm.testing import extraction_response, tool_call_response, text_response

provider = EchoProvider({"provider": "echo", "model": "test"})

# Simulate wizard stages: extraction -> tool call -> completion
provider.set_responses([
    extraction_response({"domain_id": "math-tutor", "domain_name": "Math Tutor"}),
    tool_call_response("preview_config", {"format": "yaml"}),
    text_response("Your bot is configured!"),
])
```

### ReAct Loop Testing

```python
from dataknobs_llm.testing import ResponseSequenceBuilder

# Simulate ReAct: thought -> action -> observation -> final answer
(
    ResponseSequenceBuilder()
    .add_tool_call("search", {"query": "Python async"})
    .add_tool_call("read_doc", {"url": "docs.python.org/asyncio"})
    .add_text("Based on my research, here's how async works...")
    .configure(provider)
)
```

### Multi-Tool Parallel Execution

```python
from dataknobs_llm.testing import multi_tool_response, text_response

# LLM calls multiple tools in one turn
provider.set_responses([
    multi_tool_response([
        ("preview_config", {}),
        ("validate_config", {"strict": True}),
    ]),
    text_response("Preview and validation complete!"),
])
```

## EchoProvider Features

The testing utilities work with `EchoProvider`, which offers additional features:

### Pattern Matching

```python
provider = EchoProvider({"provider": "echo", "model": "test"})

# Add pattern-based responses
provider.add_pattern_response(r"hello|hi", "Hello! How can I help?")
provider.add_pattern_response(r"weather", "I don't have weather data.")

# Falls back to pattern matching when response queue is empty
response = await provider.complete("hello there")
# response.content == "Hello! How can I help?"
```

### Call Tracking

```python
provider = EchoProvider({"provider": "echo", "model": "test"})
provider.set_responses([text_response("Response 1"), text_response("Response 2")])

await provider.complete("First message")
await provider.complete("Second message")

# Check call count
assert provider.call_count == 2
```

## Imports

All testing utilities are available from the main package:

```python
# Recommended
from dataknobs_llm.testing import (
    text_response,
    tool_call_response,
    multi_tool_response,
    extraction_response,
    ResponseSequenceBuilder,
)

# Also available from top-level
from dataknobs_llm import text_response, tool_call_response
```
