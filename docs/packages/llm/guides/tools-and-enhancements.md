# Tools, Cost Tracking, and Advanced Features

This guide covers the enhanced features added to dataknobs-llm for building sophisticated LLM applications.

## Overview

Recent enhancements to the LLM package include:

- **Tool/Function Calling**: Define tools that LLMs can call during generation
- **Cost Tracking**: Automatic token usage cost calculation and tracking
- **Rate Limiting**: Request throttling with sliding window algorithm
- **Metadata Management**: Store and retrieve conversation metadata
- **Configuration Cloning**: Runtime parameter overrides without mutation

## Tool System

The tool system provides abstractions for LLM function calling, enabling LLMs to call external functions during generation.

### Defining Tools

Create a tool by subclassing the `Tool` base class:

```python
from dataknobs_llm import Tool
from typing import Dict, Any

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations",
            metadata={"category": "math", "version": "1.0"}
        )

    @property
    def schema(self) -> Dict[str, Any]:
        """Define the tool's parameter schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(self, operation: str, a: float, b: float) -> float:
        """Execute the tool with given parameters."""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
```

### Using the Tool Registry

The `ToolRegistry` provides centralized management of available tools:

```python
from dataknobs_llm import ToolRegistry

# Create registry
registry = ToolRegistry()

# Register tools
registry.register(CalculatorTool())
registry.register(WebSearchTool())
registry.register(FileReadTool())

# List available tools
print(f"Available tools: {registry.get_tool_names()}")
# Output: ['calculator', 'web_search', 'file_read']

# Get tool definitions for LLM function calling
functions = registry.to_function_definitions()

# Or for Anthropic Claude format
tools = registry.to_anthropic_tool_definitions()

# Execute a tool
result = await registry.execute_tool(
    "calculator",
    operation="multiply",
    a=7,
    b=6
)
print(result)  # Output: 42
```

### Filtering Tools

Filter tools by metadata or name:

```python
# Get only specific tools
math_functions = registry.to_function_definitions(
    include_only={"calculator", "statistics"}
)

# Exclude dangerous tools
safe_functions = registry.to_function_definitions(
    exclude={"file_write", "system_command"}
)

# Filter by metadata
math_tools = registry.filter_by_metadata(category="math")
safe_tools = registry.filter_by_metadata(safe=True)
```

### Tool Validation

Tools automatically validate parameters against the schema:

```python
tool = registry.get_tool("calculator")

# Check if parameters are valid
is_valid = tool.validate_parameters(
    operation="add",
    a=5,
    b=3
)
print(is_valid)  # Output: True

# Missing required parameter
is_valid = tool.validate_parameters(operation="add", a=5)
print(is_valid)  # Output: False
```

## Cost Tracking

Automatic cost tracking helps monitor LLM usage and expenses.

### Enabling Cost Tracking

Cost tracking is enabled automatically when using `ConversationManager`:

```python
from dataknobs_llm import create_llm_provider, LLMConfig
from dataknobs_llm.conversations import ConversationManager

config = LLMConfig(provider="openai", model="gpt-4")
llm = create_llm_provider(config)

manager = await ConversationManager.create(
    conversation_id="conv-123",
    llm=llm,
    storage=storage
)

# Complete a turn - cost is automatically tracked
response = await manager.complete()

# Access cost information
print(f"This request cost: ${response.cost_usd:.4f}")
print(f"Total conversation cost: ${response.cumulative_cost_usd:.4f}")
```

### Getting Conversation Costs

Retrieve accumulated costs for a conversation:

```python
# Get total cost for the conversation
total_cost = manager.get_total_cost()
print(f"Total cost: ${total_cost:.4f}")

# Cost is stored in message metadata
current_node = manager.state.get_current_node()
cost = current_node.data.metadata.get('cost_usd', 0.0)
cumulative = current_node.data.metadata.get('cumulative_cost_usd', 0.0)
```

### Manual Cost Calculation

For direct LLM usage without ConversationManager:

```python
from dataknobs_llm.llm.utils import CostCalculator

# Make LLM call
response = await llm.complete(messages)

# Calculate cost
cost = CostCalculator.calculate_cost(
    response,
    model=response.model
)

print(f"Cost: ${cost:.4f}")
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
```

### Cost Information in Response

Cost data is available in `LLMResponse`:

```python
response = await manager.complete()

# Individual request cost
if response.cost_usd is not None:
    print(f"Request cost: ${response.cost_usd:.4f}")

# Running total for conversation
if response.cumulative_cost_usd is not None:
    print(f"Total cost so far: ${response.cumulative_cost_usd:.4f}")

# Token usage
if response.usage:
    print(f"Tokens used: {response.usage.total_tokens}")
```

## Rate Limiting

Rate limiting prevents excessive LLM usage and controls costs.

### Setting Up Rate Limiting

Add `RateLimitMiddleware` to your conversation:

```python
from dataknobs_llm.conversations import (
    ConversationManager,
    RateLimitMiddleware,
    RateLimitError
)

# Create rate limiter: max 10 requests per 60 seconds
rate_limiter = RateLimitMiddleware(
    max_requests=10,
    window_seconds=60,
    scope="conversation"  # or "client"
)

# Create manager with middleware
manager = await ConversationManager.create(
    conversation_id="conv-123",
    llm=llm,
    storage=storage,
    middleware=[rate_limiter]
)

# Use normally - rate limiting applies automatically
try:
    response = await manager.complete()
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
```

### Rate Limiting Scopes

Choose between conversation-level or client-level limiting:

```python
# Per-conversation rate limiting (default)
conv_limiter = RateLimitMiddleware(
    max_requests=10,
    window_seconds=60,
    scope="conversation"
)

# Per-client rate limiting (across all conversations)
client_limiter = RateLimitMiddleware(
    max_requests=100,
    window_seconds=3600,  # 100 requests per hour
    scope="client"
)
```

### Custom Rate Limit Keys

Provide a custom function to determine rate limit grouping:

```python
def get_rate_limit_key(state):
    """Group rate limits by user tier."""
    user_tier = state.metadata.get('user_tier', 'free')
    return f"tier:{user_tier}"

rate_limiter = RateLimitMiddleware(
    max_requests=50,
    window_seconds=3600,
    key_fn=get_rate_limit_key
)
```

### Handling Rate Limits

Catch and handle rate limit errors gracefully:

```python
from dataknobs_llm.conversations import RateLimitError
import asyncio

async def complete_with_retry(manager, max_retries=3):
    """Complete with automatic retry on rate limit."""
    for attempt in range(max_retries):
        try:
            return await manager.complete()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 60  # Wait before retry
                print(f"Rate limited, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

## Metadata Management

Store and retrieve custom metadata for conversations.

### Setting Metadata

Store conversation-level metadata:

```python
# Set metadata values
manager.set_metadata('user_id', 'user-123')
manager.set_metadata('session_type', 'support')
manager.set_metadata('tags', ['urgent', 'billing'])

# Metadata is automatically persisted with the conversation
```

### Getting Metadata

Retrieve metadata values:

```python
# Get specific metadata value
user_id = manager.get_metadata('user_id')

# Get with default
priority = manager.get_metadata('priority', default='normal')

# Get all metadata
all_metadata = manager.get_metadata()
print(all_metadata)
# Output: {'user_id': 'user-123', 'session_type': 'support', ...}
```

### Message-Level Metadata

Add metadata when completing a turn:

```python
# Add metadata to specific messages
response = await manager.complete(
    metadata={
        'category': 'question',
        'sentiment': 'neutral',
        'importance': 'high'
    }
)

# Retrieve from message node
node = manager.state.get_current_node()
category = node.data.metadata.get('category')
```

## Configuration Cloning

Clone LLM configurations with runtime parameter overrides.

### Basic Cloning

Create configuration variants without mutation:

```python
from dataknobs_llm import LLMConfig

# Base configuration
base_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Create variant for creative tasks
creative_config = base_config.clone(temperature=1.0)

# Create variant for precise tasks
precise_config = base_config.clone(temperature=0.1, max_tokens=500)

# Original unchanged
print(base_config.temperature)  # Output: 0.7
```

### Use Cases

Configuration cloning is useful for:

1. **Multi-tenant applications**: Different configs per client
2. **A/B testing**: Compare model configurations
3. **Dynamic adaptation**: Adjust parameters based on context
4. **Cost optimization**: Switch models based on complexity

Example:

```python
def get_llm_config(task_complexity: str) -> LLMConfig:
    """Select configuration based on task complexity."""
    base = LLMConfig(provider="openai")

    if task_complexity == "simple":
        return base.clone(model="gpt-3.5-turbo", temperature=0.3)
    elif task_complexity == "complex":
        return base.clone(model="gpt-4", temperature=0.7)
    else:
        return base.clone(model="gpt-4o", temperature=0.5)
```

## Complete Example

Putting it all together:

```python
from dataknobs_llm import (
    create_llm_provider,
    LLMConfig,
    ToolRegistry,
)
from dataknobs_llm.conversations import (
    ConversationManager,
    RateLimitMiddleware,
    RateLimitError
)

# Set up tools
registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(SearchTool())

# Configure LLM with tools
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.7
)
llm = create_llm_provider(config)

# Set up rate limiting
rate_limiter = RateLimitMiddleware(
    max_requests=20,
    window_seconds=60
)

# Create conversation manager
manager = await ConversationManager.create(
    conversation_id="conv-456",
    llm=llm,
    storage=storage,
    middleware=[rate_limiter]
)

# Set metadata
manager.set_metadata('user_id', 'user-789')
manager.set_metadata('session_type', 'research')

# Add user message
await manager.add_message(
    role="user",
    content="What is 42 times 17?"
)

# Complete with function calling
try:
    response = await manager.complete()

    print(f"Response: {response.content}")
    print(f"Cost: ${response.cost_usd:.4f}")
    print(f"Total conversation cost: ${response.cumulative_cost_usd:.4f}")

    # Check if tools were used
    if response.function_call:
        print(f"Function called: {response.function_call['name']}")

except RateLimitError:
    print("Rate limit exceeded, please try again later")

# Get conversation summary
total_cost = manager.get_total_cost()
user_id = manager.get_metadata('user_id')
print(f"Conversation {user_id}: ${total_cost:.4f} total")
```

## Best Practices

1. **Tool Security**: Validate tool inputs and restrict dangerous operations
2. **Cost Monitoring**: Set up alerts when costs exceed thresholds
3. **Rate Limiting**: Choose appropriate limits based on use case and user tier
4. **Metadata**: Store essential tracking info but avoid sensitive data
5. **Configuration**: Use cloning for thread-safe runtime parameter changes

## See Also

- [Conversation Management](conversations.md) - Managing multi-turn conversations
- [LLM API](../../../api/dataknobs-llm.md) - Core LLM interface
- [Middleware API](../api/conversations.md#middleware) - Creating custom middleware
- [Tools API](../api/tools.md) - Tool system reference
