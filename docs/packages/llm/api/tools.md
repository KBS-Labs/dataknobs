# Tools API

Tool system for LLM function calling.

## Overview

The tools API provides abstractions for defining functions that LLMs can call during generation. It includes:

- **Tool**: Abstract base class for defining callable tools
- **ToolRegistry**: Central registry for managing available tools

## Complete API Reference

For comprehensive auto-generated API documentation with all classes, methods, and full signatures, see:

**[📖 Tool & ToolRegistry Complete Reference](../../../api/reference/llm.md#dataknobs_llm.Tool)**

This page focuses on practical usage examples. The complete reference provides exhaustive technical documentation with all methods, parameters, and type annotations.

## Usage Examples

### Basic Tool Definition

```python
from dataknobs_llm import Tool
from typing import Dict, Any

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a location",
            metadata={"category": "information"}
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }

    async def execute(self, location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
        # Implementation here
        return {
            "location": location,
            "temperature": 72,
            "unit": unit,
            "conditions": "sunny"
        }
```

### Registry Operations

```python
from dataknobs_llm import ToolRegistry

# Create and populate registry
registry = ToolRegistry()
registry.register(WeatherTool())
registry.register(CalculatorTool())

# Check for tools
if registry.has_tool("get_weather"):
    tool = registry.get_tool("get_weather")

# List all tools
tools = registry.list_tools()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")

# Get function definitions for LLM
functions = registry.to_function_definitions()

# Execute a tool
result = await registry.execute_tool(
    "get_weather",
    location="San Francisco, CA",
    unit="celsius"
)
```

### Filtering and Metadata

```python
# Register tools with metadata (tools must be proper subclasses)
file_read_tool = FileReadTool()  # Assume this is defined elsewhere
search_docs_tool = SearchDocsTool()  # Assume this is defined elsewhere

registry.register(file_read_tool)
registry.register(search_docs_tool)

# Filter by metadata
safe_tools = registry.filter_by_metadata(safe=True)
fs_tools = registry.filter_by_metadata(category="filesystem")

# Selective tool definitions
safe_functions = registry.to_function_definitions(
    include_only={"search_docs", "get_weather"}
)

# Exclude dangerous tools
safe_functions = registry.to_function_definitions(
    exclude={"file_write", "system_command"}
)
```

### OpenAI Function Calling Format

The `to_function_definition()` method returns OpenAI-compatible function definitions:

```python
{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
}
```

### Anthropic Tool Format

The `to_anthropic_tool_definition()` method returns Claude-compatible tool definitions:

```python
{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
}
```

## See Also

- [Tools Guide](../guides/tools-and-enhancements.md) - Comprehensive usage guide
- [LLM API](llm.md) - LLM provider interface
- [Conversations API](conversations.md) - Conversation management
