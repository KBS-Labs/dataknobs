# Configuration-Based Tools Example

Load and configure tools entirely through configuration using the XRef system.

## Overview

This example demonstrates:

- Defining tools in configuration
- XRef pattern for tool reuse
- Mixed configuration patterns
- Tool composition

## Prerequisites

```bash
# Install Ollama: https://ollama.ai/

# Pull the required model
ollama pull phi3:mini

# Install dataknobs-bots
pip install dataknobs-bots
```

## Complete Code

```python title="06_config_based_tools.py"
--8<-- "packages/bots/examples/06_config_based_tools.py"
```

## Configuration Patterns

### Pattern 1: Direct Instantiation

Define tools inline in bot configuration:

```python
config = {
    "llm": {"provider": "ollama", "model": "phi3:mini"},
    "tools": [
        {
            "class": "my_tools.CalculatorTool",
            "params": {"precision": 2}
        },
        {
            "class": "my_tools.WeatherTool",
            "params": {"api_key": "your-key"}
        }
    ]
}
```

**Pros**: Simple, explicit
**Cons**: Duplicates configuration

### Pattern 2: XRef (Cross-Reference)

Define tools once, reference multiple times:

```yaml
# config.yaml
tools:
  calculator:
    class: my_tools.CalculatorTool
    params:
      precision: 2

  weather:
    class: my_tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}

  search:
    class: my_tools.WebSearchTool
    params:
      max_results: 10

bots:
  math_bot:
    llm:
      provider: ollama
      model: phi3:mini
    tools:
      - xref:tools[calculator]

  assistant_bot:
    llm:
      provider: ollama
      model: phi3:mini
    tools:
      - xref:tools[calculator]
      - xref:tools[weather]
      - xref:tools[search]
```

**Pros**: DRY principle, centralized configuration
**Cons**: Requires dataknobs-config

### Pattern 3: Mixed

Combine both approaches:

```python
config = {
    "llm": {"provider": "ollama", "model": "phi3:mini"},
    "tools": [
        "xref:tools[calculator]",  # Predefined tool
        {                           # Inline tool
            "class": "custom.SpecialTool",
            "params": {"mode": "debug"}
        }
    ]
}
```

**Pros**: Flexibility
**Cons**: Can be confusing

## XRef System

### Basic XRef Syntax

```python
"tools": [
    "xref:tools[calculator]"  # References tools.calculator
]
```

### Nested XRefs

```yaml
tools:
  basic_calculator:
    class: my_tools.CalculatorTool
    params:
      precision: 2

  advanced_calculator:
    xref: tools[basic_calculator]
    params:
      precision: 5  # Override precision
```

### Environment Variables

```yaml
tools:
  weather:
    class: my_tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}  # From environment
```

## Tool Composition

### Sharing Parameters

```yaml
common:
  api_timeout: 30
  retry_count: 3

tools:
  weather:
    class: my_tools.WeatherTool
    params:
      timeout: xref:common[api_timeout]
      retries: xref:common[retry_count]

  search:
    class: my_tools.WebSearchTool
    params:
      timeout: xref:common[api_timeout]
      retries: xref:common[retry_count]
```

### Tool Groups

```yaml
tool_groups:
  productivity:
    - xref:tools[calendar]
    - xref:tools[email]
    - xref:tools[tasks]

  research:
    - xref:tools[web_search]
    - xref:tools[arxiv_search]
    - xref:tools[wikipedia]

bots:
  assistant:
    tools: xref:tool_groups[productivity]

  researcher:
    tools: xref:tool_groups[research]
```

## Running the Example

```bash
cd packages/bots
python examples/06_config_based_tools.py
```

## Expected Output

```
============================================================
Configuration-Based Tools Example
============================================================

This example shows how to load tools from configuration
using different patterns.

Pattern 1: Direct Instantiation
--------------------------------
User: What is 25 times 4?
Bot: 25 times 4 equals 100.

Pattern 2: XRef Pattern
-----------------------
User: Calculate 100 divided by 4
Bot: 100 divided by 4 equals 25.

Pattern 3: Mixed Approach
-------------------------
User: What's 50 plus 75 minus 25?
Bot: The result is 100.
```

## Best Practices

### 1. Use XRef for Reusable Tools

```yaml
# Good: Centralized tool definitions
tools:
  calculator:
    class: my_tools.CalculatorTool
```

### 2. Direct Instantiation for One-Off Tools

```python
# Good: Bot-specific tool
config = {
    "tools": [
        {"class": "bots.SpecialTool", "params": {}}
    ]
}
```

### 3. Environment Variables for Secrets

```yaml
# Good: Secrets from environment
tools:
  api_tool:
    class: tools.APITool
    params:
      api_key: ${API_KEY}
      api_secret: ${API_SECRET}
```

### 4. Document Tool Parameters

```python
class CalculatorTool(Tool):
    """Calculator tool for basic arithmetic.

    Args:
        precision: Number of decimal places (default: 2)
        max_value: Maximum allowed value (default: 1e6)
    """
    def __init__(self, precision: int = 2, max_value: float = 1e6):
        # ...
```

## Tool Validation

### Runtime Validation

```python
class APITool(Tool):
    def __init__(self, api_key: str, timeout: int = 30):
        if not api_key:
            raise ValueError("api_key is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        super().__init__(name="api_tool", description="...")
        self.api_key = api_key
        self.timeout = timeout
```

### Configuration Validation

```yaml
# Use dataknobs-config schema validation
tools:
  calculator:
    class: my_tools.CalculatorTool
    params:
      precision:
        type: integer
        minimum: 0
        maximum: 10
        default: 2
```

## Key Takeaways

1. ✅ **No Code Changes** - Add tools via configuration
2. ✅ **Reusability** - XRef pattern for shared tools
3. ✅ **Flexibility** - Mix patterns as needed
4. ✅ **Environment Vars** - Keep secrets out of config

## Common Patterns

### Multi-Environment Tools

```yaml
# development.yaml
tools:
  api_tool:
    class: tools.APITool
    params:
      base_url: http://localhost:8000
      debug: true

# production.yaml
tools:
  api_tool:
    class: tools.APITool
    params:
      base_url: https://api.production.com
      debug: false
```

### Tool Variants

```yaml
tools:
  calculator_basic:
    class: tools.CalculatorTool
    params:
      precision: 2

  calculator_scientific:
    class: tools.ScientificCalculatorTool
    params:
      precision: 10
      scientific_notation: true
```

## What's Next?

You've completed all the basic examples! Now explore:

- [User Guide](../guides/user-guide.md) - In-depth tutorials
- [Tools Development Guide](../guides/tools.md) - Advanced tool patterns
- [Configuration Reference](../guides/configuration.md) - All options

## Related Examples

- [ReAct Agent](react-agent.md) - Tool-using agent basics
- [Multi-Tenant Bot](multi-tenant.md) - Multi-client setup

## Related Documentation

- [Tools Development Guide](../guides/tools.md)
- [Configuration Reference](../guides/configuration.md)
- [API Reference - Tools](../api/reference.md#tools)
