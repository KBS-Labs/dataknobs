# Tools Development Guide

Complete guide to creating and using custom tools with DataKnobs Bots.

## Table of Contents

- [Overview](#overview)
- [Tool Basics](#tool-basics)
- [Creating Custom Tools](#creating-custom-tools)
- [Tool Configuration](#tool-configuration)
- [Advanced Tool Patterns](#advanced-tool-patterns)
- [Tool Testing](#tool-testing)
- [Best Practices](#best-practices)
- [Example Tools](#example-tools)
- [Troubleshooting](#troubleshooting)

---

## Overview

Tools extend DynaBot's capabilities by allowing it to interact with external systems, perform computations, access databases, or execute any custom logic. Tools are essential for building ReAct agents that can take actions beyond text generation.

### What is a Tool?

A tool is a Python class that:
1. Inherits from `dataknobs_llm.tools.Tool`
2. Defines a schema describing its inputs
3. Implements an `execute()` method that performs the action
4. Can be loaded from configuration without code changes

### When to Use Tools?

Use tools when your bot needs to:
- Perform calculations or data processing
- Query databases or APIs
- Access external services (weather, calendar, etc.)
- Execute actions (send emails, create tickets, etc.)
- Search knowledge bases or documents
- Perform multi-step reasoning tasks

---

## Tool Basics

### Tool Interface

Every tool must implement the `Tool` interface from `dataknobs_llm.tools`:

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any
from abc import abstractmethod

class Tool(ABC):
    """Base class for tools."""

    def __init__(self, name: str, description: str):
        """Initialize tool with name and description."""
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
```

### Minimal Tool Example

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any

class HelloTool(Tool):
    """Simple tool that says hello."""

    def __init__(self):
        super().__init__(
            name="say_hello",
            description="Says hello to a person"
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the person to greet"
                }
            },
            "required": ["name"]
        }

    async def execute(self, name: str) -> str:
        return f"Hello, {name}!"
```

**Usage:**
```python
tool = HelloTool()
result = await tool.execute(name="Alice")
# Returns: "Hello, Alice!"
```

---

## Creating Custom Tools

### Step 1: Define Your Tool Class

```python
# my_tools.py
from dataknobs_llm.tools import Tool
from typing import Dict, Any

class CalculatorTool(Tool):
    """Performs basic arithmetic operations."""

    def __init__(self, precision: int = 2):
        super().__init__(
            name="calculator",
            description=(
                "Performs basic arithmetic operations: "
                "add, subtract, multiply, divide. "
                "Returns numeric results."
            )
        )
        self.precision = precision
```

### Step 2: Define the Schema

The schema describes the tool's parameters using JSON Schema:

```python
    @property
    def schema(self) -> Dict[str, Any]:
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
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
```

**Schema Guidelines:**
- Use clear, descriptive property names
- Always include descriptions
- Mark required parameters in "required" array
- Use appropriate types: string, number, integer, boolean, array, object
- Use "enum" for fixed choices
- Keep schemas simple and focused

### Step 3: Implement Execute Method

```python
    async def execute(self, operation: str, a: float, b: float) -> float:
        """Execute the calculation."""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return round(result, self.precision)
```

**Execute Method Guidelines:**
- Parameters match schema properties
- Return type should be serializable (str, int, float, dict, list)
- Raise exceptions for errors
- Use async for I/O operations
- Add logging for debugging

### Step 4: Add Error Handling

```python
    async def execute(self, operation: str, a: float, b: float) -> float:
        """Execute the calculation."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Calculator: {operation}({a}, {b})")

            operations = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else None
            }

            if operation not in operations:
                raise ValueError(f"Invalid operation: {operation}")

            result = operations[operation](a, b)

            if result is None:
                raise ValueError("Cannot divide by zero")

            result = round(result, self.precision)
            logger.info(f"Calculator result: {result}")
            return result

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            raise
```

---

## Tool Configuration

### Loading Tools via Configuration

#### Method 1: Direct Class Instantiation

```yaml
tools:
  - class: my_tools.CalculatorTool
    params:
      precision: 3

  - class: my_tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}
```

#### Method 2: XRef to Predefined Tools

```yaml
tool_definitions:
  high_precision_calc:
    class: my_tools.CalculatorTool
    params:
      precision: 5

  low_precision_calc:
    class: my_tools.CalculatorTool
    params:
      precision: 2

tools:
  - xref:tools[high_precision_calc]
```

#### Method 3: Mixed Approach

```yaml
tool_definitions:
  weather:
    class: my_tools.WeatherTool
    params:
      api_key: ${WEATHER_API_KEY}

tools:
  # Direct
  - class: my_tools.CalculatorTool
    params:
      precision: 2

  # XRef
  - xref:tools[weather]
```

### Tool Parameters

Tools can accept configuration parameters:

```python
class EmailTool(Tool):
    """Send emails via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        from_email: str = None
    ):
        super().__init__(
            name="send_email",
            description="Send an email to a recipient"
        )
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email or smtp_user
```

**Configuration:**
```yaml
tools:
  - class: my_tools.EmailTool
    params:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      smtp_user: ${SMTP_USER}
      smtp_password: ${SMTP_PASSWORD}
      from_email: bot@example.com
```

---

## Advanced Tool Patterns

### Pattern 1: API Integration Tool

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any
import httpx

class WeatherTool(Tool):
    """Get weather information from API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org"):
        super().__init__(
            name="get_weather",
            description="Get current weather information for a city"
        )
        self.api_key = api_key
        self.base_url = base_url

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'New York')"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units"
                }
            },
            "required": ["city"]
        }

    async def execute(self, city: str, units: str = "metric") -> Dict[str, Any]:
        """Fetch weather data from API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/data/2.5/weather",
                params={
                    "q": city,
                    "appid": self.api_key,
                    "units": units
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "units": units
            }
```

### Pattern 2: Database Query Tool

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any, List
import asyncpg

class DatabaseQueryTool(Tool):
    """Execute safe database queries."""

    def __init__(self, connection_string: str):
        super().__init__(
            name="query_database",
            description="Query the database for information"
        )
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "enum": ["users", "orders", "products"],  # Whitelist tables
                    "description": "Table to query"
                },
                "conditions": {
                    "type": "object",
                    "description": "WHERE conditions (key=value pairs)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return",
                    "default": 10
                }
            },
            "required": ["table"]
        }

    async def execute(
        self,
        table: str,
        conditions: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute database query safely."""
        # Build safe parameterized query
        query = f"SELECT * FROM {table}"
        params = []

        if conditions:
            where_clauses = []
            for i, (key, value) in enumerate(conditions.items(), 1):
                where_clauses.append(f"{key} = ${i}")
                params.append(value)
            query += " WHERE " + " AND ".join(where_clauses)

        query += f" LIMIT {limit}"

        # Execute query
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
```

### Pattern 3: File Operations Tool

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any
from pathlib import Path
import aiofiles

class FileOperationsTool(Tool):
    """Safe file operations."""

    def __init__(self, allowed_directory: str):
        super().__init__(
            name="file_operations",
            description="Read, write, or list files in allowed directory"
        )
        self.allowed_directory = Path(allowed_directory).resolve()

    def _validate_path(self, filepath: str) -> Path:
        """Ensure path is within allowed directory."""
        full_path = (self.allowed_directory / filepath).resolve()
        if not str(full_path).startswith(str(self.allowed_directory)):
            raise ValueError("Access denied: path outside allowed directory")
        return full_path

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "File operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                }
            },
            "required": ["operation", "path"]
        }

    async def execute(
        self,
        operation: str,
        path: str,
        content: str = None
    ) -> Any:
        """Execute file operation."""
        validated_path = self._validate_path(path)

        if operation == "read":
            async with aiofiles.open(validated_path, 'r') as f:
                return await f.read()

        elif operation == "write":
            if content is None:
                raise ValueError("Content required for write operation")
            validated_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(validated_path, 'w') as f:
                await f.write(content)
            return f"Written {len(content)} bytes to {path}"

        elif operation == "list":
            if validated_path.is_dir():
                return [str(p.relative_to(self.allowed_directory))
                        for p in validated_path.iterdir()]
            else:
                raise ValueError(f"{path} is not a directory")
```

### Pattern 4: Multi-Step Tool

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any, List

class ResearchTool(Tool):
    """Perform multi-step research."""

    def __init__(self, search_engine_api_key: str):
        super().__init__(
            name="research",
            description="Research a topic using multiple sources"
        )
        self.api_key = search_engine_api_key

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to research"
                },
                "num_sources": {
                    "type": "integer",
                    "description": "Number of sources to consult",
                    "default": 3
                }
            },
            "required": ["topic"]
        }

    async def execute(self, topic: str, num_sources: int = 3) -> Dict[str, Any]:
        """Execute research."""
        # Step 1: Search for sources
        sources = await self._search(topic, num_sources)

        # Step 2: Fetch content from each source
        contents = await self._fetch_contents(sources)

        # Step 3: Summarize findings
        summary = await self._summarize(contents)

        return {
            "topic": topic,
            "sources": sources,
            "summary": summary,
            "num_sources_consulted": len(sources)
        }

    async def _search(self, query: str, limit: int) -> List[Dict[str, str]]:
        """Search for sources."""
        # Implementation...
        pass

    async def _fetch_contents(self, sources: List[Dict]) -> List[str]:
        """Fetch content from sources."""
        # Implementation...
        pass

    async def _summarize(self, contents: List[str]) -> str:
        """Summarize findings."""
        # Implementation...
        pass
```

---

## Tool Testing

### Unit Testing

```python
# test_my_tools.py
import pytest
from my_tools import CalculatorTool

@pytest.mark.asyncio
async def test_calculator_add():
    """Test calculator addition."""
    tool = CalculatorTool(precision=2)

    result = await tool.execute(operation="add", a=5, b=3)

    assert result == 8.0
    assert isinstance(result, float)


@pytest.mark.asyncio
async def test_calculator_divide_by_zero():
    """Test divide by zero error handling."""
    tool = CalculatorTool(precision=2)

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        await tool.execute(operation="divide", a=10, b=0)


@pytest.mark.asyncio
async def test_calculator_precision():
    """Test precision parameter."""
    tool = CalculatorTool(precision=3)

    result = await tool.execute(operation="divide", a=10, b=3)

    assert result == 3.333
```

### Integration Testing with DynaBot

```python
# test_tool_integration.py
import pytest
from dataknobs_bots import DynaBot, BotContext
from my_tools import CalculatorTool

@pytest.mark.asyncio
async def test_tool_with_bot():
    """Test tool integration with bot."""
    config = {
        "llm": {"provider": "ollama", "model": "phi3:mini"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {
            "strategy": "react",
            "max_iterations": 5
        },
        "tools": [
            {
                "class": "my_tools.CalculatorTool",
                "params": {"precision": 2}
            }
        ]
    }

    bot = await DynaBot.from_config(config)
    context = BotContext(conversation_id="test", client_id="test")

    response = await bot.chat("What is 15 multiplied by 7?", context)

    assert "105" in response
```

### Mocking External Services

```python
# test_weather_tool.py
import pytest
from unittest.mock import AsyncMock, patch
from my_tools import WeatherTool

@pytest.mark.asyncio
async def test_weather_tool_with_mock():
    """Test weather tool with mocked API."""
    tool = WeatherTool(api_key="test_key")

    # Mock the HTTP client
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "name": "London",
            "main": {"temp": 15, "humidity": 70},
            "weather": [{"description": "cloudy"}]
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        result = await tool.execute(city="London")

        assert result["city"] == "London"
        assert result["temperature"] == 15
        assert result["description"] == "cloudy"
```

---

## Best Practices

### 1. Clear Naming and Descriptions

```python
# ❌ Bad
class Tool1(Tool):
    def __init__(self):
        super().__init__(name="t1", description="does stuff")

# ✅ Good
class WeatherQueryTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description=(
                "Retrieves current weather information for a specified city. "
                "Returns temperature, conditions, and humidity. "
                "Use this when the user asks about weather or temperature."
            )
        )
```

### 2. Detailed Schemas

```python
# ❌ Bad
@property
def schema(self):
    return {
        "type": "object",
        "properties": {"x": {"type": "string"}}
    }

# ✅ Good
@property
def schema(self):
    return {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "Name of the city (e.g., 'London', 'Tokyo', 'New York')",
                "minLength": 1
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units to use",
                "default": "celsius"
            }
        },
        "required": ["city_name"]
    }
```

### 3. Error Handling

```python
async def execute(self, **kwargs):
    try:
        # Validate inputs
        if not self._validate_inputs(kwargs):
            raise ValueError("Invalid inputs")

        # Perform operation
        result = await self._do_operation(kwargs)

        # Validate output
        if not self._validate_output(result):
            raise RuntimeError("Invalid output")

        return result

    except ExternalAPIError as e:
        logger.error(f"API error: {e}")
        raise RuntimeError(f"External service unavailable: {e}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 4. Logging

```python
import logging

class MyTool(Tool):
    def __init__(self):
        super().__init__(name="my_tool", description="...")
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    async def execute(self, **kwargs):
        self.logger.info(f"Executing {self.name} with params: {kwargs}")

        try:
            result = await self._do_work(**kwargs)
            self.logger.info(f"Success: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            raise
```

### 5. Security

```python
class SecureDatabaseTool(Tool):
    """Secure database access with whitelist."""

    ALLOWED_TABLES = ["users", "orders", "products"]
    ALLOWED_COLUMNS = {
        "users": ["id", "name", "email"],
        "orders": ["id", "user_id", "total"],
        "products": ["id", "name", "price"]
    }

    async def execute(self, table: str, columns: List[str] = None):
        # Validate table
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Access to table '{table}' not allowed")

        # Validate columns
        if columns:
            allowed = self.ALLOWED_COLUMNS[table]
            if not all(col in allowed for col in columns):
                raise ValueError(f"Some columns not allowed for table '{table}'")

        # Use parameterized queries
        # Never use string formatting for SQL
```

### 6. Async Best Practices

```python
class MyAsyncTool(Tool):
    async def execute(self, urls: List[str]):
        # ❌ Bad: Sequential requests
        results = []
        for url in urls:
            result = await self._fetch(url)
            results.append(result)

        # ✅ Good: Concurrent requests
        import asyncio
        tasks = [self._fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return results
```

---

## Example Tools

### Complete Calculator Tool

See [Custom Tools Example](../examples/custom-tools.md) for complete code

### Complete API Integration Tool

```python
from dataknobs_llm.tools import Tool
from typing import Dict, Any
import httpx
import logging

class GitHubTool(Tool):
    """Interact with GitHub API."""

    def __init__(self, access_token: str = None):
        super().__init__(
            name="github",
            description="Query GitHub repositories, issues, and pull requests"
        )
        self.access_token = access_token
        self.logger = logging.getLogger(__name__)

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_repo", "list_issues", "search_code"],
                    "description": "Action to perform"
                },
                "repository": {
                    "type": "string",
                    "description": "Repository in format 'owner/repo'"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search_code action)"
                }
            },
            "required": ["action", "repository"]
        }

    async def execute(
        self,
        action: str,
        repository: str,
        query: str = None
    ) -> Dict[str, Any]:
        """Execute GitHub API call."""
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"token {self.access_token}"

        try:
            async with httpx.AsyncClient() as client:
                if action == "get_repo":
                    return await self._get_repo(client, repository, headers)
                elif action == "list_issues":
                    return await self._list_issues(client, repository, headers)
                elif action == "search_code":
                    if not query:
                        raise ValueError("query required for search_code")
                    return await self._search_code(client, repository, query, headers)

        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            raise RuntimeError(f"GitHub API error: {e}")

    async def _get_repo(self, client, repo, headers):
        response = await client.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        return {
            "name": data["name"],
            "description": data["description"],
            "stars": data["stargazers_count"],
            "url": data["html_url"]
        }

    # ... other methods
```

---

## Troubleshooting

### Tool Not Being Called

**Symptoms**: Bot doesn't use the tool even when it should.

**Solutions**:
1. **Improve description**: Be very explicit about when to use the tool
2. **Use better model**: `phi3:mini` or `gpt-4` are better at tool use
3. **Increase iterations**: `max_iterations: 10`
4. **Check schema**: Ensure schema is valid JSON Schema

### Tool Execution Errors

**Symptoms**: Tool is called but crashes.

**Solutions**:
1. **Add error handling**: Try/except blocks
2. **Validate inputs**: Check parameters before use
3. **Add logging**: Debug what's happening
4. **Test independently**: Unit test the tool

### Schema Validation Fails

**Symptoms**: Parameters don't match schema.

**Solutions**:
1. **Check required fields**: Ensure all required params are marked
2. **Validate types**: Use correct JSON Schema types
3. **Test schema**: Use JSON Schema validator
4. **Simplify schema**: Start simple, add complexity gradually

---

## See Also

- [API Reference](../api/reference.md) - Complete API documentation
- [User Guide](user-guide.md) - Usage tutorials
- [Configuration Reference](configuration.md) - Configuration options
- [Examples](../examples/custom-tools.md) - Working tool examples
