"""Tests for the tool system (Tool and ToolRegistry)."""

import pytest
from dataknobs_llm.tools import Tool, ToolRegistry


class CalculatorTool(Tool):
    """Simple calculator tool for testing."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations"
        )

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }

    async def execute(self, operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")


class WebSearchTool(Tool):
    """Mock web search tool for testing."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            metadata={"category": "search"}
        )

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> str:
        return f"Search results for: {query}"


class TestTool:
    """Test the Tool base class."""

    def test_tool_initialization(self):
        """Test tool can be initialized."""
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert tool.description == "Performs basic arithmetic operations"
        assert tool.metadata == {}

    def test_tool_schema(self):
        """Test tool schema property."""
        tool = CalculatorTool()
        schema = tool.schema
        assert schema["type"] == "object"
        assert "operation" in schema["properties"]
        assert schema["required"] == ["operation", "a", "b"]

    @pytest.mark.asyncio
    async def test_tool_execute(self):
        """Test tool execution."""
        tool = CalculatorTool()
        result = await tool.execute(operation="add", a=5, b=3)
        assert result == 8

        result = await tool.execute(operation="multiply", a=4, b=7)
        assert result == 28

    @pytest.mark.asyncio
    async def test_tool_execute_error(self):
        """Test tool execution error handling."""
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Division by zero"):
            await tool.execute(operation="divide", a=10, b=0)

    def test_tool_to_function_definition(self):
        """Test conversion to function definition format."""
        tool = CalculatorTool()
        func_def = tool.to_function_definition()

        assert func_def["name"] == "calculator"
        assert func_def["description"] == "Performs basic arithmetic operations"
        assert "parameters" in func_def
        assert func_def["parameters"] == tool.schema

    def test_tool_to_anthropic_tool_definition(self):
        """Test conversion to Anthropic tool format."""
        tool = CalculatorTool()
        anthropic_def = tool.to_anthropic_tool_definition()

        assert anthropic_def["name"] == "calculator"
        assert anthropic_def["description"] == "Performs basic arithmetic operations"
        assert "input_schema" in anthropic_def
        assert anthropic_def["input_schema"] == tool.schema

    def test_tool_validate_parameters(self):
        """Test parameter validation."""
        tool = CalculatorTool()

        # Valid parameters
        assert tool.validate_parameters(operation="add", a=1, b=2) is True

        # Missing required parameter
        assert tool.validate_parameters(operation="add", a=1) is False

    def test_tool_with_metadata(self):
        """Test tool with metadata."""
        tool = WebSearchTool()
        assert tool.metadata == {"category": "search"}


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_registry_initialization(self):
        """Test registry can be initialized."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.count() == 0

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = CalculatorTool()

        registry.register_tool(tool)
        assert len(registry) == 1
        assert registry.has_tool("calculator")

    def test_register_duplicate_tool(self):
        """Test registering duplicate tool raises error."""
        from dataknobs_common import OperationError

        registry = ToolRegistry()
        tool1 = CalculatorTool()
        tool2 = CalculatorTool()

        registry.register_tool(tool1)
        with pytest.raises(OperationError, match="already registered"):
            registry.register_tool(tool2)

    def test_register_many_tools(self):
        """Test registering multiple tools at once."""
        registry = ToolRegistry()
        tools = [CalculatorTool(), WebSearchTool()]

        registry.register_many(tools)
        assert len(registry) == 2
        assert registry.has_tool("calculator")
        assert registry.has_tool("web_search")

    def test_get_tool(self):
        """Test retrieving a tool by name."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register_tool(tool)

        retrieved = registry.get_tool("calculator")
        assert retrieved is tool
        assert retrieved.name == "calculator"

    def test_get_nonexistent_tool(self):
        """Test getting nonexistent tool raises error."""
        from dataknobs_common import NotFoundError

        registry = ToolRegistry()
        with pytest.raises(NotFoundError, match="Tool not found"):
            registry.get_tool("nonexistent")

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register_tool(tool)

        assert registry.has_tool("calculator")
        registry.unregister("calculator")
        assert not registry.has_tool("calculator")

    def test_unregister_nonexistent_tool(self):
        """Test unregistering nonexistent tool raises error."""
        from dataknobs_common import NotFoundError

        registry = ToolRegistry()
        with pytest.raises(NotFoundError, match="not found"):
            registry.unregister("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        tools = registry.list_tools()
        assert len(tools) == 2
        assert any(t["name"] == "calculator" for t in tools)
        assert any(t["name"] == "web_search" for t in tools)

        # Check structure
        calc_tool = next(t for t in tools if t["name"] == "calculator")
        assert "description" in calc_tool
        assert "schema" in calc_tool
        assert "metadata" in calc_tool

    def test_get_tool_names(self):
        """Test getting list of tool names."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        names = registry.get_tool_names()
        assert set(names) == {"calculator", "web_search"}

    def test_clear_registry(self):
        """Test clearing all tools from registry."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0

    def test_to_function_definitions(self):
        """Test converting all tools to function definitions."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        definitions = registry.to_function_definitions()
        assert len(definitions) == 2
        assert any(d["name"] == "calculator" for d in definitions)
        assert any(d["name"] == "web_search" for d in definitions)

    def test_to_function_definitions_with_include(self):
        """Test converting specific tools to function definitions."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        definitions = registry.to_function_definitions(
            include_only={"calculator"}
        )
        assert len(definitions) == 1
        assert definitions[0]["name"] == "calculator"

    def test_to_function_definitions_with_exclude(self):
        """Test excluding tools from function definitions."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        definitions = registry.to_function_definitions(
            exclude={"web_search"}
        )
        assert len(definitions) == 1
        assert definitions[0]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool through the registry."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        result = await registry.execute_tool("calculator", operation="add", a=5, b=3)
        assert result == 8

    def test_filter_by_metadata(self):
        """Test filtering tools by metadata."""
        registry = ToolRegistry()
        registry.register_tool(WebSearchTool())  # Has category="search"

        calc = CalculatorTool()
        calc.metadata = {"category": "math"}
        registry.register_tool(calc)

        search_tools = registry.filter_by_metadata(category="search")
        assert len(search_tools) == 1
        assert search_tools[0].name == "web_search"

        math_tools = registry.filter_by_metadata(category="math")
        assert len(math_tools) == 1
        assert math_tools[0].name == "calculator"

    def test_clone_registry(self):
        """Test cloning a registry."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        cloned = registry.clone()
        assert len(cloned) == len(registry)
        assert cloned.has_tool("calculator")
        assert cloned.has_tool("web_search")

        # Modify clone shouldn't affect original
        cloned.unregister("calculator")
        assert not cloned.has_tool("calculator")
        assert registry.has_tool("calculator")

    def test_contains_operator(self):
        """Test 'in' operator for checking tool existence."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        assert "calculator" in registry
        assert "nonexistent" not in registry

    def test_iter_registry(self):
        """Test iterating over tools in registry."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        search = WebSearchTool()
        registry.register_tool(calc)
        registry.register_tool(search)

        tools = list(registry)
        assert len(tools) == 2
        assert calc in tools
        assert search in tools

    def test_registry_repr(self):
        """Test string representation of registry."""
        registry = ToolRegistry()
        assert repr(registry) == "ToolRegistry(tools=0)"

        registry.register_tool(CalculatorTool())
        assert repr(registry) == "ToolRegistry(tools=1)"

    def test_registry_str(self):
        """Test human-readable string of registry."""
        registry = ToolRegistry()
        assert str(registry) == "ToolRegistry(empty)"

        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())
        registry_str = str(registry)
        assert "ToolRegistry(2 tools:" in registry_str
        assert "calculator" in registry_str
        assert "web_search" in registry_str
