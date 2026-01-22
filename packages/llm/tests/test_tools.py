"""Tests for the tool system (Tool and ToolRegistry)."""

import time

import pytest

from dataknobs_llm.tools import Tool, ToolExecutionRecord, ToolRegistry


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


class TestToolRegistryExecutionTracking:
    """Test execution tracking in ToolRegistry."""

    def test_tracking_disabled_by_default(self):
        """Test that tracking is disabled by default."""
        registry = ToolRegistry()
        assert registry.tracking_enabled is False

    def test_tracking_can_be_enabled(self):
        """Test that tracking can be enabled."""
        registry = ToolRegistry(track_executions=True)
        assert registry.tracking_enabled is True

    def test_repr_shows_tracking(self):
        """Test repr shows tracking status."""
        registry = ToolRegistry()
        assert "tracking" not in repr(registry)

        registry_tracking = ToolRegistry(track_executions=True)
        assert "tracking=True" in repr(registry_tracking)

    def test_str_shows_tracking(self):
        """Test str shows tracking status."""
        registry = ToolRegistry(track_executions=True)
        assert "tracking enabled" in str(registry)

    @pytest.mark.asyncio
    async def test_execute_without_tracking(self):
        """Test that execute works without tracking enabled."""
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        result = await registry.execute_tool("calculator", operation="add", a=5, b=3)
        assert result == 8

        # No history when tracking disabled
        history = registry.get_execution_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_execute_with_tracking(self):
        """Test that execute records history when tracking enabled."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        result = await registry.execute_tool("calculator", operation="add", a=5, b=3)
        assert result == 8

        history = registry.get_execution_history()
        assert len(history) == 1

        record = history[0]
        assert record.tool_name == "calculator"
        assert record.parameters == {"operation": "add", "a": 5, "b": 3}
        assert record.result == 8
        assert record.success is True
        assert record.duration_ms >= 0  # May be 0 for very fast executions

    @pytest.mark.asyncio
    async def test_execute_tracking_records_failures(self):
        """Test that failed executions are tracked."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        with pytest.raises(ValueError, match="Division by zero"):
            await registry.execute_tool("calculator", operation="divide", a=10, b=0)

        history = registry.get_execution_history()
        assert len(history) == 1

        record = history[0]
        assert record.tool_name == "calculator"
        assert record.success is False
        assert "Division by zero" in record.error
        assert record.result is None

    @pytest.mark.asyncio
    async def test_execute_tracking_sanitizes_internal_params(self):
        """Test that internal params starting with _ are not recorded."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        # Execute with internal params (these shouldn't be recorded)
        result = await registry.execute_tool(
            "calculator",
            operation="add",
            a=5,
            b=3,
            _context={"internal": "data"},
            _other="private",
        )
        assert result == 8

        history = registry.get_execution_history()
        assert len(history) == 1

        record = history[0]
        # Internal params should not be in recorded parameters
        assert "_context" not in record.parameters
        assert "_other" not in record.parameters
        # Regular params should be recorded
        assert record.parameters == {"operation": "add", "a": 5, "b": 3}

    @pytest.mark.asyncio
    async def test_get_execution_history_filter_by_tool(self):
        """Test filtering execution history by tool name."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        await registry.execute_tool("web_search", query="test")
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)

        # All history
        all_history = registry.get_execution_history()
        assert len(all_history) == 3

        # Calculator only
        calc_history = registry.get_execution_history(tool_name="calculator")
        assert len(calc_history) == 2
        assert all(r.tool_name == "calculator" for r in calc_history)

        # Search only
        search_history = registry.get_execution_history(tool_name="web_search")
        assert len(search_history) == 1

    @pytest.mark.asyncio
    async def test_get_execution_history_filter_by_success(self):
        """Test filtering execution history by success/failure."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        try:
            await registry.execute_tool("calculator", operation="divide", a=1, b=0)
        except ValueError:
            pass
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)

        # All history
        all_history = registry.get_execution_history()
        assert len(all_history) == 3

        # Success only
        success_history = registry.get_execution_history(success_only=True)
        assert len(success_history) == 2
        assert all(r.success for r in success_history)

        # Failed only
        failed_history = registry.get_execution_history(failed_only=True)
        assert len(failed_history) == 1
        assert all(not r.success for r in failed_history)

    @pytest.mark.asyncio
    async def test_get_execution_history_filter_by_time(self):
        """Test filtering execution history by time range."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        before = time.time()
        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        middle = time.time()
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)
        after = time.time()

        # Since middle
        recent_history = registry.get_execution_history(since=middle)
        assert len(recent_history) == 1
        assert recent_history[0].parameters["operation"] == "multiply"

        # Until middle
        early_history = registry.get_execution_history(until=middle)
        assert len(early_history) == 1
        assert early_history[0].parameters["operation"] == "add"

    @pytest.mark.asyncio
    async def test_get_execution_history_with_limit(self):
        """Test limiting execution history results."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        for i in range(10):
            await registry.execute_tool("calculator", operation="add", a=i, b=1)

        history = registry.get_execution_history(limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_execution_stats(self):
        """Test getting execution statistics."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)
        try:
            await registry.execute_tool("calculator", operation="divide", a=1, b=0)
        except ValueError:
            pass

        stats = registry.get_execution_stats()
        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert stats.failed_executions == 1
        assert stats.success_rate == pytest.approx(66.67, rel=0.01)

        # Stats for specific tool
        calc_stats = registry.get_execution_stats("calculator")
        assert calc_stats.tool_name == "calculator"
        assert calc_stats.total_executions == 3

    @pytest.mark.asyncio
    async def test_clear_execution_history(self):
        """Test clearing execution history."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)

        assert registry.execution_history_count() == 2

        registry.clear_execution_history()

        assert registry.execution_history_count() == 0
        assert registry.get_execution_history() == []

    def test_clear_execution_history_no_op_when_disabled(self):
        """Test that clear_execution_history is safe when tracking disabled."""
        registry = ToolRegistry()
        # Should not raise
        registry.clear_execution_history()

    def test_execution_history_count(self):
        """Test getting execution history count."""
        registry = ToolRegistry(track_executions=True)
        assert registry.execution_history_count() == 0

        registry_disabled = ToolRegistry()
        assert registry_disabled.execution_history_count() == 0

    @pytest.mark.asyncio
    async def test_clone_preserves_tracking_settings(self):
        """Test that clone preserves tracking settings."""
        registry = ToolRegistry(track_executions=True, max_execution_history=50)
        registry.register_tool(CalculatorTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)

        cloned = registry.clone()

        # Tracking settings preserved
        assert cloned.tracking_enabled is True

        # But history is not copied by default
        assert cloned.execution_history_count() == 0

    @pytest.mark.asyncio
    async def test_clone_can_preserve_history(self):
        """Test that clone can optionally preserve history."""
        registry = ToolRegistry(track_executions=True)
        registry.register_tool(CalculatorTool())

        await registry.execute_tool("calculator", operation="add", a=1, b=2)
        await registry.execute_tool("calculator", operation="multiply", a=3, b=4)

        cloned = registry.clone(preserve_history=True)

        # History is copied
        assert cloned.execution_history_count() == 2

    @pytest.mark.asyncio
    async def test_max_execution_history(self):
        """Test that max execution history is enforced."""
        registry = ToolRegistry(track_executions=True, max_execution_history=3)
        registry.register_tool(CalculatorTool())

        for i in range(10):
            await registry.execute_tool("calculator", operation="add", a=i, b=1)

        # Should only keep last 3
        assert registry.execution_history_count() == 3
        history = registry.get_execution_history()
        # Most recent should be i=9, 8, 7
        assert history[0].parameters["a"] == 7
        assert history[1].parameters["a"] == 8
        assert history[2].parameters["a"] == 9
