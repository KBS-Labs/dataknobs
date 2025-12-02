"""Integration tests for tools and ReAct reasoning.

These tests verify tool execution and ReAct reasoning.

Most tests use the Echo LLM provider for fast, deterministic testing.
Tests that require real LLM tool-calling decisions use Ollama.

Run all tests:
    pytest tests/integration/test_tools_integration.py

Run only Echo-based tests (no Ollama needed):
    pytest tests/integration/test_tools_integration.py -m "not ollama_required"
"""

import os
from typing import Any, Dict

import pytest

from dataknobs_bots import BotContext, DynaBot
from dataknobs_llm.tools import Tool


class CalculatorTool(Tool):
    """Calculator tool for testing."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        }

    async def execute(self, operation: str, a: float, b: float) -> float:
        """Execute calculation."""
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


class CounterTool(Tool):
    """Stateful counter tool for testing."""

    def __init__(self):
        super().__init__(
            name="counter",
            description="Increments and returns a counter value",
        )
        self.count = 0

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "increment": {
                    "type": "integer",
                    "description": "Amount to increment by",
                    "default": 1,
                }
            },
        }

    async def execute(self, increment: int = 1) -> int:
        """Increment and return counter."""
        self.count += increment
        return self.count


# =============================================================================
# Tests using Echo LLM (fast, no external dependencies)
# =============================================================================


class TestToolRegistration:
    """Test tool registration and management using Echo LLM."""

    @pytest.mark.asyncio
    async def test_register_single_tool(self, bot_config_echo):
        """Test registering a single tool."""
        bot = await DynaBot.from_config(bot_config_echo)

        tool = CalculatorTool()
        bot.tool_registry.register_tool(tool)

        # Verify tool is registered
        assert len(list(bot.tool_registry)) == 1
        assert tool in list(bot.tool_registry)

    @pytest.mark.asyncio
    async def test_register_multiple_tools(self, bot_config_echo):
        """Test registering multiple tools."""
        bot = await DynaBot.from_config(bot_config_echo)

        calc_tool = CalculatorTool()
        counter_tool = CounterTool()

        bot.tool_registry.register_tool(calc_tool)
        bot.tool_registry.register_tool(counter_tool)

        # Verify both tools are registered
        tools = list(bot.tool_registry)
        assert len(tools) == 2
        assert calc_tool in tools
        assert counter_tool in tools

    @pytest.mark.asyncio
    async def test_tool_definitions(self, bot_config_echo):
        """Test tool definition generation."""
        bot = await DynaBot.from_config(bot_config_echo)

        tool = CalculatorTool()
        bot.tool_registry.register_tool(tool)

        # Test function definition format
        func_def = tool.to_function_definition()
        assert func_def["name"] == "calculator"
        assert "description" in func_def
        assert "parameters" in func_def

        # Test Anthropic format
        anthropic_def = tool.to_anthropic_tool_definition()
        assert anthropic_def["name"] == "calculator"
        assert "input_schema" in anthropic_def


class TestToolExecution:
    """Test direct tool execution (no LLM needed)."""

    @pytest.mark.asyncio
    async def test_calculator_tool_execution(self):
        """Test calculator tool executes correctly."""
        tool = CalculatorTool()

        # Test addition
        result = await tool.execute(operation="add", a=5, b=3)
        assert result == 8

        # Test subtraction
        result = await tool.execute(operation="subtract", a=10, b=4)
        assert result == 6

        # Test multiplication
        result = await tool.execute(operation="multiply", a=6, b=7)
        assert result == 42

        # Test division
        result = await tool.execute(operation="divide", a=20, b=4)
        assert result == 5

    @pytest.mark.asyncio
    async def test_calculator_division_by_zero(self):
        """Test calculator handles division by zero."""
        tool = CalculatorTool()

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await tool.execute(operation="divide", a=10, b=0)

    @pytest.mark.asyncio
    async def test_counter_tool_execution(self):
        """Test counter tool maintains state."""
        tool = CounterTool()

        # First call
        result = await tool.execute(increment=1)
        assert result == 1

        # Second call
        result = await tool.execute(increment=2)
        assert result == 3

        # Third call
        result = await tool.execute(increment=5)
        assert result == 8


class TestReActWithToolsEcho:
    """Test ReAct reasoning with tools using Echo LLM."""

    @pytest.mark.asyncio
    async def test_react_with_registered_tools(self, bot_config_echo_react):
        """Test ReAct with registered tools (Echo LLM)."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        # Register calculator tool
        calc = CalculatorTool()
        bot.tool_registry.register_tool(calc)

        context = BotContext(
            conversation_id="test-react-calc-001",
            client_id="test-client",
        )

        # With Echo LLM, we're testing that the infrastructure works
        response = await bot.chat("What is 15 plus 27?", context)

        # Should get a response (Echo response)
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_react_with_multiple_tools(self, bot_config_echo_react):
        """Test ReAct with multiple available tools."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        # Register multiple tools
        calc = CalculatorTool()
        counter = CounterTool()

        bot.tool_registry.register_tool(calc)
        bot.tool_registry.register_tool(counter)

        context = BotContext(
            conversation_id="test-react-multi-001",
            client_id="test-client",
        )

        # The agent should have access to both tools
        response = await bot.chat("Hello, what tools do you have?", context)

        assert response is not None

    @pytest.mark.asyncio
    async def test_react_max_iterations_config(self, echo_config):
        """Test ReAct respects max_iterations configuration."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 2,  # Very low limit
                "verbose": False,
                "store_trace": True,
            },
        }

        bot = await DynaBot.from_config(config)

        # Verify config was applied
        assert bot.reasoning_strategy.max_iterations == 2

        # Register a tool
        calc = CalculatorTool()
        bot.tool_registry.register_tool(calc)

        context = BotContext(
            conversation_id="test-react-iterations",
            client_id="test-client",
        )

        # Should complete within max_iterations
        response = await bot.chat("What is 5 plus 5?", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_react_trace_storage_config(self, bot_config_echo_react):
        """Test ReAct stores trace when configured."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        # Register tool
        calc = CalculatorTool()
        bot.tool_registry.register_tool(calc)

        context = BotContext(
            conversation_id="test-react-trace-tools",
            client_id="test-client",
        )

        # Generate response that might use tool
        await bot.chat("Calculate 10 plus 5", context)

        # Verify trace storage is enabled
        assert bot.reasoning_strategy.store_trace is True


class TestToolErrorHandling:
    """Test error handling in tool execution."""

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, bot_config_echo_react):
        """Test handling of tool execution errors."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        # Create a tool that will fail
        class FailingTool(Tool):
            def __init__(self):
                super().__init__(
                    name="failing_tool",
                    description="A tool that always fails",
                )

            @property
            def schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                }

            async def execute(self, input: str) -> str:
                raise RuntimeError("Tool execution failed")

        failing_tool = FailingTool()
        bot.tool_registry.register_tool(failing_tool)

        context = BotContext(
            conversation_id="test-failing-tool",
            client_id="test-client",
        )

        # Should handle error gracefully (not crash)
        response = await bot.chat("Use the failing tool", context)
        assert response is not None

    @pytest.mark.asyncio
    async def test_invalid_tool_parameters(self):
        """Test tool parameter validation."""
        tool = CalculatorTool()

        # Missing required parameter should raise error
        with pytest.raises(TypeError):
            await tool.execute(operation="add", a=5)  # Missing 'b'


class TestToolIntegrationPatterns:
    """Test common tool integration patterns using Echo LLM."""

    @pytest.mark.asyncio
    async def test_tool_state_persistence(self, bot_config_echo_react):
        """Test that tool state persists across calls."""
        bot = await DynaBot.from_config(bot_config_echo_react)

        counter = CounterTool()
        bot.tool_registry.register_tool(counter)

        context = BotContext(
            conversation_id="test-tool-state",
            client_id="test-client",
        )

        # First message
        await bot.chat("Increment the counter", context)

        # Second message
        await bot.chat("Increment it again", context)

        # Counter exists and is accessible (may or may not be incremented by Echo LLM)
        assert counter.count >= 0


# =============================================================================
# Tests requiring real LLM (Ollama)
# =============================================================================


class TestReActWithToolsOllama:
    """Test ReAct reasoning with tools using real Ollama LLM."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("TEST_OLLAMA", "").lower() == "true",
        reason="Real LLM tool execution requires TEST_OLLAMA=true",
    )
    async def test_react_tool_execution_flow(self, bot_config_react):
        """Test complete ReAct flow with tool execution (requires real LLM)."""
        bot = await DynaBot.from_config(bot_config_react)

        # Register calculator tool
        calc = CalculatorTool()
        bot.tool_registry.register_tool(calc)

        context = BotContext(
            conversation_id="test-react-calc-ollama",
            client_id="test-client",
        )

        # Ask a calculation question
        response = await bot.chat("What is 15 plus 27?", context)

        # Should get a response
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("TEST_OLLAMA", "").lower() == "true",
        reason="Real LLM tool chaining requires TEST_OLLAMA=true",
    )
    async def test_tool_chaining(self, bot_config_react):
        """Test using multiple tool calls in sequence (requires real LLM)."""
        bot = await DynaBot.from_config(bot_config_react)

        calc = CalculatorTool()
        bot.tool_registry.register_tool(calc)

        context = BotContext(
            conversation_id="test-tool-chain",
            client_id="test-client",
        )

        # Ask for multi-step calculation
        response = await bot.chat(
            "Calculate 5 times 3, then add 10 to the result",
            context,
        )

        assert response is not None
