"""ReAct agent example.

This example demonstrates:
- ReAct (Reasoning + Acting) strategy
- Tool definition and registration
- Multi-step problem solving
- Reasoning trace storage
- Verbose logging

Required Ollama model:
    ollama pull gemma3:1b
"""

import asyncio
from typing import Any, Dict

from dataknobs_bots import BotContext, DynaBot
from dataknobs_llm.tools import Tool


# Define custom tools for the agent
class CalculatorTool(Tool):
    """Tool for performing basic arithmetic operations."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations (add, subtract, multiply, divide)",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            "required": ["operation", "a", "b"],
        }

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

        print(f"  → Calculator: {a} {operation} {b} = {result}")
        return result


class WeatherTool(Tool):
    """Mock tool for getting weather information."""

    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather information for a location",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location",
                },
            },
            "required": ["location"],
        }

    async def execute(self, location: str) -> Dict[str, Any]:
        """Get mock weather data."""
        # Mock weather data
        mock_weather = {
            "location": location,
            "temperature": 72,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 8,
        }

        print(f"  → Weather: {location} is {mock_weather['condition']}, {mock_weather['temperature']}°F")
        return mock_weather


class TimeTool(Tool):
    """Tool for getting current time information."""

    def __init__(self):
        super().__init__(
            name="get_time",
            description="Get current time in a specific timezone",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'America/New_York', 'Europe/London')",
                    "default": "UTC",
                },
            },
        }

    async def execute(self, timezone: str = "UTC") -> str:
        """Get current time (mocked)."""
        import datetime

        # Simple mock - just return UTC time with timezone label
        now = datetime.datetime.now(datetime.timezone.utc)
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        print(f"  → Time: {time_str} {timezone}")
        return f"{time_str} {timezone}"


async def main():
    """Run a ReAct agent conversation."""
    print("=" * 60)
    print("ReAct Agent Example")
    print("=" * 60)
    print()
    print("This example shows an agent using ReAct reasoning with tools.")
    print("Required: ollama pull gemma3:1b")
    print()

    # Configuration with ReAct reasoning
    config = {
        "llm": {
            "provider": "ollama",
            "model": "gemma3:1b",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "conversation_storage": {
            "backend": "memory",
        },
        "reasoning": {
            "strategy": "react",
            "max_iterations": 5,
            "verbose": True,  # Enable debug logging
            "store_trace": True,  # Store reasoning trace in metadata
        },
        "prompts": {
            "agent_system": "You are a helpful AI agent with access to tools. "
            "When you need to perform calculations, get weather, or check time, "
            "use the appropriate tools. Think step by step and explain your reasoning."
        },
        "system_prompt": {
            "name": "agent_system",
        },
    }

    print("Creating ReAct agent with tools...")
    bot = await DynaBot.from_config(config)

    # Register tools
    calculator = CalculatorTool()
    weather = WeatherTool()
    time_tool = TimeTool()

    bot.tool_registry.register_tool(calculator)
    bot.tool_registry.register_tool(weather)
    bot.tool_registry.register_tool(time_tool)

    print("✓ Bot created successfully")
    print(f"✓ Reasoning: ReAct (max {config['reasoning']['max_iterations']} iterations)")
    print("✓ Tools registered:")
    print(f"  - {calculator.name}: {calculator.description}")
    print(f"  - {weather.name}: {weather.description}")
    print(f"  - {time_tool.name}: {time_tool.description}")
    print()

    # Create context for this conversation
    context = BotContext(
        conversation_id="react-agent-001",
        client_id="example-client",
        user_id="demo-user",
    )

    # Tasks that require tool use
    tasks = [
        "What is 15 multiplied by 24?",
        "What's the weather like in San Francisco?",
        "What time is it in UTC?",
        "Calculate 100 divided by 4, then multiply the result by 3",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"[Task {i}] User: {task}")
        print()

        response = await bot.chat(
            message=task,
            context=context,
        )

        print(f"[Task {i}] Agent: {response}")
        print()
        print("-" * 60)
        print()

        # Add a small delay between tasks
        if i < len(tasks):
            await asyncio.sleep(2)

    print("=" * 60)
    print("ReAct agent demonstration complete!")
    print()
    print("Notice how the agent:")
    print("- Identified which tools to use for each task")
    print("- Called tools with appropriate parameters")
    print("- Reasoned through multi-step problems")
    print("- Provided final answers based on tool results")
    print()
    print("The reasoning trace is stored in conversation metadata")
    print("and can be retrieved for audit/debugging purposes.")


if __name__ == "__main__":
    asyncio.run(main())
