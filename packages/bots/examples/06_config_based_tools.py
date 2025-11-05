"""Config-based tool loading example.

This example demonstrates:
- Loading tools directly from configuration
- Using xref to reference pre-defined tool configurations
- Tool parameter customization via config
- No need to manually instantiate and register tools

Required Ollama model:
    ollama pull phi3:mini
"""

import asyncio
from typing import Any, Dict

from dataknobs_bots import BotContext, DynaBot
from dataknobs_llm.tools import Tool


# Define custom tools that can be loaded from config
class CalculatorTool(Tool):
    """Tool for performing basic arithmetic operations."""

    def __init__(self, precision: int = 2):
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations (add, subtract, multiply, divide)",
        )
        self.precision = precision

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

        # Round to configured precision
        result = round(result, self.precision)
        print(f"  → Calculator: {a} {operation} {b} = {result}")
        return result


class StringTool(Tool):
    """Tool for string operations."""

    def __init__(self, default_case: str = "upper"):
        super().__init__(
            name="string_ops",
            description="Performs string operations like uppercase, lowercase, reverse",
        )
        self.default_case = default_case

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to transform",
                },
                "operation": {
                    "type": "string",
                    "enum": ["upper", "lower", "reverse", "length"],
                    "description": "The operation to perform",
                },
            },
            "required": ["text", "operation"],
        }

    async def execute(self, text: str, operation: str) -> str | int:
        """Execute the string operation."""
        if operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "length":
            result = len(text)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        print(f"  → String: {operation}({text!r}) = {result!r}")
        return result


async def main():
    """Run a bot with config-based tool loading."""
    print("=" * 60)
    print("Config-Based Tool Loading Example")
    print("=" * 60)
    print()
    print("This example shows how to load tools from configuration.")
    print("No need to manually instantiate and register tools!")
    print("Required: ollama pull phi3:mini")
    print()

    # Configuration with tools loaded from config
    config = {
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "conversation_storage": {
            "backend": "memory",
        },
        "reasoning": {
            "strategy": "react",
            "max_iterations": 5,
            "verbose": True,
            "store_trace": True,
        },
        "prompts": {
            "agent_system": "You are a helpful AI agent with access to tools. "
            "Use the calculator for math operations and string_ops for text manipulation."
        },
        "system_prompt": {
            "name": "agent_system",
        },
        # Tool definitions - reusable tool configurations
        "tool_definitions": {
            "basic_calculator": {
                "class": "examples.06_config_based_tools.CalculatorTool",
                "params": {"precision": 2},
            },
            "precise_calculator": {
                "class": "examples.06_config_based_tools.CalculatorTool",
                "params": {"precision": 5},
            },
            "default_string": {
                "class": "examples.06_config_based_tools.StringTool",
                "params": {"default_case": "upper"},
            },
        },
        # Tools to load - can use direct class references or xrefs
        "tools": [
            # Direct class instantiation
            {
                "class": "examples.06_config_based_tools.CalculatorTool",
                "params": {"precision": 3},
            },
            # XRef to predefined tool definition
            "xref:tools[default_string]",
        ],
    }

    print("Creating bot with config-based tools...")
    print()
    print("Config includes:")
    print("  1. Direct tool instantiation (CalculatorTool with precision=3)")
    print("  2. XRef-based tool loading (StringTool from tool_definitions)")
    print()

    bot = await DynaBot.from_config(config)

    print("✓ Bot created successfully")
    tools = list(bot.tool_registry)
    print(f"✓ {len(tools)} tools loaded from configuration:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # Create context for this conversation
    context = BotContext(
        conversation_id="config-tools-001",
        client_id="example-client",
        user_id="demo-user",
    )

    # Tasks that use the configured tools
    tasks = [
        "Calculate 15.678 divided by 3.2",
        "Convert the text 'Hello World' to uppercase",
        "What is 10 multiplied by 7?",
        "Reverse the string 'configuration'",
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
    print("Config-based tool loading demonstration complete!")
    print()
    print("Key benefits:")
    print("- Tools are defined in configuration, not hardcoded")
    print("- Tool parameters can be customized per instance")
    print("- XRef allows reusing tool definitions")
    print("- No manual tool instantiation or registration needed")
    print("- Easy to swap tools without code changes")
    print()
    print("Try modifying the config to:")
    print("- Change calculator precision")
    print("- Add more tools via xref")
    print("- Switch between tool definitions dynamically")


if __name__ == "__main__":
    asyncio.run(main())
