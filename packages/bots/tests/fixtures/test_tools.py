"""Test tools for unit testing tool resolution.

These tools are in a proper module so they can be imported via importlib.
"""

from typing import Any, Dict

from dataknobs_llm.tools import Tool


class SimpleTestTool(Tool):
    """Simple test tool without parameters."""

    def __init__(self):
        super().__init__(
            name="simple_test",
            description="A simple test tool",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self) -> str:
        return "simple"


class ParameterizedTestTool(Tool):
    """Test tool with initialization parameters."""

    def __init__(self, prefix: str = "test", multiplier: int = 1):
        super().__init__(
            name="parameterized_test",
            description="A parameterized test tool",
        )
        self.prefix = prefix
        self.multiplier = multiplier

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    async def execute(self, value: str) -> str:
        return f"{self.prefix}:{value}" * self.multiplier
