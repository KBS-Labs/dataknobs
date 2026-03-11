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


class KBDependentTestTool(Tool):
    """Test tool that requires knowledge_base injection.

    Mimics KnowledgeSearchTool's pattern: declares ``requires``
    in ``catalog_metadata()`` and takes ``knowledge_base`` as a
    required constructor argument.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        return {
            "name": "kb_dependent_test",
            "description": "Test tool requiring knowledge_base",
            "requires": ("knowledge_base",),
        }

    def __init__(self, knowledge_base: Any):
        super().__init__(
            name="kb_dependent_test",
            description="A test tool that depends on knowledge_base",
        )
        self.knowledge_base = knowledge_base

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str) -> dict[str, Any]:
        return {"query": query, "has_kb": self.knowledge_base is not None}
