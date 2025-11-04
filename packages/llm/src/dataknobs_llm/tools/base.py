"""Base tool abstraction for LLM function calling.

This module provides the base Tool class for implementing callable tools
that can be used with LLM function calling capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    """Abstract base class for LLM-callable tools.

    A Tool represents a function that can be called by an LLM during generation.
    Each tool has a name, description, parameter schema, and execution logic.

    Example:
        class CalculatorTool(Tool):
            def __init__(self):
                super().__init__(
                    name="calculator",
                    description="Performs basic arithmetic operations"
                )

            @property
            def schema(self) -> Dict[str, Any]:
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
                    return a / b
                else:
                    raise ValueError(f"Unknown operation: {operation}")
    """

    def __init__(
        self,
        name: str,
        description: str,
        metadata: Dict[str, Any] | None = None
    ):
        """Initialize a tool.

        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            metadata: Optional metadata about the tool
        """
        self.name = name
        self.description = description
        self.metadata = metadata or {}

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters.

        Returns a JSON Schema dictionary describing the parameters
        this tool accepts. The schema should follow the JSON Schema
        specification and is used by LLMs to understand how to call
        the tool.

        Returns:
            JSON Schema dictionary for tool parameters

        Example:
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters.

        This method performs the actual tool logic. Parameters are passed
        as keyword arguments matching the schema definition.

        Args:
            **kwargs: Tool parameters matching the schema

        Returns:
            Tool execution result (can be any JSON-serializable type)

        Raises:
            Exception: If tool execution fails
        """
        pass

    def to_function_definition(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format.

        Returns a dictionary in the format expected by OpenAI's function
        calling API.

        Returns:
            Function definition dictionary

        Example:
            {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
        }

    def to_anthropic_tool_definition(self) -> Dict[str, Any]:
        """Convert tool to Anthropic tool format.

        Returns a dictionary in the format expected by Anthropic's Claude API.

        Returns:
            Tool definition dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate parameters against schema.

        Optional method for parameter validation before execution.
        By default, assumes LLM provides valid parameters.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - check required fields
        required = self.schema.get("required", [])
        return all(field in kwargs for field in required)

    def __repr__(self) -> str:
        """String representation of tool."""
        return f"Tool(name={self.name!r}, description={self.description!r})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name}: {self.description}"
