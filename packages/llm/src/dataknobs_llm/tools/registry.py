"""Tool registry for managing available LLM tools.

This module provides the ToolRegistry class for registering, discovering,
and managing tools that can be used with LLMs.
"""

from typing import Dict, List, Any, Set
from dataknobs_llm.tools.base import Tool


class ToolRegistry:
    """Registry for managing available tools/functions.

    The ToolRegistry provides a central place to register and discover
    tools that can be called by LLMs. It supports tool registration,
    retrieval, listing, and conversion to function calling formats.

    Example:
        # Create registry
        registry = ToolRegistry()

        # Register tools
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())

        # Check if tool exists
        if registry.has_tool("calculator"):
            tool = registry.get_tool("calculator")
            result = await tool.execute(operation="add", a=5, b=3)

        # Get all tools for LLM function calling
        functions = registry.to_function_definitions()

        # List available tools
        tools = registry.list_tools()
        for tool_info in tools:
            print(f"{tool_info['name']}: {tool_info['description']}")  # validate: ignore-print
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool with name '{tool.name}' already registered. "
                f"Use unregister() first or choose a different name."
            )
        self._tools[tool.name] = tool

    def register_many(self, tools: List[Tool]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of Tool instances to register

        Raises:
            ValueError: If any tool name conflicts with existing tools
        """
        # Check for conflicts first
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"Tool with name '{tool.name}' already registered"
                )

        # Register all tools
        for tool in tools:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to unregister

        Raises:
            KeyError: If no tool with the given name exists
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        del self._tools[name]

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool instance

        Raises:
            KeyError: If no tool with the given name exists
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool with the given name exists.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata.

        Returns:
            List of dictionaries containing tool information

        Example:
            [
                {
                    "name": "calculator",
                    "description": "Performs arithmetic operations",
                    "schema": {...},
                    "metadata": {...}
                },
                ...
            ]
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
                "metadata": tool.metadata,
            }
            for tool in self._tools.values()
        ]

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def count(self) -> int:
        """Get number of registered tools.

        Returns:
            Number of tools in registry
        """
        return len(self._tools)

    def clear(self) -> None:
        """Remove all tools from registry."""
        self._tools.clear()

    def to_function_definitions(
        self,
        include_only: Set[str] | None = None,
        exclude: Set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format.

        Args:
            include_only: If provided, only include tools with these names
            exclude: If provided, exclude tools with these names

        Returns:
            List of function definition dictionaries

        Example:
            # Get all tools
            functions = registry.to_function_definitions()

            # Get only specific tools
            functions = registry.to_function_definitions(
                include_only={"calculator", "web_search"}
            )

            # Get all except specific tools
            functions = registry.to_function_definitions(
                exclude={"dangerous_tool"}
            )
        """
        tools_to_include = []

        for name, tool in self._tools.items():
            # Apply filters
            if include_only and name not in include_only:
                continue
            if exclude and name in exclude:
                continue

            tools_to_include.append(tool)

        return [tool.to_function_definition() for tool in tools_to_include]

    def to_anthropic_tool_definitions(
        self,
        include_only: Set[str] | None = None,
        exclude: Set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool format.

        Args:
            include_only: If provided, only include tools with these names
            exclude: If provided, exclude tools with these names

        Returns:
            List of tool definition dictionaries
        """
        tools_to_include = []

        for name, tool in self._tools.items():
            # Apply filters
            if include_only and name not in include_only:
                continue
            if exclude and name in exclude:
                continue

            tools_to_include.append(tool)

        return [tool.to_anthropic_tool_definition() for tool in tools_to_include]

    async def execute_tool(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with given parameters.

        This is a convenience method for getting and executing a tool
        in a single call.

        Args:
            name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool not found
            Exception: If tool execution fails
        """
        tool = self.get_tool(name)
        return await tool.execute(**kwargs)

    def filter_by_metadata(self, **filters: Any) -> List[Tool]:
        """Filter tools by metadata attributes.

        Args:
            **filters: Key-value pairs to match in tool metadata

        Returns:
            List of tools matching all filters

        Example:
            # Get all tools with category="math"
            math_tools = registry.filter_by_metadata(category="math")

            # Get tools with multiple criteria
            safe_tools = registry.filter_by_metadata(
                category="utility",
                safe=True
            )
        """
        matching_tools = []

        for tool in self._tools.values():
            # Check if all filters match
            matches = True
            for key, value in filters.items():
                if key not in tool.metadata or tool.metadata[key] != value:
                    matches = False
                    break

            if matches:
                matching_tools.append(tool)

        return matching_tools

    def clone(self) -> "ToolRegistry":
        """Create a shallow copy of this registry.

        Returns:
            New ToolRegistry with same tools registered
        """
        new_registry = ToolRegistry()
        new_registry._tools = self._tools.copy()
        return new_registry

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool exists using 'in' operator."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"ToolRegistry(tools={len(self._tools)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self._tools:
            return "ToolRegistry(empty)"

        tool_names = ", ".join(self._tools.keys())
        return f"ToolRegistry({len(self._tools)} tools: {tool_names})"
