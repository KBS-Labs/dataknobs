"""Tool registry for managing available LLM tools.

This module provides the ToolRegistry class for registering, discovering,
and managing tools that can be used with LLMs. Now built on the common
Registry pattern from dataknobs_common.
"""

from typing import Any, Dict, List, Set

from dataknobs_common import NotFoundError, OperationError, Registry

from dataknobs_llm.tools.base import Tool


class ToolRegistry(Registry[Tool]):
    """Registry for managing available tools/functions.

    The ToolRegistry provides a central place to register and discover
    tools that can be called by LLMs. It supports tool registration,
    retrieval, listing, and conversion to function calling formats.

    Built on dataknobs_common.Registry for consistency across the ecosystem.

    Example:
        ```python
        # Create registry
        registry = ToolRegistry()

        # Register tools
        registry.register_tool(CalculatorTool())
        registry.register_tool(WebSearchTool())

        # Check if tool exists
        if registry.has_tool("calculator"):
            tool = registry.get_tool("calculator")
            result = await tool.execute(operation="add", a=5, b=3)

        # Get all tools for LLM function calling
        functions = registry.to_function_definitions()

        # List available tools
        tools = registry.list_tools()
        for tool_info in tools:
            print(f"{tool_info['name']}: {tool_info['description']}")
        ```
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        super().__init__(name="tools", enable_metrics=True)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            OperationError: If a tool with the same name already exists

        Example:
            >>> calculator = CalculatorTool()
            >>> registry.register_tool(calculator)
        """
        try:
            self.register(
                tool.name,
                tool,
                metadata={"description": tool.description, "schema": tool.schema},
            )
        except OperationError as e:
            # Re-raise with more specific message for backward compatibility
            raise OperationError(
                f"Tool with name '{tool.name}' already registered. "
                f"Use unregister() first or choose a different name.",
                context=e.context,
            ) from e

    def register_many(self, tools: List[Tool]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of Tool instances to register

        Raises:
            OperationError: If any tool name conflicts with existing tools

        Example:
            >>> tools = [CalculatorTool(), SearchTool(), WeatherTool()]
            >>> registry.register_many(tools)
        """
        # Check for conflicts first
        for tool in tools:
            if self.has(tool.name):
                raise OperationError(
                    f"Tool with name '{tool.name}' already registered",
                    context={"tool_name": tool.name, "registry": self.name},
                )

        # Register all tools
        for tool in tools:
            self.register(
                tool.name,
                tool,
                metadata={"description": tool.description, "schema": tool.schema},
            )

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool instance

        Raises:
            NotFoundError: If no tool with the given name exists

        Example:
            >>> tool = registry.get_tool("calculator")
            >>> result = await tool.execute(operation="add", a=5, b=3)
        """
        try:
            return self.get(name)
        except NotFoundError as e:
            # Re-raise with more specific message for backward compatibility
            raise NotFoundError(
                f"Tool not found: {name}",
                context=e.context,
            ) from e

    def has_tool(self, name: str) -> bool:
        """Check if a tool with the given name exists.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool exists, False otherwise

        Example:
            ```python
            if registry.has_tool("calculator"):
                print("Calculator available")
            ```
        """
        return self.has(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata.

        Returns:
            List of dictionaries containing tool information

        Example:
            ```python
            tools = registry.list_tools()
            for tool_info in tools:
                print(f"{tool_info['name']}: {tool_info['description']}")
            ```

            Returns format:
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
            for tool in self.list_items()
        ]

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names

        Example:
            >>> names = registry.get_tool_names()
            >>> print(names)
            ['calculator', 'search', 'weather']
        """
        return self.list_keys()

    def to_function_definitions(
        self, include_only: Set[str] | None = None, exclude: Set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format.

        Args:
            include_only: If provided, only include tools with these names
            exclude: If provided, exclude tools with these names

        Returns:
            List of function definition dictionaries

        Example:
            ```python
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
            ```
        """
        tools_to_include = []

        for name, tool in self.items():
            # Apply filters
            if include_only and name not in include_only:
                continue
            if exclude and name in exclude:
                continue

            tools_to_include.append(tool)

        return [tool.to_function_definition() for tool in tools_to_include]

    def to_anthropic_tool_definitions(
        self, include_only: Set[str] | None = None, exclude: Set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool format.

        Args:
            include_only: If provided, only include tools with these names
            exclude: If provided, exclude tools with these names

        Returns:
            List of tool definition dictionaries

        Example:
            ```python
            tools = registry.to_anthropic_tool_definitions()
            # Use with Anthropic API
            response = client.messages.create(
                model="claude-3-sonnet",
                tools=tools,
                messages=[...]
            )
            ```
        """
        tools_to_include = []

        for name, tool in self.items():
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
            NotFoundError: If tool not found
            Exception: If tool execution fails

        Example:
            ```python
            result = await registry.execute_tool(
                "calculator",
                operation="add",
                a=5,
                b=3
            )
            print(result)
            # 8
            ```
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
            ```python
            # Get all tools with category="math"
            math_tools = registry.filter_by_metadata(category="math")

            # Get tools with multiple criteria
            safe_tools = registry.filter_by_metadata(
                category="utility",
                safe=True
            )
            ```
        """
        matching_tools = []

        for tool in self.list_items():
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

        Example:
            >>> original = ToolRegistry()
            >>> original.register_tool(CalculatorTool())
            >>>
            >>> copy = original.clone()
            >>> copy.count()
            1
        """
        new_registry = ToolRegistry()
        for name, tool in self.items():
            new_registry.register(name, tool, allow_overwrite=True)
        return new_registry

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"ToolRegistry(tools={self.count()})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.count() == 0:
            return "ToolRegistry(empty)"

        tool_names = ", ".join(self.list_keys())
        return f"ToolRegistry({self.count()} tools: {tool_names})"

    # Note: __len__, __contains__, and __iter__ are inherited from Registry base class
