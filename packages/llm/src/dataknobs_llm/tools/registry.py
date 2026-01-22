"""Tool registry for managing available LLM tools.

This module provides the ToolRegistry class for registering, discovering,
and managing tools that can be used with LLMs. Now built on the common
Registry pattern from dataknobs_common.
"""

import logging
import time
from typing import Any, Dict, List, Set

from dataknobs_common import NotFoundError, OperationError, Registry

from dataknobs_llm.tools.base import Tool
from dataknobs_llm.tools.observability import (
    ExecutionHistoryQuery,
    ExecutionStats,
    ExecutionTracker,
    ToolExecutionRecord,
)

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        track_executions: bool = False,
        max_execution_history: int = 100,
    ):
        """Initialize a tool registry.

        Args:
            track_executions: If True, record execution history for
                observability and debugging. Default False.
            max_execution_history: Maximum number of execution records
                to retain when tracking is enabled. Default 100.
        """
        super().__init__(name="tools", enable_metrics=True)
        self._track_executions = track_executions
        self._execution_tracker: ExecutionTracker | None = (
            ExecutionTracker(max_history=max_execution_history)
            if track_executions
            else None
        )

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
        in a single call. When execution tracking is enabled, records
        timing, parameters, and results for observability.

        Args:
            name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool. Special parameters
                starting with '_' (like _context) are passed to the tool
                but excluded from execution records.

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

            # With tracking enabled
            registry = ToolRegistry(track_executions=True)
            await registry.execute_tool("calculator", operation="add", a=5, b=3)
            history = registry.get_execution_history(tool_name="calculator")
            ```
        """
        tool = self.get_tool(name)

        # Separate internal params from tool params
        # Internal params (starting with _) are used by the registry but not passed to tools
        tool_params = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        context = kwargs.get("_context")

        if not self._track_executions or self._execution_tracker is None:
            return await tool.execute(**tool_params)

        # Extract context ID if available
        context_id = getattr(context, "conversation_id", None) if context else None

        start_time = time.time()
        try:
            result = await tool.execute(**tool_params)
            duration_ms = (time.time() - start_time) * 1000

            self._execution_tracker.record(
                ToolExecutionRecord(
                    tool_name=name,
                    timestamp=start_time,
                    parameters=tool_params,
                    result=result,
                    duration_ms=duration_ms,
                    success=True,
                    context_id=context_id,
                )
            )
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._execution_tracker.record(
                ToolExecutionRecord(
                    tool_name=name,
                    timestamp=start_time,
                    parameters=tool_params,
                    result=None,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    context_id=context_id,
                )
            )
            raise

    # ==================== Execution Tracking Methods ====================

    @property
    def tracking_enabled(self) -> bool:
        """Check if execution tracking is enabled.

        Returns:
            True if tracking is enabled, False otherwise
        """
        return self._track_executions

    def get_execution_history(
        self,
        tool_name: str | None = None,
        context_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        success_only: bool = False,
        failed_only: bool = False,
        limit: int | None = None,
    ) -> list[ToolExecutionRecord]:
        """Query tool execution history.

        Only available when tracking is enabled.

        Args:
            tool_name: Filter by tool name
            context_id: Filter by context/conversation ID
            since: Filter to records after this timestamp
            until: Filter to records before this timestamp
            success_only: Only include successful executions
            failed_only: Only include failed executions
            limit: Maximum number of records to return

        Returns:
            List of matching execution records, or empty list if
            tracking is not enabled

        Example:
            ```python
            # Get all executions for a specific tool
            calc_history = registry.get_execution_history(tool_name="calculator")

            # Get recent failed executions
            failures = registry.get_execution_history(
                since=time.time() - 3600,
                failed_only=True
            )

            # Get executions for a conversation
            conv_history = registry.get_execution_history(
                context_id="conv-123"
            )
            ```
        """
        if self._execution_tracker is None:
            return []

        query = ExecutionHistoryQuery(
            tool_name=tool_name,
            context_id=context_id,
            since=since,
            until=until,
            success_only=success_only,
            failed_only=failed_only,
            limit=limit,
        )
        return self._execution_tracker.query(query)

    def get_execution_stats(self, tool_name: str | None = None) -> ExecutionStats:
        """Get aggregated execution statistics.

        Only available when tracking is enabled.

        Args:
            tool_name: Get stats for specific tool, or None for all tools

        Returns:
            ExecutionStats with aggregated metrics, or empty stats if
            tracking is not enabled

        Example:
            ```python
            # Get stats for all tools
            all_stats = registry.get_execution_stats()
            print(f"Total: {all_stats.total_executions}")
            print(f"Success rate: {all_stats.success_rate:.1f}%")

            # Get stats for specific tool
            calc_stats = registry.get_execution_stats("calculator")
            print(f"Avg duration: {calc_stats.avg_duration_ms:.2f}ms")
            ```
        """
        if self._execution_tracker is None:
            return ExecutionStats(tool_name=tool_name)

        return self._execution_tracker.get_stats(tool_name)

    def clear_execution_history(self) -> None:
        """Clear all execution history.

        Has no effect if tracking is not enabled.

        Example:
            ```python
            # Clear history before a test run
            registry.clear_execution_history()
            ```
        """
        if self._execution_tracker is not None:
            self._execution_tracker.clear()

    def execution_history_count(self) -> int:
        """Get number of records in execution history.

        Returns:
            Number of records, or 0 if tracking is not enabled
        """
        if self._execution_tracker is None:
            return 0
        return len(self._execution_tracker)

    # ==================== Filter Methods ====================

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

    def clone(self, preserve_history: bool = False) -> "ToolRegistry":
        """Create a shallow copy of this registry.

        Args:
            preserve_history: If True and tracking is enabled, copy
                execution history to the new registry. Default False.

        Returns:
            New ToolRegistry with same tools and tracking settings

        Example:
            >>> original = ToolRegistry(track_executions=True)
            >>> original.register_tool(CalculatorTool())
            >>>
            >>> copy = original.clone()
            >>> copy.count()
            1
            >>> copy.tracking_enabled
            True
        """
        # Determine max history from current tracker
        max_history = (
            self._execution_tracker._max_history
            if self._execution_tracker
            else 100
        )

        new_registry = ToolRegistry(
            track_executions=self._track_executions,
            max_execution_history=max_history,
        )

        for name, tool in self.items():
            new_registry.register(name, tool, allow_overwrite=True)

        # Optionally copy history
        # Note: use 'is not None' to avoid __len__ evaluation (empty tracker is falsy)
        if (
            preserve_history
            and self._execution_tracker is not None
            and new_registry._execution_tracker is not None
        ):
            for record in self._execution_tracker.query():
                new_registry._execution_tracker.record(record)

        return new_registry

    def __repr__(self) -> str:
        """String representation of registry."""
        tracking = ", tracking=True" if self._track_executions else ""
        return f"ToolRegistry(tools={self.count()}{tracking})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.count() == 0:
            tracking = ", tracking enabled" if self._track_executions else ""
            return f"ToolRegistry(empty{tracking})"

        tool_names = ", ".join(self.list_keys())
        tracking = ", tracking enabled" if self._track_executions else ""
        return f"ToolRegistry({self.count()} tools: {tool_names}{tracking})"

    # Note: __len__, __contains__, and __iter__ are inherited from Registry base class
