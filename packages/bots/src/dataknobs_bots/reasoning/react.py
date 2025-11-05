"""ReAct (Reasoning + Acting) reasoning strategy."""

import logging
from typing import Any

from .base import ReasoningStrategy

logger = logging.getLogger(__name__)


class ReActReasoning(ReasoningStrategy):
    """ReAct (Reasoning + Acting) strategy.

    This strategy implements the ReAct pattern where the LLM:
    1. Reasons about what to do (Thought)
    2. Takes an action (using tools if needed)
    3. Observes the result
    4. Repeats until task is complete

    This is useful for:
    - Multi-step problem solving
    - Tasks requiring tool use
    - Complex reasoning chains

    Attributes:
        max_iterations: Maximum number of reasoning loops
        verbose: Whether to enable debug-level logging
        store_trace: Whether to store reasoning trace in conversation metadata

    Example:
        ```python
        strategy = ReActReasoning(
            max_iterations=5,
            verbose=True,
            store_trace=True
        )
        response = await strategy.generate(
            manager=conversation_manager,
            llm=llm_provider,
            tools=[search_tool, calculator_tool]
        )
        ```
    """

    def __init__(
        self,
        max_iterations: int = 5,
        verbose: bool = False,
        store_trace: bool = False,
    ):
        """Initialize ReAct reasoning strategy.

        Args:
            max_iterations: Maximum reasoning/action iterations
            verbose: Enable debug-level logging for reasoning steps
            store_trace: Store reasoning trace in conversation metadata
        """
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.store_trace = store_trace

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response using ReAct loop.

        The ReAct loop:
        1. Generate response (may include tool calls)
        2. If tool calls present, execute them
        3. Add observations to conversation
        4. Repeat until no more tool calls or max iterations

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Generation parameters

        Returns:
            Final LLM response
        """
        if not tools:
            # No tools available, fall back to simple generation
            logger.info(
                "ReAct: No tools available, falling back to simple generation",
                extra={"conversation_id": manager.conversation_id},
            )
            return await manager.complete(**kwargs)

        # Initialize trace if enabled
        trace = [] if self.store_trace else None

        # Get log level based on verbose setting
        log_level = logging.DEBUG if self.verbose else logging.INFO

        logger.log(
            log_level,
            "ReAct: Starting reasoning loop",
            extra={
                "conversation_id": manager.conversation_id,
                "max_iterations": self.max_iterations,
                "tools_available": len(tools),
            },
        )

        # ReAct loop
        for iteration in range(self.max_iterations):
            iteration_trace = {
                "iteration": iteration + 1,
                "tool_calls": [],
            }

            logger.log(
                log_level,
                "ReAct: Starting iteration",
                extra={
                    "conversation_id": manager.conversation_id,
                    "iteration": iteration + 1,
                    "max_iterations": self.max_iterations,
                },
            )

            # Generate response with tools
            response = await manager.complete(tools=tools, **kwargs)

            # Check if we have tool calls
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # No tool calls, we're done
                logger.log(
                    log_level,
                    "ReAct: No tool calls in response, finishing",
                    extra={
                        "conversation_id": manager.conversation_id,
                        "iteration": iteration + 1,
                    },
                )

                if trace is not None:
                    iteration_trace["status"] = "completed"
                    trace.append(iteration_trace)
                    await self._store_trace(manager, trace)

                return response

            num_tool_calls = len(response.tool_calls)
            logger.log(
                log_level,
                "ReAct: Executing tool calls",
                extra={
                    "conversation_id": manager.conversation_id,
                    "iteration": iteration + 1,
                    "num_tools": num_tool_calls,
                    "tools": [tc.name for tc in response.tool_calls],
                },
            )

            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_trace = {
                    "name": tool_call.name,
                    "parameters": tool_call.parameters,
                }

                try:
                    # Find the tool
                    tool = self._find_tool(tool_call.name, tools)
                    if not tool:
                        observation = f"Error: Tool '{tool_call.name}' not found"
                        tool_trace["status"] = "error"
                        tool_trace["error"] = "Tool not found"

                        logger.warning(
                            "ReAct: Tool not found",
                            extra={
                                "conversation_id": manager.conversation_id,
                                "iteration": iteration + 1,
                                "tool_name": tool_call.name,
                            },
                        )
                    else:
                        # Execute the tool
                        result = await tool.execute(**tool_call.parameters)
                        observation = f"Tool result: {result}"
                        tool_trace["status"] = "success"
                        tool_trace["result"] = str(result)

                        logger.log(
                            log_level,
                            "ReAct: Tool executed successfully",
                            extra={
                                "conversation_id": manager.conversation_id,
                                "iteration": iteration + 1,
                                "tool_name": tool_call.name,
                                "result_length": len(str(result)),
                            },
                        )

                    # Add observation to conversation
                    await manager.add_message(
                        content=f"Observation from {tool_call.name}: {observation}",
                        role="system",
                    )

                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Error executing tool {tool_call.name}: {e!s}"
                    tool_trace["status"] = "error"
                    tool_trace["error"] = str(e)

                    logger.error(
                        "ReAct: Tool execution failed",
                        extra={
                            "conversation_id": manager.conversation_id,
                            "iteration": iteration + 1,
                            "tool_name": tool_call.name,
                            "error": str(e),
                        },
                        exc_info=True,
                    )

                    await manager.add_message(content=error_msg, role="system")

                if trace is not None:
                    iteration_trace["tool_calls"].append(tool_trace)

            if trace is not None:
                iteration_trace["status"] = "continued"
                trace.append(iteration_trace)

        # Max iterations reached, generate final response without tools
        logger.log(
            log_level,
            "ReAct: Max iterations reached, generating final response",
            extra={
                "conversation_id": manager.conversation_id,
                "iterations_used": self.max_iterations,
            },
        )

        if trace is not None:
            trace.append({"status": "max_iterations_reached"})
            await self._store_trace(manager, trace)

        return await manager.complete(**kwargs)

    async def _store_trace(self, manager: Any, trace: list[dict[str, Any]]) -> None:
        """Store reasoning trace in conversation metadata.

        Args:
            manager: ConversationManager instance
            trace: Reasoning trace data
        """
        try:
            # Get existing metadata
            metadata = manager.conversation.metadata or {}

            # Add trace to metadata
            metadata["reasoning_trace"] = trace

            # Update conversation metadata
            await manager.storage.update_metadata(
                conversation_id=manager.conversation_id,
                metadata=metadata,
            )

            logger.debug(
                "ReAct: Stored reasoning trace in conversation metadata",
                extra={
                    "conversation_id": manager.conversation_id,
                    "trace_items": len(trace),
                },
            )
        except Exception as e:
            logger.warning(
                "ReAct: Failed to store reasoning trace",
                extra={
                    "conversation_id": manager.conversation_id,
                    "error": str(e),
                },
            )

    def _find_tool(self, tool_name: str, tools: list[Any]) -> Any | None:
        """Find a tool by name.

        Args:
            tool_name: Name of the tool to find
            tools: List of available tools

        Returns:
            Tool instance or None if not found
        """
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
