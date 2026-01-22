"""Context-aware tool base class for tools that need execution context.

This module provides ContextAwareTool, a base class for tools that need
access to conversation state, wizard data, or other runtime context
during execution.
"""

from abc import abstractmethod
from typing import Any

from dataknobs_llm.tools.base import Tool
from dataknobs_llm.tools.context import ToolExecutionContext


class ContextAwareTool(Tool):
    """Base class for tools that need execution context.

    Unlike the base Tool class which only receives LLM-provided parameters,
    ContextAwareTool implementations receive a ToolExecutionContext that
    provides access to:

    - Conversation metadata
    - Wizard state (current stage, collected data, history)
    - User/client identifiers
    - Custom context values

    Tools extend this class when they need information beyond what the
    LLM provides in its tool call. For example:

    - A preview tool that needs wizard-collected data
    - A save tool that needs the conversation ID
    - A search tool that filters by user ID

    The execute() method handles context extraction from kwargs,
    maintaining backwards compatibility with callers that don't
    provide context.

    Example:
        ```python
        class PreviewConfigTool(ContextAwareTool):
            def __init__(self):
                super().__init__(
                    name="preview_config",
                    description="Preview the bot configuration being built"
                )

            @property
            def schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["yaml", "json", "summary"],
                            "default": "summary"
                        }
                    }
                }

            async def execute_with_context(
                self,
                context: ToolExecutionContext,
                format: str = "summary",
                **kwargs
            ) -> dict:
                # Access wizard data from context
                wizard_data = {}
                if context.wizard_state:
                    wizard_data = context.wizard_state.collected_data

                # Build and format configuration
                config = self._build_config(wizard_data)
                return {"config": config, "format": format}
        ```

    Notes:
        - Static dependencies (databases, services) should still use
          constructor injection
        - Context is for dynamic, per-request information
        - Tools can combine both patterns
    """

    @abstractmethod
    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Execute the tool with context available.

        This is the primary method to implement. The context parameter
        provides access to conversation state, wizard data, and other
        runtime information.

        Args:
            context: Execution context with conversation/wizard state
            **kwargs: Tool parameters from LLM tool call

        Returns:
            Tool execution result (JSON-serializable)

        Raises:
            Exception: If tool execution fails
        """
        pass

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool, extracting context if provided.

        This method provides backwards compatibility. When context
        is passed via the `_context` keyword argument (by reasoning
        strategies that support context injection), it's extracted
        and passed to execute_with_context(). When called without
        context, an empty context is provided.

        Reasoning strategies inject context like this:
            ```python
            context = ToolExecutionContext.from_manager(manager)
            result = await tool.execute(**params, _context=context)
            ```

        Args:
            **kwargs: Tool parameters, optionally including _context

        Returns:
            Tool execution result
        """
        # Extract context from kwargs if provided
        context = kwargs.pop("_context", None)

        if context is None:
            context = ToolExecutionContext.empty()

        return await self.execute_with_context(context, **kwargs)


class ContextEnhancedTool(ContextAwareTool):
    """Wrapper that adds context awareness to an existing tool.

    This wrapper allows existing Tool implementations to receive
    context without modification. The context is made available
    via a callback function that the wrapped tool can optionally
    use.

    This is useful when:
    - You can't modify the tool class
    - You want to gradually migrate tools to context awareness
    - You need to inject context into third-party tools

    Example:
        ```python
        # Existing tool that checks for wizard_data in kwargs
        class LegacyPreviewTool(Tool):
            async def execute(self, format: str = "summary", **kwargs):
                wizard_data = kwargs.get("wizard_data", {})
                return {"config": wizard_data}

        # Wrap it to inject wizard_data from context
        def inject_wizard_data(context: ToolExecutionContext) -> dict:
            if context.wizard_state:
                return {"wizard_data": context.wizard_state.collected_data}
            return {}

        enhanced_tool = ContextEnhancedTool(
            LegacyPreviewTool(),
            context_injector=inject_wizard_data
        )
        ```

    Attributes:
        _inner_tool: The wrapped tool instance
        _context_injector: Function that extracts kwargs from context
    """

    def __init__(
        self,
        inner_tool: Tool,
        context_injector: Any | None = None,
    ):
        """Initialize the wrapper.

        Args:
            inner_tool: Tool instance to wrap
            context_injector: Optional callable(context) -> dict of kwargs
                to inject into the inner tool's execute call
        """
        super().__init__(
            name=inner_tool.name,
            description=inner_tool.description,
            metadata=inner_tool.metadata,
        )
        self._inner_tool = inner_tool
        self._context_injector = context_injector

    @property
    def schema(self) -> dict[str, Any]:
        """Get schema from inner tool."""
        return self._inner_tool.schema

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Execute inner tool with injected context values.

        Args:
            context: Execution context
            **kwargs: Tool parameters

        Returns:
            Inner tool's execution result
        """
        # Inject additional kwargs from context if injector provided
        if self._context_injector:
            injected = self._context_injector(context)
            if injected:
                # Injected values don't override explicit kwargs
                for key, value in injected.items():
                    if key not in kwargs:
                        kwargs[key] = value

        return await self._inner_tool.execute(**kwargs)


def default_wizard_data_injector(context: ToolExecutionContext) -> dict[str, Any]:
    """Default injector that provides wizard_data from context.

    This is a convenience injector for tools that expect a
    `wizard_data` parameter containing the wizard's collected data.

    Args:
        context: Execution context

    Returns:
        Dict with wizard_data key if wizard state available

    Example:
        ```python
        enhanced_tool = ContextEnhancedTool(
            legacy_tool,
            context_injector=default_wizard_data_injector
        )
        ```
    """
    if context.wizard_state:
        return {"wizard_data": context.wizard_state.collected_data}
    return {}
