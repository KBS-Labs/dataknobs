"""Simple reasoning strategy - direct LLM call."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer

from .base import ReasoningStrategy
from .simple_config import SimpleReasoningConfig


class SimpleReasoning(
    StructuredConfigConsumer[SimpleReasoningConfig], ReasoningStrategy
):
    """Simple reasoning strategy that makes direct LLM calls.

    This is the most straightforward strategy - it simply passes
    the conversation to the LLM and returns the response without
    any additional reasoning steps.

    Use this when:
    - You want direct, fast responses
    - The task doesn't require complex reasoning
    - You're using a powerful model that doesn't need guidance

    Example:
        ```python
        strategy = SimpleReasoning()
        response = await strategy.generate(
            manager=conversation_manager,
            llm=llm_provider,
            temperature=0.7
        )
        ```
    """

    #: Typed config consumed via the ``StructuredConfigConsumer`` mixin —
    #: ``cls.from_config({...})`` / ``cls(SimpleReasoningConfig(...))`` /
    #: ``cls(greeting_template=...)`` all reach the same typed ``self.config``.
    CONFIG_CLS: ClassVar[type[SimpleReasoningConfig]] = SimpleReasoningConfig

    def _setup(self) -> None:
        """Bind the greeting template from the typed config."""
        self._greeting_template = self.config.greeting_template

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response with a simple LLM call.

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance (not used directly)
            tools: Optional list of tools
            **kwargs: Generation parameters

        Returns:
            LLM response
        """
        # Use the conversation manager's generate method
        # which handles the LLM call with the conversation history
        return await manager.complete(tools=tools, **kwargs)

    async def stream_generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream response with true token-level streaming.

        Delegates to ``manager.stream_complete()`` which yields
        ``LLMStreamResponse`` chunks as they arrive from the provider.

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance (not used directly)
            tools: Optional list of tools
            **kwargs: Generation parameters

        Yields:
            LLM stream response chunks
        """
        async for chunk in manager.stream_complete(tools=tools, **kwargs):
            yield chunk
