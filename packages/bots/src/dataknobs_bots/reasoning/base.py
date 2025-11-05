"""Base reasoning strategy for DynaBot."""

from abc import ABC, abstractmethod
from typing import Any


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies.

    Reasoning strategies control how the bot processes information
    and generates responses. Different strategies can implement
    different levels of reasoning complexity.

    Examples:
        - Simple: Direct LLM call
        - Chain-of-Thought: Break down reasoning into steps
        - ReAct: Reason and act in a loop with tools
    """

    @abstractmethod
    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response using this reasoning strategy.

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            LLM response object

        Example:
            ```python
            response = await strategy.generate(
                manager=conversation_manager,
                llm=llm_provider,
                tools=[search_tool, calculator_tool],
                temperature=0.7,
                max_tokens=1000
            )
            ```
        """
        pass
