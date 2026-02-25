"""Base reasoning strategy for DynaBot."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReasoningManagerProtocol(Protocol):
    """Protocol defining the manager interface for reasoning strategies.

    ConversationManager implements this protocol. Test managers must also
    conform to it. This formalizes the implicit interface that reasoning
    strategies depend on, preventing interface drift between production
    and test code.

    The protocol covers the minimum interface needed by all reasoning
    strategies (Simple, ReAct, Wizard). Individual strategies may access
    additional manager features beyond this protocol.
    """

    @property
    def system_prompt(self) -> str:
        """The current system prompt for this conversation."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Conversation-level metadata dict (read/write)."""
        ...

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the conversation as dicts."""
        ...

    async def add_message(
        self,
        role: str,
        content: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional per-message metadata
            **kwargs: Additional parameters (prompt_name, params, etc.)
        """
        ...

    async def complete(
        self,
        *,
        system_prompt_override: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate an LLM completion.

        Args:
            system_prompt_override: Override system prompt for this call only
            tools: Optional list of tools available for this completion
            **kwargs: Additional parameters (branch_name, metadata, etc.)
        """
        ...


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
        manager: ReasoningManagerProtocol,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response using this reasoning strategy.

        Args:
            manager: ConversationManager or compatible manager instance
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
