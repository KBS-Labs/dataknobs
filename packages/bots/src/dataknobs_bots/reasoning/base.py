"""Base reasoning strategy for DynaBot."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import jinja2

from dataknobs_llm import LLMResponse


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

    def stream_complete(
        self,
        *,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream an LLM completion.

        Args:
            tools: Optional list of tools available for this completion
            **kwargs: Additional parameters (branch_name, metadata, etc.)
        """
        ...


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies.

    Reasoning strategies control how the bot processes information
    and generates responses. Different strategies can implement
    different levels of reasoning complexity.

    All strategies support an optional ``greeting_template`` — a Jinja2
    template string rendered with ``initial_context`` variables when
    ``greet()`` is called.  Strategies that need richer greeting behavior
    (e.g. ``WizardReasoning`` with FSM-driven stage responses) override
    ``greet()`` entirely.

    Args:
        greeting_template: Optional Jinja2 template for bot-initiated
            greetings.  Variables from ``initial_context`` are available
            as top-level template variables (e.g. ``{{ user_name }}``).

    Examples:
        - Simple: Direct LLM call
        - Chain-of-Thought: Break down reasoning into steps
        - ReAct: Reason and act in a loop with tools
    """

    def __init__(self, *, greeting_template: str | None = None) -> None:
        self._greeting_template = greeting_template

    async def greet(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        *,
        initial_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Generate an initial bot greeting before the user speaks.

        The default implementation renders ``greeting_template`` (if set)
        with ``initial_context`` variables using Jinja2 and returns the
        result as an ``LLMResponse``.  Returns ``None`` when no template
        is configured.

        ``WizardReasoning`` fully overrides this with FSM-driven greeting
        generation from the wizard's start stage.

        Args:
            manager: ConversationManager or compatible manager instance
            llm: LLM provider instance
            initial_context: Optional dict of data available as Jinja2
                template variables (e.g. ``{"user_name": "Alice"}``
                makes ``{{ user_name }}`` resolve to ``"Alice"``).
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse if a greeting was generated, or None
        """
        if self._greeting_template is None:
            return None
        context = initial_context or {}
        env = jinja2.Environment(undefined=jinja2.Undefined)
        text = env.from_string(self._greeting_template).render(**context)
        return LLMResponse(content=text, model="template", finish_reason="stop")

    def providers(self) -> dict[str, Any]:
        """Return LLM providers managed by this strategy, keyed by role.

        Subsystems declare the providers they own so that the bot can
        register them in the provider catalog without reaching into
        private attributes.  The default returns an empty dict (no
        providers).

        Returns:
            Dict mapping provider role names to provider instances.
        """
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace a provider managed by this strategy.

        Called by ``inject_providers`` to wire a test provider into the
        actual subsystem, not just the registry catalog.  The default
        returns ``False`` (role not recognized).  Concrete subclasses
        override to accept their known roles.

        Args:
            role: Provider role name (e.g. ``PROVIDER_ROLE_EXTRACTION``).
            provider: Replacement provider instance.

        Returns:
            ``True`` if the role was recognized and the provider updated,
            ``False`` otherwise.
        """
        return False

    async def close(self) -> None:  # noqa: B027
        """Release resources held by this strategy.

        Default no-op. Subclasses that hold resources (LLM providers,
        database connections, asyncio tasks) should override to release
        them. Called by ``DynaBot.close()``.
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

    async def stream_generate(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream response using this reasoning strategy.

        The default implementation wraps ``generate()`` and yields the
        complete response as a single item.  Subclasses that support true
        token-level streaming (e.g. ``SimpleReasoning``) should override
        this to yield incremental chunks.

        Args:
            manager: ConversationManager or compatible manager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Additional generation parameters

        Yields:
            LLM response or stream chunk objects
        """
        result = await self.generate(manager, llm, tools=tools, **kwargs)
        yield result
