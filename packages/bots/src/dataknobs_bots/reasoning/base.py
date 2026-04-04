"""Base reasoning strategy for DynaBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import jinja2

from dataknobs_llm import LLMResponse

if TYPE_CHECKING:
    from dataknobs_bots.bot.turn import ToolExecution


@dataclass(frozen=True)
class StrategyCapabilities:
    """Declares what a reasoning strategy manages autonomously.

    Used by ``DynaBot`` and other consumers to decide which orchestration
    steps to perform (e.g. source construction, auto-context) without
    hard-coding strategy names.

    All fields default to ``False``; concrete strategies override only
    the capabilities they possess.  New fields can be added with
    ``default=False`` without breaking existing strategies.

    Attributes:
        manages_sources: Strategy manages its own retrieval sources
            (grounded/hybrid).  When ``True``, ``DynaBot`` performs
            config-driven source construction after factory creation
            and disables redundant ``auto_context``.
    """

    manages_sources: bool = False


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
        system_prompt_override: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream an LLM completion.

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

    @classmethod
    def capabilities(cls) -> StrategyCapabilities:
        """Declare what this strategy manages autonomously.

        The default returns no capabilities.  Concrete strategies
        override to declare their actual capabilities.

        Returns:
            Frozen dataclass describing strategy capabilities.
        """
        return StrategyCapabilities()

    @classmethod
    def from_config(cls, config: dict[str, Any], **_kwargs: Any) -> Self:
        """Create a strategy instance from a configuration dict.

        The base implementation extracts ``greeting_template`` and
        passes it to the constructor.  Concrete strategies with richer
        configuration override this classmethod.

        Args:
            config: Strategy configuration dict.
            **_kwargs: Additional context (e.g. ``knowledge_base``).

        Returns:
            Configured strategy instance.
        """
        return cls(greeting_template=config.get("greeting_template"))

    @classmethod
    def get_source_configs(cls, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract source configuration dicts from a strategy config.

        ``DynaBot`` calls this after creating the strategy to discover
        which sources to construct and wire in via :meth:`add_source`.
        The default looks for a top-level ``"sources"`` key, which is
        the convention used by ``GroundedReasoning``.

        Strategies with non-standard config layouts (e.g.
        ``HybridReasoning``, where sources are nested under
        ``"grounded"``) override this to return the correct list.

        Args:
            config: The full strategy configuration dict.

        Returns:
            List of source configuration dicts (may be empty).
        """
        return config.get("sources", [])

    def __init__(self, *, greeting_template: str | None = None) -> None:
        self._greeting_template = greeting_template
        self._tool_executions: list[ToolExecution] = []

    def get_and_clear_tool_executions(self) -> list[ToolExecution]:
        """Return tool executions recorded during the last generate() call.

        Strategies that execute tools (e.g. ReAct) append
        ``ToolExecution`` records to ``self._tool_executions`` during
        their generation loop.  DynaBot calls this after
        ``generate()`` returns to collect the records and fire
        ``on_tool_executed`` middleware hooks.

        Returns:
            List of tool execution records (cleared after retrieval).
        """
        result = list(self._tool_executions)
        self._tool_executions.clear()
        return result

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

    def add_source(self, source: Any) -> None:
        """Add a retrieval source to this strategy.

        Strategies that declare ``manages_sources=True`` in their
        :meth:`capabilities` MUST override this method.  ``DynaBot``
        calls it during config-driven source construction.

        The default raises ``NotImplementedError`` so that a 3rd-party
        strategy that forgets to implement it fails loudly.

        Args:
            source: A ``GroundedSource`` instance (or compatible).

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares manages_sources=True "
            f"but does not implement add_source()"
        )

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
