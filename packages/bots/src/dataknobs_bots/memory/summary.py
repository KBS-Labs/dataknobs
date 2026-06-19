"""Summary memory implementation using LLM-based message compression."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from dataknobs_bots.prompts.resolver import PromptResolver

from dataknobs_bots.prompts.memory import DEFAULT_SUMMARY_PROMPT
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.structured_config import StructuredConfigConsumer

from .base import Memory, apply_history_redactions, compile_history_redactions
from .config import SummaryMemoryConfig

logger = logging.getLogger(__name__)


async def _resolve_summary_llm(
    llm_config: dict[str, Any] | None,
    fallback_provider: Any | None,
) -> Any:
    """Resolve the LLM provider for summary memory.

    If ``llm_config`` is provided, a dedicated provider is created and
    initialized from it. Otherwise the ``fallback_provider`` (typically
    the bot's own LLM) is used.

    Args:
        llm_config: Optional dedicated LLM-provider config.
        fallback_provider: Provider to use when no dedicated LLM is
            configured.

    Returns:
        An initialised ``AsyncLLMProvider``.

    Raises:
        ValueError: If neither a dedicated LLM config nor a fallback is
            available.
    """
    if llm_config is not None:
        from dataknobs_llm.llm import LLMProviderFactory

        factory = LLMProviderFactory(is_async=True)
        provider = factory.create(llm_config)
        await provider.initialize()
        return provider

    if fallback_provider is not None:
        return fallback_provider

    raise ValueError(
        "Summary memory requires an LLM provider. Either include an 'llm' "
        "section in the memory config or pass llm_provider to "
        "create_memory_from_config()."
    )


class SummaryMemory(StructuredConfigConsumer[SummaryMemoryConfig], Memory):
    """Memory that summarizes older messages to maintain long context windows.

    Maintains a rolling buffer of recent messages. When the buffer exceeds
    a configurable threshold, the oldest messages are compressed into a
    running summary using the LLM provider. This trades exact message recall
    for a much longer effective context window.

    ``get_context()`` returns the summary (if any) as a system message,
    followed by the recent verbatim messages.

    Construct from config (``await SummaryMemory.from_config({...},
    llm_provider=fallback)`` — a dedicated provider built from a
    ``llm`` config section is owned by the memory, otherwise the
    injected fallback is reused) or from a pre-built provider
    (``SummaryMemory.from_components({"recent_window": …},
    llm_provider=provider)`` — caller-owned).

    Attributes:
        llm_provider: LLM provider used for generating summaries
        recent_window: Number of recent messages to keep verbatim
        summary_prompt: Template for the summarization prompt
    """

    CONFIG_CLS: ClassVar[type[SummaryMemoryConfig]] = SummaryMemoryConfig

    @classmethod
    async def from_config(  # type: ignore[override]
        cls, config: Any, **components: Any
    ) -> SummaryMemory:
        """Create SummaryMemory from configuration (async warmup).

        Construction is asynchronous — a dedicated provider built from the
        ``llm`` config section must be initialized — so ``from_config``
        delegates to :meth:`from_config_async` to run ``_ainit``. Pass the
        bot's main LLM as the ``llm_provider`` keyword to use it as the
        fallback when no dedicated ``llm`` section is configured.
        """
        return await cls.from_config_async(config, **components)

    def _setup(self) -> None:
        """Initialize the message buffer and running summary.

        The LLM provider, ownership flag, and resolved prompt are bound
        by :meth:`_ainit` (config-driven) or :meth:`_adopt_components`
        (pre-built injection).
        """
        self.recent_window = self.config.recent_window
        self.llm_provider: Any = None
        self._owns_llm_provider = False
        self._prompt_resolver: PromptResolver | None = None
        self.summary_prompt: str = DEFAULT_SUMMARY_PROMPT
        self._messages: deque[dict[str, Any]] = deque()
        self._summary: str = ""
        self._compiled_redactions = compile_history_redactions(
            self.config.history_redactions
        )

    def _resolve_prompt(self, prompt_resolver: PromptResolver | None) -> None:
        """Resolve ``summary_prompt`` (explicit > library > default).

        Shared by the config and pre-built construction paths.
        """
        self._prompt_resolver = prompt_resolver
        if self.config.summary_prompt is not None:
            self.summary_prompt = self.config.summary_prompt
        elif prompt_resolver is not None:
            resolved = prompt_resolver.resolve("memory.summary")
            self.summary_prompt = resolved if resolved else DEFAULT_SUMMARY_PROMPT
        else:
            self.summary_prompt = DEFAULT_SUMMARY_PROMPT

    async def _ainit(
        self,
        *,
        llm_provider: Any = None,
        prompt_resolver: PromptResolver | None = None,
        **_: Any,
    ) -> None:
        """Resolve the summary LLM and prompt from config + injection.

        A dedicated provider built from the ``llm`` config section is
        owned by this memory; the injected ``llm_provider`` fallback
        (the bot's main LLM) is not.
        """
        if self._prebuilt:
            return
        has_dedicated_llm = self.config.llm is not None
        self.llm_provider = await _resolve_summary_llm(
            self.config.llm, llm_provider
        )
        self._owns_llm_provider = has_dedicated_llm
        self._resolve_prompt(prompt_resolver)

    def _adopt_components(
        self,
        *,
        llm_provider: Any = None,
        prompt_resolver: PromptResolver | None = None,
        **_: Any,
    ) -> None:
        """Adopt a caller-owned LLM provider for ``from_components``."""
        if llm_provider is None:
            raise TypeError(
                "SummaryMemory.from_components requires llm_provider"
            )
        self.llm_provider = llm_provider
        self._owns_llm_provider = False
        self._resolve_prompt(prompt_resolver)

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message and trigger summarization if the buffer is full.

        When the number of buffered messages exceeds ``recent_window``,
        the oldest messages are summarized into the running summary using
        the LLM provider. On LLM failure, older messages are dropped to
        keep the buffer within bounds (graceful degradation).

        Args:
            content: Message content
            role: Message role (user, assistant, system)
            metadata: Optional metadata for the message
        """
        self._messages.append(
            {"content": content, "role": role, "metadata": metadata or {}}
        )

        if len(self._messages) > self.recent_window:
            await self._summarize_oldest()

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Return the running summary followed by recent messages.

        Configured ``history_redactions`` are applied to assistant-role
        entries in the recent buffer; the summary header (system-role)
        and user messages pass through unchanged. The underlying
        ``self._messages`` deque is not mutated.

        Args:
            current_message: The current message (not used by summary memory,
                            kept for interface compatibility)

        Returns:
            List of message dicts. If a summary exists it is the first
            element with ``role="system"``; the remaining elements are
            the redacted view of the recent verbatim messages.
        """
        context: list[dict[str, Any]] = []

        if self._summary:
            context.append(
                {
                    "content": f"[Conversation summary]: {self._summary}",
                    "role": "system",
                    "metadata": {"is_summary": True},
                }
            )

        # Redact assistant content in the recent buffer; system + user
        # roles pass through (default redact_roles={"assistant"}).
        context.extend(
            apply_history_redactions(self._messages, self._compiled_redactions)
        )
        return context

    def providers(self) -> dict[str, Any]:
        """Return the summary LLM provider for catalog registration.

        Always reports the provider for discovery and observability.
        The ``_owns_llm_provider`` flag controls lifecycle (``close()``),
        not visibility — consistent with VectorMemory, RAGKnowledgeBase,
        and WizardReasoning.
        """
        from dataknobs_bots.bot.base import PROVIDER_ROLE_SUMMARY_LLM

        if self.llm_provider is not None:
            return {PROVIDER_ROLE_SUMMARY_LLM: self.llm_provider}
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace the summary LLM provider if the role matches."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_SUMMARY_LLM

        if role == PROVIDER_ROLE_SUMMARY_LLM:
            self.llm_provider = provider
            return True
        return False

    async def close(self) -> None:
        """Close the LLM provider if this instance owns it.

        When a dedicated provider was created for this memory (via the
        ``llm`` config key), this instance owns its lifecycle. When the
        bot's main LLM was passed in as a fallback, the bot owns it.
        """
        await close_if_owned(
            self.llm_provider,
            self._owns_llm_provider,
            on_error=lambda _exc: logger.exception(
                "Error closing summary LLM provider"
            ),
        )

    async def clear(self) -> None:
        """Clear both the running summary and the message buffer."""
        self._messages.clear()
        self._summary = ""

    async def pop_messages(self, count: int = 2) -> list[dict[str, Any]]:
        """Remove and return the last N messages from the recent window.

        Only messages still in the recent buffer can be popped. Messages that
        have already been summarized are irreversibly compressed and cannot be
        individually removed.

        Args:
            count: Number of messages to remove from the end.

        Returns:
            The removed messages in the order they were stored.

        Raises:
            ValueError: If count exceeds available (unsummarized) messages
                or is < 1.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > len(self._messages):
            raise ValueError(
                f"Cannot pop {count} messages, only {len(self._messages)} "
                f"unsummarized messages available in the recent window"
            )
        removed = []
        for _ in range(count):
            removed.append(self._messages.pop())
        removed.reverse()
        return removed

    async def _summarize_oldest(self) -> None:
        """Compress the oldest messages into the running summary.

        Removes messages beyond ``recent_window`` from the buffer,
        formats them into a prompt, and asks the LLM to produce an
        updated summary. On LLM failure the messages are simply
        discarded (graceful degradation to buffer-only behaviour).

        Configured ``history_redactions`` are applied to the oldest
        messages BEFORE they are formatted into the summarization
        prompt. Otherwise the running summary (a system-role header
        the default ``redact_roles`` deliberately leaves untouched on
        the read path) would carry the citation tokens forward — a
        leak that bypasses the read-time guarantee.
        """
        # Collect messages that overflow the recent window
        messages_to_summarize: list[dict[str, Any]] = []
        while len(self._messages) > self.recent_window:
            messages_to_summarize.append(self._messages.popleft())

        if not messages_to_summarize:
            return

        # Redact assistant content before the summarizer sees it. The
        # apply helper returns a fresh list and rewrites only the
        # ``content`` of redaction-eligible roles; non-eligible rows
        # pass through by identity (cheap when no redactions are
        # configured — pass-through fast path).
        redacted_for_summary = apply_history_redactions(
            messages_to_summarize, self._compiled_redactions
        )

        # Format messages for the summarization prompt
        formatted = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in redacted_for_summary
        )

        prompt = self.summary_prompt.format(
            existing_summary=self._summary or "(none)",
            new_messages=formatted,
        )

        try:
            from dataknobs_llm.llm.base import LLMMessage

            response = await self.llm_provider.complete(
                messages=[LLMMessage(role="user", content=prompt)],
            )
            self._summary = response.content
            logger.debug(
                "Summary updated, %d messages compressed",
                len(messages_to_summarize),
            )
        except Exception:
            logger.warning(
                "LLM summarization failed, %d messages dropped",
                len(messages_to_summarize),
                exc_info=True,
            )
            # Graceful degradation: messages are already removed from the
            # buffer. The previous summary (if any) is retained.
