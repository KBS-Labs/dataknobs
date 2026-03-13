"""Summary memory implementation using LLM-based message compression."""

import logging
from collections import deque
from typing import Any

from dataknobs_llm.llm.base import AsyncLLMProvider

from .base import Memory

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "into a brief summary that captures the key points, decisions, and context. "
    "Focus on information that would be useful for continuing the conversation.\n\n"
    "Current summary (if any):\n{existing_summary}\n\n"
    "New messages to incorporate:\n{new_messages}\n\n"
    "Write a concise updated summary:"
)


class SummaryMemory(Memory):
    """Memory that summarizes older messages to maintain long context windows.

    Maintains a rolling buffer of recent messages. When the buffer exceeds
    a configurable threshold, the oldest messages are compressed into a
    running summary using the LLM provider. This trades exact message recall
    for a much longer effective context window.

    ``get_context()`` returns the summary (if any) as a system message,
    followed by the recent verbatim messages.

    Attributes:
        llm_provider: LLM provider used for generating summaries
        recent_window: Number of recent messages to keep verbatim
        summary_prompt: Template for the summarization prompt
    """

    def __init__(
        self,
        llm_provider: AsyncLLMProvider,
        recent_window: int = 10,
        summary_prompt: str | None = None,
        *,
        owns_llm_provider: bool = False,
    ) -> None:
        """Initialize summary memory.

        Args:
            llm_provider: Async LLM provider for generating summaries
            recent_window: Number of recent messages to keep verbatim.
                          When the buffer has more than ``recent_window``
                          messages, the oldest are summarized.
            summary_prompt: Custom summarization prompt template. Must
                           contain ``{existing_summary}`` and
                           ``{new_messages}`` placeholders.
            owns_llm_provider: Whether this instance owns the provider's
                lifecycle. True when a dedicated provider was created for
                this memory; False when reusing the bot's main LLM.
        """
        self.llm_provider = llm_provider
        self.recent_window = recent_window
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self._owns_llm_provider = owns_llm_provider
        self._messages: deque[dict[str, Any]] = deque()
        self._summary: str = ""

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

        Args:
            current_message: The current message (not used by summary memory,
                            kept for interface compatibility)

        Returns:
            List of message dicts. If a summary exists it is the first
            element with ``role="system"``; the remaining elements are
            the recent verbatim messages.
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

        context.extend(self._messages)
        return context

    def providers(self) -> dict[str, Any]:
        """Return the summary LLM provider if this instance owns it.

        When the bot's main LLM is reused (``owns_llm_provider=False``),
        it is not reported here — the bot already knows about its own
        main provider.
        """
        from dataknobs_bots.bot.base import PROVIDER_ROLE_SUMMARY_LLM

        if self._owns_llm_provider and self.llm_provider is not None:
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
        if self._owns_llm_provider and self.llm_provider and hasattr(self.llm_provider, "close"):
            try:
                await self.llm_provider.close()
            except Exception:
                logger.exception("Error closing summary LLM provider")

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
        """
        # Collect messages that overflow the recent window
        messages_to_summarize: list[dict[str, Any]] = []
        while len(self._messages) > self.recent_window:
            messages_to_summarize.append(self._messages.popleft())

        if not messages_to_summarize:
            return

        # Format messages for the summarization prompt
        formatted = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in messages_to_summarize
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
