"""Base memory interface for bot memory implementations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class HistoryRedaction(StructuredConfig):
    """One redaction pattern applied to assistant-role history at read time.

    Memory backends that surface conversation history back into an LLM prompt
    (``BufferMemory``, ``SummaryMemory``, …) can carry citation tokens or
    other domain-specific identifiers forward verbatim. From the model's
    view those tokens look like a pool of citable sources — even when this
    turn's retrieval no longer includes them — and the model will reach for
    them. ``HistoryRedaction`` is a configurable rewrite applied as
    messages are served from memory (NOT as they are stored), so the
    underlying buffer keeps the original text while the prompt-feed sees
    a redacted form.

    The pattern is compiled eagerly in ``__post_init__`` so an invalid
    regex surfaces at config-load time (with the offending pattern in the
    exception) rather than later at backend construction time. The
    compiled form is stashed on the frozen dataclass for reuse by
    :func:`_compile_history_redactions`.

    Attributes:
        pattern: Regex pattern matched against assistant message content.
            Required and non-empty: an empty regex matches every position
            and combined with any non-empty replacement would shred
            message content.
        replacement: String substituted for each match. Defaults to ``""``
            (strip the match). Use a placeholder like ``"[prior citation]"``
            to preserve sentence flow.
    """

    pattern: str = ""
    replacement: str = ""

    def __post_init__(self) -> None:
        """Validate the pattern and eagerly compile it.

        Rejecting an empty pattern at config-load time avoids a class of
        silent footgun: ``re.compile("")`` matches at every position, so
        combined with any non-empty replacement it would corrupt every
        message. Compiling here also surfaces invalid regex syntax at the
        config-load boundary rather than at backend construction time.
        """
        if not self.pattern:
            raise ValueError(
                "HistoryRedaction.pattern is required and must be non-empty; "
                "an empty regex matches every position and would corrupt "
                "message content."
            )
        # Eagerly validate the regex syntax. Stash the compiled form on
        # the frozen dataclass for reuse — bypassing the frozen guard
        # with object.__setattr__ is the standard pattern for caching
        # derived state on a frozen StructuredConfig (see e.g.
        # PostgresAdvisoryLock._key_hash).
        compiled = re.compile(self.pattern)
        object.__setattr__(self, "_compiled_pattern", compiled)


def _compile_history_redactions(
    redactions: Sequence[HistoryRedaction],
) -> list[tuple[re.Pattern[str], str]]:
    """Compile a sequence of :class:`HistoryRedaction` once for reuse.

    Backends call this in their ``_setup`` and cache the result; the cached
    list is then handed to :func:`apply_history_redactions` per turn.
    Patterns are applied in declared order — callers list the more specific
    pattern (e.g. a bracketed header) before the more general bare token.

    Each :class:`HistoryRedaction` already eagerly compiled its pattern in
    ``__post_init__``; this function simply harvests the cached compiled
    forms into the ``(pattern, replacement)`` tuple shape that
    :func:`apply_history_redactions` expects.
    """
    return [(r._compiled_pattern, r.replacement) for r in redactions]


def apply_history_redactions(
    messages: Iterable[dict[str, Any]],
    redactions: Sequence[tuple[re.Pattern[str], str]],
    *,
    redact_roles: frozenset[str] = frozenset({"assistant"}),
) -> list[dict[str, Any]]:
    """Return a redacted copy of ``messages`` for prompt-feed.

    Each message whose ``role`` is in ``redact_roles`` (default:
    ``"assistant"`` only) has its ``content`` rewritten by applying the
    compiled ``(pattern, replacement)`` pairs in order. Other messages
    pass through unchanged. The input iterable is not mutated.

    Args:
        messages: The raw messages to transform (typically from the
            backend's internal buffer / retrieval).
        redactions: Compiled patterns from :func:`_compile_history_redactions`.
            Empty ⇒ messages are returned as a shallow-copied list with no
            rewrite.
        redact_roles: Roles whose ``content`` should be redacted. Defaults
            to assistant-only — humans rarely emit bib codes themselves.

    Returns:
        A new list of new message dicts. Original dicts are not mutated.
    """
    if not redactions:
        return list(messages)
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in redact_roles:
            # ``msg.get("content")`` returns ``None`` when the key is present
            # but the value is None (e.g. assistant tool-call messages with no
            # textual content) — ``or ""`` coerces both the missing and the
            # explicit-None case to an empty string so ``pattern.sub`` does
            # not raise ``TypeError``.
            content = msg.get("content") or ""
            for pattern, replacement in redactions:
                content = pattern.sub(replacement, content)
            new_msg = dict(msg)
            new_msg["content"] = content
            out.append(new_msg)
        else:
            out.append(dict(msg))
    return out


class Memory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add message to memory.

        Args:
            content: Message content
            role: Message role (user, assistant, system, etc.)
            metadata: Optional metadata for the message
        """
        pass

    @abstractmethod
    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        """Get relevant context for current message.

        Args:
            current_message: The current message to get context for

        Returns:
            List of relevant message dictionaries
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory."""
        pass

    def providers(self) -> dict[str, Any]:
        """Return LLM providers managed by this memory, keyed by role.

        Subsystems declare the providers they own so that the bot can
        register them in the provider catalog without reaching into
        private attributes.  The default returns an empty dict (no
        providers).

        Returns:
            Dict mapping provider role names to provider instances.
        """
        return {}

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace a provider managed by this memory.

        Called by ``inject_providers`` to wire a test provider into the
        actual subsystem, not just the registry catalog.  The default
        returns ``False`` (role not recognized).  Concrete subclasses
        override to accept their known roles.

        Args:
            role: Provider role name (e.g. ``PROVIDER_ROLE_MEMORY_EMBEDDING``).
            provider: Replacement provider instance.

        Returns:
            ``True`` if the role was recognized and the provider updated,
            ``False`` otherwise.
        """
        return False

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release resources held by this memory implementation.

        The default is a no-op.  Subclasses that create providers or open
        connections (e.g. ``VectorMemory``, ``SummaryMemory``) should
        override to clean up.
        """

    async def pop_messages(self, count: int = 2) -> list[dict[str, Any]]:
        """Remove and return the last N messages from memory.

        Used for conversation undo. The count is determined by the caller
        based on node depth difference (not a fixed 2).

        Args:
            count: Number of messages to remove from the end.

        Returns:
            The removed messages in the order they were stored.

        Raises:
            NotImplementedError: If the implementation does not support undo.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pop_messages"
        )
