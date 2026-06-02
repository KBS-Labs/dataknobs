"""Read-time conversation-history redaction primitive.

Used by:

- :class:`~dataknobs_llm.conversations.middleware.HistoryRedactionMiddleware`
  (this package) for the request-time prompt-feed rewrite path.
- Memory backends in ``dataknobs-bots`` (e.g. ``BufferMemory``) for the
  read-time ``get_context`` rewrite path.

The two callers operate on different element shapes (an
:class:`~dataknobs_llm.llm.base.LLMMessage` vs a plain ``dict[str, Any]``),
so the generic :func:`apply_history_redactions` helper is parameterized by a
trio of accessor callables — ``role_of``, ``content_of``,
``replace_content`` — that project the element into a uniform
``(role, content)`` view and produce a redacted copy. Element identity is
preserved for non-redacted roles; redacted-role elements are replaced by the
caller's ``replace_content`` factory so the input sequence is never mutated.

Why a shared primitive: both layers re-implement the same regex-by-role
rewrite. ``dataknobs-bots`` depends on ``dataknobs-llm``, so the shared home
must live here to be reachable from both the memory backends (in
``dataknobs-bots``) and the middleware (here). ``dataknobs-bots`` re-exports
these names for back-compat.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from dataknobs_common.structured_config import StructuredConfig

T = TypeVar("T")


@dataclass(frozen=True)
class HistoryRedaction(StructuredConfig):
    """One redaction pattern applied to assistant-role history at read time.

    Memory backends that surface conversation history back into an LLM prompt
    (``BufferMemory``, ``SummaryMemory``, …) can carry citation tokens or
    other domain-specific identifiers forward verbatim. From the model's view
    those tokens look like a pool of citable sources — even when this turn's
    retrieval no longer includes them — and the model will reach for them.
    ``HistoryRedaction`` is a configurable rewrite applied as messages are
    served from memory (NOT as they are stored), so the underlying buffer
    keeps the original text while the prompt-feed sees a redacted form.

    The pattern is compiled eagerly in ``__post_init__`` so an invalid regex
    (or an empty pattern, which would corrupt every message) surfaces at
    config-load time rather than later at backend construction time. The
    compiled form is stashed on the frozen dataclass for reuse by
    :func:`_compile_history_redactions`.

    Attributes:
        pattern: Regex pattern matched against assistant message content.
            Required and non-empty: an empty regex matches every position
            and combined with any non-empty replacement would shred message
            content.
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

        Stashes the compiled form on the frozen dataclass via
        ``object.__setattr__`` — the standard pattern for caching derived
        state on a frozen ``StructuredConfig`` (see e.g.
        ``PostgresAdvisoryLock._key_hash``).
        """
        if not self.pattern:
            raise ValueError(
                "HistoryRedaction.pattern is required and must be non-empty; "
                "an empty regex matches every position and would corrupt "
                "message content."
            )
        compiled = re.compile(self.pattern)
        object.__setattr__(self, "_compiled_pattern", compiled)


def _compile_history_redactions(
    redactions: Sequence[HistoryRedaction],
) -> list[tuple[re.Pattern[str], str]]:
    """Harvest the cached compiled patterns into ``(pattern, replacement)``.

    Each :class:`HistoryRedaction` eagerly compiled its pattern in
    ``__post_init__``; this helper simply collects the cached compiled forms
    into the tuple shape :func:`apply_history_redactions` expects. Backends
    call this in their ``_setup`` and cache the result; the cached list is
    then handed to :func:`apply_history_redactions` per turn. Patterns are
    applied in declared order — callers list the more specific pattern (e.g.
    a bracketed header) before the more general bare token.
    """
    return [(r._compiled_pattern, r.replacement) for r in redactions]


def apply_history_redactions(
    messages: Iterable[T],
    redactions: Sequence[tuple[re.Pattern[str], str]],
    *,
    role_of: Callable[[T], str],
    content_of: Callable[[T], str | None],
    replace_content: Callable[[T, str], T],
    redact_roles: frozenset[str] = frozenset({"assistant"}),
) -> list[T]:
    """Return a redacted copy of ``messages`` for prompt-feed.

    Args:
        messages: Iterable of message-like elements (shape opaque to the
            helper — projected via the accessor callables).
        redactions: Compiled patterns from
            :func:`_compile_history_redactions`. Empty ⇒ the elements are
            returned in a fresh list, unchanged.
        role_of: Returns the role string for one element.
        content_of: Returns the textual content of one element, or ``None``
            when the element has no textual content (e.g. an assistant
            tool-call message with ``content=None``). ``None`` is coerced to
            ``""`` before pattern application so ``pattern.sub`` is never
            called on ``None``.
        replace_content: Returns a new element with ``content`` replaced by
            the given string. MUST NOT mutate the input element.
        redact_roles: Roles whose content is rewritten. Defaults to
            assistant-only — humans rarely emit bib codes themselves.

    Returns:
        A new list. Elements whose role is not in ``redact_roles`` pass
        through by identity (the element itself is returned in the new list).
        Elements whose role IS in ``redact_roles`` are replaced by
        ``replace_content(elt, new_content)``.
    """
    if not redactions:
        return list(messages)
    out: list[T] = []
    for elt in messages:
        if role_of(elt) in redact_roles:
            # ``or ""`` coerces both a missing-string and an explicit
            # ``None`` (e.g. assistant tool-call messages with no textual
            # content) so ``pattern.sub`` does not raise ``TypeError``.
            content = content_of(elt) or ""
            for pattern, replacement in redactions:
                content = pattern.sub(replacement, content)
            out.append(replace_content(elt, content))
        else:
            out.append(elt)
    return out


# ---------------------------------------------------------------------------
# Dict-shape convenience wrapper (the memory-backend call site).
# ---------------------------------------------------------------------------


def _dict_role(msg: dict[str, Any]) -> str:
    return msg.get("role", "")


def _dict_content(msg: dict[str, Any]) -> str | None:
    # ``.get`` (no default) returns None when the key is present-but-None.
    return msg.get("content")


def _dict_replace(msg: dict[str, Any], new_content: str) -> dict[str, Any]:
    new_msg = dict(msg)
    new_msg["content"] = new_content
    return new_msg


def apply_history_redactions_to_dicts(
    messages: Iterable[dict[str, Any]],
    redactions: Sequence[tuple[re.Pattern[str], str]],
    *,
    redact_roles: frozenset[str] = frozenset({"assistant"}),
) -> list[dict[str, Any]]:
    """Apply :func:`apply_history_redactions` to dict-shaped messages.

    The dict-shape memory-backend call site uses this wrapper — it passes
    dict elements and never needs to think about the accessor trio.

    Non-redacted-role elements pass through by IDENTITY (the original dict is
    returned), matching the generic helper's contract. This is a behavior
    change from the originating dict-only helper, which shallow-copied every
    message regardless of role. The change is safe because (a) the helper
    documents the input as immutable and (b) no in-tree caller mutates the
    returned list's elements.
    """
    return apply_history_redactions(
        messages,
        redactions,
        role_of=_dict_role,
        content_of=_dict_content,
        replace_content=_dict_replace,
        redact_roles=redact_roles,
    )


__all__ = [
    "HistoryRedaction",
    "_compile_history_redactions",
    "apply_history_redactions",
    "apply_history_redactions_to_dicts",
]
