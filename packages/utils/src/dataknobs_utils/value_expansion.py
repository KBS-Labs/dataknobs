"""Conjunction-bounded value expansion for extraction grounding.

When an LLM extraction model returns a partial string value (e.g.,
``"formal"`` instead of ``"formal and academic"``), grounding checks
pass because the extracted words ARE in the user's message.  This
module provides a deterministic expansion technique that recovers
the full compound phrase from the user's message.

The algorithm finds the extracted value as a substring in the message,
then expands rightward across explicit conjunctions (``and``, ``or``,
``nor``) to include adjacent significant words that are part of the
same value phrase.  Expansion stops at natural phrase boundaries:
punctuation, field-switching patterns (``and the``, ``and set``, etc.),
or end of message.

Example::

    >>> expand_value_in_message(
    ...     "formal", "Set the tone to formal and academic",
    ...     stopwords=frozenset({"the", "to", "and", "set"}),
    ... )
    'formal and academic'
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_CONJUNCTIONS: frozenset[str] = frozenset({"and", "or", "nor"})

_DEFAULT_PHRASE_BREAK_CHARS: frozenset[str] = frozenset(".,;:!?")

_DEFAULT_FIELD_BOUNDARY_PATTERNS: tuple[str, ...] = (
    "and the",
    "and set",
    "and make",
    "and include",
    "and add",
    "and use",
    "but",
)


@dataclasses.dataclass(frozen=True)
class ValueExpansionConfig:
    """Algorithm parameters for value expansion.

    All fields have sensible defaults.  Override individual fields
    to customise expansion behaviour without rebuilding the full
    configuration.

    Attributes:
        conjunctions: Words that bridge compound value parts.
        phrase_break_chars: Characters that end a value phrase.
        field_boundary_patterns: Multi-word patterns signaling a
            new field assignment.
        require_conjunction: Only expand when an explicit conjunction
            bridges to the next word.
        expand_direction: Expansion direction.  ``"left"`` and
            ``"both"`` raise ``NotImplementedError`` (RTL-ready).
    """

    conjunctions: frozenset[str] = dataclasses.field(
        default=_DEFAULT_CONJUNCTIONS,
    )
    phrase_break_chars: frozenset[str] = dataclasses.field(
        default=_DEFAULT_PHRASE_BREAK_CHARS,
    )
    field_boundary_patterns: tuple[str, ...] = _DEFAULT_FIELD_BOUNDARY_PATTERNS
    require_conjunction: bool = True
    expand_direction: Literal["right", "left", "both"] = "right"


@runtime_checkable
class ValueExpansionPlugin(Protocol):
    """Extension point for augmented value expansion.

    A plugin receives the deterministic expansion result and can
    override it.  No concrete implementation is shipped; the protocol
    exists for future LLM-augmented expansion.
    """

    def expand(
        self,
        value: str,
        message: str,
        deterministic_result: str | None,
        config: ValueExpansionConfig,
    ) -> str | None:
        """Return an expanded value, or ``None`` to accept the deterministic result."""
        ...


_DEFAULT_CONFIG = ValueExpansionConfig()


def expand_value_in_message(
    value: str,
    message: str,
    *,
    stopwords: frozenset[str],
    config: ValueExpansionConfig | None = None,
    plugin: ValueExpansionPlugin | None = None,
) -> str | None:
    """Expand an extracted value to the full phrase in the user's message.

    Finds the extracted value as a substring in the message, then
    expands rightward to include adjacent words that appear to be part
    of the same value phrase.  Expansion stops at natural phrase
    boundaries (punctuation, field-boundary patterns, end of message).

    Tries all occurrences of the value in the message and picks the
    longest expansion.

    Args:
        value: The extracted value (e.g. ``"formal"``).
        message: The user's message.
        stopwords: Set of words to treat as non-significant.  Required
            to avoid duplicating the canonical stopword list that lives
            in the grounding module.
        config: Algorithm parameters.  Uses defaults when ``None``.
        plugin: Optional plugin that can override the deterministic
            expansion result.

    Returns:
        The expanded value using the original casing from the user's
        message if it differs from the input, or ``None`` if the
        value is already complete or not found in the message.
    """
    cfg = config or _DEFAULT_CONFIG

    if cfg.expand_direction != "right":
        raise NotImplementedError(
            f"expand_direction={cfg.expand_direction!r} is not yet supported"
        )

    msg_lower = message.lower()
    val_lower = value.lower().strip()

    if not val_lower:
        return None

    best: str | None = None
    search_start = 0

    while True:
        pos = msg_lower.find(val_lower, search_start)
        if pos == -1:
            break

        # Require word boundaries — don't match inside longer words
        if not _at_word_boundary(msg_lower, pos, len(val_lower)):
            search_start = pos + 1
            continue

        # Use original-case message for output, lowercase for matching
        original_value = message[pos:pos + len(val_lower)]
        after_original = message[pos + len(val_lower):]
        expanded_right = _expand_right(after_original, stopwords, cfg)

        candidate = (original_value + expanded_right).strip()
        if candidate.lower() != val_lower:
            if best is None or len(candidate) > len(best):
                best = candidate

        search_start = pos + 1

    if plugin is not None:
        override = plugin.expand(value, message, best, cfg)
        if override is not None:
            return override

    return best


def _at_word_boundary(text: str, pos: int, length: int) -> bool:
    """Check that the match at ``pos`` is at word boundaries on both sides."""
    if pos > 0 and text[pos - 1].isalnum():
        return False
    end = pos + length
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _expand_right(
    text: str,
    stopwords: frozenset[str],
    config: ValueExpansionConfig,
) -> str:
    """Expand rightward from the end of the extracted value.

    Only expands across explicit conjunctions to avoid absorbing
    adjacent nouns like field names.  For example, ``"formal and
    academic"`` expands, but ``"formal tone"`` does not.

    The returned text preserves the original casing from ``text``.
    Lowercase comparisons are used internally for matching.
    """
    words = text.split()
    expanded_parts: list[str] = []
    pending_connectors: list[str] = []
    has_conjunction = False

    for i, raw_word in enumerate(words):
        cleaned = raw_word.strip(".,;:!?\"'()[]{}").lower()
        original_cleaned = raw_word.strip(".,;:!?\"'()[]{}")
        has_break = any(c in config.phrase_break_chars for c in raw_word)

        # Check for field-boundary patterns (look ahead)
        remaining = " ".join(words[i:]).lower()
        is_boundary = any(
            remaining.startswith(pat)
            for pat in config.field_boundary_patterns
        )

        if is_boundary:
            break

        if cleaned in config.conjunctions:
            if has_break:
                break
            has_conjunction = True
            pending_connectors.append(raw_word)
        elif cleaned in stopwords or len(cleaned) <= 2:
            if not has_break:
                pending_connectors.append(raw_word)
            else:
                break
        else:
            # Significant word — only include if a conjunction was seen
            if config.require_conjunction and not has_conjunction:
                break
            expanded_parts.extend(pending_connectors)
            expanded_parts.append(original_cleaned)
            pending_connectors = []
            has_conjunction = False
            if has_break:
                break

    if expanded_parts:
        return " " + " ".join(expanded_parts)
    return ""
