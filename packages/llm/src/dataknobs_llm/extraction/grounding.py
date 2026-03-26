"""Extraction grounding — validate extracted values against user input.

Checks whether values produced by :class:`SchemaExtractor` (or any
extraction pipeline) are *grounded* in the user's actual message.
Ungrounded values are likely hallucinations — the extraction model
inferred a value the user never expressed.

This module provides the core grounding logic as a standalone utility.
Consumers include:

- **Wizard merge filtering** — ``SchemaGroundingFilter`` delegates its
  type-dispatched checks here, then applies wizard-specific merge
  policy (protect existing values from ungrounded overwrites).
- **Grounded reasoning** — ``_extract_intent()`` drops ungrounded
  optional intent fields (e.g. ``output_style``) so the resolution
  cascade falls through to config/session defaults.

Grounding rules by type:

- **string + enum**: enum value appears in message (word-boundary)
- **string (general)**: word overlap ratio above threshold
- **boolean**: field-related keywords found in message, with optional
  value-direction checking via negation detection
- **integer/number**: literal number appears in message (word-boundary)
- **array**: at least one element appears in message (word-boundary);
  empty arrays grounded via negation keywords

Per-field configuration via ``x-extraction`` JSON Schema annotations:

- ``grounding``: ``"exact"`` | ``"fuzzy"`` | ``"skip"``
- ``empty_allowed``: ``true`` | ``false``
- ``overlap_threshold``: ``float``
- ``check_direction``: ``true`` | ``false`` (boolean fields)
- ``negation_keywords``: ``list[str]``
- ``negation_proximity``: ``int``

Configuration cascade::

    module defaults (DEFAULT_STOPWORDS / DEFAULT_NEGATION_KEYWORDS)
      → GroundingConfig (consumer-level override)
        → per-field x-extraction annotation (field-level override)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Module-level defaults
# ──────────────────────────────────────────────────────────────────

DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "of", "for", "to", "or", "and", "whether",
    "it", "in", "on", "at", "by", "with", "this", "that", "be", "as",
    "can", "do", "does", "did", "has", "have", "had", "was", "were",
    "are", "been", "being", "will", "would", "could", "should", "may",
    "might", "shall", "must", "not", "but", "if", "then", "than",
    "so", "just", "also", "very", "too", "really", "quite",
})

DEFAULT_NEGATION_KEYWORDS: frozenset[str] = frozenset({
    "no", "skip", "none", "remove", "clear", "without", "blank",
    "empty", "delete", "disable", "disabled", "off",
})


# ──────────────────────────────────────────────────────────────────
# Configuration and result types
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroundingConfig:
    """Configuration for extraction grounding checks.

    Attributes:
        overlap_threshold: Default minimum word-overlap ratio for
            string grounding (0.0-1.0).  Overridable per-field via
            ``x-extraction.overlap_threshold``.
        stopwords: Words too common to carry grounding signal.
            Used by :func:`significant_words` for word extraction.
        negation_keywords: Keywords indicating intent to clear,
            remove, or negate a value.  Used by boolean direction
            checking, empty-string grounding, and empty-array
            grounding.  Overridable per-field via
            ``x-extraction.negation_keywords``.
    """

    overlap_threshold: float = 0.5
    stopwords: frozenset[str] = field(default=DEFAULT_STOPWORDS)
    negation_keywords: frozenset[str] = field(default=DEFAULT_NEGATION_KEYWORDS)


@dataclass(frozen=True)
class FieldGroundingResult:
    """Grounding verdict for a single extracted field.

    Attributes:
        field: Schema property name.
        grounded: Whether the value is grounded in the user message.
        reason: Human-readable explanation for tracing/debugging.
        strategy: Which grounding strategy was applied (e.g.
            ``"enum"``, ``"string_overlap"``, ``"number"``,
            ``"boolean"``, ``"array"``, ``"skip"``, ``"fuzzy"``,
            ``"exact"``).
    """

    field: str
    grounded: bool
    reason: str
    strategy: str


# ──────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────

def _word_in_text(word: str, text: str) -> bool:
    r"""Check if *word* appears as a whole word in *text*.

    Uses ``\b`` word-boundary anchors to avoid substring
    false positives (e.g. ``"base"`` matching ``"database"``).
    """
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


def significant_words(
    text: str,
    *,
    stopwords: frozenset[str] | None = None,
) -> set[str]:
    """Extract meaningful words from text, ignoring stopwords.

    Strips leading/trailing punctuation from each word before
    filtering, so ``"science,"`` becomes ``"science"``.

    Args:
        text: Input text (description, field name, etc.)
        stopwords: Custom stopword set.  Defaults to
            :data:`DEFAULT_STOPWORDS`.

    Returns:
        Set of lowercase words longer than 2 characters,
        excluding stopwords.
    """
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    return {
        w
        for raw in text.split()
        if len(w := raw.lower().strip(".,;:!?\"'()[]{}")) > 2
        and w not in sw
    }


def field_keywords(
    field_name: str,
    schema_property: dict[str, Any],
    *,
    stopwords: frozenset[str] | None = None,
) -> set[str]:
    """Derive grounding keywords from field name and schema description.

    Extracts significant words from the field's ``description`` (or the
    field name itself when no description is set) and the field name
    split on underscores.

    Args:
        field_name: Field name (e.g., ``"save_confirmed"``).
        schema_property: JSON Schema property dict for the field.
        stopwords: Custom stopword set.  Defaults to
            :data:`DEFAULT_STOPWORDS`.

    Returns:
        Set of lowercase keywords longer than 2 characters, excluding
        stopwords.  May be empty for fields with very short names and
        no description.
    """
    desc_words = significant_words(
        schema_property.get("description", field_name),
        stopwords=stopwords,
    )
    desc_words |= significant_words(
        field_name.replace("_", " "),
        stopwords=stopwords,
    )
    return desc_words


def has_negation(
    msg_lower: str,
    negation_keywords: frozenset[str],
    *,
    field_keywords: set[str] | None = None,
    proximity: int = 0,
) -> bool:
    """Check if the message contains a negation signal.

    Args:
        msg_lower: Lowercased user message.
        negation_keywords: Set of negation keywords to look for.
        field_keywords: If provided with ``proximity > 0``, require
            a negation keyword within *proximity* words of a field
            keyword.
        proximity: Maximum word distance between a negation keyword
            and a field keyword.  ``0`` (default) means any position
            in the message counts — no proximity requirement.

    Returns:
        True if a negation signal is detected.
    """
    if proximity <= 0 or field_keywords is None:
        return any(_word_in_text(w, msg_lower) for w in negation_keywords)

    # Proximity check: split message into words, find positions.
    words = [
        raw.lower().strip(".,;:!?\"'()[]{}")
        for raw in msg_lower.split()
    ]
    neg_positions: list[int] = []
    field_positions: list[int] = []
    for i, w in enumerate(words):
        if w in negation_keywords:
            neg_positions.append(i)
        if w in field_keywords:
            field_positions.append(i)

    if not neg_positions or not field_positions:
        return False

    return any(
        abs(np - fp) <= proximity
        for np in neg_positions
        for fp in field_positions
    )


def detect_boolean_signal(
    msg_lower: str,
    *,
    affirmative_signals: frozenset[str],
    affirmative_phrases: tuple[str, ...],
    negative_signals: frozenset[str],
    negative_phrases: tuple[str, ...],
    negation_keywords: frozenset[str] | None = None,
) -> bool | None:
    """Detect affirmative or negative boolean signal in a message.

    Scans the message for signal words and phrases, returning
    ``True`` for affirmative, ``False`` for negative, or ``None``
    when the signal is ambiguous or absent.

    All matching — both single words and multi-word phrases — uses
    word-boundary checks to prevent substring false positives.

    When only affirmative signals are found, the function checks for
    negation keywords (excluding words already in ``negative_signals``
    to avoid double-counting) to catch negated affirmatives like
    ``"I will skip confirming"``.

    Args:
        msg_lower: Lowercased user message.
        affirmative_signals: Single-word affirmative keywords.
        affirmative_phrases: Multi-word affirmative phrases.
        negative_signals: Single-word negative keywords.
        negative_phrases: Multi-word negative phrases.
        negation_keywords: Negation keyword set for detecting negated
            affirmatives.  Defaults to :data:`DEFAULT_NEGATION_KEYWORDS`.

    Returns:
        ``True`` if affirmative signal detected, ``False`` if negative
        signal detected, ``None`` if ambiguous or no signal.
    """
    neg_kw = (
        negation_keywords
        if negation_keywords is not None
        else DEFAULT_NEGATION_KEYWORDS
    )

    # Check for phrase matches (multi-word, word-boundary aware)
    has_affirmative_phrase = any(
        _word_in_text(p, msg_lower) for p in affirmative_phrases
    )
    has_negative_phrase = any(
        _word_in_text(p, msg_lower) for p in negative_phrases
    )

    # Check for single-word signal matches (word-boundary aware)
    has_affirmative_word = any(
        _word_in_text(w, msg_lower) for w in affirmative_signals
    )
    has_negative_word = any(
        _word_in_text(w, msg_lower) for w in negative_signals
    )

    has_aff = has_affirmative_phrase or has_affirmative_word
    has_neg = has_negative_phrase or has_negative_word

    if not has_aff and not has_neg:
        return None

    if has_neg and not has_aff:
        return False

    if has_aff and not has_neg:
        # Check for negation that could flip the affirmative.
        check_neg = neg_kw - negative_signals
        if check_neg and has_negation(msg_lower, check_neg):
            return False
        return True

    # Both affirmative and negative signals present — ambiguous.
    # Phrases carry stronger intent than single words.
    if has_negative_phrase and not has_affirmative_phrase:
        return False
    if has_affirmative_phrase and not has_negative_phrase:
        return True

    return None


# ──────────────────────────────────────────────────────────────────
# Core grounding API
# ──────────────────────────────────────────────────────────────────

def is_field_grounded(
    field_name: str,
    value: Any,
    user_message: str,
    schema_property: dict[str, Any],
    *,
    config: GroundingConfig | None = None,
) -> FieldGroundingResult:
    """Check whether a single extracted value is grounded in the user message.

    Dispatches to type-specific checks based on
    ``schema_property["type"]``.  Per-field overrides via
    ``x-extraction`` annotations on the schema property.

    Args:
        field_name: Schema property name.
        value: Extracted value to check.
        user_message: The user's raw message text.
        schema_property: JSON Schema property dict for this field.
        config: Grounding configuration.  Defaults to
            :class:`GroundingConfig` with standard defaults.

    Returns:
        A :class:`FieldGroundingResult` with the verdict, reason, and
        strategy used.
    """
    cfg = config or GroundingConfig()
    x_ext = schema_property.get("x-extraction", {})
    msg_lower = user_message.lower()

    # Explicit grounding mode overrides type-based dispatch
    grounding_mode = x_ext.get("grounding")

    if grounding_mode == "skip":
        return FieldGroundingResult(
            field=field_name, grounded=True,
            reason="grounding=skip", strategy="skip",
        )

    if grounding_mode == "exact":
        grounded = _word_in_text(str(value).lower(), msg_lower)
        return FieldGroundingResult(
            field=field_name, grounded=grounded,
            reason=f"exact match: {value!r} {'found' if grounded else 'not found'}",
            strategy="exact",
        )

    if grounding_mode == "fuzzy":
        return FieldGroundingResult(
            field=field_name, grounded=True,
            reason="grounding=fuzzy", strategy="fuzzy",
        )

    # Resolve overlap threshold: per-field > config
    overlap_threshold = x_ext.get("overlap_threshold", cfg.overlap_threshold)

    # Type-based dispatch
    field_type = schema_property.get("type", "string")

    if field_type == "boolean":
        grounded = _ground_boolean(
            field_name, value, schema_property, msg_lower, cfg,
        )
        return FieldGroundingResult(
            field=field_name, grounded=grounded,
            reason=f"boolean keywords {'found' if grounded else 'not found'}",
            strategy="boolean",
        )

    if field_type in ("integer", "number"):
        grounded = _ground_number(value, user_message)
        return FieldGroundingResult(
            field=field_name, grounded=grounded,
            reason=f"number {value!r} {'found' if grounded else 'not found'}",
            strategy="number",
        )

    if field_type == "array":
        grounded = _ground_array(
            field_name, value, msg_lower, schema_property, cfg,
        )
        return FieldGroundingResult(
            field=field_name, grounded=grounded,
            reason=f"array {'element found' if grounded else 'no elements found'}",
            strategy="array",
        )

    # String (with or without enum)
    if isinstance(value, str):
        empty_allowed = x_ext.get("empty_allowed", False)
        grounded = _ground_string(
            field_name, value, msg_lower, schema_property,
            empty_allowed=empty_allowed,
            overlap_threshold=overlap_threshold,
            config=cfg,
        )
        strategy = "enum" if "enum" in schema_property and value else "string_overlap"
        return FieldGroundingResult(
            field=field_name, grounded=grounded,
            reason=f"string {value!r} {'grounded' if grounded else 'not grounded'}",
            strategy=strategy,
        )

    # Unknown types: trust extraction
    return FieldGroundingResult(
        field=field_name, grounded=True,
        reason=f"unknown type {field_type!r}: trusted",
        strategy="unknown",
    )


def ground_extraction(
    extracted: dict[str, Any],
    user_message: str,
    schema: dict[str, Any],
    *,
    config: GroundingConfig | None = None,
) -> dict[str, FieldGroundingResult]:
    """Check all extracted fields against the user message.

    Iterates schema ``properties``, calls :func:`is_field_grounded`
    for each field present in *extracted*.  Skips ``None`` values
    (mirror wizard behavior — None filtering happens before
    grounding).

    Args:
        extracted: Dict of extracted field names → values.
        user_message: The user's raw message text.
        schema: JSON Schema dict with ``properties`` key.
        config: Grounding configuration.

    Returns:
        Dict of field name → :class:`FieldGroundingResult` for each
        field that was checked.  Fields not in the schema or with
        ``None`` values are omitted.
    """
    cfg = config or GroundingConfig()
    properties = schema.get("properties", {})
    results: dict[str, FieldGroundingResult] = {}

    for fname, fvalue in extracted.items():
        if fvalue is None:
            continue
        schema_prop = properties.get(fname)
        if schema_prop is None:
            continue
        results[fname] = is_field_grounded(
            fname, fvalue, user_message, schema_prop, config=cfg,
        )

    return results


# ──────────────────────────────────────────────────────────────────
# Private type-dispatched grounding checks
# ──────────────────────────────────────────────────────────────────

def _ground_boolean(
    field_name: str,
    value: Any,
    schema_property: dict[str, Any],
    msg_lower: str,
    config: GroundingConfig,
) -> bool:
    """Grounded if field-related keywords appear in message."""
    keywords = field_keywords(
        field_name, schema_property, stopwords=config.stopwords,
    )
    field_mentioned = any(_word_in_text(w, msg_lower) for w in keywords)
    if not field_mentioned:
        return False

    x_ext = schema_property.get("x-extraction", {})
    check_direction = x_ext.get("check_direction", True)
    if not check_direction:
        return True

    custom_neg = x_ext.get("negation_keywords")
    neg_keywords = (
        frozenset(custom_neg) if custom_neg is not None
        else config.negation_keywords
    )
    proximity = x_ext.get("negation_proximity", 0)

    has_neg = has_negation(
        msg_lower, neg_keywords,
        field_keywords=keywords,
        proximity=proximity,
    )

    if value is False:
        return has_neg
    return not has_neg


def _ground_number(value: Any, user_message: str) -> bool:
    """Grounded if the number appears as a whole word in the message."""
    pattern = r"\b" + re.escape(str(value)) + r"\b"
    return re.search(pattern, user_message) is not None


def _ground_array(
    field_name: str,
    value: Any,
    msg_lower: str,
    schema_property: dict[str, Any],
    config: GroundingConfig,
) -> bool:
    """Grounded if at least one element appears in the message."""
    if not value:
        x_ext = schema_property.get("x-extraction", {})
        if x_ext.get("empty_allowed", False):
            return True
        keywords = field_keywords(
            field_name, schema_property, stopwords=config.stopwords,
        )
        custom_neg = x_ext.get("negation_keywords")
        neg_keywords = (
            frozenset(custom_neg) if custom_neg is not None
            else config.negation_keywords
        )
        proximity = x_ext.get("negation_proximity", 0)
        return (
            any(_word_in_text(w, msg_lower) for w in keywords)
            and has_negation(
                msg_lower, neg_keywords,
                field_keywords=keywords,
                proximity=proximity,
            )
        )
    return any(
        _word_in_text(str(item).lower(), msg_lower)
        for item in value
    )


def _ground_string(
    field_name: str,
    value: str,
    msg_lower: str,
    schema_property: dict[str, Any],
    *,
    empty_allowed: bool = False,
    overlap_threshold: float = 0.5,
    config: GroundingConfig,
) -> bool:
    """Ground a string value against the user's message."""
    if not value:
        if empty_allowed:
            return True
        x_ext = schema_property.get("x-extraction", {})
        keywords = field_keywords(
            field_name, schema_property, stopwords=config.stopwords,
        )
        custom_neg = x_ext.get("negation_keywords")
        neg_keywords = (
            frozenset(custom_neg) if custom_neg is not None
            else config.negation_keywords
        )
        proximity = x_ext.get("negation_proximity", 0)
        return (
            any(_word_in_text(w, msg_lower) for w in keywords)
            and has_negation(
                msg_lower, neg_keywords,
                field_keywords=keywords,
                proximity=proximity,
            )
        )

    # Enum: check if value matches an enum entry found in message
    if "enum" in schema_property:
        return _word_in_text(value.lower(), msg_lower)

    # General string: word overlap check
    value_words = significant_words(value, stopwords=config.stopwords)
    if not value_words:
        return True  # Value is all stopwords — trust extraction
    msg_words = significant_words(msg_lower, stopwords=config.stopwords)
    overlap = value_words & msg_words
    return len(overlap) / len(value_words) >= overlap_threshold
