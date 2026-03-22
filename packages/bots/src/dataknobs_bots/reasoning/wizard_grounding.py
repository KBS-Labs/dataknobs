"""Schema-driven extraction grounding and merge filtering.

Verifies that extracted values are grounded in the user's actual
message before allowing them to overwrite existing wizard state data.
This prevents extraction models from hallucinating values for fields
the user didn't address.

Grounding Architecture::

    Layer 1 (built-in):  Skip None values (always active)
    Layer 2 (this module): Schema-driven grounding check
    Layer 3 (opt-in):     Per-field x-extraction config hints
    Layer 4 (escape):     Pluggable custom merge filter

Merge Filter Composition::

    Built-in grounding → Custom filter(s) → Merge into wizard_data

    When both grounding and a custom filter are configured, they
    compose via :class:`CompositeMergeFilter`.  Grounding runs first
    to reject ungrounded values; the custom filter sees only values
    that passed grounding.  Set ``skip_builtin_grounding: true`` to
    bypass grounding and run only the custom filter.

"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Literal, Protocol, runtime_checkable

from dataknobs_bots.reasoning.wizard_utils import word_in_text

logger = logging.getLogger(__name__)

# Words too common to carry grounding signal
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "of", "for", "to", "or", "and", "whether",
    "it", "in", "on", "at", "by", "with", "this", "that", "be", "as",
    "can", "do", "does", "did", "has", "have", "had", "was", "were",
    "are", "been", "being", "will", "would", "could", "should", "may",
    "might", "shall", "must", "not", "but", "if", "then", "than",
    "so", "just", "also", "very", "too", "really", "quite",
})

# Keywords indicating intent to clear/remove/negate a value.
# Used by boolean direction checking, empty-string grounding,
# and empty-array grounding.  Overridable per-field via
# ``x-extraction.negation_keywords``.
_NEGATION_KEYWORDS = frozenset({
    "no", "skip", "none", "remove", "clear", "without", "blank",
    "empty", "delete", "disable", "disabled", "off",
})


def significant_words(text: str) -> set[str]:
    """Extract meaningful words from text, ignoring stopwords.

    Strips leading/trailing punctuation from each word before
    filtering, so ``"science,"`` becomes ``"science"``.

    Args:
        text: Input text (description, field name, etc.)

    Returns:
        Set of lowercase words longer than 2 characters,
        excluding common stopwords.
    """
    return {
        w
        for raw in text.split()
        if len(w := raw.lower().strip(".,;:!?\"'()[]{}")) > 2
        and w not in _STOPWORDS
    }



def field_keywords(
    field: str,
    schema_property: dict[str, Any],
) -> set[str]:
    """Derive grounding keywords from field name and schema description."""
    desc_words = significant_words(
        schema_property.get("description", field)
    )
    desc_words |= significant_words(field.replace("_", " "))
    return desc_words


def _has_negation(
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
            in the message counts --- no proximity requirement.

    Returns:
        True if a negation signal is detected.
    """
    if proximity <= 0 or field_keywords is None:
        # No proximity requirement --- just check presence.
        return any(word_in_text(w, msg_lower) for w in negation_keywords)

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

    # Check if any negation keyword is within proximity of any field keyword.
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
) -> bool | None:
    """Detect affirmative or negative boolean signal in a message.

    Scans the message for signal words and phrases, returning
    ``True`` for affirmative, ``False`` for negative, or ``None``
    when the signal is ambiguous or absent.

    All matching — both single words and multi-word phrases — uses
    word-boundary checks to prevent substring false positives
    (e.g., ``"go back"`` should not match inside ``"go background"``).

    When only affirmative signals are found, the function checks for
    negation keywords (from ``_NEGATION_KEYWORDS``, excluding words
    already in ``negative_signals`` to avoid double-counting) to
    catch negated affirmatives like ``"no, I don't confirm"``.

    When both affirmative and negative signals are present, the
    result is ``None`` (ambiguous) — the function does not guess.

    Args:
        msg_lower: Lowercased user message.
        affirmative_signals: Single-word affirmative keywords.
        affirmative_phrases: Multi-word affirmative phrases.
        negative_signals: Single-word negative keywords.
        negative_phrases: Multi-word negative phrases.

    Returns:
        ``True`` if affirmative signal detected, ``False`` if negative
        signal detected, ``None`` if ambiguous or no signal.
    """
    # Check for phrase matches (multi-word, word-boundary aware)
    has_affirmative_phrase = any(
        word_in_text(p, msg_lower) for p in affirmative_phrases
    )
    has_negative_phrase = any(
        word_in_text(p, msg_lower) for p in negative_phrases
    )

    # Check for single-word signal matches (word-boundary aware)
    has_affirmative_word = any(
        word_in_text(w, msg_lower) for w in affirmative_signals
    )
    has_negative_word = any(
        word_in_text(w, msg_lower) for w in negative_signals
    )

    has_affirmative = has_affirmative_phrase or has_affirmative_word
    has_negative = has_negative_phrase or has_negative_word

    if not has_affirmative and not has_negative:
        return None  # No signals at all

    if has_negative and not has_affirmative:
        return False

    if has_affirmative and not has_negative:
        # Check for negation that could flip the affirmative.
        # Exclude words already in negative_signals to avoid
        # double-counting (e.g., "no" is both a negative signal
        # and a negation keyword — if it matched as negative it
        # would have been caught above, so here we only want
        # negation modifiers like "don't", "not", etc.).
        neg_kw = _NEGATION_KEYWORDS - negative_signals
        if neg_kw and _has_negation(msg_lower, neg_kw):
            return False
        return True

    # Both affirmative and negative signals present — ambiguous.
    # Phrases carry stronger intent than single words.
    if has_negative_phrase and not has_affirmative_phrase:
        return False
    if has_affirmative_phrase and not has_negative_phrase:
        return True

    return None  # Truly ambiguous


@dataclasses.dataclass(frozen=True)
class MergeDecision:
    """Result of a merge filter decision.

    Use the class methods :meth:`accept`, :meth:`reject`, and
    :meth:`transform` for ergonomic construction.

    Attributes:
        action: One of ``"accept"``, ``"reject"``, or ``"transform"``.
        value: The transformed value (used only when action is
            ``"transform"``).
        reason: Optional human-readable explanation for tracing and
            debugging.
    """

    action: Literal["accept", "reject", "transform"]
    value: Any = None
    reason: str | None = None

    @classmethod
    def accept(cls, *, reason: str | None = None) -> MergeDecision:
        """Accept the value as-is."""
        return cls(action="accept", reason=reason)

    @classmethod
    def reject(cls, *, reason: str | None = None) -> MergeDecision:
        """Reject the value — it will not be merged."""
        return cls(action="reject", reason=reason)

    @classmethod
    def transform(
        cls, value: Any, *, reason: str | None = None,
    ) -> MergeDecision:
        """Accept but substitute a different value."""
        return cls(action="transform", value=value, reason=reason)


@runtime_checkable
class MergeFilter(Protocol):
    """Protocol for filtering extraction results before merge.

    Implementations decide, per field, whether an extracted value
    should be merged into wizard state.  The built-in
    :class:`SchemaGroundingFilter` uses the JSON Schema definition
    and the user's message to make this decision.  Custom filters
    can compose with grounding via :class:`CompositeMergeFilter`.
    """

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        """Decide whether to accept, reject, or transform a value.

        Args:
            field: Schema property name.
            new_value: Value from extraction (never None --- None
                is filtered before this is called).
            existing_value: Current value in wizard_state.data
                (None if absent).
            user_message: Raw user message text.
            schema_property: JSON Schema property definition for
                this field.
            wizard_data: Read-only snapshot of the current wizard
                state data dict.  Do not mutate.

        Returns:
            A :class:`MergeDecision` indicating what to do with the
            value.
        """
        ...


class SchemaGroundingFilter:
    """Built-in merge filter using schema-driven grounding.

    Verifies each extracted value against the user's message using
    type-appropriate heuristics derived from the JSON Schema
    definition.

    Grounding rules by type:

    - **string**: word overlap between value and message
    - **string + enum**: enum value appears in message (word-boundary)
    - **boolean**: field-related keywords found in message, with
      optional value-direction checking via negation detection
    - **integer/number**: literal number appears in message
      (word-boundary)
    - **array**: at least one element appears in message
      (word-boundary); empty arrays grounded via negation keywords

    The ``x-extraction`` JSON Schema extension allows per-field
    overrides:

    - ``grounding``: ``"exact"`` | ``"fuzzy"`` | ``"skip"``
      (override type-based strategy)
    - ``empty_allowed``: ``true`` | ``false``
      (allow empty string/array to overwrite)
    - ``overlap_threshold``: ``float``
      (per-field word overlap ratio)
    - ``check_direction``: ``true`` | ``false``
      (boolean fields: also verify value direction via negation
      detection; default ``true``)
    - ``negation_keywords``: ``list[str]``
      (override default negation keyword set for this field)
    - ``negation_proximity``: ``int``
      (max word distance between negation and field keyword;
      ``0`` means no proximity requirement; default ``0``)

    Args:
        overlap_threshold: Default minimum word-overlap ratio for
            string grounding (0.0--1.0).  Defaults to 0.5.
    """

    def __init__(self, overlap_threshold: float = 0.5) -> None:
        self._overlap_threshold = overlap_threshold

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        """Decide whether to merge based on grounding in user message."""
        # Layer 3: per-field x-extraction hints override Layer 2
        x_ext = schema_property.get("x-extraction", {})

        grounding_mode = x_ext.get("grounding")
        if grounding_mode == "skip":
            return MergeDecision.accept(reason="grounding=skip")

        # Core decision: grounded values always merge; ungrounded
        # values merge only when there's no existing data to protect.
        grounded = self._is_grounded(
            field, new_value, user_message, schema_property,
            grounding_mode=grounding_mode,
            overlap_threshold=x_ext.get(
                "overlap_threshold", self._overlap_threshold,
            ),
        )

        if grounded:
            return MergeDecision.accept(reason="grounded")

        if existing_value is None:
            # No existing data to protect --- benefit of the doubt.
            return MergeDecision.accept(
                reason="ungrounded but no existing value",
            )

        return MergeDecision.reject(
            reason=(
                f"ungrounded: extracted {new_value!r} not in message, "
                f"existing {existing_value!r} preserved"
            ),
        )

    # ------------------------------------------------------------------
    # Type-dispatched grounding checks
    # ------------------------------------------------------------------

    def _is_grounded(
        self,
        field: str,
        value: Any,
        user_message: str,
        schema_property: dict[str, Any],
        *,
        grounding_mode: str | None = None,
        overlap_threshold: float,
    ) -> bool:
        """Check if an extracted value is grounded in the user's message."""
        msg_lower = user_message.lower()
        field_type = schema_property.get("type", "string")

        # Explicit grounding mode overrides type-based dispatch
        if grounding_mode == "exact":
            return word_in_text(str(value).lower(), msg_lower)

        if grounding_mode == "fuzzy":
            return True

        # Type-based dispatch
        if field_type == "boolean":
            return self._ground_boolean(
                field, value, schema_property, msg_lower,
            )

        if field_type in ("integer", "number"):
            return self._ground_number(value, user_message)

        if field_type == "array":
            return self._ground_array(
                field, value, msg_lower, schema_property,
            )

        # String (with or without enum)
        if isinstance(value, str):
            x_ext = schema_property.get("x-extraction", {})
            empty_allowed = x_ext.get("empty_allowed", False)
            return self._ground_string(
                field, value, msg_lower, schema_property,
                empty_allowed=empty_allowed,
                overlap_threshold=overlap_threshold,
            )

        return True  # Unknown types: trust extraction

    def _ground_boolean(
        self,
        field: str,
        value: Any,
        schema_property: dict[str, Any],
        msg_lower: str,
    ) -> bool:
        """Grounded if field-related keywords appear in message.

        When ``check_direction`` is enabled (the default), the
        extracted boolean value is also verified against negation
        signals in the message:

        - ``False`` requires a negation keyword near the field keyword
        - ``True`` requires the *absence* of negation near the field
          keyword

        This prevents extraction models that hallucinate the wrong
        boolean direction from overwriting correct existing values.

        Configurable via ``x-extraction``:

        - ``check_direction``: enable/disable direction checking
          (default ``true``)
        - ``negation_keywords``: override the default negation set
        - ``negation_proximity``: max word distance between negation
          and field keyword (``0`` = anywhere in message)
        """
        keywords = field_keywords(field, schema_property)
        field_mentioned = any(word_in_text(w, msg_lower) for w in keywords)
        if not field_mentioned:
            return False

        # Check if direction-checking is enabled (default: true)
        x_ext = schema_property.get("x-extraction", {})
        check_direction = x_ext.get("check_direction", True)
        if not check_direction:
            return True  # Field mentioned is sufficient

        # Resolve negation parameters
        custom_neg = x_ext.get("negation_keywords")
        neg_keywords = (
            frozenset(custom_neg) if custom_neg is not None
            else _NEGATION_KEYWORDS
        )
        proximity = x_ext.get("negation_proximity", 0)

        has_neg = _has_negation(
            msg_lower, neg_keywords,
            field_keywords=keywords,
            proximity=proximity,
        )

        # Direction: False requires negation, True requires no negation
        if value is False:
            return has_neg
        return not has_neg

    def _ground_number(self, value: Any, user_message: str) -> bool:
        """Grounded if the number appears as a whole word in the message.

        Uses word-boundary matching to avoid false positives like
        ``5`` matching ``"15"`` or ``"50"``.
        """
        pattern = r"\b" + re.escape(str(value)) + r"\b"
        return re.search(pattern, user_message) is not None

    def _ground_array(
        self,
        field: str,
        value: Any,
        msg_lower: str,
        schema_property: dict[str, Any],
    ) -> bool:
        """Grounded if at least one element appears in the message.

        Empty arrays are grounded when the user expresses clearing
        intent via negation keywords (e.g. "no tools", "remove all
        tools"), mirroring the empty-string grounding logic.

        Configurable via ``x-extraction``:

        - ``empty_allowed``: skip negation check for empty arrays
        - ``negation_keywords``: override negation keyword set
        - ``negation_proximity``: max word distance (``0`` = anywhere)
        """
        if not value:
            x_ext = schema_property.get("x-extraction", {})
            if x_ext.get("empty_allowed", False):
                return True
            keywords = field_keywords(field, schema_property)
            custom_neg = x_ext.get("negation_keywords")
            neg_keywords = (
                frozenset(custom_neg) if custom_neg is not None
                else _NEGATION_KEYWORDS
            )
            proximity = x_ext.get("negation_proximity", 0)
            return (
                any(word_in_text(w, msg_lower) for w in keywords)
                and _has_negation(
                    msg_lower, neg_keywords,
                    field_keywords=keywords,
                    proximity=proximity,
                )
            )
        return any(
            word_in_text(str(item).lower(), msg_lower)
            for item in value
        )

    def _ground_string(
        self,
        field: str,
        value: str,
        msg_lower: str,
        schema_property: dict[str, Any],
        *,
        empty_allowed: bool = False,
        overlap_threshold: float = 0.5,
    ) -> bool:
        """Ground a string value against the user's message."""
        if not value:
            # Empty string: grounded only if field concept + negation
            if empty_allowed:
                return True
            x_ext = schema_property.get("x-extraction", {})
            keywords = field_keywords(field, schema_property)
            custom_neg = x_ext.get("negation_keywords")
            neg_keywords = (
                frozenset(custom_neg) if custom_neg is not None
                else _NEGATION_KEYWORDS
            )
            proximity = x_ext.get("negation_proximity", 0)
            return (
                any(word_in_text(w, msg_lower) for w in keywords)
                and _has_negation(
                    msg_lower, neg_keywords,
                    field_keywords=keywords,
                    proximity=proximity,
                )
            )

        # Enum: check if value matches an enum entry found in message
        if "enum" in schema_property:
            return word_in_text(value.lower(), msg_lower)

        # General string: word overlap check
        value_words = significant_words(value)
        if not value_words:
            return True  # Value is all stopwords --- trust extraction
        msg_words = significant_words(msg_lower)
        overlap = value_words & msg_words
        return len(overlap) / len(value_words) >= overlap_threshold


class CompositeMergeFilter:
    """Chain multiple merge filters in sequence.

    Short-circuits on the first ``reject``.  When a filter returns
    ``transform``, the transformed value is passed as ``new_value``
    to subsequent filters.

    Satisfies the :class:`MergeFilter` protocol via duck typing.
    """

    def __init__(self, filters: list[MergeFilter]) -> None:
        self._filters = filters

    def filter(
        self,
        field: str,
        new_value: Any,
        existing_value: Any | None,
        user_message: str,
        schema_property: dict[str, Any],
        wizard_data: dict[str, Any],
    ) -> MergeDecision:
        """Run each filter in order, short-circuiting on reject."""
        current_value = new_value
        transformed = False
        last_reason: str | None = None
        for f in self._filters:
            decision = f.filter(
                field, current_value, existing_value,
                user_message, schema_property, wizard_data,
            )
            if decision.action == "reject":
                return decision
            if decision.action == "transform":
                current_value = decision.value
                last_reason = decision.reason
                transformed = True
        if transformed:
            return MergeDecision.transform(
                current_value, reason=last_reason,
            )
        return MergeDecision.accept(reason=last_reason)
