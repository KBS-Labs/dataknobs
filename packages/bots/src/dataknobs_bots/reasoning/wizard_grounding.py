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

"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

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

# Keywords indicating intent to clear/remove a value
_NEGATION_KEYWORDS = frozenset({
    "no", "skip", "none", "remove", "clear", "without", "blank",
    "empty", "delete",
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


@runtime_checkable
class MergeFilter(Protocol):
    """Protocol for filtering extraction results before merge.

    Implementations decide, per field, whether an extracted value
    should be merged into wizard state.  The built-in
    :class:`SchemaGroundingFilter` uses the JSON Schema definition
    and the user's message to make this decision.  Custom filters
    can replace it entirely for domain-specific logic.
    """

    def should_merge(
        self,
        field: str,
        new_value: Any,
        existing_value: Any,
        user_message: str,
        schema_property: dict[str, Any],
    ) -> bool:
        """Decide whether to merge an extracted value into wizard state.

        Args:
            field: Schema property name.
            new_value: Value from extraction (never None --- None
                is filtered before this is called).
            existing_value: Current value in wizard_state.data
                (None if absent).
            user_message: Raw user message text.
            schema_property: JSON Schema property definition for
                this field.

        Returns:
            True to merge, False to skip.
        """
        ...


class SchemaGroundingFilter:
    """Built-in merge filter using schema-driven grounding.

    Verifies each extracted value against the user's message using
    type-appropriate heuristics derived from the JSON Schema
    definition.

    Grounding rules by type:

    - **string**: word overlap between value and message
    - **string + enum**: enum value appears in message
    - **boolean**: field-related keywords found in message
    - **integer/number**: literal number appears in message
    - **array**: at least one element appears in message

    The ``x-extraction`` JSON Schema extension allows per-field
    overrides:

    - ``grounding``: ``"exact"`` | ``"fuzzy"`` | ``"skip"``
      (override type-based strategy)
    - ``empty_allowed``: ``true`` | ``false``
      (allow empty string to overwrite)
    - ``overlap_threshold``: ``float``
      (per-field word overlap ratio)

    Args:
        overlap_threshold: Default minimum word-overlap ratio for
            string grounding (0.0--1.0).  Defaults to 0.5.
    """

    def __init__(self, overlap_threshold: float = 0.5) -> None:
        self._overlap_threshold = overlap_threshold

    def should_merge(
        self,
        field: str,
        new_value: Any,
        existing_value: Any,
        user_message: str,
        schema_property: dict[str, Any],
    ) -> bool:
        """Decide whether to merge based on grounding in user message."""
        # Layer 3: per-field x-extraction hints override Layer 2
        x_ext = schema_property.get("x-extraction", {})

        grounding_mode = x_ext.get("grounding")
        if grounding_mode == "skip":
            return True  # Author says: always merge this field

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
            return True

        if existing_value is None:
            # No existing data to protect --- benefit of the doubt.
            return True

        logger.debug(
            "Grounding check blocked overwrite of '%s': "
            "extracted %r not grounded in message, existing %r preserved",
            field, new_value, existing_value,
        )
        return False

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
            return str(value).lower() in msg_lower

        if grounding_mode == "fuzzy":
            return True

        # Type-based dispatch
        if field_type == "boolean":
            return self._ground_boolean(field, schema_property, msg_lower)

        if field_type in ("integer", "number"):
            return self._ground_number(value, user_message)

        if field_type == "array":
            return self._ground_array(value, msg_lower)

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
        schema_property: dict[str, Any],
        msg_lower: str,
    ) -> bool:
        """Grounded if field-related keywords appear in message."""
        desc_words = significant_words(
            schema_property.get("description", field)
        )
        # Also include the field name itself as keywords
        desc_words |= significant_words(field.replace("_", " "))
        return any(w in msg_lower for w in desc_words)

    def _ground_number(self, value: Any, user_message: str) -> bool:
        """Grounded if the number appears as a whole word in the message.

        Uses word-boundary matching to avoid false positives like
        ``5`` matching ``"15"`` or ``"50"``.
        """
        pattern = r"\b" + re.escape(str(value)) + r"\b"
        return re.search(pattern, user_message) is not None

    def _ground_array(self, value: Any, msg_lower: str) -> bool:
        """Grounded if at least one element appears in the message."""
        if not value:
            return False
        return any(str(item).lower() in msg_lower for item in value)

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
            desc_words = significant_words(
                schema_property.get("description", field)
            )
            desc_words |= significant_words(field.replace("_", " "))
            return (
                any(w in msg_lower for w in desc_words)
                and any(w in msg_lower for w in _NEGATION_KEYWORDS)
            )

        # Enum: check if value matches an enum entry found in message
        if "enum" in schema_property:
            return value.lower() in msg_lower

        # General string: word overlap check
        value_words = significant_words(value)
        if not value_words:
            return True  # Value is all stopwords --- trust extraction
        msg_words = significant_words(msg_lower)
        overlap = value_words & msg_words
        return len(overlap) / len(value_words) >= overlap_threshold
