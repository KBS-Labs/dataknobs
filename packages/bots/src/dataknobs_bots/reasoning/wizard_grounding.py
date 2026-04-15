"""Schema-driven extraction grounding and merge filtering.

Verifies that extracted values are grounded in the user's actual
message before allowing them to overwrite existing wizard state data.
This prevents extraction models from hallucinating values for fields
the user didn't address.

Core grounding logic (type-dispatched checks, text utilities, negation
detection) lives in :mod:`dataknobs_llm.extraction.grounding` as a
standalone utility.  This module provides the **wizard-specific merge
policy** that wraps those checks: grounded values merge; ungrounded
values merge only when there is no existing data to protect
(configurable per-field via ``require_grounded``).

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
from typing import Any, Literal, Protocol, runtime_checkable

from dataknobs_utils.value_expansion import (
    ValueExpansionConfig,
    expand_value_in_message,
)

from dataknobs_llm.extraction.grounding import (
    DEFAULT_NEGATION_KEYWORDS,
    DEFAULT_STOPWORDS,
    GroundingConfig,
    detect_boolean_signal,
    field_keywords,
    has_negation,
    is_field_grounded,
    significant_words,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases for constants and functions that moved
# to dataknobs_llm.extraction.grounding.  Test files import these
# with underscore prefix.
_STOPWORDS = DEFAULT_STOPWORDS
_NEGATION_KEYWORDS = DEFAULT_NEGATION_KEYWORDS
_has_negation = has_negation

# Re-export public names so existing consumers (wizard.py, test files)
# that import from this module continue to work unchanged.
__all__ = [
    "CompositeMergeFilter",
    "MergeDecision",
    "MergeFilter",
    "SchemaGroundingFilter",
    "ValueExpansionConfig",
    # Re-exported from dataknobs_llm.extraction.grounding
    "detect_boolean_signal",
    "field_keywords",
    "significant_words",
]


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

    Delegates type-dispatched grounding checks to the shared utility
    in :mod:`dataknobs_llm.extraction.grounding`, then applies
    wizard-specific merge policy:

    - Grounded values always merge.
    - Ungrounded values merge when there is no existing data to
      protect (benefit of the doubt for first-turn fields).
    - Ungrounded values are rejected when existing data would be
      overwritten.

    The ``x-extraction`` JSON Schema extension allows per-field
    overrides:

    - ``grounding``: ``"exact"`` | ``"fuzzy"`` | ``"skip"``
      (override type-based strategy)
    - ``empty_allowed``: ``true`` | ``false``
      (allow empty string/array to overwrite)
    - ``overlap_threshold``: ``float``
      (per-field word overlap ratio)
    - ``require_grounded``: ``true`` | ``false``
      (reject ungrounded values even when no existing value;
      default ``false`` preserves benefit-of-the-doubt for first write)
    - ``check_direction``: ``true`` | ``false``
      (boolean fields: also verify value direction via negation
      detection; default ``true``)
    - ``negation_keywords``: ``list[str]``
      (override default negation keyword set for this field)
    - ``negation_proximity``: ``int``
      (max word distance between negation and field keyword;
      ``0`` means no proximity requirement; default ``0``)
    - ``expand_from_message``: ``true`` | ``false``
      (enable value expansion for this string field;
      default ``false`` — when enabled, partial extractions
      are expanded across conjunctions to recover the full
      compound phrase from the user's message; best for
      descriptive fields like tone, style, or mood where
      compound values are common)

    Note: ``grounding: "skip"`` bypasses expansion as well as
    grounding — the value is accepted as-is without any processing.

    Args:
        overlap_threshold: Default minimum word-overlap ratio for
            string grounding (0.0-1.0).  Defaults to 0.5.
        expansion_config: Algorithm parameters for value expansion.
            Uses defaults when ``None``.  Configurable via
            ``expansion_config`` in wizard settings YAML.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        expansion_config: ValueExpansionConfig | None = None,
    ) -> None:
        self._config = GroundingConfig(overlap_threshold=overlap_threshold)
        self._expansion_config = expansion_config

    def _try_expand(
        self,
        value: Any,
        user_message: str,
        schema_property: dict[str, Any],
    ) -> str | None:
        """Attempt to expand a partial string extraction to the full phrase.

        Returns the expanded value, or ``None`` if expansion is not
        applicable or yields no change.
        """
        # Gate 1: only non-empty string values
        if not isinstance(value, str) or not value:
            return None
        # Gate 2: skip non-string schema types
        if schema_property.get("type", "string") != "string":
            return None
        # Gate 3: skip enum fields (values are constrained)
        if "enum" in schema_property:
            return None
        # Gate 4: opt-in via x-extraction annotation (default off)
        x_ext = schema_property.get("x-extraction", {})
        if not x_ext.get("expand_from_message", False):
            return None
        return expand_value_in_message(
            value,
            user_message,
            stopwords=self._config.stopwords,
            config=self._expansion_config,
        )

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
        x_ext = schema_property.get("x-extraction", {})

        grounding_mode = x_ext.get("grounding")
        if grounding_mode == "skip":
            return MergeDecision.accept(reason="grounding=skip")

        result = is_field_grounded(
            field, new_value, user_message, schema_property,
            config=self._config,
        )

        if result.grounded:
            expanded = self._try_expand(
                new_value, user_message, schema_property,
            )
            if expanded is not None:
                return MergeDecision.transform(
                    expanded, reason="expanded partial extraction",
                )
            return MergeDecision.accept(reason="grounded")

        if existing_value is None:
            if x_ext.get("require_grounded", False):
                return MergeDecision.reject(
                    reason=(
                        f"require_grounded: extracted {new_value!r} "
                        f"not grounded in message"
                    ),
                )
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
