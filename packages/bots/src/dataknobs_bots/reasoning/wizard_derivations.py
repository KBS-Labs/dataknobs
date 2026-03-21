"""Field derivation rules for wizard extraction recovery.

Derivation rules define deterministic relationships between fields.
When a source field is present but a target field is missing, the
framework can derive the target without an additional LLM call.

This is the cheapest recovery strategy — pure functions, no I/O.

Configuration example::

    settings:
      derivations:
        - source: domain_id
          target: domain_name
          transform: title_case
          when: target_missing

        - source: domain_name
          target: domain_id
          transform: lower_hyphen
          when: target_missing
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ──


@runtime_checkable
class FieldTransform(Protocol):
    """Protocol for custom field transform functions.

    Implementations receive the source field value and the full
    wizard data dict, returning the derived target value.
    """

    def transform(self, value: Any, wizard_data: dict[str, Any]) -> Any:
        """Transform a source field value into a target field value."""
        ...


# ── Data types ──


@dataclass(frozen=True)
class DerivationRule:
    """A single field derivation rule.

    Attributes:
        source: Name of the field to derive from.
        target: Name of the field to derive into.
        transform_name: Transform to apply. One of the built-in names
            (``title_case``, ``lower_hyphen``, ``lower_underscore``,
            ``copy``, ``template``) or ``custom`` for a pluggable
            :class:`FieldTransform`.
        when: Guard condition controlling when derivation fires.
            ``target_missing`` (default) — derive only when the target
            field is absent or not present.  ``target_empty`` — derive
            when the target is ``None`` or empty string.  ``always`` —
            always derive, overwriting existing values.
        template: Jinja2 template string used by the ``template``
            transform.  Rendered with the full wizard data dict.
        custom_transform: Loaded :class:`FieldTransform` instance for
            ``custom`` transforms.  Populated by
            :func:`parse_derivation_rules` at config-load time.
    """

    source: str
    target: str
    transform_name: str
    when: str = "target_missing"
    template: str | None = None
    custom_transform: FieldTransform | None = field(default=None, repr=False)


# ── Built-in transforms ──
#
# Each transform is a pure function:
#   (value: Any, wizard_data: dict[str, Any]) -> Any


def _title_case(value: Any, _data: dict[str, Any]) -> str:
    """Convert a hyphen/underscore-separated ID to Title Case.

    ``"chess-champ"`` → ``"Chess Champ"``
    """
    s = str(value)
    return s.replace("-", " ").replace("_", " ").title().strip()


def _lower_hyphen(value: Any, _data: dict[str, Any]) -> str:
    """Convert a display name to a lowercase hyphenated slug.

    ``"Chess Champ"`` → ``"chess-champ"``
    """
    return re.sub(r"[\s_]+", "-", str(value)).strip("-").lower()


def _lower_underscore(value: Any, _data: dict[str, Any]) -> str:
    """Convert a display name to snake_case.

    ``"Chess Champ"`` → ``"chess_champ"``
    """
    return re.sub(r"[\s-]+", "_", str(value)).strip("_").lower()


def _copy(value: Any, _data: dict[str, Any]) -> Any:
    """Direct copy of the source value."""
    return value


BUILTIN_TRANSFORMS: dict[str, Any] = {
    "title_case": _title_case,
    "lower_hyphen": _lower_hyphen,
    "lower_underscore": _lower_underscore,
    "copy": _copy,
}

_VALID_WHEN_CONDITIONS = frozenset({"target_missing", "target_empty", "always"})


# ── Parsing ──


def parse_derivation_rules(
    config_list: list[dict[str, Any]],
) -> list[DerivationRule]:
    """Parse derivation config dicts into :class:`DerivationRule` objects.

    Called once at :meth:`WizardReasoning.from_config` time.  Custom
    transforms are loaded and cached here so they are not re-imported
    on every turn.

    Args:
        config_list: List of derivation rule dicts from wizard settings.

    Returns:
        Parsed rules ready for :func:`apply_field_derivations`.
    """
    rules: list[DerivationRule] = []
    for item in config_list:
        source = item.get("source", "")
        target = item.get("target", "")
        transform_name = item.get("transform", "copy")
        when = item.get("when", "target_missing")
        template = item.get("template")

        if not source or not target:
            logger.warning(
                "Derivation rule missing source or target: %s",
                item,
            )
            continue

        if when not in _VALID_WHEN_CONDITIONS:
            logger.warning(
                "Unknown derivation 'when' condition %r — "
                "defaulting to 'target_missing'.",
                when,
            )
            when = "target_missing"

        # Validate built-in transform name
        if (
            transform_name != "custom"
            and transform_name != "template"
            and transform_name not in BUILTIN_TRANSFORMS
        ):
            logger.warning(
                "Unknown derivation transform %r in rule %s → %s. "
                "Available: %s, 'template', 'custom'.",
                transform_name,
                source,
                target,
                sorted(BUILTIN_TRANSFORMS),
            )
            continue

        # Load custom transform if specified
        custom_transform: FieldTransform | None = None
        if transform_name == "custom":
            custom_class_path = item.get("custom_class")
            if custom_class_path:
                custom_transform = _load_custom_transform(custom_class_path)
                if custom_transform is None:
                    continue  # Loading failed — skip this rule
            else:
                logger.warning(
                    "Derivation rule with transform='custom' missing "
                    "'custom_class': %s → %s",
                    source,
                    target,
                )
                continue

        rules.append(
            DerivationRule(
                source=source,
                target=target,
                transform_name=transform_name,
                when=when,
                template=template,
                custom_transform=custom_transform,
            )
        )

    return rules


# ── Execution ──


def apply_field_derivations(
    rules: list[DerivationRule],
    data: dict[str, Any],
    field_is_present: Callable[[Any], bool] | None = None,
) -> set[str]:
    """Apply derivation rules to wizard state data (in-place).

    Rules are processed in order.  Each rule runs at most once per
    invocation — this prevents circular derivation loops.  When two
    rules derive from each other (A→B and B→A), the first rule whose
    source is present wins.

    Args:
        rules: Parsed derivation rules from
            :func:`parse_derivation_rules`.
        data: Wizard state data dict (modified in-place).
        field_is_present: Optional callable ``(value) -> bool`` to
            check whether a field value counts as "present".
            Defaults to ``lambda v: v is not None``.

    Returns:
        Set of target field keys that were derived (newly set).
    """
    if not rules:
        return set()

    is_present = field_is_present or (lambda v: v is not None)
    derived: set[str] = set()

    for rule in rules:
        source_value = data.get(rule.source)
        if not is_present(source_value):
            continue  # Source field not present — can't derive

        # Check guard condition.
        # target_missing uses is_present() semantics (consistent with
        # the confidence gate's _field_is_present): a key present with
        # value None is treated as absent and will be re-derived.
        # This differs from _apply_transition_derivations which uses
        # ``key in state.data`` — a stricter key-presence check.
        target_value = data.get(rule.target)
        if rule.when == "target_missing":
            if is_present(target_value):
                continue  # Target already present — skip
        elif rule.when == "target_empty":
            if is_present(target_value) and target_value != "":
                continue  # Target has a non-empty value — skip
        # "always" — no guard, always derive

        # Execute transform
        result = _execute_transform(rule, source_value, data)
        if result is not None:
            data[rule.target] = result
            derived.add(rule.target)
            logger.debug(
                "Derived %s = %r from %s via %s",
                rule.target,
                result,
                rule.source,
                rule.transform_name,
            )

    return derived


def _execute_transform(
    rule: DerivationRule,
    source_value: Any,
    data: dict[str, Any],
) -> Any:
    """Execute a single transform, returning the derived value or None."""
    if rule.transform_name == "template":
        if not rule.template:
            logger.warning(
                "Derivation template transform missing 'template' key "
                "for rule %s → %s",
                rule.source,
                rule.target,
            )
            return None
        import jinja2

        try:
            env = jinja2.Environment(undefined=jinja2.StrictUndefined)
            rendered = env.from_string(rule.template).render(**data)
            return rendered.strip() if rendered.strip() else None
        except jinja2.UndefinedError:
            # Not all referenced variables are present yet — skip
            # silently.  This is expected when the template references
            # multiple fields and only some have been extracted so far.
            return None
        except Exception:
            logger.warning(
                "Derivation template render failed for %s → %s",
                rule.source,
                rule.target,
                exc_info=True,
            )
            return None

    if rule.transform_name == "custom" and rule.custom_transform is not None:
        try:
            return rule.custom_transform.transform(source_value, data)
        except Exception:
            logger.warning(
                "Custom derivation transform failed for %s → %s",
                rule.source,
                rule.target,
                exc_info=True,
            )
            return None

    # Built-in transform
    transform_fn = BUILTIN_TRANSFORMS.get(rule.transform_name)
    if transform_fn is None:
        logger.warning(
            "Unknown derivation transform: %s",
            rule.transform_name,
        )
        return None

    try:
        return transform_fn(source_value, data)
    except Exception:
        logger.warning(
            "Derivation transform %s failed for %s → %s",
            rule.transform_name,
            rule.source,
            rule.target,
            exc_info=True,
        )
        return None


def _load_custom_transform(dotted_path: str) -> FieldTransform | None:
    """Load a custom :class:`FieldTransform` from a dotted import path.

    Args:
        dotted_path: Fully qualified path, e.g.
            ``mypackage.transforms.SubjectToId``.

    Returns:
        Instantiated transform, or ``None`` on failure.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls()
        if not isinstance(instance, FieldTransform):
            logger.warning(
                "Custom transform %s does not implement FieldTransform "
                "protocol",
                dotted_path,
            )
            return None
        return instance
    except Exception:
        logger.warning(
            "Failed to load custom transform %s",
            dotted_path,
            exc_info=True,
        )
        return None
