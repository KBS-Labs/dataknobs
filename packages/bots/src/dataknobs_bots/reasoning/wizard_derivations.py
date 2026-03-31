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

        # Conditional: set flag based on field value
        - source: intent
          target: kb_enabled
          transform: equals
          transform_value: research_assistant
          when: target_missing

        # Lookup table
        - source: intent
          target: synthesis_style
          transform: map
          transform_map:
            research_assistant: conversational
            tutor: socratic
          transform_default: structured
          when: target_missing

        # General-purpose expression (native-typed result)
        - source: intent
          target: max_questions
          transform: expression
          expression: "10 if value == 'quiz_maker' else 5"
          when: target_missing
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.expressions import safe_eval

logger = logging.getLogger(__name__)


# Sentinel value to signal "skip this derivation" — distinct from None,
# which is a valid derivation result (e.g., constant: null, expression
# evaluating to None).  Transforms that cannot compute a value return
# _SKIP; transforms that return None produce a None value in wizard data.
_SKIP = object()


# ── Protocol ──


@runtime_checkable
class FieldTransform(Protocol):
    """Protocol for custom field transform functions.

    Implementations receive the source field value and the full
    wizard data dict, returning the derived target value.

    Return ``None`` to signal that the derivation should be skipped
    (e.g., when preconditions are not met).  Any other value —
    including falsy values like ``0``, ``False``, or ``""`` — will
    be stored as the derived result.

    Note: ``None`` is converted to ``_SKIP`` internally so that
    custom transforms need not know about the internal sentinel.
    """

    def transform(self, value: Any, wizard_data: dict[str, Any]) -> Any:
        """Transform a source field value into a target field value.

        Returns:
            The derived value, or ``None`` to skip the derivation.
        """
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
            :class:`FieldTransform`, or one of the parameterized
            transforms (``equals``, ``not_equals``, ``constant``,
            ``map``, ``boolean``, ``one_of``, ``contains``, ``first``,
            ``last``, ``join``, ``split``, ``length``, ``regex_match``,
            ``regex_extract``, ``regex_replace``), or ``expression``
            for a safe Python expression.
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
        transform_value: Parameter for ``equals``, ``not_equals``,
            ``constant``, ``contains``, ``join`` (separator),
            ``split`` (separator), ``regex_match``, ``regex_extract``,
            ``regex_replace`` (pattern).
        transform_values: List parameter for ``one_of``.
        transform_map: Dict parameter for ``map``.
        transform_default: Fallback value for ``map`` when key not found.
            Defaults to ``_SKIP`` (derivation skipped).  Set to
            ``null``/``None`` in config to explicitly store ``None``.
        transform_replacement: Replacement string for ``regex_replace``.
        expression: Python expression string for ``expression`` transform.
        compiled_regex: Pre-compiled regex pattern for ``regex_*``
            transforms.  Populated by :func:`parse_derivation_rules`
            at config-load time.
    """

    source: str
    target: str
    transform_name: str
    when: str = "target_missing"
    template: str | None = None
    custom_transform: FieldTransform | None = field(default=None, repr=False)
    # Parameterized transform fields
    transform_value: Any = None
    transform_values: list[Any] | None = None
    transform_map: dict[str, Any] | None = None
    transform_default: Any = _SKIP
    transform_replacement: str | None = None
    expression: str | None = None
    # Pre-compiled regex (populated at parse time for regex_* transforms)
    compiled_regex: re.Pattern[str] | None = field(default=None, repr=False)


# ── Built-in transforms ──
#
# Non-parameterized transforms: (value, data) -> Any
# Parameterized transforms: (value, data, **kwargs) -> Any


def _title_case(value: Any, _data: dict[str, Any]) -> str | None:
    """Convert a hyphen/underscore-separated ID to Title Case.

    ``"chess-champ"`` → ``"Chess Champ"``

    Returns ``None`` for empty/whitespace-only input.
    """
    s = str(value)
    result = s.replace("-", " ").replace("_", " ").title().strip()
    return result or None


def _lower_hyphen(value: Any, _data: dict[str, Any]) -> str | None:
    """Convert a display name to a lowercase hyphenated slug.

    ``"Chess Champ"`` → ``"chess-champ"``

    Returns ``None`` for empty/whitespace-only input.
    """
    result = re.sub(r"[\s_]+", "-", str(value)).strip("-").lower()
    return result or None


def _lower_underscore(value: Any, _data: dict[str, Any]) -> str | None:
    """Convert a display name to snake_case.

    ``"Chess Champ"`` → ``"chess_champ"``

    Returns ``None`` for empty/whitespace-only input.
    """
    result = re.sub(r"[\s-]+", "_", str(value)).strip("_").lower()
    return result or None


def _copy(value: Any, _data: dict[str, Any]) -> Any:
    """Direct copy of the source value."""
    return value


# ── Conditional/Logical transforms (parameterized) ──


def _equals(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> bool:
    """``True`` if ``str(source) == str(transform_value)``."""
    return str(value) == str(transform_value)


def _not_equals(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> bool:
    """``True`` if ``str(source) != str(transform_value)``."""
    return str(value) != str(transform_value)


def _constant(
    _value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> Any:
    """Returns ``transform_value`` regardless of source."""
    return transform_value


def _map_transform(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_map: dict[str, Any] | None = None,
    transform_default: Any = _SKIP,
    **_kwargs: Any,
) -> Any:
    """Lookup ``str(source)`` in map; returns mapped value or default.

    When the key is not found and no ``transform_default`` was
    configured, returns ``_SKIP`` so the derivation is skipped
    rather than storing ``None``.
    """
    if transform_map is None:
        return _SKIP
    key = str(value)
    if key in transform_map:
        return transform_map[key]
    return transform_default


def _boolean(
    value: Any,
    _data: dict[str, Any],
    **_kwargs: Any,
) -> bool:
    """``True`` if source is truthy, ``False`` otherwise."""
    return bool(value)


def _one_of(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_values: list[Any] | None = None,
    **_kwargs: Any,
) -> bool:
    """``True`` if source is in the values list."""
    if transform_values is None:
        return False
    return value in transform_values


def _contains(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> bool:
    """``True`` if ``transform_value`` is a case-insensitive substring of source."""
    if transform_value is None:
        return False
    return str(transform_value).lower() in str(value).lower()


# ── Collection transforms (parameterized) ──


def _first(
    value: Any,
    _data: dict[str, Any],
    **_kwargs: Any,
) -> Any:
    """First element of iterable source; ``_SKIP`` if empty/non-iterable."""
    try:
        it = iter(value)
        return next(it, _SKIP)
    except TypeError:
        return _SKIP


def _last(
    value: Any,
    _data: dict[str, Any],
    **_kwargs: Any,
) -> Any:
    """Last element of iterable source; ``_SKIP`` if empty/non-iterable."""
    try:
        items = list(value)
        return items[-1] if items else _SKIP
    except TypeError:
        return _SKIP


def _join(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> str | Any:
    """Join list elements into string with separator."""
    separator = str(transform_value) if transform_value is not None else ", "
    try:
        items = list(value)
        return separator.join(str(item) for item in items) if items else _SKIP
    except TypeError:
        return _SKIP


def _split(
    value: Any,
    _data: dict[str, Any],
    *,
    transform_value: Any = None,
    **_kwargs: Any,
) -> list[str] | Any:
    """Split string into list, with ``strip()`` on each element."""
    if value is None:
        return _SKIP
    separator = str(transform_value) if transform_value is not None else ","
    s = str(value)
    if not s:
        return _SKIP
    return [part.strip() for part in s.split(separator)]


def _length(
    value: Any,
    _data: dict[str, Any],
    **_kwargs: Any,
) -> int | Any:
    """Length of string/list/dict; ``_SKIP`` if not measurable."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return _SKIP


# ── Regex transforms (parameterized) ──


def _regex_match(
    value: Any,
    _data: dict[str, Any],
    *,
    compiled_regex: re.Pattern[str] | None = None,
    **_kwargs: Any,
) -> bool:
    """``True`` if source matches pattern (using ``re.search``)."""
    if compiled_regex is None:
        return False
    return compiled_regex.search(str(value)) is not None


def _regex_extract(
    value: Any,
    _data: dict[str, Any],
    *,
    compiled_regex: re.Pattern[str] | None = None,
    **_kwargs: Any,
) -> str | Any:
    """First capture group match, or ``_SKIP`` if no match."""
    if compiled_regex is None:
        return _SKIP
    match = compiled_regex.search(str(value))
    if match and match.groups():
        return match.group(1)
    return _SKIP


def _regex_replace(
    value: Any,
    _data: dict[str, Any],
    *,
    compiled_regex: re.Pattern[str] | None = None,
    transform_replacement: str | None = None,
    **_kwargs: Any,
) -> str:
    """Replace all matches of pattern with replacement string."""
    if compiled_regex is None or transform_replacement is None:
        return str(value)
    return compiled_regex.sub(transform_replacement, str(value))


# ── Transform registries ──


BUILTIN_TRANSFORMS: dict[str, Any] = {
    "title_case": _title_case,
    "lower_hyphen": _lower_hyphen,
    "lower_underscore": _lower_underscore,
    "copy": _copy,
    # Conditional/Logical
    "equals": _equals,
    "not_equals": _not_equals,
    "constant": _constant,
    "map": _map_transform,
    "boolean": _boolean,
    "one_of": _one_of,
    "contains": _contains,
    # Collection
    "first": _first,
    "last": _last,
    "join": _join,
    "split": _split,
    "length": _length,
    # Regex
    "regex_match": _regex_match,
    "regex_extract": _regex_extract,
    "regex_replace": _regex_replace,
}

# Parameterized transforms receive config kwargs via the rule
PARAMETERIZED_TRANSFORMS: frozenset[str] = frozenset({
    "equals", "not_equals", "constant", "map", "boolean",
    "one_of", "contains",
    "first", "last", "join", "split", "length",
    "regex_match", "regex_extract", "regex_replace",
})

_VALID_WHEN_CONDITIONS = frozenset({"target_missing", "target_empty", "always"})

# ── Parsing ──


def parse_derivation_rules(
    config_list: list[dict[str, Any]],
) -> list[DerivationRule]:
    """Parse derivation config dicts into :class:`DerivationRule` objects.

    Called once at :meth:`WizardReasoning.from_config` time.  Custom
    transforms are loaded and cached here so they are not re-imported
    on every turn.  Regex patterns are pre-compiled and cached on
    the rule.

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
            and transform_name != "expression"
            and transform_name not in BUILTIN_TRANSFORMS
        ):
            logger.warning(
                "Unknown derivation transform %r in rule %s → %s. "
                "Available: %s, 'template', 'custom', 'expression'.",
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

        # Read parameterized transform fields
        transform_value = item.get("transform_value")
        transform_values = item.get("transform_values")
        transform_map = item.get("transform_map")
        transform_default = item.get("transform_default", _SKIP)
        transform_replacement = item.get("transform_replacement")
        expression = item.get("expression")
        compiled_regex: re.Pattern[str] | None = None

        # Validate parameterized fields per transform type
        if not _validate_transform_params(
            transform_name, source, target,
            transform_value=transform_value,
            transform_values=transform_values,
            transform_map=transform_map,
            transform_replacement=transform_replacement,
            expression=expression,
        ):
            continue

        # Pre-compile regex patterns
        if transform_name in ("regex_match", "regex_extract", "regex_replace"):
            try:
                compiled_regex = re.compile(str(transform_value))
            except re.error as exc:
                logger.warning(
                    "Invalid regex pattern %r in derivation %s → %s: %s",
                    transform_value,
                    source,
                    target,
                    exc,
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
                transform_value=transform_value,
                transform_values=transform_values,
                transform_map=transform_map,
                transform_default=transform_default,
                transform_replacement=transform_replacement,
                expression=expression,
                compiled_regex=compiled_regex,
            )
        )

    return rules


def _validate_transform_params(
    transform_name: str,
    source: str,
    target: str,
    *,
    transform_value: Any,
    transform_values: Any,
    transform_map: Any,
    transform_replacement: Any,
    expression: Any,
) -> bool:
    """Validate required config fields for parameterized transforms.

    Returns ``True`` if valid, ``False`` if the rule should be skipped.
    """
    if transform_name in ("equals", "not_equals", "contains"):
        if transform_value is None:
            logger.warning(
                "Derivation transform %r requires 'transform_value' "
                "for rule %s → %s",
                transform_name,
                source,
                target,
            )
            return False

    elif transform_name == "map":
        if not isinstance(transform_map, dict):
            logger.warning(
                "Derivation transform 'map' requires 'transform_map' "
                "(dict) for rule %s → %s",
                source,
                target,
            )
            return False

    elif transform_name == "one_of":
        if not isinstance(transform_values, list):
            logger.warning(
                "Derivation transform 'one_of' requires 'transform_values' "
                "(list) for rule %s → %s",
                source,
                target,
            )
            return False

    elif transform_name in ("regex_match", "regex_extract"):
        if transform_value is None:
            logger.warning(
                "Derivation transform %r requires 'transform_value' "
                "(regex pattern) for rule %s → %s",
                transform_name,
                source,
                target,
            )
            return False

    elif transform_name == "regex_replace":
        if transform_value is None:
            logger.warning(
                "Derivation transform 'regex_replace' requires "
                "'transform_value' (pattern) for rule %s → %s",
                source,
                target,
            )
            return False
        if transform_replacement is None:
            logger.warning(
                "Derivation transform 'regex_replace' requires "
                "'transform_replacement' for rule %s → %s",
                source,
                target,
            )
            return False

    elif transform_name == "expression":
        if not expression:
            logger.warning(
                "Derivation transform 'expression' requires "
                "'expression' for rule %s → %s",
                source,
                target,
            )
            return False

    return True


# ── Execution ──


def apply_field_derivations(
    rules: list[DerivationRule],
    data: dict[str, Any],
    field_is_present: Callable[[Any], bool] | None = None,
) -> set[str]:
    """Apply derivation rules to wizard state data (in-place).

    Rules are applied in a single pass through the list — there is
    no multi-pass re-evaluation, so each rule executes at most once.
    For bidirectional pairs (A→B and B→A) with the default
    ``target_missing`` guard, the first rule whose source is present
    wins; with ``when: always``, both rules fire in order and later
    rules may overwrite earlier results.

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
        elif rule.when == "always":
            pass  # No guard — always derive
        else:
            logger.warning(
                "Unknown 'when' condition %r at execution time for "
                "rule %s → %s — skipping.",
                rule.when,
                rule.source,
                rule.target,
            )
            continue

        # Execute transform
        result = _execute_transform(rule, source_value, data)
        if result is not _SKIP:
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
    """Execute a single transform, returning the derived value or ``_SKIP``.

    Returns ``_SKIP`` when the transform cannot compute a value (error,
    missing config, template with undefined variables, etc.).  Any other
    return value — including ``None`` — is stored as the derived result.
    """
    # Step 1: template (existing special case)
    if rule.transform_name == "template":
        if not rule.template:
            logger.warning(
                "Derivation template transform missing 'template' key "
                "for rule %s → %s",
                rule.source,
                rule.target,
            )
            return _SKIP
        import jinja2

        try:
            env = jinja2.Environment(undefined=jinja2.StrictUndefined)
            rendered = env.from_string(rule.template).render(**data)
            return rendered.strip() if rendered.strip() else _SKIP
        except jinja2.UndefinedError:
            # Not all referenced variables are present yet — skip
            # silently.  This is expected when the template references
            # multiple fields and only some have been extracted so far.
            return _SKIP
        except jinja2.TemplateError:
            logger.warning(
                "Derivation template render failed for %s → %s",
                rule.source,
                rule.target,
                exc_info=True,
            )
            return _SKIP

    # Step 2: custom (existing special case)
    # Custom transforms use None to signal "skip" per the FieldTransform
    # protocol — convert to _SKIP so the caller's sentinel check works.
    if rule.transform_name == "custom" and rule.custom_transform is not None:
        try:
            result = rule.custom_transform.transform(source_value, data)
            return _SKIP if result is None else result
        except Exception:  # Custom user code — cannot predict exception types
            logger.warning(
                "Custom derivation transform failed for %s → %s",
                rule.source,
                rule.target,
                exc_info=True,
            )
            return _SKIP

    # Step 3: expression (new special case — safe eval, native type)
    if rule.transform_name == "expression":
        return _execute_expression(rule, source_value, data)

    # Step 4: parameterized built-in transforms
    if rule.transform_name in PARAMETERIZED_TRANSFORMS:
        transform_fn = BUILTIN_TRANSFORMS.get(rule.transform_name)
        if transform_fn is None:
            logger.warning(
                "Unknown parameterized transform: %s",
                rule.transform_name,
            )
            return _SKIP
        try:
            return transform_fn(
                source_value,
                data,
                transform_value=rule.transform_value,
                transform_values=rule.transform_values,
                transform_map=rule.transform_map,
                transform_default=rule.transform_default,
                transform_replacement=rule.transform_replacement,
                compiled_regex=rule.compiled_regex,
            )
        except (TypeError, ValueError, AttributeError):
            logger.warning(
                "Derivation transform %s failed for %s → %s",
                rule.transform_name,
                rule.source,
                rule.target,
                exc_info=True,
            )
            return _SKIP

    # Step 5: existing non-parameterized built-ins
    transform_fn = BUILTIN_TRANSFORMS.get(rule.transform_name)
    if transform_fn is None:
        logger.warning(
            "Unknown derivation transform: %s",
            rule.transform_name,
        )
        return _SKIP

    try:
        result = transform_fn(source_value, data)
        # Non-parameterized built-in transforms (title_case, lower_hyphen,
        # lower_underscore, copy) use None to signal "couldn't compute"
        # (e.g., empty input) — convert to _SKIP
        if result is None:
            return _SKIP
        return result
    except (TypeError, ValueError, AttributeError):
        logger.warning(
            "Derivation transform %s failed for %s → %s",
            rule.transform_name,
            rule.source,
            rule.target,
            exc_info=True,
        )
        return _SKIP


def _execute_expression(
    rule: DerivationRule,
    source_value: Any,
    data: dict[str, Any],
) -> Any:
    """Execute an expression transform using the shared safe-eval engine.

    Available scope variables in expressions:

    - ``value`` — the source field value
    - ``data`` — snapshot of the full wizard data dict
    - ``has(key)`` — shorthand for ``data.get(key) is not None``

    Returns ``_SKIP`` when the expression fails or is missing.
    Returns the expression's native result otherwise (including ``None``).
    """
    if not rule.expression:
        logger.warning(
            "Expression transform missing 'expression' for %s → %s",
            rule.source,
            rule.target,
        )
        return _SKIP

    data_snapshot = dict(data)
    result = safe_eval(
        rule.expression,
        scope={
            "value": source_value,
            "data": data_snapshot,
            "has": lambda key: data_snapshot.get(key) is not None,
        },
        default=_SKIP,
    )
    if not result.success:
        logger.warning(
            "Expression transform failed for %s → %s: %r (%s)",
            rule.source,
            rule.target,
            rule.expression,
            result.error,
        )
        return _SKIP
    return result.value


def _load_custom_transform(dotted_path: str) -> FieldTransform | None:
    """Load a custom :class:`FieldTransform` from a dotted import path.

    Args:
        dotted_path: Fully qualified path, e.g.
            ``mypackage.transforms.SubjectToId``.

    Returns:
        Instantiated transform, or ``None`` on failure.
    """
    if not re.match(r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)+$", dotted_path):
        logger.warning(
            "Invalid dotted path for custom transform: %r",
            dotted_path,
        )
        return None

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
    except (ImportError, AttributeError, TypeError, ValueError):
        logger.warning(
            "Failed to load custom transform %s",
            dotted_path,
            exc_info=True,
        )
        return None
