"""Layered metadata-merge primitives.

Used by ``VectorMemory`` (tenant-scope enforcement),
``RAGKnowledgeBase`` (chunk-text protection), and the markdown
chunker (node-classification protection).  The primitive captures
the cross-cutting pattern: given a layered merge, force a chosen
set of keys to take their values from a designated source, and
warn when a caller attempted to override them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

__all__ = ["enforce_immutable_keys"]

_logger = logging.getLogger(__name__)


def _values_differ(a: Any, b: Any) -> bool:
    """Array-safe inequality check.

    ``a != b`` raises ``ValueError`` when ``a`` and ``b`` are numpy
    arrays (or any object whose ``__ne__`` returns an element-wise
    array). Fall back to identity, then to ``not __eq__`` interpreted
    via Python's truth rules. If even ``__eq__`` returns a non-bool
    we treat the values as differing — the warning fires (correctly
    flagging caller intent) and the caller-supplied value is
    discarded, matching the helper's contract.
    """
    if a is b:
        return False
    try:
        return bool(a != b)
    except (ValueError, TypeError):
        # numpy/array __ne__ returned an element-wise truth array, or
        # the comparison itself raised. Use __eq__ symmetrically.
        try:
            equal = a == b
            if isinstance(equal, bool):
                return not equal
            # Element-wise truth array — reduce to a single bool. Use
            # ``all()`` so any difference flags as a mismatch.
            return not bool(getattr(equal, "all", lambda: False)())
        except (ValueError, TypeError):
            return True


def enforce_immutable_keys(
    *,
    target: dict[str, Any],
    caller: dict[str, Any] | None,
    source: dict[str, Any],
    keys: Iterable[str],
    logger: logging.Logger | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Force ``keys`` in ``target`` to take their values from ``source``.

    Mutates and returns ``target``.  When ``caller`` supplied a
    different value for an immutable key that is also present in
    ``source``, emits a WARNING naming the key so misuse is
    debuggable.  When ``caller`` agrees with ``source``, or when
    ``caller`` did not provide the key at all, no warning is logged.

    Args:
        target: The merged dict to enforce against.  Mutated in-place
            and returned.  Typically the result of a layered merge
            (defaults < base < caller).
        caller: The caller-supplied layer that may have overridden
            immutable keys.  ``None`` means the caller supplied no
            metadata and no warnings will fire — including the case
            where ``target[key]`` already differs from
            ``source[key]``: the helper silently rewrites ``target``
            because there is no caller intent to flag.
        source: The authoritative source for immutable-key values
            (e.g. ``default_metadata`` for ``VectorMemory``, the
            chunker's locally-built dict for ``RAGKnowledgeBase``).
            Keys not present in ``source`` are skipped — there is
            nothing to enforce against.
        keys: The set of keys that must take their values from
            ``source``.
        logger: Optional logger to emit warnings on.  Defaults to a
            module-level logger.
        context: Optional human-readable context string included in
            warning messages (e.g. ``"VectorMemory.add_message"``).

    Returns:
        The mutated ``target`` dict.

    Note on value types:
        Equality checks between caller and source values are
        array-safe — numpy arrays, lists, and other non-scalar values
        are compared without raising ``ValueError`` from the
        ambiguous truth value of element-wise comparison.
    """
    log = logger or _logger
    for key in keys:
        if key not in source:
            continue
        source_value = source[key]
        if (
            caller is not None
            and key in caller
            and _values_differ(caller[key], source_value)
        ):
            ctx = f"{context}: " if context else ""
            log.warning(
                "%s%r is an immutable metadata key; caller-supplied value %r "
                "was discarded in favor of source value %r.",
                ctx,
                key,
                caller[key],
                source_value,
            )
        target[key] = source_value
    return target
