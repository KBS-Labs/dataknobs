"""Shared arity/await normalization for record-step callables.

Both the validation gate (:mod:`validators`) and the enrichment step
(:mod:`enrichers`) accept a user callable that may be written as
``record -> X`` or ``(record, context) -> X`` and may be sync or async, and
both must adapt it to the engine's fixed ``(record, context)`` call shape. That
adaptation is identical bar the terminal coercion (the gate coerces to ``bool``;
the enricher passes the value through), so it lives here once rather than being
copy-pasted into each module.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def normalize_record_callable(
    fn: Callable[..., Any],
    *,
    coerce: Callable[[Any], Any] | None = None,
) -> Callable[..., Any]:
    """Arity- and await-normalize a record callable to ``(record, context)``.

    The FSM engine always invokes a record step as ``fn(record, context)``. A
    supplied callable may be a bare ``record -> X`` callable, a
    ``(record, context) -> X`` callable, or an
    :class:`~dataknobs_fsm.functions.base.ITransformFunction`'s bound method
    (sync ``(data)`` or async ``(data, context)``). The returned callable always
    accepts ``(record, context)``, forwards the right number of arguments, and is
    a coroutine function iff ``fn`` is — so the caller's ``iscoroutinefunction`` /
    ``isawaitable`` check routes it correctly.

    Arity detection counts positional parameters only: a callable declaring two
    or more positionals (or ``*args``) receives ``(record, context)``; otherwise
    it receives ``record`` alone. A predicate that declares ``context`` as a
    *required keyword-only* argument (``def fn(record, *, context): ...``) is
    therefore called with the record alone and raises ``TypeError`` at evaluation
    time — write ``(record, context)`` or ``(record, context=None)`` instead.

    Args:
        fn: The user callable to normalize.
        coerce: Optional terminal coercion applied to the callable's result
            (e.g. ``bool`` for a gate). ``None`` (the default) returns the
            result unchanged (the enricher form).

    Returns:
        A ``(record, context)`` callable (a coroutine function when ``fn`` is).
    """
    try:
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        wants_context = (
            any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
            or len(params) >= 2
        )
    except (TypeError, ValueError):
        # Builtins / C-callables with no introspectable signature: be permissive
        # and pass only the record (the common ``record -> X`` shape).
        wants_context = False

    if inspect.iscoroutinefunction(fn):

        async def async_call(data: dict, context: Any = None) -> Any:
            out = await (fn(data, context) if wants_context else fn(data))
            return coerce(out) if coerce is not None else out

        return async_call

    def sync_call(data: dict, context: Any = None) -> Any:
        out = fn(data, context) if wants_context else fn(data)
        return coerce(out) if coerce is not None else out

    return sync_call
