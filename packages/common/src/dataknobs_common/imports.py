"""Resolve a dotted path from configuration to a live Python object.

Four functions, one policy. ``resolve_dotted`` imports a module and returns one
attribute of it; ``resolve_callable`` and ``resolve_class`` add a shape check;
``resolve_optional_callable`` is the ``None``-tolerant lift for a config block
whose callable references are optional.

.. warning::

   **Import is execution.** Resolving a dotted path runs the target module's
   top level — every import it performs, every decorator it applies, every
   line at module scope — before this module has looked at the attribute, let
   alone checked its shape. There is no allow-list here and no sandbox.

   A dotted path must therefore come from the same trust domain as the
   application's own code: a config file, a deployment's policy bundle, a
   declaration a platform team authored. **Never build one from end-user
   input, a request body, or a per-tenant blob supplied by the tenant.**

   ``resolve_class`` returning the class rather than an instance is a
   **partial** mitigation and is described as partial deliberately: it means a
   wrong-shape target never runs its constructor, which closes the narrow case
   of a misfiled spec triggering ctor side effects. It does nothing about the
   module import that already happened, and nothing at all about a correctly
   shaped class from a hostile path.

Separator
---------

Both ``module.path:name`` and ``module.path.name`` are accepted everywhere.
Prefer ``:`` in new configuration and documentation — it says which half is the
module without the reader having to know the package layout — but ``.`` is
accepted because existing configuration uses it.

Exactly **one** attribute lookup is performed. ``module:Outer.Inner`` is not
supported: no caller needs it, and reading a chain would make the ``.`` form
ambiguous in a way the ``:`` form exists to prevent.

Errors
------

Two sibling exception types, both :class:`~dataknobs_common.exceptions.ConfigurationError`
subclasses — see :class:`~dataknobs_common.exceptions.DottedPathError` for why
sibling and not parent-and-child.

The message a caller sees names the reference and the *type* of the underlying
failure; the underlying exception's text travels on ``__cause__``, where a log
handler can pick it up. Since the import executes the target, that text is
arbitrary consumer output — it can carry an absolute filesystem path, a
credential read at module scope, or a stack from three libraries down — and
the resolution sites here are reached from surfaces that render a
``ConfigurationError`` to an HTTP client.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeAlias, TypeVar

from dataknobs_common.exceptions import (
    DottedPathError,
    DottedPathReason,
    DottedPathTypeError,
)

__all__ = [
    "ClassConstraint",
    "resolve_callable",
    "resolve_class",
    "resolve_dotted",
    "resolve_optional_callable",
]

_T = TypeVar("_T")

#: A class or runtime-checkable protocol used as a subclass constraint.
#:
#: Spelled as a callable rather than the obvious ``type[_T]`` because mypy
#: models ``type[T]`` as *instantiable* and rejects an abstract class or a
#: protocol in that position (``type-abstract``, python/mypy#4717). Those are
#: not an edge case here — they are the entire population of real arguments,
#: since a constraint nobody can subclass constrains nothing. The obvious
#: spelling therefore rejects every correct call and accepts only meaningless
#: ones, and it would push a ``# type: ignore`` into every call site in every
#: consuming codebase.
#:
#: What the looser spelling gives up is narrow: a factory *function* also
#: satisfies it, and mypy will not object. That lands where a bad constraint
#: already landed — :func:`resolve_class` feeds this straight to
#: ``issubclass``, which raises an unwrapped ``TypeError`` naming the problem.
#: See that function's ``Raises`` section; the treatment is deliberate and
#: unchanged.
ClassConstraint: TypeAlias = Callable[..., _T]

#: How many public callables to name when an attribute is missing.
_SUGGESTION_LIMIT = 10


def _split(ref: str) -> tuple[str, str]:
    """Split *ref* into ``(module_path, attribute)``.

    Raises:
        DottedPathError: reason ``malformed`` — for a non-string, an empty or
            whitespace-only reference, a reference with no separator at all,
            or either half being empty.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise DottedPathError(
            f"Expected a dotted path of the form 'module.path:name'; got {ref!r}",
            ref=str(ref),
            reason=DottedPathReason.MALFORMED,
        )

    ref = ref.strip()
    module_path, sep, attribute = ref.rpartition(":" if ":" in ref else ".")

    if not sep or not module_path or not attribute:
        raise DottedPathError(
            f"Invalid dotted path {ref!r}; expected 'module.path:name' "
            "or 'module.path.name'",
            ref=ref,
            reason=DottedPathReason.MALFORMED,
        )

    return module_path, attribute


def _suggestions(module: Any) -> str:
    """Public callables *this* module defines, for a missing-attribute message.

    Filtered to symbols whose ``__module__`` is the module itself, because the
    unfiltered namespace is mostly imports — a message listing ``Any``,
    ``Protocol`` and ``dataclass`` before the module's own functions is worse
    than no message, and alphabetical truncation reliably cuts the useful half
    (lowercase function names sort after imported class names).

    A pure re-export module — a package ``__init__`` — defines nothing of its
    own, so the filter would empty the list exactly where a caller most needs
    it. Those fall back to the whole namespace.
    """
    own_name = getattr(module, "__name__", None)
    public = [
        name
        for name in dir(module)
        if not name.startswith("_") and callable(getattr(module, name, None))
    ]
    defined_here = [
        name
        for name in public
        if getattr(getattr(module, name, None), "__module__", None) == own_name
    ]

    available = sorted(defined_here or public)
    if not available:
        return "(none)"

    shown = ", ".join(available[:_SUGGESTION_LIMIT])
    if len(available) > _SUGGESTION_LIMIT:
        shown += f", ... ({len(available) - _SUGGESTION_LIMIT} more)"
    return shown


def resolve_dotted(ref: str) -> Any:
    """Import a module and return one attribute of it, unchecked.

    The base of the family. Use it when any object will do; prefer
    :func:`resolve_callable` or :func:`resolve_class` when the caller has an
    expectation about the target, so that a config typo naming the wrong kind
    of symbol is caught here rather than at first use.

    Args:
        ref: ``"module.path:name"`` or ``"module.path.name"``.

    Returns:
        The resolved attribute, whatever it is.

    Raises:
        DottedPathError: The path is malformed, the module cannot be
            imported, or it has no such attribute.

    Warning:
        Importing the module **executes** it. See the module docstring.
    """
    module_path, attribute = _split(ref)

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        # `Exception`, not `ImportError`: the import runs the target module's
        # top level, so the failure can be anything that module raises. The
        # text stays on `__cause__` for the same reason.
        raise DottedPathError(
            f"Cannot import module {module_path!r} from {ref!r} "
            f"({type(exc).__name__})",
            ref=ref,
            reason=DottedPathReason.MODULE_NOT_FOUND,
        ) from exc

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise DottedPathError(
            f"Module {module_path!r} has no attribute {attribute!r} "
            f"(from {ref!r}). Available: {_suggestions(module)}",
            ref=ref,
            reason=DottedPathReason.ATTRIBUTE_NOT_FOUND,
        ) from exc


def resolve_callable(ref: str) -> Callable[..., Any]:
    """Resolve *ref* to a callable.

    Args:
        ref: ``"module.path:name"`` or ``"module.path.name"``.

    Returns:
        The resolved callable.

    Raises:
        DottedPathError: The path could not be resolved, or the target is
            not callable (reason ``not_callable``).

    Warning:
        Importing the module **executes** it. See the module docstring.
    """
    target = resolve_dotted(ref)

    if not callable(target):
        # `DottedPathError`, not `DottedPathTypeError`: "callable" is not a
        # base a caller declared, so there is no `expected` type to report,
        # and no config layout to have gotten wrong -- the path simply names
        # the wrong symbol. `DottedPathTypeError` is reserved for a target
        # that failed a base the *caller* named.
        raise DottedPathError(
            f"{ref!r} resolved to {type(target).__name__}, which is not callable",
            ref=ref,
            reason=DottedPathReason.NOT_CALLABLE,
        )

    return target


def resolve_class(ref: str, base: ClassConstraint[_T]) -> type[_T]:
    """Resolve *ref* to a class that subclasses *base*.

    **Returns the class; the caller instantiates.** That is the load-bearing
    choice in this function's signature, not an inconvenience to route around:
    it makes validate-before-instantiate the only order this function can
    express. A resolver that returned an instance would have to construct the
    target before it could check it, so a mistyped path would run an unrelated
    class's ``__init__`` — arbitrary code with whatever side effects it has —
    and only then be rejected. Callers instantiate differently anyway, each
    passing its own parameters.

    Args:
        ref: ``"module.path:name"`` or ``"module.path.name"``.
        base: The class or runtime-checkable protocol the target must
            satisfy. Typed as :data:`ClassConstraint` rather than
            ``type[_T]`` so that an abstract class or a protocol — which is
            every useful argument — type-checks at the call site; see that
            alias for why the obvious spelling cannot be used.

    Returns:
        The resolved class — **not** an instance of it.

    Raises:
        DottedPathError: The path could not be resolved.
        DottedPathTypeError: It resolved, and the target is not a class or
            does not subclass *base*.
        TypeError: *base* is not usable as a subclass constraint — a
            non-runtime-checkable protocol, a data protocol, or something
            that is not a class at all (a factory function reaches here,
            since :data:`ClassConstraint` cannot exclude one statically).
            Deliberately **not** wrapped: that is a defect in the calling
            code, and dressing it as a ``ConfigurationError`` would send the
            reader to inspect a config file that is fine.

    Note:
        A structural (protocol) *base* tests member **presence**, so a class
        whose member is a non-callable attribute of the right name still
        passes. That is a property of runtime-checkable protocols, not of
        this function.

    Warning:
        Importing the module **executes** it. See the module docstring.
    """
    target = resolve_dotted(ref)

    if not isinstance(target, type) or not issubclass(target, base):
        raise DottedPathTypeError(
            f"{ref!r} must resolve to a subclass of "
            f"{base.__module__}.{base.__qualname__}; got {target!r}",
            ref=ref,
            expected=base,
        )

    return target


def resolve_optional_callable(
    ref: Any,
    *,
    field_name: str,
    owner: str,
) -> Callable[..., Any] | None:
    """Resolve an optional callable reference, naming its config site on failure.

    The standard lift for a config block that accepts callable references
    under several optional keys — identity callables on a source, transform
    hooks on a bot, on-save handlers on a tool. ``None`` in, ``None`` out; a
    reference that is present but unresolvable still raises, because "the key
    was omitted" and "the key was wrong" are different states and only the
    first is optional.

    Args:
        ref: A dotted path, or ``None`` when the config key was omitted.
        field_name: The config key, for the error message.
        owner: The object being built — a source, bot, or tool name — so a
            bad reference points back at its config site.

    Returns:
        The resolved callable, or ``None`` when *ref* is ``None``.

    Raises:
        DottedPathError: *ref* is not ``None`` and could not be resolved to
            a callable.

    Warning:
        Importing the module **executes** it. See the module docstring.
    """
    if ref is None:
        return None

    try:
        return resolve_callable(ref)
    except DottedPathError as exc:
        raise DottedPathError(
            f"{owner!r}: cannot resolve {field_name}={ref!r} — {exc}",
            ref=exc.ref,
            reason=exc.reason,
            field_name=field_name,
            owner=owner,
        ) from exc
