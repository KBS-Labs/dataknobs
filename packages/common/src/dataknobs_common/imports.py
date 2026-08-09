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

   Reading the attribute can execute code too: a module-level ``__getattr__``
   (PEP 562) runs on first access, which is how a lazy export defers an
   optional dependency. Both points are treated as execution here.

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
    "dotted_path",
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


def _peek(obj: Any, name: str) -> Any:
    """``getattr(obj, name)`` that cannot raise, for use inside a handler.

    ``getattr(..., None)`` swallows ``AttributeError`` only. A module with a
    PEP 562 ``__getattr__`` can raise anything for a name ``__dir__``
    advertises — an ``ImportError`` from a lazy export whose optional
    dependency is absent is the realistic case — and that would escape the
    message builder rather than being reported as a missing attribute.
    """
    try:
        return getattr(obj, name, None)
    except Exception:
        # Deliberately broad: this is a diagnostic, and a diagnostic that
        # raises replaces the error it was describing.
        return None


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

    Every attribute read goes through :func:`_peek`, because this runs *inside*
    an ``except`` handler: an exception escaping here would replace the
    ``DottedPathError`` the caller is owed with an unrelated one, chained
    behind "during handling of the above exception".
    """
    own_name = _peek(module, "__name__")
    public = [
        name
        for name in dir(module)
        if not name.startswith("_") and callable(_peek(module, name))
    ]
    defined_here = [
        name
        for name in public
        if _peek(_peek(module, name), "__module__") == own_name
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
        DottedPathError: The path is malformed (``malformed``), a module was
            not found (``module_not_found``), code ran and raised
            (``import_failed``), or the module has no such attribute
            (``attribute_not_found``). Branch on ``reason`` to tell an absent
            optional dependency from a broken one.

    Warning:
        Importing the module **executes** it — and so, for a PEP 562 lazy
        export, does reading the attribute. See the module docstring.
    """
    module_path, attribute = _split(ref)

    # The two clauses here, and the two at the attribute lookup below, are the
    # same classification made twice: `ModuleNotFoundError` means something is
    # **not installed** (an environment condition, and the one a caller's
    # `optional: true` may reasonably swallow), anything else means code was
    # found and **raised** (a defect, never safe to skip silently). Expressed
    # as except-clause dispatch rather than a shared helper because a helper
    # would have to receive `exc`, and an exception instance flowing into a
    # raise expression is the disclosure pattern the house error-text guard
    # exists to reject — correctly, since it cannot know what a callee does
    # with it. All four paths are pinned by tests.
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise DottedPathError(
            f"Cannot import module {module_path!r} from {ref!r} "
            f"({type(exc).__name__})",
            ref=ref,
            reason=DottedPathReason.MODULE_NOT_FOUND,
        ) from exc
    except Exception as exc:
        # `Exception`, not `ImportError`: the import runs the target module's
        # top level, so the failure can be anything that module raises. The
        # text stays on `__cause__` for the same reason. A plain `ImportError`
        # — a `from x import y` that failed inside the target — reaches here
        # rather than the clause above, which is right: the module began
        # executing.
        raise DottedPathError(
            f"Cannot import module {module_path!r} from {ref!r} "
            f"({type(exc).__name__})",
            ref=ref,
            reason=DottedPathReason.IMPORT_FAILED,
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
    except ModuleNotFoundError as exc:
        # A PEP 562 module-level `__getattr__` runs arbitrary code, so the
        # attribute lookup is a second execution point — and the standard use
        # of that hook is a *lazy export* that imports an optional dependency
        # on first access. `dataknobs_common.events:SqsEventBus` is one:
        # without `aioboto3` installed, this line raises `ModuleNotFoundError`,
        # not `AttributeError`.
        #
        # Catching only `AttributeError` let that escape as a raw exception, so
        # a caller's `except DottedPathError` did not match it and
        # `optional: true` did not cover the one case it most obviously should.
        raise DottedPathError(
            f"Module {module_path!r} raised resolving attribute "
            f"{attribute!r} (from {ref!r}) ({type(exc).__name__})",
            ref=ref,
            reason=DottedPathReason.MODULE_NOT_FOUND,
        ) from exc
    except Exception as exc:
        # The lazy export ran and failed for some reason other than a missing
        # module — present and broken, not absent.
        raise DottedPathError(
            f"Module {module_path!r} raised resolving attribute "
            f"{attribute!r} (from {ref!r}) ({type(exc).__name__})",
            ref=ref,
            reason=DottedPathReason.IMPORT_FAILED,
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


def dotted_path(target: Any) -> str:
    """Spell *target*'s dotted path, the reference ``resolve_dotted`` reads back.

    The inverse of :func:`resolve_dotted`, for the caller that has the object
    and needs the string: a test building a config block that names a class, a
    generator emitting a declaration, an error message quoting the reference a
    consumer would have to write.

    Args:
        target: Any object carrying ``__module__`` and ``__qualname__`` — a
            class or a module-level function.

    Returns:
        ``"module.path:name"``, the canonical form documented above.

    Raises:
        ValueError: *target* carries no ``__module__``/``__qualname__``; its
            module is ``__main__``, which names a different object in every
            process; its qualname is nested (``Outer.Inner``, or a closure's
            ``f.<locals>.g``), since ``resolve_dotted`` performs exactly one
            attribute lookup; or its qualname is a bracketed placeholder rather
            than a name (``<lambda>``, ``<listcomp>``), which no attribute
            lookup can resolve at all. Each of those would produce a string
            this family cannot read back, and failing here names the object
            while failing at resolution time would name only the string.

            What it does **not** check is that the module is importable under
            the name it reports. A class whose ``__module__`` was rewritten, or
            one defined in a module never placed in ``sys.modules`` under that
            name, is spelled without complaint and fails on read-back.

    Writing the path out by hand instead is the thing worth avoiding. A literal
    that disagrees with the object does not fail the way a typo does: it names
    a real module reachable under a second name, so the import succeeds and
    yields a *second* class object, equal in every respect except identity.
    Every ``isinstance`` against the locally imported one then fails, and every
    test that only checks behaviour keeps passing.
    """
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(
            f"cannot spell a dotted path for {target!r}: it carries "
            f"__module__={module!r} and __qualname__={qualname!r}"
        )
    if "." in qualname:
        raise ValueError(
            f"cannot spell a dotted path for {target!r}: __qualname__ "
            f"{qualname!r} is nested, and resolve_dotted performs exactly one "
            "attribute lookup. Move the target to module scope."
        )
    if "<" in qualname:
        # Reached only by a qualname with no dot in it, since the check above
        # already took every nested one — which in practice means a lambda
        # defined at module scope. CPython spells the unnamed with brackets
        # (`<lambda>`, `<listcomp>`, `<genexpr>`), and a bracketed placeholder
        # is not a name: no attribute of that spelling exists to look up, so
        # the dot check cannot stand in for this one.
        raise ValueError(
            f"cannot spell a dotted path for {target!r}: __qualname__ "
            f"{qualname!r} is not a name — it is what CPython writes for an "
            "object that never had one. Give it a def and a name at module "
            "scope."
        )
    if module == "__main__":
        raise ValueError(
            f"cannot spell a dotted path for {target!r}: it is defined in "
            "__main__, which no other process can import by that name."
        )
    return f"{module}:{qualname}"
