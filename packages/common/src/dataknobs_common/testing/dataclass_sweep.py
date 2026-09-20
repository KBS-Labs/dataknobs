# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Sweep every dataclass a package defines, and build one of each.

A guard about a *class* of types has to find the class, not the instances
somebody thought to list. Three suites do that today --- the hashable-contract
censuses in ``dataknobs-common`` and ``dataknobs-data``, and the
relation-spelling population --- and this is the one walk they share, so a type
added to a package is visible to all of them without any of them having to
remember it.

**Bound to a root package rather than to one.** It lived in
``dataknobs-common``'s test directory and named that package in four places, so
the second package that needed it could only get it by copying --- which is how
a sweep acquires two spellings and then two answers. It is a testing construct
about *any* package's dataclasses, so it ships beside the other testing
constructs and takes its root as an argument.

**Why the TYPE_CHECKING imports are re-executed.** Under ``from __future__
import annotations`` a name imported only under ``if TYPE_CHECKING:`` --- the
shape ruff's TC rules produce --- is absent at runtime, and
:func:`typing.get_type_hints` raises ``NameError`` for it. Eleven types in
``dataknobs-common`` resolve only once those imports are replayed. Without them
a sweep does not report a defect; it silently stops being able to see one,
which is the failure mode a sweep is least allowed to have.

**Why a value is built rather than an annotation read.** A predicate over field
types cannot settle what a value does. ``AssertionHierarchy.relation`` is
annotated ``str | RelationType`` and ``RelationType`` is unhashable --- but the
type canonicalises the field in ``__post_init__``, so every instance holds the
id and hashes; a static predicate reports a working type as broken. In the
other direction a ``Sequence`` field is hashable holding a tuple and not
holding a list, and only a value settles which.

**Why fields the CALLER supplies get a hashable stand-in.** Where a field is a
Protocol, a ``Callable``, an ``Any`` or an unbounded ``TypeVar``, the value
comes from outside and its behaviour is the caller's to determine --- a
structure hashes exactly as far as what it holds does. Substituting a hashable
stand-in isolates the type's OWN declared structure, so a finding is always a
property of the type rather than of a value chosen for it.

That isolation is also this module's limit. A generic witness settles
*hashability*, because any conforming value decides it. It cannot settle
anything needing a semantically valid value --- two :class:`Supplied` stand-ins
compare unequal, so a roundtrip asserted over one of these instances would fail
on a field the sweep invented.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import decimal
import enum
import importlib
import inspect
import pathlib
import pkgutil
import types
import typing
import uuid
from collections import abc as cabc

__all__ = ["DataclassSweep", "Supplied", "UnbuildableError"]


class Supplied:
    """Stand-in for a value the caller supplies, chosen to be hashable."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "<supplied by the caller>"


class UnbuildableError(Exception):
    """No value of this annotation can be synthesised."""


_SCALARS: dict[type, object] = {
    str: "x",
    int: 1,
    float: 1.0,
    bool: True,
    bytes: b"x",
    complex: 1j,
    type(None): None,
    pathlib.Path: pathlib.Path("x"),
    uuid.UUID: uuid.UUID(int=0),
    decimal.Decimal: decimal.Decimal(1),
    datetime.datetime: datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
    datetime.date: datetime.date(2026, 1, 1),
    datetime.timedelta: datetime.timedelta(0),
}

_MAPPINGS = (dict, cabc.Mapping, cabc.MutableMapping)
_SEQUENCES = (
    list,
    cabc.Sequence,
    cabc.MutableSequence,
    cabc.Collection,
    cabc.Iterable,
    cabc.Container,
    cabc.Reversible,
)
_SETS = (set, cabc.Set, cabc.MutableSet)

#: ``collections.abc.Callable``, bound as the object it is at runtime.
#:
#: The identity check below runs after ``base`` is narrowed to ``type``, and a
#: type checker reads ``cabc.Callable`` there as a typing special form rather
#: than as a class -- so ``base is cabc.Callable`` is a non-overlapping
#: comparison to it and a true one at runtime. Binding it to an ``object``
#: says which of the two readings this line means.
_CALLABLE: object = cabc.Callable


class DataclassSweep:
    """Every dataclass one package defines, with a value built for each.

    Args:
        root: The package to walk. Its ``__path__`` is the tree and its
            ``__name__`` is the prefix keys are reported relative to.

    Example:
        ```python
        import dataknobs_data
        sweep = DataclassSweep(dataknobs_data)
        verdict, detail = sweep.probe_hashability(sweep.every_dataclass()["query.Query"])
        ```
    """

    def __init__(self, root: types.ModuleType) -> None:
        self._root = root
        self._prefix = root.__name__
        self._type_checking_names: dict[str, object] = {}
        self._import_failures: list[tuple[str, str]] = []
        self._found: dict[str, type] | None = None

    # -- the walk ---------------------------------------------------------

    def every_dataclass(self) -> dict[str, type]:
        """The dataclasses this package defines, keyed by their path below the root.

        Computed once per instance. The walk imports every module and replays
        its ``TYPE_CHECKING`` block, so repeating it per suite would be waste
        with a side effect rather than waste alone --- and the import-failure
        list would accumulate a duplicate entry per repeat, which a caller
        asserting the list is empty would never notice and one printing it
        would misread.
        """
        if self._found is not None:
            return self._found
        modules = self._walk()
        for module in modules:
            self._replay_type_checking_imports(module)
        found: dict[str, type] = {}
        for module in modules:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if dataclasses.is_dataclass(obj) and obj.__module__.startswith(self._prefix):
                    key = f"{obj.__module__}.{obj.__qualname__}".removeprefix(f"{self._prefix}.")
                    found[key] = obj
        self._found = found
        return found

    def import_failures(self) -> list[tuple[str, str]]:
        """Modules :meth:`every_dataclass` could not import, and why.

        A sweep that silently skipped a module would report a clean tree, so
        the holes are returned rather than swallowed and a caller asserts on
        them.
        """
        self.every_dataclass()
        return list(self._import_failures)

    def _walk(self) -> list[types.ModuleType]:
        """Every module in the package, with import failures recorded rather than swallowed."""
        modules = [self._root]
        for info in pkgutil.walk_packages(self._root.__path__, prefix=f"{self._prefix}."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as exc:
                self._import_failures.append((info.name, f"{type(exc).__name__}: {exc}"))
        return modules

    def _replay_type_checking_imports(self, module: types.ModuleType) -> None:
        """Execute the module's ``if TYPE_CHECKING:`` imports into the shared namespace."""
        try:
            source = inspect.getsource(module)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return
        # Seeded with the module's own import context, so a **relative**
        # TYPE_CHECKING import replays. `exec` against a bare namespace has no
        # `__package__`, so `from .vector.types import DistanceMetric` raised
        # `ImportError` and was skipped -- and the names it would have bound
        # stayed missing, which turns up later as `NameError` from
        # `get_type_hints` and reads as an unconstructible type rather than as
        # a hole in the replay. `dataknobs-common` writes most of these
        # absolutely and barely noticed; a package that writes them relatively
        # loses every one.
        namespace: dict[str, object] = {
            "__name__": module.__name__,
            "__package__": getattr(module, "__package__", ""),
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            guarded = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if not guarded:
                continue
            for statement in node.body:
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    try:
                        exec(
                            compile(ast.Module([statement], []), "<type-checking>", "exec"),
                            namespace,
                        )
                    except Exception:
                        continue
        self._type_checking_names.update(namespace)

    # -- building a value -------------------------------------------------

    def _hints(self, cls: type) -> dict[str, object]:
        """Resolved annotations, with TYPE_CHECKING-only names supplied additively.

        ``localns`` rather than ``globalns``: :func:`typing.get_type_hints`
        walks the MRO and an explicit ``globalns`` would override every base's
        own module, which resolves the subclass and breaks the bases.
        """
        return typing.get_type_hints(cls, localns=self._type_checking_names)

    def _witness(self, annotation: object, depth: int = 0) -> object:
        """A value the annotation permits, preferring one that is not hashable."""
        if depth > 6:
            raise UnbuildableError("annotation nests too deeply")
        if annotation is typing.Any or annotation is object:
            return Supplied()
        if isinstance(annotation, typing.TypeVar):
            bound = annotation.__bound__
            if isinstance(bound, typing.ForwardRef):
                bound = self._type_checking_names.get(bound.__forward_arg__)
            if bound is not None:
                return self._witness(bound, depth + 1)
            if annotation.__constraints__:
                return self._witness(annotation.__constraints__[0], depth + 1)
            return Supplied()
        if isinstance(annotation, (str, typing.ForwardRef)):
            raise UnbuildableError(f"unresolved annotation {annotation!r}")

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is typing.Literal:
            return args[0]
        if origin in (typing.Union, types.UnionType):
            return self._union_witness(args, depth)

        base = origin if origin is not None else annotation
        if base in _SCALARS:
            return _SCALARS[base]
        if base in _MAPPINGS:
            return self._mapping_witness(args, depth)
        if base in _SEQUENCES:
            return self._sequence_witness(args, depth)
        if base in _SETS:
            return set()
        if base is frozenset:
            return frozenset()
        if base is tuple:
            return self._tuple_witness(args, depth)
        if not isinstance(base, type):
            raise UnbuildableError(f"unrecognised annotation {annotation!r}")
        if issubclass(base, enum.Enum):
            return next(iter(base))
        if base is cabc.Hashable:
            return "x"
        if base is _CALLABLE or getattr(base, "_is_protocol", False) or inspect.isabstract(base):
            return Supplied()
        if dataclasses.is_dataclass(base):
            return self.construct(base, depth + 1)
        raise UnbuildableError(f"no witness for {base.__name__}")

    def _union_witness(self, args: tuple[object, ...], depth: int) -> object:
        """The union member most likely to be unhashable, since a caller may pass it."""
        fallback: object = None
        found = False
        for arg in args:
            if arg is type(None):
                continue
            try:
                value = self._witness(arg, depth + 1)
            except UnbuildableError:
                continue
            try:
                hash(value)
            except TypeError:
                return value
            if not found:
                fallback, found = value, True
        if found:
            return fallback
        if type(None) in args:
            return None
        raise UnbuildableError("no member of the union could be built")

    def _mapping_witness(self, args: tuple[object, ...], depth: int) -> dict[object, object]:
        """A one-entry dict where the element types allow it, an empty one otherwise."""
        if args:
            try:
                return {self._witness(args[0], depth + 1): self._witness(args[1], depth + 1)}
            except UnbuildableError:
                pass
        return {}

    def _sequence_witness(self, args: tuple[object, ...], depth: int) -> list[object]:
        """A one-element list where the element type allows it, an empty one otherwise."""
        if args:
            try:
                return [self._witness(args[0], depth + 1)]
            except UnbuildableError:
                pass
        return []

    def _tuple_witness(self, args: tuple[object, ...], depth: int) -> tuple[object, ...]:
        """A tuple of the declared arity, non-empty for the variadic form."""
        if not args:
            return ()
        if args[-1] is Ellipsis:
            try:
                return (self._witness(args[0], depth + 1),)
            except UnbuildableError:
                return ()
        return tuple(self._witness(arg, depth + 1) for arg in args)

    # -- the two questions a suite asks -----------------------------------

    def construct(self, cls: type, depth: int = 0) -> object:
        """An instance built from the type's own declared field types."""
        hints = self._hints(cls)
        kwargs: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            if not field.init:
                continue
            has_default = (
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
            )
            try:
                kwargs[field.name] = self._witness(hints.get(field.name, field.type), depth + 1)
            except UnbuildableError:
                if not has_default:
                    raise
        try:
            return cls(**kwargs)
        except TypeError:
            return cls(*kwargs.values())  # a hand-written __init__ taking *args

    def probe_hashability(self, cls: type) -> tuple[str, str]:
        """``("hashes" | "raises" | "unconstructible", detail)`` for one type."""
        try:
            instance = self.construct(cls)
        except UnbuildableError as exc:
            return "unconstructible", str(exc)
        except Exception as exc:
            return "unconstructible", f"{type(exc).__name__}: {exc}"
        try:
            hash(instance)
        except TypeError as exc:
            return "raises", str(exc)
        return "hashes", ""
