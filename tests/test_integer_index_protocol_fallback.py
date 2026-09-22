# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""No shipped class falls into Python's legacy integer-indexing protocol.

A type that defines ``__getitem__`` and no ``__iter__`` is *still iterable*.
Python falls back to the protocol that predates ``__iter__``: ask for index
``0``, then ``1``, and stop at ``IndexError``. On a class whose ``__getitem__``
forwards to a string-keyed dict, the first question is ``obj[0]`` and the
answer is ``KeyError: 0``.

That is the whole defect, and every symptom of it is a face of the same
sentence:

- ``for key in obj`` and ``list(obj)`` raise ``KeyError: 0``
- ``dict(obj)`` and ``{**obj}`` raise ``KeyError: 0`` or ``TypeError``
- ``key in obj`` raises ``KeyError: 0`` too, unless the class also has
  ``__contains__`` --- in which case the membership test works and the trap
  stays hidden until something iterates

The error names a key the caller never wrote, from a line that never
mentioned an integer, so it reads as a bug in the caller's data. It was found
three times in this workspace before anyone wrote it down:
``StateDataWrapper`` (fsm), ``ResolvedConfig`` and ``BotContext`` (bots) ---
independently written, years apart, by way of the same reasonable-looking
step of adding ``__getitem__`` to a class that holds a dict.

**Why the check is ``__iter__`` and not "the whole mapping surface".** A
missing ``__len__`` is a loud, correct ``TypeError: object of type 'X' has no
len()``: nothing is silently wrong, and the caller is told exactly what is
absent. A missing ``__contains__`` is only a trap when ``__iter__`` is
*also* missing, because ``in`` prefers iteration over integer indexing. So
``__iter__`` is not one symptom among several --- it is the single condition
under which the interpreter starts inventing integer keys, and a guard keyed
to anything wider would report classes that are merely incomplete rather than
classes that are wrong.

It also keeps the guard honest about the population it is scanning. A
deliberately non-mapping ``__getitem__`` is common and fine ---
``PreserveUndefined`` in the llm package subclasses ``jinja2.Undefined``,
whose ``__getitem__`` raises for every key --- and it passes here without an
exemption because ``Undefined`` supplies ``__iter__``. A wider predicate would
have needed an allowlist entry for it, and an allowlist that has to hold the
correct cases is one nobody can read.

There is no allowlist. The finding is zero across the workspace, so an
exemption list would be an empty declaration whose only effect is to rot. A
class that genuinely wants the legacy protocol --- an integer-indexed
sequence relying on it deliberately --- takes a line-level
``# legacy-index-protocol: <reason>`` marker on its ``class`` statement,
which travels with the code rather than with a spelling of its name, and
which :func:`test_the_marker_is_not_satisfied_by_a_bare_comment` requires to
carry a reason.

This guard reads the *runtime* type rather than the AST, because the answer
depends on the MRO: ``class X(Mapping)`` inherits everything and a class
statement cannot be read for that. Importing the workspace to ask is what
makes the answer correct, and it costs about a second and a half.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import Any
from pathlib import PurePosixPath

import pytest

from tests._workspace import ROOT, tracked_python_files

#: The marker a deliberately integer-indexed class carries, on its ``class``
#: line. Nothing carries it today; see the module docstring for why it exists
#: unused rather than being added when first needed.
MARKER = "# legacy-index-protocol:"


def _workspace_modules() -> tuple[str, ...]:
    """Every importable module under ``packages/*/src``, as a dotted name.

    Tracked *and* present. A file git still tracks but the working tree has
    deleted is an ordinary in-progress state, and other guards own the
    bookkeeping question; this one only wants something it can import. The
    filter was added after a deleted module imported anyway, out of a stale
    ``__pycache__`` entry left behind by an earlier run --- so the scan was
    reading bytecode for a file that no longer exists and reporting green.
    """
    names = []
    for tracked in tracked_python_files():
        path = PurePosixPath(tracked)
        parts = path.parts
        if len(parts) < 4 or parts[0] != "packages" or parts[2] != "src":
            continue
        if not (ROOT / tracked).exists():
            continue
        dotted = list(parts[3:-1]) + [path.stem]
        if dotted[-1] == "__init__":
            dotted.pop()
        if dotted:
            names.append(".".join(dotted))
    assert names, "no package modules found — has the enumeration broken?"
    return tuple(sorted(set(names)))


def _shipped_top_level() -> frozenset[str]:
    """The import names the workspace owns, so third-party classes are skipped."""
    return frozenset(
        path.name
        for path in ROOT.glob("packages/*/src/*")
        if path.is_dir() and (path / "__init__.py").exists()
    )


def _shipped_classes() -> list[type]:
    """Every class the workspace defines, imported and deduplicated.

    Module-level classes and one level of nesting. Deeper nesting is not
    scanned and has never held one of these; a class nested two deep is not
    reachable by a consumer without going through the one above it.
    """
    owned = _shipped_top_level()
    seen: dict[int, type] = {}

    def consider(obj: object) -> None:
        if not inspect.isclass(obj):
            return
        if getattr(obj, "__module__", "").split(".")[0] not in owned:
            return
        if id(obj) in seen:
            return
        seen[id(obj)] = obj
        for nested in vars(obj).values():
            consider(nested)

    for name in _workspace_modules():
        for member in vars(importlib.import_module(name)).values():
            consider(member)

    assert len(seen) > 500, f"only {len(seen)} classes found — has the scan broken?"
    return list(seen.values())


def falls_back_to_integer_indexing(cls: type) -> bool:
    """Whether ``cls`` answers a subscript but leaves iteration to the interpreter."""
    return getattr(cls, "__getitem__", None) is not None and getattr(cls, "__iter__", None) is None


def _declaration_of(cls: type) -> str:
    """The ``class`` statement's own line, where a marker would sit."""
    try:
        source, _ = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return ""
    for line in source:
        if line.lstrip().startswith(("class ", "@")):
            if line.lstrip().startswith("class "):
                return line
    return source[0] if source else ""


def _marker_reason(cls: type) -> str:
    declaration = _declaration_of(cls)
    if MARKER not in declaration:
        return ""
    return declaration.split(MARKER, 1)[1].strip()


def test_no_shipped_class_falls_back_to_integer_indexing() -> None:
    """``__getitem__`` without ``__iter__`` makes the interpreter invent keys."""
    findings = [
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in _shipped_classes()
        if falls_back_to_integer_indexing(cls) and not _marker_reason(cls)
    ]

    assert not findings, (
        "these classes answer obj['key'] but leave iteration to the legacy "
        "integer-indexing protocol, so iterating one asks it for obj[0]:\n  "
        + "\n  ".join(sorted(findings))
        + f"\n\nDefine __iter__ (and __len__), or declare the ABC that supplies "
        f"them. A class that wants the legacy protocol deliberately takes "
        f"'{MARKER} <reason>' on its class line."
    )


def test_the_detector_reports_a_class_that_has_the_defect() -> None:
    """A positive control: a census that can find nothing always passes.

    Both halves, because a predicate stuck on ``False`` and a predicate stuck
    on ``True`` produce the same green run for opposite reasons.
    """

    class StringKeyed:
        def __getitem__(self, key: str) -> int:
            return {"a": 1}[key]

    class Complete(Mapping):
        def __getitem__(self, key: str) -> int:
            return {"a": 1}[key]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter({"a": 1})

        def __len__(self) -> int:
            return 1

    assert falls_back_to_integer_indexing(StringKeyed)
    assert not falls_back_to_integer_indexing(Complete)

    # Reached dynamically because the static and runtime answers differ, and
    # that difference is the subject: mypy reports `StringKeyed` as matching
    # no overload of `list`, which is true of the type and false of the
    # interpreter --- it iterates it anyway, through the protocol below.
    defective: Any = StringKeyed()
    with pytest.raises(KeyError, match="0"):
        list(defective)
    assert list(Complete()) == ["a"]


def test_the_marker_is_not_satisfied_by_a_bare_comment() -> None:
    """A marker with nothing after it exempts nothing.

    The reason is the part that survives review; a bare marker is a way to
    turn the guard off one class at a time without saying anything.
    """

    class Bare:  # legacy-index-protocol:
        def __getitem__(self, key: str) -> int:
            return 0

    class Reasoned:  # legacy-index-protocol: integer-indexed by design
        def __getitem__(self, index: int) -> int:
            return index

    assert _marker_reason(Bare) == ""
    assert _marker_reason(Reasoned) == "integer-indexed by design"


def test_the_scan_reaches_every_package() -> None:
    """The enumeration covers the workspace, not whichever package imports first."""
    modules = _workspace_modules()
    packages = {name.split(".")[0] for name in modules}

    assert packages == _shipped_top_level(), sorted(packages ^ _shipped_top_level())
