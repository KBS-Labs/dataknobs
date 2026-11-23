# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Source guard against reading an error's ``__cause__`` to decide what to show.

A :class:`~dataknobs_common.registry.PluginRegistry` bounds what a factory
raises -- ``"Failed to create plugin '<key>' (<type>)"`` -- because a factory
builds a backend from deployment config and its text can carry the connection
URL it was handed. The factory's error stays on ``__cause__``, for a log.

Code in front of the registry that reads ``__cause__`` back out and raises or
interpolates it undoes that bound, and choosing by the cause's *type* does not
help: a ``ValueError`` dataknobs wrote to say a summary memory has no LLM and
one a driver raised with a DSN in it are the same type. The one answer is
:func:`~dataknobs_common.registry.surfaced_cause`, which decides by who wrote
the text, and :func:`~dataknobs_common.registry.create_surfacing`, which calls
it. This guard keeps every other reader from appearing unnoticed:

    ```python
    from dataknobs_common.testing import assert_no_cause_read

    def test_nothing_reads_a_cause_but_the_registry_helpers():
        assert_no_cause_read(
            Path(__file__).parent.parent / "src",
            allowed={
                # Logs the cause server-side; never returned to a caller.
                "myapp/api/errors.py::log_error",
            },
        )
    ```

It flags **every** load of ``__cause__``, not only those in an ``except``
handler: a helper that takes the exception as a parameter reads it just the
same. One load is not flagged, because it reads no text: ``type(x.__cause__)``
as the whole of a ``type()`` call, which is how a bounded message names the
cause's type.

The same holds for the other ways to the same error, each flagged alike:

- ``__context__``, which holds the factory's error too, since a registry
  raises its bounded error inside the handler that caught it;
- the attribute named as a string, ``getattr(exc, "__cause__")`` -- any string
  constant that is exactly ``"__cause__"`` or ``"__context__"``;
- the :mod:`traceback` formatters, which render the whole chain as text:
  ``traceback.format_exception`` and its siblings, used through the module or
  imported by name. A logger's ``exc_info`` is not flagged: it writes to a
  log, not to a caller.

It is a source scan, not a runtime check, for the reason
:func:`~dataknobs_common.testing.assert_no_broad_except_in_error_text` is: the
defect is a shape in the code, and the path that reaches a given site may need
a live backend to trigger.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "CauseReadFinding",
    "assert_no_cause_read",
]

#: The attributes that hold the error a registry bounded.
_CHAIN = frozenset({"__cause__", "__context__"})

#: The :mod:`traceback` names that render an error's whole chain as text.
_CHAIN_FORMATTERS = frozenset(
    {
        "TracebackException",
        "format_exc",
        "format_exception",
        "format_exception_only",
        "print_exc",
        "print_exception",
    }
)


class CauseReadFinding(NamedTuple):
    """One read of an error's chain the guard flagged."""

    path: Path
    lineno: int
    #: The enclosing function's qualified name, or ``"<module>"``.
    qualname: str
    #: What was read: ``"__cause__"``, ``"__context__"``, a string naming
    #: one of them, or the ``traceback`` formatter used.
    read: str = "__cause__"

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno} in {self.qualname} ({self.read})"


def _qualnames(tree: ast.AST) -> dict[ast.AST, str]:
    """Every node's enclosing function (or class-qualified method), by node."""
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, scope: list[str], function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*scope, child.name], function)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = ".".join([*scope, child.name])
                visit(child, [*scope, child.name], name)
            else:
                owner[child] = function
                visit(child, scope, function)

    visit(tree, [], "<module>")
    return owner


def _reads_only_its_type(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    return (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id == "type"
        and len(parent.args) == 1
        and parent.args[0] is node
        and not parent.keywords
    )


def _scan_file(path: Path) -> list[CauseReadFinding]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - exercised via the public API
        raise AssertionError(
            f"{path}: could not be parsed, so the guard cannot vouch for it "
            f"({type(exc).__name__} at line {exc.lineno}). A guard that skips "
            "a file it cannot read reports green on whatever is inside it."
        ) from exc

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    owners = _qualnames(tree)
    findings: list[CauseReadFinding] = []

    def flag(node: ast.AST, read: str) -> None:
        lineno = getattr(node, "lineno", 0)
        findings.append(CauseReadFinding(path, lineno, owners.get(node, "<module>"), read))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _CHAIN
            and isinstance(node.ctx, ast.Load)
            and not _reads_only_its_type(node, parents)
        ):
            flag(node, node.attr)
        elif isinstance(node, ast.Constant) and node.value in _CHAIN:
            flag(node, repr(node.value))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in _CHAIN_FORMATTERS
            and isinstance(node.value, ast.Name)
            and node.value.id == "traceback"
        ):
            flag(node, f"traceback.{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "traceback":
            for alias in node.names:
                if alias.name in _CHAIN_FORMATTERS or alias.name == "*":
                    flag(node, f"traceback.{alias.name}")
    return findings


def _matches(finding: CauseReadFinding, entry: str) -> bool:
    file_part, _, function = entry.partition("::")
    posix = finding.path.as_posix()
    if not (posix == file_part or posix.endswith(f"/{file_part}")):
        return False
    return not function or finding.qualname == function


def assert_no_cause_read(*roots: Path, allowed: Iterable[str] = ()) -> None:
    """Fail if code under ``roots`` reads an error's chain outside ``allowed``.

    What counts as a read is in this module's docstring: ``__cause__`` or
    ``__context__``, either named as a string, or a :mod:`traceback`
    formatter.

    Args:
        *roots: Directories (or files) to scan for ``*.py``.
        allowed: Entries exempting a reviewed reader, each
            ``"<path-suffix>::<qualname>"`` for one function (``qualname`` as
            ``Class.method`` for a method, ``"<module>"`` for module level)
            or ``"<path-suffix>"`` for a whole file. The path suffix matches
            on a component boundary. Give each a comment saying what the
            reader does with the cause and why that shows no factory's text.
            An entry matching nothing is an error: an exemption whose site
            moved or was removed is a hole, and a silent one reads as a clean
            scan.

    Raises:
        AssertionError: Listing every flagged load, so one run reports the
            whole surface rather than the first offender.
    """
    exempt = frozenset(allowed)
    used: set[str] = set()
    findings: list[CauseReadFinding] = []

    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            for finding in _scan_file(path):
                matched = {entry for entry in exempt if _matches(finding, entry)}
                used |= matched
                if not matched:
                    findings.append(finding)

    if stale := sorted(exempt - used):
        listed = "\n".join(f"  {entry}" for entry in stale)
        raise AssertionError(
            f"{len(stale)} allowed entr(ies) matched no `__cause__` read:\n{listed}\n\n"
            "Either the reader was removed -- drop the entry -- or it moved and "
            "the exemption silently stopped covering it."
        )

    if findings:
        listed = "\n".join(f"  {f}" for f in findings)
        raise AssertionError(
            f"{len(findings)} read(s) of an error's chain outside the allowed set:\n"
            f"{listed}\n\n"
            "A registry keeps a factory's error on `__cause__` because its text "
            "can carry the connection URL the factory was handed. To decide "
            "whether a cause may be shown, call "
            "`dataknobs_common.registry.surfaced_cause`, or build through "
            "`create_surfacing`; to name it in a bounded message, use "
            "`type(exc.__cause__).__name__`."
        )
