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


class CauseReadFinding(NamedTuple):
    """One ``__cause__`` load the guard flagged."""

    path: Path
    lineno: int
    #: The enclosing function's qualified name, or ``"<module>"``.
    qualname: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno} in {self.qualname}"


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
    return [
        CauseReadFinding(path, node.lineno, owners.get(node, "<module>"))
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "__cause__"
        and isinstance(node.ctx, ast.Load)
        and not _reads_only_its_type(node, parents)
    ]


def _matches(finding: CauseReadFinding, entry: str) -> bool:
    file_part, _, function = entry.partition("::")
    posix = finding.path.as_posix()
    if not (posix == file_part or posix.endswith(f"/{file_part}")):
        return False
    return not function or finding.qualname == function


def assert_no_cause_read(*roots: Path, allowed: Iterable[str] = ()) -> None:
    """Fail if code under ``roots`` reads ``__cause__`` outside ``allowed``.

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
            f"{len(findings)} read(s) of `__cause__` outside the allowed set:\n{listed}\n\n"
            "A registry keeps a factory's error on `__cause__` because its text "
            "can carry the connection URL the factory was handed. To decide "
            "whether a cause may be shown, call "
            "`dataknobs_common.registry.surfaced_cause`, or build through "
            "`create_surfacing`; to name it in a bounded message, use "
            "`type(exc.__cause__).__name__`."
        )
