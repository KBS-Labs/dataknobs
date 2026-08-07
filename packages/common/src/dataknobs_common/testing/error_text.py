"""Source guard against unbounded text in error messages.

An error message built from a caught exception is only as bounded as the
exception is. When the ``except`` clause names a specific type the text is
predictable — ``ImportError`` yields module names, ``TypeError`` from a call
yields a signature mismatch. When it catches ``Exception``, the text comes from
whatever ran inside the ``try``, and if that includes consumer code — a
constructor, a module import, a callback — the message can carry anything the
consumer's dependencies put in *their* messages. Database and cache clients
routinely put the connection URL, credentials included, in theirs.

That matters because some of these error types are rendered at an HTTP
boundary. ``dataknobs_bots.api`` maps ``dataknobs_common.exceptions`` types to
statuses and decides per type whether the message is returned to the caller, so
a message assembled from an arbitrary third-party exception is a disclosure
channel that no single raise site looks like it is opening.

The fix at each site is to name what failed — an identifier the project already
had, plus ``type(exc).__name__`` — and let ``raise ... from exc`` carry the
original to the logs. This module is the guard that keeps the pattern from
coming back:

    ```python
    from dataknobs_common.testing import assert_no_broad_except_in_error_text

    def test_no_unbounded_text_in_configuration_errors():
        assert_no_broad_except_in_error_text(
            Path(__file__).parent.parent / "src",
            error_names=frozenset({"ConfigurationError", "ConfigError"}),
        )
    ```

It is a source scan, not a runtime check: the defect is a shape in the code,
and the runtime path that reaches any given site may need a live database to
trigger.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

__all__ = ["BroadExceptFinding", "assert_no_broad_except_in_error_text"]

#: ``except`` targets treated as unbounded. Anything narrower is assumed to
#: produce text the project can reason about.
_BROAD = frozenset({"Exception", "BaseException"})


class BroadExceptFinding(NamedTuple):
    """One flagged raise site."""

    path: Path
    lineno: int
    error_name: str
    exc_name: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.lineno}: {self.error_name} message interpolates "
            f"'{self.exc_name}' caught by a broad except"
        )


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """Whether this ``except`` catches ``Exception`` / ``BaseException``."""
    node = handler.type
    if node is None:  # bare `except:`
        return True
    names = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(isinstance(n, ast.Name) and n.id in _BROAD for n in names)


def _interpolates(node: ast.AST, name: str) -> bool:
    """Whether an f-string (or ``str()`` call) in ``node`` reads ``name``.

    Only formatted values count: ``f"failed: {exc}"`` interpolates, while
    ``f"failed ({type(exc).__name__})"`` does not, because the name appears
    inside a call whose result is a class name. That distinction is the whole
    point of the guard, so it is checked structurally rather than by looking
    for the identifier anywhere in the expression.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, ast.FormattedValue):
            continue
        value = sub.value
        if isinstance(value, ast.Name) and value.id == name:
            return True
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "str"
            and any(
                isinstance(a, ast.Name) and a.id == name for a in value.args
            )
        ):
            return True
    return False


def _scan_file(path: Path, error_names: frozenset[str]) -> list[BroadExceptFinding]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[BroadExceptFinding] = []

    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
        if not _handler_is_broad(handler) or handler.name is None:
            continue
        bound = handler.name

        for stmt in ast.walk(handler):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            call = stmt.exc
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            raised = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if raised not in error_names:
                continue
            if any(_interpolates(arg, bound) for arg in call.args):
                findings.append(
                    BroadExceptFinding(path, stmt.lineno, raised, bound)
                )

    return findings


def assert_no_broad_except_in_error_text(
    *roots: Path,
    error_names: Iterable[str],
    ignore: Iterable[str] = (),
) -> None:
    """Fail if a broad ``except`` feeds its exception into a named error's message.

    Args:
        *roots: Directories (or files) to scan for ``*.py``.
        error_names: Bare class names whose construction is guarded — for
            example ``{"ConfigurationError", "ConfigError"}``. Matched on the
            name at the raise site, so an aliased import is covered by listing
            the alias.
        ignore: ``"<path-suffix>:<lineno>"`` entries to exempt, for a site
            reviewed and judged bounded. Each needs a comment saying why.

    Raises:
        AssertionError: Listing every flagged site, so one run reports the
            whole surface rather than the first offender.
    """
    wanted = frozenset(error_names)
    exempt = frozenset(ignore)
    findings: list[BroadExceptFinding] = []

    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            for finding in _scan_file(path, wanted):
                key = f"{path.name}:{finding.lineno}"
                if key not in exempt:
                    findings.append(finding)

    if findings:
        listed = "\n".join(f"  {f}" for f in findings)
        raise AssertionError(
            f"{len(findings)} error message(s) built from a broadly-caught "
            f"exception:\n{listed}\n\n"
            "The text of an exception caught by `except Exception` comes from "
            "whatever ran in the `try`, including consumer code, so it can "
            "carry a connection URL or a credential. Name what failed plus "
            "`type(exc).__name__`, and let `raise ... from exc` carry the "
            "original to the logs."
        )
