"""Source guard against unbounded text in error messages.

An error message built from a caught exception is only as bounded as the
exception is. When the ``except`` clause names a specific type the text is
predictable — an ``AttributeError`` from ``getattr`` yields a module and an
attribute name, a ``TypeError`` from a call yields a signature mismatch. When
it catches ``Exception``, the text comes from
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

"Predictable" is not the same as "safe", so a narrow clause is not
automatically bounded. ``ImportError`` is the case: its text reads ``cannot
import name 'X' from 'pkg' (/abs/path/site-packages/pkg/__init__.py)`` — an
absolute filesystem path, which is exactly what ``dataknobs_config`` withholds
from a not-found error on the grounds that it doubles as a map of the server's
filesystem. It is in the default set for that reason, and because the reason is
a property of the exception type rather than of the package catching it: an
opt-in every caller has to remember is a guard that narrows quietly.

``unbounded_types=`` names the set, and **replaces** the default rather than
extending it — union the default in when adding a type.

It is a source scan, not a runtime check: the defect is a shape in the code,
and the runtime path that reaches any given site may need a live database to
trigger.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "GUARDED_ERROR_NAMES",
    "BroadExceptFinding",
    "assert_no_broad_except_in_error_text",
]

#: ``except`` targets treated as unbounded, for either of two reasons.
#:
#: ``Exception`` and ``BaseException`` are unbounded because they are *broad*:
#: the text comes from whatever ran in the ``try``, which can be consumer code.
#:
#: ``ImportError`` is unbounded despite being narrow, because of what its text
#: says — ``cannot import name 'X' from 'pkg' (/abs/path/site-packages/...)``
#: carries an absolute filesystem path. Anything else narrow is assumed to
#: produce text the project can reason about.
_UNBOUNDED = frozenset({"Exception", "BaseException", "ImportError"})


def _shared_error_names() -> frozenset[str]:
    """Every exception class name in :mod:`dataknobs_common.exceptions`.

    Derived rather than listed so a new shared exception is guarded the day it
    is added. A hand-maintained set is a guard that quietly narrows: the class
    lands, no test mentions it, and the scan keeps reporting green over a type
    it was never told to look at.
    """
    from dataknobs_common import exceptions

    return frozenset(
        name
        for name, obj in vars(exceptions).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseException)
        and obj.__module__ == exceptions.__name__
    )


#: The default set for :func:`assert_no_broad_except_in_error_text`. A package
#: unions its own aliases and subclasses onto this — the raise site is matched
#: on the bare name, so ``ConfigError`` must be named even though it *is* a
#: ``ConfigurationError``.
GUARDED_ERROR_NAMES = _shared_error_names()


class BroadExceptFinding(NamedTuple):
    """One flagged raise site."""

    path: Path
    lineno: int
    error_name: str
    exc_name: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.lineno}: {self.error_name} message interpolates "
            f"'{self.exc_name}', caught by an except whose text is unbounded"
        )


def _handler_is_broad(
    handler: ast.ExceptHandler, unbounded: frozenset[str]
) -> bool:
    """Whether this ``except`` catches something with unbounded text.

    A bare ``except:`` is broader than any named type and answers ``True``, but
    binds no name — so there is no identifier for a message to read, and the
    caller skips it. That is a property of the syntax, not a gap here.
    """
    node = handler.type
    if node is None:  # bare `except:`
        return True
    names = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(
        (isinstance(n, ast.Name) and n.id in unbounded)
        or (isinstance(n, ast.Attribute) and n.attr in unbounded)
        for n in names
    )


def _safe_reads(node: ast.AST) -> set[int]:
    """Identify reads of a caught exception that cannot disclose its text.

    Returns the ``id()`` of every :class:`ast.Name` node appearing in a
    construct whose result is a class name or a bool — ``type(exc).__name__``,
    ``exc.__class__.__name__``, ``isinstance(exc, X)``. Everything else is
    treated as a disclosure, which is what makes this fail closed: a shape
    nobody anticipated is flagged rather than missed.

    Identity rather than name, so ``isinstance(exc, X) and str(exc)`` marks
    only the first read safe.
    """
    safe: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "__name__":
            inner = sub.value
            # type(exc).__name__
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "type"
            ):
                safe.update(id(a) for a in inner.args if isinstance(a, ast.Name))
            # exc.__class__.__name__
            elif isinstance(inner, ast.Attribute) and inner.attr == "__class__":
                if isinstance(inner.value, ast.Name):
                    safe.add(id(inner.value))
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "isinstance"
        ):
            safe.update(id(a) for a in sub.args if isinstance(a, ast.Name))
    return safe


def _reads_unsafely(node: ast.AST, tainted: frozenset[str]) -> bool:
    """Whether ``node`` reads any tainted name other than as a class name."""
    safe = _safe_reads(node)
    return any(
        isinstance(sub, ast.Name)
        and sub.id in tainted
        and isinstance(sub.ctx, ast.Load)
        and id(sub) not in safe
        for sub in ast.walk(node)
    )


def _assigned_names(node: ast.stmt) -> list[str]:
    """The plain names a statement binds, ignoring attribute/subscript targets."""
    if isinstance(node, ast.Assign):
        targets: list[ast.expr] = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    else:
        return []
    return [
        sub.id
        for target in targets
        for sub in ast.walk(target)
        if isinstance(sub, ast.Name)
    ]


def _tainted_names(handler: ast.ExceptHandler, bound: str) -> frozenset[str]:
    """Names carrying the caught exception's text, to a fixpoint.

    ``msg = f"failed: {exc}"`` puts the text in ``msg``; a later
    ``raise X(msg)`` discloses exactly as much as interpolating at the raise
    would. Three of the sites this guard was written to protect were that
    shape, so tracking it is the difference between a guard and a decoration.

    Order is not modelled — a name assigned unsafely anywhere in the handler
    is tainted throughout it. That over-approximates when a name is reassigned
    to something bounded before the raise, and over-approximating is the right
    direction for a guard; ``ignore=`` covers the rare case.
    """
    tainted = {bound}
    assignments = [
        node
        for node in ast.walk(handler)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
    ]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            if node.value is None:  # a bare `x: int` annotation
                continue
            names = [n for n in _assigned_names(node) if n not in tainted]
            if names and _reads_unsafely(node.value, frozenset(tainted)):
                tainted.update(names)
                changed = True
    return frozenset(tainted)


def _raised_name(call: ast.Call) -> str | None:
    """The bare class name a ``raise X(...)`` constructs, if it is one."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _scan_file(
    path: Path, error_names: frozenset[str], unbounded: frozenset[str]
) -> list[BroadExceptFinding]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - exercised via the public API
        raise AssertionError(
            f"{path}: could not be parsed, so the guard cannot vouch for it "
            f"({type(exc).__name__} at line {exc.lineno}). A guard that skips "
            "a file it cannot read reports green on whatever is inside it."
        ) from exc

    findings: list[BroadExceptFinding] = []

    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
        if not _handler_is_broad(handler, unbounded) or handler.name is None:
            continue
        bound = handler.name
        tainted = _tainted_names(handler, bound)

        for stmt in ast.walk(handler):
            if not isinstance(stmt, ast.Raise) or stmt.exc is None:
                continue
            call = stmt.exc
            if not isinstance(call, ast.Call):
                continue
            if _raised_name(call) not in error_names:
                continue
            # `raise X(...) from exc` is the prescribed fix, so the cause is
            # deliberately not scanned — only what the constructor receives.
            operands: list[ast.AST] = [*call.args]
            operands.extend(kw.value for kw in call.keywords)
            if any(_reads_unsafely(operand, tainted) for operand in operands):
                findings.append(
                    BroadExceptFinding(
                        path, stmt.lineno, _raised_name(call) or "?", bound
                    )
                )

    return findings


def assert_no_broad_except_in_error_text(
    *roots: Path,
    error_names: Iterable[str],
    ignore: Iterable[str] = (),
    unbounded_types: Iterable[str] = _UNBOUNDED,
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
            Matched on a path-component boundary, so ``"base.py:120"`` exempts
            that line in *every* ``base.py`` under the roots while
            ``"pkg/base.py:120"`` exempts one — give as much path as you mean.
            An entry matching nothing is an error: a suppression whose site
            moved is a hole, and a silent one reads as a clean scan.
        unbounded_types: ``except`` targets whose text is treated as
            unbounded. Defaults to ``Exception``, ``BaseException`` and
            ``ImportError``. **Replaces** the default rather than extending
            it, so union it in when adding a type.

    Raises:
        AssertionError: Listing every flagged site, so one run reports the
            whole surface rather than the first offender.
    """
    wanted = frozenset(error_names)
    exempt = frozenset(ignore)
    treat_as_unbounded = frozenset(unbounded_types)
    used: set[str] = set()
    findings: list[BroadExceptFinding] = []

    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            for finding in _scan_file(path, wanted, treat_as_unbounded):
                key = f"{path.as_posix()}:{finding.lineno}"
                matched = {
                    entry
                    for entry in exempt
                    if key == entry or key.endswith(f"/{entry}")
                }
                used |= matched
                if not matched:
                    findings.append(finding)

    if stale := sorted(exempt - used):
        listed = "\n".join(f"  {entry}" for entry in stale)
        raise AssertionError(
            f"{len(stale)} ignore entr(ies) matched no flagged site:\n{listed}\n\n"
            "Either the site was fixed — drop the entry — or it moved and the "
            "suppression silently stopped covering it."
        )

    if findings:
        listed = "\n".join(f"  {f}" for f in findings)
        raise AssertionError(
            f"{len(findings)} error message(s) built from an exception whose "
            f"text is unbounded:\n{listed}\n\n"
            "Under `except Exception` the text comes from whatever ran in the "
            "`try`, including consumer code, so it can carry a connection URL "
            "or a credential. Under `except ImportError` it carries an "
            "absolute filesystem path. Name what failed plus "
            "`type(exc).__name__`, and let `raise ... from exc` carry the "
            "original to the logs."
        )
