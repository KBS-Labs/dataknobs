"""Reproduce-first guard for a consumer callback dispatched without judgement.

A method that takes a callback from its caller and calls it inside an ``async
def`` has to decide whether the result needs awaiting. There are two ways to
get that wrong, and this repository held both:

**Asking the wrong question.** ``asyncio.iscoroutinefunction`` answers for
*functions* and reports a callable object whose ``__call__`` is an ``async
def`` as synchronous --- the shape anything stateful takes. Ten sites in the
vector package were fixed by adopting ``is_async_callable``.

**Asking no question at all.** The commoner spelling by far: call the
callback and move on. Eleven such sites were found in ``packages/data/src``
alone, including two whose *return value* was used --- a transform yielding a
coroutine object into the caller's stream, and a filter predicate answering
``True`` unconditionally, because a coroutine is truthy.

The second shape is why this guard is written against the AST rather than
against the token ``iscoroutinefunction``. A census keyed to that token can
only find sites that already reached for the right idea and got it slightly
wrong; it is structurally blind to a site that never asked, which is the
larger population and the one that produces the worse failures. The first
version of this check was keyed to the token, scanned one subpackage, and
reported green over three unchecked dispatches in a file it was reading.

There is no allowlist. Both findings are zero in scope, so an exemption
would be an empty declaration whose only effect is to rot --- and the one
entry the token-keyed version *did* carry turned out to rest on a reason that
did not survive reading the constructor it was about. A site that genuinely
needs an exception takes a line-level ``# async-dispatch-exempt: <reason>``
marker, which travels with the line rather than with a spelling of its
argument, and which cannot be satisfied by a bare marker.

Related runtime proof: ``packages/data/tests/test_consumer_callback_dispatch.py``
drives every dispatch this guard covers with a callable object and asserts
the callback actually ran.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tests._workspace import ROOT, rel, tracked_python_files

#: The scanned scope, as directory prefixes.
#:
#: One package rather than one subpackage: the first version of this check
#: covered ``dataknobs_data/vector/`` and so reported green over sites in
#: ``database.py``, ``streaming.py`` and ``migration/`` that hold the same
#: defect. A scope narrower than the unit a contributor edits is a scope that
#: certifies the region already cleaned.
#:
#: Widening is this constant plus the work to bring the added package to zero
#: --- which is the whole cost, and it is not small. The remaining packages
#: measure 37 sites of the first shape and 12 of the second, and they are not
#: a backlog of oversights: each is a published dispatch whose behaviour
#: changes when it starts awaiting, so each needs its own reproduce-first
#: proof. Several are tracked already, at high severity, because the failure
#: is not a dropped notification but a circuit breaker recording a success it
#: never executed and a streaming sink dropping records it has already
#: counted. Adding a package here without doing that work turns a guard that
#: passes into a guard that is disabled.
SCOPE = ("packages/data/src",)

#: Enough scanned files that an empty finding list means "clean" rather than
#: "matched nothing". Set well under the real count (95 when this was
#: written) so ordinary growth and deletion do not move it, but far enough
#: above zero that a scope expression resolving to nothing fails instead of
#: passing. This is the only anti-vacuity anchor the guard has: with no
#: allowlist there is no declaration whose staleness would otherwise catch a
#: scope that stopped matching.
MINIMUM_FILES_SCANNED = 70

#: A line-level opt-out. The reason is mandatory --- a bare marker would make
#: this an escape hatch from the guard rather than a documented exception to
#: it, which is the failure mode the allowlist it replaced actually had.
EXEMPT = "# async-dispatch-exempt:"

#: Names that answer "is this async?", however they were imported. Matched on
#: the attribute or function name, so ``asyncio.iscoroutinefunction``,
#: ``inspect.iscoroutinefunction`` and a bare ``iscoroutinefunction`` from an
#: ``as``-aliased import all resolve the same way.
CLASSIFIERS = frozenset({"iscoroutinefunction", "is_async_callable", "isawaitable", "iscoroutine"})

#: Calls that hand a callable somewhere else to be run, rather than running it
#: here. The callable is not dispatched at this site, so this site owes no
#: judgement about it.
DELEGATORS = frozenset({"to_thread", "run_in_executor", "run_callback", "submit"})


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _scanned_files() -> list[Path]:
    """Every tracked ``*.py`` under a declared scope prefix."""
    return [
        ROOT / name
        for name in tracked_python_files()
        if any(name.startswith(f"{prefix}/") for prefix in SCOPE)
    ]


def _call_name(node: ast.Call) -> str:
    """The callee's own name, ignoring whatever module it hangs off."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _callback_parameters(fn: ast.AsyncFunctionDef) -> set[str]:
    """Parameters annotated as something callable.

    Read off the annotation rather than resolved, for the reason the file
    handle guard gives for matching ``.open`` on the attribute name: proving
    what a parameter really holds means reimplementing type inference to
    answer a question a reader settles by looking. Over-reporting is the safe
    direction --- a false finding is a line-level marker with a reason, and a
    missed one is the defect shipping again.
    """
    args = fn.args
    found = set()
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is None or arg.annotation is None:
            continue
        annotation = ast.unparse(arg.annotation)
        if "Callable" in annotation or "Awaitable" in annotation:
            found.add(arg.arg)
    return found


def _judged_names(fn: ast.AsyncFunctionDef) -> set[str]:
    """Names this function decides the async-ness of, by either route.

    Two routes, because both are legitimate and the repository uses each
    where it is the only one that works:

    - **Before the call**, ``is_async_callable(cb)``. The only form available
      to a caller whose sync branch *offloads*, since there is no offloading
      a call already made.
    - **After the call**, assigning the result and asking
      ``isawaitable(result)``. Strictly more robust --- it also catches a
      plain ``def`` that returns a coroutine --- and what ``run_callback``
      does once so that call sites do not have to.
    """
    judged: set[str] = set()
    assigned_from: dict[str, str] = {}

    for node in ast.walk(fn):
        # `result = cb(...)` --- remember which callback `result` came from.
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value, targets = node.value, []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif node.target is not None:
                targets = [node.target]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                for target in targets:
                    if isinstance(target, ast.Name):
                        assigned_from[target.id] = value.func.id

    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or _call_name(node) not in CLASSIFIERS:
            continue
        for argument in node.args:
            if isinstance(argument, ast.Name):
                judged.add(argument.id)
                # `isawaitable(result)` judges whatever produced `result`.
                judged.add(assigned_from.get(argument.id, argument.id))
            elif isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name):
                judged.add(argument.value.id)
    return judged


def _findings(source: str, path: Path) -> list[Finding]:
    """Both shapes, from one parse and one traversal.

    One function rather than two, so the two tests below cannot disagree
    about what they are reading. The token-keyed version this replaces had
    the scan written twice, with subtly different traversals in each copy.
    """
    tree = ast.parse(source, str(path))
    lines = source.splitlines()
    name = rel(path)

    def exempt(lineno: int) -> bool:
        """The marker on the line, or in the comment block directly above it.

        Same-line only --- the ``# sweep-exempt:`` idiom --- does not fit
        here. A reason worth accepting is a sentence about who supplies the
        callback, and a sentence does not go beside a call and stay under the
        line limit; forcing it to would select for reasons short enough to be
        useless. Contiguous comment lines only, so the marker cannot drift
        away from what it exempts.
        """
        if EXEMPT in lines[lineno - 1]:
            return True
        for above in range(lineno - 2, -1, -1):
            stripped = lines[above].strip()
            if not stripped.startswith("#"):
                return False
            if EXEMPT in stripped:
                return True
        return False

    found: list[Finding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "iscoroutinefunction":
            if not exempt(node.lineno):
                found.append(
                    Finding(
                        name,
                        node.lineno,
                        f"`{ast.unparse(node.func)}(...)` misreads a callable object as "
                        "synchronous — use `is_async_callable`, or `run_callback` if the "
                        "sync branch just calls inline",
                    )
                )

    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        callbacks = _callback_parameters(fn)
        if not callbacks:
            continue
        judged = _judged_names(fn)
        awaited = {id(node.value) for node in ast.walk(fn) if isinstance(node, ast.Await)}
        delegated = {
            argument.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and _call_name(node) in DELEGATORS
            for argument in node.args
            if isinstance(argument, ast.Name)
        }
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or id(node) in awaited:
                continue
            if not isinstance(node.func, ast.Name):
                continue
            callback = node.func.id
            if callback not in callbacks or callback in judged or callback in delegated:
                continue
            if exempt(node.lineno):
                continue
            found.append(
                Finding(
                    name,
                    node.lineno,
                    f"`{fn.name}` calls its `{callback}` callback without deciding whether "
                    "the result needs awaiting — an async callable returns a coroutine that "
                    f"is discarded, or used as the value. Use `await run_callback({callback}"
                    ", ...)`",
                )
            )

    return found


def _scan() -> tuple[list[Finding], int]:
    scanned = _scanned_files()
    findings = [
        finding for path in scanned for finding in _findings(path.read_text(encoding="utf-8"), path)
    ]
    return findings, len(scanned)


def test_no_dispatch_misreads_a_callable_object() -> None:
    """Shape one: the site asked, and asked the question that gets it wrong."""
    findings, _ = _scan()
    offenders = [str(f) for f in findings if "misreads" in f.detail]

    assert not offenders, (
        "`iscoroutinefunction` reports a callable object whose `__call__` is an "
        "`async def` as synchronous, and that shape is how a stateful callback is "
        "written:\n  " + "\n  ".join(offenders) + "\n\n"
        "`dataknobs_common.callbacks.is_async_callable` is a TypeGuard and so is a "
        "drop-in. If the subject genuinely cannot be a callable object, say why on "
        f"the line with `{EXEMPT} <reason>`."
    )


def test_no_consumer_callback_is_dispatched_without_judgement() -> None:
    """Shape two: the site never asked at all."""
    findings, _ = _scan()
    offenders = [str(f) for f in findings if "without deciding" in f.detail]

    assert not offenders, (
        "a consumer-supplied callback called inside an `async def` with no decision "
        "about whether its result needs awaiting:\n  " + "\n  ".join(offenders) + "\n\n"
        "`await dataknobs_common.callbacks.run_callback(cb, ...)` calls it and awaits "
        "the result if it is awaitable, which is correct for every callable shape. If "
        f"this callback can only ever be synchronous, say why with `{EXEMPT} <reason>`."
    )


def test_the_scope_still_matches_files() -> None:
    """Anti-vacuity. Two empty finding lists agree with a scope matching nothing."""
    _, scanned = _scan()

    assert scanned >= MINIMUM_FILES_SCANNED, (
        f"the declared scope {SCOPE} matched {scanned} files, under the floor of "
        f"{MINIMUM_FILES_SCANNED}. With no allowlist to go stale, this is the only "
        "thing standing between a scope expression that stopped resolving and two "
        "tests that pass because they read nothing."
    )
