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

**And the first AST version repeated that mistake one level up.** It read a
callback only where it was a *parameter of the dispatching function*, matched
as a bare ``ast.Name``. Almost no real callback is spelled that way: it is
taken by ``__init__``, stored, and dispatched later as ``self.overflow_handler``
or ``self.progress_callback``. So the structurally superior check covered none
of the four sites that motivated it, and the token check --- the one this
paragraph disparages --- was the only half doing any work. Attribute-held
callbacks are now first-class; see :func:`_callback_attributes` and
:func:`_dispatch_key`. Turning it on found four unjudged dispatches in
``keyed_store.py``, a file the guard had been reading and passing all along.

Still not covered, and named here so the next reader knows the edge rather
than discovering it: a callback reached through a container
(``route["condition"]``) or bound by a loop over one. Deciding what those hold
is type inference, and the over-reporting this guard prefers stops being safe
once a finding cannot be settled by reading one function.

There is no allowlist. Both findings are zero in scope, so an exemption
would be an empty declaration whose only effect is to rot --- and the one
entry the token-keyed version *did* carry turned out to rest on a reason that
did not survive reading the constructor it was about. A site that genuinely
needs an exception takes a line-level ``# async-dispatch-exempt: <reason>``
marker, which travels with the line rather than with a spelling of its
argument, and which cannot be satisfied by a bare marker.

:data:`UNADOPTED` is not that. It excludes whole *packages* rather than
findings, it records what they still hold between them, and a ratchet fails it
the moment an entry stops being true --- which is the property the allowlist
above lacked.

This guard is structural, so it can only ever say that a judgement was
*made*. Whether the judgement is *right* is a runtime question, and these
files answer it by driving each dispatch with a real callable object and
asserting the callback actually ran:

- ``packages/common/tests/test_callbacks_dispatch.py`` --- the two helpers
  themselves, against every callable shape including the ones they refuse.
- ``packages/data/tests/test_consumer_callback_dispatch.py``
- ``packages/fsm/tests/`` --- ``test_consumer_callback_dispatch``,
  ``test_execution_callback_dispatch``, ``test_engine_callable_object_dispatch``,
  ``test_streaming_callback_dispatch``, ``test_resilience_callback_dispatch``,
  ``test_async_detection_delegation``.

The split matters when reading a failure. This file failing means somebody
dispatched without asking; one of those failing means somebody asked and got
the wrong answer. Only the second kind has ever cost data.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tests._workspace import ROOT, load_bin_module, rel, tracked_python_files

#: Packages left outside the scope, each because its sites are not clear yet.
#:
#: Not a backlog of oversights. Every one is a published dispatch whose
#: behaviour changes the moment it starts awaiting, so each needs its own
#: reproduce-first proof before its package can move --- adding a package to
#: the scope without doing that work turns a guard that passes into a guard
#: that is disabled. The four hold **9 sites of the first shape and 30 of the
#: second** as this was written, and that figure is worth re-measuring rather
#: than quoting: a census is a property of the question asked, and this one
#: moved once already when the detector learned about ``self.<attr>``
#: callbacks (see :func:`_callback_attributes`), which roughly tripled the
#: second shape against unchanged code.
#:
#: Distinct from the finding allowlist the module docstring says this guard
#: does not have, and distinct in the way that matters: an entry here cannot
#: quietly stop meaning anything, because
#: :func:`test_an_unadopted_package_still_has_something_to_adopt` fails when
#: one of these becomes clean. The list can only shrink, and it shrinks by
#: being noticed.
#:
#: What the ``fsm`` pass cost, since the paragraph above is worth calibrating
#: against a case where the bill came in: 23 findings across 9 modules, and
#: fewer distinct defects than that --- four of the six in ``async_engine``
#: were two hand-written copies of one judgement, counted twice because the
#: token appears twice. Eight sat in ``patterns/`` modules that no *shipped*
#: module imports, which had looked like a reason to defer them; it was not,
#: because their fix turned out to be one delegation each and the tests that
#: prove it are the tests those modules did not otherwise have.
UNADOPTED = frozenset({"bots", "common", "llm", "xization"})


def _undeprecated_packages() -> tuple[str, ...]:
    """Every package the registry declares and does not mark deprecated.

    Through ``bin/list-packages.py`` rather than by parsing the JSON here. That
    script is the registry's reader, and its own comment gives the reason this
    file should not become a second one: the shape is owned by
    ``.dataknobs/packages.json``, and a second declaration of it is one more
    thing to keep in step with the file it describes.
    """
    registry = load_bin_module("list-packages")
    declared = registry.filter_packages(
        registry.load_registry()["packages"], exclude_deprecated=True
    )
    return tuple(sorted(package["name"] for package in declared))


#: The scanned scope, as directory prefixes.
#:
#: One package rather than one subpackage: the first version of this check
#: covered ``dataknobs_data/vector/`` and so reported green over sites in
#: ``database.py``, ``streaming.py`` and ``migration/`` that hold the same
#: defect. A scope narrower than the unit a contributor edits is a scope that
#: certifies the region already cleaned.
#:
#: Derived rather than listed, and the difference is which way the default
#: falls. A hand-written tuple leaves a *new* package outside this guard
#: silently and for good: nothing prompts the edit, and a package nobody
#: thought about is indistinguishable from one deliberately left out. Deriving
#: inverts that --- a package is in scope the day it is registered, and if it
#: holds findings the two checks below fail and name them. The only way out is
#: :data:`UNADOPTED`, which is a sentence someone has to write.
#:
#: **Deprecated packages are out, by the registry's own flag rather than by an
#: opinion recorded here.** That distinction is the whole value: a name in this
#: file claims a package is deprecated, and claims it in a place nothing
#: revisits, so the day a package stops being deprecated --- or the day
#: somebody adds a name for a different reason and calls it the same thing ---
#: the exclusion has started hiding live code. Reading ``deprecated`` from the
#: declaration every other tool reads cannot drift that way.
#:
#: It costs no coverage here either way. ``legacy`` is 202 lines: four
#: ``__init__`` shims re-exporting the modular packages, and the ``sys.modules``
#: aliasing helper behind them. There is no ``async def`` anywhere in it, and
#: this guard reads nothing but ``AsyncFunctionDef`` bodies and
#: ``iscoroutinefunction`` calls --- so there is no finding for it to have, now
#: or after any edit that keeps it a compatibility alias.
#:
#: ``config``, ``structures`` and ``utils`` arrived the same way and are worth
#: naming for contrast with the cost paragraph on :data:`UNADOPTED`: all three
#: already measured zero, so they joined for the price of being registered.
#: Widening is expensive where there is something to fix and free where there
#: is not, and only a measurement tells you which you are looking at.
SCOPE = tuple(f"packages/{name}/src" for name in _undeprecated_packages() if name not in UNADOPTED)

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
#:
#: ``run_callback`` and ``run_callback_off_loop`` differ only in whether a
#: synchronous callback runs on the loop or on a worker thread. Both judge the
#: callable, which is the question this guard asks, so both discharge it; which
#: of the two a site should use is a property of the surface and is argued at
#: the site rather than checked here.
DELEGATORS = frozenset(
    {"to_thread", "run_in_executor", "run_callback", "run_callback_off_loop", "submit"}
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _files_under(prefix: str) -> list[Path]:
    """Every tracked ``*.py`` under one directory prefix.

    Per prefix rather than over the whole scope at once, because two checks
    want it that way: the anti-vacuity one asks each entry separately, and the
    ratchet on :data:`UNADOPTED` asks about a package the scope excludes.
    """
    return [ROOT / name for name in tracked_python_files() if name.startswith(f"{prefix}/")]


def _scanned_files() -> list[Path]:
    """Every tracked ``*.py`` under a declared scope prefix."""
    return [path for prefix in SCOPE for path in _files_under(prefix)]


def _call_name(node: ast.Call) -> str:
    """The callee's own name, ignoring whatever module it hangs off."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _dispatch_key(node: ast.expr) -> str | None:
    """How one callback is spelled, canonicalised so the sets can compare.

    Two spellings reach the same callback and both have to answer to the same
    judgement: a bare parameter (``callback``) and an instance attribute
    holding one (``self.progress_callback``). Keying on a string rather than
    on the node lets ``judged``, ``delegated`` and the dispatch walk all speak
    one vocabulary instead of three.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _callback_parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
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


def _callback_attributes(cls: ast.ClassDef) -> set[str]:
    """``self.x`` keys for attributes holding a constructor's callback.

    The shape this guard was blind to, and the one every defect it has
    actually caught was written in. A callback taken by ``__init__`` and
    stored is not a parameter of the method that later dispatches it, so a
    detector reading only ``fn.args`` cannot see ``self.overflow_handler(...)``
    or ``self.progress_callback(...)`` however carefully it reads them ---
    which is how a check whose whole argument is that the structural shape
    beats the token came to cover none of the four sites that motivated it.

    Two sources, both read off the annotation as above:

    - ``self.cb = cb`` in ``__init__``, where ``cb`` is annotated callable.
      Covers the overwhelming majority, including every site here.
    - ``self.cb: Callable[...] = ...``, an annotated attribute assignment,
      wherever it appears in the class body.

    Not covered, deliberately: an attribute assembled from a container or
    rebound in a method the constructor never sees. Deciding what those hold
    is type inference, and the over-reporting this guard prefers stops being
    safe once a finding cannot be settled by looking at one function.
    """
    found: set[str] = set()

    for node in cls.body:
        # `self.cb: Callable[...] = ...` --- annotated in the class body.
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            annotation = ast.unparse(node.annotation)
            key = _dispatch_key(node.target)
            if (
                key
                and key.startswith("self.")
                and ("Callable" in annotation or "Awaitable" in annotation)
            ):
                found.add(key)

        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name != "__init__":
            continue
        callable_parameters = _callback_parameters(node)
        for statement in ast.walk(node):
            if isinstance(statement, ast.AnnAssign):
                targets: list[ast.expr] = [statement.target]
                value = statement.value
            elif isinstance(statement, ast.Assign):
                targets = list(statement.targets)
                value = statement.value
            else:
                continue
            # `self.cb = cb`, and `self.cb = cb or default` --- the second
            # because a `None`-defaulted callback is normally coalesced.
            sources: list[ast.expr] = []
            if value is not None:
                sources = [value]
                if isinstance(value, ast.BoolOp):
                    sources = list(value.values)
            if not any(
                isinstance(source, ast.Name) and source.id in callable_parameters
                for source in sources
            ):
                continue
            for target in targets:
                key = _dispatch_key(target)
                if key and key.startswith("self."):
                    found.add(key)

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
            if isinstance(value, ast.Call):
                source = _dispatch_key(value.func)
                if source is not None:
                    for target in targets:
                        key = _dispatch_key(target)
                        if key is not None:
                            assigned_from[key] = source

    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or _call_name(node) not in CLASSIFIERS:
            continue
        for argument in node.args:
            key = _dispatch_key(argument)
            if key is None:
                continue
            judged.add(key)
            # `isawaitable(result)` judges whatever produced `result`.
            judged.add(assigned_from.get(key, key))
            # `isawaitable(wrapper.attr)` also settles `wrapper` itself.
            if isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name):
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

    # A callback stored by `__init__` is dispatched by methods that do not
    # take it as a parameter, so the class is the unit that knows about it.
    attributes_in_scope: dict[int, set[str]] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        attributes = _callback_attributes(cls)
        if not attributes:
            continue
        for node in ast.walk(cls):
            if isinstance(node, ast.AsyncFunctionDef):
                attributes_in_scope.setdefault(id(node), set()).update(attributes)

    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        callbacks = _callback_parameters(fn) | attributes_in_scope.get(id(fn), set())
        if not callbacks:
            continue
        judged = _judged_names(fn)
        awaited = {id(node.value) for node in ast.walk(fn) if isinstance(node, ast.Await)}
        delegated = {
            key
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and _call_name(node) in DELEGATORS
            for argument in node.args
            if (key := _dispatch_key(argument)) is not None
        }
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or id(node) in awaited:
                continue
            callback = _dispatch_key(node.func)
            if callback is None or callback not in callbacks:
                continue
            if callback in judged or callback in delegated:
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


def _scan_paths(paths: list[Path]) -> list[Finding]:
    return [
        finding for path in paths for finding in _findings(path.read_text(encoding="utf-8"), path)
    ]


def _scan() -> tuple[list[Finding], int]:
    scanned = _scanned_files()
    return _scan_paths(scanned), len(scanned)


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


def test_every_scope_entry_still_matches_files() -> None:
    """Anti-vacuity. Two empty finding lists agree with a scope matching nothing.

    Per entry rather than against a total, because a total cannot guard entries
    this unequal. The scope spans 209 files of which ``structures`` is five, so
    any floor loose enough to survive ordinary deletion is also loose enough to
    lose that entry whole. The floor this replaces was 120 --- set, correctly
    for a two-element scope, above the larger package's own count so that
    losing either one would fail. Against five entries the same number would
    have watched ``config``, ``structures`` and ``utils`` all leave together
    and still reported green, which is the failure it was written to catch.

    One file per entry needs no maintenance and catches what actually happens:
    a prefix that stops resolving because a package was renamed or moved.
    """
    assert SCOPE, (
        "the scope derived from the package registry is empty, so both checks "
        "above passed by reading nothing. Either the registry declares no "
        f"undeprecated package, or UNADOPTED ({sorted(UNADOPTED)}) now covers "
        "every one that it does."
    )

    empty = [prefix for prefix in SCOPE if not _files_under(prefix)]
    assert not empty, (
        f"{len(empty)} scope entr(ies) matched no tracked Python file: "
        f"{', '.join(empty)}. A prefix that stopped resolving reads exactly like "
        "a package with nothing wrong in it — the two checks above have simply "
        "stopped covering it. Either the package moved, or it is gone and its "
        "name should leave the registry."
    )


def test_an_unadopted_package_still_has_something_to_adopt() -> None:
    """A package that has become clean belongs in the scope, not beside it.

    The ratchet under :data:`UNADOPTED`, and the reason that list is not the
    kind of allowlist this guard refuses to keep. An exception whose staleness
    nothing detects is indistinguishable from a decision, and this one would go
    stale in the direction that costs: findings get fixed one at a time by
    people who are not thinking about this file, and the last fix in a package
    would silently leave it certified by nobody.
    """
    registered = set(_undeprecated_packages())

    unknown = sorted(UNADOPTED - registered)
    assert not unknown, (
        f"UNADOPTED names {', '.join(unknown)}, which the registry does not "
        "declare as undeprecated packages. An entry naming nothing excludes "
        "nothing, and reads as a decision that is still holding."
    )

    clean = sorted(
        name for name in UNADOPTED if not _scan_paths(_files_under(f"packages/{name}/src"))
    )
    assert not clean, (
        f"{', '.join(clean)} now measures zero findings while still sitting in "
        "UNADOPTED, so nothing is stopping the package from joining SCOPE — and "
        "until it does, a new unjudged dispatch there is invisible. Drop the "
        "name from UNADOPTED; the scope follows from the registry."
    )
