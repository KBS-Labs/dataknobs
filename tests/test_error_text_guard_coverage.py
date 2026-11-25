"""Every package wires the error-text guard, and the set is derived.

``assert_no_broad_except_in_error_text`` scans one package's source for a
message that interpolates an exception caught by an unbounded ``except``. It is
called from a small file in each package's own suite, and for a long time from
seven of the ten — ``utils``, ``structures`` and ``legacy`` had none, which
nothing reported, because a package with no wiring produces no failing test.
That is the shape this whole family of guards is about: a registration set
whose membership nothing checks *is* the membership, and it looks identical
from the outside to a complete one.

So the set is derived here, from the same package discovery every other reader
uses, and the day an eleventh package is created is the day this fails.

The names a wiring guards are derived too, by ``error_names_in``, and the second
check here holds each wiring to that. They used to be listed by hand beside each
package, and a class nobody added to the list was a class the scan did not see:
ten raise sites in two packages quoted a caught exception under a green guard.

What stays in the package is the wiring itself. Each one's docstring says why
that package's text is risky in its own terms — which of its errors a caller
sees, and what third-party words could reach one — and that prose does not
survive a move to one central loop.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._workspace import ROOT, load_bin_module

#: The function a package's wiring has to call. Named once here rather than in
#: each check below, because a rename that this file did not follow would make
#: every package look unwired rather than making this guard look stale.
GUARD_CALL = "assert_no_broad_except_in_error_text("

#: Package discovery, read from the script that owns it rather than by globbing
#: here. A second walk could disagree about what counts as a package, and the
#: disagreement would show up as a package silently exempt from this check.
ALL_PACKAGES: list[str] = load_bin_module("changed-packages").ALL_PACKAGES


def _packages_wiring_the_guard() -> set[str]:
    """Every package whose own suite calls the guard from a collected file.

    Restricted to ``test_*.py``, and that restriction is the point rather than
    tidiness: the call sitting in a helper module beside the tests would satisfy
    a looser scan while pytest never collected it, so the guard would be present
    and never run — which is the same green-while-checking-nothing state this
    file exists to detect, reached by a different route.
    """
    wired: set[str] = set()
    for package in ALL_PACKAGES:
        tests = ROOT / "packages" / package / "tests"
        if not tests.is_dir():
            continue
        for source in tests.rglob("test_*.py"):
            if GUARD_CALL in source.read_text(encoding="utf-8"):
                wired.add(package)
                break
    return wired


def test_the_guard_call_this_looks_for_still_exists() -> None:
    """Non-vacuity, in the direction that fails open.

    If ``assert_no_broad_except_in_error_text`` were renamed, every package
    would read as unwired and this file would fail loudly — noisy, but safe.
    The dangerous direction is the discovery going empty, since a check over no
    packages passes. Both are pinned here so the failure names the cause.
    """
    assert len(ALL_PACKAGES) > 5, (
        f"only {len(ALL_PACKAGES)} packages discovered — the walk broke, and "
        "the check below would pass by asking about almost none of them"
    )

    from dataknobs_common import testing

    assert hasattr(testing, GUARD_CALL.rstrip("(")), (
        f"dataknobs_common.testing no longer exports {GUARD_CALL.rstrip('(')} — "
        "if it was renamed, update GUARD_CALL here and in every package wiring, "
        "rather than deleting this guard"
    )


def test_every_package_wires_the_error_text_guard() -> None:
    """A package with no wiring is not a package with nothing to guard.

    All three that were missing pass the assertion on arrival, which is exactly
    why nobody noticed: the gap costs nothing until the first error path is
    added, and by then the absence looks like a decision.
    """
    unwired = sorted(set(ALL_PACKAGES) - _packages_wiring_the_guard())
    assert not unwired, (
        f"{len(unwired)} of {len(ALL_PACKAGES)} packages never call "
        f"{GUARD_CALL.rstrip('(')}, so nothing checks whether their error "
        "messages carry text from an unbounded except:\n"
        + "\n".join(f"  - {name}" for name in unwired)
        + "\n\nAdd packages/<name>/tests/test_error_text_guard.py following any "
        "of the existing ones. If the package raises nothing today the test "
        "passes immediately — which is the useful moment to add it, not a "
        "reason to skip it."
    )


def _guarded_names_arguments(source: str) -> list[ast.expr]:
    """What each call to the guard in ``source`` passes as ``error_names``."""
    return [
        keyword.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == GUARD_CALL.rstrip("(")
        for keyword in node.keywords
        if keyword.arg == "error_names"
    ]


def _derives(argument: ast.expr, import_names: set[str]) -> bool:
    """Whether an ``error_names`` argument is ``error_names_in`` of this package.

    Exactly that call. ``error_names_in(...) | {"X"}`` is a list growing back.
    """
    return (
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Name)
        and argument.func.id == "error_names_in"
        and len(argument.args) == 1
        and isinstance(argument.args[0], ast.Constant)
        and argument.args[0].value in import_names
        and not argument.keywords
    )


def _import_names(package_root: Path) -> set[str]:
    return {path.parent.name for path in (package_root / "src").glob("*/__init__.py")}


def test_every_wiring_derives_the_names_it_guards() -> None:
    """A hand-written list is the shape that let ten sites through."""
    listed = []
    for package in ALL_PACKAGES:
        root = ROOT / "packages" / package
        wiring = root / "tests" / "test_error_text_guard.py"
        arguments = _guarded_names_arguments(wiring.read_text(encoding="utf-8"))
        if not any(_derives(argument, _import_names(root)) for argument in arguments):
            listed.append(package)
    assert not listed, (
        f"{listed}: the guard is not passed `error_names=error_names_in(<this package>)`. "
        "A list written by hand misses the next error class."
    )


def test_a_list_unioned_onto_the_derived_names_is_not_derived() -> None:
    call = GUARD_CALL + "SRC, error_names={})"
    for names, expected in {
        'error_names_in("pkg")': True,
        'error_names_in("pkg") | {"ExtraError"}': False,
        'GUARDED_ERROR_NAMES | {"ExtraError"}': False,
        'error_names_in("another")': False,
        'error_names_in("pkg", skip_modules=["pkg.x"])': False,
        "NAMES": False,
    }.items():
        (argument,) = _guarded_names_arguments(call.format(names))
        assert _derives(argument, {"pkg"}) is expected, names
