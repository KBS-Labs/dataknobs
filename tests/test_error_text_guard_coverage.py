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

import pytest

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


#: How a wiring spells the root it scans: the package's own ``src``.
SOURCE_ROOT = "Path(__file__).resolve().parents[1] / 'src'"


def _scans_of_source(source: str) -> list[ast.Call]:
    """Each call to the guard in ``source`` that scans the package's ``src``.

    The root is a name, so the assignment is what is compared. The guard's own
    suite also calls it on planted files, which are not a wiring.
    """
    tree = ast.parse(source)
    roots = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == SOURCE_ROOT
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == GUARD_CALL.rstrip("(")
        and any(isinstance(arg, ast.Name) and arg.id in roots for arg in node.args)
    ]


def _derives(call: ast.Call, import_names: set[str]) -> bool:
    """Whether a call's ``error_names`` is ``error_names_in`` of this package.

    Exactly that call. ``error_names_in(...) | {"X"}`` is a list growing back.
    """
    return any(
        keyword.arg == "error_names"
        and isinstance(argument := keyword.value, ast.Call)
        and isinstance(argument.func, ast.Name)
        and argument.func.id == "error_names_in"
        and len(argument.args) == 1
        and isinstance(argument.args[0], ast.Constant)
        and argument.args[0].value in import_names
        and not argument.keywords
        for keyword in call.keywords
    )


def _wired_by_derivation(source: str, import_names: set[str]) -> bool:
    """Whether every scan of ``src`` in a wiring derives its names, and one exists."""
    calls = _scans_of_source(source)
    return bool(calls) and all(_derives(call, import_names) for call in calls)


def _import_names(package_root: Path) -> set[str]:
    return {path.parent.name for path in (package_root / "src").glob("*/__init__.py")}


def test_every_wiring_derives_the_names_it_guards() -> None:
    """A hand-written list is the shape that let ten sites through."""
    listed = []
    for package in ALL_PACKAGES:
        root = ROOT / "packages" / package
        wiring = root / "tests" / "test_error_text_guard.py"
        if not wiring.exists() or not _wired_by_derivation(
            wiring.read_text(encoding="utf-8"), _import_names(root)
        ):
            listed.append(package)
    assert not listed, (
        f"{listed}: tests/test_error_text_guard.py does not scan the package's src "
        "with `error_names=error_names_in(<this package>)` in every call. "
        "A list written by hand misses the next error class."
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{guard}_SRC, error_names=error_names_in("pkg"))', True),
        ('{guard}_SRC, error_names=error_names_in("pkg") | {{"ExtraError"}})', False),
        ('{guard}_SRC, error_names=GUARDED_ERROR_NAMES | {{"ExtraError"}})', False),
        ('{guard}_SRC, error_names=error_names_in("another"))', False),
        ('{guard}_SRC, error_names=error_names_in("pkg", skip_modules=["pkg.x"]))', False),
        ("{guard}_SRC, error_names=NAMES)", False),
        ("{guard}_SRC)", False),
        ('{guard}OTHER, error_names=error_names_in("pkg"))', False),
        (
            '{guard}_SRC, error_names=error_names_in("pkg"))\n{guard}_SRC, error_names=NAMES)',
            False,
        ),
        ('{guard}_SRC, error_names=error_names_in("pkg"))\n{guard}OTHER, error_names=NAMES)', True),
    ],
)
def test_what_counts_as_a_derived_wiring(body: str, expected: bool) -> None:
    source = f"_SRC = {SOURCE_ROOT}\nOTHER = tmp_path\n" + body.format(guard=GUARD_CALL)
    assert _wired_by_derivation(source, {"pkg"}) is expected
