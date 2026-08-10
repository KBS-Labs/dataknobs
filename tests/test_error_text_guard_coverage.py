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

What is deliberately **not** derived is the content. The tempting next step —
one loop in this file over ``packages/*/src`` — needs the per-package extra
error names to live somewhere, and they would end up in a table here: a
hand-maintained registration set recreated one layer up, no longer beside the
package whose error types it names. It would also strand the reason. Each
wiring's docstring says why that package's text is risky in its own terms —
which of its errors a caller sees, and what third-party words could reach one —
and that prose does not survive the move to a central loop.

Derive membership; keep parameters beside what they parameterize.
"""

from __future__ import annotations

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


def test_the_guard_call_this_looks_for_still_exists():
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


def test_every_package_wires_the_error_text_guard():
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
