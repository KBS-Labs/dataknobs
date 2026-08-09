"""Reproduce-first guard: the shell half of ``bin/`` is linted by something.

Every gate defect this batch tripped over was in a shell script — the change
detection that decided which packages to check, a ``show_usage`` ending in
``exit 0``, a ``print_fail`` reaching no failure path, a lint-target enumeration
that named one directory. Four for four, each found by hand-reading the script,
because 42 shell files totalling ~9,900 lines were linted by nothing at all
while every ``*.py`` beside them went through ruff and mypy.

``shellcheck`` detects the two classes those defects fall into. ``SC2086``: an
unquoted expansion, where an empty variable becomes *zero* arguments rather than
one empty one — so a target set silently comes out empty and the run reports
success over nothing. ``SC2155``: ``local x=$(cmd)``, where ``local``'s exit
status wins and the command's is discarded — so a failing command reads as a
succeeding one.

The tiers below are a **ratchet**, and that is the whole design. A script that
is clean must stay clean; a script that is not is held to errors only, and is
promoted the moment it reaches zero. So the deferred set can only shrink, and
nobody has to remember to revisit it — :func:`test_no_baseline_script_is_already_clean`
fails when one becomes eligible.

These guards read every tracked shell file rather than the check's own
declaration, because a guard derived from the thing it guards moves when that
thing moves. The enumeration here is ``git ls-files``; the check's is its own.
They have to agree, and that disagreement is the failure this file reports.
"""

from __future__ import annotations

import subprocess

import pytest

from tests._workspace import ROOT, load_bin_module

LINT_SHELL = ROOT / "bin" / "lint-shell.sh"

#: The scripts that decide whether a quality run passes, pinned literally.
#:
#: Not derived from ``lint-shell.sh``'s own strict list, for the same reason
#: ``REQUIRED_DEFAULT_TARGETS`` is not derived from ``workspace_targets``: an
#: assertion computed from the declaration it guards cannot notice that
#: declaration shrinking. Demoting one of these to the baseline tier would
#: otherwise be a passing move, and it is precisely the move that would let a
#: defect back into the code that decides the verdict.
REQUIRED_STRICT = frozenset(
    {
        "bin/run-quality-checks.sh",  # owns the verdict and writes the artifact
        "bin/validate.sh",  # the ruff / mypy / print step
        "bin/test.sh",  # the test step
        "bin/validate-quality-artifacts.sh",  # what CI runs in place of the gate
        "bin/package-discovery.sh",  # which packages exist, and what else to check
        "bin/lint-workflows.sh",  # a recorded check in its own right
        "bin/docs-checks.sh",  # produces the documentation status
        "bin/dk",  # the entry point every other invocation goes through
        "bin/fix.sh",  # rewrites the code the above then judges
    }
)


def _tracked_shell_files() -> list[str]:
    """Every tracked shell script, by ``git ls-files`` and then by shebang.

    Extension alone would miss ``bin/dk``, which is the entry point and carries
    no suffix — and missing exactly the file everything else is invoked through
    is the shape of gap this file exists to close.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for name in listing.split("\0"):
        if not name:
            continue
        path = ROOT / name
        if name.endswith(".sh"):
            found.append(name)
            continue
        if not path.is_file():
            continue
        try:
            first = path.open("rb").readline()
        except OSError:  # pragma: no cover - unreadable tracked file
            continue
        if first.startswith(b"#!") and b"sh" in first.split(b"\n")[0]:
            found.append(name)
    assert found, "no tracked shell files found — has the enumeration broken?"
    return sorted(found)


def _lint_shell(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LINT_SHELL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _declared(mode: str) -> set[str]:
    """Ask the check which files it covers, rather than parsing what it says.

    Same reasoning as ``validate.sh --print-targets``: the declaration is a
    filesystem walk plus a tier split, so reading it as text reports what it
    says while the question is what it returns.
    """
    result = _lint_shell(mode)
    assert result.returncode == 0, (
        f"bin/lint-shell.sh {mode} failed ({result.returncode}):\n{result.stderr}"
    )
    return {line.strip() for line in result.stdout.split() if line.strip()}


def test_the_shell_lint_exists_and_is_executable():
    """The seam every other test here trusts. Without it they assert nothing."""
    assert LINT_SHELL.is_file(), (
        f"{LINT_SHELL.relative_to(ROOT)} does not exist — the shell half of bin/ "
        "is linted by nothing, which is the gap this file is written for."
    )
    import os

    assert os.access(LINT_SHELL, os.X_OK), f"{LINT_SHELL.relative_to(ROOT)} is not executable"


def test_every_tracked_shell_file_is_checked():
    """No shell script sits outside the check.

    A file in neither tier is not deferred, it is invisible: nothing reports it
    and no entry anywhere records that it was skipped.
    """
    covered = _declared("--print-targets")
    uncovered = sorted(set(_tracked_shell_files()) - covered)

    assert not uncovered, (
        "Tracked shell files that bin/lint-shell.sh does not check:\n"
        + "\n".join(f"  - {path}" for path in uncovered)
        + "\nEvery shell file belongs to a tier. If one genuinely should not be "
        "checked, exclude it explicitly in lint-shell.sh so the exclusion is "
        "readable — silence is what this guard exists to prevent."
    )


def test_the_check_claims_no_file_that_does_not_exist():
    """The other direction: a stale entry naming a deleted or renamed script.

    Cheap to check and the failure is otherwise invisible, since a tier listing
    a path that matches nothing simply does less work and still reports pass.
    """
    tracked = set(_tracked_shell_files())
    phantom = sorted(_declared("--print-targets") - tracked)

    assert not phantom, (
        "bin/lint-shell.sh names files that are not tracked shell scripts:\n"
        + "\n".join(f"  - {path}" for path in phantom)
    )


def test_the_scripts_that_decide_the_verdict_are_held_to_zero():
    """A literal pin, so demotion to the baseline tier cannot be a passing move."""
    strict = _declared("--print-strict")
    missing = sorted(REQUIRED_STRICT - strict)

    assert not missing, (
        f"These scripts are no longer held to zero shellcheck findings: {missing}. "
        "They decide whether a quality run passes — moving one to the baseline "
        "tier is not a fix, it is the regression this guard exists to catch."
    )


def test_the_strict_pin_has_not_drifted_from_the_declaration():
    """Containment in the other direction, so the pin cannot quietly go stale.

    A script promoted into the strict tier and later dropped from both the tier
    and this list would be missing from each without either noticing. This does
    not require the two to be equal — the strict tier grows as the ratchet turns
    — only that every pinned name is a real one.
    """
    tracked = set(_tracked_shell_files())
    phantom = sorted(REQUIRED_STRICT - tracked)

    assert not phantom, (
        f"REQUIRED_STRICT names scripts that no longer exist: {phantom}. "
        "Rename or remove the entry; a pin matching nothing pins nothing."
    )


def test_every_linted_shell_script_is_covered_by_a_hash_scope():
    """A checked script outside every hash scope lets its own edit go unvalidated.

    The gate records a ``shell_lint`` verdict and these scripts are what it is a
    verdict *about*. Edit one and no package is touched — every suite that passed
    still passes — but the recorded verdict was computed over different code, and
    the stored hashes have no way to say so. CI validates the committed artifact
    instead of re-running the gate, so it accepts the artifact the edit has just
    invalidated. The pull request that changes a checked script is exactly the one
    the gate cannot check, and it reports green.

    Not hypothetical, and not narrowly avoided: when the shell lint was first
    wired in, 38 of the 46 scripts it reports on were outside every scope,
    because a directory entry expanded to ``*.py`` — right while ruff and mypy
    were the only readers, wrong the moment a shell checker joined them.

    Coverage is asked through ``workspace_scope_files``, the function the hash
    itself uses, rather than by re-deriving which paths an entry expands to. A
    second implementation of that rule could answer for a rule nothing follows.
    """
    hashes = load_bin_module("package-hashes")
    covered = {
        str(path.relative_to(ROOT))
        for scope in hashes.WORKSPACE_QUALITY_INPUTS
        for path in hashes.workspace_scope_files(scope)
    }

    targets = _declared("--print-targets")
    assert targets, "the shell lint reports no targets — this guard would check nothing"

    uncovered = sorted(targets - covered)
    assert not uncovered, (
        "The shell lint reports on these scripts, but no hash scope covers them, "
        "so editing one leaves every stored hash intact and its own change "
        "unvalidated:\n"
        + "\n".join(f"  - {path}" for path in uncovered)
        + "\nAdd them to a scope in bin/changed-packages.py, or widen the "
        "directory-entry rule in bin/package-hashes.py so it reaches them."
    )


@pytest.mark.skipif(
    not LINT_SHELL.is_file(), reason="lint-shell.sh does not exist yet"
)
def test_no_baseline_script_is_already_clean():
    """The ratchet. A deferred script that reaches zero must be promoted.

    Without this the baseline is a list nobody revisits: a script cleaned as a
    side effect of unrelated work stays deferred forever, and the next change to
    it can reintroduce findings with nothing to say so. Turning the ratchet is
    the fix — move the name into the strict tier.
    """
    result = _lint_shell("--print-promotable")
    assert result.returncode == 0, f"--print-promotable failed:\n{result.stderr}"
    promotable = sorted(line.strip() for line in result.stdout.split() if line.strip())

    assert not promotable, (
        "These scripts are in the baseline tier but have no shellcheck findings "
        "at all:\n"
        + "\n".join(f"  - {path}" for path in promotable)
        + "\nMove them to the strict tier in bin/lint-shell.sh. A clean script "
        "left deferred can regress without anything reporting it."
    )
