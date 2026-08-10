"""The two roles are separate: checkers check, and one gate produces evidence.

``bin/run-quality-checks.sh`` is a checker. ``bin/dk pr`` is the gate — it runs
the same checks and additionally writes ``.quality-artifacts/``, the evidence CI
verifies against the committed tree.

Three of the checker's four modes used to write that directory, including the
no-argument default, and the comment CI leaves on a failed gate told the
developer to run exactly that form. So the documented remedy for a red gate was
the command that rewrites the evidence the gate reads: run it, commit what came
out, and the check passes because the artifacts now agree with the tree — which
is the one thing they were never supposed to be able to do on their own.

Nothing about that was visible. A checker that also writes produces no error, no
warning, and no diff a reviewer would question; the only symptom is a gate that
has stopped being able to fail.

Two properties, and they are load-bearing together rather than separately:

1. **Reachability** — the artifacts directory is named in exactly one place, the
   line that resolves ``OUTPUT_DIR``. Every write in the script targets
   ``OUTPUT_DIR``, so a run that does not resolve it to the artifacts directory
   has no way to name it at all.
2. **Resolution** — ``OUTPUT_DIR`` is the artifacts directory only under
   ``--emit-artifacts``, in every mode, asked through the script's own
   resolution rather than re-derived here.

Either alone is satisfiable while the property fails: a correct resolution does
not stop a second write path from naming the directory directly, and an
unreachable name does not stop the resolution from pointing at it in dev mode.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests._workspace import ROOT, tracked_shell_files

#: The checker under test, and the gate that wraps it.
CHECKER = ROOT / "bin" / "run-quality-checks.sh"
GATE = ROOT / "bin" / "dk"

#: The directory whose contents CI reads back.
ARTIFACTS = ROOT / ".quality-artifacts"

#: Every scope selector the checker accepts, plus the bare invocation. Each is
#: probed, because the defect was not that one mode wrote artifacts — it was
#: that the *default* did, and the three that did were the ones nobody thought
#: of as artifact modes.
CHECK_ONLY_INVOCATIONS = (
    (),
    ("--pr",),
    ("--all",),
    ("--full",),
    ("--dev",),
    ("--dev", "data"),
    ("data",),
)

#: A shell construct that creates or replaces a file, or moves the working
#: directory somewhere a later relative write would land. ``cd`` belongs here:
#: the coverage step changes directory and then writes relative paths, so a
#: ``cd`` to the artifacts directory is a write path with no artifacts directory
#: in the writing line at all.
WRITE_CONTEXT_RE = re.compile(r"(?:^|\s|\|)(?:>>?|mv|cp|mkdir|touch|tee|rm|ln|cd)\s")

#: A read of the variable, in either spelling. The assignment is not a read.
ARTIFACTS_READ_RE = re.compile(r"\$\{?ARTIFACTS_DIR\b")

#: What that single read is allowed to be.
ARTIFACTS_READ_SITE = 'OUTPUT_DIR="$ARTIFACTS_DIR"'


def _checker_lines() -> list[tuple[int, str]]:
    return list(enumerate(CHECKER.read_text(encoding="utf-8").splitlines(), start=1))


def _output_dir_for(*args: str) -> subprocess.CompletedProcess[str]:
    """Ask the checker where this invocation's writes would land.

    Through ``--print-output-dir``, which resolves and prints without running a
    check — the shape ``validate.sh --print-targets`` already uses here. Asking
    the script rather than restating its rule is the point: a guard that
    recomputes the answer agrees with itself, not with the script.
    """
    return subprocess.run(
        [str(CHECKER), *args, "--print-output-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # The refusal case exits non-zero on purpose, and a test asserts it.
        check=False,
    )


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------


def test_the_artifacts_directory_is_named_once_where_it_is_resolved():
    """One read, on the line that resolves ``OUTPUT_DIR``.

    This is what makes check-only a property of the code rather than a claim
    about it. Gating each write site on a flag would put the property in the
    hands of whoever adds the next write; leaving one name means a run that did
    not ask for the artifacts directory cannot reach it however it is edited.
    """
    reads = [
        (n, line.strip())
        for n, line in _checker_lines()
        if ARTIFACTS_READ_RE.search(line) and not line.lstrip().startswith("#")
    ]

    assert len(reads) == 1, (
        f"bin/run-quality-checks.sh reads $ARTIFACTS_DIR {len(reads)} time(s):\n"
        + "\n".join(f"  - line {n}: {text}" for n, text in reads)
        + f"\n  Expected exactly one, {ARTIFACTS_READ_SITE!r}. Every other write "
        "must target $OUTPUT_DIR, which is the artifacts directory only under "
        "--emit-artifacts."
    )

    _, text = reads[0]
    assert ARTIFACTS_READ_SITE in text, (
        f"the one read of $ARTIFACTS_DIR is {text!r}, not the OUTPUT_DIR "
        "resolution — so something other than the resolution can reach the "
        "artifacts directory"
    )


def test_no_write_names_the_artifacts_directory_literally():
    """The single-read rule is only worth having if the path has one spelling.

    ``.quality-artifacts`` written out longhand routes around the variable and
    around the resolution with it. Mentioning it in a comment or a message is
    fine — the directory is what those are about.
    """
    offenders = [
        (n, line.strip())
        for n, line in _checker_lines()
        if ".quality-artifacts" in line
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("ARTIFACTS_DIR=")
        and WRITE_CONTEXT_RE.search(line)
    ]

    assert not offenders, (
        "bin/run-quality-checks.sh builds a path from the literal "
        ".quality-artifacts:\n"
        + "\n".join(f"  - line {n}: {text}" for n, text in offenders)
        + "\n  Write to $OUTPUT_DIR instead."
    )


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("args", CHECK_ONLY_INVOCATIONS, ids=lambda a: " ".join(a) or "(no args)")
def test_a_run_without_the_flag_writes_outside_the_repository(args: tuple[str, ...]):
    """Every mode, because the default was one of the three that wrote."""
    result = _output_dir_for(*args)
    assert result.returncode == 0, result.stderr or result.stdout

    target = Path(result.stdout.strip())
    assert target != ARTIFACTS, (
        f"bin/run-quality-checks.sh {' '.join(args) or '(no args)'} writes to "
        f"{ARTIFACTS.name}/ without --emit-artifacts"
    )
    assert ROOT not in target.parents, (
        f"bin/run-quality-checks.sh {' '.join(args) or '(no args)'} writes "
        f"inside the repository, at {target}"
    )


def test_the_flag_resolves_to_the_artifacts_directory():
    """The other direction, so the fix cannot be "never write anything"."""
    result = _output_dir_for("--pr", "--emit-artifacts")
    assert result.returncode == 0, result.stderr or result.stdout
    assert Path(result.stdout.strip()) == ARTIFACTS


def test_the_flag_is_refused_where_it_would_write_a_partial_set():
    """Dev mode runs none of the steps that produce the artifacts.

    Accepting the flag there would leave a directory holding whichever files
    happened to fall out of a quick run — which still carries a signature, and
    so still passes the check that exists to notice a stale one.
    """
    result = _output_dir_for("--dev", "--emit-artifacts")
    assert result.returncode != 0, (
        "--dev --emit-artifacts was accepted; it can only half-write the set"
    )


# --------------------------------------------------------------------------
# Who passes the flag
# --------------------------------------------------------------------------


def _dk_checker_invocations() -> list[str]:
    """Lines in ``bin/dk`` that run the checker."""
    return [
        line.strip()
        for line in GATE.read_text(encoding="utf-8").splitlines()
        if "run-quality-checks.sh" in line and not line.lstrip().startswith("#")
    ]


def test_the_pr_commands_emit_and_the_check_commands_do_not():
    """``bin/dk pr`` is the gate; ``bin/dk check`` is a checker.

    Read off the scope selector each invocation passes rather than off the
    command name above it, so a new PR-preparation verb that forgets the flag
    fails here instead of shipping a command that reports a clean gate and
    leaves the evidence untouched.
    """
    invocations = _dk_checker_invocations()
    assert invocations, "bin/dk no longer invokes bin/run-quality-checks.sh"

    violations = []
    for line in invocations:
        emits = "--emit-artifacts" in line
        gate_scope = any(flag in line for flag in ("--pr", "--all", "--full"))
        if gate_scope and not emits:
            violations.append(f"produces no artifacts: {line}")
        if not gate_scope and emits:
            violations.append(f"emits artifacts outside a PR scope: {line}")

    assert not violations, "bin/dk:\n" + "\n".join(f"  - {v}" for v in violations)


def test_nothing_else_in_the_workspace_passes_the_flag():
    """One gate. A second one is a second thing that can rewrite the evidence."""
    offenders = [
        f"{name}:{n}: {line.strip()}"
        for name in tracked_shell_files()
        if name not in {"bin/dk", "bin/run-quality-checks.sh"}
        for n, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1)
        if "--emit-artifacts" in line and not line.lstrip().startswith("#")
    ]

    assert not offenders, (
        "only bin/dk may ask for artifacts:\n" + "\n".join(f"  - {o}" for o in offenders)
    )
