"""Behavioural guards for ``bin/diagnose-quality-failures.sh``.

The gate records eight checks. This tool displayed four of them, because it
asked for each one by name — the same hand-kept-list defect
``validate-quality-artifacts.sh`` was already fixed for, left standing in the
other reader of the same file. A run that failed shell lint, workflow lint, or
any of the three documentation checks therefore reported four passing rows and
no failure at all, and then printed "To fix these issues:" with nothing under
it, because the remedy section named the same four.

That is worse than an incomplete display. The tool exists to answer *what broke*
after a red gate, so for five of the eight checks the answer it gave was silence
that read like an all-clear.

Both tests here drive the script over a synthetic run whose checks include a
name that appears nowhere in its source. A reader that enumerates passes; a
reader that asks by name cannot, whatever list it holds — which is the property
worth pinning, rather than today's set of eight.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests._workspace import ROOT, rel

DIAGNOSE = ROOT / "bin" / "diagnose-quality-failures.sh"

#: A check the gate does not have and this tool has never heard of. The point of
#: the tests below is that it is reported anyway.
INVENTED = "nightingale_check"


def _summary(checks: dict[str, dict[str, object]], overall: str) -> dict[str, object]:
    return {
        "timestamp": "2026-08-10T12:00:00Z",
        "overall_status": overall,
        "run_mode": "pr",
        "environment": "host",
        "packages": "all",
        "coverage_percent": 91.2,
        "checks": checks,
    }


def _passing(tool: str = "") -> dict[str, object]:
    entry: dict[str, object] = {"status": "pass", "exit_code": 0, "skipped": False}
    if tool:
        entry["tool"] = tool
    return entry


def _failing(tool: str = "") -> dict[str, object]:
    entry: dict[str, object] = {"status": "fail", "exit_code": 1, "skipped": False}
    if tool:
        entry["tool"] = tool
    return entry


def _diagnose(tmp_path: Path, summary: dict[str, object] | None, started: str = "") -> str:
    """Run the tool over a synthetic run directory and return what it printed.

    ``started`` writes the in-progress marker, which is how the producer records
    that a run began and never closed out.
    """
    if summary is not None:
        (tmp_path / "quality-summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    if started:
        (tmp_path / ".run-in-progress").write_text(f"{started}\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(DIAGNOSE), "--from", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert "Unknown option" not in result.stdout, (
        f"{rel(DIAGNOSE)} rejected --from:\n{result.stdout}"
    )
    return result.stdout + result.stderr


def test_every_recorded_check_is_reported(tmp_path):
    """A check the summary records is a check the developer gets told about.

    The failure this pins is silent by construction: the missing rows are the
    ones nobody was looking at, so an incomplete display looks exactly like a
    complete one.
    """
    names = [
        "documentation",
        "documentation_versions",
        "documentation_mirrors",
        "validation",
        "shell_lint",
        "workflow_lint",
        "unit_tests",
        "integration_tests",
        INVENTED,
    ]
    output = _diagnose(tmp_path, _summary({name: _passing() for name in names}, "PASS"))

    # The label the projection derives, which is what a reader that enumerates
    # would print without having been told the name exists.
    missing = [name for name in names if name.replace("_", " ").capitalize() not in output]
    assert not missing, (
        f"{rel(DIAGNOSE)} recorded these checks and reported none of them: "
        f"{', '.join(missing)}\n\n{output}"
    )


def test_a_failing_check_with_no_dedicated_remedy_is_still_named(tmp_path):
    """ "To fix these issues:" with nothing under it is not a diagnosis.

    Only tests, style and linting had remedy lines. A run red on shell lint
    alone reached the end of this tool having printed a heading, a blank list,
    and an instruction to re-run the gate that would fail the same way.
    """
    summary = _summary(
        {
            "unit_tests": _passing(),
            "integration_tests": _passing(),
            "shell_lint": _failing(tool="lint-shell.sh"),
        },
        "FAIL",
    )
    output = _diagnose(tmp_path, summary)

    next_steps = output.split("Next Steps", 1)[-1]
    assert "shell" in next_steps.lower(), (
        f"{rel(DIAGNOSE)} diagnosed a run that failed shell lint without "
        f"mentioning it in the remedy:\n{next_steps}"
    )


def test_a_run_that_never_finished_is_not_reported_as_the_current_one(tmp_path):
    """The worst version of a stale read is a green one.

    Seven checks in the gate exit before the summary is written, and the gate
    cannot clear its output directory on entry — it holds committed files. So an
    abort leaves the previous run's summary beside logs the aborted run has
    already overwritten. Ask what broke and the answer is the older run's, which
    if that one passed is a full set of ticks for a tree that has not been
    checked.

    The verdict is still shown: those logs are worth reading, and the marker
    cannot say which of them are new. What must not survive is the impression
    that it describes the run you just did.
    """
    passing = _summary({"unit_tests": _passing(), "shell_lint": _passing()}, "PASS")
    output = _diagnose(tmp_path, passing, started="2026-08-10T12:34:56Z")

    assert "2026-08-10T12:34:56Z" in output, (
        f"{rel(DIAGNOSE)} read a run that never finished and said nothing about "
        f"it:\n{output}"
    )
    # Before the summary, not after: under a "Quality Check Summary" heading the
    # same words read as a note about the summary rather than about which run it
    # came from.
    warning = output.find("2026-08-10T12:34:56Z")
    heading = output.find("Quality Check Summary")
    assert heading == -1 or warning < heading, (
        f"{rel(DIAGNOSE)} reported the summary before saying it belongs to an "
        f"earlier run:\n{output}"
    )


def test_an_aborted_run_with_no_summary_at_all_says_so(tmp_path):
    """A first-ever run that aborts leaves a marker and nothing else.

    "No quality run found to diagnose" is wrong there and sends the developer to
    do the thing they just did. One did run; it stopped before it could record
    anything.
    """
    output = _diagnose(tmp_path, None, started="2026-08-10T12:34:56Z")

    assert "2026-08-10T12:34:56Z" in output, (
        f"{rel(DIAGNOSE)} told a developer no run had happened, over the marker "
        f"of one that had:\n{output}"
    )


def test_the_marker_brackets_everything_that_can_abort_a_run():
    """The marker is only worth having if it spans the whole exposed stretch.

    Written after an abort path, that path leaves no marker and the stale
    summary is read as current — the defect, untouched, for whichever check
    happens to be first. Removed before the summary is written, every run looks
    aborted and the warning becomes noise to scroll past, which is the same
    failure arriving from the other side.

    Structural rather than behavioural: exercising it for real means running the
    gate from inside the gate's own test suite, and pointing it at the developer's
    diagnostics directory would destroy the run they are reading.
    """
    lines = (ROOT / "bin" / "run-quality-checks.sh").read_text(encoding="utf-8").splitlines()

    def first(predicate) -> int:
        return next(
            (n for n, line in enumerate(lines, 1) if predicate(line.strip())),
            -1,
        )

    write = first(lambda line: line.startswith("printf") and ".run-in-progress" in line)
    remove = first(lambda line: line.startswith("rm -f") and ".run-in-progress" in line)
    summary = first(lambda line: line.startswith("cat >") and "quality-summary.json" in line)
    aborts = [n for n, line in enumerate(lines, 1) if line.strip() == "exit 1"]

    assert write > 0, "nothing writes the in-progress marker"
    assert remove > 0, "nothing removes the in-progress marker"
    assert summary > 0, "the summary write moved; this guard needs its new shape"

    early = [n for n in aborts if n < write]
    assert not early, (
        "these lines abort the run before it records that one is in progress, so "
        f"a stale summary is read as current: {early}. Move the marker write "
        "above them. (A function *defined* above it but called later is a false "
        "positive — teach this guard about it rather than moving the write down.)"
    )
    assert remove > summary, (
        f"the marker is removed at line {remove}, before the summary is written "
        f"at line {summary}. Between the two, an abort leaves a directory that "
        "looks closed out and holds the previous run's verdict."
    )
