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


def _diagnose(tmp_path: Path, summary: dict[str, object]) -> str:
    """Run the tool over a synthetic run directory and return what it printed."""
    (tmp_path / "quality-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
