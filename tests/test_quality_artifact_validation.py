"""Behavioural tests for how ``bin/validate-quality-artifacts.sh`` reads a summary.

That script is what CI runs instead of re-running the gate, so what it can read
decides what CI can see. Nothing had ever executed it: the guards in
``test_quality_gate_accounting.py`` inspect its *source*, which is why it
accumulated seven line-offset greps —
``grep -A2 '"unit_tests"' | grep '"status"'`` for a status,
``grep -A3`` for a skipped flag — without anything noticing.

Those greps read the file by POSITION. JSON objects are unordered by definition,
so a field added or moved above the one a window wanted pushed it out, the grep
returned nothing, and the validator rejected an artifact it had merely failed to
read — reporting ``Unit tests:`` with an empty status as the reason. The
reproduction is ``_REORDERED`` below: valid, passing, and unreadable by the old
parser.

Each test drives ``--read-summary``, which prints the projection the main path
consumes. It is the same function, so these assert about the reader the
validator actually uses rather than a second copy of it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests._workspace import ROOT, rel

VALIDATOR = ROOT / "bin" / "validate-quality-artifacts.sh"


def _read_summary(path: Path) -> dict[str, object]:
    """Run the validator's reader and parse its tab-delimited projection."""
    result = subprocess.run(
        ["bash", str(VALIDATOR), "--read-summary", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"reader exited {result.returncode}: {result.stderr}"

    overall, checks, error = "", {}, None
    for line in result.stdout.splitlines():
        if not line:
            continue
        kind, _, rest = line.partition("\t")
        if kind == "OVERALL":
            overall = rest
        elif kind == "ERROR":
            error = rest
        elif kind == "CHECK":
            name, status, skipped, label = rest.split("\t")
            checks[name] = {
                "status": status,
                "skipped": skipped == "true",
                "label": label,
            }
    return {"overall": overall, "checks": checks, "error": error, "raw": result.stdout}


def _write(path: Path, doc: object) -> Path:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


#: The producer's key order today: status first, durations appended last.
_PRODUCER_ORDER = {
    "overall_status": "PASS",
    "checks": {
        "unit_tests": {
            "status": "pass",
            "exit_code": 0,
            "skipped": False,
            "duration_seconds": 402,
        },
        "workflow_lint": {
            "status": "fail",
            "exit_code": 1,
            "tool": "lint-workflows.sh",
            "duration_seconds": 1,
        },
    },
}

#: The same data with each check's keys in a different order. Semantically
#: identical JSON; the old positional greps read every status here as empty.
_REORDERED = {
    "overall_status": "PASS",
    "checks": {
        "unit_tests": {
            "duration_seconds": 402,
            "skipped": False,
            "status": "pass",
            "exit_code": 0,
        },
        "workflow_lint": {
            "tool": "lint-workflows.sh",
            "duration_seconds": 1,
            "exit_code": 1,
            "status": "fail",
        },
    },
}


def test_a_status_reads_the_same_whatever_order_the_keys_are_in(tmp_path):
    """The reproduction: key order must not change what the validator sees.

    Before this was a JSON parse, the second form read as no status at all, so a
    valid passing artifact failed CI and the reported reason was a blank.
    """
    produced = _read_summary(_write(tmp_path / "produced.json", _PRODUCER_ORDER))
    reordered = _read_summary(_write(tmp_path / "reordered.json", _REORDERED))

    assert produced["checks"] == reordered["checks"], (
        "the same summary read differently depending on key order, which is the "
        "positional-parsing defect this reader replaced"
    )
    assert reordered["checks"]["unit_tests"]["status"] == "pass"
    assert reordered["checks"]["workflow_lint"]["status"] == "fail"


def test_a_failing_check_is_never_read_as_absent(tmp_path):
    """A status the reader cannot find is indistinguishable from a passing run.

    ``workflow_lint`` is recorded as failing in both fixtures. The old greps
    returned an empty string for it, and an empty status is not "fail" — it only
    happened to fail the gate because empty is also not "pass".
    """
    for name, doc in (("produced", _PRODUCER_ORDER), ("reordered", _REORDERED)):
        summary = _read_summary(_write(tmp_path / f"{name}.json", doc))
        assert summary["checks"]["workflow_lint"]["status"] == "fail", (
            f"{name}: a recorded failure must read back as a failure"
        )


def test_every_recorded_check_is_reported_not_a_hand_kept_three(tmp_path):
    """The old reader named unit_tests, integration_tests and workflow_lint only.

    The other five reached CI as a bare ``Overall status: FAIL`` naming nothing.
    """
    doc = {
        "overall_status": "PASS",
        "checks": {
            name: {"status": "pass", "exit_code": 0}
            for name in (
                "documentation",
                "documentation_versions",
                "documentation_mirrors",
                "validation",
                "shell_lint",
                "workflow_lint",
                "unit_tests",
                "integration_tests",
            )
        },
    }
    summary = _read_summary(_write(tmp_path / "all.json", doc))
    assert set(summary["checks"]) == set(doc["checks"]), (
        "a check the summary records but the reader drops is one CI cannot see"
    )


def test_a_check_name_becomes_a_readable_label(tmp_path):
    """Derived from the name, so a check added later is not left unlabelled."""
    doc = {"overall_status": "PASS", "checks": {"shell_lint": {"status": "pass"}}}
    summary = _read_summary(_write(tmp_path / "label.json", doc))
    assert summary["checks"]["shell_lint"]["label"] == "Shell lint"


def test_a_skipped_check_reads_as_skipped(tmp_path):
    """Only ``true`` counts. A missing flag is not a skip."""
    doc = {
        "overall_status": "PASS_WITH_SKIPS",
        "checks": {
            "integration_tests": {"status": "pass", "skipped": True},
            "unit_tests": {"status": "pass", "skipped": False},
            "shell_lint": {"status": "pass"},
        },
    }
    summary = _read_summary(_write(tmp_path / "skips.json", doc))
    assert summary["checks"]["integration_tests"]["skipped"] is True
    assert summary["checks"]["unit_tests"]["skipped"] is False
    assert summary["checks"]["shell_lint"]["skipped"] is False


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("malformed", "{not json"),
        ("truncated", '{"overall_status": "PASS", "checks": {'),
        ("not_an_object", "[1, 2, 3]"),
        ("empty", ""),
    ],
)
def test_an_unreadable_summary_says_so_rather_than_reading_as_empty(
    tmp_path, name, content
):
    """Fails loud, not blank.

    "Could not parse the attestation" and "the attestation says the run passed"
    must not reach the same verdict, and a reader that returns nothing for both
    leaves the caller unable to tell them apart.
    """
    path = tmp_path / f"{name}.json"
    path.write_text(content, encoding="utf-8")
    summary = _read_summary(path)

    assert summary["error"], f"{name}: expected an ERROR line, got {summary['raw']!r}"
    assert not summary["checks"], f"{name}: no checks should be reported"


def test_a_missing_summary_is_an_error_not_a_silent_pass(tmp_path):
    summary = _read_summary(tmp_path / "does-not-exist.json")
    assert summary["error"], "a summary that is not there cannot report a passing run"


def test_the_reader_prints_the_projection_and_nothing_else(tmp_path):
    """No banner on this path.

    The validator's decorative header goes to stdout on the normal path. If it
    were printed here too, every caller would have to strip it, and the first one
    to forget gets a parse failure that reads as a data problem.
    """
    result = subprocess.run(
        ["bash", str(VALIDATOR), "--read-summary", str(_write(tmp_path / "s.json", _PRODUCER_ORDER))],
        capture_output=True,
        text=True,
        check=False,
    )
    kinds = {line.split("\t", 1)[0] for line in result.stdout.splitlines() if line}
    assert kinds <= {"OVERALL", "CHECK", "ERROR"}, (
        f"unexpected output on the read-summary path: {result.stdout!r}"
    )


def test_the_summary_is_not_read_by_line_offset_again(tmp_path):
    """Recurrence guard for the defect class, not for one instance of it.

    ``grep -A<n>`` against the summary is the shape that made field order
    load-bearing. It reads whatever sits n lines below a match, which is a
    position rather than a name, so it silently returns the wrong thing the
    moment the producer adds a field. Nothing but a parser should read this file.
    """
    source = VALIDATOR.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "quality-summary.json" in line and ("grep -A" in line or "grep -B" in line)
    ]
    assert not offenders, (
        f"{rel(VALIDATOR)} reads quality-summary.json by line offset again:\n  "
        + "\n  ".join(offenders)
        + "\nUse read_summary(), which parses it as JSON."
    )
