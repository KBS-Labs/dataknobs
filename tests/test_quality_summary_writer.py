"""Guards for the producer of ``quality-summary.json`` and of the banner.

The summary used to be a shell heredoc over variables initialised near the top
of ``bin/run-quality-checks.sh``, and the defect in that shape shipped twice:
**a status variable has a default, and the default is a verdict.** ``0`` renders
as ``"pass"``, so a check no code path assigned reported as one that ran and
passed. The duration fields escaped only by defaulting to ``null`` — the absence
of a measurement rather than a passing one.

The producer is now a writer over records each check appends as it runs, so the
defect closes by construction rather than by a guard that reads the producer's
source for shapes it must not have. The tests here check the construction: that
an unrecorded check is absent rather than passing, and that the fields land
where the document says they land.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._workspace import load_bin_module

writer = load_bin_module("quality-summary")


#: The ``checks`` object of a real gate run, as the shell heredoc produced it.
#: Copied from the committed artifact at the commit before the writer replaced
#: it, and kept verbatim: it is the reference the swap is measured against.
HEREDOC_CHECKS = {
    "documentation": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "tool": "mkdocs", "duration_seconds": 25,
    },
    "documentation_versions": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "tool": "docs-update-versions.sh", "duration_seconds": 0,
    },
    "documentation_mirrors": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "tool": "docs-mirror-check.py", "duration_seconds": 0,
    },
    "validation": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "tool": "validate.sh", "duration_seconds": 39,
    },
    # No "skipped": nothing gates these two, so there is no state in which they
    # did not run. The absence is part of what the writer has to reproduce.
    "shell_lint": {
        "status": "pass", "exit_code": 0,
        "tool": "lint-shell.sh", "duration_seconds": 4,
    },
    "workflow_lint": {
        "status": "pass", "exit_code": 0,
        "tool": "lint-workflows.sh", "duration_seconds": 1,
    },
    # No "tool": a suite is not one tool's verdict. The extra span is the
    # workspace guards, folded into the unit status but timed separately.
    "unit_tests": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "duration_seconds": 78, "workspace_guards_seconds": 27,
    },
    "integration_tests": {
        "status": "pass", "exit_code": 0, "skipped": False,
        "duration_seconds": 54,
    },
}

#: Every top-level field, with a value of the right JSON type. Built from the
#: writer's own tuple so a field added there is supplied here rather than
#: turning every build below into a "missing field" error that says nothing
#: about what the test meant to check.
STRINGS = {"timestamp", "overall_status", "run_mode", "environment", "packages"}


def _metadata(omit: str | None = None, **overrides: object) -> list[str]:
    """``--str``/``--json`` arguments covering every top-level field.

    ``omit`` drops a field's flag *and* its value: dropping the value alone
    would leave the flag to swallow whatever followed, and the build would then
    fail for a reason the test did not mean to create.
    """
    stock: dict[str, object] = {
        "timestamp": "2026-08-10T00:00:00Z",
        "overall_status": "PASS",
        "run_mode": "pr",
        "environment": "host",
        "packages": "all",
        "tested_packages": [],
        "coverage_percent": 91.5,
        "package_hashes": {},
        "workspace_hashes": {},
        "total_seconds": 12,
    }
    stock.update(overrides)
    args = []
    for field in writer.TOP_LEVEL_FIELDS:
        if field not in stock or field == omit:
            continue
        if field in STRINGS:
            args += ["--str", f"{field}={stock[field]}"]
        else:
            args += ["--json", f"{field}={json.dumps(stock[field])}"]
    return args


def _record(records: Path, name: str, code: int, *rest: str) -> None:
    writer.main(
        ["quality-summary.py", "record", "--records", str(records), "--name", name,
         "--exit-code", str(code), *rest]
    )


def _build(tmp_path: Path, records: Path, **overrides: object) -> dict:
    output = tmp_path / "quality-summary.json"
    writer.main(
        ["quality-summary.py", "build", "--records", str(records),
         "--output", str(output), *_metadata(**overrides)]
    )
    return json.loads(output.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------


def test_a_check_that_records_nothing_is_absent_rather_than_passing(tmp_path):
    """The whole reason the producer changed shape.

    Under the heredoc every check had a status variable initialised to ``0``, so
    a check that no path assigned was reported as one that ran and passed. Here
    there is no variable and no default: the check that wrote no record simply
    is not in the document, and a reader asking whether it passed gets nothing
    rather than a yes.
    """
    records = tmp_path / "records.jsonl"
    _record(records, "shell_lint", 0, "--duration", "3", "--tool", "lint-shell.sh")

    document = _build(tmp_path, records)

    assert set(document["checks"]) == {"shell_lint"}
    assert "unit_tests" not in document["checks"]


def test_a_failing_check_is_reported_as_failing(tmp_path):
    records = tmp_path / "records.jsonl"
    _record(records, "shell_lint", 2, "--duration", "3")

    entry = _build(tmp_path, records)["checks"]["shell_lint"]

    assert entry["status"] == "fail"
    assert entry["exit_code"] == 2


def test_an_unmeasured_duration_is_null_and_an_instant_one_is_zero(tmp_path):
    """``0`` is a measurement; ``null`` is the absence of one.

    Conflating them is how this defect class started: a stage that never ran had
    a duration of ``0`` and was indistinguishable from one that ran instantly.
    ``--duration null`` is what a shell variable holding no measurement spells,
    and it must not arrive as a number.
    """
    records = tmp_path / "records.jsonl"
    _record(records, "skipped_check", 0, "--duration", "null")
    _record(records, "instant_check", 0, "--duration", "0")

    checks = _build(tmp_path, records)["checks"]

    assert checks["skipped_check"]["duration_seconds"] is None
    assert checks["instant_check"]["duration_seconds"] == 0


def test_the_fields_of_a_check_land_in_the_documented_order(tmp_path):
    """Order is for the reader of the raw file; the field set is the contract.

    ``skipped`` and ``tool`` are present only when the record carries them, which
    is how ``shell_lint`` has no ``skipped`` — it cannot be skipped — and
    ``unit_tests`` has no ``tool``, being no single tool's verdict. A field that
    can hold only one value is one more thing to get wrong.
    """
    records = tmp_path / "records.jsonl"
    _record(
        records, "unit_tests", 0,
        "--skipped", "false", "--duration", "41",
        "--field", "workspace_guards_seconds=7",
    )
    _record(records, "shell_lint", 0, "--tool", "lint-shell.sh", "--duration", "3")

    checks = _build(tmp_path, records)["checks"]

    assert list(checks["unit_tests"]) == [
        "status", "exit_code", "skipped", "duration_seconds",
        "workspace_guards_seconds",
    ]
    assert list(checks["shell_lint"]) == [
        "status", "exit_code", "tool", "duration_seconds",
    ]


def test_the_checks_appear_in_the_order_they_ran(tmp_path):
    records = tmp_path / "records.jsonl"
    for name in ("workflow_lint", "shell_lint", "validation"):
        _record(records, name, 0, "--duration", "1")

    document = _build(tmp_path, records)

    assert list(document["checks"]) == ["workflow_lint", "shell_lint", "validation"]


def test_two_records_for_one_check_are_refused(tmp_path):
    """Two answers to one question, and picking either would hide a bug.

    A second record means two sites believe they ran the same check, so one of
    them is describing a run that did not happen. Last-one-wins would make that
    invisible for as long as the two agreed.
    """
    records = tmp_path / "records.jsonl"
    _record(records, "shell_lint", 0, "--duration", "3")
    _record(records, "shell_lint", 1, "--duration", "4")

    with pytest.raises(SystemExit, match="already recorded"):
        _build(tmp_path, records)


def test_a_forgotten_top_level_field_fails_the_build(tmp_path):
    """Rather than quietly leaving it out of the artifact CI validates."""
    records = tmp_path / "records.jsonl"
    _record(records, "shell_lint", 0, "--duration", "3")
    output = tmp_path / "quality-summary.json"

    with pytest.raises(SystemExit, match="coverage_percent"):
        writer.main(
            ["quality-summary.py", "build", "--records", str(records),
             "--output", str(output), *_metadata(omit="coverage_percent")]
        )


def test_a_top_level_field_the_writer_does_not_know_is_refused(tmp_path):
    """A field with no declared position has no position, so it is not guessed."""
    records = tmp_path / "records.jsonl"
    _record(records, "shell_lint", 0, "--duration", "3")
    output = tmp_path / "quality-summary.json"

    with pytest.raises(SystemExit, match="does not know how to place"):
        writer.main(
            ["quality-summary.py", "build", "--records", str(records),
             "--output", str(output), *_metadata(), "--str", "mystery=1"]
        )


def test_the_writer_reproduces_a_heredoc_summary_exactly(tmp_path):
    """The acceptance for the swap: same run, same document.

    ``HEREDOC_CHECKS`` is the ``checks`` object of a real gate run, taken from
    the committed artifact as the shell heredoc last produced it. Turning it
    back into records and rebuilding must return it unchanged — every field, its
    type, and the two shapes where a field is absent rather than null
    (``shell_lint`` has no ``skipped``; ``unit_tests`` has no ``tool``).

    Frozen here rather than read from ``.quality-artifacts/`` on disk. A guard
    that reads the artifact would make the artifact one of its inputs, and the
    gate rewrites it on every run — so each run would invalidate the guard that
    validates it. A sample of the format this replaced does not go stale,
    because what it pins is that the replacement can still produce it.
    """
    records = tmp_path / "records.jsonl"
    for name, entry in HEREDOC_CHECKS.items():
        rest = ["--duration", "null" if entry["duration_seconds"] is None
                else str(entry["duration_seconds"])]
        if "tool" in entry:
            rest += ["--tool", entry["tool"]]
        if "skipped" in entry:
            rest += ["--skipped", "true" if entry["skipped"] else "false"]
        rest += [
            f"--field={key}={json.dumps(value)}"
            for key, value in entry.items()
            if key not in {"status", "exit_code", "skipped", "tool", "duration_seconds"}
        ]
        _record(records, name, entry["exit_code"], *rest)

    rebuilt = _build(tmp_path, records)

    assert rebuilt["checks"] == HEREDOC_CHECKS
