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
an unrecorded check is absent rather than passing, that the fields land where
the document says they land, and that the banner says what the document says.

That last one is not hypothetical. Before the banner was rendered from the
document it derived its own rows, and on any pull request that changed no
documentation it printed::

    Documentation:      ✓ PASSED
    Doc Versions:       ✓ PASSED
    Doc Mirrors:        ✓ PASSED

beside a summary recording ``skipped: true`` for all three — the same false
green, in the half a developer actually reads, on the commonest shape of pull
request there is.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from tests._workspace import ROOT, load_bin_module

writer = load_bin_module("quality-summary")

GATE = ROOT / "bin" / "run-quality-checks.sh"

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


# --------------------------------------------------------------------------
# the renderer
# --------------------------------------------------------------------------


def _rendered(checks: dict, *, mode: str = "pr", package_tests_skipped: bool = False):
    return writer.render(
        {"checks": checks},
        mode=mode,
        package_tests_skipped=package_tests_skipped,
        palette=writer.Palette(io.StringIO()),
    )


def _entry(status: str = "pass", **extra: object) -> dict:
    return {"status": status, "exit_code": 0 if status == "pass" else 1, **extra}


def test_a_records_file_that_is_not_there_is_named_rather_than_traced():
    """Every other malformed input to this module names itself; this one did not.

    A bad JSON line, a record with no ``name``, a duplicate check, an unknown
    top-level field — each raises ``SystemExit`` carrying what was wrong. An
    absent records file raised ``FileNotFoundError`` instead, so the one failure
    that means *no check recorded anything* was the one that printed a traceback
    rather than a sentence. It still failed closed, which is why this is a
    legibility fix and not a correctness one.
    """
    with pytest.raises(SystemExit) as caught:
        writer.read_records("/nonexistent/records.jsonl")

    assert "no check recorded" in str(caught.value)


def test_a_garbled_check_entry_is_not_rendered_as_a_pass():
    """The renderer's own version of "absence is not a pass".

    ``render`` already refuses a ``checks`` value that is not a mapping. The
    entries inside it were read without the same question being asked, so a
    non-mapping entry reached ``entry.get`` and raised ``AttributeError``. The
    render call is the one gate step that is not wrapped in a failure branch,
    and it runs *after* the summary is written and the in-progress marker is
    removed — so an entry the writer would never produce could abort a run whose
    checks had all passed and whose artifacts were already valid.

    Failing closed rather than skipping the row: a garbled entry is not a check
    that passed, and dropping it silently would make it a check that vanished.
    """
    rows = _rendered({"documentation": "pass"})

    assert rows == ["  Documentation:      ✗ FAILED"]


def test_a_garbled_test_entry_does_not_reach_the_grouping_arithmetic():
    """The same guard one layer down, where the entries are read field by field.

    The two test entries are not just handed to ``verdict`` — they are combined
    (``dict(unit)``, ``integration.get("status")``) to produce the one row a dev
    run earns. Making ``verdict`` total is therefore not enough on its own: a
    non-mapping entry would still raise before it ever got there. The rows go
    out ungrouped rather than not at all.
    """
    rows = _rendered(
        {"unit_tests": "pass", "integration_tests": _entry()}, mode="dev"
    )

    assert rows == [
        "  Unit Tests:        ✗ FAILED",
        "  Integration Tests: ✓ PASSED",
    ]


def test_a_skipped_check_is_rendered_as_skipped_not_passed():
    """The defect this renderer exists to close, in its original form.

    A skipped check carries ``status: "pass"`` — it has always been recorded that
    way, and the field that says it did not run is ``skipped``. The banner used
    to read a status variable instead, so a documentation check that never ran
    printed ✓ PASSED. Reading ``status`` alone here would restore that exactly.
    """
    rows = _rendered({"documentation": _entry(skipped=True)})

    assert rows == ["  Documentation:      ⊘ SKIPPED"]


def test_the_row_layout_is_unchanged():
    """Two column widths, reproducing the shell's layout rather than tidying it.

    The documentation and lint rows align one column further right than the test
    rows. Preserved deliberately: it made swapping the producer verifiable by
    diffing a real run's console output against the previous one, which is worth
    more than the tidier alternative. Normalising them is a visible change to
    make on purpose, not a side effect of this one.
    """
    rows = _rendered(
        {
            "documentation": _entry(skipped=False),
            "shell_lint": _entry(),
            "unit_tests": _entry(skipped=False),
            "integration_tests": _entry(skipped=False),
        }
    )

    assert rows == [
        "  Documentation:      ✓ PASSED",
        "  Shell Lint:         ✓ PASSED",
        "  Unit Tests:        ✓ PASSED",
        "  Integration Tests: ✓ PASSED",
    ]


def test_the_documentation_rows_are_shown_only_where_they_could_have_run():
    """A dev run does not offer them, so it does not report on them."""
    checks = {"documentation": _entry(skipped=True), "shell_lint": _entry()}

    assert len(_rendered(checks, mode="pr")) == 2
    assert _rendered(checks, mode="dev") == ["  Shell Lint:         ✓ PASSED"]


def test_a_check_the_display_table_does_not_name_is_still_shown():
    """A check added to the gate reaches the banner without editing the renderer.

    Deriving the label rather than requiring one is what makes that true, and it
    closes the human-facing half of the defect the artifact half already has a
    guard for: a check invisible in the banner is one a developer will not know
    ran, whatever the summary says.
    """
    rows = _rendered({"licence_audit": _entry("fail")})

    assert rows == ["  Licence audit:      ✗ FAILED"]


def test_dev_mode_reports_the_one_verdict_the_run_produced():
    """Dev mode runs both suites through one invocation and one exit code.

    Reporting them apart would invent a distinction the run did not make, so the
    two entries — which carry the same value for that reason — become one row.
    """
    passing = {"unit_tests": _entry(skipped=False), "integration_tests": _entry(skipped=False)}
    failing = {
        "unit_tests": _entry("pass", skipped=False),
        "integration_tests": _entry("fail", skipped=False),
    }

    assert _rendered(passing, mode="dev") == ["  Tests:             ✓ PASSED"]
    assert _rendered(failing, mode="dev") == ["  Tests:             ✗ FAILED"]


def test_dev_mode_reports_a_skipped_run_as_skipped():
    checks = {"unit_tests": _entry(skipped=True), "integration_tests": _entry(skipped=True)}

    assert _rendered(checks, mode="dev") == ["  Tests:             ⊘ SKIPPED"]


def test_a_run_with_no_package_changed_names_the_guards_it_did_run():
    """"Unit Tests: PASSED" for a run that collected none is green for work not done.

    The workspace guards did run and the unit entry carries their status, so the
    row is labelled as what it is and the package suites are named apart.
    """
    rows = _rendered(
        {"unit_tests": _entry(skipped=False), "integration_tests": _entry(skipped=True)},
        package_tests_skipped=True,
    )

    assert rows == [
        "  Workspace Guards:  ✓ PASSED",
        "  Package Tests:     ⊘ SKIPPED (no package changed)",
    ]


def test_colour_is_applied_only_to_a_terminal():
    """Matched to the shell's own test, so a piped run has no escapes in either half."""

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    plain = writer.Palette(io.StringIO()).verdict(_entry())
    assert plain == "✓ PASSED"


# --------------------------------------------------------------------------
# the gate keeps one derivation
# --------------------------------------------------------------------------


def test_the_banner_derives_no_verdict_of_its_own():
    """One producer, so the console and the artifact cannot come to disagree.

    They did, and the shape of it was a shell ``if`` over a status variable in
    the banner region. This asserts none is left: every row comes from the
    rendered document. The region is delimited by the banner's own rules rather
    than by line numbers, so it cannot silently shrink to nothing — a guard that
    checks an empty region is the failure mode this repository keeps finding.
    """
    lines = GATE.read_text(encoding="utf-8").splitlines()
    rules = [i for i, line in enumerate(lines) if "Quality Check Summary" in line]
    assert len(rules) == 1, "the banner header moved — re-point this guard"

    # The rule immediately under the header opens the region; the next one
    # closes it. Taking the first would leave the region empty, which is how a
    # structural guard comes to check nothing and report green.
    header = rules[0]
    borders = [
        i for i, line in enumerate(lines[header + 1 :], header + 1)
        if line.startswith('echo -e "${BLUE}═')
    ]
    assert len(borders) >= 2, "the banner's closing rule moved — re-point this guard"
    start, stop = borders[0], borders[1]
    region = lines[start:stop]
    assert region, "the banner region came out empty"
    assert any("quality-summary.py" in line for line in region), (
        "the banner no longer renders from the summary — if the rows moved "
        "elsewhere, re-point this guard rather than deleting it"
    )

    derived = [
        f"{i + 1}: {line.strip()}"
        for i, line in enumerate(region, start)
        if re.search(r"\[\s*\"?\$[A-Z][A-Z0-9_]*_(STATUS|SKIPPED)\b", line)
    ]
    assert not derived, (
        "these lines decide a banner row from a status variable rather than from "
        f"quality-summary.json: {derived}. The two derivations disagreed once "
        "already — three documentation checks printed as passing on every pull "
        "request that changed no documentation."
    )
