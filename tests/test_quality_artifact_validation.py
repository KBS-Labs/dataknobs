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
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests._workspace import ROOT, tracked_shell_files

VALIDATOR = ROOT / "bin" / "validate-quality-artifacts.sh"

#: The projection's field delimiter. See bin/read-quality-summary.py for why it
#: is not a tab: `read` collapses runs of IFS whitespace and drops empty fields.
SEP = "\x1f"


def _read_summary_raw(path: Path) -> str:
    """Run the validator's reader and return its projection verbatim."""
    result = subprocess.run(
        ["bash", str(VALIDATOR), "--read-summary", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"reader exited {result.returncode}: {result.stderr}"
    return result.stdout


def _read_summary(path: Path) -> dict[str, Any]:
    """Run the validator's reader and parse its delimited projection."""
    raw = _read_summary_raw(path)

    overall, checks, error = "", {}, None
    for line in raw.splitlines():
        if not line:
            continue
        kind, _, rest = line.partition(SEP)
        if kind == "OVERALL":
            overall = rest
        elif kind == "ERROR":
            error = rest
        elif kind == "CHECK":
            name, status, skipped, exit_code, tool, label = rest.split(SEP)
            checks[name] = {
                "status": status,
                "skipped": skipped == "true",
                "exit_code": exit_code,
                "tool": tool,
                "label": label,
            }
    return {"overall": overall, "checks": checks, "error": error, "raw": raw}


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


def test_a_status_reads_the_same_whatever_order_the_keys_are_in(tmp_path: Path) -> None:
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


def test_a_failing_check_is_never_read_as_absent(tmp_path: Path) -> None:
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


def test_every_recorded_check_is_reported_not_a_hand_kept_three(tmp_path: Path) -> None:
    """The old reader named unit_tests, integration_tests and workflow_lint only.

    The other five reached CI as a bare ``Overall status: FAIL`` naming nothing.
    """
    doc: dict[str, Any] = {
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


def test_a_check_name_becomes_a_readable_label(tmp_path: Path) -> None:
    """Derived from the name, so a check added later is not left unlabelled."""
    doc: dict[str, Any] = {"overall_status": "PASS", "checks": {"shell_lint": {"status": "pass"}}}
    summary = _read_summary(_write(tmp_path / "label.json", doc))
    assert summary["checks"]["shell_lint"]["label"] == "Shell lint"


def test_a_skipped_check_reads_as_skipped(tmp_path: Path) -> None:
    """Only ``true`` counts. A missing flag is not a skip."""
    doc: dict[str, Any] = {
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
    tmp_path: Path, name: str, content: str
) -> None:
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


def test_a_missing_summary_is_an_error_not_a_silent_pass(tmp_path: Path) -> None:
    summary = _read_summary(tmp_path / "does-not-exist.json")
    assert summary["error"], "a summary that is not there cannot report a passing run"


def test_the_reader_prints_the_projection_and_nothing_else(tmp_path: Path) -> None:
    """No banner on this path.

    The validator's decorative header goes to stdout on the normal path. If it
    were printed here too, every caller would have to strip it, and the first one
    to forget gets a parse failure that reads as a data problem.
    """
    result = subprocess.run(
        [
            "bash",
            str(VALIDATOR),
            "--read-summary",
            str(_write(tmp_path / "s.json", _PRODUCER_ORDER)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    kinds = {line.split(SEP, 1)[0] for line in result.stdout.splitlines() if line}
    assert kinds <= {"OVERALL", "CHECK", "ERROR"}, (
        f"unexpected output on the read-summary path: {result.stdout!r}"
    )


#: Commands that read a file as text, or as a second parse of JSON that already
#: has a parser. ``jq`` is here for two reasons: it is a build-time dependency
#: this repository does not pin, so every use needs a branch for its absence,
#: and every such branch is a second reader that drifts from the first. That is
#: not hypothetical — the ``jq`` path in the diagnostics tool asked for two
#: checks the producer has never emitted, while the no-``jq`` path left every
#: status variable unset, so it reported four failures for a passing run.
_TEXT_READERS = ("grep ", "sed ", "awk ", "cut ", "head ", "tail ", "jq ")

#: Reads that do not parse the file: existence, mtime and modification time.
_NOT_A_READ = ("-nt ", "-ot ", "! -f ", "-f ", "date -r")


def test_nothing_reads_the_summary_except_the_parser() -> None:
    """Recurrence guard for the defect class, not for one instance of it.

    ``grep -A<n>`` against the summary is the shape that made field order
    load-bearing: it reads whatever sits n lines below a match, which is a
    position rather than a name, so it silently returns the wrong thing the
    moment the producer adds a field.

    Scoped to every tracked shell file rather than to the validator. The narrow
    version of this test passed for as long as the diagnostics tool held a
    second reader of the same file, because the second reader was in the file it
    did not look at. A guard scoped to where you already looked reports green
    over the place you didn't.
    """
    offenders = []
    for name in tracked_shell_files():
        path = ROOT / name
        if path.name == "read-quality-summary.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "quality-summary.json" not in line or line.lstrip().startswith("#"):
                continue
            if any(token in line for token in _NOT_A_READ):
                continue
            if any(token in line for token in _TEXT_READERS):
                offenders.append(f"{name}:{number}: {line.strip()}")

    assert not offenders, (
        "quality-summary.json is read by something other than its parser:\n  "
        + "\n  ".join(offenders)
        + "\nUse bin/read-quality-summary.py, which parses it as JSON once."
    )


def test_every_consumer_of_the_projection_names_every_field(tmp_path: Path) -> None:
    """A shell ``read`` with too few variables is silently wrong, not an error.

    ``read -r kind name status skipped label`` against a six-field record does
    not fail: bash assigns the leftover fields to the final variable, so ``label``
    silently becomes ``"<label><TAB><tool>"``. Adding a field to the projection
    therefore corrupts the last-named field of every consumer that was not
    updated with it, in a way no consumer can detect at run time.

    So the count is asserted here instead, against what the projection actually
    emits rather than against a number written down twice.
    """
    projection = _read_summary_raw(_write(tmp_path / "s.json", _PRODUCER_ORDER))
    widest = max(
        len(line.split(SEP)) for line in projection.splitlines() if line.startswith("CHECK")
    )

    # `kind` first is the convention that makes a consumer findable: every loop
    # over the projection has to name that field to branch on the record type.
    #
    # Naming `label` is what makes one a CHECK consumer. It is the last field, so
    # it is the one that silently absorbs whatever a short read leaves over —
    # which makes it both the symptom and a precise way to ask the question. A
    # loop that filters to OVERALL or META and names two or three fields is not
    # reading a CHECK record and is left alone.
    consumers = []
    for name in tracked_shell_files():
        for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"read -r kind\b([^;|#]*)", line)
            if not match:
                continue
            fields = ("kind" + match.group(1)).split()
            if any(field.lstrip("_") == "label" for field in fields):
                consumers.append((f"{name}:{number}", len(fields), line.strip()))

    assert consumers, "no consumer of the projection found — has the loop shape changed?"
    wrong = [
        f"{where}: names {named} of {widest} fields\n    {text}"
        for where, named, text in consumers
        if named != widest
    ]
    assert not wrong, (
        "these read the projection with the wrong number of variables:\n  "
        + "\n  ".join(wrong)
        + f"\nA CHECK record has {widest} fields; name all of them, using `_` "
        "for the ones the loop ignores."
    )


# --------------------------------------------------------------------------
# The main path — everything above drives --read-summary, which returns before
# the main path begins.
# --------------------------------------------------------------------------

#: What the validator requires to be present before it will read anything. Their
#: contents do not matter to the assertions below; their existence does, because
#: the script exits at the missing-files check.
_REQUIRED_ARTIFACTS = (
    "environment.json",
    "signature.sha256",
    "unit-test-results.xml",
)


def _artifacts_dir(tmp_path: Path, summary: object | str) -> Path:
    """Build a directory shaped like ``.quality-artifacts/`` around one summary."""
    directory = tmp_path / "artifacts"
    directory.mkdir()
    for name in _REQUIRED_ARTIFACTS:
        (directory / name).write_text("", encoding="utf-8")

    target = directory / "quality-summary.json"
    if isinstance(summary, str):
        target.write_text(summary, encoding="utf-8")
    else:
        _write(target, summary)
    return directory


def _validate(directory: Path) -> str:
    """Run the validator's main path over ``directory`` and return its report.

    The exit code is deliberately not asserted on. This script also revalidates
    package content hashes against the working tree, which is a real check with
    a real verdict that has nothing to do with the summary — so the status lines
    are the subject here, not the overall result.
    """
    result = subprocess.run(
        ["bash", str(VALIDATOR), "--from", str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def test_the_overall_status_reaches_the_main_path(tmp_path: Path) -> None:
    r"""A passing summary must be reported as passing.

    The projection moved from tab-delimited to ``\037`` records. Three
    ``while IFS`` loops moved with it and two ``sed`` extractions did not, so
    ``OVERALL_STATUS`` resolved to the empty string on every run and the
    validator reported ``Overall status: `` — failing CI on every pull request,
    whatever the artifacts said.

    Nothing caught it because every test above drives ``--read-summary``, which
    exits before this code. The reader was right; its callers were not.
    """
    report = _validate(_artifacts_dir(tmp_path, _PRODUCER_ORDER))

    assert "Overall status: PASS" in report, (
        "the validator did not read PASS from a passing summary:\n" + report
    )


def test_an_unreadable_summary_is_named_on_the_main_path(tmp_path: Path) -> None:
    """A summary that cannot be parsed must not read as one that says nothing.

    These are the two verdicts the script's own comment says must never meet:
    "could not parse it" and "it says the run passed". With the ERROR extraction
    matching a separator the projection no longer uses, the parse error was
    dropped and the run fell through to an empty overall status — so a corrupt
    artifact and a passing one produced the same message.
    """
    report = _validate(_artifacts_dir(tmp_path, "[not, an, object]"))

    assert "Could not read quality-summary.json" in report, (
        "an unparseable summary was not reported as unreadable:\n" + report
    )


def test_the_projection_is_split_never_pattern_matched() -> None:
    """Recurrence guard for the class the two stale ``sed`` calls belong to.

    ``test_nothing_reads_the_summary_except_the_parser`` covers readers of the
    *file*. These two read the *projection*, on lines that never mention
    ``quality-summary.json``, so that guard could not see them — and the
    separator changed underneath them silently.

    The projection is delimited text with a declared separator. The one correct
    way to read it is to let ``IFS`` split it; spelling the separator into a
    regex is a second copy of that decision, and the two do not change together.
    The carriers are discovered from the assignment rather than named here, so a
    third one is covered the day it is written.
    """
    offenders = []
    for name in tracked_shell_files():
        text = (ROOT / name).read_text(encoding="utf-8")
        carriers = set(re.findall(r"(\w+)=\$\(\s*read_summary\b", text)) | set(
            re.findall(r"(\w+)=\$\([^)]*--read-summary", text)
        )
        if not carriers:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if not any(f"${carrier}" in line or f"${{{carrier}}}" in line for carrier in carriers):
                continue
            if any(token in line for token in _TEXT_READERS):
                offenders.append(f"{name}:{number}: {line.strip()}")

    assert not offenders, (
        "the projection is being pattern-matched rather than split:\n  "
        + "\n  ".join(offenders)
        + "\nRead it with `while IFS=\"$(printf '\\037')\" read -r ...`, which "
        "takes the separator from one place."
    )
