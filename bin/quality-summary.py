#!/usr/bin/env python3
r"""Build ``quality-summary.json`` from records the checks write as they run.

``quality-summary.json`` is the only channel by which a check's result reaches
CI, which validates the committed artifacts rather than re-running the gate. It
used to be written by a shell heredoc: ~60 lines of copy-pasted per-check
stanza, each status field an inline ``$([ "$X" -eq 0 ] && echo '"pass"')`` over
a variable initialised near the top of the script.

That shape has one defect and it shipped twice. **A status variable has a
default, and the default is a verdict.** ``0`` renders as ``"pass"``, so a check
that no code path assigned reported as one that ran and passed —
``unit_tests``/``integration_tests`` on every dev run, and the three
documentation checks on every run that did not perform them. The duration
fields escaped only by the accident of defaulting to ``null``: the absence of a
measurement rather than a passing one.

Three commands, and the split is the fix:

``record``
    Append one check's outcome, called from the site that ran it (or from the
    arm that deliberately skipped it). A check no path reaches writes no record,
    and an absent record cannot be read as a passing one, because there is no
    default left to fall through to.

``build``
    Merge the records with the run's metadata and emit the document. Every
    top-level field must be supplied — a forgotten one fails the run rather than
    vanishing from the artifact.

``render``
    Print the console status lines from the finished document, so the terminal
    banner and the artifact cannot disagree about what happened. They did: a
    run that skipped code validation because no package changed printed
    ``Code Validation: ✓ PASSED`` beside a summary recording ``skipped: true``.

Records are JSON Lines, one object per check, written to a file under the run's
output directory. It is a file rather than an in-shell accumulation so that the
outcome is durable at the moment the check produces it, and readable afterwards
by anyone diagnosing a run that never reached its summary.

The check entries appear in the document in the order the checks ran, which no
consumer depends on — ``bin/read-quality-summary.py`` sorts by name — and which
tells a reader of the raw file what the run did, in sequence.

This lives in its own file rather than inside a shell string because a program
embedded in a shell string is checked by nothing: not ruff, not mypy, not even a
syntax check until the moment it runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

#: The document's top-level fields, in the order they are written.
#:
#: Order is not load-bearing for any consumer — every reader parses the file as
#: JSON — but a stable order keeps the committed artifact's diffs readable. What
#: *is* load-bearing is that the tuple is exhaustive: ``build`` refuses a field
#: it does not name and refuses to run without one it does, so a field added to
#: the caller and not here fails loudly instead of being dropped.
TOP_LEVEL_FIELDS = (
    "timestamp",
    "overall_status",
    "run_mode",
    "environment",
    "packages",
    "tested_packages",
    "coverage_percent",
    "package_hashes",
    "workspace_hashes",
    "total_seconds",
)

#: Field order within one check entry. ``status`` is derived; the rest are
#: written only when the record carries them, which is how ``shell_lint`` has no
#: ``skipped`` (it cannot be skipped) and ``unit_tests`` has no ``tool`` (it is
#: not one tool's verdict). Anything else a record carries follows, in the order
#: the record wrote it.
CHECK_FIELD_ORDER = ("exit_code", "skipped", "tool", "duration_seconds")


# --------------------------------------------------------------------------
# argparse value types — each one fails loudly rather than substituting a default
# --------------------------------------------------------------------------


def optional_int(text: str) -> int | None:
    """An integer, or ``None`` for the literal ``null`` a shell variable holds.

    A duration is ``null`` when the stage did not run. That is not the same as
    ``0``, which is a stage that ran and took no measurable time, and keeping
    the two apart is why the duration fields never grew the defect the statuses
    did.
    """
    if text in ("", "null"):
        return None
    return int(text)


def json_bool(text: str) -> bool:
    """``true``/``false`` as a shell spells them, and nothing else.

    Deliberately not truthiness: the shell variables that reach here hold the
    strings ``"true"`` and ``"false"``, and Python considers both non-empty
    strings true. A silent misread here would put ``"skipped": true`` on a check
    that ran.
    """
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected 'true' or 'false', got {text!r}")


def key_value(text: str) -> tuple[str, str]:
    """``KEY=VALUE``, split on the first ``=`` so values may contain one."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def key_json(text: str) -> tuple[str, Any]:
    """``KEY=<json>``, with the value parsed rather than quoted as a string."""
    key, raw = key_value(text)
    try:
        return key, json.loads(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{key}: {exc}") from exc


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    """Append one check's outcome to the records file."""
    record: dict[str, Any] = {"name": args.name, "exit_code": args.exit_code}
    if args.skipped is not None:
        record["skipped"] = args.skipped
    if args.tool is not None:
        record["tool"] = args.tool
    record["duration_seconds"] = args.duration
    for key, value in args.field:
        record[key] = value

    with open(args.records, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def read_records(path: str) -> dict[str, dict[str, Any]]:
    """The records file as ``name -> check entry``, in the order recorded.

    A duplicate name is an error rather than a last-one-wins: two records for
    one check are two answers to one question, and picking either silently would
    hide whichever site is wrong.
    """
    try:
        return _read_records(path)
    except FileNotFoundError as exc:
        # The gate truncates this file into existence before the first check, so
        # its absence does not mean "nothing failed" — it means no check reached
        # its recording site at all. Named here rather than left to a traceback,
        # because every other malformed input to this module names itself.
        # Narrowed to this one OSError: the rest are genuinely exceptional and
        # read better as themselves than as a sentence about recording.
        raise SystemExit(f"{path}: no check recorded an outcome here: {exc}") from exc


def _read_records(path: str) -> dict[str, dict[str, Any]]:
    """``read_records`` without the absent-file translation. See it for the contract."""
    checks: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"{path}:{number}: not a JSON record: {exc}") from exc
            if not isinstance(record, dict) or "name" not in record:
                raise SystemExit(f"{path}:{number}: record has no 'name'")

            name = record.pop("name")
            if name in checks:
                raise SystemExit(
                    f"{path}:{number}: {name!r} was already recorded — two sites "
                    "reported this check, so one of them is describing a run that "
                    "did not happen"
                )
            if "exit_code" not in record:
                raise SystemExit(f"{path}:{number}: {name!r} has no 'exit_code'")

            entry: dict[str, Any] = {
                "status": "pass" if record["exit_code"] == 0 else "fail"
            }
            for field in CHECK_FIELD_ORDER:
                if field in record:
                    entry[field] = record.pop(field)
            entry.update(record)
            checks[name] = entry
    return checks


def cmd_build(args: argparse.Namespace) -> int:
    """Merge records and metadata into the summary document."""
    values: dict[str, Any] = dict(args.str)
    for key, value in args.json:
        values[key] = value

    unknown = sorted(set(values) - set(TOP_LEVEL_FIELDS))
    if unknown:
        raise SystemExit(
            f"build was given fields it does not know how to place: {unknown}. "
            "Add them to TOP_LEVEL_FIELDS, in the position they should occupy."
        )
    missing = [field for field in TOP_LEVEL_FIELDS if field not in values]
    if missing:
        raise SystemExit(
            f"build is missing top-level fields: {missing}. Every one is written "
            "on every run; a caller that stops supplying one must remove it here "
            "rather than leaving the field to disappear from the artifact."
        )

    document: dict[str, Any] = {field: values[field] for field in TOP_LEVEL_FIELDS}
    document["checks"] = read_records(args.records)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return 0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

#: Display rules for the checks the gate ships, as
#: ``name -> (label, width, pr_only)``.
#:
#: Labels and column widths only — every verdict comes from the document. The
#: two widths reproduce the layout the shell had: the documentation and lint
#: rows align one column further right than the test rows. Preserved rather than
#: normalised, so that swapping the producer could be verified by diffing a real
#: run's console output against the previous one and seeing nothing.
#:
#: A check absent from this table is still displayed, under a label derived from
#: its name. That is the point of deriving it: a check added to the gate appears
#: in the banner without this table being edited, and cannot be silently dropped
#: from the human-facing half the way it once could from the artifact.
ROWS = {
    "documentation": ("Documentation:", 20, True),
    "documentation_versions": ("Doc Versions:", 20, True),
    "documentation_mirrors": ("Doc Mirrors:", 20, True),
    "validation": ("Code Validation:", 20, False),
    "workflow_lint": ("Workflow Lint:", 20, False),
    "shell_lint": ("Shell Lint:", 20, False),
}

#: The test rows, which are grouped for display rather than shown one per check.
TEST_CHECKS = ("unit_tests", "integration_tests")

#: Width of the label column for every test row.
TEST_WIDTH = 19


class Palette:
    """Terminal colours, on the same condition the shell script uses.

    Matched deliberately: this runs as a child of that script with its stdout
    inherited, so the same test gives the same answer and a piped run stays
    free of escape sequences on both halves of the banner.
    """

    def __init__(self, stream: Any) -> None:
        term = os.environ.get("TERM", "")
        enabled = stream.isatty() and term not in ("", "dumb")
        self.red = "\033[0;31m" if enabled else ""
        self.green = "\033[0;32m" if enabled else ""
        self.cyan = "\033[0;36m" if enabled else ""
        self.off = "\033[0m" if enabled else ""

    def verdict(self, entry: Any) -> str:
        if not isinstance(entry, dict):
            # Not a shape the writer produces, so it arrived from a hand-edited
            # or foreign document. Reported as failed rather than raised: this
            # runs after the summary is written and the in-progress marker is
            # gone, so raising would fail a run whose checks all passed. And
            # reported rather than dropped, because a row that vanishes is a
            # check the developer never learns ran.
            return f"{self.red}✗ FAILED{self.off}"
        if entry.get("skipped") is True:
            return f"{self.cyan}⊘ SKIPPED{self.off}"
        if entry.get("status") == "pass":
            return f"{self.green}✓ PASSED{self.off}"
        return f"{self.red}✗ FAILED{self.off}"


def _row(label: str, width: int, verdict: str) -> str:
    return f"  {label:<{width}}{verdict}"


def render(
    document: dict[str, Any],
    *,
    mode: str,
    package_tests_skipped: bool,
    palette: Palette,
) -> list[str]:
    """The banner's per-check lines, in display order.

    ``mode`` and ``package_tests_skipped`` are presentation, not verdicts: the
    first decides whether the documentation rows and the split unit/integration
    rows are shown, the second whether the unit row is labelled as the workspace
    guards it is reduced to when no package changed. Neither can change what any
    row says — that comes from the document, which is the property that keeps
    this half and the artifact in step.
    """
    checks = document.get("checks")
    if not isinstance(checks, dict):
        return []

    lines = []
    for name, (label, width, pr_only) in ROWS.items():
        entry = checks.get(name)
        if entry is None or (pr_only and mode != "pr"):
            continue
        lines.append(_row(label, width, palette.verdict(entry)))

    # Every check the gate recorded that has no row above and is not one of the
    # test entries grouped below. Shown rather than dropped: the banner is the
    # half a developer reads, and a check invisible there is one they will not
    # know ran.
    for name, entry in checks.items():
        if name in ROWS or name in TEST_CHECKS:
            continue
        label = f"{name.replace('_', ' ').capitalize()}:"
        lines.append(_row(label, 20, palette.verdict(entry)))

    unit = checks.get("unit_tests")
    integration = checks.get("integration_tests")
    if unit is None or integration is None:
        return lines

    if not isinstance(unit, dict) or not isinstance(integration, dict):
        # Making verdict() total is not enough here: the grouping below reads
        # fields off both entries to produce the single row a dev run earns, so
        # a non-mapping entry would raise before reaching it. Same disposition
        # as verdict() — reported, not raised, not dropped — with the rows going
        # out ungrouped, which is the most the document supports.
        lines.append(_row("Unit Tests:", TEST_WIDTH, palette.verdict(unit)))
        lines.append(
            _row("Integration Tests:", TEST_WIDTH, palette.verdict(integration))
        )
        return lines

    if mode != "pr":
        # One line: dev mode hands test.sh no --type, so both suites go through
        # one invocation and come back as one exit code. Reporting them
        # separately would be inventing a distinction the run did not make.
        combined = dict(unit)
        if integration.get("status") != "pass":
            combined["status"] = "fail"
        if not (unit.get("skipped") and integration.get("skipped")):
            combined["skipped"] = False
        lines.append(_row("Tests:", TEST_WIDTH, palette.verdict(combined)))
    elif unit.get("skipped") and integration.get("skipped"):
        lines.append(_row("Unit Tests:", TEST_WIDTH, palette.verdict(unit)))
        lines.append(
            _row("Integration Tests:", TEST_WIDTH, palette.verdict(integration))
        )
    elif package_tests_skipped:
        # No package suite ran, so "Unit Tests: PASSED" would be green for work
        # not done. The workspace guards did run, and their status is what the
        # unit entry carries here.
        lines.append(_row("Workspace Guards:", TEST_WIDTH, palette.verdict(unit)))
        lines.append(
            _row(
                "Package Tests:",
                TEST_WIDTH,
                f"{palette.cyan}⊘ SKIPPED (no package changed){palette.off}",
            )
        )
    else:
        lines.append(_row("Unit Tests:", TEST_WIDTH, palette.verdict(unit)))
        lines.append(
            _row("Integration Tests:", TEST_WIDTH, palette.verdict(integration))
        )
    return lines


def cmd_render(args: argparse.Namespace) -> int:
    """Print the banner's per-check lines from a finished summary."""
    try:
        with open(args.summary, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        # Reported where the rows would have been, rather than raised: the
        # banner is a report on a run that has already finished, and aborting it
        # here would replace a readable failure with a stack trace.
        print(f"  (could not read {args.summary}: {exc})", file=sys.stderr)
        return 0

    if not isinstance(document, dict):
        print(f"  (summary in {args.summary} is not a JSON object)", file=sys.stderr)
        return 0

    for line in render(
        document,
        mode=args.mode,
        package_tests_skipped=args.package_tests_skipped,
        palette=Palette(sys.stdout),
    ):
        print(line)
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    recorder = commands.add_parser("record", help="append one check's outcome")
    recorder.add_argument("--records", required=True, help="the JSON Lines file")
    recorder.add_argument("--name", required=True, help="the check's key")
    recorder.add_argument("--exit-code", required=True, type=int)
    recorder.add_argument("--duration", required=True, type=optional_int)
    recorder.add_argument("--tool", default=None)
    recorder.add_argument("--skipped", default=None, type=json_bool)
    recorder.add_argument(
        "--field",
        action="append",
        default=[],
        type=key_json,
        metavar="KEY=JSON",
        help="an additional field, its value parsed as JSON",
    )
    recorder.set_defaults(run=cmd_record)

    builder = commands.add_parser("build", help="emit the summary document")
    builder.add_argument("--records", required=True)
    builder.add_argument("--output", required=True)
    builder.add_argument(
        "--str",
        action="append",
        default=[],
        type=key_value,
        metavar="KEY=VALUE",
        help="a top-level string field",
    )
    builder.add_argument(
        "--json",
        action="append",
        default=[],
        type=key_json,
        metavar="KEY=JSON",
        help="a top-level field whose value is parsed as JSON",
    )
    builder.set_defaults(run=cmd_build)

    renderer = commands.add_parser("render", help="print the console status lines")
    renderer.add_argument("--summary", required=True)
    renderer.add_argument("--mode", required=True, choices=("pr", "dev"))
    renderer.add_argument("--package-tests-skipped", action="store_true")
    renderer.set_defaults(run=cmd_render)

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
