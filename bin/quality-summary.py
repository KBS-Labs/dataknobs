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

Two commands, and the split is the fix:

``record``
    Append one check's outcome, called from the site that ran it (or from the
    arm that deliberately skipped it). A check no path reaches writes no record,
    and an absent record cannot be read as a passing one, because there is no
    default left to fall through to.

``build``
    Merge the records with the run's metadata and emit the document. Every
    top-level field must be supplied — a forgotten one fails the run rather than
    vanishing from the artifact.

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

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
