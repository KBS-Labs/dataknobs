#!/usr/bin/env python3
r"""Project ``quality-summary.json`` into delimited lines a shell can read.

``bin/validate-quality-artifacts.sh`` is what CI runs instead of re-running the
gate, so what it can read decides what CI can see. It used to read the summary
with line-offset greps — ``grep -A2 '"unit_tests"'`` for a status, ``grep -A3``
for a skipped flag — which located fields by POSITION. JSON objects have no
order, so a field added or moved above the one a window wanted pushed it out,
the grep returned nothing, and the validator rejected an artifact it had merely
failed to read, reporting an empty status as the reason.

Output, one record per line, fields separated by ASCII Unit Separator (``\x1f``)::

    OVERALL<US><overall_status>
    META<US><key><US><value>
    CHECK<US><name><US><status><US><skipped><US><exit_code><US><tool><US><label>
    ERROR<US><message>          (unreadable input; nothing else is emitted)

Not spaces, because the label is a phrase containing spaces and a delimiter that
can appear inside a field is not a delimiter. Not tabs either, which is less
obvious and was found the hard way: a tab is IFS *whitespace*, and shell ``read``
treats a run of IFS whitespace as one delimiter and discards empty fields. A
check with no ``tool`` recorded therefore shifted ``label`` into ``tool``'s
variable and left ``label`` empty — every row displayed blank, from a record that
was correct. ``\x1f`` exists for exactly this and is not whitespace, so empty
fields survive. Any separator or newline inside a value is flattened to a space.

``label`` is last because it is the only field holding a phrase written for a
person; a field added later goes before it. Every consumer must name every
field, because a shell ``read`` given too few variables does not complain — it
assigns the leftover fields to its last one, so adding a field here silently
corrupts the final variable of any consumer not updated alongside it. The guard
for that is in ``tests/test_quality_artifact_validation.py``.

``META`` carries the summary's top-level scalars, so that a consumer wanting the
timestamp or the environment does not have to open the file a second way. That
is not a stylistic preference: the second way was ``jq``, which is not pinned
here, so it came with a branch for its own absence — and that branch was a
second reader, which drifted.

Exits 0 even for an unreadable summary: the ERROR record *is* the result, and
the caller distinguishes it by reading the line rather than by branching on an
exit code it would have to map back to a cause anyway. A non-zero exit is
reserved for being unable to produce output at all.

This lives in its own file rather than inside the shell script because a program
embedded in a shell string is checked by nothing — not ruff, not mypy, not even
a syntax check until the moment it runs. That is the same reason the repository
grew a shell linter.
"""

from __future__ import annotations

import json
import sys
from typing import Any

#: Emitted when the summary cannot be read at all.
ERROR = "ERROR"

#: ASCII Unit Separator. See the module docstring for why not a tab.
SEP = "\x1f"


def clean(value: Any) -> str:
    """Flatten a value to a single separator-free, newline-free field."""
    return str(value).replace(SEP, " ").replace("\t", " ").replace("\n", " ")


def project(doc: Any) -> list[str]:
    """Render the summary as the delimited records documented above."""
    if not isinstance(doc, dict):
        return [f"{ERROR}{SEP}summary is not a JSON object"]

    records = [f"OVERALL{SEP}{clean(doc.get('overall_status') or '')}"]

    # Every top-level scalar, rather than a chosen few: a field added to the
    # summary becomes readable without this file being edited, which is the
    # whole reason the checks below are enumerated rather than named.
    # Objects and lists are skipped — the hash maps are not for reading this
    # way, and `checks` has its own record type.
    for key, value in doc.items():
        if key in {"overall_status", "checks"} or value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        records.append(SEP.join(("META", clean(key), clean(value))))

    checks = doc.get("checks")
    if isinstance(checks, dict):
        for name in sorted(checks):
            entry = checks[name] if isinstance(checks[name], dict) else {}
            exit_code = entry.get("exit_code")
            records.append(
                SEP.join(
                    (
                        "CHECK",
                        clean(name),
                        clean(entry.get("status") or ""),
                        "true" if entry.get("skipped") is True else "false",
                        "" if exit_code is None else clean(exit_code),
                        clean(entry.get("tool") or ""),
                        # Derived from the name, so a check added to the gate
                        # later is reported readably without being registered
                        # anywhere here.
                        clean(name).replace("_", " ").capitalize(),
                    )
                )
            )
    return records


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: {argv[0] if argv else 'read-quality-summary.py'} <summary.json>",
            file=sys.stderr,
        )
        return 2

    try:
        with open(argv[1], encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        # Reported as data, not raised: "could not parse the attestation" and
        # "the attestation says the run passed" must never reach the same
        # verdict, and the caller can only tell them apart if the first one says
        # so in the output it already reads.
        print(f"{ERROR}{SEP}{clean(exc)}")
        return 0

    for record in project(doc):
        print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
