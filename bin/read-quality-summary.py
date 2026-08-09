#!/usr/bin/env python3
"""Project ``quality-summary.json`` into tab-delimited lines a shell can read.

``bin/validate-quality-artifacts.sh`` is what CI runs instead of re-running the
gate, so what it can read decides what CI can see. It used to read the summary
with line-offset greps — ``grep -A2 '"unit_tests"'`` for a status, ``grep -A3``
for a skipped flag — which located fields by POSITION. JSON objects have no
order, so a field added or moved above the one a window wanted pushed it out,
the grep returned nothing, and the validator rejected an artifact it had merely
failed to read, reporting an empty status as the reason.

Output, one record per line, fields separated by tabs::

    OVERALL<TAB><overall_status>
    CHECK<TAB><name><TAB><status><TAB><skipped><TAB><label>
    ERROR<TAB><message>          (unreadable input; nothing else is emitted)

Tabs rather than spaces because the label is a human-readable phrase containing
spaces, and a delimiter that can appear inside a field is not a delimiter. Any
tab or newline within a value is flattened to a space for the same reason.

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


def clean(value: Any) -> str:
    """Flatten a value to a single tab-free, newline-free field."""
    return str(value).replace("\t", " ").replace("\n", " ")


def project(doc: Any) -> list[str]:
    """Render the summary as the tab-delimited records documented above."""
    if not isinstance(doc, dict):
        return [f"{ERROR}\tsummary is not a JSON object"]

    records = [f"OVERALL\t{clean(doc.get('overall_status') or '')}"]

    checks = doc.get("checks")
    if isinstance(checks, dict):
        for name in sorted(checks):
            entry = checks[name] if isinstance(checks[name], dict) else {}
            # The label is derived from the name, so a check added to the gate
            # later is reported readably without being registered anywhere here.
            label = clean(name).replace("_", " ").capitalize()
            records.append(
                "\t".join(
                    (
                        "CHECK",
                        clean(name),
                        clean(entry.get("status") or ""),
                        "true" if entry.get("skipped") is True else "false",
                        label,
                    )
                )
            )
    return records


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0] if argv else 'read-quality-summary.py'} <summary.json>",
              file=sys.stderr)
        return 2

    try:
        with open(argv[1], encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        # Reported as data, not raised: "could not parse the attestation" and
        # "the attestation says the run passed" must never reach the same
        # verdict, and the caller can only tell them apart if the first one says
        # so in the output it already reads.
        print(f"{ERROR}\t{clean(exc)}")
        return 0

    for record in project(doc):
        print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
