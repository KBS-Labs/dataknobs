"""Shipped code reads the current UTC time in one place: ``dataknobs_common.clock``.

Every aware timestamp the library writes goes through
:func:`dataknobs_common.clock.utc_now` or :func:`~dataknobs_common.clock.now_iso`,
so :func:`~dataknobs_common.clock.use_clock` fixes all of them in a test. A
``datetime.now(UTC)`` written anywhere else is a stamp that seam cannot reach:
it reads wall-clock time under a fixed clock, and a test asserting an ordering
across it and a fixed stamp is quietly wrong. Before the module existed there
were 86 such lines across four packages.

**What counts.** A ``now(`` call given an argument, in any tracked file under
``packages/*/src``: ``datetime.now(UTC)``, ``datetime.now(tz=...)``,
``datetime.now(timezone.utc)`` and the same through an alias. The argument is
what makes the read an aware one, so the pattern keys on it rather than on a
spelling of the receiver.

**What does not, by design.** A naive ``datetime.now()`` is a different
defect with a different fix. A stamp written in local time needs a timezone
policy for the values already stored, and an elapsed time measured as the
difference of two wall-clock reads needs :func:`time.monotonic` rather than
any wall clock. Routing either through ``utc_now()`` without that decision
would change stored values or stay wrong. When those reads move, this guard
widens to them.

It reads ``git ls-files``, so a new file is seen once it is added.
"""

from __future__ import annotations

import re
import subprocess

from tests._workspace import ROOT

#: The one file allowed to read the system clock for a timestamp.
CLOCK_MODULE = "packages/common/src/dataknobs_common/clock.py"

#: A ``now(`` call with an argument: an aware read of the current time.
AWARE_NOW = re.compile(r"\.now\(\s*[^)\s]")


def _source_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "--", ":(glob)packages/*/src/**/*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert listed, "git ls-files found no package sources; the guard would pass vacuously"
    return listed


def _aware_reads(text: str) -> list[int]:
    return [number for number, line in enumerate(text.splitlines(), 1) if AWARE_NOW.search(line)]


def test_the_clock_module_is_tracked_and_reads_the_system_clock() -> None:
    files = _source_files()
    assert CLOCK_MODULE in files, f"{CLOCK_MODULE} is not tracked (run after `git add`)"
    assert _aware_reads((ROOT / CLOCK_MODULE).read_text(encoding="utf-8"))


def test_no_other_shipped_file_reads_the_current_utc_time() -> None:
    found = [
        f"{path}:{number}"
        for path in _source_files()
        if path != CLOCK_MODULE
        for number in _aware_reads((ROOT / path).read_text(encoding="utf-8"))
    ]
    assert not found, (
        "read the current time through dataknobs_common.clock (utc_now / now_iso), "
        "so use_clock reaches the stamp:\n  " + "\n  ".join(found)
    )


def test_the_pattern_catches_each_spelling() -> None:
    for line in (
        "now = datetime.now(UTC)",
        "x = datetime.now(tz=timezone.utc)",
        "stamp = _datetime.now(UTC).isoformat()",
        "t = dt.datetime.now( datetime.UTC )",
    ):
        assert _aware_reads(line), line
    for line in ("t = datetime.now()", "t = utc_now()", "t = time.monotonic()"):
        assert not _aware_reads(line), line
