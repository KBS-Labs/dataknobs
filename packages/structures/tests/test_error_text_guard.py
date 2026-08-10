"""No unbounded third-party text in this package's error messages.

This package raises nothing at all today — not one ``raise`` statement in its
source — so the scan is green on arrival and will stay green for as long as
that holds. Wiring it anyway is a deliberate choice about *when* the guard
arrives rather than *whether*.

The alternative is to wire it the day the package first raises something, and
that day is exactly the wrong one: the author is adding an error path, deciding
what its message says, and has no reason to think a workspace convention
applies to a package that never needed it. A guard already present is a
constraint on the new code; a guard added afterwards is a review of it.

The cost of being early is one file that asserts about nothing. The cost of
being late is that nobody is told — which is why the closure guard under
``tests/`` derives the package list rather than trusting that someone will
remember, and why this file exists to satisfy it honestly rather than by
exemption.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=GUARDED_ERROR_NAMES)
