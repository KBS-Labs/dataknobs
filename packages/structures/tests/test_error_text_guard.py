"""No unbounded third-party text in this package's error messages.

This file was written while the package raised nothing at all — not one
``raise`` statement in its source — so the scan was green on arrival and
asserted about nothing. Wiring it anyway was a deliberate choice about *when*
the guard arrives rather than *whether*:

    The alternative is to wire it the day the package first raises something,
    and that day is exactly the wrong one: the author is adding an error path,
    deciding what its message says, and has no reason to think a workspace
    convention applies to a package that never needed it. A guard already
    present is a constraint on the new code; a guard added afterwards is a
    review of it.

That day has since arrived. ``Tree``'s writers refuse a child that would make a
node its own ancestor, and ``ValidationError`` is the package's first raise.
The two messages are constant text and the two nodes' data travels in
``context``, which is what this scan exists to keep true — an error that renders
a caught exception into its own message is how unbounded third-party words reach
a log line.

The cost of being early was one file that asserted about nothing, for one
release. The cost of being late is that nobody is told — which is why the
closure guard under ``tests/`` derives the package list rather than trusting
that someone will remember, and why this file was here to constrain the raise
rather than to review it.
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
