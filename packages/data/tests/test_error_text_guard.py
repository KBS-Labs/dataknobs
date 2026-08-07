"""No unbounded driver text in this package's error messages.

``RecordValidationError`` is a ``dataknobs_common.exceptions.ValidationError``,
which the ``dataknobs-bots`` API layer returns to the caller as a 422 *with its
message shown* — the one row in the default policy that discloses both halves.
So this package's error text is read by clients, not just operators, and a
driver's own words are the wrong thing to put in it: a constraint violation is
reported by naming the constraint, which names a column.

``test_constraint_disclosure.py`` covers the constraint path behaviourally.
This is the source-level guard over everything else.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

_GUARDED = GUARDED_ERROR_NAMES | {
    "RecordValidationError",
    "DatabaseError",
    "QueryError",
}


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_GUARDED)
