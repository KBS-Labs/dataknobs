"""No unbounded third-party text in this package's error messages.

The exceptions in ``dataknobs_fsm.functions.base`` now reach the shared
hierarchy, so a boundary that maps types onto HTTP statuses resolves them —
a ``ResourceError`` to 503, a ``TransformError`` to 500 by way of
``OperationError``. That is what makes their message text a boundary concern
rather than a private one.

This package wraps ``except Exception`` around more third-party code than any
other: every resource provider hands its config to a driver, a session
factory, or a pool, and every transform in the function library runs a
user-supplied callable over a record. Those failures are reported by quoting
what failed — a DSN, an endpoint, a constraint, or the record's own field
value — none of which this package wrote.

Both rows are masked today, so nothing is disclosed. The guard is here because
"masked" is a policy row rather than a property of the message, and a
deployment can pass ``error_policy=`` to change it.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    assert_no_broad_except_in_error_text,
    error_names_in,
)

_SRC = Path(__file__).resolve().parents[1] / "src"


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=error_names_in("dataknobs_fsm"))
