"""No unbounded third-party text in this package's error messages.

This package raises nothing today. It is a deprecation shim — a warning, a
version string, four re-export namespaces, and a small Flask module — so the
scan is green on arrival for the plainest possible reason: there is almost no
code to scan.

Which makes it the package most likely to be forgotten, and the argument for
wiring it is about coverage of the *set* rather than risk in this member. A
closure guard that derives the package list has to be satisfiable honestly by
every package it finds; the moment one is exempted because it seems too small
to matter, the exemption is what the next reader copies. There is no size below
which a re-export package cannot grow an error path — the modules it re-exports
have them already.

The Flask module is the part that would matter if it did. A route handler's
uncaught text is what the HTTP layer renders, and ``listdir`` takes a path from
a query string, so its failures name the filesystem it ran on.
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
