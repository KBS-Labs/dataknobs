"""No unbounded third-party text in the error types rendered over HTTP.

``dataknobs_bots.api`` maps ``dataknobs_common.exceptions`` types to statuses
and decides per type whether the message reaches the caller. A message built
from an exception caught by ``except Exception`` is only as bounded as whatever
ran in the ``try`` — and in this package that includes consumer code: tool and
middleware constructors, and the module imports that resolve them. A tool whose
constructor opens a database raises with its connection URL in the message.

Bots are built lazily on the request path, so such a site is reachable from an
ordinary HTTP request. The guard is a source scan because the runtime path
needs a real failing dependency to trigger.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import assert_no_broad_except_in_error_text

_SRC = Path(__file__).resolve().parents[1] / "src"

#: Types this package raises that the API layer renders. ``ValidationError``
#: and ``ConfigurationError`` are both disclosed-or-masked by policy rather
#: than never shown, so both are in scope.
_RENDERED = frozenset(
    {
        "ConfigurationError",
        "ValidationError",
        "BotCreationError",
    }
)


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_RENDERED)
