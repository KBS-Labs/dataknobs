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

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

#: The shared names plus this package's own API family. Every one of them is
#: rendered by the policy table or by ``client_safe``, so all are in scope.
_RENDERED = GUARDED_ERROR_NAMES | {
    "BotCreationError",
    "APIError",
    "BotNotFoundError",
    "ConsentRequiredError",
}

#: Sites reviewed and judged bounded.
_ALLOWED = {
    # wizard_response.py: the message is already assembled from the strategy
    # name, the stage name and `type(e).__name__`. What the scan sees is
    # `hint`, which comes from `_maybe_strict_signature_hint(e, forwarded)` —
    # a helper it cannot look inside, so it assumes the worst. The helper
    # returns authored text plus collaborator *names* and a doc path, none of
    # which is the exception's own message.
    "dataknobs_bots/reasoning/wizard_response.py:1694",
}


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_RENDERED, ignore=_ALLOWED)
