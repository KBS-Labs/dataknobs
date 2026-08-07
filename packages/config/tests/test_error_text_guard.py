"""No unbounded third-party text in this package's config error messages.

``ConfigError`` is a ``dataknobs_common.exceptions.ConfigurationError``, which
the ``dataknobs-bots`` API layer renders at the HTTP boundary. Resolving a
dotted class path here calls ``importlib.import_module``, which *executes* the
target module — so an ``except Exception`` around it catches text produced by
code the deployment supplied.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import assert_no_broad_except_in_error_text

_SRC = Path(__file__).resolve().parents[1] / "src"

_RENDERED = frozenset({"ConfigError", "ConfigurationError", "ValidationError"})


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_RENDERED)
