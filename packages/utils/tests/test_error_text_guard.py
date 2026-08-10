"""No unbounded third-party text in this package's error messages.

This package raises none of the shared exceptions today — its own
``ElasticsearchConflictError`` carries a document id and nothing else, and the
rest is ``ValueError`` and ``NotImplementedError``. So the scan is green on
arrival, and that is the point rather than an argument against it: what it
guards is the *next* raise site, in a package whose whole job is talking to
things that produce text of their own.

The risk here is not hypothetical for the same reason the package exists. Its
modules wrap Elasticsearch, PostgreSQL, pandas and HTTP, and each of those
reports failure in words it composed — a driver's message names a column, an
HTTP client's names a URL with its credentials in the query string, and an
``ImportError`` for an optional dependency names an absolute path into
site-packages. An ``except Exception`` whose text reaches a caller hands that
along, and a caller of this package is often the API layer of another one.

``ElasticsearchConflictError`` is named alongside the shared set because it is
the one error type here a consumer catches by name, so it is the one most
likely to grow a message.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

_GUARDED = GUARDED_ERROR_NAMES | {"ElasticsearchConflictError"}


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_GUARDED)
