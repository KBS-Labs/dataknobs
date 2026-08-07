"""No unbounded parser or path text in this package's error messages.

``KnowledgeBaseConfig.load`` reads a config file whose parse failure quotes the
line it choked on — an unterminated quote on an ``api_key`` puts the key in the
text — and reports it against a resolved path. ``dataknobs_config`` withholds
both for exactly those reasons; this is the sibling loader, held to the same
rule so the two cannot disagree about the same failure.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import (
    GUARDED_ERROR_NAMES,
    assert_no_broad_except_in_error_text,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

_GUARDED = GUARDED_ERROR_NAMES | {"IngestionConfigError"}


def test_no_broad_except_feeds_a_rendered_error_message():
    assert_no_broad_except_in_error_text(_SRC, error_names=_GUARDED)
