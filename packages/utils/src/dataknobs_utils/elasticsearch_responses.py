# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What the body of an Elasticsearch response says went wrong.

A missing document and a missing index are both a 404, and only one of them is
an ordinary answer. These read the parsed body, so they serve a caller of the
official client and a caller speaking HTTP alike; nothing here imports either.
"""

from __future__ import annotations

from typing import Any


def error_type(body: Any) -> str | None:
    """The type of error a response body or bulk item names, if it names one."""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("type"))
    return None if error is None else str(error)


def error_reason(body: Any) -> str | None:
    """The reason a response body or bulk item gives for its error, if any."""
    error = body.get("error") if isinstance(body, dict) else None
    reason = error.get("reason") if isinstance(error, dict) else None
    return None if reason is None else str(reason)


def says_document_missing(body: Any) -> bool:
    """Whether the body of a 404 is about a document rather than its index.

    A GET or DELETE of a missing document has no ``error``, and says
    ``found: false`` or ``result: not_found``. An update has one, of its own type.
    """
    if not isinstance(body, dict):
        return False
    return (
        body.get("found") is False
        or body.get("result") == "not_found"
        or error_type(body) == "document_missing_exception"
    )


def says_index_exists(body: Any) -> bool:
    """Whether a refused index creation says the index is already there."""
    return error_type(body) == "resource_already_exists_exception"
