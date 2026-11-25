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


def says_rrf_unsupported(body: Any) -> bool:
    """Whether a refused search says the node cannot run an ``rrf`` retriever.

    A node too old for retrievers, or for ``rank_window_size``, cannot parse the
    request, and one whose licence leaves RRF out refuses it.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    kind, reason = error_type(body), error_reason(body) or ""
    if kind == "security_exception":
        # Also the type of a refused permission, which names no feature.
        return "(RRF)" in str(error.get("license.expired.feature"))
    if kind == "parsing_exception":
        return reason.startswith("Unknown key for a ") and reason.endswith("in [retriever].")
    return (
        kind == "x_content_parse_exception" and "[rrf] unknown field [rank_window_size]" in reason
    )


def incomplete_reason(body: Any) -> str | None:
    """Why a successful response leaves something out; ``None`` if it does not.

    Elasticsearch answers a search or count with a 200 when only some shards
    did, and a field capabilities request when only some indices did.
    """
    if not isinstance(body, dict):
        return None
    shards = body.get("_shards")
    shards = shards if isinstance(shards, dict) else {}
    parts = []
    if failed := shards.get("failed"):
        parts.append(f"{failed} of {shards.get('total')} shards failed")
    if indices := body.get("failed_indices"):
        parts.append(f"{indices} indices failed")
    if body.get("timed_out"):
        parts.append("timed out")
    if not parts:
        return None
    named = {_failure_text(failure) for failure in incomplete_failures(body)}
    return "; ".join(parts + sorted(named))


def incomplete_failures(body: Any) -> list[dict[str, Any]]:
    """The failures a response gives for what it leaves out.

    Elasticsearch may give fewer than it counts.
    """
    if not isinstance(body, dict):
        return []
    shards = body.get("_shards")
    of_shards = shards.get("failures") if isinstance(shards, dict) else None
    return [*(of_shards or []), *(body.get("failures") or [])]


def _failure_text(failure: dict[str, Any]) -> str:
    # A shard's failure carries its reason; an index's wraps it in an error.
    nested = failure.get("failure")
    reason = failure.get("reason")
    if isinstance(nested, dict):
        return f"{error_type(nested)}: {error_reason(nested)}"
    if isinstance(reason, dict):
        return f"{reason.get('type')}: {reason.get('reason')}"
    return str(reason)
