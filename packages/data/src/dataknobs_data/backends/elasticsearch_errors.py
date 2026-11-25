# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Which Elasticsearch failures are an answer, shared by both backends.

A missing document is an answer: ``read`` returns ``None`` and ``delete``
returns ``False``. Everything else Elasticsearch refuses is a failure and
raises, as does a search that only some shards answered, which arrives with a
200. The first two arrive as errors, and a missing *index* shares the status of a
missing document, so the two are told apart here by what the body says.

Nothing in this module does I/O or imports the client, so one copy serves the
sync and async twins.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dataknobs_utils.elasticsearch_responses import (
    error_reason,
    error_type,
    incomplete_failures,
    incomplete_reason,
    says_document_missing,
    says_index_exists,
)

from ..exceptions import DatabaseError, DuplicateRecordError

#: How many failed items a bulk error's message spells out. All of them are in
#: its ``context``.
_FAILURES_IN_MESSAGE = 5


def is_missing_document(error: Any) -> bool:
    """Whether an ``ApiError`` says the document, and not its index, is missing."""
    return error.status_code == 404 and says_document_missing(error.body)


def is_existing_index(error: Any) -> bool:
    """Whether an ``ApiError`` says the index being created is already there."""
    return says_index_exists(error.body)


def require_complete(response: Any, operation: str) -> Any:
    """``response``, unless it leaves something out.

    Args:
        response: The client's response to a search, scroll, count or field
            capabilities request. An incomplete delete-by-query is refused.
        operation: What the caller was doing, for the error message.

    Raises:
        DatabaseError: A shard or index did not answer, or the request timed
            out. Its ``context["failures"]`` lists those Elasticsearch gave.
    """
    body = getattr(response, "body", response)
    reason = incomplete_reason(body)
    if reason is None:
        return response
    raise DatabaseError(
        f"Elasticsearch {operation} is incomplete: {reason}",
        context={"operation": operation, "failures": incomplete_failures(body)},
    )


def bulk_outcomes(items: Iterable[dict[str, Any]], operation: str) -> list[bool]:
    """What each item of a bulk request did, in order.

    Args:
        items: The response's items, each ``{op_type: {...}}``.
        operation: What the caller was doing, for the error message.

    Returns:
        ``True`` for an item that was applied, ``False`` for one whose document
        is missing.

    Raises:
        DuplicateRecordError: A ``create`` collided with an existing id, and
            nothing else failed.
        DatabaseError: Any other item failed. Its ``context["failures"]`` lists
            every one, collisions included. A bulk request is not atomic, so
            the items that did not fail have been written.
    """
    outcomes: list[bool] = []
    failures: list[dict[str, Any]] = []
    collided: list[str] = []
    for entry in items:
        op_type, item = next(iter(entry.items()))
        status = item.get("status", 500)
        applied = 200 <= status < 300
        outcomes.append(applied)
        if applied or (status == 404 and says_document_missing(item)):
            continue
        if op_type == "create" and status == 409:
            collided.append(str(item.get("_id")))
        failures.append(
            {
                "id": item.get("_id"),
                "status": status,
                "type": error_type(item),
                "reason": error_reason(item),
            }
        )

    if not failures:
        return outcomes
    if len(collided) == len(failures):
        raise DuplicateRecordError(collided[0])
    named = [
        f"{failure['id']!r} ({failure['status']} {failure['type']}: {failure['reason']})"
        for failure in failures[:_FAILURES_IN_MESSAGE]
    ]
    if len(failures) > len(named):
        named.append(f"and {len(failures) - len(named)} more")
    raise DatabaseError(
        f"Elasticsearch {operation}: {len(failures)} of {len(outcomes)} items failed: "
        + "; ".join(named),
        context={"operation": operation, "failures": failures},
    )
