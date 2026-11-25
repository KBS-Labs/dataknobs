"""What a bulk response means, item by item.

The live-node half is ``integration/test_elasticsearch_failures.py``; these pin
the cases a node cannot be made to produce on demand.
"""

from __future__ import annotations

import pytest

from dataknobs_data.backends.elasticsearch_errors import bulk_outcomes
from dataknobs_data.exceptions import DatabaseError, DuplicateRecordError


def _refused(op: str, record_id: str, status: int, error_type: str) -> dict[str, dict]:
    error = {"type": error_type, "reason": f"why {record_id}"}
    return {op: {"_id": record_id, "status": status, "error": error}}


def test_every_item_applied() -> None:
    items = [{"index": {"_id": "a", "status": 201}}, {"update": {"_id": "b", "status": 200}}]
    assert bulk_outcomes(items, "x") == [True, True]


def test_no_items() -> None:
    assert bulk_outcomes([], "x") == []


def test_a_missing_document_is_false() -> None:
    items = [
        {"delete": {"_id": "a", "status": 404, "result": "not_found"}},
        _refused("update", "b", 404, "document_missing_exception"),
    ]
    assert bulk_outcomes(items, "x") == [False, False]


def test_a_missing_index_is_a_failure() -> None:
    with pytest.raises(DatabaseError, match="index_not_found_exception"):
        bulk_outcomes([_refused("delete", "a", 404, "index_not_found_exception")], "x")


def test_a_conflict_is_a_duplicate_only_for_a_create() -> None:
    with pytest.raises(DuplicateRecordError):
        bulk_outcomes([_refused("create", "a", 409, "version_conflict_engine_exception")], "x")
    with pytest.raises(DatabaseError, match="version_conflict") as caught:
        bulk_outcomes([_refused("update", "a", 409, "version_conflict_engine_exception")], "x")
    assert not isinstance(caught.value, DuplicateRecordError)


def test_an_item_with_no_status_is_a_failure() -> None:
    with pytest.raises(DatabaseError):
        bulk_outcomes([{"index": {"_id": "a"}}], "x")


def test_a_long_list_of_failures_is_summarised_and_kept_whole() -> None:
    items = [_refused("index", f"r{n}", 429, "es_rejected_execution_exception") for n in range(8)]
    with pytest.raises(DatabaseError, match=r"8 of 8 items failed: 'r0'.*and 3 more") as caught:
        bulk_outcomes(items, "upsert_batch")
    assert len(caught.value.context["failures"]) == 8
    assert caught.value.context["failures"][7]["reason"] == "why r7"
