"""What an Elasticsearch response says it leaves out."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_utils.elasticsearch_responses import incomplete_failures, incomplete_reason

SHARD = {"shard": 1, "reason": {"type": "no_shard_available_action_exception", "reason": None}}
INDEX = {"indices": ["b"], "failure": {"error": {"type": "no_shard", "reason": "down"}}}


@pytest.mark.parametrize(
    "body",
    [
        None,
        "text",
        {},
        {"_shards": {"total": 2, "successful": 2, "failed": 0}, "timed_out": False},
        {"_shards": None},
        {"deleted": 3, "failures": [], "timed_out": False},
    ],
)
def test_a_complete_response(body: Any) -> None:
    assert incomplete_reason(body) is None
    assert incomplete_failures(body) == []


def test_a_failed_shard() -> None:
    body = {"_shards": {"total": 2, "failed": 1, "failures": [SHARD, SHARD]}}
    reason = incomplete_reason(body)
    assert reason == "1 of 2 shards failed; no_shard_available_action_exception: None"
    assert incomplete_failures(body) == [SHARD, SHARD]


def test_a_failed_shard_without_a_failure_given() -> None:
    body = {"_shards": {"total": 2, "failed": 1}}
    assert incomplete_reason(body) == "1 of 2 shards failed"


def test_a_failed_index() -> None:
    body = {"failed_indices": 1, "failures": [INDEX], "fields": {}}
    assert incomplete_reason(body) == "1 indices failed; no_shard: down"
    assert incomplete_failures(body) == [INDEX]


def test_a_timeout() -> None:
    assert incomplete_reason({"timed_out": True, "_shards": {"failed": 0}}) == "timed out"


def test_a_failure_of_another_shape_is_still_named() -> None:
    assert incomplete_reason({"failed_indices": 1, "failures": [{"reason": "why"}]}) == (
        "1 indices failed; why"
    )
