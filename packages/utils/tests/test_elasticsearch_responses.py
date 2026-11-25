"""What an Elasticsearch response says it leaves out, or cannot do."""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_utils.elasticsearch_responses import (
    incomplete_failures,
    incomplete_reason,
    says_rrf_unsupported,
)

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


RRF = "Reciprocal Rank Fusion (RRF)"


@pytest.mark.parametrize(
    "error",
    [
        {"type": "parsing_exception", "reason": "Unknown key for a START_OBJECT in [retriever]."},
        {
            "type": "x_content_parse_exception",
            "reason": "[1:237] [rrf] unknown field [rank_window_size] did you mean [window_size]?",
        },
        {"type": "security_exception", "reason": "no", "license.expired.feature": RRF},
    ],
)
def test_rrf_is_unsupported(error: dict[str, Any]) -> None:
    assert says_rrf_unsupported({"error": error})


@pytest.mark.parametrize(
    "body",
    [
        None,
        "text",
        {},
        {"error": RRF},
        {"error": {"type": "security_exception", "reason": f"unauthorized for [{RRF}]"}},
        {"error": {"type": "security_exception", "license.expired.feature": "field security"}},
        {"error": {"type": "parsing_exception", "reason": "unknown query [retriever]."}},
        {"error": {"type": "parsing_exception", "reason": "Unknown key for a VALUE in [size]."}},
        {"error": {"type": "parsing_exception", "reason": None}},
        {"error": {"type": "x_content_parse_exception", "reason": "[rrf] unknown field [k]"}},
        {"error": {"type": "status_exception", "reason": "[rrf] unknown field [rank_window_size]"}},
    ],
)
def test_another_refusal_is_not_about_rrf(body: Any) -> None:
    assert not says_rrf_unsupported(body)
