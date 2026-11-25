"""The stored form both S3 twins write, without S3."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from dataknobs_common.clock import use_clock
from dataknobs_common.testing import FixedClock

from dataknobs_data import Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.backends.s3_mixins import S3ObjectMixin

NOW = datetime(2026, 1, 1, tzinfo=UTC)
STORED = "2024-01-01T00:00:00+00:00"
CALLERS = "2020-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    ("stored", "callers", "expected"),
    [
        (STORED, CALLERS, STORED),
        (STORED, None, STORED),
        (None, CALLERS, CALLERS),
        (None, datetime(2020, 1, 1, tzinfo=UTC), CALLERS),
        (None, date(2020, 1, 1), "2020-01-01"),
        (None, 1577836800.5, 1577836800.5),
        (None, None, NOW.isoformat()),
    ],
)
def test_created_at_is_the_stored_then_the_callers_then_the_clocks(
    stored: Any, callers: Any, expected: Any
) -> None:
    record = Record({"v": 1}, metadata={"created_at": callers, "mine": 1})
    clock = FixedClock(NOW - timedelta(seconds=1))
    with use_clock(lambda: clock.advance(timedelta(seconds=1))):
        written = S3ObjectMixin()._object_for_write(record, "k", {"created_at": stored})
    assert written["metadata"] == {
        "mine": 1,
        "id": "k",
        "created_at": expected,
        "updated_at": NOW.isoformat(),
    }
    assert record.metadata == {"created_at": callers, "mine": 1}


def test_the_id_written_under_wins_over_every_id_the_record_carries() -> None:
    record = Record({"id": "payload"}, metadata={"id": "meta"}, id="other")
    written = S3ObjectMixin()._object_for_write(record, "k")
    assert written["id"] == written["metadata"]["id"] == "k"
    assert written["fields"]["id"]["value"] == "payload"
    assert (record.id, record.metadata) == ("other", {"id": "meta"})


@pytest.mark.parametrize("body", [{"v": 1}, {"metadata": None}, {"metadata": []}])
def test_a_stored_object_holding_no_metadata_reads_as_none(body: Any) -> None:
    assert S3ObjectMixin._stored_metadata(json.dumps(body).encode()) == {}


@pytest.mark.parametrize(
    "body", [b"{not json", b'{"a": "\xff\xfe"}', b"[1]", b"null", b'"text"', b""]
)
def test_a_stored_object_that_is_not_a_record_is_not_read_as_empty(body: bytes) -> None:
    with pytest.raises(ValueError):
        S3ObjectMixin._stored_metadata(body)


def test_both_twins_build_what_they_store_here() -> None:
    for twin in (SyncS3Database, AsyncS3Database):
        assert twin._object_for_write is S3ObjectMixin._object_for_write
