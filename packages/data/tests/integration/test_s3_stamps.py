"""Both S3 twins keep a record's creation time, by one rule.

A write over a stored object keeps that object's ``created_at`` and advances
``updated_at``, whichever write it is. Every test drives both twins through
one awaiting helper, so a difference between them fails here.

Requires LocalStack (``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import pytest
from dataknobs_common.clock import use_clock
from dataknobs_common.testing import (
    FixedClock,
    assert_no_blocking,
    requires_blockbuster,
    requires_localstack,
)

from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.streaming import ConflictPolicy, StreamConfig

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-stamps"
CREATED = datetime(2026, 1, 1, tzinfo=UTC)
LATER = CREATED + timedelta(hours=1)
CALLERS = "2020-01-01T00:00:00+00:00"


async def _value(result: Any) -> Any:
    """``result``, awaited when it is the async twin's."""
    return await result if inspect.isawaitable(result) else result


class Twin:
    """One S3 database, and the raw objects under it."""

    def __init__(self, db: Any, config: dict[str, Any]) -> None:
        self.db = db
        self._bucket = config["bucket"]
        self._prefix = config["prefix"]
        self._raw = boto3.client(
            "s3",
            endpoint_url=config["endpoint_url"],
            region_name=config["region"],
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
        )

    def key(self, record_id: str) -> str:
        return f"{self._prefix}{record_id}.json"

    def put(self, record_id: str, body: Any) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self._raw.put_object(Bucket=self._bucket, Key=self.key(record_id), Body=data)

    def delete(self, record_id: str) -> None:
        self._raw.delete_object(Bucket=self._bucket, Key=self.key(record_id))

    def stored(self, record_id: str) -> dict[str, Any]:
        response = self._raw.get_object(Bucket=self._bucket, Key=self.key(record_id))
        return json.loads(response["Body"].read())

    def stamps(self, record_id: str) -> tuple[Any, Any]:
        metadata = self.stored(record_id)["metadata"]
        return metadata["created_at"], metadata["updated_at"]

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return await _value(getattr(self.db, method)(*args, **kwargs))

    async def create(self, record_id: str, at: datetime = CREATED) -> None:
        with use_clock(FixedClock(at)):
            await self.call("create", Record({"v": 0}, id=record_id))


@pytest.fixture
def s3_config(
    make_localstack_s3_bucket: Callable[[str], Iterator[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    for base in make_localstack_s3_bucket(_BUCKET):
        yield {**base, "prefix": f"stamps-{uuid.uuid4().hex[:10]}/"}


@pytest.fixture(params=[SyncS3Database, AsyncS3Database], ids=["sync", "async"])
async def twin(request: pytest.FixtureRequest, s3_config: dict[str, Any]) -> AsyncIterator[Twin]:
    db = request.param(s3_config)
    await _value(db.connect())
    try:
        yield Twin(db, s3_config)
    finally:
        await _value(db.clear())
        await _value(db.close())


async def _update(twin: Twin, record: Record) -> None:
    assert await twin.call("update", "k", record) is True


async def _conditional_update(twin: Twin, record: Record) -> None:
    token = await twin.call("get_version", "k")
    assert await twin.call("update", "k", record, expected_version=token) is True


async def _update_batch(twin: Twin, record: Record) -> None:
    assert await twin.call("update_batch", [("k", record)]) == [True]


async def _upsert(twin: Twin, record: Record) -> None:
    assert await twin.call("upsert", "k", record) == "k"


async def _conditional_upsert(twin: Twin, record: Record) -> None:
    token = await twin.call("get_version", "k")
    assert await twin.call("upsert", "k", record, expected_version=token) == "k"


async def _upsert_batch(twin: Twin, record: Record) -> None:
    record.storage_id = "k"
    assert await twin.call("upsert_batch", [record]) == ["k"]


async def _stream_upsert(twin: Twin, record: Record) -> None:
    record.storage_id = "k"
    config = StreamConfig(on_conflict=ConflictPolicy.UPSERT)
    if isinstance(twin.db, SyncS3Database):
        result = twin.db.stream_write(iter([record]), config)
    else:

        async def one() -> AsyncIterator[Record]:
            yield record

        result = await twin.db.stream_write(one(), config)
    assert result.successful == 1


Write = Callable[[Twin, Record], Awaitable[None]]
_WRITES: list[Write] = [
    _update,
    _conditional_update,
    _update_batch,
    _upsert,
    _conditional_upsert,
    _upsert_batch,
    _stream_upsert,
]
every_write = pytest.mark.parametrize("write", _WRITES, ids=lambda w: w.__name__.strip("_"))


@every_write
@pytest.mark.parametrize(
    "metadata",
    [{}, {"created_at": CALLERS}, {"created_at": None}],
    ids=["none", "callers", "null"],
)
async def test_a_write_over_a_stored_record_keeps_its_created_at(
    twin: Twin, write: Write, metadata: dict[str, Any]
) -> None:
    """The stored value wins over the clock and over the caller's."""
    await twin.create("k")
    with use_clock(FixedClock(LATER)):
        await write(twin, Record({"v": 1}, metadata=dict(metadata)))
    assert twin.stamps("k") == (CREATED.isoformat(), LATER.isoformat())
    assert (await twin.call("read", "k")).get_value("v") == 1


async def test_a_record_read_changed_and_updated_keeps_its_created_at(twin: Twin) -> None:
    await twin.create("k")
    record = await twin.call("read", "k")
    with use_clock(FixedClock(LATER)):
        await _update(twin, record)
    assert twin.stamps("k") == (CREATED.isoformat(), LATER.isoformat())


async def test_one_record_written_to_two_ids_keeps_each_ones_created_at(twin: Twin) -> None:
    """A stamp written into the caller's record would be stored under the second id."""
    await twin.create("a")
    await twin.create("b", at=CREATED + timedelta(days=1))
    record = Record({"v": 1}, metadata={"mine": 1})
    for record_id in ("a", "b"):
        with use_clock(FixedClock(LATER)):
            assert await twin.call("update", record_id, record)
    assert twin.stamps("a")[0] == CREATED.isoformat()
    assert twin.stamps("b")[0] == (CREATED + timedelta(days=1)).isoformat()
    assert twin.stored("a")["metadata"]["id"] == "a"
    assert twin.stored("b")["metadata"]["id"] == "b"


@every_write
async def test_a_write_leaves_the_callers_metadata_alone(twin: Twin, write: Write) -> None:
    await twin.create("k")
    record = Record({"v": 1}, metadata={"mine": 1})
    await write(twin, record)
    assert record.metadata == {"mine": 1}
    assert twin.stored("k")["metadata"]["mine"] == 1


async def test_a_record_read_under_one_id_is_updated_under_another(twin: Twin) -> None:
    """The id written under is the stored one, so a read of it answers that id."""
    await twin.create("a")
    await twin.create("b")
    record = await twin.call("read", "a")
    assert await twin.call("update", "b", record)
    assert record.id == "a"
    assert twin.stored("b")["id"] == "b"
    assert (await twin.call("read", "b")).id == "b"


@every_write
@pytest.mark.parametrize(
    "stored",
    [
        {"fields": {"v": {"value": 0}}},
        {"fields": {"v": {"value": 0}}, "metadata": None},
        {"fields": {"v": {"value": 0}}, "metadata": {"created_at": None}},
    ],
    ids=["no-metadata", "null-metadata", "null-created_at"],
)
async def test_a_stored_object_with_no_created_at_takes_the_callers_or_the_clocks(
    twin: Twin, write: Write, stored: dict[str, Any]
) -> None:
    for metadata, created in (({}, LATER.isoformat()), ({"created_at": CALLERS}, CALLERS)):
        twin.put("k", stored)
        with use_clock(FixedClock(LATER)):
            await write(twin, Record({"v": 1}, metadata=dict(metadata)))
        assert twin.stamps("k") == (created, LATER.isoformat())


@every_write
async def test_a_stored_object_that_is_not_json_is_an_error(twin: Twin, write: Write) -> None:
    """Replacing it would discard what is there; a delete is how it is cleared."""
    twin.put("k", b"{not json")
    if write is _stream_upsert:
        with pytest.raises(AssertionError):
            await write(twin, Record({"v": 1}))
    else:
        with pytest.raises(json.JSONDecodeError):
            await write(twin, Record({"v": 1}))
    assert await twin.call("delete", "k") is True
    await twin.create("k")


@pytest.mark.parametrize(
    "callers", [CALLERS, datetime(2020, 1, 1, tzinfo=UTC)], ids=["text", "datetime"]
)
@pytest.mark.parametrize("method", ["create", "upsert"])
async def test_a_new_record_keeps_the_callers_created_at(
    twin: Twin, method: str, callers: Any
) -> None:
    """So a copied or migrated record keeps its time."""
    record = Record({"v": 1}, id="k", metadata={"created_at": callers})
    with use_clock(FixedClock(LATER)):
        assert await twin.call(method, record) == "k"
    assert twin.stamps("k") == (CALLERS, LATER.isoformat())
    assert record.metadata == {"created_at": callers}


@pytest.mark.parametrize("method", ["create", "upsert"])
async def test_a_new_record_with_no_created_at_takes_one_read_of_the_clock(
    twin: Twin, method: str
) -> None:
    clock = FixedClock(CREATED)
    with use_clock(lambda: clock.advance(timedelta(seconds=1))):
        await twin.call(method, Record({"v": 1}, id="k", metadata={"created_at": None}))
    created, updated = twin.stamps("k")
    assert created == updated


async def test_the_verdicts_on_a_missing_id_and_a_stale_token_are_unchanged(twin: Twin) -> None:
    for token in (None, '"deadbeef"'):
        assert await twin.call("update", "gone", Record({}), expected_version=token) is False
    with pytest.raises(ConcurrencyError):
        await twin.call("upsert", "gone", Record({}), expected_version='"deadbeef"')
    await twin.create("k")
    stale = await twin.call("get_version", "k")
    await _update(twin, Record({"v": 1}))
    with pytest.raises(ConcurrencyError):
        await twin.call("update", "k", Record({"v": 2}), expected_version=stale)
    assert (await twin.call("read", "k")).get_value("v") == 1


async def test_a_conditional_update_losing_to_a_delete_mid_write_answers_false(
    twin: Twin, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The object goes after the read and before the guarded PUT.

    Deleted from the step between the two, so the PUT itself meets an absent
    key, which S3 answers with ``NoSuchKey`` rather than a failed condition.
    """
    await twin.create("k")
    token = await twin.call("get_version", "k")
    build = type(twin.db)._object_for_write

    def racing(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        twin.delete("k")
        return build(self, *args, **kwargs)

    monkeypatch.setattr(type(twin.db), "_object_for_write", racing)
    assert await twin.call("update", "k", Record({}), expected_version=token) is False
    assert await twin.call("exists", "k") is False


@requires_blockbuster
@every_write
async def test_no_async_write_blocks_the_loop(twin: Twin, write: Write) -> None:
    await twin.create("k")
    guard = assert_no_blocking() if isinstance(twin.db, AsyncS3Database) else nullcontext()
    with guard:
        await write(twin, Record({"v": 1}))
