"""How the S3 backends list and fetch objects (real transport).

S3 has no server-side query, so ``search`` and ``stream_read`` list a prefix
and fetch each object. Three properties of that, on both twins:

- an object that cannot be read is an error, never a shorter answer, while one
  deleted between the listing and the fetch is absent;
- fetches run concurrently, up to ``max_pool_connections``;
- a listing covers this database's prefix and no other, and the id it reports
  is the one the record was stored under.

The failures are produced by the service: botocore's event hooks delete an
object, or point a request at a bucket that does not exist, at the moment a
fetch is issued. Requires LocalStack (``bin/dk up``).
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from dataknobs_common.testing import assert_no_blocking, requires_localstack

from dataknobs_data import Query, Record
from dataknobs_data.backends.config import AsyncS3DatabaseConfig, SyncS3DatabaseConfig
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-s3-fetching"
_POOL = 3
_IDS = [f"r{n:02d}" for n in range(12)]


@pytest.fixture
def s3_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    for base_cfg in make_localstack_s3_bucket(_BUCKET):
        yield {
            **base_cfg,
            "prefix": f"fetch-{uuid.uuid4().hex[:10]}",
            "max_pool_connections": _POOL,
        }


@pytest.fixture
def raw(s3_config: dict[str, Any]) -> Any:
    """A client of the test's own, for changing objects behind the backend."""
    return boto3.client(
        "s3",
        endpoint_url=s3_config["endpoint_url"],
        region_name=s3_config.get("region_name") or s3_config.get("region") or "us-east-1",
        aws_access_key_id=s3_config.get("aws_access_key_id", "test"),
        aws_secret_access_key=s3_config.get("aws_secret_access_key", "test"),
    )


@pytest.fixture
def sync_db(s3_config: dict[str, Any]) -> Iterator[SyncS3Database]:
    db = SyncS3Database(s3_config)
    db.connect()
    for n, record_id in enumerate(_IDS):
        db.create(Record({"n": n}, id=record_id))
    try:
        yield db
    finally:
        db.clear()
        db.close()


@pytest.fixture
async def async_db(s3_config: dict[str, Any]) -> AsyncIterator[AsyncS3Database]:
    db = AsyncS3Database(s3_config)
    await db.connect()
    for n, record_id in enumerate(_IDS):
        await db.create(Record({"n": n}, id=record_id))
    try:
        yield db
    finally:
        await db.clear()
        await db.close()


@contextmanager
def _hook(emitter: Any, event: str, handler: Callable[..., Any]) -> Iterator[None]:
    """Run ``handler`` on a botocore event for the duration of the block.

    The async twin's clients copy their session's emitter, and the session is
    cached across tests, so the handler has to come off again.
    """
    emitter.register(event, handler)
    try:
        yield
    finally:
        emitter.unregister(event, handler)


def _sync_search(db: SyncS3Database) -> list[Record]:
    return db.search(Query())


def _sync_stream(db: SyncS3Database) -> list[Record]:
    return list(db.stream_read())


async def _async_search(db: AsyncS3Database) -> list[Record]:
    return await db.search(Query())


async def _async_stream(db: AsyncS3Database) -> list[Record]:
    return [record async for record in db.stream_read()]


_SYNC_READS = pytest.mark.parametrize(
    "read", [_sync_search, _sync_stream], ids=["search", "stream"]
)
_ASYNC_READS = pytest.mark.parametrize(
    "read", [_async_search, _async_stream], ids=["search", "stream"]
)


def _elsewhere(params: dict[str, Any], **_: Any) -> None:
    params["Bucket"] = "dataknobs-no-such-bucket"


# --- an object that cannot be read -------------------------------------------


@_SYNC_READS
def test_sync_corrupt_object_raises(sync_db: SyncS3Database, raw: Any, read: Any) -> None:
    raw.put_object(Bucket=_BUCKET, Key=sync_db._get_object_key("r05"), Body=b"{nope")
    with pytest.raises(json.JSONDecodeError):
        read(sync_db)


@_ASYNC_READS
async def test_async_corrupt_object_raises(async_db: AsyncS3Database, raw: Any, read: Any) -> None:
    raw.put_object(Bucket=_BUCKET, Key=async_db._get_key("r05"), Body=b"{nope")
    with pytest.raises(json.JSONDecodeError):
        await read(async_db)


@_SYNC_READS
def test_sync_failed_fetch_raises(sync_db: SyncS3Database, read: Any) -> None:
    with (
        _hook(sync_db.s3_client.meta.events, "before-parameter-build.s3.GetObject", _elsewhere),
        pytest.raises(ClientError, match="NoSuchBucket"),
    ):
        read(sync_db)


@_ASYNC_READS
async def test_async_failed_fetch_raises(async_db: AsyncS3Database, read: Any) -> None:
    with (
        _hook(async_db._session.events, "before-parameter-build.s3.GetObject", _elsewhere),
        pytest.raises(ClientError, match="NoSuchBucket"),
    ):
        await read(async_db)


@_SYNC_READS
def test_sync_object_deleted_after_listing_is_absent(
    sync_db: SyncS3Database, raw: Any, read: Any
) -> None:
    gone = sync_db._get_object_key("r05")

    def _delete(params: dict[str, Any], **_: Any) -> None:
        if params["Key"] == gone:
            raw.delete_object(Bucket=_BUCKET, Key=gone)

    with _hook(sync_db.s3_client.meta.events, "before-parameter-build.s3.GetObject", _delete):
        found = read(sync_db)
    assert [r.id for r in found] == [i for i in _IDS if i != "r05"]


@_ASYNC_READS
async def test_async_object_deleted_after_listing_is_absent(
    async_db: AsyncS3Database, raw: Any, read: Any
) -> None:
    gone = async_db._get_key("r05")

    def _delete(params: dict[str, Any], **_: Any) -> None:
        if params["Key"] == gone:
            raw.delete_object(Bucket=_BUCKET, Key=gone)

    with _hook(async_db._session.events, "before-parameter-build.s3.GetObject", _delete):
        found = await read(async_db)
    assert [r.id for r in found] == [i for i in _IDS if i != "r05"]


async def test_async_read_reports_what_it_could_not_read(
    async_db: AsyncS3Database, raw: Any
) -> None:
    raw.put_object(Bucket=_BUCKET, Key=async_db._get_key("r05"), Body=b"{nope")
    with pytest.raises(json.JSONDecodeError):
        await async_db.read("r05")
    with (
        _hook(async_db._session.events, "before-parameter-build.s3.GetObject", _elsewhere),
        pytest.raises(ClientError, match="NoSuchBucket"),
    ):
        await async_db.read("r01")
    assert await async_db.read("absent") is None


async def test_async_delete_reports_a_failed_delete(async_db: AsyncS3Database) -> None:
    with (
        _hook(async_db._session.events, "before-parameter-build.s3.DeleteObject", _elsewhere),
        pytest.raises(ClientError, match="NoSuchBucket"),
    ):
        await async_db.delete("r01")
    assert await async_db.exists("r01")


async def test_async_exists_reports_a_failed_check(async_db: AsyncS3Database) -> None:
    """A malformed version id is refused, which is neither a hit nor a miss."""

    def _refused(params: dict[str, Any], **_: Any) -> None:
        params["VersionId"] = "not-a-version"

    with (
        _hook(async_db._session.events, "before-parameter-build.s3.HeadObject", _refused),
        pytest.raises(ClientError),
    ):
        await async_db.exists("r01")
    assert await async_db.exists("r01")
    assert not await async_db.exists("absent")


@_SYNC_READS
def test_sync_empty_record_is_returned(sync_db: SyncS3Database, read: Any) -> None:
    """A record with no fields is falsy, and still a record."""
    sync_db.create(Record({}, id="r99"))
    assert "r99" in [r.id for r in read(sync_db)]


# --- concurrent fetches -------------------------------------------------------


@_SYNC_READS
def test_sync_fetches_overlap_within_the_pool(sync_db: SyncS3Database, read: Any) -> None:
    """The first fetches wait for each other, so sequential ones time out."""
    meeting = threading.Barrier(_POOL)
    arrivals = iter(range(len(_IDS)))
    lock = threading.Lock()
    waiting = most = 0

    def _meet(**_: Any) -> None:
        nonlocal waiting, most
        with lock:
            arrival = next(arrivals)
            waiting += 1
            most = max(most, waiting)
        try:
            if arrival < _POOL:
                meeting.wait(timeout=5)
        finally:
            with lock:
                waiting -= 1

    with _hook(sync_db.s3_client.meta.events, "before-parameter-build.s3.GetObject", _meet):
        found = read(sync_db)
    assert [r.id for r in found] == _IDS
    assert most == _POOL


@_ASYNC_READS
async def test_async_fetches_overlap_within_the_pool(async_db: AsyncS3Database, read: Any) -> None:
    meeting = asyncio.Barrier(_POOL)
    arrivals = iter(range(len(_IDS)))
    waiting = most = 0

    async def _meet(**_: Any) -> None:
        nonlocal waiting, most
        waiting += 1
        most = max(most, waiting)
        try:
            if next(arrivals) < _POOL:
                async with asyncio.timeout(5):
                    await meeting.wait()
            else:
                await asyncio.sleep(0.01)
        finally:
            waiting -= 1

    with _hook(async_db._session.events, "before-parameter-build.s3.GetObject", _meet):
        found = await read(async_db)
    assert [r.id for r in found] == _IDS
    assert most == _POOL


def test_sync_abandoned_stream_stops_fetching(sync_db: SyncS3Database) -> None:
    fetched: list[str] = []
    with _hook(
        sync_db.s3_client.meta.events,
        "before-parameter-build.s3.GetObject",
        lambda params, **_: fetched.append(params["Key"]),
    ):
        stream = sync_db.stream_read()
        assert next(stream).id == "r00"
        stream.close()
    assert len(fetched) <= _POOL + 1


async def test_async_abandoned_stream_stops_fetching(async_db: AsyncS3Database) -> None:
    fetched: list[str] = []

    def _note(params: dict[str, Any], **_: Any) -> None:
        fetched.append(params["Key"])

    with _hook(async_db._session.events, "before-parameter-build.s3.GetObject", _note):
        stream = async_db.stream_read()
        assert (await anext(stream)).id == "r00"
        await stream.aclose()
    assert len(fetched) <= _POOL + 1
    current = asyncio.current_task()
    assert [t for t in asyncio.all_tasks() if t is not current] == []


async def test_async_reads_do_not_block(async_db: AsyncS3Database) -> None:
    with assert_no_blocking():
        assert len(await _async_search(async_db)) == len(_IDS)
        assert len(await _async_stream(async_db)) == len(_IDS)


# --- the keys a listing covers ------------------------------------------------


async def test_async_listing_stays_inside_its_prefix(
    async_db: AsyncS3Database, s3_config: dict[str, Any]
) -> None:
    """``p`` is not a prefix of ``pq``'s records, which clear() would delete."""
    neighbour = AsyncS3Database({**s3_config, "prefix": s3_config["prefix"] + "q"})
    await neighbour.connect()
    try:
        await neighbour.create(Record({"n": 1}, id="other"))
        assert await async_db.list_all() == _IDS
        assert await async_db.count() == len(_IDS)
        assert [r.id for r in await _async_search(async_db)] == _IDS
        assert [r.id for r in await _async_stream(async_db)] == _IDS
        assert await async_db.clear() == len(_IDS)
        assert await neighbour.list_all() == ["other"]
    finally:
        await neighbour.clear()
        await neighbour.close()


@pytest.mark.parametrize("record_id", ["a/b", "a.json.b", "a.json"])
async def test_async_listing_reports_the_stored_id(
    async_db: AsyncS3Database, record_id: str
) -> None:
    await async_db.create(Record({"n": 1}, id=record_id))
    assert record_id in await async_db.list_all()
    assert record_id in [r.id for r in await _async_search(async_db)]
    assert record_id in [r.metadata["id"] for r in await _async_stream(async_db)]


@pytest.fixture
def bare_config(make_localstack_s3_bucket: Any, raw: Any) -> Iterator[dict[str, Any]]:
    """A bucket of its own, for a database that has no prefix to stay inside."""
    bucket = f"dataknobs-bare-{uuid.uuid4().hex[:10]}"
    for base_cfg in make_localstack_s3_bucket(bucket):
        yield {**base_cfg, "prefix": ""}
        raw.delete_bucket(Bucket=bucket)


async def test_async_listing_without_a_prefix_keeps_slashes(bare_config: dict[str, Any]) -> None:
    db = AsyncS3Database(bare_config)
    await db.connect()
    try:
        await db.create(Record({"n": 1}, id="a/b"))
        assert await db.list_all() == ["a/b"]
        assert [r.id for r in await _async_search(db)] == ["a/b"]
    finally:
        await db.clear()
        await db.close()


# --- the pool bound -----------------------------------------------------------


@pytest.mark.parametrize("config_cls", [SyncS3DatabaseConfig, AsyncS3DatabaseConfig])
def test_pool_bound_is_one_setting_on_both_twins(config_cls: Any) -> None:
    assert config_cls.from_dict({"bucket": "b"}).max_pool_connections == 10
    assert config_cls.from_dict({"bucket": "b", "max_workers": "4"}).max_pool_connections == 4
    with pytest.raises(ValueError, match="max_pool_connections"):
        config_cls.from_dict({"bucket": "b", "max_pool_connections": 0})
