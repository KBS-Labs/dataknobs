"""The S3 backends stamp ``created_at`` / ``updated_at`` from the common clock.

Both twins read :func:`dataknobs_common.clock.utc_now`, so
:func:`~dataknobs_common.clock.use_clock` fixes the stamps a write makes, and
each write reads the clock **once**: a record the write has to give a
``created_at`` gets the same instant as its ``updated_at``. The clock used
here advances on every read, so a write reading it twice would stamp two
different instants.

Requires LocalStack (``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from dataknobs_common.clock import use_clock
from dataknobs_common.testing import FixedClock, requires_localstack

from dataknobs_data import Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-clock"
AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _ticking(clock: FixedClock) -> Any:
    """A clock that moves one second forward on every read."""
    return lambda: clock.advance(timedelta(seconds=1))


@pytest.fixture
def s3_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    for base_cfg in make_localstack_s3_bucket(_BUCKET):
        yield {**base_cfg, "prefix": f"clock-{uuid.uuid4().hex[:10]}/"}


@pytest.fixture
def sync_db(s3_config: dict[str, Any]) -> Iterator[SyncS3Database]:
    db = SyncS3Database(s3_config)
    db.connect()
    try:
        yield db
    finally:
        db.clear()
        db.close()


@pytest.fixture
async def async_db(s3_config: dict[str, Any]) -> AsyncIterator[AsyncS3Database]:
    db = AsyncS3Database(s3_config)
    await db.connect()
    try:
        yield db
    finally:
        await db.clear()
        await db.close()


def test_sync_create_stamps_the_clock_in_use(sync_db: SyncS3Database) -> None:
    with use_clock(FixedClock(AT)):
        record_id = sync_db.create(Record({"name": "a"}))
    stored = sync_db.read(record_id)
    assert stored is not None
    assert stored.metadata["created_at"] == AT.isoformat()
    assert stored.metadata["updated_at"] == AT.isoformat()


def test_sync_update_keeps_created_at_and_stamps_updated_at(sync_db: SyncS3Database) -> None:
    with use_clock(FixedClock(AT)):
        record_id = sync_db.create(Record({"name": "a"}))
    later = AT + timedelta(hours=1)
    with use_clock(FixedClock(later)):
        assert sync_db.update(record_id, Record({"name": "b"}))
    stored = sync_db.read(record_id)
    assert stored is not None
    assert stored.metadata["created_at"] == AT.isoformat()
    assert stored.metadata["updated_at"] == later.isoformat()


def test_sync_update_stamps_a_missing_created_at_from_the_same_read(
    sync_db: SyncS3Database, s3_config: dict[str, Any]
) -> None:
    # An object another writer stored without a ``created_at``.
    record_id = "written-elsewhere"
    body = Record({"name": "a"}).to_dict(include_metadata=True, flatten=False)
    sync_db.s3_client.put_object(
        Bucket=s3_config["bucket"],
        Key=f"{s3_config['prefix']}{record_id}.json",
        Body=json.dumps(body),
        ContentType="application/json",
    )
    with use_clock(_ticking(FixedClock(AT))):
        assert sync_db.update(record_id, Record({"name": "b"}))
    stored = sync_db.read(record_id)
    assert stored is not None
    assert stored.metadata["created_at"] == stored.metadata["updated_at"]


async def test_async_create_stamps_the_clock_in_use(async_db: AsyncS3Database) -> None:
    with use_clock(FixedClock(AT)):
        record_id = await async_db.create(Record({"name": "a"}))
    stored = await async_db.read(record_id)
    assert stored is not None
    assert stored.metadata["created_at"] == AT.isoformat()
    assert stored.metadata["updated_at"] == AT.isoformat()


async def test_async_write_reads_the_clock_once(async_db: AsyncS3Database) -> None:
    with use_clock(_ticking(FixedClock(AT))):
        record_id = await async_db.create(Record({"name": "a"}))
    stored = await async_db.read(record_id)
    assert stored is not None
    assert stored.metadata["created_at"] == stored.metadata["updated_at"]
