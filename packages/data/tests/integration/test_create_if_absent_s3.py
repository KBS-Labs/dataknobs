"""Atomic create-if-absent on the S3 backends (real transport).

``create()`` issues a conditional PUT (``If-None-Match: *``) so a colliding
id raises ``DuplicateRecordError`` instead of silently overwriting. The
guarantee depends on the S3 implementation honoring conditional writes —
real AWS S3 and recent LocalStack do; older LocalStack / moto ignore the
header and degrade to last-writer-wins. A capability probe skips the module
when the store under test does not honor ``If-None-Match``, so the tests
validate the atomic path where it is actually enforceable rather than
producing a false failure against a non-conforming store.

Sync path runs against real boto3; async path against aioboto3 and is wrapped
in :func:`assert_no_blocking` so the offload contract is proven too. Requires
LocalStack (``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from dataknobs_common.testing import (
    assert_no_blocking,
    requires_localstack,
)

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-create-if-absent"


def _store_honors_if_none_match(cfg: dict[str, Any]) -> bool:
    """True when the S3 store fails a second conditional PUT on an existing key."""
    client = boto3.client(
        "s3",
        endpoint_url=cfg.get("endpoint_url"),
        region_name=cfg.get("region", "us-east-1"),
        aws_access_key_id=cfg.get("aws_access_key_id", "test"),
        aws_secret_access_key=cfg.get("aws_secret_access_key", "test"),
    )
    key = f"_probe/{uuid.uuid4().hex}"
    client.put_object(Bucket=cfg["bucket"], Key=key, Body=b"first", IfNoneMatch="*")
    try:
        client.put_object(Bucket=cfg["bucket"], Key=key, Body=b"second", IfNoneMatch="*")
        return False  # second PUT succeeded → header ignored
    except ClientError as e:
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = e.response.get("Error", {}).get("Code")
        return status == 412 or code in ("PreconditionFailed", "412")
    finally:
        client.delete_object(Bucket=cfg["bucket"], Key=key)


@pytest.fixture
def s3_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    for base_cfg in make_localstack_s3_bucket(_BUCKET):
        if not _store_honors_if_none_match(base_cfg):
            pytest.skip(
                "S3 store under test does not honor If-None-Match conditional "
                "writes; atomic create-if-absent is not enforceable here"
            )
        yield {**base_cfg, "prefix": f"cia-{uuid.uuid4().hex[:10]}/"}


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


def test_sync_duplicate_create_raises(sync_db: SyncS3Database) -> None:
    sync_db.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_db.create(Record({"v": "loser"}, id="dup"))
    assert excinfo.value.id == "dup"
    assert sync_db.read("dup").get_value("v") == "winner"


async def test_async_duplicate_create_raises(async_db: AsyncS3Database) -> None:
    await async_db.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_db.create(Record({"v": "loser"}, id="dup"))
    assert excinfo.value.id == "dup"
    got = await async_db.read("dup")
    assert got.get_value("v") == "winner"


async def test_async_create_does_not_block_loop(async_db: AsyncS3Database) -> None:
    """The conditional-PUT create path stays off the event loop (aioboto3)."""
    with assert_no_blocking():
        await async_db.create(Record({"v": 1}, id="noblock-1"))
        with pytest.raises(DuplicateRecordError):
            await async_db.create(Record({"v": 2}, id="noblock-1"))
