"""Fail-closed streaming INSERT on the S3 backends (real transport).

S3 has no cheaper bulk write than a per-key PUT, so ``create_batch`` inherits the
ABC-default per-record ``create()`` loop (already fail-closed) and the streaming
INSERT path routes through per-record ``create()`` too (``insert_batch_func=None``
— a non-transactional bulk + per-record fallback would double-write). A colliding
source id in a re-run into a populated target is therefore recorded as a failure
with the source id preserved.

Fail-closed create on S3 depends on the store honoring conditional writes
(``If-None-Match: *``); a capability probe skips the module when it does not.
Requires LocalStack (``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from dataknobs_common.testing import assert_no_blocking, requires_localstack

from dataknobs_data import Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.streaming import StreamConfig

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-create-batch-fc"


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
        return False
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
                "S3 store under test does not honor If-None-Match; fail-closed "
                "streaming INSERT is not enforceable here"
            )
        yield {**base_cfg, "prefix": f"cbfc-{uuid.uuid4().hex[:10]}/"}


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


def test_sync_streaming_insert_fails_closed(sync_db: SyncS3Database) -> None:
    sync_db.create(Record({"v": "old"}, id="2"))
    records = [Record({"v": "src"}, id=str(i)) for i in (1, 2, 3)]
    result = sync_db.stream_write(
        iter(records), StreamConfig(on_error=lambda e, r: True)
    )
    assert result.failed == 1
    assert result.successful == 2
    assert sync_db.read("2").get_value("v") == "old"
    assert sync_db.read("1").get_value("v") == "src"


async def test_async_streaming_insert_fails_closed(async_db: AsyncS3Database) -> None:
    await async_db.create(Record({"v": "old"}, id="2"))

    async def gen() -> AsyncIterator[Record]:
        for i in (1, 2, 3):
            yield Record({"v": "src"}, id=str(i))

    with assert_no_blocking():
        result = await async_db.stream_write(
            gen(), StreamConfig(on_error=lambda e, r: True)
        )
    assert result.failed == 1
    assert result.successful == 2
    assert (await async_db.read("2")).get_value("v") == "old"
    assert (await async_db.read("1")).get_value("v") == "src"
