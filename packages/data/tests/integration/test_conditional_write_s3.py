"""Optimistic-concurrency (conditional write) on the S3 backends (real transport).

``get_version(id)`` returns the object's ``ETag``; passing it back as
``expected_version`` issues a conditional PUT (``If-Match: <ETag>``) so the
compare-and-set is enforced by S3. A stale token raises ``ConcurrencyError``
instead of last-writer-wins. The guarantee depends on the S3 implementation
honoring ``If-Match`` — real AWS S3 and recent LocalStack do; older stores
ignore the header. A capability probe skips the module when the store under
test does not honor it, so the tests validate the atomic path where it is
actually enforceable rather than producing a false failure.

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
from dataknobs_common.testing import assert_no_blocking, requires_localstack

from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-conditional-write"


def _store_honors_if_match(cfg: dict[str, Any]) -> bool:
    """True when the S3 store rejects a PUT carrying a non-matching ``If-Match``."""
    client = boto3.client(
        "s3",
        endpoint_url=cfg.get("endpoint_url"),
        region_name=cfg.get("region", "us-east-1"),
        aws_access_key_id=cfg.get("aws_access_key_id", "test"),
        aws_secret_access_key=cfg.get("aws_secret_access_key", "test"),
    )
    key = f"_probe/{uuid.uuid4().hex}"
    client.put_object(Bucket=cfg["bucket"], Key=key, Body=b"first")
    try:
        # A deliberately-wrong ETag must be rejected.
        client.put_object(
            Bucket=cfg["bucket"], Key=key, Body=b"second", IfMatch='"0deadbeef0"'
        )
        return False  # accepted a bogus If-Match → header ignored
    except ClientError as e:
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = e.response.get("Error", {}).get("Code")
        return status in (412, 409) or code in (
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
        )
    finally:
        client.delete_object(Bucket=cfg["bucket"], Key=key)


@pytest.fixture
def s3_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    for base_cfg in make_localstack_s3_bucket(_BUCKET):
        if not _store_honors_if_match(base_cfg):
            pytest.skip(
                "S3 store under test does not honor If-Match conditional writes; "
                "optimistic concurrency is not enforceable here"
            )
        yield {**base_cfg, "prefix": f"cw-{uuid.uuid4().hex[:10]}/"}


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


def test_sync_conditional_update_fresh_token_succeeds(sync_db: SyncS3Database) -> None:
    sync_db.create(Record({"v": 0}, id="k"))
    token = sync_db.get_version("k")
    assert token is not None
    assert sync_db.update("k", Record({"v": 1}, id="k"), expected_version=token) is True
    assert sync_db.read("k").get_value("v") == 1
    assert sync_db.get_version("k") != token


def test_sync_conditional_update_stale_token_raises(sync_db: SyncS3Database) -> None:
    sync_db.create(Record({"v": 0}, id="k"))
    stale = sync_db.get_version("k")
    sync_db.update("k", Record({"v": "A"}, id="k"), expected_version=stale)
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_db.update("k", Record({"v": "B"}, id="k"), expected_version=stale)
    assert excinfo.value.context["id"] == "k"
    assert sync_db.read("k").get_value("v") == "A"


def test_sync_conditional_upsert_absent_raises(sync_db: SyncS3Database) -> None:
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_db.upsert("ghost", Record({"v": 1}, id="ghost"), expected_version='"x"')
    assert excinfo.value.context["actual_version"] is None


async def test_async_conditional_update_stale_token_raises(
    async_db: AsyncS3Database,
) -> None:
    with assert_no_blocking():
        await async_db.create(Record({"v": 0}, id="k"))
        stale = await async_db.get_version("k")
        await async_db.update("k", Record({"v": "A"}, id="k"), expected_version=stale)
        with pytest.raises(ConcurrencyError):
            await async_db.update("k", Record({"v": "B"}, id="k"), expected_version=stale)
    got = await async_db.read("k")
    assert got.get_value("v") == "A"
