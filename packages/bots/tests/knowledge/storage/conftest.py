"""Shared LocalStack fixtures for S3 knowledge-backend integration tests.

``S3KnowledgeBackend`` now drives its S3 I/O through aioboto3 (so the
event loop is never blocked). ``moto``'s in-process ``mock_aws`` is
incompatible with aiobotocore, so these tests run against a real
LocalStack S3 instead — the same harness the data package's S3 backend
and the common package's ``SqsEventBus`` tests use. Start it with
``bin/dk up`` (LocalStack runs ``SERVICES=s3,sqs``); the whole S3
integration suite skips when it is unavailable.

The bucket-creation factory comes from dataknobs-common's pytest11
plugin (``make_localstack_s3_bucket``). LocalStack persists buckets for
the session, so each fixture stamps a **unique key prefix per test** to
keep object state disjoint across tests and reruns (the snapshot-diff
tests assert exact change sets — they cannot share a key namespace).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

#: One shared unversioned bucket for snapshot-mode / functional tests.
_UNVERSIONED_BUCKET = "kb-s3-int-unversioned"
#: A separate bucket with S3 versioning enabled, for ``s3_versioning``
#: change-detection mode (the metadata object's own version history IS
#: the snapshot store). Versioning is bucket-level, so it cannot share a
#: bucket with the versioning-disabled fallback test.
_VERSIONED_BUCKET = "kb-s3-int-versioned"


def _unique_prefix() -> str:
    """A per-test key prefix so object state never collides across runs."""
    return f"kb-{uuid.uuid4().hex[:10]}/"


async def _enable_bucket_versioning(cfg: dict[str, Any]) -> None:
    """Turn on S3 versioning for the bucket named in ``cfg``."""
    import aioboto3

    session = aioboto3.Session(
        region_name=cfg["region"],
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
    )
    async with session.client(
        "s3", endpoint_url=cfg["endpoint_url"], use_ssl=False
    ) as s3:
        await s3.put_bucket_versioning(
            Bucket=cfg["bucket"],
            VersioningConfiguration={"Status": "Enabled"},
        )


@pytest.fixture
def s3_kb_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    """Config dict for an S3KnowledgeBackend on an unversioned bucket.

    Yields the dataknobs S3 config shape (bucket / endpoint_url / region
    / credentials) with a unique per-test ``prefix`` so each test owns a
    disjoint key namespace.
    """
    for base_cfg in make_localstack_s3_bucket(_UNVERSIONED_BUCKET):
        yield {**base_cfg, "prefix": _unique_prefix()}


@pytest.fixture
def s3_kb_versioned_config(
    make_localstack_s3_bucket,
) -> Iterator[dict[str, Any]]:
    """Config dict for an S3KnowledgeBackend on a versioning-enabled bucket.

    Used by the ``s3_versioning`` change-detection-mode tests. Versioning
    is enabled on the bucket during setup (idempotent on LocalStack).
    """
    for base_cfg in make_localstack_s3_bucket(_VERSIONED_BUCKET):
        asyncio.run(_enable_bucket_versioning(base_cfg))
        yield {**base_cfg, "prefix": _unique_prefix()}
