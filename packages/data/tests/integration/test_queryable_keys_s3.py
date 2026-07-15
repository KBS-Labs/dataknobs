"""Queryable record identifiers on the S3 backends (real transport).

``Filter("id", ...)`` resolves to the object's storage key, and
``Operator.STARTS_WITH`` is a literal, case-sensitive prefix scan. This module
pins the async S3 backend at parity with the sync one — reproduce-first, since
the async ``_matches_filters`` previously looked up ``id`` as a *data field*
(never the storage key) and silently dropped ``BETWEEN`` / ``EXISTS`` filters.

Sync path runs against real boto3; async path against aioboto3 and is wrapped in
:func:`assert_no_blocking` so the offload contract holds. Requires LocalStack
(``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_localstack

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data.backends.s3_async import AsyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]

_BUCKET = "dataknobs-queryable-keys"
_KEYS = [
    "artifacts/alice/report/final",
    "artifacts/alice/report/draft",
    "artifacts/alice/notes/1",
    "artifacts/bob/report/final",
    "a_b/1",
    "axb/1",
]


@pytest.fixture
def s3_config(make_localstack_s3_bucket) -> Iterator[dict[str, Any]]:
    for base_cfg in make_localstack_s3_bucket(_BUCKET):
        yield {**base_cfg, "prefix": f"qk-{uuid.uuid4().hex[:10]}/"}


@pytest.fixture
def sync_db(s3_config: dict[str, Any]) -> Iterator[SyncS3Database]:
    db = SyncS3Database(s3_config)
    db.connect()
    for k in _KEYS:
        db.create(Record({"payload": k, "n": len(k)}, id=k))
    try:
        yield db
    finally:
        db.clear()
        db.close()


@pytest.fixture
async def async_db(s3_config: dict[str, Any]) -> AsyncIterator[AsyncS3Database]:
    db = AsyncS3Database(s3_config)
    await db.connect()
    for k in _KEYS:
        await db.create(Record({"payload": k, "n": len(k)}, id=k))
    try:
        yield db
    finally:
        await db.clear()
        await db.close()


def _ids(records: list[Record]) -> set[str]:
    return {r.id for r in records}


def test_sync_id_prefix_and_operators(sync_db: SyncS3Database) -> None:
    assert _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/report/")]))
    ) == {"artifacts/alice/report/final", "artifacts/alice/report/draft"}
    assert _ids(sync_db.search(Query(filters=[Filter("id", Operator.EQ, "a_b/1")]))) == {"a_b/1"}
    # Literal prefix: the '_' does not act as a wildcard, so axb/1 is excluded.
    assert _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "a_b/")]))
    ) == {"a_b/1"}


async def test_async_id_resolves_to_storage_key(async_db: AsyncS3Database) -> None:
    """Reproduce-first: async ``Filter("id", ...)`` now targets the storage key.

    Previously ``_matches_filters`` looked up ``get_field("id")`` (a data field),
    so these returned the wrong rows or none.
    """
    with assert_no_blocking():
        eq = await async_db.search(Query(filters=[Filter("id", Operator.EQ, "a_b/1")]))
        in_ = await async_db.search(
            Query(filters=[Filter("id", Operator.IN, ["a_b/1", "axb/1"])])
        )
        prefix = await async_db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/report/")])
        )
        literal = await async_db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "a_b/")])
        )
    assert _ids(eq) == {"a_b/1"}
    assert _ids(in_) == {"a_b/1", "axb/1"}
    assert _ids(prefix) == {"artifacts/alice/report/final", "artifacts/alice/report/draft"}
    assert _ids(literal) == {"a_b/1"}  # literal '_', axb/1 excluded


async def test_async_previously_dropped_operators_now_honored(
    async_db: AsyncS3Database,
) -> None:
    """Reproduce-first: BETWEEN / EXISTS were silently ignored (matched every row).

    The inline switch had no branch for them, so an unmatched operator fell
    through as a match. Delegating to ``Filter.matches`` honors them.
    """
    with assert_no_blocking():
        between = await async_db.search(
            Query(filters=[Filter("n", Operator.BETWEEN, [5, 5])])
        )
        exists = await async_db.search(
            Query(filters=[Filter("missing_field", Operator.EXISTS, None)])
        )
    assert _ids(between) == {"a_b/1", "axb/1"}  # only the length-5 keys
    assert exists == []  # no record has that field → EXISTS matches none


async def test_async_matches_sync(async_db: AsyncS3Database, s3_config: dict[str, Any]) -> None:
    """Async and sync return the same rows for the same id query."""
    sync_db = SyncS3Database(s3_config)
    sync_db.connect()
    q = Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")])
    with assert_no_blocking():
        async_rows = _ids(await async_db.search(q))
    sync_rows = _ids(sync_db.search(q))
    sync_db.close()
    assert async_rows == sync_rows
