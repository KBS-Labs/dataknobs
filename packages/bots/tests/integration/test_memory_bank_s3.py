"""A bank over S3 reads records stored with either form of ``created_at``.

Requires LocalStack (``bin/dk up``); the module skips when unavailable.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_common.testing import requires_localstack
from dataknobs_data import Record
from dataknobs_data.backends.s3 import SyncS3Database

pytestmark = [pytest.mark.integration, pytest.mark.s3, requires_localstack]


@pytest.fixture
def db(
    make_localstack_s3_bucket: Callable[[str], Iterator[dict[str, Any]]],
) -> Iterator[SyncS3Database]:
    for base in make_localstack_s3_bucket("dataknobs-bank-stamps"):
        database = SyncS3Database({**base, "prefix": f"bank-{uuid.uuid4().hex[:10]}/"})
        database.connect()
        try:
            yield database
        finally:
            database.clear()
            database.close()


def test_a_bank_with_a_record_stamped_in_text_keeps_working(db: SyncS3Database) -> None:
    db.create(
        Record(
            {"n": "first"},
            storage_id="first",
            metadata={"record_id": "first", "created_at": "2026-01-01T00:00:00+00:00"},
        )
    )
    bank = MemoryBank("b", schema={"required": ["n"]}, db=db)
    second = bank.add({"n": "second"})
    assert [r.record_id for r in bank.all()] == ["first", second]
    assert len(bank.find()) == 2
    assert bank.update("first", {"n": "changed"})
    records = bank.all()
    assert [r.data["n"] for r in records] == ["changed", "second"]
    for record in records:
        assert isinstance(record.created_at, float)
        assert isinstance(record.updated_at, float)
    bank.clear()
    assert bank.count() == 0
