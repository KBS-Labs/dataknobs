"""Sync S3 atomic create-if-absent, exercised deterministically against moto.

The real-transport LocalStack coverage lives in ``test_create_if_absent_s3``
and skips when the running S3 store does not honor ``If-None-Match``. This
module pins the sync backend's conditional-PUT create path against ``moto``'s
``mock_aws`` (which honors ``If-None-Match`` and returns 412
``PreconditionFailed``), so the ``DuplicateRecordError`` raise is exercised in
CI without depending on a conforming LocalStack. ``moto``'s ``mock_aws`` is
incompatible with the aiobotocore transport, so only the sync backend is
covered here; the async backend is covered by the LocalStack module.

Depends on ``isolate_aws_env`` (integration conftest) so the ambient
``AWS_ENDPOINT_URL`` that ``bin/test.sh`` exports cannot route these "mock"
tests to the shared LocalStack container.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.s3 import SyncS3Database

pytestmark = pytest.mark.s3

_CONFIG = {"bucket": "dk-create-if-absent-moto", "region": "us-east-1"}


@pytest.fixture
def moto_s3(isolate_aws_env) -> Iterator[SyncS3Database]:
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        db = SyncS3Database(dict(_CONFIG))
        db.connect()
        try:
            yield db
        finally:
            db.clear()
            db.close()


def test_duplicate_create_raises(moto_s3: SyncS3Database) -> None:
    moto_s3.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        moto_s3.create(Record({"v": "loser"}, id="dup"))
    assert excinfo.value.id == "dup"


def test_no_clobber_on_collision(moto_s3: SyncS3Database) -> None:
    moto_s3.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        moto_s3.create(Record({"v": "loser"}, id="dup"))
    assert moto_s3.read("dup").get_value("v") == "winner"


def test_distinct_ids_still_create(moto_s3: SyncS3Database) -> None:
    moto_s3.create(Record({"v": 1}, id="a"))
    moto_s3.create(Record({"v": 2}, id="b"))
    assert moto_s3.read("a").get_value("v") == 1
    assert moto_s3.read("b").get_value("v") == 2
