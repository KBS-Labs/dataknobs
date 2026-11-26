"""``seed_elasticsearch_records`` seeds a shared corpus as the backend would.

A corpus seeded this way has to look, to a backend opened on it, exactly like
an index the backend created and filled itself: the same mapping, the same
ids. Otherwise a test moved onto a corpus would be testing a different index
from the one it tested before.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_data import Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.testing import seed_elasticsearch_records

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

RECORDS = [Record({"n": n, "word": f"w{n}"}, id=f"r{n}") for n in range(3)]


def _records() -> Iterator[Record]:
    """A generator, read once by the helper."""
    yield from RECORDS
    yield Record({"n": 99, "word": "minted"})


@pytest.fixture
def corpus(make_elasticsearch_corpus: Any) -> dict[str, Any]:
    return make_elasticsearch_corpus(
        "test_seed_records_",
        seed_elasticsearch_records(
            _records(),
            mappings={"properties": {"data": {"properties": {"word": {"type": "keyword"}}}}},
        ),
    )


def test_every_record_is_there_with_its_id(corpus: dict[str, Any]) -> None:
    db = SyncElasticsearchDatabase(corpus)
    db.connect()
    try:
        found = {record.id: record.get_value("n") for record in db.search(Query())}
    finally:
        db.close()

    assert {k: v for k, v in found.items() if k in {"r0", "r1", "r2"}} == {
        "r0": 0,
        "r1": 1,
        "r2": 2,
    }
    # The one record without an id was given one.
    assert len(found) == 4
    assert found[next(k for k in found if k not in {"r0", "r1", "r2"})] == 99


async def test_the_index_has_the_backends_mapping_and_the_extra_config(
    corpus: dict[str, Any],
) -> None:
    from elasticsearch import AsyncElasticsearch

    client = AsyncElasticsearch([f"http://{corpus['host']}:{corpus['port']}"])
    try:
        mapping = await client.indices.get_mapping(index=corpus["index"])
    finally:
        await client.close()
    properties = mapping[corpus["index"]]["mappings"]["properties"]

    assert properties["id"] == {"type": "keyword"}
    assert properties["data"]["properties"]["word"] == {"type": "keyword"}


async def test_the_async_backend_reads_it_too(corpus: dict[str, Any]) -> None:
    db = AsyncElasticsearchDatabase(corpus)
    await db.connect()
    try:
        assert await db.count() == 4
    finally:
        await db.close()
