"""A filter or sort reaches a field by how the index maps it, on both twins.

Both backends used to guess whether a field has a ``.keyword`` sub-field: the
sync sort from a list of English names, the async sort by never using one, and
a range by always taking the analyzed text. Each case here is asked of the sync
and the async backend over the same index.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.query import SortOrder

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

#: An index somebody else made: each field mapped a different way.
FOREIGN_MAPPINGS: dict[str, Any] = {
    "properties": {
        "id": {"type": "keyword"},
        "data": {
            "properties": {
                "kw": {"type": "keyword"},
                "raw": {"type": "text", "fields": {"exact": {"type": "keyword"}}},
                "day": {"type": "date"},
                "ip": {"type": "ip"},
                "words": {"type": "text"},
                "release": {"type": "version"},
                "flat": {"type": "flattened"},
                "alias": {"type": "alias", "path": "data.kw"},
            }
        },
    }
}
ROWS: dict[str, dict[str, Any]] = {
    "a": {
        "kw": "Banana",
        "raw": "Banana",
        "words": "banana",
        "day": "2024-01-01",
        "rank": 3,
        "flat": {"k": "x"},
        "ip": "10.0.0.1",
        "release": "1.2.0",
        "obj": {"k": 1},
        "flag": True,
    },
    "b": {
        "kw": "apple",
        "raw": "apple",
        "words": "apple",
        "day": "2023-01-01",
        "rank": 1,
        "flat": {"k": "y"},
        "ip": "10.0.0.2",
        "release": "1.10.0",
        "obj": {"k": 1},
        "flag": False,
    },
    "c": {
        "kw": "Cherry",
        "raw": "Cherry",
        "words": "cherry",
        "day": "2025-01-01",
        "rank": 2,
        "flat": {"k": "z"},
        "ip": "10.0.0.3",
        "release": "2.0.0",
        "obj": {"k": 1},
        "flag": False,
    },
}


@pytest.fixture
def config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_paths_")


@pytest.fixture
def index_client(config: dict[str, Any]) -> Iterator[Any]:
    from elasticsearch import Elasticsearch

    client = Elasticsearch([f"http://{config['host']}:{config['port']}"])
    yield client
    client.close()


class Twins:
    """One question put to both backends, whose answers must be the same."""

    def __init__(
        self, sync_db: SyncElasticsearchDatabase, async_db: AsyncElasticsearchDatabase
    ) -> None:
        self.sync_db = sync_db
        self.async_db = async_db

    async def search(self, query: Query) -> list[str]:
        answers: list[Any] = []
        try:
            answers.append(
                [r.storage_id for r in await asyncio.to_thread(self.sync_db.search, query)]
            )
        except Exception as exc:
            answers.append(type(exc).__name__)
        try:
            answers.append([r.storage_id for r in await self.async_db.search(query)])
        except Exception as exc:
            answers.append(type(exc).__name__)
        assert answers[0] == answers[1], "sync and async disagree"
        return answers[0]

    async def matches(self, field: str, operator: Operator, value: Any) -> list[str] | str:
        query = Query(filters=[Filter(field, operator, value)])
        found = await self.search(query)
        if isinstance(found, str):
            return found
        # count and stream_read go through the same translation.
        assert await asyncio.to_thread(self.sync_db.count, query) == len(found)
        assert await self.async_db.count(query) == len(found)
        assert sorted([r.storage_id async for r in self.async_db.stream_read(query)]) == sorted(
            found
        )
        return sorted(found)

    async def sorted_by(self, field: str, order: SortOrder = SortOrder.ASC) -> list[str]:
        return await self.search(Query().sort(field, order))


@pytest.fixture
async def twins(config: dict[str, Any]) -> AsyncIterator[Twins]:
    sync_db = SyncElasticsearchDatabase(config)
    await asyncio.to_thread(sync_db.connect)
    async_db = AsyncElasticsearchDatabase(config)
    await async_db.connect()
    yield Twins(sync_db, async_db)
    await async_db.close()
    await asyncio.to_thread(sync_db.close)


def _make_foreign(client: Any, index: str) -> None:
    """Create ``index`` with :data:`FOREIGN_MAPPINGS` and write :data:`ROWS` to it."""
    client.indices.create(index=index, mappings=FOREIGN_MAPPINGS)
    for row_id, data in ROWS.items():
        client.index(index=index, id=row_id, document={"id": row_id, "data": data})
    client.indices.refresh(index=index)


def _seed_foreign(config: dict[str, Any]) -> None:
    from elasticsearch import Elasticsearch

    client = Elasticsearch([f"http://{config['host']}:{config['port']}"])
    try:
        _make_foreign(client, config["index"])
    finally:
        client.close()


@pytest.fixture
def foreign_index(config: dict[str, Any], index_client: Any) -> None:
    """The foreign index, made in this test's own index, for a test that changes it."""
    _make_foreign(index_client, config["index"])


class TestAnIndexMappedElsewhere:
    """Every case only reads, so they share one foreign index, seeded once."""

    @pytest.fixture
    def config(self, make_elasticsearch_corpus: Any) -> dict[str, Any]:
        return make_elasticsearch_corpus("test_paths_foreign_", _seed_foreign)

    @pytest.mark.parametrize(
        ("field", "ascending"),
        [
            ("rank", ["b", "c", "a"]),
            ("kw", ["a", "c", "b"]),
            ("raw", ["a", "c", "b"]),
            ("raw.exact", ["a", "c", "b"]),
            ("day", ["b", "a", "c"]),
            ("flat.k", ["a", "b", "c"]),
            ("alias", ["a", "c", "b"]),
        ],
    )
    async def test_every_field_sorts(self, twins: Twins, field: str, ascending: list[str]) -> None:
        assert await twins.sorted_by(field) == ascending
        assert await twins.sorted_by(field, SortOrder.DESC) == ascending[::-1]

    @pytest.mark.parametrize(
        ("field", "operator", "value", "expected"),
        [
            ("kw", Operator.EQ, "apple", ["b"]),
            ("kw", Operator.IN, ["apple", "Cherry"], ["b", "c"]),
            ("kw", Operator.LIKE, "BAN%", ["a"]),
            ("kw", Operator.STARTS_WITH, "Ch", ["c"]),
            ("kw", Operator.GT, "a", ["b"]),
            ("raw", Operator.EQ, "Banana", ["a"]),
            ("raw", Operator.GT, "a", ["b"]),
            ("raw", Operator.BETWEEN, ["B", "D"], ["a", "c"]),
            ("day", Operator.GT, "2024-06-01", ["c"]),
            ("day", Operator.LT, datetime(2023, 6, 1), ["b"]),
            ("flat.k", Operator.EQ, "y", ["b"]),
            ("flat.k", Operator.BETWEEN, ["y", "z"], ["b", "c"]),
            ("alias", Operator.EQ, "apple", ["b"]),
            # Equality reads a date as a range does.
            ("day", Operator.EQ, "2024-01-01", ["a"]),
            ("day", Operator.EQ, "2024-01-01T00:00:00Z", ["a"]),
            ("day", Operator.EQ, datetime(2024, 1, 1), ["a"]),
            ("day", Operator.NEQ, "2024-01-01", ["b", "c"]),
            ("day", Operator.IN, ["2024-01-01", "2023-01-01"], ["a", "b"]),
            ("day", Operator.NOT_IN, ["2024-01-01"], ["b", "c"]),
            ("day", Operator.EQ, "apple", "BadRequestError"),
            ("day", Operator.GT, "apple", "BadRequestError"),
            ("ip", Operator.EQ, "10.0.0.2", ["b"]),
            ("ip", Operator.NEQ, "10.0.0.2", ["a", "c"]),
            ("ip", Operator.IN, ["10.0.0.1", "10.0.0.3"], ["a", "c"]),
            # A string is never equal to a number or a boolean, whatever it spells.
            ("rank", Operator.EQ, "apple", []),
            ("rank", Operator.EQ, "3", []),
            ("rank", Operator.NEQ, "3", ["a", "b", "c"]),
            ("rank", Operator.IN, ["3"], []),
            ("flag", Operator.EQ, "true", []),
            ("flag", Operator.EQ, True, ["a"]),
            # Only words are reachable here, and no comparison is about a word.
            ("words", Operator.EQ, "apple", "ValueError"),
            ("words", Operator.NEQ, "apple", "ValueError"),
            ("words", Operator.IN, ["apple"], "ValueError"),
            ("words", Operator.NOT_IN, ["apple"], "ValueError"),
            ("words", Operator.GT, "apple", "ValueError"),
            ("words", Operator.NOT_BETWEEN, ["a", "b"], "ValueError"),
            ("words", Operator.LIKE, "app%", "ValueError"),
            ("words", Operator.NOT_LIKE, "app%", "ValueError"),
            ("words", Operator.EQ, None, []),
            ("words", Operator.EXISTS, None, ["a", "b", "c"]),
            # Nor can a pattern be tried against a date or an address.
            ("day", Operator.STARTS_WITH, "20", "ValueError"),
            ("day", Operator.NOT_LIKE, "20%", "ValueError"),
            ("ip", Operator.LIKE, "10%", "ValueError"),
            ("ip", Operator.REGEX, "10.*", "ValueError"),
            # These hold no strings, so a pattern matches none of them.
            ("rank", Operator.LIKE, "1%", []),
            ("rank", Operator.NOT_LIKE, "1%", []),
            ("obj", Operator.LIKE, "1%", []),
            ("obj", Operator.NOT_LIKE, "1%", []),
            ("release", Operator.STARTS_WITH, "1.", ["a", "b"]),
            ("release", Operator.LIKE, "1.%", ["a", "b"]),
            ("release", Operator.NOT_LIKE, "1.%", ["c"]),
        ],
    )
    async def test_every_field_filters(
        self, twins: Twins, field: str, operator: Operator, value: Any, expected: list[str]
    ) -> None:
        assert await twins.matches(field, operator, value) == expected


class TestANewIndex:
    async def test_sorting_a_field_no_record_has(self, twins: Twins) -> None:
        assert await twins.sorted_by("absent") == []
        twins.sync_db.create(Record({"x": 1}, storage_id="a"))
        assert await twins.sorted_by("absent") == ["a"]

    async def test_a_field_is_read_again_once_it_exists(self, twins: Twins) -> None:
        assert await twins.matches("late", Operator.GT, "a") == []
        twins.sync_db.create(Record({"late": "Zebra"}, storage_id="a"))
        twins.sync_db.create(Record({"late": "ant"}, storage_id="b"))
        assert await twins.matches("late", Operator.GT, "a") == ["b"]
        assert await twins.sorted_by("late") == ["a", "b"]

    async def test_a_string_past_256_characters_still_compares(self, twins: Twins) -> None:
        for row_id, text in {"a": "b" * 300, "b": "a" * 300}.items():
            twins.sync_db.create(Record({"t": text, "n": {"t": text}}, storage_id=row_id))
        assert await twins.matches("t", Operator.GT, "ab") == ["a"]
        assert await twins.matches("n.t", Operator.EQ, "a" * 300) == ["b"]
        assert await twins.sorted_by("t") == ["b", "a"]

    async def test_a_string_too_long_for_a_term_is_still_stored(self, twins: Twins) -> None:
        text = "é" * 40_000
        twins.sync_db.create(Record({"t": text}, storage_id="a"))
        found = twins.sync_db.read("a")
        assert found is not None and found.get_value("t") == text

    async def test_a_string_field_does_not_become_a_date(self, twins: Twins) -> None:
        for row_id, text in {"a": "2024-01-01", "b": "not a date", "c": "2023-01-01"}.items():
            twins.sync_db.create(Record({"d": text}, storage_id=row_id))
        assert await twins.matches("d", Operator.GTE, "2024") == ["a", "b"]
        assert await twins.matches("d", Operator.EQ, "2024-01-01") == ["a"]
        assert await twins.sorted_by("d") == ["c", "a", "b"]

    async def test_a_datetime_bound_compares_as_the_text_it_is_written_as(
        self, twins: Twins
    ) -> None:
        for row_id, hour in {"a": 9, "b": 12}.items():
            when = datetime(2024, 1, 1, hour)
            twins.sync_db.create(Record({"when": when}, storage_id=row_id))
        # Another spelling of 10:00, which as text comes before any "T".
        twins.sync_db.create(Record({"when": "2024-01-01 10:00:00"}, storage_id="c"))
        bound = datetime(2024, 1, 1, 9, 30)
        assert await twins.matches("when", Operator.GT, bound) == ["b"]
        assert await twins.matches("when", Operator.LT, bound) == ["a", "c"]

    async def test_text_search_still_reads_the_words(
        self, twins: Twins, config: dict[str, Any], index_client: Any
    ) -> None:
        twins.sync_db.create(Record({"t": "The Quick fox"}, storage_id="a"))
        found = index_client.count(index=config["index"], query={"match": {"data.t": "quick"}})
        assert found["count"] == 1


def _open_sync(config: dict[str, Any]) -> None:
    db = SyncElasticsearchDatabase(config)
    db.connect()
    db.close()


async def _open(flavour: str, config: dict[str, Any]) -> None:
    if flavour == "sync":
        await asyncio.to_thread(_open_sync, config)
        return
    db = AsyncElasticsearchDatabase(config)
    await db.connect()
    await db.close()


@pytest.mark.parametrize("flavour", ["sync", "async"])
async def test_configured_mappings_and_settings_reach_the_index(
    flavour: str, config: dict[str, Any], index_client: Any
) -> None:
    """Configured mappings add to the declared ones, and win where both speak."""
    await _open(
        flavour,
        {
            **config,
            "mappings": {"properties": {"data": {"properties": {"due": {"type": "date"}}}}},
            "settings": {"number_of_shards": 2, "number_of_replicas": 0},
        },
    )
    made = index_client.indices.get(index=config["index"])[config["index"]]
    assert made["mappings"]["properties"]["data"]["properties"]["due"] == {"type": "date"}
    assert made["mappings"]["properties"]["id"] == {"type": "keyword"}
    assert made["mappings"]["date_detection"] is False
    assert made["mappings"]["dynamic_templates"]
    assert made["settings"]["index"]["number_of_shards"] == "2"


@pytest.mark.parametrize("data", [{"type": "flattened"}, {"enabled": False}, {"dynamic": False}])
async def test_the_data_object_can_be_mapped_another_way(
    data: dict[str, Any], config: dict[str, Any], index_client: Any
) -> None:
    await _open("sync", {**config, "mappings": {"properties": {"data": data}}})
    made = index_client.indices.get_mapping(index=config["index"])[config["index"]]
    mapped = made["mappings"]["properties"]["data"]
    # Elasticsearch reports ``dynamic`` back as a string.
    assert {k: str(mapped[k]).lower() for k in data} == {k: str(v).lower() for k, v in data.items()}


async def test_configured_dynamic_templates_are_tried_first(
    config: dict[str, Any], index_client: Any
) -> None:
    whole = {"codes": {"path_match": "data.code", "mapping": {"type": "keyword"}}}
    await _open("sync", {**config, "mappings": {"dynamic_templates": [whole]}})
    index_client.index(index=config["index"], document={"data": {"code": "a b", "t": "x"}})
    made = index_client.indices.get_mapping(index=config["index"])[config["index"]]
    fields = made["mappings"]["properties"]["data"]["properties"]
    assert fields["code"] == {"type": "keyword"}
    assert fields["t"]["fields"]["keyword"]["ignore_above"] == 8191


async def test_both_twins_create_the_same_mapping(
    config: dict[str, Any], index_client: Any
) -> None:
    made = []
    for flavour in ("sync", "async"):
        await _open(flavour, config)
        made.append(index_client.indices.get_mapping(index=config["index"])[config["index"]])
        index_client.indices.delete(index=config["index"])
    assert made[0] == made[1]


@pytest.mark.parametrize("flavour", ["sync", "async"])
@pytest.mark.usefixtures("foreign_index")
async def test_a_vector_pre_filter_reads_the_mapping(
    flavour: str, twins: Twins, config: dict[str, Any], index_client: Any
) -> None:
    index_client.indices.put_mapping(
        index=config["index"],
        properties={"data": {"properties": {"v": {"type": "dense_vector", "dims": 2}}}},
    )
    for row_id in ROWS:
        index_client.update(
            index=config["index"], id=row_id, doc={"data": {"v": [1.0, 2.0]}}, refresh=True
        )
    query = Query(filters=[Filter("kw", Operator.EQ, "apple")])
    if flavour == "sync":
        hits = await asyncio.to_thread(
            lambda: twins.sync_db.vector_search([1.0, 2.0], vector_field="v", k=3, filter=query)
        )
    else:
        hits = await twins.async_db.vector_search([1.0, 2.0], vector_field="v", k=3, filter=query)
    assert [h.record.storage_id for h in hits] == ["b"]
