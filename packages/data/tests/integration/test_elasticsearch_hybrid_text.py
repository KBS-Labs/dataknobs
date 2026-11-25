"""The text half of a hybrid search, which both twins ask Elasticsearch the same way.

The sync twin sent a ``LIKE`` on the first field. On a text field with no
keyword sub-field, the mapping full-text search wants, that cannot be answered.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import Elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.vector.hybrid import FusionStrategy, HybridSearchConfig

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

Database = SyncElasticsearchDatabase | AsyncElasticsearchDatabase

ROWS = {
    "a": {"title": "red fox", "content": "a quick animal", "kind": "animal"},
    "b": {"title": "blue whale", "content": "a slow fox hunter", "kind": "animal"},
    "c": {"title": "green tea", "content": "a drink", "kind": "drink"},
}
#: ``title`` is text and no more; ``kind`` is the keyword a filter can use.
MAPPINGS: dict[str, Any] = {
    "properties": {
        "id": {"type": "keyword"},
        "data": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "kind": {"type": "keyword"},
                "v": {"type": "dense_vector", "dims": 3, "index": True, "similarity": "cosine"},
            }
        },
    }
}


async def settle(value: Any) -> Any:
    """The result of a call on either twin."""
    return await value if inspect.isawaitable(value) else value


@pytest.fixture
def es_config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_hybrid_text_")


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(request: Any, es_config: dict[str, Any]) -> AsyncIterator[Database]:
    with Elasticsearch(f"http://{es_config['host']}:{es_config['port']}") as admin:
        admin.indices.create(index=es_config["index"], mappings=MAPPINGS)
    database = request.param({**es_config, "refresh": True})
    await settle(database.connect())
    for record_id, row in ROWS.items():
        await settle(database.create(Record(data={**row, "v": [1.0, 0.0, 0.0]}, id=record_id)))
    yield database
    await settle(database.close())


async def found(db: Database, text: str, fields: list[str] | None, **more: Any) -> list[str]:
    hits = await settle(db._text_search_for_hybrid(text, fields, k=more.pop("k", 5), **more))
    assert all(0.0 < score <= 1.0 for _, score in hits)
    return [record.storage_id for record, _ in hits]


async def test_a_word_of_a_text_field_is_found(db: Database) -> None:
    assert await found(db, "fox", ["title"]) == ["a"]


async def test_any_word_is_enough(db: Database) -> None:
    assert set(await found(db, "whale fox", ["title"])) == {"a", "b"}


async def test_every_field_is_searched(db: Database) -> None:
    assert set(await found(db, "fox", ["title", "content"])) == {"a", "b"}


async def test_the_fields_searched_when_none_are_named(db: Database) -> None:
    assert set(await found(db, "fox", None)) == {"a", "b"}


async def test_the_best_match_scores_one(db: Database) -> None:
    hits = await settle(db._text_search_for_hybrid("red fox whale", ["title"], k=5))
    assert hits[0][0].storage_id == "a"
    assert hits[0][1] == 1.0
    assert hits[1][1] < 1.0


async def test_no_more_than_k(db: Database) -> None:
    assert len(await found(db, "fox", ["title", "content"], k=1)) == 1


async def test_nothing_is_found(db: Database) -> None:
    assert await found(db, "absent", ["title"]) == []


async def test_the_filter_applies(db: Database) -> None:
    drinks = Query(filters=[Filter("kind", Operator.EQ, "drink")])
    assert await found(db, "fox tea", ["title"], filter=drinks) == ["c"]


@pytest.mark.parametrize("strategy", FusionStrategy)
async def test_a_hybrid_search_on_plain_text(db: Database, strategy: FusionStrategy) -> None:
    results = await settle(
        db.hybrid_search(
            "tea",
            np.array([1.0, 0.0, 0.0]),
            text_fields=["title"],
            vector_field="v",
            k=3,
            config=HybridSearchConfig(fusion_strategy=strategy),
        )
    )
    # The vectors are equal, so the text decides what comes first.
    assert results[0].record.storage_id == "c"
    assert {result.record.storage_id for result in results} == set(ROWS)


#: Fields a word cannot be looked up in, each with a value a record can hold.
NOT_TEXT: dict[str, tuple[dict[str, Any], Any]] = {
    "long": ({"type": "long"}, 5),
    "date": ({"type": "date"}, "2024-01-01"),
    "boolean": ({"type": "boolean"}, True),
    "ip": ({"type": "ip"}, "10.0.0.1"),
    "integer_range": ({"type": "integer_range"}, {"gte": 1, "lte": 5}),
    "geo_point": ({"type": "geo_point"}, {"lat": 1.0, "lon": 1.0}),
}


async def _with_field(db: Database, name: str, kind: str) -> None:
    """Map ``name`` as ``kind``, and have a record that holds a value there."""
    mapping, value = NOT_TEXT[kind]
    client = db._require_client()
    properties = {"data": {"properties": {name: mapping}}}
    await settle(client.indices.put_mapping(index=db.index_name, properties=properties))
    row = {"title": "grey fox", name: value, "v": [1.0, 0.0, 0.0]}
    await settle(db.create(Record(data=row, id="d")))


@pytest.mark.parametrize("kind", NOT_TEXT)
async def test_a_field_that_holds_no_words_matches_none(db: Database, kind: str) -> None:
    await _with_field(db, "other", kind)
    assert await found(db, "fox", ["other"]) == []
    assert set(await found(db, "fox", ["title", "other"])) == {"a", "d"}


async def test_a_vector_field_matches_no_word(db: Database) -> None:
    assert await found(db, "fox", ["title", "v"]) == ["a"]


async def test_a_default_field_that_holds_no_words(db: Database) -> None:
    """``text`` is searched when no field is named, whatever the index holds there."""
    await _with_field(db, "text", "boolean")
    assert set(await found(db, "fox", None)) == {"a", "b", "d"}


async def test_a_value_a_number_field_holds_is_found(db: Database) -> None:
    await _with_field(db, "other", "long")
    assert await found(db, "5", ["title", "other"]) == ["d"]


UNANSWERABLE = Query(filters=[Filter("title", Operator.NEQ, "red fox")])


@pytest.mark.parametrize("name", ["vector", "hybrid", "native", "text"])
async def test_a_filter_the_index_cannot_answer_raises(db: Database, name: str) -> None:
    vector = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="cannot answer NEQ on 'title'"):
        if name == "vector":
            await settle(db.vector_search(vector, vector_field="v", k=2, filter=UNANSWERABLE))
        elif name == "text":
            await settle(db._text_search_for_hybrid("fox", ["content"], 2, UNANSWERABLE))
        else:
            chosen = FusionStrategy.NATIVE if name == "native" else FusionStrategy.RRF
            await settle(
                db.hybrid_search(
                    "fox",
                    vector,
                    text_fields=["content"],
                    vector_field="v",
                    k=2,
                    config=HybridSearchConfig(fusion_strategy=chosen),
                    filter=UNANSWERABLE,
                )
            )
