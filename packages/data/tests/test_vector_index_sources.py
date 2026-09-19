# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Index sources over a bare table, and the index seen through the retrieval stack.

Nothing in the configuration contract's acceptance suite exercises either ---
that suite's call site has no bare table in it, and no grounded pipeline. So
these are asserted against what declares them: the reference table that names
the two live sources, the local-id boundary, and the retrieval source's own
result shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataknobs_common.records import Record

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.query import Filter, Operator, Query
from dataknobs_data.sources import RetrievalIntent, SemanticIndexSource
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import MultiFieldSource, RecordFieldSource, SemanticIndex
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

DIMENSIONS = 16

ROWS = [
    {"id": "sku-4471", "title": "Acme Widget", "summary": "a widget", "kind": "tool"},
    {"id": "sku-8802", "title": "Bolt", "summary": "", "kind": "part"},
    {"id": "sku-9110", "title": "", "summary": "no title at all", "kind": "part"},
]


@pytest.fixture
async def catalogue() -> AsyncIterator[AsyncMemoryDatabase]:
    database = AsyncMemoryDatabase()
    await database.connect()
    for row in ROWS:
        await database.create(Record(data=dict(row)))
    yield database
    await database.close()


async def test_one_field_yields_one_item_per_row_that_has_text(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """A row with nothing in the field is skipped rather than yielded empty.

    An embedded empty string is a vector that matches everything weakly and
    nothing well, which is worse inside a corpus than an absence.
    """
    source = RecordFieldSource(catalogue, "title")

    items = {item.id: item.text async for item in source.stream_items()}

    assert items == {"sku-4471": "Acme Widget", "sku-8802": "Bolt"}


async def test_the_ids_are_local_and_that_is_the_boundary(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """A source over a bare table was given no namespace, so it qualifies into none.

    The failure this boundary prevents is the quiet one: an index built over
    local ids and later read through a resolver expecting qualified ones
    returns candidates that resolve to nothing --- and *unresolved* is a
    legitimate answer, so nothing reports an error.
    """
    source = RecordFieldSource(catalogue, "title")

    assert source.declares() == frozenset()
    assert all(":" not in item.id for item in [item async for item in source.stream_items()])


async def test_a_query_narrows_what_is_indexed(catalogue: AsyncMemoryDatabase) -> None:
    """Half a table is a legitimate thing to index, and the narrowing stays in the backend."""
    source = RecordFieldSource(
        catalogue,
        "title",
        query=Query(filters=[Filter("kind", Operator.EQ, "tool")]),
    )

    assert [item.id async for item in source.stream_items()] == ["sku-4471"]


async def test_two_fields_compose_and_one_value_leaves_no_dangling_separator(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """The one-value case is the ordinary path, not an edge one.

    Two of these three rows carry one of the two fields. A separator applied
    after each value rather than between them would be invisible over a table
    where every row has both.
    """
    source = MultiFieldSource(catalogue, ["title", "summary"])

    items = {item.id: item.text async for item in source.stream_items()}

    assert items == {
        "sku-4471": "Acme Widget — a widget",
        "sku-8802": "Bolt",
        "sku-9110": "no title at all",
    }


async def test_what_the_text_was_composed_from_travels_to_the_store(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """Without it a hit from this index is indistinguishable from any other row.

    Both answer an absent source field at the published read door, so the
    absence would have to mean two things at once.
    """
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    try:
        source = MultiFieldSource(catalogue, ["title", "summary"])
        assert source.source_field == "title — summary"

        index = SemanticIndex(source, DeterministicEmbedder(dimensions=DIMENSIONS), store)
        assert await index.build() == 3

        [hit] = [h for h in await index.search("Acme Widget", k=10) if h.record.id == "sku-4471"]
        assert hit.vector_field == "title — summary"
        assert hit.source_text == "Acme Widget — a widget"
    finally:
        await store.close()


# --------------------------------------------------------------------------
# The index, seen through the retrieval stack
# --------------------------------------------------------------------------


async def test_text_queries_come_back_as_results_scored_by_the_index(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """The adapter's whole contract: intent in, normalised results out.

    An adapter rather than a base class --- the index stays usable without the
    retrieval stack, so the dependency runs one way and the index knows
    nothing about any of this.
    """
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    try:
        index = SemanticIndex(
            MultiFieldSource(catalogue, ["title", "summary"]),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        await index.build()
        source = SemanticIndexSource(index, name="catalogue")

        results = await source.query(RetrievalIntent(text_queries=["Acme Widget"]), top_k=2)

        assert len(results) == 2
        assert results[0].source_name == "catalogue"
        assert results[0].source_type == "semantic_index"
        assert results[0].relevance >= results[1].relevance
        assert results[0].content
        assert source.get_schema() is None, "a text-only source declares no filter dimensions"
    finally:
        await store.close()


async def test_two_phrasings_of_one_question_do_not_return_one_row_twice(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """Merged by id, keeping the best score.

    A row reached by two queries is one piece of evidence, and the score a
    caller acts on should be the best found rather than whichever query
    happened to run last.
    """
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    try:
        index = SemanticIndex(
            RecordFieldSource(catalogue, "title"),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        await index.build()
        source = SemanticIndexSource(index)

        results = await source.query(
            RetrievalIntent(text_queries=["Acme Widget", "Acme Widget"]), top_k=10
        )

        ids = [result.source_id for result in results]
        assert len(ids) == len(set(ids))
    finally:
        await store.close()


async def test_an_intent_carrying_no_text_asks_the_store_nothing(
    catalogue: AsyncMemoryDatabase,
) -> None:
    """An empty query list is something a pipeline produces, not something it intends."""
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    try:
        index = SemanticIndex(
            RecordFieldSource(catalogue, "title"),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        await index.build()

        assert await SemanticIndexSource(index).query(RetrievalIntent()) == []
    finally:
        await store.close()
