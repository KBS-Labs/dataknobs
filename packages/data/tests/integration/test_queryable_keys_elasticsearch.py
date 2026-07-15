"""Queryable record identifiers on the Elasticsearch backends.

``Filter("id", ...)`` resolves to the document's storage key, and
``Operator.STARTS_WITH`` compiles to a case-sensitive ``prefix`` query on the
``id`` keyword field. This module pins the async backend at parity with the sync
one — reproduce-first, since the async ``search`` / ``count`` loops previously
built ``data.id`` unconditionally, so an id filter never resolved to the storage
key.

Enable with ``TEST_ELASTICSEARCH=true`` and a running Elasticsearch; the module
skips otherwise.
"""

from __future__ import annotations

import os

import pytest

from dataknobs_data import Filter, Operator, Query, Record, SortOrder, SortSpec
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.query_logic import ComplexQuery, LogicCondition, LogicOperator
from dataknobs_data.query_logic import FilterCondition

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_ELASTICSEARCH") != "true",
    reason="Elasticsearch integration tests not enabled",
)

_KEYS = [
    "artifacts/alice/report/final",
    "artifacts/alice/report/draft",
    "artifacts/bob/report/final",
    "orders/1",
    "orders/2",
]


def _ids(records: list[Record]) -> set[str]:
    return {r.id for r in records}


def _seed_sync(db: SyncElasticsearchDatabase) -> None:
    for k in _KEYS:
        db.create(Record({"payload": k}, id=k))


async def _seed_async(db: AsyncElasticsearchDatabase) -> None:
    for k in _KEYS:
        await db.create(Record({"payload": k}, id=k))


def test_sync_id_prefix_and_operators(elasticsearch_test_index) -> None:
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        _seed_sync(db)
        assert _ids(
            db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/")]))
        ) == {"artifacts/alice/report/final", "artifacts/alice/report/draft"}
        assert _ids(db.search(Query(filters=[Filter("id", Operator.EQ, "orders/1")]))) == {
            "orders/1"
        }
        assert _ids(
            db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "orders/")]))
        ) == {"orders/1", "orders/2"}
    finally:
        db.close()


async def test_async_id_resolves_to_storage_key(elasticsearch_test_index) -> None:
    """Reproduce-first: async id filters now target the storage key, not data.id."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await _seed_async(db)
        eq = await db.search(Query(filters=[Filter("id", Operator.EQ, "orders/1")]))
        prefix = await db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/")])
        )
        in_ = await db.search(
            Query(filters=[Filter("id", Operator.IN, ["orders/1", "orders/2"])])
        )
        assert _ids(eq) == {"orders/1"}
        assert _ids(prefix) == {
            "artifacts/alice/report/final",
            "artifacts/alice/report/draft",
        }
        assert _ids(in_) == {"orders/1", "orders/2"}
    finally:
        await db.close()


async def test_async_count_id_prefix(elasticsearch_test_index) -> None:
    """The async ``count`` loop honors the same id resolution + prefix."""
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await _seed_async(db)
        n = await db.count(Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")]))
        assert n == 3
        n_orders = await db.count(Query(filters=[Filter("id", Operator.EQ, "orders/2")]))
        assert n_orders == 1
    finally:
        await db.close()


async def test_async_matches_sync(elasticsearch_test_index) -> None:
    """Async and sync ES return the same rows for the same id prefix query."""
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_sync(sync_db)
        q = Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")])
        assert _ids(await async_db.search(q)) == _ids(sync_db.search(q))
    finally:
        sync_db.close()
        await async_db.close()


async def test_async_minted_id_is_filterable(elasticsearch_test_index) -> None:
    """Reproduce-first: a record created without an explicit id (minted uuid)
    is still findable by ``Filter("id", ...)``.

    The async backend built the document *before* minting the id and passed the
    minted value only as ES ``_id`` — never stamping it into the document body.
    Since id filters target the top-level ``id`` keyword field (mirroring
    ``_id``), that field was absent for minted-id records, so EQ / STARTS_WITH /
    IN on the minted id returned nothing. The sync backend never had this gap
    (it always stamps ``doc["id"]``), so this is a sync/async parity fix.
    """
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        minted = await db.create(Record({"payload": "no-id"}))
        assert minted  # create() minted and returned a uuid
        eq = await db.search(Query(filters=[Filter("id", Operator.EQ, minted)]))
        assert _ids(eq) == {minted}
        prefix = await db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, minted[:8])])
        )
        assert _ids(prefix) == {minted}
        in_ = await db.search(Query(filters=[Filter("id", Operator.IN, [minted])]))
        assert _ids(in_) == {minted}
    finally:
        await db.close()


async def test_async_batch_minted_ids_are_filterable(elasticsearch_test_index) -> None:
    """The minting write paths beyond ``create()`` stamp the resolved id into the
    document body too, so a record they mint stays findable by ``Filter("id", ...)``.

    Covers the same ``_record_to_doc(record, id)`` id-stamping fix through
    ``create_batch`` / ``upsert_batch`` / ``upsert(record)`` — the write paths
    that resolve their own id when the record has none.
    """
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        # create_batch mints a uuid per id-less record.
        created = await db.create_batch(
            [Record({"payload": "a"}), Record({"payload": "b"})]
        )
        assert len(created) == 2
        for rid in created:
            assert _ids(
                await db.search(Query(filters=[Filter("id", Operator.EQ, rid)]))
            ) == {rid}

        # upsert_batch mints too, and the id remains prefix-filterable.
        upserted = await db.upsert_batch([Record({"payload": "c"})])
        assert len(upserted) == 1
        batch_id = upserted[0]
        assert _ids(
            await db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, batch_id[:8])]))
        ) == {batch_id}

        # upsert(record) with no id mints and must stay filterable.
        single_id = await db.upsert(Record({"payload": "d"}))
        assert _ids(
            await db.search(Query(filters=[Filter("id", Operator.EQ, single_id)]))
        ) == {single_id}
    finally:
        await db.close()


def test_sync_minted_id_is_filterable(elasticsearch_test_index) -> None:
    """Parity guard: the sync backend already stamps the minted id — keep a
    minted-id record findable by its id so the two backends cannot drift.
    """
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        minted = db.create(Record({"payload": "no-id"}))
        assert _ids(
            db.search(Query(filters=[Filter("id", Operator.EQ, minted)]))
        ) == {minted}
    finally:
        db.close()


async def test_async_sort_by_id(elasticsearch_test_index) -> None:
    """Reproduce-first: sorting by ``id`` orders by the storage key.

    The async sort loop built ``data.{field}`` for every sort field, so
    ``SortSpec("id")`` sorted on the (absent) ``data.id`` field and did not
    order by the storage key. Sync ES special-cases ``id``; async now matches.
    """
    db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await db.connect()
    try:
        await _seed_async(db)
        asc = await db.search(Query(sort_specs=[SortSpec("id")]))
        assert [r.id for r in asc] == sorted(_KEYS)
        desc = await db.search(
            Query(sort_specs=[SortSpec("id", SortOrder.DESC)])
        )
        assert [r.id for r in desc] == sorted(_KEYS, reverse=True)
    finally:
        await db.close()


def test_sync_count_between_is_filtered(elasticsearch_test_index) -> None:
    """Reproduce-first: a BETWEEN-only ``count()`` returns the filtered count.

    Before the sync search/count paths were routed through the single shared
    per-filter translator, ``count()`` had its own inline loop with no
    ``BETWEEN`` branch, so the filter was dropped and the query fell back to
    ``match_all`` — returning the total (5) instead of the filtered count (3).
    """
    db = SyncElasticsearchDatabase(elasticsearch_test_index)
    db.connect()
    try:
        for i in range(1, 6):
            db.create(Record({"n": i}, id=f"r{i}"))
        assert db.count() == 5
        assert db.count(Query(filters=[Filter("n", Operator.BETWEEN, [2, 4])])) == 3
    finally:
        db.close()


# --- full-operator sync/async parity matrix --------------------------------
#
# The sync and async backends and the vector pre-filter now share one
# filter->DSL translator. These pin, against a live Elasticsearch, that the two
# query backends return the *same* rows for the same query across the full
# operator set — the class of drift the shared translator exists to close
# (async previously dropped REGEX / EXISTS / NOT_EXISTS / NOT_LIKE and used a
# substring LIKE).

_MATRIX_RECORDS = [
    Record({"tag": "alpha", "n": 1}, id="orders/1"),
    Record({"tag": "beta", "n": 2}, id="orders/2"),
    Record({"tag": "gamma", "n": 3}, id="artifacts/a"),
    Record({"tag": "Alpha", "n": 4}, id="artifacts/b"),
]

# One Query per row, exercised on a data field and on the id storage key.
_MATRIX_QUERIES = [
    Query(filters=[Filter("tag", Operator.EQ, "alpha")]),
    Query(filters=[Filter("tag", Operator.NEQ, "alpha")]),
    Query(filters=[Filter("n", Operator.GT, 2)]),
    Query(filters=[Filter("n", Operator.GTE, 2)]),
    Query(filters=[Filter("n", Operator.LT, 3)]),
    Query(filters=[Filter("n", Operator.LTE, 3)]),
    Query(filters=[Filter("tag", Operator.LIKE, "alph%")]),
    Query(filters=[Filter("tag", Operator.NOT_LIKE, "alph%")]),
    Query(filters=[Filter("tag", Operator.IN, ["alpha", "beta"])]),
    Query(filters=[Filter("tag", Operator.NOT_IN, ["alpha", "beta"])]),
    Query(filters=[Filter("tag", Operator.EXISTS)]),
    Query(filters=[Filter("missing", Operator.NOT_EXISTS)]),
    Query(filters=[Filter("tag", Operator.REGEX, "al.*")]),
    Query(filters=[Filter("tag", Operator.STARTS_WITH, "al")]),
    Query(filters=[Filter("n", Operator.BETWEEN, [2, 3])]),
    Query(filters=[Filter("n", Operator.NOT_BETWEEN, [2, 3])]),
    Query(filters=[Filter("id", Operator.EQ, "orders/1")]),
    Query(filters=[Filter("id", Operator.NEQ, "orders/1")]),
    Query(filters=[Filter("id", Operator.IN, ["orders/1", "orders/2"])]),
    Query(filters=[Filter("id", Operator.NOT_IN, ["orders/1", "orders/2"])]),
    Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/")]),
    Query(filters=[Filter("id", Operator.LIKE, "orders/%")]),
    Query(filters=[Filter("id", Operator.REGEX, "orders/.*")]),
]


def _seed_matrix_sync(db: SyncElasticsearchDatabase) -> None:
    for record in _MATRIX_RECORDS:
        db.create(record)


@pytest.mark.parametrize("query", _MATRIX_QUERIES, ids=lambda q: q.filters[0].operator.name + ":" + q.filters[0].field)
async def test_sync_async_search_parity(elasticsearch_test_index, query) -> None:
    """Sync and async ES return identical id sets for every operator."""
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_matrix_sync(sync_db)
        assert _ids(await async_db.search(query)) == _ids(sync_db.search(query))
    finally:
        sync_db.close()
        await async_db.close()


@pytest.mark.parametrize("query", _MATRIX_QUERIES, ids=lambda q: q.filters[0].operator.name + ":" + q.filters[0].field)
async def test_sync_async_count_parity(elasticsearch_test_index, query) -> None:
    """Sync and async ``count`` agree for every operator (the class of the
    BETWEEN-only count-drop bug).
    """
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_matrix_sync(sync_db)
        assert await async_db.count(query) == sync_db.count(query)
    finally:
        sync_db.close()
        await async_db.close()


async def test_like_is_case_insensitive(elasticsearch_test_index) -> None:
    """``LIKE`` matches case-insensitively, like the in-memory and SQL backends.

    ``alpha`` and ``Alpha`` both match ``LIKE "alpha"`` — the ES ``wildcard``
    now carries ``case_insensitive: true``.
    """
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_matrix_sync(sync_db)
        q = Query(filters=[Filter("tag", Operator.LIKE, "alpha")])
        assert _ids(sync_db.search(q)) == {"orders/1", "artifacts/b"}
        assert _ids(await async_db.search(q)) == {"orders/1", "artifacts/b"}
    finally:
        sync_db.close()
        await async_db.close()


async def test_regex_matches_full_value_across_tokens(elasticsearch_test_index) -> None:
    """``REGEX`` matches the full field value, not a single analyzed token.

    ``regexp`` on the analyzed ``data.<field>`` path runs per-indexed-term, so a
    cross-token pattern (``alice.*smith`` against ``"alice smith"``) matched
    nothing — the analyzer split the value into ``alice`` / ``smith`` and the
    pattern matched neither. Routing ``REGEX`` to the ``.keyword`` sub-field
    matches the whole value, consistent with the in-memory (``re.search``) and
    SQL backends. Both query backends agree.
    """
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        sync_db.create(Record({"name": "alice smith"}, id="p/1"))
        sync_db.create(Record({"name": "alice jones"}, id="p/2"))
        q = Query(filters=[Filter("name", Operator.REGEX, "alice.*smith")])
        assert _ids(sync_db.search(q)) == {"p/1"}
        assert _ids(await async_db.search(q)) == {"p/1"}
    finally:
        sync_db.close()
        await async_db.close()


async def test_like_escapes_literal_wildcard_metacharacter(elasticsearch_test_index) -> None:
    """A literal ``*`` in a SQL ``LIKE`` pattern matches verbatim, not as a
    wildcard.

    Only ``%``/``_`` are SQL wildcards; an unescaped ``*`` reaching the ES
    ``wildcard`` query would match anything. With escaping, ``LIKE "a*b"`` finds
    only the value ``"a*b"`` — not ``"axb"``. Both query backends agree.
    """
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        sync_db.create(Record({"code": "a*b"}, id="c/star"))
        sync_db.create(Record({"code": "axb"}, id="c/other"))
        q = Query(filters=[Filter("code", Operator.LIKE, "a*b")])
        assert _ids(sync_db.search(q)) == {"c/star"}
        assert _ids(await async_db.search(q)) == {"c/star"}
    finally:
        sync_db.close()
        await async_db.close()


async def test_async_complex_query_matches_sync(elasticsearch_test_index) -> None:
    """The async backend honors ComplexQuery (AND/OR/NOT) at parity with sync."""
    sync_db = SyncElasticsearchDatabase(elasticsearch_test_index)
    sync_db.connect()
    async_db = AsyncElasticsearchDatabase(elasticsearch_test_index)
    await async_db.connect()
    try:
        _seed_matrix_sync(sync_db)
        # id starts with "orders/" AND n < 2  ->  {orders/1}
        cq = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.AND,
                conditions=[
                    FilterCondition(Filter("id", Operator.STARTS_WITH, "orders/")),
                    FilterCondition(Filter("n", Operator.LT, 2)),
                ],
            )
        )
        assert _ids(await async_db.search(cq)) == {"orders/1"}
        assert _ids(await async_db.search(cq)) == _ids(sync_db.search(cq))
    finally:
        sync_db.close()
        await async_db.close()
