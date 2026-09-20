# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""One place a Postgres vector lives, written by every path and read by both twins.

The two ``SyncPostgresDatabase`` / ``AsyncPostgresDatabase`` vector searches
read different storage on the same table. The async twin read a dedicated
``vector_<field>`` pgvector column; the sync twin extracted the vector back
out of the JSON ``data`` column. Each twin was internally consistent, which
is why the divergence survived --- it is only visible when one writes and the
other reads.

The column was the narrower of the two. Measured before the fix, it was
written by **three** of the fourteen write paths across the two classes:
``create``, ``update`` and ``upsert`` on the async twin, all three via
``_collect_vector_inserts``. Every batch path on either twin, and every path
at all on the sync twin, wrote only ``data`` --- so the async twin's own
``create_batch`` produced records its own ``vector_search`` could not see,
and a sync-written corpus was invisible to an async search with no column at
all (``UndefinedColumnError``) or, once any async single-write had created
one, as an empty result list with no error.

``data`` is written by all fourteen. So unifying there loses nothing and
needs no backfill, where unifying on the column would have orphaned every
row ever written by a batch.

The third consequence this pins is the index. ``create_vector_index`` existed
only on the async twin and built its index over the JSON expression --- which
was not the expression that twin's own query used, so the one implementation
of index creation produced an index nothing could use. An expression index
serves only a query whose expression matches exactly, so the guard is a plan,
not a string comparison.
"""

from __future__ import annotations

import numpy as np
import psycopg2
import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.backends.postgres_vector import build_vector_search_sql
from dataknobs_data.fields import VectorField
from dataknobs_data.vector.types import DistanceMetric

pytestmark = requires_postgres

QUERY = np.array([1.0, 0.0], dtype=np.float32)


def _record(name: str, vector: list[float]) -> Record:
    return Record(
        {"name": name, "embedding": VectorField(value=np.array(vector, dtype=np.float32))}
    )


def _identified(name: str, vector: list[float]) -> Record:
    """A record whose id is its name, so each write path owns its own row."""
    record = _record(name, vector)
    record.id = name
    return record


def _blank(name: str) -> Record:
    """A row for an update path to write a vector *into*, so the write under
    test is the one that puts the vector there rather than a create before it.
    """
    record = Record({"name": name})
    record.id = name
    return record


@pytest.fixture
def vector_db_config(postgres_test_db):
    """The per-test table, with vector support asked for explicitly.

    ``vector_enabled`` defaults to False and both ``connect()`` methods gate
    ``_detect_vector_support`` behind it, so a config without it produces a
    database that refuses every vector call.
    """
    return {**postgres_test_db, "vector_enabled": True}


@pytest.fixture
async def twins(vector_db_config):
    """Both twins on one table, connected, closed together."""
    sync_db = SyncPostgresDatabase(vector_db_config)
    sync_db.connect()
    async_db = AsyncPostgresDatabase(vector_db_config)
    await async_db.connect()
    if not sync_db.has_vector_support():
        pytest.skip("pgvector extension unavailable on the test server")
    try:
        yield sync_db, async_db
    finally:
        await async_db.close()
        sync_db.close()


def _names(hits) -> list[str]:
    return sorted(hit.record.get_value("name") for hit in hits)


def _connect(config):
    return psycopg2.connect(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"],
    )


def _index_names(config) -> list[str]:
    with _connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = %s", (config["table"],))
        return [row[0] for row in cur.fetchall()]


def _explain(config, sql: str) -> str:
    """The plan for ``sql``, with sequential scans discouraged.

    ``enable_seqscan = off`` does not force an index; it prices sequential
    scans out of contention. An index the planner cannot use for this
    expression still does not appear, which is the distinction the test
    needs --- a twenty-row table would otherwise always be scanned.
    """
    with _connect(config) as conn, conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN " + sql)
        return "\n".join(row[0] for row in cur.fetchall())


class TestEveryWritePathIsVisibleToBothTwins:
    """The property the divergence broke, over every door that writes a vector."""

    @pytest.mark.asyncio
    async def test_a_sync_written_corpus_is_visible_to_the_async_search(self, twins):
        sync_db, async_db = twins
        sync_db.create(_record("sync_create", [1.0, 0.0]))

        hits = await async_db.vector_search(QUERY, vector_field="embedding", k=10)

        assert _names(hits) == ["sync_create"], (
            "the async twin searched storage the sync twin never wrote"
        )

    @pytest.mark.asyncio
    async def test_an_async_written_corpus_is_visible_to_the_sync_search(self, twins):
        sync_db, async_db = twins
        await async_db.create(_record("async_create", [1.0, 0.0]))

        hits = sync_db.vector_search(QUERY, vector_field="embedding", k=10)

        assert _names(hits) == ["async_create"]

    @pytest.mark.asyncio
    async def test_a_batch_written_corpus_is_visible_to_the_twin_that_wrote_it(self, twins):
        """The async twin could not see its own ``create_batch`` output."""
        _, async_db = twins
        await async_db.create_batch(
            [_record("batch_a", [1.0, 0.0]), _record("batch_b", [0.99, 0.11])]
        )

        hits = await async_db.vector_search(QUERY, vector_field="embedding", k=10)

        assert _names(hits) == ["batch_a", "batch_b"]

    @pytest.mark.asyncio
    async def test_every_write_path_lands_in_one_place(self, twins):
        """The sweep: every door, one corpus, both readers agree.

        Fourteen record-writing entry points across the two classes ---
        ``create``, ``update``, ``upsert``, ``create_batch``,
        ``upsert_batch``, ``update_batch`` and ``stream_write``, twice.
        Measured before the fix, three filled the ``vector_<field>`` column.

        Enumerated rather than parametrised because the point is the *union*:
        a fix that made any single path visible while leaving another behind
        is precisely the state this replaces.
        """
        sync_db, async_db = twins

        async def async_stream():
            yield _identified("async.stream_write", [1.0, 0.0])

        sync_db.create(_identified("sync.create", [1.0, 0.0]))
        sync_db.create(_blank("sync.update"))
        sync_db.update("sync.update", _identified("sync.update", [1.0, 0.0]))
        sync_db.upsert("sync.upsert", _identified("sync.upsert", [1.0, 0.0]))
        sync_db.create_batch([_identified("sync.create_batch", [1.0, 0.0])])
        sync_db.upsert_batch([_identified("sync.upsert_batch", [1.0, 0.0])])
        sync_db.create(_blank("sync.update_batch"))
        sync_db.update_batch([("sync.update_batch", _identified("sync.update_batch", [1.0, 0.0]))])
        sync_db.stream_write(iter([_identified("sync.stream_write", [1.0, 0.0])]))

        await async_db.create(_identified("async.create", [1.0, 0.0]))
        await async_db.create(_blank("async.update"))
        await async_db.update("async.update", _identified("async.update", [1.0, 0.0]))
        await async_db.upsert("async.upsert", _identified("async.upsert", [1.0, 0.0]))
        await async_db.create_batch([_identified("async.create_batch", [1.0, 0.0])])
        await async_db.upsert_batch([_identified("async.upsert_batch", [1.0, 0.0])])
        await async_db.create(_blank("async.update_batch"))
        await async_db.update_batch(
            [("async.update_batch", _identified("async.update_batch", [1.0, 0.0]))]
        )
        await async_db.stream_write(async_stream())

        expected = sorted(
            f"{lane}.{door}"
            for lane in ("sync", "async")
            for door in (
                "create",
                "update",
                "upsert",
                "create_batch",
                "upsert_batch",
                "update_batch",
                "stream_write",
            )
        )
        assert len(expected) == 14
        assert _names(await async_db.vector_search(QUERY, k=50)) == expected
        assert _names(sync_db.vector_search(QUERY, k=50)) == expected

    @pytest.mark.asyncio
    async def test_an_updated_vector_is_what_both_twins_find(self, twins):
        """``update`` rewrote the column on one twin and the JSON on both."""
        sync_db, async_db = twins
        sync_db.upsert("r", _record("r", [0.0, 1.0]))
        await async_db.update("r", _record("r", [1.0, 0.0]))

        for hits in (
            await async_db.vector_search(QUERY, k=10),
            sync_db.vector_search(QUERY, k=10),
        ):
            assert len(hits) == 1
            assert hits[0].score == pytest.approx(1.0, abs=1e-4)


class TestBothTwinsScoreTheSameCorpusAlike:
    @pytest.mark.asyncio
    async def test_the_two_twins_return_the_same_scores(self, twins):
        """Cosine was ``1 - d`` on one twin and ``1 - min(d, 2)/2`` on the other.

        Two numbers for one corpus and one query, which nothing compared
        until ``score_threshold`` arrived to compare them against a constant.
        """
        sync_db, async_db = twins
        for name, vector in (("same", [1.0, 0.0]), ("near", [0.99, 0.11]), ("orth", [0.0, 1.0])):
            sync_db.create(_record(name, vector))

        sync_hits = sync_db.vector_search(QUERY, k=10)
        async_hits = await async_db.vector_search(QUERY, k=10)

        sync_scored = {h.record.get_value("name"): h.score for h in sync_hits}
        async_scored = {h.record.get_value("name"): h.score for h in async_hits}
        assert sync_scored.keys() == async_scored.keys()
        for name, score in sync_scored.items():
            assert score == pytest.approx(async_scored[name], abs=1e-5)
        assert sync_scored["same"] == pytest.approx(1.0, abs=1e-4)
        assert sync_scored["orth"] == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.asyncio
    async def test_score_threshold_agrees_across_the_twins(self, twins):
        sync_db, async_db = twins
        for name, vector in (("same", [1.0, 0.0]), ("orth", [0.0, 1.0])):
            sync_db.create(_record(name, vector))

        assert _names(sync_db.vector_search(QUERY, k=10, score_threshold=0.5)) == ["same"]
        assert _names(await async_db.vector_search(QUERY, k=10, score_threshold=0.5)) == ["same"]


class TestTheIndexServesTheQueryThatAsksForIt:
    """``create_vector_index`` built an index over an expression nothing queried."""

    @pytest.mark.asyncio
    async def test_both_twins_can_create_an_index(self, twins, vector_db_config):
        """Only the async twin had an implementation; the sync twin inherited
        the mixin's ``return True`` no-op, so it reported success and built
        nothing.
        """
        sync_db, async_db = twins
        sync_db.create(_record("r", [1.0, 0.0]))

        assert await async_db.create_vector_index("embedding", dimensions=2) is True
        assert await async_db.drop_vector_index("embedding") is True
        assert sync_db.create_vector_index("embedding", dimensions=2) is True

        assert any("embedding" in name for name in _index_names(vector_db_config)), (
            "the sync twin reported success and created no vector index"
        )

    @pytest.mark.asyncio
    async def test_both_twins_report_the_same_index_stats(self, twins):
        """``get_vector_index_stats`` was the mixin's empty default on the sync twin."""
        sync_db, async_db = twins
        sync_db.create(_record("r", [1.0, 0.0]))
        assert sync_db.create_vector_index("embedding", dimensions=2) is True

        sync_stats = sync_db.get_vector_index_stats("embedding")
        async_stats = await async_db.get_vector_index_stats("embedding")

        assert sync_stats == async_stats
        assert sync_stats["vector_count"] == 1
        assert sync_stats["indexed"] is True

    @pytest.mark.asyncio
    async def test_the_planner_can_use_the_index_for_the_search(self, twins, vector_db_config):
        """The claim is that the index and the query name the same expression.

        A textual comparison cannot show that --- Postgres rewrites an index
        definition on the way into the catalog, so the generated SQL and the
        stored definition never match as strings even when they mean the same
        thing. A plan can: with sequential scans discouraged, a matching index
        is chosen and a non-matching one is unreachable.

        The SQL here is the backend's own, from the builder both twins call,
        rather than a re-spelling of it --- an index that matched a
        reconstruction and not the real query would pass a weaker test.
        """
        sync_db, async_db = twins
        for i in range(20):
            sync_db.create(_record(f"r{i}", [1.0, i / 100.0]))
        assert (
            await async_db.create_vector_index("embedding", dimensions=2, index_type="hnsw") is True
        )

        sql = build_vector_search_sql(
            q_qualified=f'"{sync_db.schema_name}"."{sync_db.table_name}"',
            vector_field="embedding",
            dimensions=2,
            metric=DistanceMetric.COSINE,
            vector_placeholder="'[1,0]'",
            field_placeholder="'embedding'",
            limit_clause="LIMIT 5",
        )

        plan = _explain(vector_db_config, sql)

        assert "Index Scan" in plan, f"the index is unreachable for this query:\n{plan}"
        assert "embedding" in plan


@pytest.mark.asyncio
class TestARowThatCannotHoldAVectorIsNotSearched:
    """``data ? 'embedding'`` is a key-presence test, and a key can hold null.

    The async twin's predicate used to be ``WHERE vector_<field> IS NOT
    NULL``, which a column type makes true or false with nothing in between.
    Moving both twins onto the JSON ``data`` column replaced that with a test
    the JSON document passes while holding ``null`` --- and
    ``record_to_json`` writes exactly ``{"embedding": null}`` for a record
    whose vector field carries ``None``.

    Such a row produces ``distance = NULL``. ``ORDER BY distance`` sorts it
    last, so it is invisible until the corpus holds fewer than ``k`` real
    vectors; then ``float(None)`` raises ``TypeError`` and the entire search
    fails rather than the one row. A non-vector string under the key is worse
    --- the cast raises inside Postgres and nothing comes back at all.
    """

    async def test_a_null_vector_neither_ranks_nor_breaks_the_search(self, twins):
        sync_db, async_db = twins
        sync_db.create(_record("real", [1.0, 0.0]))
        sync_db.create(Record({"name": "null-vector", "embedding": None}, id="null-vector"))

        # k above the corpus size, so a surviving null row cannot hide behind
        # the ORDER BY that sorts it last.
        assert _names(sync_db.vector_search(QUERY, k=10)) == ["real"]
        assert _names(await async_db.vector_search(QUERY, k=10)) == ["real"]

    async def test_a_non_vector_value_does_not_fail_the_whole_query(self, twins):
        """The cast is what raises, so one bad row used to cost every row."""
        sync_db, async_db = twins
        sync_db.create(_record("real", [1.0, 0.0]))
        sync_db.create(Record({"name": "text", "embedding": "not a vector"}, id="text"))

        assert _names(sync_db.vector_search(QUERY, k=10)) == ["real"]
        assert _names(await async_db.vector_search(QUERY, k=10)) == ["real"]


@pytest.mark.asyncio
class TestTheNativeHybridSearchHonoursItsArguments:
    """``filter`` was declared, bound into nothing, and never consulted.

    The native path built its parameter list as ``[text, vector, field]`` and
    stopped, so a filtered hybrid search read the whole table and reported no
    error. The fallback path one branch above it *did* apply the filter, so
    the same call answered differently depending on a fusion strategy the
    caller may not have set.
    """

    async def _corpus(self, sync_db):
        for i in range(6):
            record = Record(
                {
                    "name": f"doc{i}",
                    "content": "widget gadget",
                    "tenant": "a" if i < 3 else "b",
                    "embedding": VectorField(value=np.array([1.0, i / 100.0], dtype=np.float32)),
                }
            )
            record.id = f"doc{i}"
            sync_db.create(record)

    async def test_a_filter_reaches_the_native_path(self, twins):
        from dataknobs_data.query import Filter, Operator, Query

        sync_db, async_db = twins
        await self._corpus(sync_db)

        results = await async_db.hybrid_search(
            query_text="widget",
            query_vector=QUERY,
            text_fields=["content"],
            vector_field="embedding",
            k=10,
            filter=Query(filters=[Filter(field="tenant", operator=Operator.EQ, value="a")]),
        )

        assert sorted(r.record.get_value("name") for r in results) == ["doc0", "doc1", "doc2"]

    async def test_both_arms_rank_before_they_truncate(self, twins):
        """Each arm's ``LIMIT`` used to take an arbitrary ``fetch_k``.

        With ``k=1`` the arms fetch three rows each from a corpus of six, so
        an unordered ``LIMIT`` is free to hand the fusion the three *worst*
        vector matches. The nearest neighbour is the one row that cannot be
        missing from a correctly ordered arm.
        """
        sync_db, async_db = twins
        await self._corpus(sync_db)

        results = await async_db.hybrid_search(
            query_text="nothing matches this text",
            query_vector=QUERY,
            text_fields=["content"],
            vector_field="embedding",
            k=1,
        )

        assert [r.record.get_value("name") for r in results] == ["doc0"]
