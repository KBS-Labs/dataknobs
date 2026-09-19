# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``score_threshold`` and ``include_source`` do what the declaration says.

Both are declared on ``Sync/AsyncVectorOperationsMixin.vector_search`` and
neither was implemented anywhere. Twelve backends answered them three ways:
two honoured them, eight swallowed them into ``**kwargs``, two raised
``TypeError``. The swallow is the one worth reproducing, because it is the
only answer that is silent --- measured on ``AsyncMemoryDatabase`` before the
fix, three records scoring 1.0, 0.994 and 0.0 all came back from the call
that asked for ``score_threshold=0.99``, in the same order as the call that
asked for nothing.

``include_source`` had no working implementation at all.
``VectorSearchResult.record`` is declared required, so a hit without a record
cannot be constructed --- which is why both Elasticsearch twins read
``record = None; if include_source: record = ...; if record is None:
continue`` and returned an **empty list** for ``include_source=False``. The
comment above that guard said *"shouldn't happen if include_source is True"*,
so the branch was written knowing it only worked one way.

The structural half of this --- that there is exactly one implementation, and
no backend can answer differently again --- is
``TestOneSearchOverTwelveHooks`` in ``test_vector_mixin_lane_parity.py``,
beside the rest of the family sweep it extends.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase

# Cosine against ``[1, 0]``: 1.0, 0.994 and 0.0. Chosen so that one threshold
# drops the orthogonal record only and a tighter one keeps the exact match
# alone --- a single threshold that dropped two at once could not tell a
# working post-filter from a truncation.
FIXTURE = (
    ("same", [1.0, 0.0]),
    ("near", [0.99, 0.11]),
    ("orth", [0.0, 1.0]),
)
QUERY = np.array([1.0, 0.0])


@pytest.fixture(params=["memory", "file", "sqlite"])
def sync_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SyncDatabase]:
    """Every sync backend whose vector search runs in process.

    Parametrised rather than pinned to memory: the eight that swallowed
    ``score_threshold`` did so by forwarding ``**kwargs`` to one shared
    helper, so a single-backend cell would prove the helper and leave the
    other seven signatures unread. ``SyncS3Database``,
    ``SyncPostgresDatabase`` and ``SyncElasticsearchDatabase`` need a
    service; the signature sweep in the parity guard is what covers them.
    """
    backend = request.param
    if backend == "memory":
        database: SyncDatabase = SyncMemoryDatabase(config={"vector_enabled": True})
    elif backend == "file":
        database = SyncFileDatabase(
            config={"path": str(tmp_path / "records"), "vector_enabled": True}
        )
    else:
        database = SyncSQLiteDatabase(
            config={"path": str(tmp_path / "records.db"), "vector_enabled": True}
        )
    database.connect()
    for record_id, vector in FIXTURE:
        database.create(Record(data={"id": record_id, "embedding": vector}))
    try:
        yield database
    finally:
        database.close()


@pytest.fixture(params=["memory", "file", "sqlite"])
async def async_db(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncDatabase:
    """The async half of the same three."""
    backend = request.param
    if backend == "memory":
        database: AsyncDatabase = AsyncMemoryDatabase(config={"vector_enabled": True})
    elif backend == "file":
        database = AsyncFileDatabase(
            config={"path": str(tmp_path / "records"), "vector_enabled": True}
        )
    else:
        database = AsyncSQLiteDatabase(
            config={"path": str(tmp_path / "records.db"), "vector_enabled": True}
        )
    await database.connect()
    for record_id, vector in FIXTURE:
        await database.create(Record(data={"id": record_id, "embedding": vector}))
    return database


def _ids(results: list) -> list[str]:
    return [result.record.id for result in results]


class TestScoreThresholdIsHonoured:
    """Red on ten of the twelve: eight swallowed it, two refused it."""

    def test_sync_no_threshold_returns_everything(self, sync_db: SyncDatabase) -> None:
        """The control. Without it, a threshold test passes on a broken search
        that returns nothing at all.
        """
        assert sorted(_ids(sync_db.vector_search(QUERY, vector_field="embedding", k=10))) == [
            "near",
            "orth",
            "same",
        ]

    def test_sync_threshold_drops_the_hit_below_it(self, sync_db: SyncDatabase) -> None:
        results = sync_db.vector_search(QUERY, vector_field="embedding", k=10, score_threshold=0.99)

        assert sorted(_ids(results)) == ["near", "same"], "orth scores 0.0 and asked for 0.99"

    def test_sync_a_tighter_threshold_drops_more(self, sync_db: SyncDatabase) -> None:
        """Two thresholds, two answers --- so the cell above cannot be passed
        by a search that happens to return two results for another reason.
        """
        results = sync_db.vector_search(
            QUERY, vector_field="embedding", k=10, score_threshold=0.999
        )

        assert _ids(results) == ["same"]

    def test_sync_a_threshold_may_return_fewer_than_k(self, sync_db: SyncDatabase) -> None:
        """The defined behaviour, stated rather than discovered.

        The threshold is a post-filter over what the backend's k-NN returned,
        which is what the Elasticsearch implementation always did. Refilling
        ``k`` by over-fetching is a different promise and is not made here.
        """
        k = 3
        results = sync_db.vector_search(QUERY, vector_field="embedding", k=k, score_threshold=0.999)

        assert len(results) == 1
        assert len(results) < k

    @pytest.mark.asyncio
    async def test_async_threshold_drops_the_hit_below_it(self, async_db: AsyncDatabase) -> None:
        results = await async_db.vector_search(
            QUERY, vector_field="embedding", k=10, score_threshold=0.99
        )

        assert sorted(_ids(results)) == ["near", "same"]

    @pytest.mark.asyncio
    async def test_async_a_tighter_threshold_drops_more(self, async_db: AsyncDatabase) -> None:
        results = await async_db.vector_search(
            QUERY, vector_field="embedding", k=10, score_threshold=0.999
        )

        assert _ids(results) == ["same"]


class TestAnUnrecognisedKeywordRaises:
    """``**kwargs`` was the mechanism, and this is what removing it buys.

    Red on the eight that carried it: the keyword bound at the signature,
    forwarded into a helper whose own ``**kwargs`` is read nowhere, and
    vanished. The two Postgres twins already raised here --- loudly, and
    therefore correctly.
    """

    def test_sync(self, sync_db: SyncDatabase) -> None:
        with pytest.raises(TypeError, match="bogus"):
            sync_db.vector_search(QUERY, vector_field="embedding", bogus=1)

    @pytest.mark.asyncio
    async def test_async(self, async_db: AsyncDatabase) -> None:
        with pytest.raises(TypeError, match="bogus"):
            await async_db.vector_search(QUERY, vector_field="embedding", bogus=1)


class TestIncludeSourceDerivesTheText:
    """What ``include_source`` always meant.

    Not records-versus-ids --- the record is always returned. The original
    design spells it *"automatic source retrieval"*: populate
    ``VectorSearchResult.source_text`` with the text the vector was made
    from, read off the record already in hand. No query and no id round-trip:
    ``content_hash_metadata`` stored the ordered field list and the separator
    on the vector field for exactly this, and ``assemble_source_text``
    reproduces the text from them.
    """

    @pytest.fixture
    def embedded(self, tmp_path: Path) -> Iterator[SyncDatabase]:
        database = SyncMemoryDatabase(config={"vector_enabled": True})
        database.connect()
        database.bulk_embed_and_store(
            [Record(data={"id": "multi", "title": "Widget", "body": "A widget for widgeting."})],
            text_field=["title", "body"],
            embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
        )
        # Written without going through bulk_embed_and_store, so its vector
        # carries no assembly description --- the shape of every vector
        # written before ``content_hash_metadata`` existed.
        database.create(
            Record(data={"id": "bare", "title": "Undescribed", "embedding": [0.99, 0.11]})
        )
        try:
            yield database
        finally:
            database.close()

    def _hit(self, database: SyncDatabase, record_id: str, **kwargs: object):
        results = database.vector_search(QUERY, vector_field="embedding", k=10, **kwargs)
        return next(r for r in results if r.record.id == record_id)

    def test_a_multi_field_embed_round_trips_to_the_text_it_was_made_from(
        self, embedded: SyncDatabase
    ) -> None:
        assert self._hit(embedded, "multi").source_text == "Widget A widget for widgeting."

    def test_a_single_field_embed_derives_from_the_legacy_scalar_key(self, tmp_path: Path) -> None:
        """The middle of the three answers, which the multi-field case hides.

        ``attach_vector_field`` writes the ordered field list *and* a legacy
        scalar ``source_field`` --- the one field name when one was embedded,
        the names comma-joined when several were. The derivation prefers the
        list, so the scalar is only reached by a record written before the
        list existed; this asserts that when it is reached it resolves,
        rather than that the branch is merely present.
        """
        from dataknobs_data.fields import VectorField
        from dataknobs_data.vector.content import derive_source_text

        record = Record(
            data={"id": "legacy", "body": "just one field"},
        )
        # No `metadata`, so `stored_assembly` reports nothing and the scalar
        # is what remains --- the shape of a pre-description vector.
        record.fields["embedding"] = VectorField(
            name="embedding", value=[1.0, 0.0], source_field="body"
        )

        assert derive_source_text(record, "embedding") == "just one field"

    def test_a_comma_joined_scalar_is_not_read_as_a_field_name(self) -> None:
        """...and the same branch correctly misses for a multi-field vector.

        ``"title,body"`` is not a field, so the lookup fails and the answer is
        ``None`` rather than a ``KeyError`` or an invented value. Graceful,
        not fixed: a multi-field vector from before descriptions existed
        cannot be reassembled, and saying so is the honest answer.
        """
        from dataknobs_data.fields import VectorField
        from dataknobs_data.vector.content import derive_source_text

        record = Record(data={"id": "legacy", "title": "Widget", "body": "A widget."})
        record.fields["embedding"] = VectorField(
            name="embedding", value=[1.0, 0.0], source_field="title,body"
        )

        assert derive_source_text(record, "embedding") is None

    def test_a_vector_with_no_description_yields_none(self, embedded: SyncDatabase) -> None:
        """Graceful rather than fixed: the field list is what the derivation
        reads, and a vector written before descriptions existed has none.
        """
        assert self._hit(embedded, "bare").source_text is None

    def test_include_source_false_still_returns_the_hit(self, embedded: SyncDatabase) -> None:
        """The defect on both Elasticsearch twins, in the one place it can be
        observed without a server.

        There, ``include_source=False`` told Elasticsearch not to return
        ``_source``, ``_doc_to_record`` therefore had nothing, and the
        ``if record is None: continue`` that followed returned an empty list.
        The record is not the knob's subject; the assembly is.
        """
        hit = self._hit(embedded, "multi", include_source=False)

        assert hit.record.get_value("title") == "Widget"
        assert hit.source_text is None

    def test_the_knob_is_the_only_difference(self, embedded: SyncDatabase) -> None:
        """Same hits either way --- which is what makes the empty list a bug
        rather than a documented meaning of ``False``.
        """
        with_source = embedded.vector_search(QUERY, vector_field="embedding", k=10)
        without = embedded.vector_search(
            QUERY, vector_field="embedding", k=10, include_source=False
        )

        assert _ids(with_source) == _ids(without)
