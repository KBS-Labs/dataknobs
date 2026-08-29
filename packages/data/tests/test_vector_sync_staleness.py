"""The staleness contract between VectorTextSynchronizer and ChangeTracker.

Every test here is a reproduce-first cell. Six of the seven fail against the
tree these were written on; the seventh describes a state that tree cannot
produce at all, because nothing writes the metadata it reads.

The contract they pin, stated once:

* ``VectorTextSynchronizer`` re-embeds a record if and only if the text it
  would feed the embedder differs from the text it fed last time. Not "if a
  field changed", not "always" — the embedder's own input is the staleness
  condition, and the stored digest is a digest of exactly that string.
* ``ChangeTracker`` answers the same question about the same records without
  being the class that wrote them, so it has to *reproduce* that string. It
  reproduces it from what the record carries, never from how the tracker
  happened to be configured. That is what the seventh cell is for, and it is
  the only one that distinguishes the fix from a plausible near-miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType, VectorField
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.testing import text_embedding
from dataknobs_data.vector.sync import VectorTextSynchronizer
from dataknobs_data.vector.tracker import ChangeTracker


class CountingEmbedder:
    """An embedding function that records how often it was asked.

    The fifth cell is a call count and nothing else, so the count has to be
    real rather than inferred from ``sync_all``'s ``updated`` tally — the
    naive union of the two fixes double-embeds a record while reporting it
    updated once.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return text_embedding(text)

    @property
    def count(self) -> int:
        return len(self.calls)


async def _plain_database() -> AsyncMemoryDatabase:
    """A database with no vector field declared in its schema.

    The simplified ``text_fields=`` API is the whole surface under test here,
    and it does not consult the schema.
    """
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.connect()
    return db


async def _schema_database() -> AsyncMemoryDatabase:
    """A database declaring ``embedding`` with ``content`` as its source."""
    schema = DatabaseSchema()
    schema.add_field(FieldSchema(name="content", type=FieldType.TEXT))
    schema.add_field(
        FieldSchema(
            name="embedding",
            type=FieldType.VECTOR,
            metadata={"dimensions": 384, "source_field": "content"},
        )
    )
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    db.schema = schema
    await db.connect()
    return db


@pytest.fixture
async def plain_db():
    db = await _plain_database()
    yield db
    await db.close()


@pytest.fixture
async def schema_db():
    db = await _schema_database()
    yield db
    await db.close()


# Two texts that must embed differently. `text_embedding` seeds from the sum of
# the first ten code points, so a pair differing only after character ten — or
# only in ordering within it — shares a vector and would make an edit
# undetectable for a reason that has nothing to do with the code under test.
ORIGINAL_CONTENT = "Content one"
EDITED_CONTENT = "Wholly different prose"


class TestSourceTextEditIsDetected:
    """Cells 1-4: a source-text edit must re-embed, on both API halves."""

    @pytest.mark.asyncio
    async def test_sync_on_update_re_embeds_after_edit_on_text_fields_api(self, plain_db):
        """Cell 1 — ``text_fields=`` + ``sync_on_update`` after an edit.

        Returns ``False`` today: ``_initialize_field_mappings`` populates
        ``_source_fields`` from the schema only, so on the simplified API the
        loop ``sync_on_update`` walks to decide whether anything changed is
        empty, and it returns before doing any work.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        record_id = await plain_db.create(Record(data={"content": ORIGINAL_CONTENT}))
        await sync.sync_record(await plain_db.read(record_id))

        did_sync = await sync.sync_on_update(
            record_id,
            {"content": ORIGINAL_CONTENT},
            {"content": EDITED_CONTENT},
        )

        assert did_sync is True
        assert EDITED_CONTENT in embedder.calls

    @pytest.mark.asyncio
    async def test_sync_all_skips_unchanged_records_on_text_fields_api(self, plain_db):
        """Cell 2 — ``text_fields=`` + ``sync_all()`` with nothing changed.

        Re-embeds everything today: on the simplified path ``sync_record``
        never consults ``_needs_update`` at all, so a second sweep over an
        untouched corpus costs one embedding per record.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        for i in range(3):
            await plain_db.create(Record(data={"content": f"Document number {i}"}))

        first = await sync.sync_all()
        assert first["updated"] == 3
        after_first_sweep = embedder.count

        second = await sync.sync_all()

        assert second["updated"] == 0, "an unchanged corpus must not be re-embedded"
        assert embedder.count == after_first_sweep

    @pytest.mark.asyncio
    async def test_sync_all_re_embeds_after_edit_on_schema_api(self, schema_db):
        """Cell 3 — schema-declared + ``sync_all()`` after an edit.

        Reports ``updated=0`` today. ``_has_current_vector`` returns ``True``
        for any ``VectorField`` whose model version matches, without reading
        the ``content_hash`` the same class wrote, so the edit is invisible.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=schema_db,
            embedding_fn=embedder,
            model_version="v1",
        )

        record_id = await schema_db.create(Record(data={"content": ORIGINAL_CONTENT}))
        assert (await sync.sync_all())["updated"] == 1

        record = await schema_db.read(record_id)
        record.set_value("content", EDITED_CONTENT)
        await schema_db.update(record_id, record)

        assert (await sync.sync_all())["updated"] == 1, "an edited record must be re-embedded"
        assert EDITED_CONTENT in embedder.calls

    @pytest.mark.asyncio
    async def test_has_current_vector_is_false_after_source_edit(self, schema_db):
        """Cell 4 — the unit-level statement of cell 3.

        ``_has_current_vector`` is where the omission lives, so this is the
        cell that names it. It is ``True`` today for a record whose text no
        longer matches the digest stored beside its vector.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=schema_db,
            embedding_fn=embedder,
            model_version="v1",
        )

        record_id = await schema_db.create(Record(data={"content": ORIGINAL_CONTENT}))
        await sync.sync_all()

        record = await schema_db.read(record_id)
        assert sync._has_current_vector(record, "embedding") is True

        record.set_value("content", EDITED_CONTENT)
        assert sync._has_current_vector(record, "embedding") is False


class TestPartialFixRegression:
    """Cell 5: the guard against the fix that arrives twice."""

    @pytest.mark.asyncio
    async def test_one_fresh_record_costs_exactly_one_embedding(self, plain_db):
        """Cell 5 — one fresh record, one ``sync_all()``, one embedding.

        This cell passes today and passes after either half of the change
        alone. It fails only against their naive union, where the simplified
        path embeds once via ``text_fields`` and again via the field mapping
        that the first half of the fix newly registers.

        It has to exist before the fix rather than after it: a doubling that
        arrives together with a correctness improvement reads as the cost of
        the improvement.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        await plain_db.create(Record(data={"content": ORIGINAL_CONTENT}))
        await sync.sync_all()

        assert embedder.count == 1, f"embedded {embedder.count}x: {embedder.calls}"


class TestUnstoredRecordIsNotReportedAsSynced:
    """A record with no id cannot be written, and must not claim it was.

    Not a staleness cell. This is what the type checker was pointing at once
    ``database:`` stopped being annotated with a class that does not exist:
    ``AsyncDatabase.update`` takes a ``str`` and both call sites passed
    ``str | None``. Passing ``None`` does not raise — the write is dropped and
    the method reports success, so a caller holding an unstored record is told
    its vector was persisted when the database is still empty.
    """

    @pytest.mark.asyncio
    async def test_sync_record_reports_failure_when_there_is_no_id_to_write_under(self, plain_db):
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        unstored = Record(data={"content": ORIGINAL_CONTENT})
        assert unstored.id is None

        success, updated = await sync.sync_record(unstored)

        # The vector really is on the record the caller holds, so it is still
        # reported — what is false is the claim that it was stored.
        assert updated == ["embedding"]
        assert success is False
        assert await plain_db.search(Query()) == []

    @pytest.mark.asyncio
    async def test_sync_on_create_reports_failure_when_there_is_no_id(self, plain_db):
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        assert await sync.sync_on_create(Record(data={"content": ORIGINAL_CONTENT})) is False
        assert await plain_db.search(Query()) == []


class TestTrackerAgreesWithSynchronizer:
    """Cells 6-7: the unwritten contract between the two classes."""

    @pytest.mark.asyncio
    async def test_tracker_reports_nothing_outdated_on_non_default_separator(self, plain_db):
        """Cell 6 — a freshly synced, unedited corpus is not outdated.

        Measured, not predicted. ``ChangeTracker`` hardcodes ``" ".join`` while
        the synchronizer joins on its configured ``field_separator``, so for
        any non-default separator the comparison cannot match on any record,
        ever — a fresh sweep leaves the whole corpus permanently "outdated".

        Both existing callers sit on the default separator, one of them
        explicitly, which is why nobody has seen this.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator="\n",
        )
        await plain_db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
        await sync.sync_all()

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])

        assert await tracker.get_outdated_records() == []

    @pytest.mark.asyncio
    async def test_tracker_reproduces_the_digest_from_the_record(self, plain_db):
        """Cell 7 — the tracker is mis-configured on purpose.

        Cell 6 can be satisfied by handing ``ChangeTracker`` a
        ``field_separator=`` and asking every caller to keep two constructor
        arguments in step. That closes today's disagreement while leaving the
        next one available, so this cell withholds the argument: the tracker is
        built exactly as a caller would build one today, and must still agree.

        It can only pass if the tracker reproduces the embedder's input from
        what the *record* carries rather than from how it was *configured*.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator=" | ",
        )
        await plain_db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
        await sync.sync_all()

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])

        assert await tracker.get_outdated_records() == []

    @pytest.mark.asyncio
    async def test_tracker_still_detects_a_real_edit(self, plain_db):
        """Cell 7's first companion, and the one that gets skipped.

        Cell 7 alone is satisfied by a tracker that returns ``[]``
        unconditionally — a regression wearing the fix's clothes. This is the
        same shape as cell 5 and exists for the same reason.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator=" | ",
        )
        record_id = await plain_db.create(
            Record(data={"title": "Doc 1", "content": ORIGINAL_CONTENT})
        )
        await sync.sync_all()

        record = await plain_db.read(record_id)
        record.set_value("content", EDITED_CONTENT)
        await plain_db.update(record_id, record)

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])
        outdated = await tracker.get_outdated_records()

        assert [r.id for r in outdated] == [record_id]

    @pytest.mark.asyncio
    async def test_record_written_before_the_change_behaves_as_it_does_today(self, plain_db):
        """Cell 7's second companion: the backwards-compatibility claim.

        A record carrying a ``content_hash`` but no assembly description is
        what every record written before this change looks like. The tracker
        must fall back to the defaults it hardcodes today — a space-joined
        digest over its own ``tracked_fields`` — so that upgrading invalidates
        no stored hash and re-embeds nothing.

        Nothing else in the suite would notice if that claim were false.
        """
        legacy_digest = "d41d8cd98f00b204e9800998ecf8427e"
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1 Content 1"),
            name="embedding",
            metadata={"content_hash": legacy_digest},
        )
        record_id = await plain_db.create(record)

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])
        outdated = await tracker.get_outdated_records()

        # The stored digest is deliberately not the digest of "Doc 1 Content 1",
        # so today's space-joined comparison reports it stale. What is being
        # pinned is that the comparison still happens against the old defaults
        # rather than being skipped for want of the new metadata keys.
        assert [r.id for r in outdated] == [record_id]
