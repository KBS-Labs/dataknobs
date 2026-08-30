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
from dataknobs_data.vector.content import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    compute_content_hash,
)
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

        The digest here is the *correct* one for the old defaults, and the
        assertion is that nothing is outdated. An earlier version of this cell
        seeded a deliberately wrong digest and asserted the record *was*
        outdated, which any fallback producing any text at all satisfies --- a
        wrong field list, a wrong separator, an empty string. It could not
        distinguish the claim from a near-miss, which is the one job it had.
        """
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1 Content 1"),
            name="embedding",
            metadata={CONTENT_HASH_KEY: compute_content_hash("Doc 1 Content 1")},
        )
        record_id = await plain_db.create(record)

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])

        assert await tracker.get_outdated_records() == [], (
            "a record digested under the old defaults must survive the upgrade"
        )

        # ...and the comparison is still live, not skipped for want of the new
        # keys: edit the record and it must go stale.
        stored = await plain_db.read(record_id)
        stored.set_value("content", EDITED_CONTENT)
        await plain_db.update(record_id, stored)

        assert [r.id for r in await tracker.get_outdated_records()] == [record_id]


class TestTheWriterAsksAboutItsOwnConfiguration:
    """The half of the contract that is *not* "read it off the record".

    A reader reproduces the assembly the record describes, because it has no
    standing to impose its own. A writer must do the opposite: it maintains
    the field, so its configuration is the authority and the record's account
    of itself is history. Collapsing the two questions into one function reads
    as a simplification and is a defect --- it makes the writer's own
    configuration unchangeable.
    """

    @pytest.mark.asyncio
    async def test_a_changed_separator_takes_effect_on_the_next_sweep(self, plain_db):
        """Re-point the separator and the corpus must be rebuilt under it.

        Fails against a writer that recomputes from the record's stored
        description: every record keeps matching the assembly it was written
        under, so the sweep meant to apply the new separator reports nothing
        to do. Silently, and permanently --- there is no later sweep that
        would notice, and no error anywhere.
        """
        embedder = CountingEmbedder()
        original = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator=" ",
        )
        await plain_db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
        await original.sync_all()
        assert embedder.calls == ["Doc 1 Content 1"]

        rebuilt = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator="\n",
        )

        assert (await rebuilt.sync_all())["updated"] == 1, (
            "a changed field_separator must re-embed the corpus"
        )
        assert "Doc 1\nContent 1" in embedder.calls

    @pytest.mark.asyncio
    async def test_a_changed_text_fields_takes_effect_on_the_next_sweep(self, plain_db):
        """The same statement for the field list rather than the separator."""
        embedder = CountingEmbedder()
        narrow = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )
        await plain_db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
        await narrow.sync_all()

        widened = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
        )

        assert (await widened.sync_all())["updated"] == 1
        assert "Doc 1 Content 1" in embedder.calls


class TestALegacyCorpusHeals:
    """A record digested before the assembly was described must become
    self-describing without being re-embedded.

    Otherwise the upgrade is one-way and the two halves deadlock: a tracker
    falls back to a space and reports the whole corpus outdated, while the
    synchronizer correctly finds every record current and so never rewrites
    one. Each half is right and the corpus stays stuck forever.
    """

    @staticmethod
    async def _legacy_record(db, separator: str) -> str:
        """A record carrying a digest and no account of how it was produced."""
        text = separator.join(["Doc 1", "Content 1"])
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding(text),
            name="embedding",
            metadata={CONTENT_HASH_KEY: compute_content_hash(text)},
        )
        return await db.create(record)

    @pytest.mark.asyncio
    async def test_a_sweep_describes_the_assembly_without_re_embedding(self, plain_db):
        record_id = await self._legacy_record(plain_db, "\n")

        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator="\n",
        )

        assert (await sync.sync_all())["updated"] == 0, "the record is current; do not re-embed it"
        assert embedder.count == 0, "describing an assembly costs no embedding"

        stored = await plain_db.read(record_id)
        metadata = stored.fields["embedding"].metadata
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "content"]
        assert metadata[FIELD_SEPARATOR_KEY] == "\n"

    @pytest.mark.asyncio
    async def test_the_tracker_agrees_once_the_sweep_has_described_it(self, plain_db):
        """The deadlock, stated end to end.

        Fails against a tree where the synchronizer never writes the
        description: the tracker falls back to a space, disagrees with the
        digest, and reports a freshly-swept unedited record outdated.
        """
        await self._legacy_record(plain_db, "\n")

        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=CountingEmbedder(),
            text_fields=["title", "content"],
            field_separator="\n",
        )
        await sync.sync_all()

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])
        assert await tracker.get_outdated_records() == []

    @pytest.mark.asyncio
    async def test_describing_an_assembly_does_not_make_a_stale_record_look_fresh(self, plain_db):
        """The companion. A description is only written for a record the
        synchronizer has already judged current under its own configuration,
        so an edited legacy record must still be re-embedded rather than
        relabelled.
        """
        record_id = await self._legacy_record(plain_db, "\n")
        stored = await plain_db.read(record_id)
        stored.set_value("content", EDITED_CONTENT)
        await plain_db.update(record_id, stored)

        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["title", "content"],
            field_separator="\n",
        )

        assert (await sync.sync_all())["updated"] == 1
        assert f"Doc 1\n{EDITED_CONTENT}" in embedder.calls


class TestSyncOnUpdateDoesNotDestroyTheRecord:
    """`new_data` is what changed, not necessarily the whole record."""

    @pytest.mark.asyncio
    async def test_fields_the_caller_did_not_mention_survive(self, plain_db):
        """Fails by dropping `title` and `author` entirely.

        `sync_record` persists the record it is handed, whole. Handing it one
        built out of `new_data` alone replaces the stored record with the
        changed fields plus the vector --- and an `(old_data, new_data)`
        signature is an invitation to pass exactly that.

        This path only became reachable when `text_fields=` started
        registering into `_source_fields`; before that it returned early, which
        is why the loss had never been seen.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )
        record_id = await plain_db.create(
            Record(data={"title": "Keep me", "content": ORIGINAL_CONTENT, "author": "Ada"})
        )
        await sync.sync_record(await plain_db.read(record_id))

        did_sync = await sync.sync_on_update(
            record_id,
            {"content": ORIGINAL_CONTENT},
            {"content": EDITED_CONTENT},
        )

        assert did_sync is True
        stored = await plain_db.read(record_id)
        assert stored.get_value("content") == EDITED_CONTENT
        assert stored.get_value("title") == "Keep me", "an unmentioned field was destroyed"
        assert stored.get_value("author") == "Ada", "an unmentioned field was destroyed"

    @pytest.mark.asyncio
    async def test_only_the_vector_fields_the_change_feeds_are_re_embedded(self, plain_db):
        """`force=True` covers the fields whose sources changed, not all of them.

        `fields_to_update` was computed, used as an early-out and then
        discarded, so an edit to one source re-embedded every registered
        vector field --- including ones the edit cannot have affected.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )
        # A second vector field, fed by a source the edit does not touch.
        sync._vector_fields["summary_embedding"] = {
            "source_fields": ["summary"],
            "field_separator": " ",
        }
        sync._source_fields["summary"].append("summary_embedding")

        record_id = await plain_db.create(
            Record(data={"content": ORIGINAL_CONTENT, "summary": "A summary"})
        )
        await sync.sync_record(await plain_db.read(record_id))
        embedder.calls.clear()

        await sync.sync_on_update(
            record_id,
            {"content": ORIGINAL_CONTENT, "summary": "A summary"},
            {"content": EDITED_CONTENT, "summary": "A summary"},
        )

        assert embedder.calls == [EDITED_CONTENT], (
            f"re-embedded a field the change cannot affect: {embedder.calls}"
        )


class TestAWriteThatDidNotLandIsNotSuccess:
    """The `None` id was one case of a larger one: the write did not happen.

    `AsyncDatabase.update` reports whether it found anything to write, and
    every call site in this package discarded that. Guarding only the `None`
    id fixes the symptom that was measured and leaves the one beside it.
    """

    @pytest.mark.asyncio
    async def test_sync_record_reports_failure_when_no_record_is_stored_under_the_id(
        self, plain_db
    ):
        """`Record` falls back to an `id` data field, so a record can carry an
        id without ever having been written. `update` then returns `False` and
        the caller used to be told its vector was persisted.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        never_stored = Record(data={"id": "no-such-record", "content": ORIGINAL_CONTENT})
        assert never_stored.id == "no-such-record"

        success, updated = await sync.sync_record(never_stored)

        assert updated == ["embedding"]
        assert success is False, "reported a write that the database refused"
        assert await plain_db.search(Query()) == []

    @pytest.mark.asyncio
    async def test_sync_on_create_reports_failure_when_the_write_does_not_land(self, plain_db):
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=plain_db,
            embedding_fn=embedder,
            text_fields=["content"],
        )

        never_stored = Record(data={"id": "no-such-record", "content": ORIGINAL_CONTENT})

        assert await sync.sync_on_create(never_stored) is False
        assert await plain_db.search(Query()) == []


class TestACorruptDescriptionFallsBack:
    """The assembly description crosses a persistence trust boundary.

    It comes back from whatever store wrote it, and is not guaranteed to be
    the shape that was written.
    """

    @pytest.mark.asyncio
    async def test_a_string_where_the_field_list_belongs_does_not_silence_the_check(self, plain_db):
        """Fails by reporting an edited record current.

        A bare string iterates as characters, so every lookup misses, the
        assembled text is empty, and the digest comes back `None` --- which
        the tracker reads as "nothing to compare" and skips. A corrupt
        description silently switched staleness detection off.
        """
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1 Content 1"),
            name="embedding",
            metadata={
                CONTENT_HASH_KEY: compute_content_hash("Doc 1 Content 1"),
                SOURCE_FIELDS_KEY: "content",  # a string, not a list of names
            },
        )
        record_id = await plain_db.create(record)

        stored = await plain_db.read(record_id)
        stored.set_value("content", EDITED_CONTENT)
        await plain_db.update(record_id, stored)

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])

        assert [r.id for r in await tracker.get_outdated_records()] == [record_id]

    @pytest.mark.asyncio
    async def test_a_non_string_separator_does_not_raise(self, plain_db):
        """`separator.join(...)` on a non-string raises `TypeError`, which
        `sync_all` does not catch --- one corrupt record would abort the sweep
        for every record behind it.
        """
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1 Content 1"),
            name="embedding",
            metadata={
                CONTENT_HASH_KEY: compute_content_hash("Doc 1 Content 1"),
                SOURCE_FIELDS_KEY: ["title", "content"],
                FIELD_SEPARATOR_KEY: 0,  # not a separator
            },
        )
        await plain_db.create(record)

        tracker = ChangeTracker(database=plain_db, tracked_fields=["title", "content"])

        assert await tracker.get_outdated_records() == []


class TestTheOverrideReplacesTheSchemaRatherThanAddingToIt:
    """`text_fields=` overrides what the schema said about the same vector
    field, so a source the schema named and `text_fields` does not no longer
    feeds it.
    """

    @pytest.mark.asyncio
    async def test_a_replaced_schema_source_no_longer_triggers_a_re_embed(self, schema_db):
        """Fails by re-embedding on an edit to a field the vector is not
        derived from --- and the re-embed produces byte-identical text, so
        nothing downstream can notice the work was pointless.
        """
        embedder = CountingEmbedder()
        sync = VectorTextSynchronizer(
            database=schema_db,
            embedding_fn=embedder,
            text_fields=["title"],
        )

        assert sync._source_fields.get("content", []) == [], (
            "the schema's source survived an override that replaced it"
        )

        record_id = await schema_db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
        await sync.sync_record(await schema_db.read(record_id))
        before = embedder.count

        did_sync = await sync.sync_on_update(
            record_id,
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 1", "content": EDITED_CONTENT},
        )

        assert did_sync is False
        assert embedder.count == before
