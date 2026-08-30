"""Every other class that assembles the embedder's input, or stores its digest.

The staleness contract was extracted into ``dataknobs_data.vector.content``
for two classes. It had four more copies of the same loop, and one backend
that dropped the digest on the way to storage --- so a corpus could be judged
by a rule that never reached it.

Each cell here is reproduce-first against the tree that extracted the
assembler without converting these consumers.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from dataknobs_data.backends.elasticsearch_mixins import (
    ElasticsearchRecordSerializer,
    vector_tracking_metadata,
)
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.records import Record
from dataknobs_data.testing import text_embedding
from dataknobs_data.vector.content import (
    CONTENT_HASH_KEY,
    FIELD_SEPARATOR_KEY,
    SOURCE_FIELDS_KEY,
    compute_content_hash,
)
from dataknobs_data.vector.migration import IncrementalVectorizer
from dataknobs_data.vector.mixins import VectorSyncMixin
from dataknobs_data.vector.sync import VectorTextSynchronizer


def _embed_batch(texts: list[str]) -> np.ndarray:
    return np.array([text_embedding(t) for t in texts], dtype=np.float32)


class RecordingBatchEmbedder:
    """A batch embedding function that keeps what it was asked to embed."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.texts.extend(texts)
        return _embed_batch(texts)


class TextSyncHost(VectorSyncMixin):
    """The mixin under test, hosted on nothing else.

    ``VectorSyncMixin`` reads and writes only the records it is handed, so it
    needs no database to exercise. Nothing in the tree composes it with one,
    which is itself worth knowing.
    """


class TestVectorSyncMixinAssembly:
    """``sync_vectors_with_text`` --- the third copy of the loop."""

    @pytest.mark.asyncio
    async def test_the_separator_is_configurable_and_reaches_the_embedder(self):
        """Fails against a hardcoded ``" ".join``."""
        embedder = RecordingBatchEmbedder()
        record = Record(data={"title": "Doc 1", "content": "Content 1"})

        updated = await TextSyncHost().sync_vectors_with_text(
            [record],
            text_fields=["title", "content"],
            embedding_fn=embedder,
            field_separator="\n",
        )

        assert updated == 1
        assert embedder.texts == ["Doc 1\nContent 1"]

    @pytest.mark.asyncio
    async def test_it_records_a_digest_so_the_field_can_be_judged(self):
        """Fails by writing a vector with no digest.

        A field with no digest is treated as current by every reader of the
        contract, forever --- so a corpus built through this method could not
        be re-embedded by a ``VectorTextSynchronizer`` sweeping the same
        records.
        """
        record = Record(data={"title": "Doc 1", "content": "Content 1"})

        await TextSyncHost().sync_vectors_with_text(
            [record],
            text_fields=["title", "content"],
            embedding_fn=RecordingBatchEmbedder(),
            field_separator=" | ",
        )

        metadata = record.fields["embedding"].metadata
        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("Doc 1 | Content 1")
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "content"]
        assert metadata[FIELD_SEPARATOR_KEY] == " | "

    @pytest.mark.asyncio
    async def test_an_edited_source_is_re_embedded(self):
        """Fails by never re-embedding.

        The staleness check compared the *set of source fields* and nothing
        else, so a vector went on being reported current after its text was
        edited --- the same omission ``_has_current_vector`` carried, in a
        second class.
        """
        host = TextSyncHost()
        embedder = RecordingBatchEmbedder()
        record = Record(data={"title": "Doc 1", "content": "Content 1"})

        await host.sync_vectors_with_text(
            [record], text_fields=["title", "content"], embedding_fn=embedder
        )
        record.set_value("content", "Wholly different prose")
        embedder.texts.clear()

        updated = await host.sync_vectors_with_text(
            [record], text_fields=["title", "content"], embedding_fn=embedder
        )

        assert updated == 1
        assert embedder.texts == ["Doc 1 Wholly different prose"]

    @pytest.mark.asyncio
    async def test_an_unchanged_record_is_not_re_embedded(self):
        """The companion: the cell above must not be satisfied by a method
        that re-embeds unconditionally.
        """
        host = TextSyncHost()
        embedder = RecordingBatchEmbedder()
        record = Record(data={"title": "Doc 1", "content": "Content 1"})

        await host.sync_vectors_with_text(
            [record], text_fields=["title", "content"], embedding_fn=embedder
        )
        embedder.texts.clear()

        updated = await host.sync_vectors_with_text(
            [record], text_fields=["title", "content"], embedding_fn=embedder
        )

        assert updated == 0
        assert embedder.texts == []

    @pytest.mark.asyncio
    async def test_a_vector_field_with_no_source_field_does_not_raise(self):
        """Fails with ``AttributeError: 'NoneType' object has no attribute 'split'``.

        ``VectorField`` writes ``source_field`` into its metadata as ``None``
        when it has none, so the key is *present* --- and
        ``metadata.get("source_field", "")`` returns ``None``, not the default.
        Any multi-field vector built without an explicit source name reached
        that line.
        """
        record = Record(data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1 Content 1"), name="embedding"
        )
        assert record.fields["embedding"].metadata["source_field"] is None

        updated = await TextSyncHost().sync_vectors_with_text(
            [record],
            text_fields=["title", "content"],
            embedding_fn=RecordingBatchEmbedder(),
        )

        assert updated == 1


class TestBulkEmbedAssembly:
    """``BulkEmbedMixin`` --- the fourth and fifth copies of the loop."""

    def test_the_separator_is_configurable_and_a_digest_is_recorded(self):
        """Fails on both counts against the unconverted mixin.

        The records come from ``all()`` rather than ``read()`` for no reason
        beyond convenience now. It used to be a workaround:
        ``SyncMemoryDatabase.read`` dropped the id, and ``bulk_embed_and_store``
        reads ``record.id`` to choose between update and create, so a ``read()``
        result made it store a *duplicate*. That is fixed at its own layer --- see
        ``test_read_preserves_storage_id.py``, which pins the contract for every
        backend rather than for this caller.
        """
        db = SyncMemoryDatabase(config={"vector_enabled": True})
        db.connect()
        try:
            embedder = RecordingBatchEmbedder()
            record_id = db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))

            db.bulk_embed_and_store(
                db.all(),
                text_field=["title", "content"],
                embedding_fn=embedder,
                field_separator="\n",
            )

            assert embedder.texts == ["Doc 1\nContent 1"]
            assert len(db.all()) == 1, "bulk_embed_and_store stored a duplicate record"

            metadata = db.read(record_id).fields["embedding"].metadata
            assert metadata[CONTENT_HASH_KEY] == compute_content_hash("Doc 1\nContent 1")
            assert metadata[SOURCE_FIELDS_KEY] == ["title", "content"]
            assert metadata[FIELD_SEPARATOR_KEY] == "\n"
        finally:
            db.close()


class TestIncrementalVectorizerAssembly:
    """``IncrementalVectorizer`` --- the sixth copy, and the one that already
    honoured its own separator, so only the duplication was at stake.
    """

    @pytest.mark.asyncio
    async def test_it_assembles_through_the_shared_assembler(self):
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            seen: list[str] = []

            def embed(text: str) -> np.ndarray:
                seen.append(text)
                return text_embedding(text)

            record_id = await db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
            vectorizer = IncrementalVectorizer(
                database=db,
                embedding_fn=embed,
                text_fields=["title", "content"],
                field_separator="\n",
                model_name="test-model",
            )

            await vectorizer._process_record(await db.read(record_id))

            assert seen == ["Doc 1\nContent 1"]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_the_source_field_list_survives_a_non_default_separator(self):
        """Fails by joining *field names* on the *content* separator.

        The only reader of this key splits it on a comma, so joining the names
        on ``field_separator`` produced one unsplittable string for every
        non-default separator.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            record_id = await db.create(Record(data={"title": "Doc 1", "content": "Content 1"}))
            vectorizer = IncrementalVectorizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["title", "content"],
                field_separator="\n",
                model_name="test-model",
            )

            await vectorizer._process_record(await db.read(record_id))

            sidecar = (await db.read(record_id)).get_value("embedding_metadata")
            assert sidecar["source_field"].split(",") == ["title", "content"]
        finally:
            await db.close()


class TestPlainValueModelVersion:
    """``_has_current_vector``'s other branch reads a sidecar nothing wrote."""

    @pytest.mark.asyncio
    async def test_a_vectorizer_written_sidecar_is_read_back(self):
        """Fails by reporting a version mismatch on every sweep.

        ``VectorMetadata.to_dict`` nests the version as
        ``{"model": {"version": ...}}``; the reader asked for a flat
        ``model_version`` key that nothing in the tree writes. So a record
        vectorized by ``IncrementalVectorizer`` was re-embedded by a
        ``VectorTextSynchronizer`` every single sweep, forever.

        This branch was recorded as dead on the grounds that nothing writes
        the field. Something does.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            record_id = await db.create(Record(data={"content": "Content 1"}))
            vectorizer = IncrementalVectorizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                model_name="test-model",
                model_version="v1",
            )
            await vectorizer._process_record(await db.read(record_id))

            sync = VectorTextSynchronizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                model_version="v1",
            )
            stored = await db.read(record_id)

            assert stored.get_value("embedding_metadata")["model"]["version"] == "v1"
            assert sync._has_current_vector(stored, "embedding") is True
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_a_genuine_version_mismatch_is_still_detected(self):
        """The companion, so the cell above is not satisfied by a reader that
        stopped checking.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            record_id = await db.create(Record(data={"content": "Content 1"}))
            await IncrementalVectorizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                model_name="test-model",
                model_version="v1",
            )._process_record(await db.read(record_id))

            sync = VectorTextSynchronizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                model_version="v2",
            )

            assert sync._has_current_vector(await db.read(record_id), "embedding") is False
        finally:
            await db.close()


class TestElasticsearchPreservesTheDigest:
    """A backend that drops the digest turns "no digest means current" from a
    benign default into permanent staleness.

    ``_record_to_document`` stores only the vector's numbers; everything else
    about the field travels in ``record.metadata["vector_fields"]``, which
    both Elasticsearch backends populated from an identical five-key
    whitelist that did not include the field's metadata. So the digest the
    synchronizer had just written never reached the index, came back absent,
    and every record read as current --- for good, since nothing re-embeds a
    current record.

    These exercise the serialization contract directly, which is where the
    drop was; they need no server.
    """

    @staticmethod
    def _vector_record() -> Record:
        record = Record(id="doc-1", data={"title": "Doc 1", "content": "Content 1"})
        record.fields["embedding"] = VectorField(
            value=text_embedding("Doc 1\nContent 1"),
            name="embedding",
            source_field=None,
            model_name="test-model",
            model_version="v1",
            metadata={
                CONTENT_HASH_KEY: compute_content_hash("Doc 1\nContent 1"),
                SOURCE_FIELDS_KEY: ["title", "content"],
                FIELD_SEPARATOR_KEY: "\n",
            },
        )
        return record

    def test_the_tracking_entry_carries_the_field_metadata(self):
        record = self._vector_record()

        entry = vector_tracking_metadata(record.fields["embedding"], dimensions=384)

        assert entry["metadata"][CONTENT_HASH_KEY] == compute_content_hash("Doc 1\nContent 1")
        assert entry["metadata"][SOURCE_FIELDS_KEY] == ["title", "content"]
        assert entry["metadata"][FIELD_SEPARATOR_KEY] == "\n"

    def test_the_digest_survives_a_document_round_trip(self):
        record = self._vector_record()
        record.metadata["vector_fields"] = {
            "embedding": vector_tracking_metadata(record.fields["embedding"], dimensions=384)
        }

        doc = ElasticsearchRecordSerializer._record_to_document(record)
        restored = ElasticsearchRecordSerializer._document_to_record(doc, doc_id="doc-1")

        metadata = restored.fields["embedding"].metadata
        assert metadata[CONTENT_HASH_KEY] == compute_content_hash("Doc 1\nContent 1")
        assert metadata[SOURCE_FIELDS_KEY] == ["title", "content"]
        assert metadata[FIELD_SEPARATOR_KEY] == "\n"

    def test_a_restored_record_is_still_judged_by_the_synchronizer(self):
        """The point of the round trip, stated as the behaviour it protects:
        an edited record that has been through Elasticsearch must still come
        back stale.
        """
        record = self._vector_record()
        record.metadata["vector_fields"] = {
            "embedding": vector_tracking_metadata(record.fields["embedding"], dimensions=384)
        }
        doc = ElasticsearchRecordSerializer._record_to_document(record)
        restored = ElasticsearchRecordSerializer._document_to_record(doc, doc_id="doc-1")

        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        sync = VectorTextSynchronizer(
            database=db,
            embedding_fn=text_embedding,
            text_fields=["title", "content"],
            field_separator="\n",
        )

        assert sync._has_current_vector(restored, "embedding") is True

        restored.set_value("content", "Wholly different prose")
        assert sync._has_current_vector(restored, "embedding") is False


class TestIncrementalVectorizerLoadsItsQueue:
    """`_load_queue` called a method no database has.

    `AsyncDatabase` defines no `filter`, and neither does any backend. The
    call raised `AttributeError` on the first iteration --- into an
    `except Exception` that logs, sleeps ten seconds and loops --- so the
    vectorizer enqueued nothing, forever, while reporting only a recurring
    log line.

    It survived because the class annotated its database as `Database`, a
    name `dataknobs_data.database` does not define. The checker could say
    nothing about any call made through it. This is the same defect as the
    `str | None` writes that the same wrong annotation hid in the two sibling
    files, one file further along.
    """

    @pytest.mark.asyncio
    async def test_records_without_a_vector_reach_the_queue(self):
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            await db.create(Record(data={"content": "Needs a vector"}))
            already = Record(data={"content": "Has one"})
            already.fields["embedding"] = VectorField(
                value=text_embedding("Has one"), name="embedding"
            )
            await db.create(already)

            vectorizer = IncrementalVectorizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                batch_size=10,
            )

            queued = await vectorizer._load_pending_records()

            assert [r.get_value("content") for r in queued] == ["Needs a vector"]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_the_loader_terminates_instead_of_refilling_the_queue(self):
        """The defect the fix above uncovered, pinned so it cannot come back.

        With a working fetch the loop re-queried the instant it had enqueued a
        batch, got back the records the workers had not written yet, and
        enqueued them again --- growing the queue faster than it drained and
        never finishing. It was invisible while the fetch itself raised.

        Fails by hanging, so it is written with a timeout rather than a bare
        `await`.
        """
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.connect()
        try:
            for i in range(3):
                await db.create(Record(data={"content": f"Document number {i}"}))

            vectorizer = IncrementalVectorizer(
                database=db,
                embedding_fn=text_embedding,
                text_fields=["content"],
                batch_size=2,
                max_workers=1,
            )

            async def every_record_vectorized() -> None:
                # Polled rather than `wait_for_completion`, which returns the
                # moment the queue is empty --- true before the loader has
                # filled it for the first time as well as after it is done.
                while True:
                    vectored = [r for r in await db.all() if r.get_value("embedding") is not None]
                    if len(vectored) == 3:
                        return
                    await asyncio.sleep(0.02)

            await vectorizer.start()
            try:
                await asyncio.wait_for(every_record_vectorized(), timeout=10.0)
            finally:
                # Bounded: against the unfixed loader the queue grows without
                # limit, and an unbounded `stop()` turns a failing test into a
                # hanging suite.
                await asyncio.wait_for(vectorizer.stop(timeout=2.0), timeout=5.0)

            assert vectorizer._queue.qsize() == 0, (
                "the loader was still refilling the queue after all records were vectorized"
            )
        finally:
            await db.close()
