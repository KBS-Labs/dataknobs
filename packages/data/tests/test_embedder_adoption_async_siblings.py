"""The three async classes that embed text and had not adopted the seam.

Twenty-five parameter declarations gained an ``embedder`` alongside their
``embedding_fn``. These four did not, and they are the ones where it matters
most: each is a *long-running* embedding path --- a sweep, a migration, a
background vectorizer, a dedup pass --- so a caller holding an embedder had to
unwrap it into a callable and, in doing so, throw away the identity the seam
exists to carry.

``VectorTextSynchronizer`` is the sharpest case, because it is both halves of
the staleness contract. It *writes* ``model_name`` and it *reads* it back in
``_has_current_vector``. Until it took an embedder, a caller with one had to
name the model twice --- once by passing the embedder's ``embed`` and once by
passing ``model_name=`` --- with nothing checking that the two agreed. That is
the exact class of error ``model_id`` was introduced to close, surviving in the
one class positioned to close it.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.dedup import DedupChecker, DedupConfig
from dataknobs_data.fields import VectorField
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector.migration import IncrementalVectorizer, VectorMigration
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.sync import VectorTextSynchronizer

pytestmark = pytest.mark.asyncio


async def _corpus(*texts: str) -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.connect()
    for text in texts:
        await db.create(Record(data={"content": text}))
    return db


class TestTheSynchronizerClosesItsOwnLoop:
    """It writes the key and it reads the key. Now from one source."""

    async def test_a_swept_record_records_the_embedders_identity(self) -> None:
        db = await _corpus("the first document")
        try:
            sync = VectorTextSynchronizer(
                database=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v1"),
                text_fields=["content"],
            )
            assert sync.model_name == "v1", "model_name defaults from the embedder"

            assert (await sync.sync_all())["updated"] == 1

            [record] = await db.all()
            field = record.get_field("embedding")
            assert isinstance(field, VectorField)
            assert field.model_name == "v1"
        finally:
            await db.close()

    async def test_the_same_embedder_reports_its_own_vectors_current(self) -> None:
        """The half that must not regress when the comparison was added."""
        db = await _corpus("the first document")
        try:
            embedder = DeterministicEmbedder(dimensions=8, model_id="v1")
            sync = VectorTextSynchronizer(database=db, embedder=embedder, text_fields=["content"])
            await sync.sync_all()

            assert (await sync.sync_all())["updated"] == 0, (
                "a corpus embedded by this very embedder must not re-embed"
            )
        finally:
            await db.close()

    async def test_swapping_the_embedder_re_embeds_the_corpus(self) -> None:
        """One object passed, one key written, one key compared.

        Before the parameter existed a caller had to pass ``embed`` *and*
        ``model_name=``, and nothing checked they described the same model. The
        point of the adoption is that there is now no second thing to keep in
        step: the identity travels with the embedder.
        """
        db = await _corpus("the first document")
        try:
            v1 = VectorTextSynchronizer(
                database=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v1"),
                text_fields=["content"],
            )
            await v1.sync_all()

            v2 = VectorTextSynchronizer(
                database=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v2"),
                text_fields=["content"],
            )
            assert (await v2.sync_all())["updated"] == 1, (
                "vectors from another model sit in another vector space and "
                "must be regenerated, not compared"
            )

            [record] = await db.all()
            field = record.get_field("embedding")
            assert isinstance(field, VectorField)
            assert field.model_name == "v2"
        finally:
            await db.close()

    async def test_an_explicit_model_name_still_wins(self) -> None:
        """A caller who said what they meant is not overridden."""
        db = await _corpus("the first document")
        try:
            sync = VectorTextSynchronizer(
                database=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v1"),
                text_fields=["content"],
                model_name="a-name-of-my-own",
            )
            assert sync.model_name == "a-name-of-my-own"
        finally:
            await db.close()

    async def test_neither_source_is_refused_at_construction(self) -> None:
        """Where the sweep is built, not part-way through it.

        This class exists only to embed, so a synchronizer with no source is
        never useful. Discovering that on the first record means the failure
        arrives after a query, a batch and a partial write.
        """
        db = await _corpus()
        try:
            with pytest.raises(ValueError, match="embedder is required"):
                VectorTextSynchronizer(database=db, text_fields=["content"])
        finally:
            await db.close()

    async def test_both_sources_are_refused(self) -> None:
        db = await _corpus()
        try:
            with pytest.raises(ValueError, match="not both"):
                VectorTextSynchronizer(
                    database=db,
                    embedder=DeterministicEmbedder(dimensions=8),
                    embedding_fn=lambda text: np.zeros(8),
                    text_fields=["content"],
                )
        finally:
            await db.close()

    async def test_the_callable_path_is_unchanged(self) -> None:
        """Adoption is additive: nothing that worked before stops working."""
        db = await _corpus("the first document")
        try:
            sync = VectorTextSynchronizer(
                database=db,
                embedding_fn=lambda text: np.asarray([float(len(text))] * 8, dtype=np.float32),
                text_fields=["content"],
            )
            assert (await sync.sync_all())["updated"] == 1
        finally:
            await db.close()


class TestTheVectorizerAndTheMigration:
    """Both take an embedder; only one of them requires a source."""

    async def test_the_vectorizer_stores_a_vector_and_names_its_model(self) -> None:
        db = await _corpus("the first document")
        try:
            vectorizer = IncrementalVectorizer(
                database=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v1"),
                text_fields="content",
            )
            assert vectorizer.model_name == "v1"

            [record] = await db.all()
            assert await vectorizer._process_record(record) is True

            stored = await db.read(record.id)
            assert stored is not None
            value = stored.get_value("embedding")
            assert value is not None
            assert len(value) == 8
            assert all(isinstance(component, float) for component in value)
        finally:
            await db.close()

    async def test_the_vectorizer_refuses_no_source(self) -> None:
        """It has no mode that does not embed, so this is a construction error."""
        db = await _corpus()
        try:
            with pytest.raises(ValueError, match="embedder is required"):
                IncrementalVectorizer(database=db, text_fields="content")
        finally:
            await db.close()

    async def test_a_migration_may_be_built_with_no_source_at_all(self) -> None:
        """And that asymmetry with the vectorizer is deliberate.

        Adding the schema field is a migration in its own right, so demanding a
        source at construction would refuse a supported use. The demand is made
        where a vector is actually produced instead.
        """
        db = await _corpus()
        try:
            migration = VectorMigration(source_db=db)
            assert migration.has_embedding_source is False

            with pytest.raises(ValueError, match="embedding source is required"):
                await migration.add_vectors_to_existing({"embedding": "content"})
        finally:
            await db.close()

    async def test_a_migration_with_an_embedder_names_its_model(self) -> None:
        db = await _corpus()
        try:
            migration = VectorMigration(
                source_db=db,
                embedder=DeterministicEmbedder(dimensions=8, model_id="v1"),
                text_fields=["content"],
            )
            assert migration.has_embedding_source is True
            assert migration.model_name == "v1"
        finally:
            await db.close()

    async def test_a_migration_refuses_two_sources(self) -> None:
        db = await _corpus()
        try:
            with pytest.raises(ValueError, match="not both"):
                VectorMigration(
                    source_db=db,
                    embedder=DeterministicEmbedder(dimensions=8),
                    embedding_fn=lambda text: np.zeros(8),
                )
        finally:
            await db.close()


class TestTheDedupChecker:
    """Its semantic pass is optional, so its source is too."""

    async def test_an_embedder_drives_both_halves_of_the_semantic_pass(self) -> None:
        """The write half stores a vector; the read half embeds and searches.

        The threshold is floored so that *any* neighbour counts. This test is
        about which component produced the vectors, not about whether
        ``DeterministicEmbedder`` thinks two unrelated sentences are alike ---
        it deliberately does not, which is the property that makes it usable
        for ranking assertions elsewhere.
        """
        db = AsyncMemoryDatabase()
        await db.connect()
        store = MemoryVectorStore(dimensions=8)
        await store.initialize()
        try:
            embedder = DeterministicEmbedder(dimensions=8, model_id="v1")
            checker = DedupChecker(
                db=db,
                config=DedupConfig(semantic_check=True, similarity_threshold=-1.0),
                vector_store=store,
                embedder=embedder,
            )

            await checker.register({"content": "a question about arithmetic"}, "q-1")

            [(stored_vector, _metadata)] = await store.get_vectors(["q-1"])
            expected = (await embedder.embed(["a question about arithmetic"]))[0]
            assert stored_vector is not None
            assert np.allclose(stored_vector, np.array(expected, dtype=np.float32)), (
                "register must store the vector this embedder produces"
            )

            result = await checker.check({"content": "an entirely unrelated remark"})

            assert result.is_exact_duplicate is False, "different content, different hash"
            assert [item.record_id for item in result.similar_items] == ["q-1"], (
                "check must embed the candidate and search the store"
            )
        finally:
            await store.close()
            await db.close()

    async def test_no_source_leaves_the_semantic_pass_switched_off(self) -> None:
        """Exact-hash matching still works, which is why neither is allowed."""
        db = AsyncMemoryDatabase()
        await db.connect()
        try:
            checker = DedupChecker(db=db, config=DedupConfig(semantic_check=True))

            await checker.register({"content": "hello"}, "doc-1")
            result = await checker.check({"content": "hello"})

            assert result.is_exact_duplicate is True
            assert result.similar_items == []
        finally:
            await db.close()

    async def test_two_sources_are_refused(self) -> None:
        db = AsyncMemoryDatabase()
        await db.connect()
        try:

            async def one(text: str) -> list[float]:
                return [1.0] * 8

            with pytest.raises(ValueError, match="not both"):
                DedupChecker(
                    db=db,
                    config=DedupConfig(),
                    embedder=DeterministicEmbedder(dimensions=8),
                    embedding_fn=one,
                )
        finally:
            await db.close()
