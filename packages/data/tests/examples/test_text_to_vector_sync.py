"""Tests for the text-to-vector synchronization example."""

import pytest
import sys
import zlib
from pathlib import Path
from typing import List

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField
from dataknobs_data.vector import VectorTextSynchronizer, ChangeTracker

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

# The example imports sentence-transformers lazily, so this costs nothing and
# needs no optional dependency present. It used to load a model at module
# scope, which is why nothing here imported it -- and why six defects in it
# went unnoticed while every test in this file stayed green.
from text_to_vector_sync import (  # noqa: E402 - must follow the sys.path.insert above
    main as run_sync_example,
)


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def encode(self, text: str) -> List[float]:
        """Generate deterministic fake embeddings."""
        # crc32, not hash(): str.__hash__ is salted per process
        # (PYTHONHASHSEED), so hash() would break the determinism this
        # docstring promises.
        hash_val = zlib.crc32(text.encode()) % 1000
        return [float((hash_val + i) % 100) / 100.0 for i in range(384)]


def mock_generate_embedding(text: str) -> List[float]:
    """Mock embedding generation function."""
    model = MockEmbeddingModel()
    return model.encode(text)


@pytest.fixture
async def test_db():
    """Create a test database."""
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite", path=":memory:", vector_enabled=True, vector_metric="cosine"
    )
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def populated_db(test_db):
    """Create a database with test documents."""
    documents = [
        {
            "id": 1,
            "title": "Python Basics",
            "content": "Introduction to Python programming.",
            "author": "Alice",
        },
        {
            "id": 2,
            "title": "Advanced Python",
            "content": "Deep dive into Python features.",
            "author": "Bob",
        },
        {
            "id": 3,
            "title": "Data Science",
            "content": "Using Python for data analysis.",
            "author": "Carol",
        },
    ]

    for doc in documents:
        await test_db.create(Record(doc))

    return test_db


class TestVectorTextSynchronization:
    """Test text-to-vector synchronization functionality."""

    @pytest.mark.asyncio
    async def test_synchronizer_initialization(self, test_db):
        """Test VectorTextSynchronizer initialization."""
        sync = VectorTextSynchronizer(
            test_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
        )

        assert sync.database == test_db
        assert callable(sync.embedding_function)

    @pytest.mark.asyncio
    async def test_synchronizer_setup(self, test_db):
        """Test synchronizer setup."""
        sync = VectorTextSynchronizer(
            test_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
        )

        # Synchronizer should be configured on initialization
        assert sync.text_fields == ["title", "content"]
        assert sync.vector_field == "embedding"

    @pytest.mark.asyncio
    async def test_bulk_sync(self, populated_db):
        """Test bulk synchronization of records."""
        sync = VectorTextSynchronizer(
            populated_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
        )

        # Track progress
        progress_calls = []

        def progress_callback(done, total):
            progress_calls.append((done, total))

        # Run bulk sync
        results = await sync.sync_all(batch_size=2, progress_callback=progress_callback)

        # Verify all records were processed
        assert results["processed"] == 3
        assert len(progress_calls) > 0

        # Check that records now have embeddings
        from dataknobs_data import Query

        all_records = await populated_db.search(Query())
        for record in all_records:
            assert "embedding" in record.fields
            assert len(record.fields["embedding"].value) == 384

    @pytest.mark.asyncio
    async def test_change_tracker(self, populated_db):
        """Test change tracking functionality."""
        tracker = ChangeTracker(
            populated_db, tracked_fields=["title", "content"], vector_field="embedding"
        )
        await tracker.start_processing()

        # Initially all records should be outdated (no embeddings)
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 3

        # Add embedding to one record
        first_record = outdated[0]
        first_record.fields["embedding"] = VectorField(
            mock_generate_embedding("test"), dimensions=384
        )
        await populated_db.update(first_record.id, first_record)

        # Mark as updated - tracker automatically detects updates

        # Now should have 2 outdated records
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 2

    @pytest.mark.asyncio
    async def test_sync_single_record(self, populated_db):
        """Test synchronizing a single record."""
        sync = VectorTextSynchronizer(
            populated_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
        )

        # Get first record
        from dataknobs_data import Query

        records = await populated_db.search(Query())
        first_record = records[0]

        # Sync single record
        await sync.sync_record(first_record.id)

        # Verify embedding was added
        updated = await populated_db.read(first_record.id)
        assert "embedding" in updated.fields
        assert len(updated.fields["embedding"].value) == 384

    @pytest.mark.asyncio
    async def test_auto_sync_flag(self, test_db):
        """Test auto-sync enable/disable."""
        sync = VectorTextSynchronizer(
            test_db, mock_generate_embedding, text_fields=["title"], auto_sync=False
        )

        # Initially disabled
        assert not sync.auto_sync

        # Enable auto-sync
        sync.auto_sync = True
        assert sync.auto_sync

        # Disable auto-sync
        sync.auto_sync = False
        assert not sync.auto_sync

    @pytest.mark.asyncio
    async def test_update_detection(self, populated_db):
        """Test detection of updated records."""
        tracker = ChangeTracker(
            populated_db, tracked_fields=["title", "content"], vector_field="embedding"
        )
        await tracker.start_processing()

        # Add embeddings to all records
        sync = VectorTextSynchronizer(
            populated_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
        )
        await sync.sync_all()

        # No outdated records
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 0

        # Update a document's content
        from dataknobs_data import Query

        records = await populated_db.search(Query())
        first_record = records[0]
        first_record.set_value("content", "Updated content that needs new embedding")
        await populated_db.update(first_record.id, first_record)

        # Should detect the update
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1
        assert outdated[0].id == records[0].id

    @pytest.mark.asyncio
    async def test_batch_processing(self, populated_db):
        """Test batch processing with different batch sizes."""
        sync = VectorTextSynchronizer(
            populated_db,
            mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            field_separator=" ",
        )

        # Test with batch_size=1
        results = await sync.sync_all(batch_size=1)
        assert results["processed"] == 3

        # Clear embeddings
        from dataknobs_data import Query

        for record in await populated_db.search(Query()):
            record.fields.pop("embedding", None)
            await populated_db.update(record.id, record)

        # Test with batch_size=10 (larger than record count)
        results = await sync.sync_all(batch_size=10)
        assert results["processed"] == 3


class TestDocumentSyncClass:
    """Test the DocumentSync helper class from the example."""

    @pytest.mark.asyncio
    async def test_document_sync_setup(self, test_db):
        """Test DocumentSync class setup."""
        # Skip this test as it requires sentence_transformers
        pytest.skip("Requires sentence_transformers module")

    @pytest.mark.asyncio
    async def test_show_sync_status(self, populated_db):
        """Test sync status display."""
        # Skip this test as it requires sentence_transformers
        pytest.skip("Requires sentence_transformers module")


@pytest.mark.asyncio
async def test_example_workflow():
    """Test the complete synchronization workflow."""
    # Create database
    factory = AsyncDatabaseFactory()
    db = factory.create(backend="sqlite", path=":memory:", vector_enabled=True)
    await db.connect()

    try:
        # Create documents without embeddings
        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
            {"title": "Doc 3", "content": "Content 3"},
        ]

        record_ids = []
        for doc in docs:
            record_id = await db.create(Record(doc))
            record_ids.append(record_id)

        # Setup synchronization
        sync = VectorTextSynchronizer(
            db, mock_generate_embedding, text_fields=["title", "content"], vector_field="embedding"
        )

        # Bulk sync
        results = await sync.sync_all()
        assert results["processed"] == 3

        # Verify embeddings
        for record_id in record_ids:
            record = await db.read(record_id)
            assert "embedding" in record.fields
            assert len(record.fields["embedding"].value) == 384

        # Update a document
        record = await db.read(record_ids[0])
        record.set_value("content", "Updated content")
        await db.update(record_ids[0], record)

        # Track changes
        tracker = ChangeTracker(db, tracked_fields=["title", "content"], vector_field="embedding")
        await tracker.start_processing()

        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1

        # Re-sync outdated record
        await sync.sync_record(outdated[0].id)

        # Verify no more outdated
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 0

    finally:
        await db.close()


class TestSyncExample:
    """Drive the shipped example itself, rather than the library it calls.

    Everything above exercises VectorTextSynchronizer and ChangeTracker
    directly. Useful, but not a test of the example: the example can call
    those classes wrongly -- and did, at six sites -- with none of it
    visible from here.
    """

    @pytest.mark.asyncio
    async def test_example_runs_end_to_end(self, capsys):
        """The example's own ``main()``, start to finish, nothing stubbed.

        Only the embedding function is supplied, through the parameter the
        example exposes for it. A real in-memory SQLite database does the
        rest.
        """
        await run_sync_example(mock_generate_embedding)

        out = capsys.readouterr().out
        assert "Synchronization Performance Metrics" in out

    @pytest.mark.asyncio
    async def test_example_reaches_every_numbered_step(self, capsys):
        """Each step prints a heading; assert the run got to the last one.

        The example used to die at step 4 on a progress callback whose
        arity did not match ``sync_all``'s ``(done, total)`` contract, then
        at three separate ``db.update(id, dict)`` calls -- ``update`` takes
        a Record -- then on a numpy array's ambiguous truth value.
        """
        await run_sync_example(mock_generate_embedding)

        out = capsys.readouterr().out
        for heading in (
            "1. Setting up database",
            "4. Running bulk synchronization",
            "6. Updating document and tracking changes",
            "7. Re-syncing the record whose text changed",
            "9. Batch updating multiple documents",
            "10. Synchronization Performance Metrics",
        ):
            assert heading in out, heading

    @pytest.mark.asyncio
    async def test_example_syncs_every_record(self, capsys):
        """By the final status check, nothing is left without an embedding."""
        await run_sync_example(mock_generate_embedding)

        out = capsys.readouterr().out
        # The example prints a sync status block several times; the last one
        # is the state it leaves the database in.
        last_status = out.rsplit("Sync Status:", 1)[-1]
        assert "Without embeddings: 0" in last_status

    @pytest.mark.asyncio
    async def test_example_actually_resyncs_the_edited_record(self, capsys):
        """Step 7 does work, rather than iterating an empty list.

        It used to loop over whatever `show_sync_status` called outdated --
        a list that only ever holds records with *no* embedding, so an
        edited record was never in it and the loop body never ran.
        """
        await run_sync_example(mock_generate_embedding)

        out = capsys.readouterr().out
        assert "changed source fields: ['title', 'content']" in out
        assert "sync_on_update -> True" in out

    @pytest.mark.asyncio
    async def test_sync_record_skips_a_record_already_brought_up_to_date(self, capsys):
        """Step 7's second call has nothing to do, and says so.

        This replaces `test_sync_on_update_is_inert_for_this_configuration`,
        which pinned today's wrong answer (`sync_on_update -> False`) so that
        the example would stop agreeing with the library the moment the library
        was fixed. It fired, as designed, and the step-7 commentary it guarded
        was rewritten in the same change.

        What is worth pinning now is the other half of that fix: `sync_record`
        skips a vector field whose stored digest still matches its source text,
        so calling it on a record `sync_on_update` just re-embedded costs
        nothing. Before the fix this printed `updated=['embedding']` -- a second
        embedding of text that had not changed since the first.
        """
        await run_sync_example(mock_generate_embedding)

        assert "sync_record -> success=True, updated=[]" in capsys.readouterr().out
