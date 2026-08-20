"""Tests for the migration example."""

import pytest
import sys
import time
import zlib
from pathlib import Path
from typing import List

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField
from dataknobs_data.vector import VectorMigration, IncrementalVectorizer

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

# The example imports sentence-transformers lazily, so this import costs
# nothing and needs no optional dependency present. It used to load a model
# at module scope, which made the example unimportable here -- so these
# tests grew up testing the library the example calls, and re-declaring its
# MigrationStats rather than importing it. Nothing then covered the example
# itself, and every record in it failed to migrate for as long as that held.
from migrate_existing_data import (  # noqa: E402 - must follow the sys.path.insert above
    MigrationStats,
    main as run_migration_example,
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
async def legacy_db():
    """Create a legacy database without vector support."""
    factory = AsyncDatabaseFactory()
    db = factory.create(backend="sqlite", path=":memory:", vector_enabled=False)
    await db.connect()

    # Add legacy data
    legacy_data = [
        {
            "id": 1,
            "type": "article",
            "title": "Cloud Computing",
            "content": "Introduction to cloud services.",
            "author": "John Doe",
        },
        {
            "id": 2,
            "type": "tutorial",
            "title": "Docker Basics",
            "content": "Learn containerization with Docker.",
            "author": "Jane Smith",
        },
        {
            "id": 3,
            "type": "guide",
            "title": "API Design",
            "content": "Best practices for RESTful APIs.",
            "author": "Bob Wilson",
        },
    ]

    for data in legacy_data:
        await db.create(Record(data))

    yield db
    await db.close()


@pytest.fixture
async def vector_db():
    """Create a vector-enabled database."""
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite", path=":memory:", vector_enabled=True, vector_metric="cosine"
    )
    await db.connect()
    yield db
    await db.close()


class TestVectorMigration:
    """Test vector migration functionality."""

    @pytest.mark.asyncio
    async def test_migration_initialization(self, legacy_db, vector_db):
        """Test VectorMigration initialization."""
        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
        )

        assert migration.source_db == legacy_db
        assert migration.target_db == vector_db
        assert callable(migration.embedding_function)

    @pytest.mark.asyncio
    async def test_migration_configuration(self, legacy_db, vector_db):
        """Test migration configuration."""
        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            batch_size=2,
        )

        assert migration.text_fields == ["title", "content"]
        assert migration.vector_field == "embedding"
        assert migration.batch_size == 2

    @pytest.mark.asyncio
    async def test_basic_migration(self, legacy_db, vector_db):
        """Test basic migration from legacy to vector database."""
        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
        )

        # Track progress
        progress_calls = []

        def progress_callback(status):
            progress_calls.append(status)

        # Run migration
        results = await migration.run(progress_callback=progress_callback)

        # Verify migration results
        assert results.total_processed == 3
        assert results.failed_count == 0
        assert len(progress_calls) > 0

        # Verify records in target database
        from dataknobs_data import Query

        migrated_records = await vector_db.search(Query())
        assert len(migrated_records) == 3

        # Check embeddings
        for record in migrated_records:
            assert "embedding" in record.fields
            assert len(record.fields["embedding"].value) == 384

    @pytest.mark.asyncio
    async def test_migration_with_retry(self, legacy_db, vector_db):
        """Test migration with retry logic."""
        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            max_retries=3,
            retry_delay=0.1,
        )

        # Run migration
        results = await migration.run()

        assert results.total_processed == 3

    @pytest.mark.asyncio
    async def test_migration_failure_handling(self, legacy_db, vector_db):
        """Test handling of migration failures."""
        # Create a failing embedding function
        call_count = 0

        def failing_embedding(text: str) -> List[float]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise ValueError("Embedding generation failed")
            return mock_generate_embedding(text)

        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=failing_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            max_retries=0,
        )

        # Run migration
        results = await migration.run()

        # Should have one failure
        assert results.failed_count == 1
        assert results.total_processed - results.failed_count == 2


class TestIncrementalVectorizer:
    """Test incremental vectorization functionality."""

    @pytest.mark.asyncio
    async def test_incremental_vectorizer_init(self, vector_db):
        """Test IncrementalVectorizer initialization."""
        vectorizer = IncrementalVectorizer(
            database=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields="title",
            vector_field="embedding",
        )

        assert vectorizer.database == vector_db
        assert callable(vectorizer.embedding_function)

    @pytest.mark.asyncio
    async def test_incremental_configuration(self, vector_db):
        """Test incremental vectorizer configuration."""
        vectorizer = IncrementalVectorizer(
            database=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            batch_size=2,
            checkpoint_interval=5,
        )

        assert vectorizer.text_fields == ["title", "content"]
        assert vectorizer.vector_field == "embedding"
        assert vectorizer.batch_size == 2
        assert vectorizer.checkpoint_interval == 5

    @pytest.mark.asyncio
    async def test_incremental_processing(self, vector_db):
        """Test incremental processing of records."""
        # Add records without embeddings
        records = [{"title": f"Doc {i}", "content": f"Content {i}"} for i in range(5)]

        for record in records:
            await vector_db.create(Record(record))

        vectorizer = IncrementalVectorizer(
            database=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            batch_size=2,
        )

        # Track progress
        progress_calls = []

        async def progress_callback(completed, total, current_batch):
            progress_calls.append((completed, total, len(current_batch)))

        # Run vectorization
        results = await vectorizer.run(progress_callback=progress_callback, max_workers=1)

        assert results["processed"] == 5
        assert results["failed"] == 0
        assert len(progress_calls) > 0

        # Verify embeddings added
        from dataknobs_data import Query

        all_records = await vector_db.search(Query())
        for record in all_records:
            assert "embedding" in record.fields

    @pytest.mark.asyncio
    async def test_vectorizer_status(self, vector_db):
        """Test getting vectorizer status."""
        vectorizer = IncrementalVectorizer(
            database=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title"],
            vector_field="embedding",
        )

        # Add some records
        for i in range(3):
            await vector_db.create(Record({"title": f"Doc {i}"}))

        # Get initial status
        status = await vectorizer.get_status()
        assert status["total"] == 3
        assert status["completed"] == 0

        # Run vectorization
        await vectorizer.run()

        # Get final status
        status = await vectorizer.get_status()
        assert status["completed"] == 3


class TestMigrationStats:
    """Test the example's own statistics tracking."""

    def test_stats_initialization(self):
        """Test MigrationStats initialization."""
        stats = MigrationStats()
        assert stats.total_records == 0
        assert stats.migrated_records == 0
        assert stats.failed_records == 0

    def test_stats_duration(self):
        """Test duration calculation."""
        stats = MigrationStats()
        stats.start_time = time.time()
        time.sleep(0.1)
        stats.end_time = time.time()

        assert stats.duration >= 0.1

    def test_stats_success_rate(self):
        """Test success rate calculation."""
        stats = MigrationStats()
        stats.total_records = 10
        stats.migrated_records = 8
        stats.failed_records = 2

        assert stats.success_rate == 80.0

        # Test with no records
        stats.total_records = 0
        assert stats.success_rate == 0


@pytest.mark.asyncio
async def test_complete_migration_workflow():
    """Test the complete migration workflow."""
    # Create legacy database
    factory = AsyncDatabaseFactory()
    legacy_db = factory.create(backend="sqlite", path=":memory:", vector_enabled=False)
    await legacy_db.connect()

    # Create vector database
    vector_db = factory.create(backend="sqlite", path=":memory:", vector_enabled=True)
    await vector_db.connect()

    try:
        # Add legacy data
        legacy_data = [
            {"id": i, "title": f"Document {i}", "content": f"Content for document {i}"}
            for i in range(5)
        ]

        for data in legacy_data:
            await legacy_db.create(Record(data))

        # Create migration
        migration = VectorMigration(
            source_db=legacy_db,
            target_db=vector_db,
            embedding_fn=mock_generate_embedding,
            text_fields=["title", "content"],
            vector_field="embedding",
            batch_size=2,
        )

        # Run migration
        results = await migration.run()

        assert results.total_processed == 5
        assert results.failed_count == 0

        # Verify target database
        from dataknobs_data import Query

        migrated = await vector_db.search(Query())
        assert len(migrated) == 5

        # Test vector search on migrated data
        query_embedding = mock_generate_embedding("Document search")
        search_results = await vector_db.vector_search(
            query_vector=query_embedding, k=3, vector_field="embedding"
        )

        assert len(search_results) <= 3

    finally:
        await legacy_db.close()
        await vector_db.close()


@pytest.mark.asyncio
async def test_migration_verification():
    """Test migration verification functionality."""
    factory = AsyncDatabaseFactory()
    vector_db = factory.create(backend="sqlite", path=":memory:", vector_enabled=True)
    await vector_db.connect()

    try:
        # Add records with and without embeddings
        with_embedding = Record(
            {
                "title": "With Vector",
                "embedding": VectorField(mock_generate_embedding("test"), dimensions=384),
            }
        )
        without_embedding = Record({"title": "Without Vector"})

        await vector_db.create(with_embedding)
        await vector_db.create(without_embedding)

        # Count records with vectors
        from dataknobs_data import Query

        all_records = await vector_db.search(Query())
        records_with_vectors = sum(
            1
            for r in all_records
            if "embedding" in r.fields and r.fields["embedding"].value is not None
        )
        records_without_vectors = len(all_records) - records_with_vectors

        assert records_with_vectors == 1
        assert records_without_vectors == 1

    finally:
        await vector_db.close()


class TestMigrationExample:
    """Drive the shipped example itself, rather than the library it calls.

    Everything above exercises VectorMigration and IncrementalVectorizer
    directly. That is worth doing, but it is not a test of the example: the
    example can call those classes wrongly -- and did, at six separate
    sites -- while every test here stays green.
    """

    @pytest.mark.asyncio
    async def test_example_migrates_every_record(self, capsys):
        """The example's own ``main()``, start to finish, no step stubbed.

        Only the embedding function is supplied, through the parameter the
        example exposes for it; there is nothing to patch and nothing to
        mock. A real in-memory SQLite database does the rest, in about a
        second.
        """
        stats = await run_migration_example(mock_generate_embedding)

        assert stats.total_records == 8
        assert stats.migrated_records == 8
        assert stats.failed_records == 0

        out = capsys.readouterr().out
        assert "✗ Failed:" not in out
        assert "Migration example completed successfully" in out

    @pytest.mark.asyncio
    async def test_example_reaches_every_numbered_step(self, capsys):
        """Each step prints a heading; assert the run got to the last one.

        Step 6 is here because it used to be defined and never called --
        the example's one demonstration of the library's own migration path
        was dead code, and a test that never ran main() could not notice.
        """
        await run_migration_example(mock_generate_embedding)

        out = capsys.readouterr().out
        for heading in (
            "1. Creating legacy database",
            "2. Creating new database with vector support",
            "3. Manual Migration Process",
            "4. Incremental Vectorization",
            "5. Verifying Migration",
            "6. Migration with Retry Logic",
        ):
            assert heading in out, heading

    @pytest.mark.asyncio
    async def test_example_reports_success_rate_as_a_percentage(self, capsys):
        """A fully successful migration reads 100.0%, not 1.0%.

        ``MigrationStatus.success_rate`` is a 0..1 ratio, so the example's
        ``:.1f%`` format string needed scaling. Printing "1.0%" next to
        "Failed: 0 records" is the kind of wrong that survives review.
        """
        await run_migration_example(mock_generate_embedding)

        out = capsys.readouterr().out
        assert "Success rate: 100.0%" in out
        assert "Success rate: 1.0%" not in out

    @pytest.mark.asyncio
    async def test_example_verifies_vectors_are_present(self, capsys):
        """Verification counts vectors by field presence, not truthiness.

        An embedding's value is a numpy array, so the ``if record.get(...)``
        this used to do raised "truth value of an array ... is ambiguous"
        the moment a record actually had one.
        """
        await run_migration_example(mock_generate_embedding)

        out = capsys.readouterr().out
        assert "Records with vectors: 8" in out
        assert "Records without vectors: 0" in out
