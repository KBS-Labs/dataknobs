"""Tests for vector migration functionality."""

import asyncio
import os
from datetime import datetime

import numpy as np
import pytest

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Vector migration tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.vector.migration import (
    IncrementalVectorizer,
    MigrationConfig,
    MigrationStatus,
    VectorMigration,
)


@pytest.fixture
async def source_database():
    """Create source database with existing data."""
    # Create schema
    schema = DatabaseSchema()
    schema.add_field(FieldSchema(name="content", type=FieldType.TEXT))
    schema.add_field(FieldSchema(name="title", type=FieldType.TEXT))
    schema.add_field(FieldSchema(name="category", type=FieldType.TEXT))
    schema.add_field(FieldSchema(name="metadata", type=FieldType.JSON))
    
    db = AsyncMemoryDatabase()
    db.schema = schema
    await db.connect()
    
    # Add some test data
    for i in range(10):
        await db.create(Record(data={
            "content": f"This is document {i} content",
            "title": f"Document {i}",
            "category": f"category_{i % 3}",
            "metadata": {"index": i, "processed": False},
        }))
    
    yield db
    await db.close()


@pytest.fixture
async def target_database():
    """Create target database with vector fields."""
    # Create schema with vector fields
    schema = DatabaseSchema()
    schema.add_field(FieldSchema(name="content", type=FieldType.TEXT))
    schema.add_field(FieldSchema(name="title", type=FieldType.TEXT))
    
    schema.add_field(FieldSchema(name="category", type=FieldType.TEXT))
    schema.add_field(FieldSchema(name="metadata", type=FieldType.JSON))
    schema.add_field(FieldSchema(
        name="content_embedding",
        type=FieldType.VECTOR,
        metadata={"dimensions": 384, "source_field": "content"}
    ))
    schema.add_field(FieldSchema(
        name="title_embedding",
        type=FieldType.VECTOR,
        metadata={"dimensions": 384, "source_field": "title"}
    ))
    
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    db.schema = schema
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def simple_embedding_fn():
    """Create a simple deterministic embedding function."""
    def embedding_fn(text: str) -> np.ndarray:
        if not text:
            return None
        # Create deterministic embeddings based on text
        np.random.seed(sum(ord(c) for c in text[:10]))
        return np.random.rand(384)
    return embedding_fn


@pytest.fixture
def async_embedding_fn():
    """Create an async embedding function."""
    async def embedding_fn(text: str) -> np.ndarray:
        await asyncio.sleep(0.001)  # Simulate async work
        if not text:
            return None
        np.random.seed(sum(ord(c) for c in text[:10]))
        return np.random.rand(384)
    return embedding_fn


class TestMigrationConfig:
    """Test MigrationConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MigrationConfig()
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.checkpoint_interval == 1000
        assert config.enable_rollback is True
        assert config.verify_migration is True
        assert config.retry_failed is True
        assert config.max_retries == 3
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            config = MigrationConfig(batch_size=0)
            config.validate()
        
        # Invalid workers
        with pytest.raises(ValueError, match="Max workers must be positive"):
            config = MigrationConfig(max_workers=0)
            config.validate()


class TestMigrationStatus:
    """Test MigrationStatus calculations."""
    
    def test_success_rate(self):
        """Test success rate calculation."""
        status = MigrationStatus()
        assert status.success_rate == 0.0
        
        status.total_records = 100
        status.migrated_records = 85
        assert status.success_rate == 0.85
    
    def test_duration_and_speed(self):
        """Test duration and speed calculations."""
        status = MigrationStatus()
        assert status.duration is None
        assert status.records_per_second == 0.0
        
        status.start_time = datetime(2024, 1, 1, 10, 0, 0)
        status.end_time = datetime(2024, 1, 1, 10, 0, 10)
        status.migrated_records = 100
        
        assert status.duration == 10.0
        assert status.records_per_second == 10.0
    
    def test_add_checkpoint(self):
        """Test adding checkpoints."""
        status = MigrationStatus()
        status.migrated_records = 50
        status.failed_records = 5
        
        status.add_checkpoint("batch_1", "record_123")
        
        assert len(status.checkpoints) == 1
        checkpoint = status.checkpoints[0]
        assert checkpoint["name"] == "batch_1"
        assert checkpoint["record_id"] == "record_123"
        assert checkpoint["migrated"] == 50
        assert checkpoint["failed"] == 5
        assert "timestamp" in checkpoint
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        status = MigrationStatus(
            total_records=100,
            migrated_records=90,
            verified_records=85,
            failed_records=5,
            rollback_records=0,
        )
        
        result = status.to_dict()
        assert result["total_records"] == 100
        assert result["migrated_records"] == 90
        assert result["verified_records"] == 85
        assert result["failed_records"] == 5
        assert result["success_rate"] == 0.9


class TestVectorMigration:
    """Test VectorMigration functionality with real databases."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, source_database, simple_embedding_fn):
        """Test migration initialization."""
        migration = VectorMigration(
            source_db=source_database,
            embedding_fn=simple_embedding_fn,
            model_name="test-model",
            model_version="v1.0",
        )
        
        assert migration.source_db == source_database
        assert migration.target_db == source_database  # In-place by default
        assert migration.embedding_fn == simple_embedding_fn
        assert migration.model_name == "test-model"
        assert migration.model_version == "v1.0"
    
    @pytest.mark.asyncio
    async def test_add_vectors_to_existing(self, source_database, simple_embedding_fn):
        """Test adding vectors to existing records."""
        # Add vector fields to the database schema
        source_database.schema.add_vector_field(
            "content_embedding", 
            dimensions=384, 
            source_field="content"
        )
        source_database.schema.add_vector_field(
            "title_embedding", 
            dimensions=384, 
            source_field="title"
        )
        
        migration = VectorMigration(
            source_db=source_database,
            embedding_fn=simple_embedding_fn,
            config=MigrationConfig(batch_size=3),
            model_name="test-model",
            model_version="v1.0",
        )
        
        vector_fields = {
            "content_embedding": "content",
            "title_embedding": "title",
        }
        
        progress_updates = []
        def progress_callback(status):
            progress_updates.append(status.migrated_records)
        
        status = await migration.add_vectors_to_existing(
            vector_fields=vector_fields,
            progress_callback=progress_callback,
        )
        
        assert status.total_records == 10
        assert status.migrated_records == 10
        assert status.failed_records == 0
        assert status.verified_records == 10  # With verification enabled
        
        # Check that vectors were added
        records = await source_database.all()
        for record in records:
            assert record.get_value("content_embedding") is not None
            assert record.get_value("title_embedding") is not None
            assert len(record.get_value("content_embedding")) == 384
            assert len(record.get_value("title_embedding")) == 384
    
    @pytest.mark.asyncio
    async def test_add_vectors_with_filter(self, source_database, simple_embedding_fn):
        """Test adding vectors to filtered records."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        migration = VectorMigration(
            source_db=source_database,
            embedding_fn=simple_embedding_fn,
        )
        
        # Only migrate category_0 records
        filter_query = {"category": "category_0"}
        
        status = await migration.add_vectors_to_existing(
            vector_fields={"content_embedding": "content"},
            filter_query=filter_query,
        )
        
        # Should only migrate ~3-4 records (category_0)
        assert status.migrated_records < 10
        assert status.failed_records == 0
        
        # Verify only filtered records have vectors
        all_records = await source_database.all()
        for record in all_records:
            if record.get_value("category") == "category_0":
                assert record.get_value("content_embedding") is not None
            else:
                assert record.get_value("content_embedding") is None
    
    @pytest.mark.asyncio
    async def test_error_handling_and_rollback(self, source_database):
        """Test error handling and rollback on failure."""
        # Embedding function that fails after some successes
        call_count = 0
        async def failing_embedding(text):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise Exception("Embedding service failed")
            np.random.seed(sum(ord(c) for c in text[:10]))
            return np.random.rand(384)
        
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        migration = VectorMigration(
            source_db=source_database,
            embedding_fn=failing_embedding,
            config=MigrationConfig(enable_rollback=True, batch_size=2),
        )
        
        with pytest.raises(Exception, match="Embedding service failed"):
            await migration.add_vectors_to_existing(
                vector_fields={"content_embedding": "content"},
            )
        
        # Check rollback occurred
        assert len(migration._rollback_data) > 0
        assert migration._rollback_data  # Should have stored original data
    
    @pytest.mark.asyncio
    async def test_migrate_between_backends(self, source_database, target_database):
        """Test migrating data between different backends."""
        # Add some vectors to source
        for record in await source_database.all():
            embedding = np.random.rand(384)
            record.set_value("content_embedding", embedding.tolist())
            record.set_value("title_embedding", embedding.tolist())
            await source_database.update(record.id, record)
        
        migration = VectorMigration(
            source_db=source_database,
            target_db=target_database,
        )
        
        status = await migration.migrate_between_backends()
        
        assert status.total_records == 10
        assert status.migrated_records == 10
        assert status.failed_records == 0
        
        # Verify all records in target
        target_records = await target_database.all()
        assert len(target_records) == 10
        
        for record in target_records:
            assert "content" in record.data
            assert "title" in record.data
            assert "category" in record.data
    
    @pytest.mark.asyncio
    async def test_migrate_with_field_mapping(self, source_database, target_database):
        """Test migration with field name mapping."""
        migration = VectorMigration(
            source_db=source_database,
            target_db=target_database,
        )
        
        # Map fields with different names
        field_mapping = {
            "content": "content",
            "title": "title",
            "category": "category",
            "metadata": "metadata",
        }
        
        # Transform function to modify records
        def transform_fn(record):
            metadata = record.get_value("metadata") or {}
            metadata["migrated"] = True
            record.set_value("metadata", metadata)
            return record
        
        status = await migration.migrate_between_backends(
            field_mapping=field_mapping,
            transform_fn=transform_fn,
        )
        
        assert status.migrated_records == 10
        
        # Verify transformation was applied
        target_records = await target_database.all()
        for record in target_records:
            metadata = record.get_value("metadata")
            assert metadata and metadata.get("migrated") is True
    
    @pytest.mark.asyncio
    async def test_verification(self, source_database, simple_embedding_fn):
        """Test migration verification."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        migration = VectorMigration(
            source_db=source_database,
            embedding_fn=simple_embedding_fn,
            config=MigrationConfig(verify_migration=True),
        )
        
        status = await migration.add_vectors_to_existing(
            vector_fields={"content_embedding": "content"},
        )
        
        assert status.verified_records == status.migrated_records
        assert status.verified_records == 10


class TestIncrementalVectorizer:
    """Test IncrementalVectorizer functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, source_database, simple_embedding_fn):
        """Test vectorizer initialization."""
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=simple_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
            batch_size=5,
            max_workers=2,
            model_name="test-model",
            model_version="v1.0",
        )
        
        assert vectorizer.database == source_database
        assert vectorizer.vector_field == "content_embedding"
        assert vectorizer.text_fields == ["content"]
        assert vectorizer.batch_size == 5
        assert vectorizer.max_workers == 2
        assert vectorizer.model_name == "test-model"
    
    @pytest.mark.asyncio
    async def test_process_record(self, source_database, simple_embedding_fn):
        """Test processing a single record."""
        # Add vector field
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=simple_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
            model_name="test-model",
            model_version="v1.0",
        )
        
        # Get a record
        records = await source_database.all()
        record = records[0]
        
        # Process it
        await vectorizer._process_record(record)
        
        # Verify vector was added
        updated_record = await source_database.read(record.id)
        assert updated_record.get_value("content_embedding") is not None
        assert len(updated_record.get_value("content_embedding")) == 384
        assert "content_embedding_metadata" in updated_record.data
    
    @pytest.mark.asyncio
    async def test_worker_processing(self, source_database, async_embedding_fn):
        """Test worker task processing."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=async_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
        )
        
        # Add records to queue
        records = await source_database.all()
        for record in records[:3]:
            await vectorizer._queue.put(record)
        
        # Start a worker
        worker_task = asyncio.create_task(vectorizer._worker(0))
        
        # Let it process
        await asyncio.sleep(0.1)
        
        # Stop the worker
        vectorizer._shutdown_event.set()
        await asyncio.wait_for(worker_task, timeout=1.0)
        
        # Check stats
        assert vectorizer._stats["processed"] >= 3
    
    @pytest.mark.asyncio
    async def test_start_stop(self, source_database, simple_embedding_fn):
        """Test starting and stopping vectorization."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=simple_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
            max_workers=2,
        )
        
        # Start vectorization
        await vectorizer.start()
        assert vectorizer._processing_task is not None
        assert len(vectorizer._workers) == 2
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop vectorization
        await vectorizer.stop()
        assert vectorizer._processing_task is None
        assert len(vectorizer._workers) == 0
    
    @pytest.mark.asyncio
    async def test_get_stats(self, source_database, simple_embedding_fn):
        """Test getting vectorization statistics."""
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=simple_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
        )
        
        stats = vectorizer.get_stats()
        assert "processed" in stats
        assert "failed" in stats
        assert "queued" in stats
        assert "queue_size" in stats
        assert "workers" in stats
        assert "is_running" in stats
        assert stats["is_running"] is False
    
    @pytest.mark.asyncio
    async def test_skip_existing_vectors(self, source_database, simple_embedding_fn):
        """Test that existing vectors are skipped."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        # Add vector to first record
        records = await source_database.all()
        first_record = records[0]
        existing_vector = np.random.rand(384).tolist()
        first_record.set_value("content_embedding", existing_vector)
        await source_database.update(first_record.id, first_record)
        
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=simple_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
        )
        
        # Process the record
        updated_first = await source_database.read(first_record.id)
        await vectorizer._process_record(updated_first)
        
        # Vector should remain unchanged (already exists)
        final_record = await source_database.read(first_record.id)
        assert final_record.get_value("content_embedding") == existing_vector
    
    @pytest.mark.asyncio
    async def test_wait_for_completion(self, source_database, async_embedding_fn):
        """Test waiting for queue completion."""
        source_database.schema.add_vector_field(
            "content_embedding",
            dimensions=384,
            source_field="content"
        )
        
        vectorizer = IncrementalVectorizer(
            database=source_database,
            embedding_fn=async_embedding_fn,
            vector_field="content_embedding",
            text_fields="content",
            max_workers=1,
        )
        
        # Add items to queue
        records = await source_database.all()
        for record in records[:3]:
            await vectorizer._queue.put(record)
        
        # Start processing
        await vectorizer.start()
        
        # Wait for completion
        await vectorizer.wait_for_completion(check_interval=0.05)
        
        # Queue should be empty
        assert vectorizer._queue.qsize() == 0
        
        # Stop
        await vectorizer.stop()