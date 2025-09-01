"""Tests for vector synchronization functionality."""

import asyncio
import os
from datetime import datetime

import numpy as np
import pytest

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="Vector sync tests require TEST_POSTGRES=true and a running PostgreSQL instance with pgvector"
)

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema, FieldSchema
from dataknobs_data.vector.sync import (
    SyncConfig,
    SyncStatus,
    VectorTextSynchronizer,
)


@pytest.fixture
async def memory_database():
    """Create an in-memory database for testing."""
    # Create schema for the database
    schema = DatabaseSchema()
    schema.add_field(FieldSchema(name="content", type=FieldType.TEXT))
    schema.add_field(FieldSchema(
        name="embedding", 
        type=FieldType.VECTOR,
        metadata={"dimensions": 384, "source_field": "content"}
    ))
    schema.add_field(FieldSchema(name="title", type=FieldType.TEXT))
    schema.add_field(FieldSchema(
        name="title_embedding",
        type=FieldType.VECTOR,
        metadata={"dimensions": 384, "source_field": "title"}
    ))
    schema.add_field(FieldSchema(name="summary", type=FieldType.TEXT))
    
    # Create database with vector support
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    db.schema = schema
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def simple_embedding_fn():
    """Create a simple deterministic embedding function."""
    def embedding_fn(text: str) -> np.ndarray:
        # Create deterministic embeddings based on text
        if not text:
            return None
        # Use text length and char codes for deterministic output
        np.random.seed(sum(ord(c) for c in text[:10]))
        return np.random.rand(384)
    return embedding_fn


@pytest.fixture
def async_embedding_fn():
    """Create an async embedding function."""
    async def embedding_fn(text: str) -> np.ndarray:
        # Simulate async processing
        await asyncio.sleep(0.001)
        if not text:
            return None
        np.random.seed(sum(ord(c) for c in text[:10]))
        return np.random.rand(384)
    return embedding_fn


class TestSyncConfig:
    """Test SyncConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SyncConfig()
        assert config.auto_embed_on_create is True
        assert config.auto_update_on_text_change is True
        assert config.batch_size == 100
        assert config.track_model_version is True
        assert config.embedding_timeout == 30.0
        assert config.max_retries == 3
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            config = SyncConfig(batch_size=0)
            config.validate()
        
        # Invalid timeout
        with pytest.raises(ValueError, match="Embedding timeout must be positive"):
            config = SyncConfig(embedding_timeout=-1)
            config.validate()
        
        # Invalid retries
        with pytest.raises(ValueError, match="Max retries cannot be negative"):
            config = SyncConfig(max_retries=-1)
            config.validate()


class TestSyncStatus:
    """Test SyncStatus calculations."""
    
    def test_success_rate(self):
        """Test success rate calculation."""
        status = SyncStatus()
        assert status.success_rate == 0.0
        
        status.processed_records = 100
        status.failed_records = 20
        assert status.success_rate == 0.8
    
    def test_duration(self):
        """Test duration calculation."""
        status = SyncStatus()
        assert status.duration is None
        
        status.start_time = datetime(2024, 1, 1, 10, 0, 0)
        status.end_time = datetime(2024, 1, 1, 10, 5, 30)
        assert status.duration == 330.0  # 5 minutes 30 seconds
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        status = SyncStatus(
            total_records=100,
            processed_records=90,
            updated_records=85,
            failed_records=5,
        )
        
        result = status.to_dict()
        assert result["total_records"] == 100
        assert result["processed_records"] == 90
        assert result["updated_records"] == 85
        assert result["failed_records"] == 5
        assert result["success_rate"] == 85/90


class TestVectorTextSynchronizer:
    """Test VectorTextSynchronizer functionality with real database."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, memory_database, simple_embedding_fn):
        """Test synchronizer initialization."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
            model_name="test-model",
            model_version="v1.0",
        )
        
        assert sync.database == memory_database
        assert sync.embedding_fn == simple_embedding_fn
        assert sync.model_name == "test-model"
        assert sync.model_version == "v1.0"
        assert "embedding" in sync._vector_fields
        assert "content" in sync._source_fields
    
    def test_content_hash(self, memory_database, simple_embedding_fn):
        """Test content hash computation."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
        )
        
        hash1 = sync._compute_content_hash("test content")
        hash2 = sync._compute_content_hash("test content")
        hash3 = sync._compute_content_hash("different content")
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_has_current_vector(self, memory_database, simple_embedding_fn):
        """Test vector currency checking with real data."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
            model_version="v1.0",
        )
        
        # Create record without vector
        record_id = await memory_database.create(Record(data={"content": "test content"}))
        record = await memory_database.read(record_id)
        
        # No vector present
        assert sync._has_current_vector(record, "embedding") is False
        
        # Add vector manually with proper content hash
        embedding = simple_embedding_fn("test content")
        content_hash = sync._compute_content_hash("test content")
        # Create a new record with the embedding and content hash
        updated_record = Record(id=record_id, data={
            "content": "test content", 
            "embedding": embedding.tolist(),
            "embedding_content_hash": content_hash
        })
        await memory_database.update(record_id, updated_record)
        record = await memory_database.read(record_id)
        
        # Vector present but no version tracking
        sync.config.track_model_version = False
        assert sync._has_current_vector(record, "embedding") is True
        
        # With version tracking but no metadata
        sync.config.track_model_version = True
        assert sync._has_current_vector(record, "embedding") is False
        
        # Add correct metadata (need to include the embedding too)
        metadata = {
            "model_version": "v1.0",
        }
        await memory_database.update(record_id, Record(id=record_id, data={
            "content": "test content",
            "embedding": embedding.tolist(),
            "embedding_metadata": metadata,
            "embedding_content_hash": content_hash,  # content_hash already computed above
        }))
        record = await memory_database.read(record_id)
        assert sync._has_current_vector(record, "embedding") is True
    
    @pytest.mark.asyncio
    async def test_embed_text(self, memory_database, async_embedding_fn):
        """Test text embedding with async function."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=async_embedding_fn,
        )
        
        # Successful embedding
        result = await sync._embed_text("test text")
        assert isinstance(result, np.ndarray)
        assert len(result) == 384
        
        # Empty text
        result = await sync._embed_text("")
        assert result is None
        
        # Test retry on failure
        fail_count = 0
        async def failing_fn(text):
            nonlocal fail_count
            fail_count += 1
            if fail_count < 2:
                raise Exception("Temporary failure")
            return np.random.rand(384)
        
        sync.embedding_fn = failing_fn
        sync.config.max_retries = 3
        sync.config.retry_delay = 0.01
        
        result = await sync._embed_text("test")
        assert result is not None
        assert fail_count == 2
    
    @pytest.mark.asyncio
    async def test_sync_record(self, memory_database, simple_embedding_fn):
        """Test single record synchronization with real database."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
            model_name="test-model",
            model_version="v1.0",
        )
        
        # Create record
        record_id = await memory_database.create(Record(data={
            "content": "This is test content",
            "title": "Test Title",
            "summary": "No vector for this field",
        }))
        record = await memory_database.read(record_id)
        
        success, updated_fields = await sync.sync_record(record)
        
        assert success is True
        assert "embedding" in updated_fields
        assert "title_embedding" in updated_fields
        
        # Check the vector fields exist and have correct properties
        embedding_field = record.fields.get("embedding")
        assert embedding_field is not None
        assert embedding_field.value is not None
        assert len(embedding_field.value) == 384
        assert embedding_field.metadata.get("model", {}).get("version") == "v1.0"
        
        title_embedding_field = record.fields.get("title_embedding")
        assert title_embedding_field is not None
        assert title_embedding_field.value is not None
        assert len(title_embedding_field.value) == 384
        
        # Summary should not have vector (no vector field defined)
        assert record.get_value("summary_embedding") is None
    
    @pytest.mark.asyncio
    async def test_sync_record_force(self, memory_database, simple_embedding_fn):
        """Test forced record synchronization."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
            model_version="v1.0",
        )
        
        # Create record with existing vector
        from dataknobs_data.fields import VectorField
        existing_vector = [0.1] * 384
        record_id = await memory_database.create(Record(data={
            "content": "Test content",
        }))
        
        # Manually add the vector field to simulate an already-synced record
        record = await memory_database.read(record_id)
        record.fields["embedding"] = VectorField(
            value=existing_vector,
            name="embedding",
            source_field="content",
            model_version="v1.0"
        )
        await memory_database.update(record_id, record)
        record = await memory_database.read(record_id)
        
        # Without force - should skip (vector is current)
        success, updated_fields = await sync.sync_record(record, force=False)
        assert success is True
        assert len(updated_fields) == 0
        
        # With force - should update
        success, updated_fields = await sync.sync_record(record, force=True)
        assert success is True
        assert "embedding" in updated_fields
        
        # Vector should be different from the original
        new_embedding = record.fields.get("embedding")
        assert new_embedding is not None
        # Compare first few values to ensure it's different
        assert not all(new_embedding.value[i] == existing_vector[i] for i in range(5))
    
    @pytest.mark.asyncio
    async def test_bulk_sync(self, memory_database, simple_embedding_fn):
        """Test bulk record synchronization with real database."""
        # Create test records
        record_ids = []
        for i in range(10):
            record_id = await memory_database.create(Record(data={
                "content": f"Content {i}",
                "title": f"Title {i}",
            }))
            record_ids.append(record_id)
        
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
        )
        sync.config.batch_size = 3
        
        status_updates = []
        def progress_callback(status):
            status_updates.append({
                "processed": status.processed_records,
                "updated": status.updated_records,
            })
        
        status = await sync.bulk_sync(progress_callback=progress_callback)
        
        assert status.total_records == 10
        assert status.processed_records == 10
        assert status.updated_records == 10
        assert status.failed_records == 0
        
        # Verify all records have vectors
        for record_id in record_ids:
            record = await memory_database.read(record_id)
            assert record.get_value("embedding") is not None
            assert record.get_value("title_embedding") is not None
            assert len(record.get_value("embedding")) == 384
            assert len(record.get_value("title_embedding")) == 384
    
    @pytest.mark.asyncio
    async def test_sync_on_update(self, memory_database, simple_embedding_fn):
        """Test synchronization on record update."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
        )
        
        # Create initial record
        record_id = await memory_database.create(Record(data={
            "content": "Old content",
            "title": "Old title",
        }))
        
        old_data = {"content": "Old content", "title": "Old title"}
        new_data = {"content": "New content", "title": "Old title"}
        
        # Should trigger sync (content changed)
        result = await sync.sync_on_update(record_id, old_data, new_data)
        assert result is True
        
        # Verify vector was updated
        record = await memory_database.read(record_id)
        assert record.get_value("embedding") is not None
        
        # No change - should not trigger
        result = await sync.sync_on_update(record_id, new_data, new_data)
        assert result is False
        
        # Disabled auto-update
        sync.config.auto_update_on_text_change = False
        newer_data = {"content": "Even newer content", "title": "Old title"}
        result = await sync.sync_on_update(record_id, new_data, newer_data)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_sync_on_create(self, memory_database, simple_embedding_fn):
        """Test synchronization on record creation."""
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=simple_embedding_fn,
        )
        
        record = Record(
            id="test-1",
            data={"content": "New content", "title": "New title"}
        )
        
        # Create the record first
        await memory_database.create(record)
        
        # Should trigger sync
        result = await sync.sync_on_create(record)
        assert result is True
        
        # Verify vectors were added
        updated_record = await memory_database.read(record.id)
        assert updated_record.get_value("embedding") is not None
        assert updated_record.get_value("title_embedding") is not None
        
        # Disabled auto-embed
        sync.config.auto_embed_on_create = False
        record2 = Record(
            id="test-2",
            data={"content": "Another content"}
        )
        await memory_database.create(Record(id=record2.id, data=record2.data))
        result = await sync.sync_on_create(record2)
        assert result is False
        
        # Verify no vectors were added
        record2_data = await memory_database.read(record2.id)
        assert "embedding" not in record2_data.data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_database):
        """Test error handling in synchronization."""
        # Embedding function that always fails
        async def failing_embedding(text):
            raise Exception("Embedding service unavailable")
        
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=failing_embedding,
        )
        sync.config.max_retries = 1
        sync.config.retry_delay = 0.01
        
        record_id = await memory_database.create(Record(data={"content": "Test"}))
        record = await memory_database.read(record_id)
        
        success, updated_fields = await sync.sync_record(record)
        
        assert success is False
        assert len(updated_fields) == 0
    
    @pytest.mark.asyncio
    async def test_bulk_sync_with_errors(self, memory_database):
        """Test bulk sync with some failures."""
        # Create test records
        record_ids = []
        for i in range(5):
            record_id = await memory_database.create(Record(data={"content": f"Content {i}"}))
            record_ids.append(record_id)
        
        # Fail on specific records
        call_count = 0
        async def selective_embedding(text):
            nonlocal call_count
            call_count += 1
            if call_count in [2, 4]:  # Fail on 2nd and 4th calls
                return None
            np.random.seed(sum(ord(c) for c in text[:10]))
            return np.random.rand(384)
        
        sync = VectorTextSynchronizer(
            database=memory_database,
            embedding_fn=selective_embedding,
        )
        
        status = await sync.bulk_sync()
        
        assert status.total_records == 5
        assert status.processed_records == 5
        assert status.updated_records == 3
        assert status.failed_records == 2
        
        # Verify which records have vectors
        success_count = 0
        for record_id in record_ids:
            record = await memory_database.read(record_id)
            if record.get_value("embedding") is not None:
                success_count += 1
        assert success_count == 3
