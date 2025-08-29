"""Tests for the text-to-vector synchronization example."""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any

# Add examples to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from dataknobs_data import DatabaseFactory, Record, VectorField
from dataknobs_data.vector import VectorTextSynchronizer, ChangeTracker


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def encode(self, text: str) -> List[float]:
        """Generate deterministic fake embeddings."""
        hash_val = hash(text) % 1000
        return [float((hash_val + i) % 100) / 100.0 for i in range(384)]


def mock_generate_embedding(text: str) -> List[float]:
    """Mock embedding generation function."""
    model = MockEmbeddingModel()
    return model.encode(text).tolist()


@pytest.fixture
async def test_db():
    """Create a test database."""
    db = await DatabaseFactory.create_async(
        backend="sqlite",
        database=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    await db.initialize()
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
            "author": "Alice"
        },
        {
            "id": 2,
            "title": "Advanced Python",
            "content": "Deep dive into Python features.",
            "author": "Bob"
        },
        {
            "id": 3,
            "title": "Data Science",
            "content": "Using Python for data analysis.",
            "author": "Carol"
        }
    ]
    
    for doc in documents:
        await test_db.create(Record(doc))
    
    return test_db


class TestVectorTextSynchronization:
    """Test text-to-vector synchronization functionality."""
    
    @pytest.mark.asyncio
    async def test_synchronizer_initialization(self, test_db):
        """Test VectorTextSynchronizer initialization."""
        sync = VectorTextSynchronizer(test_db, mock_generate_embedding)
        
        assert sync.database == test_db
        assert sync.embedding_function == mock_generate_embedding
    
    @pytest.mark.asyncio
    async def test_synchronizer_setup(self, test_db):
        """Test synchronizer setup."""
        sync = VectorTextSynchronizer(test_db, mock_generate_embedding)
        
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding",
            separator=" "
        )
        
        # Setup should configure the synchronizer
        assert hasattr(sync, 'text_fields')
        assert sync.text_fields == ["title", "content"]
        assert sync.vector_field == "embedding"
    
    @pytest.mark.asyncio
    async def test_bulk_sync(self, populated_db):
        """Test bulk synchronization of records."""
        sync = VectorTextSynchronizer(populated_db, mock_generate_embedding)
        
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding",
            separator=" "
        )
        
        # Track progress
        progress_calls = []
        
        def progress_callback(done, total):
            progress_calls.append((done, total))
        
        # Run bulk sync
        results = await sync.bulk_sync(
            batch_size=2,
            progress_callback=progress_callback
        )
        
        # Verify all records were processed
        assert results['processed'] == 3
        assert len(progress_calls) > 0
        
        # Check that records now have embeddings
        all_records = await populated_db.find()
        for record in all_records:
            assert 'embedding' in record
            assert len(record['embedding']) == 384
    
    @pytest.mark.asyncio
    async def test_change_tracker(self, populated_db):
        """Test change tracking functionality."""
        tracker = ChangeTracker(populated_db)
        
        await tracker.start_tracking(
            tracked_fields=["title", "content"],
            vector_field="embedding"
        )
        
        # Initially all records should be outdated (no embeddings)
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 3
        
        # Add embedding to one record
        first_record = outdated[0]
        await populated_db.update(
            first_record['id'],
            {"embedding": VectorField(mock_generate_embedding("test"), dimensions=384)}
        )
        
        # Mark as updated
        await tracker.mark_updated(first_record['id'])
        
        # Now should have 2 outdated records
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 2
    
    @pytest.mark.asyncio
    async def test_sync_single_record(self, populated_db):
        """Test synchronizing a single record."""
        sync = VectorTextSynchronizer(populated_db, mock_generate_embedding)
        
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding",
            separator=" "
        )
        
        # Get first record
        records = await populated_db.find()
        first_record = records[0]
        
        # Sync single record
        await sync.sync_record(first_record['id'])
        
        # Verify embedding was added
        updated = await populated_db.read(first_record['id'])
        assert 'embedding' in updated
        assert len(updated['embedding']) == 384
    
    @pytest.mark.asyncio
    async def test_auto_sync_flag(self, test_db):
        """Test auto-sync enable/disable."""
        sync = VectorTextSynchronizer(test_db, mock_generate_embedding)
        
        # Initially disabled
        assert not sync.is_auto_sync_enabled()
        
        # Enable auto-sync
        sync.enable_auto_sync()
        assert sync.is_auto_sync_enabled()
        
        # Disable auto-sync
        sync.disable_auto_sync()
        assert not sync.is_auto_sync_enabled()
    
    @pytest.mark.asyncio
    async def test_update_detection(self, populated_db):
        """Test detection of updated records."""
        tracker = ChangeTracker(populated_db)
        
        await tracker.start_tracking(
            tracked_fields=["title", "content"],
            vector_field="embedding"
        )
        
        # Add embeddings to all records
        sync = VectorTextSynchronizer(populated_db, mock_generate_embedding)
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding",
            separator=" "
        )
        await sync.bulk_sync()
        
        # No outdated records
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 0
        
        # Update a document's content
        records = await populated_db.find()
        await populated_db.update(
            records[0]['id'],
            {"content": "Updated content that needs new embedding"}
        )
        
        # Should detect the update
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1
        assert outdated[0]['id'] == records[0]['id']
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, populated_db):
        """Test batch processing with different batch sizes."""
        sync = VectorTextSynchronizer(populated_db, mock_generate_embedding)
        
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding",
            separator=" "
        )
        
        # Test with batch_size=1
        results = await sync.bulk_sync(batch_size=1)
        assert results['processed'] == 3
        
        # Clear embeddings
        for record in await populated_db.find():
            await populated_db.update(record['id'], {"embedding": None})
        
        # Test with batch_size=10 (larger than record count)
        results = await sync.bulk_sync(batch_size=10)
        assert results['processed'] == 3


class TestDocumentSyncClass:
    """Test the DocumentSync helper class from the example."""
    
    @pytest.mark.asyncio
    async def test_document_sync_setup(self, test_db):
        """Test DocumentSync class setup."""
        # Import from example
        with patch('text_to_vector_sync.generate_embedding', mock_generate_embedding):
            from text_to_vector_sync import DocumentSync
            
            doc_sync = DocumentSync(test_db)
            await doc_sync.setup()
            
            assert doc_sync.synchronizer is not None
            assert doc_sync.tracker is not None
    
    @pytest.mark.asyncio
    async def test_show_sync_status(self, populated_db):
        """Test sync status display."""
        with patch('text_to_vector_sync.generate_embedding', mock_generate_embedding):
            from text_to_vector_sync import DocumentSync
            
            doc_sync = DocumentSync(populated_db)
            await doc_sync.setup()
            
            outdated = await doc_sync.show_sync_status()
            assert len(outdated) == 3  # All records lack embeddings initially


@pytest.mark.asyncio
async def test_example_workflow():
    """Test the complete synchronization workflow."""
    # Create database
    db = await DatabaseFactory.create_async(
        backend="sqlite",
        database=":memory:",
        vector_enabled=True
    )
    await db.initialize()
    
    try:
        # Create documents without embeddings
        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
            {"title": "Doc 3", "content": "Content 3"}
        ]
        
        record_ids = []
        for doc in docs:
            record_id = await db.create(Record(doc))
            record_ids.append(record_id)
        
        # Setup synchronization
        sync = VectorTextSynchronizer(db, mock_generate_embedding)
        await sync.setup(
            text_fields=["title", "content"],
            vector_field="embedding"
        )
        
        # Bulk sync
        results = await sync.bulk_sync()
        assert results['processed'] == 3
        
        # Verify embeddings
        for record_id in record_ids:
            record = await db.read(record_id)
            assert 'embedding' in record
            assert len(record['embedding']) == 384
        
        # Update a document
        await db.update(record_ids[0], {"content": "Updated content"})
        
        # Track changes
        tracker = ChangeTracker(db)
        await tracker.start_tracking(
            tracked_fields=["title", "content"],
            vector_field="embedding"
        )
        
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 1
        
        # Re-sync outdated record
        await sync.sync_record(outdated[0]['id'])
        
        # Verify no more outdated
        outdated = await tracker.get_outdated_records()
        assert len(outdated) == 0
        
    finally:
        await db.close()