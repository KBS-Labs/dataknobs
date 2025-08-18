"""Test unified batch processing improvements."""

import pytest
from typing import List
from dataknobs_data import (
    Record, StreamConfig, StreamResult, StreamProcessor,
    SyncDatabase, AsyncDatabase
)
from dataknobs_data.backends.memory import SyncMemoryDatabase, AsyncMemoryDatabase
from dataknobs_data.streaming import process_batch_with_fallback, async_process_batch_with_fallback


class TestStreamResultEnhancements:
    """Test enhanced StreamResult with total_batches and failed_indices."""
    
    def test_stream_result_new_properties(self):
        """Test that StreamResult has new tracking properties."""
        result = StreamResult()
        
        # Verify new properties exist and have correct defaults
        assert hasattr(result, 'total_batches')
        assert hasattr(result, 'failed_indices')
        assert result.total_batches == 0
        assert result.failed_indices == []
    
    def test_add_error_with_index(self):
        """Test adding errors with indices."""
        result = StreamResult()
        
        # Add error without index
        result.add_error("record_1", ValueError("Invalid value"))
        assert len(result.errors) == 1
        assert result.errors[0]["index"] is None
        assert result.failed_indices == []
        
        # Add error with index
        result.add_error("record_2", TypeError("Wrong type"), index=5)
        assert len(result.errors) == 2
        assert result.errors[1]["index"] == 5
        assert result.failed_indices == [5]
        
        # Add another error with index
        result.add_error("record_3", RuntimeError("Runtime error"), index=10)
        assert len(result.errors) == 3
        assert result.failed_indices == [5, 10]
    
    def test_merge_with_new_properties(self):
        """Test merging results includes new properties."""
        result1 = StreamResult()
        result1.total_processed = 100
        result1.successful = 95
        result1.failed = 5
        result1.total_batches = 10
        result1.failed_indices = [3, 7, 15]
        
        result2 = StreamResult()
        result2.total_processed = 50
        result2.successful = 48
        result2.failed = 2
        result2.total_batches = 5
        result2.failed_indices = [25, 42]
        
        result1.merge(result2)
        
        assert result1.total_processed == 150
        assert result1.successful == 143
        assert result1.failed == 7
        assert result1.total_batches == 15
        assert result1.failed_indices == [3, 7, 15, 25, 42]


class TestBatchProcessingWithIndices:
    """Test batch processing functions track indices correctly."""
    
    def test_process_batch_with_fallback_tracks_batches(self):
        """Test that batch processing tracks batch counts."""
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(5)
        ]
        
        config = StreamConfig(batch_size=10)
        result = StreamResult()
        
        # Successful batch creation
        def batch_create(recs):
            return [r.id for r in recs]
        
        def single_create(rec):
            return rec.id
        
        success = process_batch_with_fallback(
            records,
            batch_create,
            single_create,
            result,
            config,
            batch_index=0
        )
        
        assert success
        assert result.total_batches == 1
        assert result.successful == 5
        assert result.failed == 0
        assert result.failed_indices == []
    
    def test_process_batch_fallback_tracks_failed_indices(self):
        """Test that failed record indices are tracked correctly."""
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(5)
        ]
        
        config = StreamConfig(batch_size=5)
        result = StreamResult()
        
        # Batch creation that always fails
        def batch_create(recs):
            raise ValueError("Batch operation failed")
        
        # Single creation that fails for specific records
        def single_create(rec):
            if rec.id in ["rec_1", "rec_3"]:
                raise ValueError(f"Failed to create {rec.id}")
            return rec.id
        
        # Error handler that continues processing
        config.on_error = lambda exc, rec: True
        
        success = process_batch_with_fallback(
            records,
            batch_create,
            single_create,
            result,
            config,
            batch_index=2  # Third batch (index 2)
        )
        
        assert success
        assert result.total_batches == 1
        assert result.successful == 3  # rec_0, rec_2, rec_4
        assert result.failed == 2  # rec_1, rec_3
        
        # Check failed indices (batch_index=2, batch_size=5)
        # rec_1 is at position 1 in batch, so global index = 2*5 + 1 = 11
        # rec_3 is at position 3 in batch, so global index = 2*5 + 3 = 13
        assert 11 in result.failed_indices
        assert 13 in result.failed_indices


class TestStreamProcessorAdapters:
    """Test list to async iterator adapters."""
    
    def test_list_to_iterator(self):
        """Test converting list to iterator."""
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(3)
        ]
        
        iterator = StreamProcessor.list_to_iterator(records)
        
        # Verify it's an iterator
        assert hasattr(iterator, '__iter__')
        assert hasattr(iterator, '__next__')
        
        # Consume iterator
        result = list(iterator)
        assert len(result) == 3
        assert all(isinstance(r, Record) for r in result)
        assert result[0].id == "rec_0"
    
    @pytest.mark.asyncio
    async def test_list_to_async_iterator(self):
        """Test converting list to async iterator."""
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(3)
        ]
        
        async_iter = StreamProcessor.list_to_async_iterator(records)
        
        # Verify it's an async iterator
        assert hasattr(async_iter, '__aiter__')
        assert hasattr(async_iter, '__anext__')
        
        # Consume async iterator
        result = []
        async for record in async_iter:
            result.append(record)
        
        assert len(result) == 3
        assert all(isinstance(r, Record) for r in result)
        assert result[0].id == "rec_0"
    
    @pytest.mark.asyncio
    async def test_iterator_to_async_iterator(self):
        """Test converting sync iterator to async iterator."""
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(3)
        ]
        
        sync_iter = iter(records)
        async_iter = StreamProcessor.iterator_to_async_iterator(sync_iter)
        
        # Consume async iterator
        result = []
        async for record in async_iter:
            result.append(record)
        
        assert len(result) == 3
        assert all(isinstance(r, Record) for r in result)


class TestIntegrationWithDatabases:
    """Test that databases use the enhanced streaming correctly."""
    
    def test_sync_database_tracks_batches(self):
        """Test sync database streaming tracks batch information."""
        db = SyncMemoryDatabase()
        
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(25)
        ]
        
        config = StreamConfig(batch_size=10)
        
        # Convert list to iterator
        record_iter = StreamProcessor.list_to_iterator(records)
        result = db.stream_write(record_iter, config)
        
        assert result.total_processed == 25
        assert result.successful == 25
        assert result.failed == 0
        assert result.total_batches == 3  # 10 + 10 + 5
        assert result.failed_indices == []
    
    @pytest.mark.asyncio
    async def test_async_database_tracks_batches(self):
        """Test async database streaming tracks batch information."""
        db = AsyncMemoryDatabase()
        
        records = [
            Record(id=f"rec_{i}", data={"value": i})
            for i in range(25)
        ]
        
        config = StreamConfig(batch_size=10)
        
        # Convert list to async iterator
        async_iter = StreamProcessor.list_to_async_iterator(records)
        result = await db.stream_write(async_iter, config)
        
        assert result.total_processed == 25
        assert result.successful == 25
        assert result.failed == 0
        assert result.total_batches == 3  # 10 + 10 + 5
        assert result.failed_indices == []


class TestBatchConfigCompatibility:
    """Test that BatchConfig and StreamConfig can be used appropriately."""
    
    def test_configs_have_similar_properties(self):
        """Test that both configs have similar batch size properties."""
        from dataknobs_data import StreamConfig
        from dataknobs_data.pandas import BatchConfig
        
        stream_config = StreamConfig(batch_size=500)
        batch_config = BatchConfig(chunk_size=500)
        
        # Both have size configuration
        assert stream_config.batch_size == 500
        assert batch_config.chunk_size == 500
        
        # Both have error handling
        assert hasattr(stream_config, 'on_error')
        assert hasattr(batch_config, 'error_handling')
    
    def test_usage_documentation_exists(self):
        """Test that documentation exists for when to use each config."""
        import os
        docs_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "BATCH_PROCESSING_GUIDE.md"
        )
        
        assert os.path.exists(docs_path)
        
        with open(docs_path) as f:
            content = f.read()
            
        # Verify key sections exist
        assert "When to Use Each Configuration" in content
        assert "StreamConfig" in content
        assert "BatchConfig" in content
        assert "Key Differences" in content
        assert "Conversion Utilities" in content