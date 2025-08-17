"""Tests for pandas batch operations."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import os

from dataknobs_data.pandas.batch_ops import (
    BatchConfig,
    ChunkedProcessor,
    BatchOperations
)
from dataknobs_data.pandas.converter import DataFrameConverter, ConversionOptions
from dataknobs_data.records import Record
from dataknobs_data.query import Query
from dataknobs_data.backends.memory import SyncMemoryDatabase


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.chunk_size == 1000
        assert config.parallel is False
        assert config.max_workers == 4
        assert config.progress_callback is None
        assert config.error_handling == "raise"
        assert config.memory_efficient is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        callback = Mock()
        config = BatchConfig(
            chunk_size=500,
            parallel=True,
            max_workers=8,
            progress_callback=callback,
            error_handling="skip",
            memory_efficient=False
        )
        assert config.chunk_size == 500
        assert config.parallel is True
        assert config.max_workers == 8
        assert config.progress_callback is callback
        assert config.error_handling == "skip"
        assert config.memory_efficient is False


class TestChunkedProcessor:
    """Tests for ChunkedProcessor class."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = ChunkedProcessor(chunk_size=100)
        assert processor.chunk_size == 100
    
    def test_iter_chunks(self):
        """Test chunking a DataFrame."""
        df = pd.DataFrame({"a": range(250), "b": range(250, 500)})
        processor = ChunkedProcessor(chunk_size=100)
        
        chunks = list(processor.iter_chunks(df))
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50
        
        # Verify data integrity
        combined = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(combined, df)
    
    def test_process_dataframe_without_combine(self):
        """Test processing DataFrame without combining results."""
        df = pd.DataFrame({"value": range(10)})
        processor = ChunkedProcessor(chunk_size=3)
        
        def sum_chunk(chunk):
            return chunk["value"].sum()
        
        results = processor.process_dataframe(df, sum_chunk)
        assert len(results) == 4  # 3 chunks of 3 + 1 chunk of 1
        assert results == [3, 12, 21, 9]  # 0+1+2, 3+4+5, 6+7+8, 9
    
    def test_process_dataframe_with_combine(self):
        """Test processing DataFrame with result combination."""
        df = pd.DataFrame({"value": range(10)})
        processor = ChunkedProcessor(chunk_size=3)
        
        def sum_chunk(chunk):
            return chunk["value"].sum()
        
        def combine_results(results):
            return sum(results)
        
        total = processor.process_dataframe(df, sum_chunk, combine_results)
        assert total == 45  # Sum of 0-9
    
    def test_read_csv_chunked(self):
        """Test reading CSV in chunks."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                "id": range(250),
                "value": np.random.randn(250)
            })
            df.to_csv(f.name, index=False)
            filepath = f.name
        
        try:
            processor = ChunkedProcessor(chunk_size=100)
            
            def process_chunk(chunk):
                return len(chunk)
            
            results = processor.read_csv_chunked(filepath, process_chunk)
            assert len(results) == 3  # 100, 100, 50
            assert sum(results) == 250
            
        finally:
            os.unlink(filepath)
    
    def test_empty_dataframe(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame()
        processor = ChunkedProcessor(chunk_size=10)
        
        chunks = list(processor.iter_chunks(df))
        assert len(chunks) == 0


class TestBatchOperations:
    """Tests for BatchOperations class."""
    
    def test_init_with_sync_database(self):
        """Test initialization with sync database."""
        db = SyncMemoryDatabase()
        converter = DataFrameConverter()
        batch_ops = BatchOperations(db, converter)
        
        assert batch_ops.database is db
        assert batch_ops.converter is converter
        assert batch_ops.is_async is False
    
    def test_init_with_default_converter(self):
        """Test initialization with default converter."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        assert batch_ops.database is db
        assert isinstance(batch_ops.converter, DataFrameConverter)
    
    @patch('dataknobs_data.pandas.batch_ops.asyncio')
    def test_init_with_async_database(self, mock_asyncio):
        """Test initialization with async database."""
        db = Mock()
        db.create = Mock()
        mock_asyncio.iscoroutinefunction.return_value = True
        
        batch_ops = BatchOperations(db)
        assert batch_ops.is_async is True
    
    def test_bulk_insert_dataframe_simple(self):
        """Test simple bulk insert of DataFrame."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        
        stats = batch_ops.bulk_insert_dataframe(df)
        
        assert stats["total_rows"] == 3
        assert stats["inserted"] == 3
        assert stats["failed"] == 0
        assert len(stats["errors"]) == 0
        
        # Verify records in database using search
        all_records = db.search(Query())
        assert len(all_records) == 3
    
    def test_bulk_insert_with_chunking(self):
        """Test bulk insert with chunking."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Create DataFrame larger than chunk size
        df = pd.DataFrame({
            "id": range(250),
            "value": np.random.randn(250)
        })
        
        config = BatchConfig(chunk_size=100, memory_efficient=True)
        stats = batch_ops.bulk_insert_dataframe(df, config)
        
        assert stats["total_rows"] == 250
        assert stats["inserted"] == 250
        assert stats["failed"] == 0
        
        all_records = db.search(Query())
        assert len(all_records) == 250
    
    def test_bulk_insert_with_error_handling_skip(self):
        """Test bulk insert with error handling set to skip."""
        db = Mock()
        db.create = Mock(side_effect=[None, Exception("Error"), None])
        
        batch_ops = BatchOperations(db)
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        config = BatchConfig(error_handling="skip", memory_efficient=False)
        stats = batch_ops.bulk_insert_dataframe(df, config)
        
        assert stats["inserted"] == 2
        assert stats["failed"] == 1
        assert len(stats["errors"]) == 0
    
    def test_bulk_insert_with_error_handling_log(self):
        """Test bulk insert with error handling set to log."""
        db = Mock()
        db.create = Mock(side_effect=[None, Exception("Test error"), None])
        
        batch_ops = BatchOperations(db)
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        config = BatchConfig(error_handling="log", memory_efficient=False)
        
        with patch('dataknobs_data.pandas.batch_ops.logger') as mock_logger:
            stats = batch_ops.bulk_insert_dataframe(df, config)
            
            assert stats["inserted"] == 2
            assert stats["failed"] == 1
            assert len(stats["errors"]) == 1
            assert "Test error" in stats["errors"][0]
            mock_logger.error.assert_called()
    
    def test_bulk_insert_with_error_handling_raise(self):
        """Test bulk insert with error handling set to raise."""
        # Create a custom database that fails on certain values
        class FailingDatabase(SyncMemoryDatabase):
            def create(self, record):
                # Fail on specific value
                if record.get_value("value") == 2:
                    raise ValueError("Test error on value 2")
                return super().create(record)
            
            def create_batch(self, records):
                # Force individual processing by always failing batch
                raise ValueError("Batch creation disabled for testing")
        
        db = FailingDatabase()
        batch_ops = BatchOperations(db)
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        config = BatchConfig(error_handling="raise", memory_efficient=False)
        
        with pytest.raises(ValueError, match="Test error on value 2"):
            batch_ops.bulk_insert_dataframe(df, config)
    
    def test_bulk_insert_with_progress_callback(self):
        """Test bulk insert with progress callback."""
        # Create a database that forces individual record processing
        class IndividualProcessingDB(SyncMemoryDatabase):
            def create_batch(self, records):
                # Force fallback to individual processing
                raise ValueError("Batch disabled to test progress callbacks")
        
        db = IndividualProcessingDB()
        batch_ops = BatchOperations(db)
        
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        df = pd.DataFrame({"value": range(5)})
        config = BatchConfig(
            progress_callback=progress_callback,
            memory_efficient=False
        )
        
        batch_ops.bulk_insert_dataframe(df, config)
        
        # Should get progress callbacks for each individual record
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)
    
    def test_query_as_dataframe(self):
        """Test querying and converting to DataFrame."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert some records
        for i in range(5):
            db.create(Record({"value": i, "type": "test"}))
        
        # Query all records
        query = Query()
        df = batch_ops.query_as_dataframe(query)
        
        assert len(df) == 5
        assert "value" in df.columns
        assert "type" in df.columns
        assert list(df["value"]) == [0, 1, 2, 3, 4]
    
    def test_query_as_dataframe_async(self):
        """Test querying with async database flag set."""
        # For simplicity, test that the is_async flag is properly detected
        db = Mock()
        db.create = Mock()
        
        # Mock asyncio.iscoroutinefunction to return True
        with patch('asyncio.iscoroutinefunction', return_value=True):
            batch_ops = BatchOperations(db)
            assert batch_ops.is_async is True
        
        # Mock asyncio.iscoroutinefunction to return False
        with patch('asyncio.iscoroutinefunction', return_value=False):
            batch_ops = BatchOperations(db)
            assert batch_ops.is_async is False
    
    def test_aggregate_with_groupby(self):
        """Test aggregation with group by."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert test data
        for i in range(10):
            db.create(Record({
                "category": "A" if i < 5 else "B",
                "value": i,
                "count": 1
            }))
        
        query = Query()
        aggregations = {
            "value": "mean",
            "count": "sum"
        }
        
        result = batch_ops.aggregate(query, aggregations, group_by=["category"])
        
        assert len(result) == 2
        assert result.loc["A", "value"] == 2.0  # Mean of 0,1,2,3,4
        assert result.loc["B", "value"] == 7.0  # Mean of 5,6,7,8,9
        assert result.loc["A", "count"] == 5
        assert result.loc["B", "count"] == 5
    
    def test_aggregate_without_groupby(self):
        """Test aggregation without group by."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert test data
        for i in range(5):
            db.create(Record({"value": i}))
        
        query = Query()
        aggregations = {
            "value": "sum"
        }
        
        result = batch_ops.aggregate(query, aggregations)
        
        assert len(result) == 1
        assert result.iloc[0]["value_sum"] == 10  # Sum of 0,1,2,3,4
    
    def test_aggregate_with_custom_function(self):
        """Test aggregation with custom function."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert test data
        for i in range(5):
            db.create(Record({"value": i}))
        
        query = Query()
        
        def custom_agg(series):
            return series.max() - series.min()
        
        aggregations = {
            "value": custom_agg
        }
        
        result = batch_ops.aggregate(query, aggregations)
        
        assert len(result) == 1
        assert result.iloc[0]["value_agg"] == 4  # 4 - 0
    
    def test_aggregate_empty_result(self):
        """Test aggregation with empty query result."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        query = Query().filter("nonexistent", "=", "value")
        aggregations = {"value": "sum"}
        
        result = batch_ops.aggregate(query, aggregations)
        
        assert len(result) == 0
    
    def test_export_to_csv(self):
        """Test exporting query results to CSV."""
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert test data
        for i in range(3):
            db.create(Record({"id": i, "value": i * 10}))
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            query = Query()
            batch_ops.export_to_csv(query, filepath, index=False)
            
            # Read back and verify
            df = pd.read_csv(filepath)
            assert len(df) == 3
            assert list(df["value"]) == [0, 10, 20]
            
        finally:
            os.unlink(filepath)
    
    def test_export_to_parquet(self):
        """Test exporting query results to Parquet."""
        pytest.importorskip("pyarrow")  # Skip test if pyarrow not installed
        
        db = SyncMemoryDatabase()
        batch_ops = BatchOperations(db)
        
        # Insert test data
        for i in range(3):
            db.create(Record({"id": i, "value": i * 10}))
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            filepath = f.name
        
        try:
            query = Query()
            batch_ops.export_to_parquet(query, filepath)
            
            # Read back and verify
            df = pd.read_parquet(filepath)
            assert len(df) == 3
            assert list(df["value"]) == [0, 10, 20]
            
        finally:
            os.unlink(filepath)