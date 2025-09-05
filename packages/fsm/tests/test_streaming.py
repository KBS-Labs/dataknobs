"""Tests for streaming functionality."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.records import Record

from dataknobs_fsm.streaming import (
    AsyncStreamContext,
    CompressionFormat,
    DatabaseBulkLoader,
    DatabaseStreamSink,
    DatabaseStreamSource,
    DirectoryStreamSource,
    FileFormat,
    FileStreamSink,
    FileStreamSource,
    IStreamSink,
    IStreamSource,
    StreamChunk,
    StreamConfig,
    StreamContext,
    StreamMetrics,
    StreamStatus,
)


class TestStreamCore:
    """Test core streaming functionality."""
    
    def test_stream_config_defaults(self):
        """Test StreamConfig default values."""
        config = StreamConfig()
        assert config.chunk_size == 1000
        assert config.buffer_size == 10000
        assert config.parallelism == 1
        assert config.memory_limit_mb == 512
        assert config.backpressure_threshold == 5000
        assert config.timeout_seconds is None
        assert config.enable_metrics is True
        assert config.retry_on_error is True
        assert config.max_retries == 3
    
    def test_stream_chunk_creation(self):
        """Test StreamChunk creation."""
        data = [1, 2, 3, 4, 5]
        chunk = StreamChunk(
            data=data,
            sequence_number=1,
            metadata={'test': 'value'},
            is_last=False
        )
        
        assert chunk.data == data
        assert chunk.sequence_number == 1
        assert chunk.metadata['test'] == 'value'
        assert chunk.is_last is False
        assert chunk.chunk_id is not None
        assert chunk.timestamp is not None
    
    def test_stream_metrics(self):
        """Test StreamMetrics calculations."""
        metrics = StreamMetrics(
            chunks_processed=10,
            bytes_processed=1024 * 1024 * 10,  # 10 MB
            items_processed=1000,
            start_time=time.time() - 10.0  # 10 seconds ago
        )
        
        duration = metrics.duration_seconds()
        assert duration is not None
        assert 9 <= duration <= 11  # Allow some timing variance
        
        throughput_items = metrics.throughput_items_per_second()
        assert 90 <= throughput_items <= 110  # ~100 items/sec
        
        throughput_mb = metrics.throughput_mb_per_second()
        assert 0.9 <= throughput_mb <= 1.1  # ~1 MB/sec
    
    def test_stream_context_basic(self):
        """Test basic StreamContext operations."""
        context = StreamContext(
            StreamConfig(chunk_size=10, parallelism=2)
        )
        
        assert context.status == StreamStatus.IDLE
        assert context.metrics.chunks_processed == 0
        
        # Add a simple processor
        def double_data(chunk: StreamChunk) -> StreamChunk:
            chunk.data = [x * 2 for x in chunk.data]
            return chunk
        
        context.add_processor(double_data)
        assert len(context._processors) == 1
        
        context.close()
        assert context.status == StreamStatus.COMPLETED


class TestFileStreaming:
    """Test file streaming functionality."""
    
    def test_file_format_detection(self):
        """Test file format detection."""
        assert FileFormat.detect(Path("test.json")) == FileFormat.JSON
        assert FileFormat.detect(Path("test.jsonl")) == FileFormat.JSONL
        assert FileFormat.detect(Path("test.csv")) == FileFormat.CSV
        assert FileFormat.detect(Path("test.txt")) == FileFormat.TEXT
        assert FileFormat.detect(Path("test.dat")) == FileFormat.BINARY
    
    def test_compression_format_detection(self):
        """Test compression format detection."""
        assert CompressionFormat.detect(Path("test.gz")) == CompressionFormat.GZIP
        assert CompressionFormat.detect(Path("test.json.gz")) == CompressionFormat.GZIP
        assert CompressionFormat.detect(Path("test.txt")) == CompressionFormat.NONE
    
    def test_json_file_streaming(self):
        """Test streaming JSON files."""
        # Create test JSON file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            test_data = [
                {'id': 1, 'name': 'Item 1'},
                {'id': 2, 'name': 'Item 2'},
                {'id': 3, 'name': 'Item 3'}
            ]
            json.dump(test_data, f)
            temp_file = Path(f.name)
        
        try:
            # Create source
            source = FileStreamSource(
                temp_file,
                chunk_size=2
            )
            
            # Read chunks
            chunks = list(source)
            assert len(chunks) == 2  # 3 items with chunk_size=2
            
            # Check first chunk
            assert len(chunks[0].data) == 2
            assert chunks[0].data[0]['id'] == 1
            assert chunks[0].data[1]['id'] == 2
            assert not chunks[0].is_last
            
            # Check second chunk
            assert len(chunks[1].data) == 1
            assert chunks[1].data[0]['id'] == 3
            assert chunks[1].is_last
            
            source.close()
            
        finally:
            temp_file.unlink()
    
    def test_jsonl_file_streaming(self):
        """Test streaming JSONL files."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.jsonl',
            delete=False
        ) as f:
            f.write('{"id": 1, "value": "a"}\n')
            f.write('{"id": 2, "value": "b"}\n')
            f.write('{"id": 3, "value": "c"}\n')
            temp_file = Path(f.name)
        
        try:
            source = FileStreamSource(temp_file, chunk_size=2)
            chunks = list(source)
            
            assert len(chunks) == 2
            assert len(chunks[0].data) == 2
            assert chunks[0].data[0]['id'] == 1
            assert chunks[0].data[1]['id'] == 2
            assert len(chunks[1].data) == 1
            assert chunks[1].data[0]['id'] == 3
            
            source.close()
            
        finally:
            temp_file.unlink()
    
    def test_csv_file_streaming(self):
        """Test streaming CSV files."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            f.write("id,name,value\n")
            f.write("1,Item1,100\n")
            f.write("2,Item2,200\n")
            f.write("3,Item3,300\n")
            temp_file = Path(f.name)
        
        try:
            source = FileStreamSource(temp_file, chunk_size=2)
            chunks = list(source)
            
            assert len(chunks) == 2
            assert chunks[0].data[0]['id'] == '1'
            assert chunks[0].data[0]['name'] == 'Item1'
            assert chunks[0].data[0]['value'] == '100'
            
            source.close()
            
        finally:
            temp_file.unlink()
    
    def test_file_sink_json(self):
        """Test writing to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.json"
            
            sink = FileStreamSink(output_file)
            
            # Write chunks
            chunk1 = StreamChunk(
                data=[{'id': 1}, {'id': 2}],
                sequence_number=0
            )
            chunk2 = StreamChunk(
                data=[{'id': 3}],
                sequence_number=1,
                is_last=True
            )
            
            assert sink.write_chunk(chunk1)
            assert sink.write_chunk(chunk2)
            
            sink.close()
            
            # Verify output
            with open(output_file) as f:
                result = json.load(f)
            
            assert len(result) == 3
            assert result[0]['id'] == 1
            assert result[1]['id'] == 2
            assert result[2]['id'] == 3
    
    def test_file_sink_jsonl(self):
        """Test writing to JSONL file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.jsonl"
            
            sink = FileStreamSink(output_file)
            
            chunk = StreamChunk(
                data=[{'id': 1}, {'id': 2}, {'id': 3}],
                is_last=True
            )
            
            assert sink.write_chunk(chunk)
            sink.close()
            
            # Verify output
            with open(output_file) as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            assert json.loads(lines[0])['id'] == 1
            assert json.loads(lines[1])['id'] == 2
            assert json.loads(lines[2])['id'] == 3
    
    def test_file_sink_csv(self):
        """Test writing to CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.csv"
            
            sink = FileStreamSink(output_file)
            
            chunk = StreamChunk(
                data=[
                    {'id': 1, 'name': 'Item1'},
                    {'id': 2, 'name': 'Item2'}
                ],
                is_last=True
            )
            
            assert sink.write_chunk(chunk)
            sink.close()
            
            # Verify output
            with open(output_file) as f:
                lines = f.readlines()
            
            assert len(lines) == 3  # Header + 2 data rows
            assert "id,name" in lines[0]
            assert "1,Item1" in lines[1]
            assert "2,Item2" in lines[2]
    
    def test_directory_stream_source(self):
        """Test streaming from directory of files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "file1.json").write_text(json.dumps([1, 2, 3]))
            (temp_path / "file2.json").write_text(json.dumps([4, 5, 6]))
            (temp_path / "other.txt").write_text("ignored")
            
            # Create directory source
            source = DirectoryStreamSource(
                temp_path,
                pattern="*.json",
                chunk_size=2
            )
            
            assert len(source.files) == 2
            
            # Read all chunks
            chunks = list(source)
            
            # Should have chunks from both files
            assert len(chunks) >= 2
            
            # Verify metadata
            assert 'source_file' in chunks[0].metadata
            assert 'file_index' in chunks[0].metadata
            assert 'total_files' in chunks[0].metadata
            
            source.close()


class TestDatabaseStreaming:
    """Test database streaming functionality."""
    
    def test_database_stream_source(self):
        """Test streaming from database."""
        # Create test database
        factory = DatabaseFactory()
        db = factory.create(backend="memory")
        
        # Add test records
        for i in range(5):
            record = Record(id=str(i), data={'value': i})
            db.create(record)
        
        # Create stream source
        source = DatabaseStreamSource(
            database=db,
            batch_size=2
        )
        
        # Read chunks
        chunks = list(source)
        
        # Should have 3 chunks (5 records with batch_size=2)
        assert len(chunks) == 3
        
        # Check first chunk
        assert len(chunks[0].data) == 2
        assert chunks[0].data[0]['_id'] == '0'
        assert chunks[0].data[1]['_id'] == '1'
        assert not chunks[0].is_last
        
        # Check last chunk
        assert len(chunks[2].data) == 1
        assert chunks[2].data[0]['_id'] == '4'
        assert chunks[2].is_last
        
        source.close()
    
    def test_database_stream_sink(self):
        """Test streaming to database."""
        # Create test database
        factory = DatabaseFactory()
        db = factory.create(backend="memory")
        
        # Create sink
        sink = DatabaseStreamSink(
            database=db,
            batch_size=2
        )
        
        # Write chunks
        chunk1 = StreamChunk(
            data=[
                {'id': '1', 'value': 'a'},
                {'id': '2', 'value': 'b'}
            ]
        )
        chunk2 = StreamChunk(
            data=[{'id': '3', 'value': 'c'}],
            is_last=True
        )
        
        assert sink.write_chunk(chunk1)
        assert sink.write_chunk(chunk2)
        
        sink.close()
        
        # Verify records were created
        assert db.count() == 3
        assert db.read('1').value == 'a'
        assert db.read('2').value == 'b'
        assert db.read('3').value == 'c'
    
    def test_database_bulk_loader(self):
        """Test bulk loading into database."""
        # Create source data
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.jsonl',
            delete=False
        ) as f:
            for i in range(10):
                f.write(json.dumps({'id': str(i), 'value': i}) + '\n')
            temp_file = Path(f.name)
        
        try:
            # Create database and loader
            factory = DatabaseFactory()
            db = factory.create(backend="memory")
            loader = DatabaseBulkLoader(db)
            
            # Create file source
            source = FileStreamSource(temp_file, chunk_size=3)
            
            # Load data
            stats = loader.load_from_source(source, batch_size=3)
            
            # Check statistics
            assert stats['records_loaded'] == 10
            assert stats['errors'] == 0
            assert stats['start_time'] is not None
            assert stats['end_time'] is not None
            
            # Verify database content
            assert db.count() == 10
            for i in range(10):
                record = db.read(str(i))
                assert record is not None
                assert record.value == i
                
        finally:
            temp_file.unlink()
    
    def test_database_export(self):
        """Test exporting from database."""
        # Create database with test data
        factory = DatabaseFactory()
        db = factory.create(backend="memory")
        
        for i in range(5):
            record = Record(id=str(i), data={'value': i * 10})
            db.create(record)
        
        # Create file sink
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "export.json"
            sink = FileStreamSink(output_file)
            
            # Export data
            loader = DatabaseBulkLoader(db)
            stats = loader.export_to_sink(sink, batch_size=2)
            
            # Check statistics
            assert stats['records_loaded'] == 5
            assert stats['errors'] == 0
            
            # Verify exported file
            with open(output_file) as f:
                result = json.load(f)
            
            assert len(result) == 5
            for i in range(5):
                assert result[i]['_id'] == str(i)
                assert result[i]['value'] == i * 10


class TestEndToEndStreaming:
    """Test end-to-end streaming scenarios."""
    
    def test_file_to_file_streaming(self):
        """Test streaming from file to file with transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input file
            input_file = temp_path / "input.jsonl"
            with open(input_file, 'w') as f:
                for i in range(10):
                    f.write(json.dumps({'id': i, 'value': i}) + '\n')
            
            # Create source and sink
            source = FileStreamSource(input_file, chunk_size=3)
            sink = FileStreamSink(temp_path / "output.jsonl")
            
            # Create stream context with transformation
            context = StreamContext(StreamConfig(chunk_size=3))
            
            def transform(data):
                """Double the value."""
                if isinstance(data, list):
                    return [{'id': d['id'], 'value': d['value'] * 2} for d in data]
                return data
            
            # Stream with transformation
            metrics = context.stream(source, sink, transform)
            
            # Check metrics
            assert metrics.chunks_processed > 0
            assert metrics.items_processed == 10
            
            # Verify output
            with open(temp_path / "output.jsonl") as f:
                lines = f.readlines()
            
            assert len(lines) == 10
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data['id'] == i
                assert data['value'] == i * 2
    
    def test_file_to_database_streaming(self):
        """Test streaming from file to database."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            f.write("id,name,score\n")
            for i in range(5):
                f.write(f"{i},User{i},{i*100}\n")
            temp_file = Path(f.name)
        
        try:
            # Create database
            factory = DatabaseFactory()
            db = factory.create(backend="memory")
            
            # Create source and sink
            source = FileStreamSource(temp_file, chunk_size=2)
            sink = DatabaseStreamSink(db, batch_size=2)
            
            # Stream data
            context = StreamContext()
            metrics = context.stream(source, sink)
            
            # Check results
            assert metrics.chunks_processed > 0
            assert db.count() == 5
            
            # Verify data
            for i in range(5):
                record = db.read(str(i))
                assert record is not None
                assert record.name == f"User{i}"
                assert record.score == str(i * 100)
                
        finally:
            temp_file.unlink()
    
    def test_database_to_file_streaming(self):
        """Test streaming from database to file."""
        # Create database with test data
        factory = DatabaseFactory()
        db = factory.create(backend="memory")
        
        for i in range(5):
            record = Record(
                id=str(i),
                data={'value': i * 10}
            )
            db.create(record)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "export.json"
            
            # Create source and sink
            source = DatabaseStreamSource(db, batch_size=2)
            sink = FileStreamSink(output_file)
            
            # Stream data
            context = StreamContext()
            metrics = context.stream(source, sink)
            
            # Check results
            assert metrics.chunks_processed > 0
            assert metrics.items_processed == 5
            
            # Verify file content
            with open(output_file) as f:
                result = json.load(f)
            
            assert len(result) == 5
            for i in range(5):
                assert result[i]['_id'] == str(i)
                assert result[i]['value'] == i * 10