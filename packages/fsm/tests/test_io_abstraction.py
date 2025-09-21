"""Tests for I/O abstraction layer."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import Any, List

from dataknobs_fsm.io.base import (
    IOConfig, IOMode, IOFormat, IOProvider,
    AsyncIOProvider, SyncIOProvider
)
from dataknobs_fsm.io.adapters import (
    FileIOAdapter, AsyncFileProvider, SyncFileProvider
)
from dataknobs_fsm.io.utils import (
    create_io_provider, batch_iterator, async_batch_iterator,
    transform_pipeline, async_transform_pipeline,
    IORouter, IOBuffer, IOMetrics, retry_io_operation
)


class TestIOBase:
    """Test base I/O components."""
    
    def test_io_config_creation(self):
        """Test IOConfig creation."""
        config = IOConfig(
            mode=IOMode.READ,
            format=IOFormat.JSON,
            source="/path/to/file.json",
            batch_size=100,
            encoding="utf-8"
        )
        
        assert config.mode == IOMode.READ
        assert config.format == IOFormat.JSON
        assert config.source == "/path/to/file.json"
        assert config.batch_size == 100
        assert config.encoding == "utf-8"
        
    def test_io_modes(self):
        """Test IOMode enum."""
        assert IOMode.READ.value == "read"
        assert IOMode.WRITE.value == "write"
        assert IOMode.APPEND.value == "append"
        assert IOMode.STREAM.value == "stream"
        assert IOMode.BATCH.value == "batch"
        
    def test_io_formats(self):
        """Test IOFormat enum."""
        assert IOFormat.JSON.value == "json"
        assert IOFormat.CSV.value == "csv"
        assert IOFormat.TEXT.value == "text"
        assert IOFormat.DATABASE.value == "database"
        assert IOFormat.API.value == "api"


class TestFileIOAdapter:
    """Test file I/O adapter."""
    
    def test_adapt_config(self):
        """Test configuration adaptation."""
        adapter = FileIOAdapter()
        config = IOConfig(
            mode=IOMode.READ,
            format=IOFormat.JSON,
            source="/test/file.json",
            encoding="utf-8",
            buffer_size=8192
        )
        
        adapted = adapter.adapt_config(config)
        assert adapted['path'] == "/test/file.json"
        assert adapted['mode'] == 'r'
        assert adapted['encoding'] == "utf-8"
        assert adapted['buffering'] == 8192
        
    def test_get_file_mode(self):
        """Test file mode conversion."""
        adapter = FileIOAdapter()
        
        assert adapter._get_file_mode(IOMode.READ) == 'r'
        assert adapter._get_file_mode(IOMode.WRITE) == 'w'
        assert adapter._get_file_mode(IOMode.APPEND) == 'a'
        assert adapter._get_file_mode(IOMode.STREAM) == 'r'
        assert adapter._get_file_mode(IOMode.BATCH) == 'r'
        
    def test_adapt_data_json(self):
        """Test JSON data adaptation."""
        adapter = FileIOAdapter()
        adapter.format = IOFormat.JSON
        
        # Write direction
        data = {"key": "value"}
        adapted = adapter.adapt_data(data, IOMode.WRITE)
        assert adapted == json.dumps(data)
        
        # Read direction
        json_str = '{"key": "value"}'
        adapted = adapter.adapt_data(json_str, IOMode.READ)
        assert adapted == {"key": "value"}


@pytest.mark.asyncio
class TestAsyncFileProvider:
    """Test async file provider."""
    
    async def test_file_operations(self):
        """Test basic file operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            test_data = {"test": "data", "number": 42}
            json.dump(test_data, f)
            temp_path = f.name
            
        try:
            # Test read
            config = IOConfig(
                mode=IOMode.READ,
                format=IOFormat.JSON,
                source=temp_path
            )
            provider = AsyncFileProvider(config)
            
            async with provider:
                # Validate
                assert await provider.validate()
                
                # Read
                data = await provider.read()
                assert "test" in data
                assert data["test"] == "data"
                assert data["number"] == 42
                
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    async def test_stream_read(self):
        """Test streaming read."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"line": 1}\n')
            f.write('{"line": 2}\n')
            f.write('{"line": 3}\n')
            temp_path = f.name
            
        try:
            config = IOConfig(
                mode=IOMode.STREAM,
                format=IOFormat.JSON,
                source=temp_path
            )
            provider = AsyncFileProvider(config)
            
            lines = []
            async with provider:
                async for line in provider.stream_read():
                    lines.append(line)
                    
            assert len(lines) == 3
            assert lines[0]["line"] == 1
            assert lines[1]["line"] == 2
            assert lines[2]["line"] == 3
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    async def test_batch_read(self):
        """Test batch reading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            for i in range(10):
                f.write(f'{{"line": {i}}}\n')
            temp_path = f.name
            
        try:
            config = IOConfig(
                mode=IOMode.BATCH,
                format=IOFormat.JSON,
                source=temp_path,
                batch_size=3
            )
            provider = AsyncFileProvider(config)
            
            batches = []
            async with provider:
                async for batch in provider.batch_read(batch_size=3):
                    batches.append(batch)
                    
            assert len(batches) == 4  # 10 items / 3 per batch = 4 batches
            assert len(batches[0]) == 3
            assert len(batches[1]) == 3
            assert len(batches[2]) == 3
            assert len(batches[3]) == 1
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIOUtils:
    """Test I/O utility functions."""
    
    def test_create_io_provider(self):
        """Test provider creation."""
        # File provider
        config = IOConfig(
            mode=IOMode.READ,
            format=IOFormat.JSON,
            source="/path/to/file.json"
        )
        provider = create_io_provider(config, is_async=True)
        assert isinstance(provider, AsyncFileProvider)
        
        # API provider (HTTP)
        config = IOConfig(
            mode=IOMode.READ,
            format=IOFormat.API,
            source="https://api.example.com/data"
        )
        provider = create_io_provider(config, is_async=True)
        assert provider is not None
        
    def test_batch_iterator(self):
        """Test batch iterator."""
        data = list(range(10))
        batches = list(batch_iterator(iter(data), batch_size=3))
        
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]
        
    @pytest.mark.asyncio
    async def test_async_batch_iterator(self):
        """Test async batch iterator."""
        async def data_generator():
            for i in range(10):
                yield i
                
        batches = []
        async for batch in async_batch_iterator(data_generator(), batch_size=3):
            batches.append(batch)
            
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[3] == [9]
        
    def test_transform_pipeline(self):
        """Test transformation pipeline."""
        def add_one(x):
            return x + 1
            
        def multiply_two(x):
            return x * 2
            
        def to_string(x):
            return str(x)
            
        pipeline = transform_pipeline(add_one, multiply_two, to_string)
        result = pipeline(5)
        
        assert result == "12"  # (5 + 1) * 2 = 12
        
    @pytest.mark.asyncio
    async def test_async_transform_pipeline(self):
        """Test async transformation pipeline."""
        async def add_one(x):
            await asyncio.sleep(0.01)
            return x + 1
            
        def multiply_two(x):
            return x * 2
            
        pipeline = async_transform_pipeline(add_one, multiply_two)
        result = await pipeline(5)
        
        assert result == 12
        
    @pytest.mark.asyncio
    async def test_io_buffer(self):
        """Test I/O buffer."""
        buffer = IOBuffer(max_size=5)
        
        # Add items
        for i in range(3):
            await buffer.add(i)
            
        # Flush
        items = await buffer.flush()
        assert items == [0, 1, 2]
        
        # Buffer should be empty
        items = await buffer.flush()
        assert items == []
        
    def test_io_metrics(self):
        """Test I/O metrics."""
        metrics = IOMetrics()
        
        # Record operations
        metrics.record_read(100)
        metrics.record_read(200)
        metrics.record_write(50)
        metrics.record_error()
        metrics.record_retry()
        
        # Check metrics
        result = metrics.get_metrics()
        assert result['read_count'] == 2
        assert result['write_count'] == 1
        assert result['bytes_read'] == 300
        assert result['bytes_written'] == 50
        assert result['errors'] == 1
        assert result['retries'] == 1
        
        # Reset
        metrics.reset()
        result = metrics.get_metrics()
        assert result['read_count'] == 0
        
    @pytest.mark.asyncio
    async def test_retry_io_operation(self):
        """Test retry operation."""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
            
        result = await retry_io_operation(
            failing_operation,
            max_retries=3,
            delay=0.01,
            exceptions=(ValueError,)
        )
        
        assert result == "success"
        assert call_count == 3
        
    @pytest.mark.asyncio
    async def test_retry_io_operation_failure(self):
        """Test retry operation that fails."""
        call_count = 0
        
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")
            
        with pytest.raises(ValueError, match="Permanent error"):
            await retry_io_operation(
                always_failing,
                max_retries=2,
                delay=0.01,
                exceptions=(ValueError,)
            )
            
        assert call_count == 3  # Initial + 2 retries