"""Tests for streaming infrastructure extensions.

This module tests the completed loose end implementations for streaming:
- BasicStreamProcessor (lines 545-630 in streaming/core.py)
- MemoryStreamSource (lines 633-675 in streaming/core.py)
- MemoryStreamSink (lines 678-704 in streaming/core.py)
- AsyncStreamContext stream_async method (lines 466-542 in streaming/core.py)
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Any, Dict

from dataknobs_fsm.streaming.core import (
    BasicStreamProcessor, MemoryStreamSource, MemoryStreamSink,
    AsyncStreamContext, StreamConfig, StreamChunk, StreamStatus, StreamMetrics,
    IStreamSource, IStreamSink
)


class TestBasicStreamProcessor:
    """Test the BasicStreamProcessor class."""
    
    def test_processor_initialization(self):
        """Test BasicStreamProcessor initialization."""
        source = MemoryStreamSource([1, 2, 3])
        sink = MemoryStreamSink()
        
        processor = BasicStreamProcessor(source, sink)
        
        assert processor.source == source
        assert processor.sink == sink
        assert processor.transform_func is None
        assert processor.buffer_size == 1000
        assert processor.processed_chunks == 0
        assert processor.processed_records == 0
        assert processor.errors == []
        
    def test_processor_with_transform(self):
        """Test processor with transformation function."""
        source = MemoryStreamSource([1, 2, 3, 4, 5], chunk_size=2)
        sink = MemoryStreamSink()
        
        # Transform function to double values
        def transform(chunk):
            chunk.data = [x * 2 for x in chunk.data]
            return chunk
        
        processor = BasicStreamProcessor(source, sink, transform_func=transform)
        stats = processor.process()
        
        # Verify results
        assert stats['success'] is True
        assert stats['processed_chunks'] == 3  # 5 items in chunks of 2
        assert stats['processed_records'] == 5
        assert len(stats['errors']) == 0
        
        # Verify transformed data
        assert sink.records == [2, 4, 6, 8, 10]
        
    def test_processor_basic_workflow(self):
        """Test basic stream processing workflow."""
        data = list(range(10))
        source = MemoryStreamSource(data, chunk_size=3)
        sink = MemoryStreamSink()
        
        processor = BasicStreamProcessor(source, sink)
        stats = processor.process()
        
        # Verify statistics
        assert stats['success'] is True
        assert stats['processed_chunks'] == 4  # 10 items in chunks of 3
        assert stats['processed_records'] == 10
        assert stats['duration'] > 0
        
        # Verify data was written correctly
        assert sink.records == data
        
    def test_processor_error_handling(self):
        """Test error handling in stream processing."""
        source = MemoryStreamSource([1, 2, 3], chunk_size=1)  # 3 chunks
        sink = Mock()
        sink.write_chunk = Mock(side_effect=[True, False, True])  # Fail on second chunk
        sink.flush = Mock()
        
        processor = BasicStreamProcessor(source, sink)
        stats = processor.process()
        
        # Should continue processing despite one failure
        assert stats['success'] is False
        assert stats['processed_chunks'] == 2  # Only 2 successful (1st and 3rd)
        assert len(stats['errors']) == 1
        assert "Failed to write chunk 1" in stats['errors'][0]  # Failed on chunk index 1
        
    def test_processor_transform_exception(self):
        """Test exception handling in transform function."""
        source = MemoryStreamSource([1, 2, 3], chunk_size=1)  # 3 separate chunks
        sink = MemoryStreamSink()
        
        def failing_transform(chunk):
            if chunk.data[0] == 2:
                raise ValueError("Transform error")
            return chunk
        
        processor = BasicStreamProcessor(source, sink, transform_func=failing_transform)
        stats = processor.process()
        
        # Should continue processing despite transform error
        assert stats['success'] is False
        assert len(stats['errors']) == 1
        assert "Error processing chunk" in stats['errors'][0]
        # Should have processed chunks 1 and 3 successfully
        assert stats['processed_chunks'] == 2
        assert sink.records == [1, 3]  # Only 1 and 3 should be in sink
        
    async def test_processor_async(self):
        """Test async processing method."""
        source = MemoryStreamSource([1, 2, 3])
        sink = MemoryStreamSink()
        
        processor = BasicStreamProcessor(source, sink)
        stats = await processor.process_async()
        
        assert stats['success'] is True
        assert stats['processed_chunks'] == 1
        assert sink.records == [1, 2, 3]


class TestMemoryStreamSource:
    """Test the MemoryStreamSource class."""
    
    def test_source_initialization(self):
        """Test MemoryStreamSource initialization."""
        data = [1, 2, 3, 4, 5]
        source = MemoryStreamSource(data, chunk_size=2)
        
        assert source.data == data
        assert source.chunk_size == 2
        assert source.current_index == 0
        
    def test_source_read_chunks(self):
        """Test reading chunks from source."""
        data = list(range(10))
        source = MemoryStreamSource(data, chunk_size=3)
        
        chunks = []
        chunk = source.read_chunk()
        while chunk is not None:
            chunks.append(chunk)
            chunk = source.read_chunk()
        
        # Should have 4 chunks (10 items / 3 per chunk)
        assert len(chunks) == 4
        assert chunks[0].data == [0, 1, 2]
        assert chunks[1].data == [3, 4, 5]
        assert chunks[2].data == [6, 7, 8]
        assert chunks[3].data == [9]
        assert chunks[3].is_last is True
        
    def test_source_iteration(self):
        """Test iterating over source chunks."""
        data = list(range(5))
        source = MemoryStreamSource(data, chunk_size=2)
        
        chunks = list(source)
        
        assert len(chunks) == 3
        assert all(isinstance(c, StreamChunk) for c in chunks)
        assert chunks[-1].is_last is True
        
    def test_source_empty_data(self):
        """Test source with empty data."""
        source = MemoryStreamSource([])
        
        chunk = source.read_chunk()
        assert chunk is None
        
        chunks = list(source)
        assert chunks == []
        
    def test_source_single_item(self):
        """Test source with single item."""
        source = MemoryStreamSource([42])
        
        chunk = source.read_chunk()
        assert chunk.data == [42]
        assert chunk.is_last is True
        
        # Second read should return None
        assert source.read_chunk() is None
        
    def test_source_chunk_metadata(self):
        """Test chunk metadata generation."""
        source = MemoryStreamSource([1, 2, 3], chunk_size=1)
        
        chunks = list(source)
        
        # Verify chunk IDs
        assert chunks[0].chunk_id == "chunk_0"
        assert chunks[1].chunk_id == "chunk_1"
        assert chunks[2].chunk_id == "chunk_2"
        
        # Verify timestamps
        assert all(c.timestamp > 0 for c in chunks)


class TestMemoryStreamSink:
    """Test the MemoryStreamSink class."""
    
    def test_sink_initialization(self):
        """Test MemoryStreamSink initialization."""
        sink = MemoryStreamSink()
        
        assert sink.chunks == []
        assert sink.records == []
        
    def test_sink_write_chunk(self):
        """Test writing chunks to sink."""
        sink = MemoryStreamSink()
        
        chunk1 = StreamChunk(data=[1, 2, 3], chunk_id="1")
        chunk2 = StreamChunk(data=[4, 5], chunk_id="2")
        
        assert sink.write_chunk(chunk1) is True
        assert sink.write_chunk(chunk2) is True
        
        assert len(sink.chunks) == 2
        assert sink.records == [1, 2, 3, 4, 5]
        
    def test_sink_write_non_iterable(self):
        """Test writing non-iterable data."""
        sink = MemoryStreamSink()
        
        chunk = StreamChunk(data=42, chunk_id="1")
        assert sink.write_chunk(chunk) is True
        
        assert len(sink.chunks) == 1
        assert sink.records == [42]
        
    def test_sink_write_failure(self):
        """Test write failure handling."""
        sink = MemoryStreamSink()
        
        # Create a problematic chunk that will cause an exception
        # when trying to extend records
        chunk = Mock(spec=StreamChunk)
        
        # Create a data object that looks iterable but fails when extended
        class FailingIterable:
            def __iter__(self):
                raise Exception("Iteration error")
        
        chunk.data = FailingIterable()
        
        result = sink.write_chunk(chunk)
        
        assert result is False
        # The chunk gets added before the error, but that's the implementation
        # The important thing is that write_chunk returns False
        assert len(sink.chunks) == 1  # Chunk was added before error
        assert sink.records == []  # But records weren't extended due to error
        
    def test_sink_flush_and_close(self):
        """Test flush and close operations (no-ops for memory)."""
        sink = MemoryStreamSink()
        
        # These should not raise errors
        sink.flush()
        sink.close()
        
        # Add data and verify flush doesn't clear it
        chunk = StreamChunk(data=[1, 2], chunk_id="1")
        sink.write_chunk(chunk)
        sink.flush()
        
        assert sink.records == [1, 2]


class TestAsyncStreamContext:
    """Test the AsyncStreamContext class stream_async method."""
    
    @pytest.mark.asyncio
    async def test_context_basic_processing(self):
        """Test basic stream processing with AsyncStreamContext."""
        config = StreamConfig(
            buffer_size=10,
            parallelism=2,
            chunk_size=2  # Use chunk_size instead of batch_size
        )
        
        context = AsyncStreamContext(config)
        
        # Create source and sink
        data = list(range(10))
        chunks = [
            StreamChunk(data=data[i:i+3], chunk_id=f"chunk_{i}", 
                       is_last=(i+3 >= len(data)))
            for i in range(0, len(data), 3)
        ]
        
        async def async_source():
            for chunk in chunks:
                yield chunk
        
        collected = []
        def sink_func(chunk):
            collected.append(chunk)
            return True
        
        # Process stream
        metrics = await context.stream_async(async_source(), sink_func)
        
        # Verify metrics
        assert metrics.chunks_processed == len(chunks)
        assert metrics.errors_count == 0
        assert metrics.start_time > 0
        assert metrics.end_time > metrics.start_time
        
        # Verify all chunks were processed
        assert len(collected) == len(chunks)
        
    @pytest.mark.asyncio
    async def test_context_with_transform(self):
        """Test stream processing with transformation."""
        config = StreamConfig(parallelism=1)
        context = AsyncStreamContext(config)
        
        # Create source
        async def async_source():
            yield StreamChunk(data=[1, 2, 3], chunk_id="1")
            yield StreamChunk(data=[4, 5, 6], chunk_id="2", is_last=True)
        
        collected = []
        def sink_func(chunk):
            collected.append(chunk)
            return True
        
        # Transform to double values
        def transform(data):
            return [x * 2 for x in data]
        
        # Process with transform
        metrics = await context.stream_async(async_source(), sink_func, transform=transform)
        
        assert metrics.chunks_processed == 2
        assert collected[0].data == [2, 4, 6]
        assert collected[1].data == [8, 10, 12]
        
    @pytest.mark.asyncio
    async def test_context_error_handling(self):
        """Test error handling in stream context."""
        config = StreamConfig(parallelism=1)
        context = AsyncStreamContext(config)
        
        # Create source that fails
        async def failing_source():
            yield StreamChunk(data=[1], chunk_id="1")
            raise ValueError("Source error")
        
        def sink_func(chunk):
            return True
        
        # Process should handle error
        metrics = await context.stream_async(failing_source(), sink_func)
        
        assert context.status == StreamStatus.ERROR
        
    @pytest.mark.asyncio
    async def test_context_sink_failure(self):
        """Test handling of sink failures."""
        config = StreamConfig(parallelism=1)
        context = AsyncStreamContext(config)
        
        async def async_source():
            yield StreamChunk(data=[1], chunk_id="1")
            yield StreamChunk(data=[2], chunk_id="2", is_last=True)
        
        call_count = 0
        def failing_sink(chunk):
            nonlocal call_count
            call_count += 1
            return False  # Always fail
        
        metrics = await context.stream_async(async_source(), failing_sink)
        
        assert metrics.chunks_processed == 2
        assert metrics.errors_count == 2
        
    @pytest.mark.asyncio
    async def test_context_parallel_processing(self):
        """Test parallel processing with multiple workers."""
        config = StreamConfig(parallelism=3)
        context = AsyncStreamContext(config)
        
        # Create source with many chunks
        async def async_source():
            for i in range(10):
                yield StreamChunk(
                    data=[i], 
                    chunk_id=f"chunk_{i}",
                    is_last=(i == 9)
                )
        
        collected = []
        def sink_func(chunk):
            collected.append(chunk.data[0])
            return True
        
        metrics = await context.stream_async(async_source(), sink_func)
        
        assert metrics.chunks_processed == 10
        # Order might vary due to parallel processing
        assert sorted(collected) == list(range(10))
        
    @pytest.mark.asyncio
    async def test_context_stop_event(self):
        """Test stopping stream processing."""
        config = StreamConfig(parallelism=1)
        context = AsyncStreamContext(config)
        
        # Create infinite source
        async def infinite_source():
            i = 0
            while True:
                yield StreamChunk(data=[i], chunk_id=f"chunk_{i}")
                i += 1
                await asyncio.sleep(0.01)
        
        collected = []
        def sink_func(chunk):
            collected.append(chunk)
            if len(collected) >= 3:
                context._stop_event.set()
            return True
        
        # Process should stop after collecting 3 chunks
        metrics = await context.stream_async(infinite_source(), sink_func)
        
        # May collect a few more due to buffering
        assert len(collected) >= 3
        assert len(collected) < 10  # Should not run forever


class TestStreamLifecycle:
    """Test stream lifecycle management."""
    
    def test_source_close(self):
        """Test closing stream source."""
        source = MemoryStreamSource([1, 2, 3])
        
        # Close should not raise
        source.close()
        
        # Should still be able to read after close (for memory source)
        chunk = source.read_chunk()
        assert chunk is not None
        
    def test_sink_lifecycle(self):
        """Test sink lifecycle operations."""
        sink = MemoryStreamSink()
        
        # Write some data
        chunk = StreamChunk(data=[1, 2], chunk_id="1")
        sink.write_chunk(chunk)
        
        # Flush should not clear data
        sink.flush()
        assert sink.records == [1, 2]
        
        # Close should not clear data
        sink.close()
        assert sink.records == [1, 2]
        
    def test_processor_with_custom_buffer_size(self):
        """Test processor with custom buffer size."""
        source = MemoryStreamSource(list(range(100)), chunk_size=10)
        sink = MemoryStreamSink()
        
        processor = BasicStreamProcessor(source, sink, buffer_size=50)
        stats = processor.process()
        
        assert stats['success'] is True
        assert processor.buffer_size == 50
        assert len(sink.records) == 100


class TestBufferManagement:
    """Test buffer management and overflow handling."""
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_handling(self):
        """Test handling of buffer overflow."""
        config = StreamConfig(
            buffer_size=2,  # Very small buffer
            parallelism=1
        )
        
        context = AsyncStreamContext(config)
        
        # Create source with many chunks
        async def async_source():
            for i in range(10):
                yield StreamChunk(
                    data=[i], 
                    chunk_id=f"chunk_{i}",
                    is_last=(i == 9)
                )
        
        collected = []
        def slow_sink(chunk):
            time.sleep(0.01)  # Simulate slow processing
            collected.append(chunk.data[0])
            return True
        
        # Should handle buffer constraints
        metrics = await context.stream_async(async_source(), slow_sink)
        
        assert metrics.chunks_processed == 10
        assert sorted(collected) == list(range(10))
        
    def test_empty_stream_handling(self):
        """Test handling of empty streams."""
        source = MemoryStreamSource([])
        sink = MemoryStreamSink()
        
        processor = BasicStreamProcessor(source, sink)
        stats = processor.process()
        
        assert stats['success'] is True
        assert stats['processed_chunks'] == 0
        assert stats['processed_records'] == 0
        assert sink.records == []