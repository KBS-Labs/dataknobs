"""Tests for stream execution module using real FSM components."""

import time
from typing import Any, List
import pytest

from dataknobs_fsm.execution.stream import (
    StreamExecutor,
    StreamPipeline,
    StreamProgress
)
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import State
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.streaming.core import StreamChunk, StreamConfig, IStreamSource, IStreamSink
from dataknobs_fsm.functions.base import BaseFunction, FunctionContext
from dataknobs_fsm.functions.manager import FunctionManager


class SimpleStreamSource(IStreamSource):
    """Simple implementation of stream source for testing."""

    def __init__(self, data_chunks: List[List[Any]]):
        """Initialize with list of data chunks."""
        self.data_chunks = data_chunks
        self.current_index = 0
        self.closed = False

    def read_chunk(self) -> StreamChunk | None:
        """Read next chunk."""
        if self.current_index >= len(self.data_chunks):
            return None

        chunk_data = self.data_chunks[self.current_index]
        chunk = StreamChunk(
            data=chunk_data,
            chunk_id=self.current_index,
            is_last=(self.current_index == len(self.data_chunks) - 1)
        )
        self.current_index += 1
        return chunk

    def close(self) -> None:
        """Close the source."""
        self.closed = True


class SimpleStreamSink(IStreamSink):
    """Simple implementation of stream sink for testing."""

    def __init__(self):
        """Initialize sink."""
        self.chunks_received = []
        self.flushed = False
        self.closed = False

    def write_chunk(self, chunk: StreamChunk) -> None:
        """Write chunk to sink."""
        self.chunks_received.append(chunk)

    def flush(self) -> None:
        """Flush the sink."""
        self.flushed = True

    def close(self) -> None:
        """Close the sink."""
        self.closed = True


class ProcessingFunction(BaseFunction):
    """Simple processing function for testing."""

    def __init__(self, multiplier: int = 2):
        """Initialize with multiplier."""
        self.multiplier = multiplier

    def execute(self, context: FunctionContext) -> Any:
        """Process the data."""
        data = context.data
        if isinstance(data, dict) and 'value' in data:
            data['processed_value'] = data['value'] * self.multiplier
            data['processed'] = True
        return data


class ErrorFunction(BaseFunction):
    """Function that raises errors for testing."""

    def __init__(self, error_on_value: Any = None):
        """Initialize with value that triggers error."""
        self.error_on_value = error_on_value

    def execute(self, context: FunctionContext) -> Any:
        """Process data, raising error for specific values."""
        data = context.data
        if isinstance(data, dict) and data.get('value') == self.error_on_value:
            raise ValueError(f"Test error on value {self.error_on_value}")
        return data


class TestStreamProgress:
    """Test suite for StreamProgress class."""

    def test_stream_progress_initialization(self):
        """Test StreamProgress initialization."""
        progress = StreamProgress()
        assert progress.chunks_processed == 0
        assert progress.records_processed == 0
        assert progress.bytes_processed == 0
        assert len(progress.errors) == 0
        assert progress.start_time > 0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        progress = StreamProgress()
        time.sleep(0.1)
        elapsed = progress.elapsed_time
        assert 0.09 < elapsed < 0.2

    def test_processing_rates(self):
        """Test rate calculations."""
        progress = StreamProgress()
        progress.start_time = time.time() - 10.0
        progress.chunks_processed = 100
        progress.records_processed = 1000

        assert 9.5 < progress.chunks_per_second < 10.5
        assert 95 < progress.records_per_second < 105


class TestStreamExecutorReal:
    """Test suite for StreamExecutor with real FSM."""

    def create_test_fsm(self, with_functions: bool = True) -> FSM:
        """Create a real FSM for testing."""
        fsm = FSM(name="test_stream_fsm")

        # Create network
        network = StateNetwork(name="main")

        # Create states
        start_state = State(name="start", type="start")
        process_state = State(name="process", type="normal")
        end_state = State(name="end", type="end")

        # Add states to network
        network.add_state(start_state, initial=True)
        network.add_state(process_state)
        network.add_state(end_state, final=True)

        # Add arcs
        network.add_arc("start", "process")
        network.add_arc("process", "end")

        # Add network to FSM
        fsm.add_network(network)

        # Add function manager with functions if requested
        if with_functions:
            func_manager = FunctionManager()
            func_manager.register_function("process", ProcessingFunction())
            fsm.function_manager = func_manager

        return fsm

    def test_executor_initialization(self):
        """Test StreamExecutor initialization with real FSM."""
        fsm = self.create_test_fsm()
        stream_config = StreamConfig(
            chunk_size=100,
            memory_limit_mb=128,
            backpressure_threshold=0.8
        )

        executor = StreamExecutor(
            fsm=fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        assert executor.fsm == fsm
        assert executor.stream_config == stream_config
        assert executor.enable_backpressure is True
        assert executor._memory_limit == 128 * 1024 * 1024

    def test_execute_stream_basic(self):
        """Test basic stream execution with real components."""
        fsm = self.create_test_fsm()
        executor = StreamExecutor(fsm=fsm)

        # Create real source and sink
        source = SimpleStreamSource([
            [{'id': 1, 'value': 10}, {'id': 2, 'value': 20}],
            [{'id': 3, 'value': 30}, {'id': 4, 'value': 40}]
        ])
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(source=source, sink=sink)

        # Execute stream
        result = executor.execute_stream(pipeline)

        # Verify results
        assert result['total_processed'] == 4
        assert result['chunks_processed'] == 2
        assert result['successful'] == 4
        assert result['failed'] == 0
        assert len(sink.chunks_received) == 2
        assert source.closed is True
        assert sink.closed is True

    def test_execute_stream_with_transformations(self):
        """Test stream execution with transformations."""
        fsm = self.create_test_fsm()
        executor = StreamExecutor(fsm=fsm)

        # Create source with data
        source = SimpleStreamSource([
            [{'id': 1, 'value': 5}],
            [{'id': 2, 'value': 10}]
        ])
        sink = SimpleStreamSink()

        # Define transformations
        def add_timestamp(data):
            data['timestamp'] = time.time()
            return data

        def double_value(data):
            data['value'] = data['value'] * 2
            return data

        pipeline = StreamPipeline(
            source=source,
            sink=sink,
            transformations=[add_timestamp, double_value]
        )

        # Execute
        result = executor.execute_stream(pipeline)

        # Verify transformations were applied
        assert result['total_processed'] == 2
        assert len(sink.chunks_received) == 2

        # Check that transformations modified the data
        first_chunk_data = sink.chunks_received[0].data[0]
        assert 'timestamp' in first_chunk_data
        assert first_chunk_data['value'] == 10  # Original 5 * 2

    def test_execute_stream_with_chunk_processors(self):
        """Test stream execution with chunk processors."""
        fsm = self.create_test_fsm()
        executor = StreamExecutor(fsm=fsm)

        source = SimpleStreamSource([
            [{'id': 1}, {'id': 2}],
            [{'id': 3}, {'id': 4}]
        ])
        sink = SimpleStreamSink()

        # Chunk processor that adds metadata
        def add_chunk_metadata(chunk):
            chunk.metadata['processed_at'] = time.time()
            chunk.metadata['item_count'] = len(chunk.data)
            return chunk

        pipeline = StreamPipeline(
            source=source,
            sink=sink,
            chunk_processors=[add_chunk_metadata]
        )

        result = executor.execute_stream(pipeline)

        # Verify chunk processor was applied
        assert result['chunks_processed'] == 2
        for chunk in sink.chunks_received:
            assert 'processed_at' in chunk.metadata
            assert 'item_count' in chunk.metadata
            assert chunk.metadata['item_count'] == 2

    def test_execute_stream_with_errors(self):
        """Test stream execution with real error handling."""
        # Create FSM with error function
        fsm = self.create_test_fsm(with_functions=False)
        func_manager = FunctionManager()
        func_manager.register_function("error_check", ErrorFunction(error_on_value=20))
        fsm.function_manager = func_manager

        executor = StreamExecutor(fsm=fsm)

        source = SimpleStreamSource([
            [{'id': 1, 'value': 10}],  # OK
            [{'id': 2, 'value': 20}],  # Will error
            [{'id': 3, 'value': 30}]   # OK
        ])
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(source=source, sink=sink)

        # Execute - errors should be caught and processing continues
        result = executor.execute_stream(pipeline)

        # Verify processing continued despite error
        assert result['chunks_processed'] == 3
        assert result['total_processed'] == 3
        # All items processed (errors are caught per-item in _process_chunk)
        assert len(sink.chunks_received) == 3

    def test_execute_stream_with_progress_callback(self):
        """Test stream execution with progress tracking."""
        fsm = self.create_test_fsm()

        progress_updates = []

        def track_progress(progress):
            progress_updates.append({
                'chunks': progress.chunks_processed,
                'records': progress.records_processed,
                'errors': len(progress.errors)
            })

        executor = StreamExecutor(
            fsm=fsm,
            progress_callback=track_progress
        )

        source = SimpleStreamSource([
            [{'id': 1}, {'id': 2}],
            [{'id': 3}, {'id': 4}],
            [{'id': 5}]
        ])
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(source=source, sink=sink)
        result = executor.execute_stream(pipeline)

        # Verify progress was tracked
        assert len(progress_updates) == 3  # One per chunk
        assert progress_updates[-1]['chunks'] == 3
        assert progress_updates[-1]['records'] == 5
        assert result['total_processed'] == 5

    def test_execute_stream_with_custom_context(self):
        """Test stream execution with custom context."""
        fsm = self.create_test_fsm()
        executor = StreamExecutor(fsm=fsm)

        source = SimpleStreamSource([[{'id': 1, 'value': 100}]])
        sink = SimpleStreamSink()

        # Create custom context with specific settings
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=TransactionMode.NONE
        )

        pipeline = StreamPipeline(source=source, sink=sink)

        result = executor.execute_stream(
            pipeline=pipeline,
            context_template=context
        )

        assert result['total_processed'] == 1
        assert result['successful'] == 1

    def test_execute_stream_with_backpressure(self):
        """Test backpressure mechanism with real data."""
        stream_config = StreamConfig(
            memory_limit_mb=1,  # Very low limit
            backpressure_threshold=0.5
        )

        fsm = self.create_test_fsm()
        executor = StreamExecutor(
            fsm=fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        # Set high memory usage
        executor._memory_usage = 600 * 1024  # 600KB

        source = SimpleStreamSource([[{'id': 1}], [{'id': 2}]])
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(source=source, sink=sink)

        # Should still complete despite backpressure
        result = executor.execute_stream(pipeline)

        assert result['total_processed'] == 2
        assert result['chunks_processed'] == 2

    def test_should_apply_backpressure_logic(self):
        """Test backpressure decision logic."""
        stream_config = StreamConfig(
            memory_limit_mb=100,
            backpressure_threshold=0.8
        )

        fsm = self.create_test_fsm()
        executor = StreamExecutor(
            fsm=fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        # Test under threshold
        executor._memory_usage = 50 * 1024 * 1024  # 50MB
        assert executor._should_apply_backpressure() is False

        # Test over memory limit
        executor._memory_usage = 101 * 1024 * 1024  # 101MB
        assert executor._should_apply_backpressure() is True

        # Test disabled
        executor.enable_backpressure = False
        assert executor._should_apply_backpressure() is False

    def test_stream_pipeline_initialization(self):
        """Test StreamPipeline creation."""
        source = SimpleStreamSource([[]])
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(
            source=source,
            sink=sink,
            transformations=[lambda x: x],
            chunk_processors=[lambda c: c]
        )

        assert pipeline.source == source
        assert pipeline.sink == sink
        assert len(pipeline.transformations) == 1
        assert len(pipeline.chunk_processors) == 1

    def test_empty_stream(self):
        """Test execution with empty stream."""
        fsm = self.create_test_fsm()
        executor = StreamExecutor(fsm=fsm)

        source = SimpleStreamSource([])  # Empty
        sink = SimpleStreamSink()

        pipeline = StreamPipeline(source=source, sink=sink)
        result = executor.execute_stream(pipeline)

        assert result['total_processed'] == 0
        assert result['chunks_processed'] == 0
        assert result['successful'] == 0
        assert result['failed'] == 0