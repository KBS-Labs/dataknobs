"""Tests for stream execution module."""

import time
from unittest.mock import Mock, patch
import pytest

from dataknobs_fsm.execution.stream import (
    StreamExecutor,
    StreamPipeline,
    StreamProgress
)
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.streaming.core import StreamChunk, StreamConfig


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
        assert progress.last_chunk_time > 0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        progress = StreamProgress()
        progress.start_time = time.time() - 5.0  # 5 seconds ago

        elapsed = progress.elapsed_time
        assert 4.9 < elapsed < 5.1  # Allow for small timing variations

    def test_chunks_per_second_zero_elapsed(self):
        """Test chunks per second with zero elapsed time."""
        progress = StreamProgress()
        progress.start_time = time.time()
        progress.chunks_processed = 10

        # When elapsed time is effectively 0, should return 0
        with patch('time.time', return_value=progress.start_time):
            assert progress.chunks_per_second == 0.0

    def test_chunks_per_second_normal(self):
        """Test chunks per second calculation."""
        progress = StreamProgress()
        progress.start_time = time.time() - 10.0  # 10 seconds ago
        progress.chunks_processed = 100

        cps = progress.chunks_per_second
        assert 9.9 < cps < 10.1  # Should be ~10 chunks/second

    def test_records_per_second_zero_elapsed(self):
        """Test records per second with zero elapsed time."""
        progress = StreamProgress()
        progress.start_time = time.time()
        progress.records_processed = 50

        with patch('time.time', return_value=progress.start_time):
            assert progress.records_per_second == 0.0

    def test_records_per_second_normal(self):
        """Test records per second calculation."""
        progress = StreamProgress()
        progress.start_time = time.time() - 5.0  # 5 seconds ago
        progress.records_processed = 250

        rps = progress.records_per_second
        assert 49.0 < rps < 51.0  # Should be ~50 records/second

    def test_error_tracking(self):
        """Test error tracking in progress."""
        progress = StreamProgress()

        # Add some errors
        error1 = ValueError("Test error 1")
        error2 = RuntimeError("Test error 2")

        progress.errors.append((1, error1))
        progress.errors.append((5, error2))

        assert len(progress.errors) == 2
        assert progress.errors[0] == (1, error1)
        assert progress.errors[1] == (5, error2)


class TestStreamPipeline:
    """Test suite for StreamPipeline class."""

    def test_pipeline_initialization(self):
        """Test StreamPipeline initialization."""
        mock_source = Mock()
        mock_sink = Mock()

        pipeline = StreamPipeline(
            source=mock_source,
            sink=mock_sink
        )

        assert pipeline.source == mock_source
        assert pipeline.sink == mock_sink
        assert len(pipeline.transformations) == 0
        assert len(pipeline.chunk_processors) == 0

    def test_pipeline_with_transformations(self):
        """Test pipeline with transformations."""
        mock_source = Mock()
        transform1 = lambda x: x * 2
        transform2 = lambda x: x + 1

        pipeline = StreamPipeline(
            source=mock_source,
            transformations=[transform1, transform2]
        )

        assert len(pipeline.transformations) == 2
        assert pipeline.transformations[0] == transform1
        assert pipeline.transformations[1] == transform2


class TestStreamExecutor:
    """Test suite for StreamExecutor class."""

    @pytest.fixture
    def simple_fsm(self):
        """Create a simple FSM for testing."""
        from dataknobs_fsm.core.state import State

        # Create network
        network = StateNetwork(name="main")

        # Create and add states
        start_state = State(name="start", type="start")
        end_state = State(name="end", type="end")

        network.add_state(start_state, initial=True)
        network.add_state(end_state, final=True)

        # Add arc from start to end
        network.add_arc("start", "end")

        # Create FSM
        fsm = FSM(name="test_fsm")
        fsm.add_network(network)

        return fsm

    @pytest.fixture
    def stream_config(self):
        """Create a stream configuration."""
        return StreamConfig(
            chunk_size=100,
            parallelism=2,
            memory_limit_mb=128,
            backpressure_threshold=0.8
        )

    def test_executor_initialization(self, simple_fsm, stream_config):
        """Test StreamExecutor initialization."""
        executor = StreamExecutor(
            fsm=simple_fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        assert executor.fsm == simple_fsm
        assert executor.stream_config == stream_config
        assert executor.enable_backpressure is True
        assert executor._memory_limit == 128 * 1024 * 1024
        assert executor._backpressure_threshold == 0.8

    def test_executor_default_config(self, simple_fsm):
        """Test executor with default configuration."""
        executor = StreamExecutor(fsm=simple_fsm)

        assert executor.stream_config is not None
        assert isinstance(executor.stream_config, StreamConfig)
        assert executor.enable_backpressure is True

    def test_executor_with_progress_callback(self, simple_fsm):
        """Test executor with progress callback."""
        callback = Mock()
        executor = StreamExecutor(
            fsm=simple_fsm,
            progress_callback=callback
        )

        assert executor.progress_callback == callback

    def test_execute_stream_basic(self, simple_fsm, stream_config):
        """Test basic stream execution."""
        executor = StreamExecutor(fsm=simple_fsm, stream_config=stream_config)

        # Mock pipeline
        mock_source = Mock()
        mock_sink = Mock()

        # Configure source to return chunks then None
        chunk1 = StreamChunk(data=[1, 2, 3], chunk_id=0)
        chunk2 = StreamChunk(data=[4, 5, 6], chunk_id=1)
        mock_source.read_chunk.side_effect = [chunk1, chunk2, None]

        pipeline = StreamPipeline(source=mock_source, sink=mock_sink)

        # Mock engine execution
        with patch.object(executor.engine, 'execute', return_value=(True, "result")):
            result = executor.execute_stream(pipeline)

        # Verify results
        assert 'total_processed' in result
        assert 'successful' in result
        assert result['chunks_processed'] == 2
        assert result['total_processed'] == 6

    def test_execute_stream_with_transformations(self, simple_fsm):
        """Test stream execution with transformations."""
        executor = StreamExecutor(fsm=simple_fsm)

        # Mock pipeline with transformations
        mock_source = Mock()
        mock_sink = Mock()

        # Transformation functions
        transform_calls = []
        def transform1(data):
            transform_calls.append(('transform1', data))
            return data * 2

        def transform2(data):
            transform_calls.append(('transform2', data))
            return data + 1

        pipeline = StreamPipeline(
            source=mock_source,
            sink=mock_sink,
            transformations=[transform1, transform2]
        )

        # Configure source
        chunk = StreamChunk(data=[1, 2], chunk_id=0)
        mock_source.read_chunk.side_effect = [chunk, None]

        # Mock engine execution
        with patch.object(executor.engine, 'execute', return_value=(True, "result")):
            result = executor.execute_stream(pipeline)

        # Verify transformations were applied
        assert len(transform_calls) == 4  # 2 transforms x 2 records

    def test_execute_stream_with_chunk_processors(self, simple_fsm):
        """Test stream execution with chunk processors."""
        executor = StreamExecutor(fsm=simple_fsm)

        mock_source = Mock()
        processor_calls = []

        def chunk_processor(chunk):
            processor_calls.append(chunk.chunk_id)
            return chunk

        pipeline = StreamPipeline(
            source=mock_source,
            chunk_processors=[chunk_processor]
        )

        # Configure source
        chunk1 = StreamChunk(data=[1, 2], chunk_id=0)
        chunk2 = StreamChunk(data=[3, 4], chunk_id=1)
        mock_source.read_chunk.side_effect = [chunk1, chunk2, None]

        with patch.object(executor.engine, 'execute', return_value=(True, "result")):
            executor.execute_stream(pipeline)

        # Verify chunk processors were called
        assert processor_calls == [0, 1]

    def test_execute_stream_with_progress_callback(self, simple_fsm):
        """Test stream execution with progress callback."""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append({
                'chunks': progress.chunks_processed,
                'records': progress.records_processed
            })

        executor = StreamExecutor(
            fsm=simple_fsm,
            progress_callback=progress_callback
        )

        mock_source = Mock()
        mock_source.read_chunk.side_effect = [
            StreamChunk(data=[1, 2], chunk_id=0),
            StreamChunk(data=[3, 4], chunk_id=1),
            None
        ]

        pipeline = StreamPipeline(source=mock_source)

        with patch.object(executor.engine, 'execute', return_value=(True, "result")):
            executor.execute_stream(pipeline)

        # Verify progress was reported
        assert len(progress_updates) > 0
        # Verify progress was reported
        # Just check that we got some updates
        assert len(progress_updates) >= 2  # At least 2 chunks worth of updates

    def test_should_apply_backpressure_disabled(self, simple_fsm):
        """Test backpressure check when disabled."""
        executor = StreamExecutor(
            fsm=simple_fsm,
            enable_backpressure=False
        )

        # Even with high memory usage, should not apply backpressure
        executor._memory_usage = 999999999
        assert executor._should_apply_backpressure() is False

    def test_should_apply_backpressure_under_threshold(self, simple_fsm):
        """Test backpressure check under threshold."""
        stream_config = StreamConfig(
            memory_limit_mb=100,
            backpressure_threshold=0.8
        )

        executor = StreamExecutor(
            fsm=simple_fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        # Under threshold
        executor._memory_usage = 50 * 1024 * 1024  # 50 MB
        assert executor._should_apply_backpressure() is False

    def test_should_apply_backpressure_over_threshold(self, simple_fsm):
        """Test backpressure check over threshold."""
        stream_config = StreamConfig(
            memory_limit_mb=100,
            backpressure_threshold=0.8
        )

        executor = StreamExecutor(
            fsm=simple_fsm,
            stream_config=stream_config,
            enable_backpressure=True
        )

        # Over memory limit
        executor._memory_usage = 101 * 1024 * 1024  # 101 MB (over 100MB limit)
        assert executor._should_apply_backpressure() is True
