"""Comprehensive tests for stream mode execution."""

import asyncio
import pytest
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig, 
    StateConfig, 
    ArcConfig, 
    NetworkConfig,
    FunctionReference
)
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.streaming.core import StreamContext, StreamConfig, StreamChunk


class TestStreamMode:
    """Test stream mode execution with real chunks."""
    
    def create_stream_fsm_config(self) -> FSMConfig:
        """Create an FSM configuration for stream processing."""
        # Create a network that processes stream data
        network = NetworkConfig(
            name="stream_network",
            states=[
                StateConfig(
                    name="start",
                    is_start=True,
                    arcs=[
                        ArcConfig(
                            target="process",
                            priority=1
                        )
                    ]
                ),
                StateConfig(
                    name="process",
                    functions={
                        "transform": FunctionReference(
                            type="inline",
                            code="data['processed'] = True; data"
                        )
                    },
                    arcs=[
                        ArcConfig(
                            target="end",
                            priority=1
                        )
                    ]
                ),
                StateConfig(
                    name="end",
                    is_end=True
                )
            ]
        )
        
        return FSMConfig(
            name="stream_fsm",
            description="Stream processing FSM",
            networks=[network],
            main_network="stream_network"
        )
    
    @pytest.mark.asyncio
    async def test_stream_mode_single_chunk(self):
        """Test stream mode with a single chunk."""
        config = self.create_stream_fsm_config()
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Execute with initial data as a single chunk
        result = await fsm.execute_async({"value": 42})
        
        assert result["status"] == "completed"
        assert result["data"]["chunks_processed"] == 1
        assert result["data"]["records_processed"] == 1
        assert result["data"]["errors"] == []
    
    @pytest.mark.asyncio
    async def test_stream_mode_multiple_chunks(self):
        """Test stream mode with multiple chunks."""
        config = self.create_stream_fsm_config()
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create execution context with stream
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        # Create stream context and add multiple chunks
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Add multiple chunks with different data
        test_data = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"},
            {"id": 4, "value": "fourth"},
            {"id": 5, "value": "fifth"}
        ]
        
        for i, data in enumerate(test_data):
            is_last = (i == len(test_data) - 1)
            context.stream_context.add_data(data, chunk_id=f"chunk_{i}", is_last=is_last)
        
        # Execute the stream
        engine = fsm.get_async_engine()
        success, result = await engine.execute(context)
        
        assert success
        assert result["chunks_processed"] == len(test_data)
        assert result["records_processed"] == len(test_data)
        assert result["errors"] == []
    
    @pytest.mark.asyncio
    async def test_stream_mode_with_batch_chunks(self):
        """Test stream mode with chunks containing multiple records."""
        config = self.create_stream_fsm_config()
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create execution context
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        # Create stream context
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Add chunks with multiple records each
        chunk1_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        chunk2_data = [{"id": 4}, {"id": 5}]
        chunk3_data = [{"id": 6}, {"id": 7}, {"id": 8}, {"id": 9}]
        
        context.stream_context.add_chunk(StreamChunk(
            data=chunk1_data,
            chunk_id="batch_1",
            is_last=False
        ))
        
        context.stream_context.add_chunk(StreamChunk(
            data=chunk2_data,
            chunk_id="batch_2",
            is_last=False
        ))
        
        context.stream_context.add_chunk(StreamChunk(
            data=chunk3_data,
            chunk_id="batch_3",
            is_last=True
        ))
        
        # Execute the stream
        engine = fsm.get_async_engine()
        success, result = await engine.execute(context)
        
        assert success
        assert result["chunks_processed"] == 3
        total_records = len(chunk1_data) + len(chunk2_data) + len(chunk3_data)
        assert result["records_processed"] == total_records
        assert result["errors"] == []
    
    @pytest.mark.asyncio
    async def test_stream_mode_empty_stream(self):
        """Test stream mode with no chunks."""
        config = self.create_stream_fsm_config()
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create execution context with empty stream
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Don't add any chunks
        
        # Execute the empty stream
        engine = fsm.get_async_engine()
        success, result = await engine.execute(context)
        
        assert success
        assert result["chunks_processed"] == 0
        assert result["records_processed"] == 0
        assert result["errors"] == []
    
    @pytest.mark.asyncio
    async def test_stream_mode_with_errors(self):
        """Test stream mode with processing errors."""
        # Create FSM with a failing condition
        network = NetworkConfig(
            name="failing_stream",
            states=[
                StateConfig(
                    name="start",
                    is_start=True,
                    arcs=[
                        ArcConfig(
                            target="process",
                            # Condition that fails for even IDs
                            condition=FunctionReference(
                                type="inline",
                                code="data.get('id', 0) % 2 != 0"
                            )
                        )
                    ]
                ),
                StateConfig(
                    name="process",
                    arcs=[
                        ArcConfig(target="end", priority=1)
                    ]
                ),
                StateConfig(name="end", is_end=True)
            ]
        )
        
        config = FSMConfig(
            name="failing_stream_fsm",
            networks=[network],
            main_network="failing_stream"
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create execution context
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Add data with both odd and even IDs
        test_data = [
            {"id": 1},  # Will succeed
            {"id": 2},  # Will fail
            {"id": 3},  # Will succeed
            {"id": 4},  # Will fail
        ]
        
        for i, data in enumerate(test_data):
            is_last = (i == len(test_data) - 1)
            context.stream_context.add_data(data, is_last=is_last)
        
        # Execute the stream
        engine = fsm.get_async_engine()
        success, result = await engine.execute(context)
        
        # Some records failed, so overall success should be False
        assert not success
        assert result["chunks_processed"] == len(test_data)
        assert result["records_processed"] == len(test_data)
        # Should have 2 errors (for IDs 2 and 4)
        assert len(result["errors"]) == 2
    
    @pytest.mark.asyncio
    async def test_stream_mode_performance(self):
        """Test stream mode with many chunks for performance."""
        config = self.create_stream_fsm_config()
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create execution context
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Add many chunks
        num_chunks = 100
        for i in range(num_chunks):
            is_last = (i == num_chunks - 1)
            context.stream_context.add_data(
                {"chunk_id": i, "data": f"chunk_{i}"}, 
                is_last=is_last
            )
        
        # Execute the stream
        engine = fsm.get_async_engine()
        success, result = await engine.execute(context)
        
        assert success
        assert result["chunks_processed"] == num_chunks
        assert result["records_processed"] == num_chunks
        assert result["errors"] == []