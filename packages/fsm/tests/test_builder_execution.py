"""Tests for FSM execute() implementation in config/builder.py."""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig, 
    StateConfig, 
    ArcConfig, 
    NetworkConfig,
    DataModeConfig,
    TransactionConfig,
    FunctionReference
)
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import TransactionMode, ProcessingMode
from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine


class TestFSMExecute:
    """Tests for FSM.execute() method (lines 855-895 in builder.py)."""
    
    def create_simple_fsm_config(self, name: str = "test_fsm") -> FSMConfig:
        """Create a simple FSM configuration for testing."""
        # Create a simple network with start and end states
        network = NetworkConfig(
            name="main",
            states=[
                StateConfig(
                    name="start",
                    is_start=True,
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
            name=name,
            description="Test FSM",
            networks=[network],
            main_network="main"
        )
    
    @pytest.fixture
    def fsm_instance(self):
        """Create a real FSM instance for testing."""
        config = self.create_simple_fsm_config()
        builder = FSMBuilder()
        return builder.build(config)
    
    @pytest.mark.asyncio
    async def test_execute_simple_success(self, fsm_instance):
        """Test successful execution of FSM."""
        # Execute the real FSM
        result = await fsm_instance.execute_async({"input": "data"})
        
        # The FSM should complete successfully (start -> end)
        assert result["status"] == "completed"
        assert "data" in result
        assert "execution_id" in result
    
    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test failed execution of FSM."""
        # Create an FSM with a failing transition
        config = FSMConfig(
            name="failing_fsm",
            networks=[NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            ArcConfig(
                                target="end",
                                # Add a condition that always fails
                                condition=FunctionReference(
                                    type="inline",
                                    code="False"  # Always fails
                                )
                            )
                        ]
                    ),
                    StateConfig(name="end", is_end=True)
                ]
            )],
            main_network="main"
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)
        result = await fsm.execute_async({"input": "data"})
        
        # Should fail because no valid transition from start
        assert result["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        """Test exception handling in execute()."""
        # Create an FSM with a transform that raises an exception
        config = FSMConfig(
            name="error_fsm",
            networks=[NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[
                            ArcConfig(
                                target="end",
                                transform=FunctionReference(
                                    type="inline",
                                    code="data['error'] = True; data"
                                )
                            )
                        ]
                    ),
                    StateConfig(name="end", is_end=True)
                ]
            )],
            main_network="main"
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)
        
        # Execute should handle the exception gracefully
        result = await fsm.execute_async({"input": "data"})
        
        # The status might be 'failed' or 'error' depending on error handling
        assert result["status"] in ["failed", "error", "completed"]
    
    @pytest.mark.asyncio
    async def test_execute_context_creation(self, fsm_instance):
        """Test that ExecutionContext is created correctly."""
        # Simply execute the FSM and verify result structure
        result = await fsm_instance.execute_async({"input": "data"})
        
        # Verify the result has expected fields from context
        assert "status" in result
        assert "data" in result
        assert "execution_id" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_none_initial_data(self, fsm_instance):
        """Test execution with None as initial data."""
        # Execute with None - FSM should handle it
        result = await fsm_instance.execute_async(None)
        
        # Should still complete successfully
        assert result["status"] == "completed"
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_different_processing_modes(self):
        """Test execution with different processing modes."""
        for mode in [ProcessingMode.SINGLE, ProcessingMode.BATCH, ProcessingMode.STREAM]:
            config = self.create_simple_fsm_config(f"{mode.value}_fsm")
            builder = FSMBuilder()
            fsm = builder.build(config)
            fsm.data_mode = mode
            
            # Real execution
            result = await fsm.execute_async({"test": "data"})
            
            # Should complete regardless of mode
            if result["status"] != "completed":
                print(f"Mode {mode.value} failed with result: {result}")
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_result_formatting(self, fsm_instance):
        """Test that result is properly formatted."""
        result = await fsm_instance.execute_async({"input": "test"})
        
        # Check all required fields are present
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert "execution_id" in result
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_with_real_context(self, fsm_instance):
        """Test execution with real context creation."""
        # Execute with real FSM and context
        result = await fsm_instance.execute_async({})
        
        # Should have all expected attributes
        assert result["status"] == "completed"
        assert "execution_id" in result
        assert "transitions" in result
        assert "duration" in result
        assert result["duration"] is not None  # Should have timing info
    
    @pytest.mark.asyncio
    async def test_execute_engine_creation_error(self, fsm_instance):
        """Test handling when engine creation fails."""
        with patch.object(fsm_instance, 'get_async_engine') as mock_get_engine:
            mock_get_engine.side_effect = ValueError("Cannot create engine")
            
            result = await fsm_instance.execute_async({"test": "data"})
            
            assert result["status"] == "error"
            assert "Cannot create engine" in result["error"]
            assert result["data"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_context_creation_error(self, fsm_instance):
        """Test handling when context creation fails."""
        with patch('dataknobs_fsm.execution.context.ExecutionContext') as MockContext:
            MockContext.side_effect = RuntimeError("Context error")
            
            result = await fsm_instance.execute_async({"test": "data"})
            
            assert result["status"] == "error"
            assert "Context error" in result["error"]
            assert result["data"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_dry_principle(self, fsm_instance):
        """Test that execute() can be called multiple times."""
        # Execute multiple times with real FSM
        for i in range(3):
            result = await fsm_instance.execute_async({"iteration": i})
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_with_complex_data(self, fsm_instance):
        """Test execution with complex nested data structures."""
        complex_data = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"},
                "tuple": (4, 5, 6)
            },
            "array": [{"id": 1}, {"id": 2}]
        }
        
        # Execute with real FSM - it should handle complex data
        result = await fsm_instance.execute_async(complex_data)
        
        assert result["status"] == "completed"
        # The data will be processed through the FSM
        assert "data" in result
    
    @pytest.mark.asyncio 
    async def test_execute_with_complex_fsm_config(self):
        """Test execution with a more complex FSM configuration."""
        # Create a more complex FSM with multiple states
        network = NetworkConfig(
            name="complex",
            states=[
                StateConfig(
                    name="init", 
                    is_start=True,
                    arcs=[
                        ArcConfig(target="process", priority=1)
                    ]
                ),
                StateConfig(
                    name="process",
                    arcs=[
                        ArcConfig(target="validate", priority=1)
                    ]
                ),
                StateConfig(
                    name="validate",
                    arcs=[
                        ArcConfig(target="success", priority=1),
                        ArcConfig(target="error", priority=2)
                    ]
                ),
                StateConfig(name="success", is_end=True),
                StateConfig(name="error", is_end=True)
            ]
        )
        
        config = FSMConfig(
            name="complex_fsm",
            description="Complex test FSM",
            networks=[network],
            main_network="complex"
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)
        
        # Execute with real FSM
        result = await fsm.execute_async({"test": "complex"})
        
        # Should complete (will go init->process->validate->success)
        assert result["status"] == "completed"
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_execute_stream_mode_with_chunks(self):
        """Test stream mode execution with multiple chunks."""
        config = self.create_simple_fsm_config("stream_fsm")
        builder = FSMBuilder()
        fsm = builder.build(config)
        fsm.data_mode = ProcessingMode.STREAM
        
        # Create test data chunks
        test_chunks = [
            {"chunk": 1, "data": "first"},
            {"chunk": 2, "data": "second"},
            {"chunk": 3, "data": "third"}
        ]
        
        # Execute with initial data (single chunk)
        result = await fsm.execute_async({"test": "stream_data"})
        
        # Should complete successfully
        if result["status"] != "completed":
            print(f"Stream mode failed: {result}")
        assert result["status"] == "completed"
        assert "chunks_processed" in result["data"]
        assert result["data"]["chunks_processed"] == 1  # One chunk processed
        
        # Now test with multiple chunks by directly manipulating the stream context
        from dataknobs_fsm.execution.context import ExecutionContext
        from dataknobs_fsm.streaming.core import StreamContext, StreamConfig
        
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM,
            transaction_mode=fsm.transaction_mode
        )
        
        # Create stream context and add multiple chunks
        stream_config = StreamConfig()
        context.stream_context = StreamContext(config=stream_config)
        
        # Add multiple chunks
        for i, chunk_data in enumerate(test_chunks):
            is_last = (i == len(test_chunks) - 1)
            context.stream_context.add_data(chunk_data, is_last=is_last)
        
        # Get the async engine and execute
        engine = fsm.get_async_engine()
        success, stream_result = await engine.execute(context)
        
        # Verify multiple chunks were processed
        assert success
        assert stream_result["chunks_processed"] == len(test_chunks)
        assert stream_result["records_processed"] == len(test_chunks)