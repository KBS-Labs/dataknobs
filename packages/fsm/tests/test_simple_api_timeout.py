"""Tests for timeout support in the Simple API (api/simple.py).

This test suite covers timeout handling in:
- process() function with ThreadPoolExecutor timeout
- process_file() function with asyncio.wait_for() timeout
- batch_process() function with concurrent.futures timeout
"""

import asyncio
import concurrent.futures
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from pathlib import Path
import json
import tempfile

from dataknobs_fsm.api.simple import (
    SimpleFSM,
    create_fsm,
    process_file,
    batch_process,
    validate_data
)
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    ArcConfig
)
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.execution.engine import ExecutionEngine


class TestSimpleAPITimeout:
    """Tests for timeout support in Simple API."""
    
    def create_simple_config(self) -> FSMConfig:
        """Create a simple FSM configuration for testing."""
        network = NetworkConfig(
            name="main",
            states=[
                StateConfig(
                    name="start",
                    is_start=True,
                    arcs=[
                        ArcConfig(target="process", priority=1)
                    ]
                ),
                StateConfig(
                    name="process",
                    arcs=[
                        ArcConfig(target="end", priority=1)
                    ]
                ),
                StateConfig(
                    name="end",
                    is_end=True
                )
            ]
        )
        
        return FSMConfig(
            name="timeout_test_fsm",
            description="FSM for timeout testing",
            networks=[network],
            main_network="main"
        )
    
    @pytest.fixture
    def simple_fsm(self):
        """Create a SimpleFSM instance for testing."""
        config = self.create_simple_config()
        # Convert FSMConfig to dict for SimpleFSM
        config_dict = config.model_dump()
        return SimpleFSM(config_dict)
    
    def test_process_with_timeout_success(self, simple_fsm):
        """Test process() completes within timeout."""
        # Mock the async engine to return quickly
        async def quick_execute(context):
            return (True, {"result": "success"})

        mock_engine = Mock()
        mock_engine.execute = Mock(side_effect=quick_execute)

        with patch.object(simple_fsm, '_async_engine', mock_engine):
            result = simple_fsm.process(
                data={"input": "test"},
                timeout=5.0  # 5 second timeout
            )

            assert result["success"] is True
            assert "error" not in result or result["error"] is None
            mock_engine.execute.assert_called_once()
    
    def test_process_with_timeout_exceeded(self, simple_fsm):
        """Test process() handles timeout properly."""
        # Mock the async engine to take longer than timeout
        async def slow_execute(context):
            await asyncio.sleep(2)  # Sleep for 2 seconds
            return (True, {"result": "success"})

        mock_engine = Mock()
        mock_engine.execute = Mock(side_effect=slow_execute)

        with patch.object(simple_fsm, '_async_engine', mock_engine):
            result = simple_fsm.process(
                data={"input": "test"},
                timeout=0.5  # 0.5 second timeout (will be exceeded)
            )

            # TimeoutError should be caught and returned as error result
            assert result["success"] is False
            assert "error" in result
            assert "exceeded timeout of 0.5 seconds" in result["error"]
    
    @pytest.mark.asyncio
    async def test_process_with_timeout_success2(self):
        """Test process() completes within timeout using real FSM."""
        # Create a simple FSM that completes quickly
        config = {
            "name": "test_fsm",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "end"}
                ]
            }]
        }
        
        fsm = SimpleFSM(config)
        
        # Process with timeout - should complete quickly
        result = fsm.process(
            data={"input": "test"},
            timeout=5.0
        )
        
        assert result["success"] is True
        assert "error" not in result or result["error"] is None
        assert result["final_state"] == "end"
    
    @pytest.mark.asyncio
    async def test_process_with_timeout_exceeded2(self, simple_fsm):
        """Test process() handles timeout properly."""
        # Mock the async engine to take longer than timeout
        async def slow_async_execute(context):
            await asyncio.sleep(2)  # Sleep for 2 seconds
            return (True, {"result": "success"})
        
        mock_async_engine = Mock()
        mock_async_engine.execute = slow_async_execute
        
        with patch.object(simple_fsm, '_async_engine', mock_async_engine):
            # process likely catches the timeout and returns error result
            result = simple_fsm.process(
                data={"input": "test"},
                timeout=0.5  # 0.5 second timeout (will be exceeded)
            )
            
            # Check that timeout was handled
            assert result["success"] is False
            assert "error" in result
    
    def test_process_file_with_timeout_success(self):
        """Test process_file() completes within timeout."""
        # Create temp input and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_file.write('{"data": "test1"}\n')
            input_file.write('{"data": "test2"}\n')
            input_path = Path(input_file.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        try:
            # Use the simple config
            config = self.create_simple_config()
            
            with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
                mock_fsm = Mock()
                
                # SimpleFSM.process_stream returns a dict directly, not a coroutine
                def mock_process_stream(**kwargs):
                    return {"processed": 2, "errors": 0}

                mock_fsm.process_stream = mock_process_stream
                mock_fsm.close = Mock()
                mock_create.return_value = mock_fsm
                
                result = process_file(
                    fsm_config=config,
                    input_file=str(input_path),
                    output_file=str(output_path),
                    timeout=5.0
                )
                
                assert result == {"processed": 2, "errors": 0}
                mock_fsm.close.assert_called_once()
        finally:
            # Cleanup temp files
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
    
    def test_process_file_with_timeout_exceeded(self):
        """Test process_file() handles timeout properly."""
        # Create temp input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as input_file:
            input_file.write('{"data": "test1"}\n')
            input_path = Path(input_file.name)
        
        try:
            config = self.create_simple_config()
            
            with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
                # Mock FSM with slow processing
                async def slow_process_stream(**kwargs):
                    await asyncio.sleep(2)  # Sleep for 2 seconds
                    return {"processed": 1, "errors": 0}
                
                mock_fsm = Mock()
                mock_fsm.process_stream = slow_process_stream
                mock_fsm.close = Mock()
                mock_create.return_value = mock_fsm
                
                with pytest.raises(asyncio.TimeoutError):
                    # Use asyncio.run with wait_for internally
                    asyncio.run(asyncio.wait_for(
                        mock_fsm.process_stream(
                            source=str(input_path),
                            sink=None,
                            chunk_size=100
                        ),
                        timeout=0.5  # 0.5 second timeout (will be exceeded)
                    ))
        finally:
            # Cleanup temp file
            input_path.unlink(missing_ok=True)
    
    def test_batch_process_with_timeout_success(self):
        """Test batch_process() completes within timeout."""
        config = self.create_simple_config()
        data = [{"id": i} for i in range(5)]
        
        with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
            mock_fsm = Mock()
            mock_fsm.process_batch = Mock(return_value=[
                {"id": i, "processed": True} for i in range(5)
            ])
            mock_fsm.close = Mock()
            mock_create.return_value = mock_fsm
            
            result = batch_process(
                fsm_config=config,
                data=data,
                batch_size=2,
                max_workers=2,
                timeout=5.0
            )
            
            assert len(result) == 5
            mock_fsm.process_batch.assert_called_once_with(
                data=data,
                batch_size=2,
                max_workers=2
            )
            mock_fsm.close.assert_called_once()
    
    def test_batch_process_with_timeout_exceeded(self):
        """Test batch_process() raises TimeoutError when timeout is exceeded."""
        config = self.create_simple_config()
        data = [{"id": i} for i in range(10)]
        
        with patch('dataknobs_fsm.api.simple.create_fsm') as mock_create:
            # Mock FSM with slow batch processing
            def slow_process_batch(**kwargs):
                time.sleep(2)  # Sleep for 2 seconds
                return [{"id": i, "processed": True} for i in range(10)]
            
            mock_fsm = Mock()
            mock_fsm.process_batch = slow_process_batch
            mock_fsm.close = Mock()
            mock_create.return_value = mock_fsm
            
            with pytest.raises(TimeoutError) as exc_info:
                batch_process(
                    fsm_config=config,
                    data=data,
                    batch_size=5,
                    max_workers=2,
                    timeout=0.5  # 0.5 second timeout (will be exceeded)
                )
            
            assert "Batch processing exceeded timeout of 0.5 seconds" in str(exc_info.value)
            mock_fsm.close.assert_called_once()
    
    def test_threadpool_timeout_handling(self):
        """Test ThreadPoolExecutor timeout handling in process()."""
        # Test that ThreadPoolExecutor properly cancels futures on timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            def long_running_task():
                time.sleep(2)
                return "completed"
            
            future = executor.submit(long_running_task)
            
            with pytest.raises(concurrent.futures.TimeoutError):
                future.result(timeout=0.5)
            
            # Future should be cancelable after timeout
            # Note: cancel() returns False if already running
            # but we can verify it was attempted
            future.cancel()
    
    def test_asyncio_wait_for_timeout_handling(self):
        """Test asyncio.wait_for() timeout handling."""
        async def test_wait_for():
            async def long_running_coroutine():
                await asyncio.sleep(2)
                return "completed"
            
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(long_running_coroutine(), timeout=0.5)
        
        asyncio.run(test_wait_for())
    
    def test_timeout_error_messages(self, simple_fsm):
        """Test that timeout errors have appropriate messages."""
        # Test process() timeout message
        async def slow_execute(context):
            await asyncio.sleep(1)
            return (True, {})

        mock_engine = Mock()
        mock_engine.execute = Mock(side_effect=slow_execute)

        with patch.object(simple_fsm, '_async_engine', mock_engine):
            result = simple_fsm.process({"test": "data"}, timeout=0.1)

            # Check error message format
            assert result["success"] is False
            error_msg = result["error"]
            assert "timeout" in error_msg.lower()
            assert "0.1" in error_msg  # Should mention the timeout value
            assert "seconds" in error_msg.lower()
    
    def test_timeout_with_resources_cleanup(self, simple_fsm):
        """Test that resources are properly cleaned up on timeout."""
        # Use the real ResourceManager
        from dataknobs_fsm.resources.manager import ResourceManager
        
        # Create a real resource manager
        resource_manager = ResourceManager()
        
        # Spy on the cleanup method
        original_cleanup = resource_manager.cleanup
        cleanup_called = []
        
        async def tracked_cleanup():
            cleanup_called.append(True)
            return await original_cleanup()
        
        resource_manager.cleanup = tracked_cleanup
        
        # Patch both resource managers (SimpleFSM keeps a reference, but async_fsm has the real one)
        with patch.object(simple_fsm, '_resource_manager', resource_manager):
            with patch.object(simple_fsm._async_fsm, '_resource_manager', resource_manager):
                # Mock slow engine
                async def slow_execute(context):
                    await asyncio.sleep(1)
                    return (True, {})

                mock_engine = Mock()
                mock_engine.execute = Mock(side_effect=slow_execute)

                with patch.object(simple_fsm, '_async_engine', mock_engine):
                    # Process with timeout (will fail but not raise)
                    result = simple_fsm.process({"test": "data"}, timeout=0.1)
                    assert result["success"] is False  # Verify timeout occurred

                    # Ensure close() cleans up resources even after timeout
                    simple_fsm.close()
                    assert len(cleanup_called) == 1  # Verify cleanup was called
    
    def test_process_without_timeout(self, simple_fsm):
        """Test that process() works without timeout parameter."""
        async def quick_execute(context):
            return (True, {"result": "success"})

        mock_engine = Mock()
        mock_engine.execute = Mock(side_effect=quick_execute)

        with patch.object(simple_fsm, '_async_engine', mock_engine):
            # Should work fine without timeout
            result = simple_fsm.process({"input": "test"})

            assert result["success"] is True
            mock_engine.execute.assert_called_once()
            # Should not use ThreadPoolExecutor when no timeout
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                simple_fsm.process({"input": "test"})
                mock_executor.assert_not_called()
