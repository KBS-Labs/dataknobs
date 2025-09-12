"""Tests for FSM execute() implementation in config/builder.py."""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

from dataknobs_fsm.config.builder import FSM, FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig, 
    StateConfig, 
    ArcConfig, 
    NetworkConfig,
    DataModeConfig,
    TransactionConfig
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
        # Mock just the engine to control execution
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"result": "success"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            result = await fsm_instance.execute({"input": "data"})
            
            assert result["status"] == "completed"
            assert result["data"] == {"result": "success"}
            assert "execution_id" in result
            assert "transitions" in result
            assert "duration" in result
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, fsm_instance):
        """Test failed execution of FSM."""
        # Mock the engine to return failure
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (False, {"error": "Failed"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            result = await fsm_instance.execute({"input": "data"})
            
            assert result["status"] == "failed"
            assert result["data"] == {"error": "Failed"}
    
    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, fsm_instance):
        """Test exception handling in execute()."""
        # Mock the engine to raise an exception
        with patch.object(fsm_instance, 'get_engine') as mock_get_engine:
            mock_get_engine.side_effect = RuntimeError("Engine error")
            
            result = await fsm_instance.execute({"input": "data"})
            
            assert result["status"] == "error"
            assert "Engine error" in result["error"]
            assert result["data"] == {"input": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_context_creation(self, fsm_instance):
        """Test that ExecutionContext is created correctly."""
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"result": "data"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            with patch('dataknobs_fsm.execution.context.ExecutionContext') as MockContext:
                mock_context = Mock()
                mock_context.execution_id = "test-123"
                mock_context.transition_count = 3
                mock_context.duration = 1.5
                MockContext.return_value = mock_context
                
                result = await fsm_instance.execute({"input": "data"})
                
                # Verify context was created with correct params
                # Note: ExecutionContext doesn't take initial data
                MockContext.assert_called_once_with(
                    data_mode=fsm_instance.core_fsm.data_mode,
                    transaction_mode=fsm_instance.core_fsm.transaction_mode
                )
                
                # Verify engine.execute was called with context and data
                mock_engine.execute.assert_called_once_with(mock_context, {"input": "data"})
                
                # Verify result includes context attributes
                assert result["execution_id"] == "test-123"
                assert result["transitions"] == 3
                assert result["duration"] == 1.5
    
    @pytest.mark.asyncio
    async def test_execute_with_none_initial_data(self, fsm_instance):
        """Test execution with None as initial data."""
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"processed": "data"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            result = await fsm_instance.execute(None)
            
            assert result["status"] == "completed"
            assert result["data"] == {"processed": "data"}
            
            # Verify engine was called with None
            mock_engine.execute.assert_called_once()
            args = mock_engine.execute.call_args[0]
            assert args[1] is None
    
    @pytest.mark.asyncio
    async def test_execute_with_different_processing_modes(self):
        """Test execution with different processing modes."""
        # Note: FSM uses ProcessingMode, not DataHandlingMode
        for mode in [ProcessingMode.SINGLE, ProcessingMode.BATCH, ProcessingMode.STREAM]:
            config = self.create_simple_fsm_config(f"{mode.value}_fsm")
            # We can't easily set the processing mode through config yet,
            # so we'll build the FSM and then modify it
            builder = FSMBuilder()
            fsm = builder.build(config)
            fsm.core_fsm.data_mode = mode  # Directly set the mode
            
            mock_engine = Mock(spec=ExecutionEngine)
            mock_engine.execute.return_value = (True, {"mode": mode.value})
            
            with patch.object(fsm, 'get_engine', return_value=mock_engine):
                result = await fsm.execute({"test": "data"})
                
                assert result["status"] == "completed"
                assert result["data"] == {"mode": mode.value}
    
    @pytest.mark.asyncio
    async def test_execute_result_formatting(self, fsm_instance):
        """Test that result is properly formatted."""
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"output": "value"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            with patch('dataknobs_fsm.execution.context.ExecutionContext') as MockContext:
                mock_context = Mock()
                mock_context.execution_id = "exec-456"
                mock_context.transition_count = 5
                mock_context.duration = 2.3
                MockContext.return_value = mock_context
                
                result = await fsm_instance.execute({"input": "test"})
                
                # Check all required fields are present
                assert isinstance(result, dict)
                assert result["status"] == "completed"
                assert result["data"] == {"output": "value"}
                assert result["execution_id"] == "exec-456"
                assert result["transitions"] == 5
                assert result["duration"] == 2.3
    
    @pytest.mark.asyncio
    async def test_execute_missing_context_attributes(self, fsm_instance):
        """Test handling when context is missing expected attributes."""
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"result": "ok"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            with patch('dataknobs_fsm.execution.context.ExecutionContext') as MockContext:
                # Create context without execution_id, transition_count, duration
                mock_context = Mock(spec=[])  # Empty spec means no attributes
                MockContext.return_value = mock_context
                
                result = await fsm_instance.execute({})
                
                # Should handle missing attributes gracefully
                assert result["status"] == "completed"
                assert result["execution_id"] is None
                assert result["transitions"] == 0
                assert result["duration"] is None
    
    @pytest.mark.asyncio
    async def test_execute_engine_creation_error(self, fsm_instance):
        """Test handling when engine creation fails."""
        with patch.object(fsm_instance, 'get_engine') as mock_get_engine:
            mock_get_engine.side_effect = ValueError("Cannot create engine")
            
            result = await fsm_instance.execute({"test": "data"})
            
            assert result["status"] == "error"
            assert "Cannot create engine" in result["error"]
            assert result["data"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_context_creation_error(self, fsm_instance):
        """Test handling when context creation fails."""
        mock_engine = Mock(spec=ExecutionEngine)
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            with patch('dataknobs_fsm.execution.context.ExecutionContext') as MockContext:
                MockContext.side_effect = RuntimeError("Context error")
                
                result = await fsm_instance.execute({"test": "data"})
                
                assert result["status"] == "error"
                assert "Context error" in result["error"]
                assert result["data"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_dry_principle(self, fsm_instance):
        """Test that execute() follows DRY principle by delegating to engine."""
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"result": "data"})
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            # Execute multiple times
            for i in range(3):
                result = await fsm_instance.execute({"iteration": i})
                assert result["status"] == "completed"
            
            # Verify engine.execute was called each time (no caching issues)
            assert mock_engine.execute.call_count == 3
            
            # Verify get_engine was called each time
            assert fsm_instance.get_engine.call_count == 3
    
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
        
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, complex_data)
        
        with patch.object(fsm_instance, 'get_engine', return_value=mock_engine):
            result = await fsm_instance.execute(complex_data)
            
            assert result["status"] == "completed"
            assert result["data"] == complex_data
    
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
        
        # Mock the engine
        mock_engine = Mock(spec=ExecutionEngine)
        mock_engine.execute.return_value = (True, {"path": "init->process->validate->success"})
        
        with patch.object(fsm, 'get_engine', return_value=mock_engine):
            result = await fsm.execute({"test": "complex"})
            
            assert result["status"] == "completed"
            assert "path" in result["data"]