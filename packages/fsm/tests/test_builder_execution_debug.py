"""Debug test for FSM execute()."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import FSMConfig
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import TransactionMode
from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.execution.engine import ExecutionEngine


@pytest.mark.asyncio
async def test_execute_debug():
    """Debug execute method to see what's happening."""
    # Create CoreFSM instance directly
    from dataknobs_fsm.core.modes import ProcessingMode
    fsm = CoreFSM(
        name="test_fsm",
        data_mode=ProcessingMode.SINGLE,
        transaction_mode=TransactionMode.NONE,
        resource_manager=ResourceManager()
    )
    
    # Mock the engine
    mock_engine = Mock(spec=ExecutionEngine)
    mock_engine.execute.return_value = (True, {"result": "success"})
    
    with patch.object(fsm, 'get_engine', return_value=mock_engine):
        result = fsm.execute({"input": "data"})  # Note: execute is now synchronous
        print(f"Result: {result}")
        if result["status"] == "error":
            print(f"Error: {result.get('error')}")
            print(f"Data: {result.get('data')}")