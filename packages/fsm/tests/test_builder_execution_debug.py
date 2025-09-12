"""Debug test for FSM execute()."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from dataknobs_fsm.config.builder import FSM
from dataknobs_fsm.config.schema import FSMConfig
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.modes import TransactionMode
from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.execution.engine import ExecutionEngine


@pytest.mark.asyncio
async def test_execute_debug():
    """Debug execute method to see what's happening."""
    # Create mock CoreFSM
    mock_core_fsm = Mock(spec=CoreFSM)
    mock_core_fsm.name = "test_fsm"
    mock_core_fsm.data_mode = DataHandlingMode.COPY
    mock_core_fsm.transaction_mode = TransactionMode.NONE
    mock_core_fsm.networks = {}
    mock_core_fsm.main_network = "main"
    mock_core_fsm.function_registry = Mock()
    
    # Create mock config
    config = Mock(spec=FSMConfig)
    config.name = "test_fsm"
    
    # Create FSM instance
    fsm = FSM(
        core_fsm=mock_core_fsm,
        config=config,
        resource_manager=ResourceManager()
    )
    
    # Mock the engine
    mock_engine = Mock(spec=ExecutionEngine)
    mock_engine.execute.return_value = (True, {"result": "success"})
    
    with patch.object(fsm, 'get_engine', return_value=mock_engine):
        result = await fsm.execute({"input": "data"})
        print(f"Result: {result}")
        if result["status"] == "error":
            print(f"Error: {result.get('error')}")
            print(f"Data: {result.get('data')}")