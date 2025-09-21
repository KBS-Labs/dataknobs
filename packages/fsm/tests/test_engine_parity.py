"""Tests to ensure parity between sync and async execution engines.

This module verifies that both the synchronous ExecutionEngine and
asynchronous AsyncExecutionEngine produce identical results for the
same FSM configurations and input data.
"""

import asyncio
import pytest
from typing import Any, Dict

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.execution.engine import ExecutionEngine, TraversalStrategy
from dataknobs_fsm.execution.async_engine import AsyncExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.core.modes import ProcessingMode


class TestEngineParity:
    """Test suite for verifying engine parity."""

    @pytest.fixture
    def simple_config(self) -> Dict[str, Any]:
        """Create a simple FSM configuration."""
        return {
            "name": "test_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True
                        },
                        {
                            "name": "process"
                        },
                        {
                            "name": "end",
                            "is_end": True
                        }
                    ],
                    "arcs": [
                        {
                            "from": "start",
                            "to": "process",
                            "name": "start_to_process"
                        },
                        {
                            "from": "process",
                            "to": "end",
                            "name": "process_to_end"
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def transform_config(self) -> Dict[str, Any]:
        """Create an FSM configuration with transforms."""
        return {
            "name": "transform_fsm",
            "main_network": "main",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "transforms": [{
                                "type": "inline",
                                "code": "lambda state: {**state.data, 'processed': True} if isinstance(state.data, dict) else {'value': state.data, 'processed': True}"
                            }]
                        },
                        {
                            "name": "process",
                            "transforms": [{
                                "type": "inline",
                                "code": "lambda state: {**state.data, 'value': state.data.get('value', 0) * 2} if isinstance(state.data, dict) else {'value': state.data * 2}"
                            }]
                        },
                        {
                            "name": "end",
                            "is_end": True
                        }
                    ],
                    "arcs": [
                        {
                            "from": "start",
                            "to": "process",
                            "name": "start_to_process"
                        },
                        {
                            "from": "process",
                            "to": "end",
                            "name": "process_to_end"
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def simple_fsm(self, simple_config):
        """Build a simple FSM from config."""
        loader = ConfigLoader()
        config = loader.load_from_dict(simple_config)
        builder = FSMBuilder()
        return builder.build(config)

    @pytest.fixture
    def transform_fsm(self, transform_config):
        """Build an FSM with transforms from config."""
        loader = ConfigLoader()
        config = loader.load_from_dict(transform_config)
        builder = FSMBuilder()
        return builder.build(config)

    def test_simple_execution_parity(self, simple_fsm):
        """Test that both engines produce the same result for simple execution."""
        input_data = {"value": 42}

        # Execute with sync engine
        sync_engine = ExecutionEngine(simple_fsm)
        sync_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        sync_context.data = input_data.copy()

        sync_success, sync_result = sync_engine.execute(sync_context)

        # Execute with async engine
        async_engine = AsyncExecutionEngine(simple_fsm)
        async_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        async_context.data = input_data.copy()

        async_success, async_result = asyncio.run(
            async_engine.execute(async_context)
        )

        # Compare results
        assert sync_success == async_success
        assert sync_context.current_state == async_context.current_state
        assert sync_context.data == async_context.data

    def test_transform_execution_parity(self, transform_fsm):
        """Test that both engines execute state transforms identically."""
        input_data = {"value": 10}

        # Execute with sync engine
        sync_engine = ExecutionEngine(transform_fsm)
        sync_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        sync_context.data = input_data.copy()

        sync_success, sync_result = sync_engine.execute(sync_context)

        # Execute with async engine
        async_engine = AsyncExecutionEngine(transform_fsm)
        async_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        async_context.data = input_data.copy()

        async_success, async_result = asyncio.run(
            async_engine.execute(async_context)
        )

        # Compare results
        assert sync_success == async_success
        assert sync_context.current_state == async_context.current_state

        # Both should have processed flag and doubled value
        assert sync_context.data.get("processed") == True
        assert async_context.data.get("processed") == True
        assert sync_context.data.get("value") == 20
        assert async_context.data.get("value") == 20
        assert sync_context.data == async_context.data

    def test_initial_state_finding_parity(self, simple_fsm):
        """Test that both engines find the same initial state."""
        sync_engine = ExecutionEngine(simple_fsm)
        async_engine = AsyncExecutionEngine(simple_fsm)

        sync_initial = sync_engine._find_initial_state()
        async_initial = asyncio.run(async_engine._find_initial_state())

        assert sync_initial == async_initial
        assert sync_initial == "start"

    def test_final_state_detection_parity(self, simple_fsm):
        """Test that both engines detect final states identically."""
        sync_engine = ExecutionEngine(simple_fsm)
        async_engine = AsyncExecutionEngine(simple_fsm)

        # Test various states
        states_to_test = ["start", "process", "end", None, "nonexistent"]

        for state in states_to_test:
            sync_is_final = sync_engine._is_final_state(state)
            async_is_final = asyncio.run(async_engine._is_final_state(state))

            assert sync_is_final == async_is_final, f"Mismatch for state '{state}'"

            # Verify expected results
            if state == "end":
                assert sync_is_final == True
            else:
                assert sync_is_final == False

    def test_error_handling_parity(self, simple_config):
        """Test that both engines handle errors identically."""
        # Add a transform that will fail
        error_config = simple_config.copy()
        error_config["networks"][0]["states"][1]["transform"] = {
            "type": "inline",
            "code": "lambda state: 1/0"  # This will raise ZeroDivisionError
        }

        loader = ConfigLoader()
        config = loader.load_from_dict(error_config)
        builder = FSMBuilder()
        error_fsm = builder.build(config)

        input_data = {"value": 42}

        # Execute with sync engine
        sync_engine = ExecutionEngine(error_fsm)
        sync_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        sync_context.data = input_data.copy()

        sync_success, sync_result = sync_engine.execute(sync_context)

        # Execute with async engine
        async_engine = AsyncExecutionEngine(error_fsm)
        async_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        async_context.data = input_data.copy()

        async_success, async_result = asyncio.run(
            async_engine.execute(async_context)
        )

        # Both should have marked the process state as failed
        assert hasattr(sync_context, 'failed_states')
        assert hasattr(async_context, 'failed_states')
        assert "process" in sync_context.failed_states
        assert "process" in async_context.failed_states

        # Both should still complete execution (reach end state)
        assert sync_context.current_state == "end"
        assert async_context.current_state == "end"

    def test_statistics_parity(self, simple_fsm):
        """Test that both engines track statistics similarly."""
        input_data = {"value": 42}

        # Execute with sync engine
        sync_engine = ExecutionEngine(simple_fsm)
        sync_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        sync_context.data = input_data.copy()

        sync_engine.execute(sync_context)
        sync_stats = sync_engine.get_execution_statistics()

        # Execute with async engine
        async_engine = AsyncExecutionEngine(simple_fsm)
        async_context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        async_context.data = input_data.copy()

        asyncio.run(async_engine.execute(async_context))
        async_stats = async_engine.get_execution_statistics()

        # Compare statistics
        assert sync_stats['execution_count'] == async_stats['execution_count']
        assert sync_stats['transition_count'] == async_stats['transition_count']
        assert sync_stats['error_count'] == async_stats['error_count']

    def test_multiple_execution_parity(self, simple_fsm):
        """Test that both engines handle multiple executions identically."""
        test_data = [
            {"value": 1},
            {"value": 2},
            {"value": 3}
        ]

        sync_results = []
        async_results = []

        # Execute with sync engine
        sync_engine = ExecutionEngine(simple_fsm)
        for data in test_data:
            context = ExecutionContext(
                data_mode=ProcessingMode.SINGLE
            )
            context.data = data.copy()
            success, result = sync_engine.execute(context)
            sync_results.append({
                'success': success,
                'final_state': context.current_state,
                'data': context.data
            })

        # Execute with async engine
        async_engine = AsyncExecutionEngine(simple_fsm)

        async def run_async_tests():
            results = []
            for data in test_data:
                context = ExecutionContext(
                    data_mode=ProcessingMode.SINGLE
                )
                context.data = data.copy()
                success, result = await async_engine.execute(context)
                results.append({
                    'success': success,
                    'final_state': context.current_state,
                    'data': context.data
                })
            return results

        async_results = asyncio.run(run_async_tests())

        # Compare all results
        assert len(sync_results) == len(async_results)
        for sync_res, async_res in zip(sync_results, async_results):
            assert sync_res == async_res

    def test_network_selection_parity(self, simple_fsm):
        """Test that both engines select networks identically."""
        sync_engine = ExecutionEngine(simple_fsm)
        async_engine = AsyncExecutionEngine(simple_fsm)

        # Create a context
        context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )

        # Get current network using base class method
        sync_network = sync_engine.get_current_network_common(context)
        async_network = async_engine.get_current_network_common(context)

        assert sync_network == async_network
        assert sync_network is not None
        assert sync_network.name == "main"
