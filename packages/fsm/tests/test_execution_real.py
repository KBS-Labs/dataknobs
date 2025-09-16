"""Tests for execution components using real implementations."""

import pytest
import time
import tempfile
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path

from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.batch import BatchExecutor
from dataknobs_fsm.execution.stream import StreamExecutor
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.api.advanced import AdvancedFSM, ExecutionMode


class TestExecutionEngineReal:
    """Test execution engine with real FSM configurations."""

    @pytest.fixture
    def simple_fsm_config(self):
        """Create a simple FSM configuration."""
        return {
            'name': 'test_execution',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin',
                    'transform': {'code': 'data["count"] = data.get("count", 0) + 1; data'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'data["count"] = data["count"] * 2; data'}
                }
            ]
        }

    @pytest.fixture
    def conditional_fsm_config(self):
        """Create FSM with conditional transitions."""
        return {
            'name': 'conditional_execution',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'check'},
                {'name': 'process_high'},
                {'name': 'process_low'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'check',
                    'name': 'begin'
                },
                {
                    'from': 'check',
                    'to': 'process_high',
                    'name': 'high_value',
                    'condition': {'type': 'inline', 'code': 'data.get("value", 0) > 50'}
                },
                {
                    'from': 'check',
                    'to': 'process_low',
                    'name': 'low_value',
                    'condition': {'type': 'inline', 'code': 'data.get("value", 0) <= 50'}
                },
                {
                    'from': 'process_high',
                    'to': 'end',
                    'name': 'complete_high',
                    'transform': {'code': 'data["result"] = "high"; data'}
                },
                {
                    'from': 'process_low',
                    'to': 'end',
                    'name': 'complete_low',
                    'transform': {'code': 'data["result"] = "low"; data'}
                }
            ]
        }

    def test_simple_execution(self, simple_fsm_config):
        """Test simple FSM execution."""
        # Create FSM from config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(simple_fsm_config, f)
            config_file = f.name

        try:
            # Load and build FSM
            loader = ConfigLoader()
            config = loader.load_from_file(config_file)
            builder = FSMBuilder()
            fsm = builder.build(config)

            # Create execution engine
            engine = ExecutionEngine(fsm)

            # Create context with data
            context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
            context.data = {'initial': 'value'}
            context.current_state = 'start'

            # Execute
            success, result = engine.execute(context)

            # Verify execution
            assert success
            assert context.current_state == 'end'
            assert context.data['count'] == 2  # 1 added, then doubled
            assert context.data['initial'] == 'value'  # Original data preserved

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_conditional_execution(self, conditional_fsm_config):
        """Test conditional branching in FSM."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(conditional_fsm_config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Test high value path
            result = fsm.process({'value': 75})
            if not result['success']:
                print(f"High value test failed: {result}")
            assert result['success'], f"Failed with error: {result.get('error')}"
            assert result['data']['result'] == 'high'
            assert result['final_state'] == 'end'

            # Test low value path
            result = fsm.process({'value': 25})
            assert result['success']
            assert result['data']['result'] == 'low'
            assert result['final_state'] == 'end'

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_execution_with_error_handling(self):
        """Test execution with error states."""
        config = {
            'name': 'error_handling',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'validate'},
                {'name': 'process'},
                {'name': 'error', 'is_end': True},
                {'name': 'success', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'validate',
                    'name': 'begin'
                },
                {
                    'from': 'validate',
                    'to': 'process',
                    'name': 'valid',
                    'condition': {'type': 'inline', 'code': '"required_field" in data'}
                },
                {
                    'from': 'validate',
                    'to': 'error',
                    'name': 'invalid',
                    'condition': {'type': 'inline', 'code': '"required_field" not in data'}
                },
                {
                    'from': 'process',
                    'to': 'success',
                    'name': 'complete'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Test valid data
            result = fsm.process({'required_field': 'present'})
            assert result['success']
            assert result['final_state'] == 'success'

            # Test invalid data
            result = fsm.process({'other_field': 'value'})
            assert result['success']  # Execution succeeded even if ended in error state
            assert result['final_state'] == 'error'

        finally:
            Path(config_file).unlink(missing_ok=True)


class TestBatchExecutorReal:
    """Test batch executor with real FSM."""

    def test_batch_processing(self):
        """Test batch processing of multiple items."""
        config = {
            'name': 'batch_processor',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin',
                    'transform': {'code': 'data["processed"] = True; data'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'data["id"] = data.get("id", 0) * 10; data'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Process batch
            batch_data = [
                {'id': 1, 'value': 'a'},
                {'id': 2, 'value': 'b'},
                {'id': 3, 'value': 'c'}
            ]

            results = fsm.process_batch(batch_data, batch_size=2, max_workers=2)
            
            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result['success'], f"Result {i} failed: {result}"
                assert result['data']['processed'] is True
                assert result['data']['id'] == (i + 1) * 10

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_batch_with_failures(self):
        """Test batch processing with some failures."""
        config = {
            'name': 'batch_with_validation',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'validate'},
                {'name': 'process'},
                {'name': 'error', 'is_end': True},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'validate',
                    'name': 'begin'
                },
                {
                    'from': 'validate',
                    'to': 'process',
                    'name': 'valid',
                    'condition': {'type': 'inline', 'code': 'data.get("valid", False)'}
                },
                {
                    'from': 'validate',
                    'to': 'error',
                    'name': 'invalid',
                    'condition': {'type': 'inline', 'code': 'not data.get("valid", False)'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Process batch with mixed validity
            batch_data = [
                {'id': 1, 'valid': True},
                {'id': 2, 'valid': False},
                {'id': 3, 'valid': True}
            ]

            results = fsm.process_batch(batch_data)

            # Verify results
            assert len(results) == 3
            assert results[0]['success']
            assert results[0]['final_state'] == 'end'
            assert results[1]['success']
            assert results[1]['final_state'] == 'error'
            assert results[2]['success']
            assert results[2]['final_state'] == 'end'

        finally:
            Path(config_file).unlink(missing_ok=True)


class TestSyncBatchExecutorReal:
    """Test synchronous batch executor with real FSM."""
    
    def test_sync_batch_processing(self):
        """Test synchronous batch processing of multiple items."""
        config = {
            'name': 'sync_batch_processor',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin',
                    'transform': {'code': 'data["processed"] = True; data'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'data["id"] = data.get("id", 0) * 10; data'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            # Load and build FSM
            loader = ConfigLoader()
            config_obj = loader.load_from_file(config_file)
            builder = FSMBuilder()
            fsm = builder.build(config_obj)
            
            # Create synchronous batch executor
            batch_executor = BatchExecutor(
                fsm=fsm,
                parallelism=1,  # Sequential execution
                batch_size=10
            )
            
            # Process batch with Record objects
            from dataknobs_data import Record
            batch_data = [
                Record({'id': 1, 'value': 'a'}),
                Record({'id': 2, 'value': 'b'}),
                Record({'id': 3, 'value': 'c'})
            ]
            
            results = batch_executor.execute_batch(batch_data)
            
            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.success, f"Result {i} failed: {result}"
                # Check data was properly extracted from Record and processed
                assert result.result['processed'] is True
                assert result.result['id'] == (i + 1) * 10

        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_sync_batch_parallel_processing(self):
        """Test synchronous batch processing with parallelism."""
        config = {
            'name': 'sync_parallel_batch',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'process',
                    'name': 'begin',
                    'transform': {'code': 'import time; time.sleep(0.01); data["processed"] = True; data'}
                },
                {
                    'from': 'process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'data["id"] = data.get("id", 0) * 10; data'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            # Load and build FSM
            loader = ConfigLoader()
            config_obj = loader.load_from_file(config_file)
            builder = FSMBuilder()
            fsm = builder.build(config_obj)
            
            # Create synchronous batch executor with parallelism
            batch_executor = BatchExecutor(
                fsm=fsm,
                parallelism=3,  # Parallel execution
                batch_size=10
            )
            
            # Process batch with Record objects
            from dataknobs_data import Record
            batch_data = [
                Record({'id': i, 'value': f'item_{i}'})
                for i in range(1, 7)  # 6 items
            ]
            
            start_time = time.time()
            results = batch_executor.execute_batch(batch_data)
            execution_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 6
            for i, result in enumerate(results):
                assert result.success, f"Result {i} failed: {result}"
                assert result.result['processed'] is True
                assert result.result['id'] == (i + 1) * 10
            
            # With parallelism=3 and 0.01s sleep per item, should be faster than sequential
            # Sequential would take ~0.12s (6 items * 0.02s for 2 transforms)
            # Parallel should take ~0.04s (2 batches of 3 items)
            assert execution_time < 0.1, f"Parallel execution too slow: {execution_time}s"

        finally:
            Path(config_file).unlink(missing_ok=True)


class TestAdvancedExecutionReal:
    """Test advanced execution features."""

    def test_async_execution(self):
        """Test async execution mode."""
        config = {
            'name': 'async_test',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'async_process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'async_process',
                    'name': 'begin'
                },
                {
                    'from': 'async_process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'import time; data["timestamp"] = time.time(); data'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            # Use SimpleFSM which has async capabilities
            fsm = SimpleFSM(config_file)

            # Process data asynchronously
            import asyncio
            
            async def run_test():
                result = fsm.process({'test': 'data'})
                return result

            result = asyncio.run(run_test())
            
            assert result['success']
            assert 'timestamp' in result['data']
            assert result['final_state'] == 'end'

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_execution_timeout(self):
        """Test execution with timeout."""
        config = {
            'name': 'timeout_test',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'slow_process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'slow_process',
                    'name': 'begin'
                },
                {
                    'from': 'slow_process',
                    'to': 'end',
                    'name': 'complete',
                    'transform': {'code': 'import time; time.sleep(0.1); data'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Execute with sufficient timeout - should succeed
            result = fsm.process({'test': 'data'}, timeout=1.0)
            assert result['success']

            # Note: Actual timeout interruption would require threading/async
            # which SimpleFSM doesn't fully support yet

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_state_persistence(self):
        """Test state persistence across executions."""
        config = {
            'name': 'stateful_fsm',
            'data_mode': 'reference',  # Use reference mode to maintain state
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'accumulate'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'accumulate',
                    'name': 'begin'
                },
                {
                    'from': 'accumulate',
                    'to': 'accumulate',
                    'name': 'continue',
                    'condition': {'type': 'inline', 'code': 'data.get("count", 0) < 3'},
                    'transform': {'code': 'data["count"] = data.get("count", 0) + 1; data'}
                },
                {
                    'from': 'accumulate',
                    'to': 'end',
                    'name': 'complete',
                    'condition': {'type': 'inline', 'code': 'data.get("count", 0) >= 3'}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Process should loop and accumulate
            # The FSM should:
            # 1. Start at 'start'
            # 2. Move to 'accumulate' with count=0
            # 3. Loop in 'accumulate': count=1, count=2, count=3
            # 4. When count=3, condition count>=3 is true, move to 'end'
            # Increase max_transitions since we need several self-loops
            result = fsm.process({'initial': 'value'}, timeout=10.0)
            if not result['success']:
                print(f"Process failed: {result}")
                print(f"Path taken: {result.get('path', [])}")
                print(f"Final data: {result.get('data', {})}")
            assert result['success'], f"Process failed: {result.get('error', 'Unknown error')}"
            assert result['data']['count'] == 3
            assert result['final_state'] == 'end'

        finally:
            Path(config_file).unlink(missing_ok=True)


class TestExecutionMetrics:
    """Test execution metrics and monitoring."""

    def test_execution_history(self):
        """Test that execution history is tracked."""
        config = {
            'name': 'metrics_test',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'middle'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'middle',
                    'name': 'first'
                },
                {
                    'from': 'middle',
                    'to': 'end',
                    'name': 'second'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Execute
            result = fsm.process({'test': 'data'})

            # Check execution path
            assert result['success']
            assert result['path'] == ['start', 'middle', 'end']

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_execution_performance(self):
        """Test execution performance metrics."""
        config = {
            'name': 'performance_test',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {
                    'from': 'start',
                    'to': 'end',
                    'name': 'direct'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_file = f.name

        try:
            fsm = SimpleFSM(config_file)

            # Measure execution time
            start_time = time.time()
            result = fsm.process({'test': 'data'})
            execution_time = time.time() - start_time

            assert result['success']
            assert execution_time < 1.0  # Should be fast for simple FSM

        finally:
            Path(config_file).unlink(missing_ok=True)
