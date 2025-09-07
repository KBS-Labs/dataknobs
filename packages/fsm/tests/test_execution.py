"""Tests for Phase 4 execution components."""

import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.records import Record

from dataknobs_fsm.core.arc import ArcDefinition, DataIsolationMode, PushArc
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import DataMode, TransactionMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import State
from dataknobs_fsm.execution import (
    BatchExecutor,
    ExecutionContext,
    ExecutionEngine,
    NetworkExecutor,
    ResourceAllocation,
    ResourceStatus,
    StreamExecutor,
    StreamPipeline,
    TraversalStrategy,
)
from dataknobs_fsm.functions.base import Function, FunctionContext, FunctionRegistry
from dataknobs_fsm.streaming.core import IStreamSink, IStreamSource, StreamChunk, StreamConfig


# Test functions
class AlwaysTrueFunction(Function):
    """Function that always returns True."""
    
    def execute(self, data: Any, context: FunctionContext) -> Any:
        return True


class AlwaysFalseFunction(Function):
    """Function that always returns False."""
    
    def execute(self, data: Any, context: FunctionContext) -> Any:
        return False


class IncrementFunction(Function):
    """Function that increments numeric data."""
    
    def execute(self, data: Any, context: FunctionContext) -> Any:
        if isinstance(data, (int, float)):
            return data + 1
        return data


class ValidatePositiveFunction(Function):
    """Function that validates positive numbers."""
    
    def execute(self, data: Any, context: FunctionContext) -> Any:
        return isinstance(data, (int, float)) and data > 0


class DoubleFunction(Function):
    """Function that doubles numeric data."""
    
    def execute(self, data: Any, context: FunctionContext) -> Any:
        if isinstance(data, (int, float)):
            return data * 2
        return data


# Test stream source
class TestStreamSource(IStreamSource):
    """Test stream source with configurable data."""
    
    def __init__(self, data_chunks: List[List[Any]]):
        self.data_chunks = data_chunks
        self.current = 0
    
    def read_chunk(self) -> Optional[StreamChunk]:
        if self.current >= len(self.data_chunks):
            return None
        
        chunk = StreamChunk(
            data=self.data_chunks[self.current],
            sequence_number=self.current,
            metadata={'chunk_index': self.current},
            is_last=(self.current == len(self.data_chunks) - 1)
        )
        self.current += 1
        return chunk
    
    def close(self) -> None:
        pass


# Test stream sink
class TestStreamSink(IStreamSink):
    """Test stream sink that collects data."""
    
    def __init__(self):
        self.chunks: List[StreamChunk] = []
        self.all_data: List[Any] = []
    
    def write_chunk(self, chunk: StreamChunk) -> bool:
        self.chunks.append(chunk)
        self.all_data.extend(chunk.data)
        return True
    
    def flush(self) -> None:
        pass
    
    def close(self) -> None:
        pass


class TestExecutionContext(unittest.TestCase):
    """Tests for ExecutionContext."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        context = ExecutionContext(
            data_mode=DataMode.BATCH,
            transaction_mode=TransactionMode.PER_BATCH
        )
        
        self.assertEqual(context.data_mode, DataMode.BATCH)
        self.assertEqual(context.transaction_mode, TransactionMode.PER_BATCH)
        self.assertIsNone(context.current_state)
        self.assertEqual(len(context.state_history), 0)
    
    def test_state_management(self):
        """Test state tracking."""
        context = ExecutionContext()
        
        context.set_state("state1")
        self.assertEqual(context.current_state, "state1")
        self.assertIsNone(context.previous_state)
        
        context.set_state("state2")
        self.assertEqual(context.current_state, "state2")
        self.assertEqual(context.previous_state, "state1")
        self.assertEqual(context.state_history, ["state1"])
    
    def test_network_stack(self):
        """Test network push/pop."""
        context = ExecutionContext()
        
        context.push_network("network1", "return_state1")
        context.push_network("network2", None)
        
        self.assertEqual(len(context.network_stack), 2)
        
        network, return_state = context.pop_network()
        self.assertEqual(network, "network2")
        self.assertIsNone(return_state)
        
        network, return_state = context.pop_network()
        self.assertEqual(network, "network1")
        self.assertEqual(return_state, "return_state1")
    
    def test_resource_allocation(self):
        """Test resource management."""
        context = ExecutionContext()
        
        # Allocate resource
        success = context.allocate_resource("cpu", "cpu1", {"cores": 4})
        self.assertTrue(success)
        
        # Try to allocate again
        success = context.allocate_resource("cpu", "cpu1")
        self.assertFalse(success)  # Already allocated
        
        # Release resource
        success = context.release_resource("cpu", "cpu1")
        self.assertTrue(success)
        
        # Check resource usage
        usage = context.get_resource_usage()
        self.assertEqual(usage['available'], 1)
    
    def test_transaction_management(self):
        """Test transaction handling."""
        db = Mock()
        context = ExecutionContext(
            transaction_mode=TransactionMode.PER_BATCH,
            database=db
        )
        
        # Start transaction
        success = context.start_transaction("txn1")
        self.assertTrue(success)
        self.assertIsNotNone(context.current_transaction)
        
        # Log operations
        context.log_operation("insert", {"record": 1})
        self.assertEqual(len(context.current_transaction.operations), 1)
        
        # Commit transaction
        success = context.commit_transaction()
        self.assertTrue(success)
        self.assertIsNone(context.current_transaction)
        self.assertEqual(len(context.transaction_history), 1)
    
    def test_batch_management(self):
        """Test batch data handling."""
        context = ExecutionContext(data_mode=DataMode.BATCH)
        
        # Add batch items
        context.add_batch_item({"id": 1})
        context.add_batch_item({"id": 2})
        
        self.assertEqual(len(context.batch_data), 2)
        
        # Add results
        context.add_batch_result({"id": 1, "processed": True})
        self.assertEqual(len(context.batch_results), 1)
        
        # Add error
        context.add_batch_error(1, ValueError("Test error"))
        self.assertEqual(len(context.batch_errors), 1)
    
    def test_child_context(self):
        """Test child context creation."""
        parent = ExecutionContext()
        parent.variables['key1'] = 'value1'
        
        child = parent.create_child_context("path1")
        
        self.assertTrue(child.is_child_context)
        self.assertEqual(child.parent_context, parent)
        self.assertEqual(child.variables['key1'], 'value1')
        
        # Modify child
        child.metadata['child_key'] = 'child_value'
        
        # Merge back
        success = parent.merge_child_context("path1")
        self.assertTrue(success)
        self.assertEqual(parent.metadata['child_key'], 'child_value')
    
    def test_context_clone(self):
        """Test context cloning."""
        original = ExecutionContext()
        original.current_state = "state1"
        original.variables['key'] = 'value'
        
        clone = original.clone()
        
        self.assertEqual(clone.current_state, original.current_state)
        self.assertEqual(clone.variables['key'], 'value')
        self.assertIsNot(clone.variables, original.variables)  # Deep copy
    
    def test_performance_stats(self):
        """Test performance statistics."""
        context = ExecutionContext()
        context.set_state("state1")
        context.function_call_count['func1'] = 5
        
        stats = context.get_performance_stats()
        
        self.assertEqual(stats['states_visited'], 0)  # History count
        self.assertEqual(stats['current_state'], "state1")
        self.assertEqual(stats['function_calls']['func1'], 5)
        self.assertIn('elapsed_time', stats)


class TestExecutionEngine(unittest.TestCase):
    """Tests for ExecutionEngine."""
    
    def setUp(self):
        """Set up test FSM."""
        self.fsm = FSM("test_fsm")
        self.registry = FunctionRegistry()
        
        # Register test functions
        self.registry.register("always_true", AlwaysTrueFunction())
        self.registry.register("always_false", AlwaysFalseFunction())
        self.registry.register("increment", IncrementFunction())
        self.registry.register("validate_positive", ValidatePositiveFunction())
        self.registry.register("double", DoubleFunction())
        
        self.fsm.function_registry = self.registry
        
        # Create simple network
        network = StateNetwork("main")
        network.add_state(State("start"), initial=True)
        network.add_state(State("process"))
        network.add_state(State("end"), final=True)
        
        # Add arcs
        arc1 = ArcDefinition(
            target_state="process",
            pre_test="validate_positive"
        )
        arc2 = ArcDefinition(
            target_state="end",
            transform="double"
        )
        
        network.add_arc("start", "process", pre_test="validate_positive")
        network.add_arc("process", "end")
        
        self.fsm.networks["test_fsm"] = network
        
        self.engine = ExecutionEngine(self.fsm)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ExecutionEngine(
            self.fsm,
            strategy=TraversalStrategy.BREADTH_FIRST,
            max_retries=5
        )
        
        self.assertEqual(engine.strategy, TraversalStrategy.BREADTH_FIRST)
        self.assertEqual(engine.max_retries, 5)
        self.assertEqual(engine.fsm, self.fsm)
    
    def test_single_mode_execution(self):
        """Test single record execution."""
        context = ExecutionContext(data_mode=DataMode.SINGLE)
        
        success, result = self.engine.execute(context, 5)
        
        self.assertTrue(success)
        self.assertEqual(context.current_state, "end")
    
    def test_batch_mode_execution(self):
        """Test batch mode execution."""
        context = ExecutionContext(data_mode=DataMode.BATCH)
        context.batch_data = [1, 2, 3, -1, 5]
        
        success, results = self.engine.execute(context)
        
        self.assertTrue(success)
        self.assertIn('results', results)
        self.assertIn('errors', results)
        self.assertEqual(len(results['results']), 4)  # 4 positive numbers
        self.assertEqual(len(results['errors']), 1)   # 1 negative number
    
    def test_traversal_strategies(self):
        """Test different traversal strategies."""
        # Depth-first
        self.engine.strategy = TraversalStrategy.DEPTH_FIRST
        context = ExecutionContext()
        success, _ = self.engine.execute(context, 1)
        self.assertTrue(success)
        
        # Breadth-first
        self.engine.strategy = TraversalStrategy.BREADTH_FIRST
        context = ExecutionContext()
        success, _ = self.engine.execute(context, 1)
        self.assertTrue(success)
        
        # Resource-optimized
        self.engine.strategy = TraversalStrategy.RESOURCE_OPTIMIZED
        context = ExecutionContext()
        success, _ = self.engine.execute(context, 1)
        self.assertTrue(success)
    
    def test_execution_hooks(self):
        """Test execution hooks."""
        pre_hook_called = False
        post_hook_called = False
        
        def pre_hook(context, arc):
            nonlocal pre_hook_called
            pre_hook_called = True
        
        def post_hook(context, arc):
            nonlocal post_hook_called
            post_hook_called = True
        
        self.engine.add_pre_transition_hook(pre_hook)
        self.engine.add_post_transition_hook(post_hook)
        
        context = ExecutionContext()
        self.engine.execute(context, 1)
        
        self.assertTrue(pre_hook_called)
        self.assertTrue(post_hook_called)
    
    def test_max_transitions_limit(self):
        """Test maximum transitions limit."""
        # Create circular network
        network = StateNetwork("circular")
        network.add_state(State("a"), initial=True)
        network.add_state(State("b"))
        network.add_arc("a", "b")
        network.add_arc("b", "a")
        
        self.fsm.networks["circular"] = network
        self.fsm.name = "circular"
        
        context = ExecutionContext()
        engine = ExecutionEngine(self.fsm)
        
        success, result = engine.execute(context, 1, max_transitions=10)
        
        self.assertFalse(success)
        self.assertIn("Maximum transitions", result)
    
    def test_execution_stats(self):
        """Test execution statistics."""
        context = ExecutionContext()
        self.engine.execute(context, 1)
        
        stats = self.engine.get_execution_stats()
        
        self.assertEqual(stats['executions'], 1)
        self.assertGreater(stats['transitions'], 0)
        self.assertEqual(stats['strategy'], TraversalStrategy.DEPTH_FIRST.value)


class TestNetworkExecutor(unittest.TestCase):
    """Tests for NetworkExecutor."""
    
    def setUp(self):
        """Set up test FSM with multiple networks."""
        self.fsm = FSM("main")
        self.registry = FunctionRegistry()
        
        # Register functions
        self.registry.register("increment", IncrementFunction())
        self.registry.register("double", DoubleFunction())
        self.fsm.function_registry = self.registry
        
        # Create main network
        main_network = StateNetwork("main")
        main_network.add_state(State("start"), initial=True)
        main_network.add_state(State("call_sub"))
        main_network.add_state(State("end"), final=True)
        
        # Add regular arcs
        main_network.add_arc("start", "call_sub")
        main_network.add_arc("call_sub", "end")  # Add arc to make end reachable
        
        # Also add a push arc for testing hierarchical execution
        # This would be used in hierarchical test but not affect basic flow
        push_arc = PushArc(
            target_state="sub",  # Target in sub-network
            target_network="sub",
            return_state="end"
        )
        # Store push_arc in metadata for hierarchical test
        self.push_arc = push_arc
        
        self.fsm.networks["main"] = main_network
        
        # Create sub-network
        sub_network = StateNetwork("sub")
        sub_network.add_state(State("sub_start"), initial=True)
        sub_network.add_state(State("sub_process"))
        sub_network.add_state(State("sub_end"), final=True)
        
        sub_network.add_arc("sub_start", "sub_process")
        sub_network.add_arc("sub_process", "sub_end")
        
        self.fsm.networks["sub"] = sub_network
        
        self.executor = NetworkExecutor(self.fsm)
    
    def test_network_execution(self):
        """Test basic network execution."""
        success, result = self.executor.execute_network("main", data=10)
        
        self.assertTrue(success)
        self.assertIsNotNone(result)
    
    def test_hierarchical_execution(self):
        """Test hierarchical network execution."""
        context = ExecutionContext()
        success, result = self.executor.execute_network("main", context, data=5)
        
        # Check that network stack was used
        self.assertTrue(success)
    
    def test_parallel_network_execution(self):
        """Test parallel network execution."""
        configs = [
            {'network_name': 'sub', 'data': 1},
            {'network_name': 'sub', 'data': 2},
            {'network_name': 'sub', 'data': 3},
        ]
        
        results = self.executor.execute_parallel_networks(configs)
        
        self.assertEqual(len(results), 3)
        for success, result in results:
            self.assertTrue(success)
    
    def test_network_validation(self):
        """Test network validation."""
        results = self.executor.validate_all_networks()
        
        self.assertIn('main', results)
        self.assertIn('sub', results)
        
        for network_name, (valid, errors) in results.items():
            self.assertTrue(valid, f"Network {network_name} validation failed: {errors}")
    
    def test_network_stats(self):
        """Test network statistics."""
        stats = self.executor.get_network_stats("main")
        
        self.assertIn('states', stats)
        self.assertIn('arcs', stats)
        self.assertIn('initial_states', stats)
        self.assertIn('final_states', stats)
        self.assertTrue(stats['is_valid'])
    
    def test_active_networks_tracking(self):
        """Test active networks tracking."""
        # Start execution in background (mock)
        self.executor._active_networks["test"] = ExecutionContext()
        
        active = self.executor.get_active_networks()
        self.assertIn("test", active)
        
        del self.executor._active_networks["test"]
        active = self.executor.get_active_networks()
        self.assertNotIn("test", active)


class TestBatchExecutor(unittest.TestCase):
    """Tests for BatchExecutor."""
    
    def setUp(self):
        """Set up test environment."""
        self.fsm = FSM("batch_test")
        self.registry = FunctionRegistry()
        
        self.registry.register("double", DoubleFunction())
        self.fsm.function_registry = self.registry
        
        # Create simple processing network
        network = StateNetwork("batch_test")
        network.add_state(State("start"), initial=True)
        network.add_state(State("process"))
        network.add_state(State("end"), final=True)
        
        network.add_arc("start", "process")
        network.add_arc("process", "end")
        
        self.fsm.networks["batch_test"] = network
        
        self.executor = BatchExecutor(self.fsm, parallelism=2, batch_size=10)
    
    def test_batch_execution(self):
        """Test batch processing."""
        items = list(range(10))
        
        results = self.executor.execute_batch(items)
        
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.result)
    
    def test_parallel_batch_execution(self):
        """Test parallel batch processing."""
        items = list(range(100))
        
        # Sequential
        self.executor.parallelism = 1
        start = time.time()
        seq_results = self.executor.execute_batch(items)
        seq_time = time.time() - start
        
        # Parallel
        self.executor.parallelism = 4
        start = time.time()
        par_results = self.executor.execute_batch(items)
        par_time = time.time() - start
        
        # Both should produce same number of results
        self.assertEqual(len(seq_results), len(par_results))
        self.assertEqual(len(seq_results), 100)
    
    def test_batch_progress_tracking(self):
        """Test progress tracking."""
        items = list(range(20))
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append({
                'completed': progress.completed,
                'progress': progress.progress
            })
        
        self.executor.progress_callback = progress_callback
        results = self.executor.execute_batch(items)
        
        self.assertGreater(len(progress_updates), 0)
        # Final progress should be 1.0
        self.assertEqual(progress_updates[-1]['completed'], 20)
    
    def test_batch_error_handling(self):
        """Test error handling in batch."""
        # Create items that will cause errors
        items = [1, "invalid", 3, None, 5]
        
        results = self.executor.execute_batch(items)
        
        success_count = sum(1 for r in results if r.success)
        error_count = sum(1 for r in results if not r.success)
        
        self.assertEqual(len(results), 5)
        # Some should succeed, some should fail
        self.assertGreater(success_count, 0)
    
    def test_execute_batches(self):
        """Test processing in multiple batches."""
        items = list(range(25))
        self.executor.batch_size = 10
        
        aggregated = self.executor.execute_batches(items)
        
        self.assertEqual(aggregated['total'], 25)
        self.assertIn('succeeded', aggregated)
        self.assertIn('failed', aggregated)
        self.assertIn('success_rate', aggregated)
        self.assertIn('average_processing_time', aggregated)
    
    def test_benchmark(self):
        """Test benchmarking functionality."""
        items = list(range(50))
        
        configurations = [
            {'name': 'sequential', 'parallelism': 1, 'batch_size': 10},
            {'name': 'parallel_2', 'parallelism': 2, 'batch_size': 10},
            {'name': 'parallel_4', 'parallelism': 4, 'batch_size': 5},
        ]
        
        results = self.executor.create_benchmark(items, configurations)
        
        self.assertEqual(len(results), 3)
        for name, metrics in results.items():
            self.assertIn('elapsed_time', metrics)
            self.assertIn('throughput', metrics)
            self.assertIn('success_rate', metrics)


class TestStreamExecutor(unittest.TestCase):
    """Tests for StreamExecutor."""
    
    def setUp(self):
        """Set up test environment."""
        self.fsm = FSM("stream_test")
        self.registry = FunctionRegistry()
        
        self.registry.register("increment", IncrementFunction())
        self.fsm.function_registry = self.registry
        
        # Create processing network
        network = StateNetwork("stream_test")
        network.add_state(State("start"), initial=True)
        network.add_state(State("process"))
        network.add_state(State("end"), final=True)
        
        network.add_arc("start", "process")
        network.add_arc("process", "end")
        
        self.fsm.networks["stream_test"] = network
        
        self.executor = StreamExecutor(self.fsm)
    
    def test_stream_execution(self):
        """Test stream processing."""
        # Create test stream
        data_chunks = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        source = TestStreamSource(data_chunks)
        sink = TestStreamSink()
        
        pipeline = StreamPipeline(source=source, sink=sink)
        
        stats = self.executor.execute_stream(pipeline)
        
        self.assertEqual(stats['chunks_processed'], 3)
        self.assertEqual(stats['records_processed'], 9)
        self.assertEqual(len(sink.all_data), 9)
    
    def test_stream_transformations(self):
        """Test stream with transformations."""
        data_chunks = [[1, 2, 3]]
        source = TestStreamSource(data_chunks)
        sink = TestStreamSink()
        
        # Add transformation
        def multiply_by_10(x):
            return x * 10
        
        pipeline = StreamPipeline(
            source=source,
            sink=sink,
            transformations=[multiply_by_10]
        )
        
        stats = self.executor.execute_stream(pipeline)
        
        self.assertEqual(stats['records_processed'], 3)
        # Check transformed values
        self.assertEqual(sink.all_data[0], 10)
        self.assertEqual(sink.all_data[1], 20)
        self.assertEqual(sink.all_data[2], 30)
    
    def test_stream_progress_tracking(self):
        """Test stream progress tracking."""
        data_chunks = [[1], [2], [3]]
        source = TestStreamSource(data_chunks)
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.chunks_processed)
        
        self.executor.progress_callback = progress_callback
        
        pipeline = StreamPipeline(source=source)
        stats = self.executor.execute_stream(pipeline)
        
        self.assertEqual(len(progress_updates), 3)
        self.assertEqual(progress_updates, [1, 2, 3])
    
    def test_stream_error_handling(self):
        """Test error handling in stream."""
        # Create chunks with invalid data
        data_chunks = [[1, "invalid", 3]]
        source = TestStreamSource(data_chunks)
        
        pipeline = StreamPipeline(source=source)
        stats = self.executor.execute_stream(pipeline)
        
        # Some errors should be recorded
        self.assertGreater(stats['errors'], 0)
        self.assertIn('error_details', stats)
    
    def test_backpressure_handling(self):
        """Test backpressure mechanism."""
        config = StreamConfig(
            chunk_size=10,
            backpressure_threshold=2
        )
        
        executor = StreamExecutor(
            self.fsm,
            stream_config=config,
            enable_backpressure=True
        )
        
        # Create large stream
        data_chunks = [[i] * 10 for i in range(10)]
        source = TestStreamSource(data_chunks)
        
        pipeline = StreamPipeline(source=source)
        stats = executor.execute_stream(pipeline)
        
        self.assertEqual(stats['chunks_processed'], 10)
    
    def test_multi_stage_pipeline(self):
        """Test multi-stage pipeline creation."""
        stages = [
            {'type': 'source', 'source': TestStreamSource([[1, 2, 3]])},
            {'type': 'transform', 'function': lambda x: x * 2},
            {'type': 'transform', 'function': lambda x: x + 1},
            {'type': 'sink', 'sink': TestStreamSink()},
        ]
        
        pipeline = self.executor.create_multi_stage_pipeline(stages)
        
        self.assertIsNotNone(pipeline.source)
        self.assertIsNotNone(pipeline.sink)
        self.assertEqual(len(pipeline.transformations), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 4 components."""
    
    def test_end_to_end_batch_processing(self):
        """Test end-to-end batch processing."""
        # Create FSM
        fsm = FSM("integration")
        registry = FunctionRegistry()
        registry.register("validate_positive", ValidatePositiveFunction())
        registry.register("double", DoubleFunction())
        fsm.function_registry = registry
        
        # Create network
        network = StateNetwork("integration")
        network.add_state(State("validate"), initial=True)
        network.add_state(State("transform"))
        network.add_state(State("complete"), final=True)
        
        network.add_arc("validate", "transform", pre_test="validate_positive")
        network.add_arc("transform", "complete")
        
        fsm.networks["integration"] = network
        
        # Process batch
        executor = BatchExecutor(fsm, parallelism=4)
        items = list(range(-5, 15))
        
        results = executor.execute_batches(items)
        
        # Only positive numbers should succeed
        self.assertEqual(results['total'], 20)
        self.assertEqual(results['succeeded'], 14)  # 1-14 are positive
        self.assertEqual(results['failed'], 6)       # -5 to 0 fail
    
    def test_stream_to_database(self):
        """Test streaming data to database."""
        # Create database
        factory = DatabaseFactory()
        db = factory.create(backend="memory")
        
        # Create FSM
        fsm = FSM("db_stream")
        registry = FunctionRegistry()
        registry.register("increment", IncrementFunction())
        fsm.function_registry = registry
        
        network = StateNetwork("db_stream")
        network.add_state(State("start"), initial=True)
        network.add_state(State("end"), final=True)
        network.add_arc("start", "end")
        
        fsm.networks["db_stream"] = network
        
        # Create stream pipeline
        from dataknobs_fsm.streaming.db_stream import DatabaseStreamSink
        
        data_chunks = [[1, 2, 3], [4, 5, 6]]
        source = TestStreamSource(data_chunks)
        sink = DatabaseStreamSink(db, batch_size=2)
        
        pipeline = StreamPipeline(source=source, sink=sink)
        
        # Execute
        executor = StreamExecutor(fsm)
        stats = executor.execute_stream(pipeline)
        
        # Verify data in database
        self.assertEqual(stats['chunks_processed'], 2)
        self.assertEqual(stats['records_processed'], 6)
    
    def test_hierarchical_network_with_context(self):
        """Test hierarchical networks with context passing."""
        # Create complex FSM
        fsm = FSM("hierarchical")
        registry = FunctionRegistry()
        registry.register("increment", IncrementFunction())
        fsm.function_registry = registry
        
        # Main network
        main = StateNetwork("hierarchical")
        main.add_state(State("init"), initial=True)
        main.add_state(State("delegate"))
        main.add_state(State("finalize"), final=True)
        
        # Add arcs
        main.add_arc("init", "delegate")
        
        # Add push arc
        push_arc = PushArc(
            target_state="worker",  # Target network name
            target_network="worker",
            return_state="finalize",
            isolation_mode=DataIsolationMode.COPY  # Use proper enum
        )
        from dataknobs_fsm.core.network import Arc
        arc = Arc(source_state="delegate", target_state="worker")
        arc.metadata = {"push_arc": push_arc, "arc_type": "push"}
        main._arcs.append(arc)
        
        fsm.networks["hierarchical"] = main
        
        # Worker network
        worker = StateNetwork("worker")
        worker.add_state(State("work_start"), initial=True)
        worker.add_state(State("work_end"), final=True)
        worker.add_arc("work_start", "work_end")
        
        fsm.networks["worker"] = worker
        
        # Execute
        executor = NetworkExecutor(fsm)
        context = ExecutionContext()
        context.variables['shared'] = 'data'
        
        success, result = executor.execute_network(
            "hierarchical",
            context,
            data=100
        )
        
        self.assertTrue(success)
        self.assertEqual(context.variables['shared'], 'data')
    
    def test_resource_aware_execution(self):
        """Test resource-aware execution."""
        # Create FSM with resource requirements
        fsm = FSM("resource_test")
        registry = FunctionRegistry()
        fsm.function_registry = registry
        
        network = StateNetwork("resource_test")
        network.add_state(State("start"), initial=True)
        network.add_state(State("heavy_compute"))
        network.add_state(State("end"), final=True)
        
        # Add arc with resource requirements
        # Note: We'll add resource requirements to metadata since Arc doesn't have that field
        from dataknobs_fsm.core.network import Arc
        arc = Arc(source_state="start", target_state="heavy_compute")
        arc.metadata = {"resource_requirements": {"cpu": 4, "memory": 1024}}
        network._arcs.append(arc)
        
        network.add_arc("heavy_compute", "end")
        
        fsm.networks["resource_test"] = network
        
        # Execute with resource constraints
        engine = ExecutionEngine(
            fsm,
            strategy=TraversalStrategy.RESOURCE_OPTIMIZED
        )
        
        context = ExecutionContext(
            resources={"cpu": 8, "memory": 2048}
        )
        
        success, result = engine.execute(context, "test_data")
        
        self.assertTrue(success)
        
        # Check resource usage
        usage = context.get_resource_usage()
        self.assertIsNotNone(usage)


if __name__ == '__main__':
    unittest.main()