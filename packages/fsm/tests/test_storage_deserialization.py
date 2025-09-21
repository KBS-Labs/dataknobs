"""Tests for ExecutionHistory deserialization."""

import time
import uuid
from typing import Any, Dict

import pytest

from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.execution.history import ExecutionHistory, ExecutionStep, ExecutionStatus
from dataknobs_structures import Tree


class TestExecutionStepDeserialization:
    """Tests for ExecutionStep.from_dict()."""
    
    def test_basic_step_deserialization(self):
        """Test deserializing a basic ExecutionStep."""
        step_data = {
            'step_id': 'test-123',
            'state_name': 'process',
            'network_name': 'main',
            'timestamp': 1234567890.5,
            'data_mode': 'copy',
            'status': 'completed',
            'start_time': 1234567890.5,
            'end_time': 1234567891.2,
            'arc_taken': 'success',
            'error': None,
            'metrics': {'processing_time': 0.7},
            'resource_usage': {'memory': 1024, 'cpu': 0.5}
        }
        
        step = ExecutionStep.from_dict(step_data)
        
        assert step.step_id == 'test-123'
        assert step.state_name == 'process'
        assert step.network_name == 'main'
        assert step.timestamp == 1234567890.5
        assert step.data_mode == DataHandlingMode.COPY
        assert step.status == ExecutionStatus.COMPLETED
        assert step.start_time == 1234567890.5
        assert step.end_time == 1234567891.2
        assert step.arc_taken == 'success'
        assert step.error is None
        assert step.metrics == {'processing_time': 0.7}
        assert step.resource_usage == {'memory': 1024, 'cpu': 0.5}
    
    def test_step_with_error_deserialization(self):
        """Test deserializing a step with an error."""
        step_data = {
            'step_id': 'error-456',
            'state_name': 'validate',
            'network_name': 'validation',
            'timestamp': 1234567890.0,
            'data_mode': 'reference',
            'status': 'failed',
            'error': 'Validation failed: invalid input'
        }
        
        step = ExecutionStep.from_dict(step_data)
        
        assert step.status == ExecutionStatus.FAILED
        assert step.error is not None
        assert str(step.error) == 'Validation failed: invalid input'
    
    def test_step_with_stream_data(self):
        """Test deserializing a step with streaming data."""
        step_data = {
            'step_id': 'stream-789',
            'state_name': 'stream_process',
            'network_name': 'streaming',
            'timestamp': 1234567890.0,
            'data_mode': 'direct',
            'status': 'in_progress',
            'stream_progress': 0.75,
            'chunks_processed': 150,
            'records_processed': 7500
        }
        
        step = ExecutionStep.from_dict(step_data)
        
        assert step.status == ExecutionStatus.IN_PROGRESS
        assert step.stream_progress == 0.75
        assert step.chunks_processed == 150
        assert step.records_processed == 7500
    
    def test_step_with_missing_optional_fields(self):
        """Test deserializing with minimal required fields."""
        step_data = {
            'step_id': 'minimal-001',
            'state_name': 'init',
            'network_name': 'default',
            'timestamp': 1234567890.0,
            'data_mode': 'copy',
            'status': 'pending'
        }
        
        step = ExecutionStep.from_dict(step_data)
        
        assert step.step_id == 'minimal-001'
        assert step.status == ExecutionStatus.PENDING
        assert step.arc_taken is None
        assert step.error is None
        assert step.metrics == {}
        assert step.resource_usage == {}
        assert step.chunks_processed == 0
        assert step.records_processed == 0
    
    def test_round_trip_serialization(self):
        """Test that to_dict() and from_dict() are inverses."""
        original = ExecutionStep(
            step_id='round-trip-123',
            state_name='transform',
            network_name='etl',
            timestamp=time.time(),
            data_mode=DataHandlingMode.REFERENCE,
            status=ExecutionStatus.COMPLETED
        )
        original.arc_taken = 'next'
        original.metrics = {'rows': 1000}
        original.resource_usage = {'threads': 4}
        original.stream_progress = 0.5
        original.chunks_processed = 50
        original.records_processed = 500
        
        # Serialize and deserialize
        serialized = original.to_dict()
        deserialized = ExecutionStep.from_dict(serialized)
        
        assert deserialized.step_id == original.step_id
        assert deserialized.state_name == original.state_name
        assert deserialized.network_name == original.network_name
        assert deserialized.timestamp == original.timestamp
        assert deserialized.data_mode == original.data_mode
        assert deserialized.status == original.status
        assert deserialized.arc_taken == original.arc_taken
        assert deserialized.metrics == original.metrics
        assert deserialized.resource_usage == original.resource_usage
        assert deserialized.stream_progress == original.stream_progress
        assert deserialized.chunks_processed == original.chunks_processed
        assert deserialized.records_processed == original.records_processed


class TestExecutionHistoryDeserialization:
    """Tests for ExecutionHistory.from_dict()."""
    
    def test_basic_history_deserialization(self):
        """Test deserializing a basic ExecutionHistory."""
        history_data = {
            'summary': {
                'fsm_name': 'test_fsm',
                'execution_id': 'exec-123',
                'data_mode': 'copy',
                'start_time': 1234567890.0,
                'end_time': 1234567900.0,
                'total_steps': 3,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': []
        }
        
        history = ExecutionHistory.from_dict(history_data)
        
        assert history.fsm_name == 'test_fsm'
        assert history.execution_id == 'exec-123'
        assert history.data_mode == DataHandlingMode.COPY
        assert history.start_time == 1234567890.0
        assert history.end_time == 1234567900.0
        assert history.total_steps == 3
        assert history.failed_steps == 0
        assert history.skipped_steps == 0
    
    def test_history_with_single_path(self):
        """Test deserializing history with a single execution path."""
        history_data = {
            'summary': {
                'fsm_name': 'linear_fsm',
                'execution_id': 'exec-456',
                'data_mode': 'reference',
                'start_time': 1234567890.0,
                'end_time': 1234567895.0,
                'total_steps': 3,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': [
                [
                    {
                        'step_id': 'step-1',
                        'state_name': 'start',
                        'network_name': 'main',
                        'timestamp': 1234567890.0,
                        'data_mode': 'reference',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'step-2',
                        'state_name': 'process',
                        'network_name': 'main',
                        'timestamp': 1234567892.0,
                        'data_mode': 'reference',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'step-3',
                        'state_name': 'end',
                        'network_name': 'main',
                        'timestamp': 1234567894.0,
                        'data_mode': 'reference',
                        'status': 'completed'
                    }
                ]
            ]
        }
        
        history = ExecutionHistory.from_dict(history_data)
        
        # Check tree structure was rebuilt
        assert len(history.tree_roots) == 1
        root = history.tree_roots[0]
        assert root.data.step_id == 'step-1'
        assert root.data.state_name == 'start'
        
        # Check children
        assert len(root.children) == 1
        child = root.children[0]
        assert child.data.step_id == 'step-2'
        assert child.data.state_name == 'process'
        
        # Check grandchild
        assert len(child.children) == 1
        grandchild = child.children[0]
        assert grandchild.data.step_id == 'step-3'
        assert grandchild.data.state_name == 'end'
        
        # Check current node points to last step
        assert history.current_node == grandchild
    
    def test_history_with_multiple_paths(self):
        """Test deserializing history with multiple execution paths."""
        history_data = {
            'summary': {
                'fsm_name': 'branching_fsm',
                'execution_id': 'exec-789',
                'data_mode': 'direct',
                'start_time': 1234567890.0,
                'end_time': None,
                'total_steps': 4,
                'failed_steps': 1,
                'skipped_steps': 0
            },
            'paths': [
                [
                    {
                        'step_id': 'step-1',
                        'state_name': 'start',
                        'network_name': 'main',
                        'timestamp': 1234567890.0,
                        'data_mode': 'direct',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'step-2a',
                        'state_name': 'branch_a',
                        'network_name': 'branch',
                        'timestamp': 1234567891.0,
                        'data_mode': 'direct',
                        'status': 'completed'
                    }
                ],
                [
                    {
                        'step_id': 'step-3',
                        'state_name': 'start',
                        'network_name': 'alt',
                        'timestamp': 1234567892.0,
                        'data_mode': 'direct',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'step-4',
                        'state_name': 'branch_b',
                        'network_name': 'alt',
                        'timestamp': 1234567893.0,
                        'data_mode': 'direct',
                        'status': 'failed',
                        'error': 'Processing error'
                    }
                ]
            ]
        }
        
        history = ExecutionHistory.from_dict(history_data)
        
        # Check multiple roots
        assert len(history.tree_roots) == 2
        
        # Check first path
        root1 = history.tree_roots[0]
        assert root1.data.step_id == 'step-1'
        assert len(root1.children) == 1
        assert root1.children[0].data.step_id == 'step-2a'
        
        # Check second path  
        root2 = history.tree_roots[1]
        assert root2.data.step_id == 'step-3'
        assert len(root2.children) == 1
        assert root2.children[0].data.step_id == 'step-4'
        assert root2.children[0].data.status == ExecutionStatus.FAILED
        assert str(root2.children[0].data.error) == 'Processing error'
    
    def test_mode_specific_storage_reconstruction(self):
        """Test that mode-specific storage is properly reconstructed."""
        history_data = {
            'summary': {
                'fsm_name': 'mixed_mode_fsm',
                'execution_id': 'exec-mixed',
                'data_mode': 'copy',
                'start_time': 1234567890.0,
                'end_time': 1234567900.0,
                'total_steps': 3,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': [
                [
                    {
                        'step_id': 'copy-1',
                        'state_name': 'copy_step',
                        'network_name': 'main',
                        'timestamp': 1234567890.0,
                        'data_mode': 'copy',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'ref-1',
                        'state_name': 'ref_step',
                        'network_name': 'main',
                        'timestamp': 1234567892.0,
                        'data_mode': 'reference',
                        'status': 'completed'
                    },
                    {
                        'step_id': 'direct-1',
                        'state_name': 'direct_step',
                        'network_name': 'main',
                        'timestamp': 1234567894.0,
                        'data_mode': 'direct',
                        'status': 'completed'
                    }
                ]
            ]
        }
        
        history = ExecutionHistory.from_dict(history_data)
        
        # Check mode-specific storage
        assert len(history._mode_storage[DataHandlingMode.COPY]) == 1
        assert history._mode_storage[DataHandlingMode.COPY][0].step_id == 'copy-1'
        
        assert len(history._mode_storage[DataHandlingMode.REFERENCE]) == 1
        assert history._mode_storage[DataHandlingMode.REFERENCE][0].step_id == 'ref-1'
        
        assert len(history._mode_storage[DataHandlingMode.DIRECT]) == 1
        assert history._mode_storage[DataHandlingMode.DIRECT][0].step_id == 'direct-1'
    
    def test_empty_history_deserialization(self):
        """Test deserializing an empty history."""
        history_data = {
            'summary': {
                'fsm_name': 'empty_fsm',
                'execution_id': 'exec-empty',
                'data_mode': 'copy',
                'start_time': 1234567890.0,
                'end_time': None,
                'total_steps': 0,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': []
        }
        
        history = ExecutionHistory.from_dict(history_data)
        
        assert history.fsm_name == 'empty_fsm'
        assert history.execution_id == 'exec-empty'
        assert history.total_steps == 0
        assert len(history.tree_roots) == 0
        assert history.current_node is None
    
    def test_large_history_deserialization_performance(self):
        """Test deserializing a large history for performance."""
        # Create a large history with many steps
        num_steps = 1000
        path_data = []
        
        for i in range(num_steps):
            path_data.append({
                'step_id': f'step-{i}',
                'state_name': f'state_{i}',
                'network_name': 'main',
                'timestamp': 1234567890.0 + i,
                'data_mode': 'copy',
                'status': 'completed',
                'metrics': {'index': i},
                'resource_usage': {'step': i}
            })
        
        history_data = {
            'summary': {
                'fsm_name': 'large_fsm',
                'execution_id': 'exec-large',
                'data_mode': 'copy',
                'start_time': 1234567890.0,
                'end_time': 1234567890.0 + num_steps,
                'total_steps': num_steps,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': [path_data]
        }
        
        # Measure deserialization time
        start_time = time.time()
        history = ExecutionHistory.from_dict(history_data)
        deserialize_time = time.time() - start_time
        
        # Should deserialize in reasonable time (< 1 second for 1000 steps)
        assert deserialize_time < 1.0
        
        # Verify structure
        assert history.total_steps == num_steps
        assert len(history.tree_roots) == 1
        
        # Verify deep nesting
        current = history.tree_roots[0]
        for i in range(min(10, num_steps - 1)):  # Check first 10 levels
            assert current.data.step_id == f'step-{i}'
            if current.children:
                current = current.children[0]
    
    def test_history_with_invalid_data_mode(self):
        """Test handling of invalid data mode."""
        history_data = {
            'summary': {
                'fsm_name': 'invalid_fsm',
                'execution_id': 'exec-invalid',
                'data_mode': 'INVALID_MODE',
                'start_time': 1234567890.0,
                'end_time': None,
                'total_steps': 0,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': []
        }
        
        with pytest.raises(ValueError):
            ExecutionHistory.from_dict(history_data)
    
    def test_history_with_invalid_status(self):
        """Test handling of invalid execution status."""
        history_data = {
            'summary': {
                'fsm_name': 'test_fsm',
                'execution_id': 'exec-123',
                'data_mode': 'copy',
                'start_time': 1234567890.0,
                'end_time': None,
                'total_steps': 1,
                'failed_steps': 0,
                'skipped_steps': 0
            },
            'paths': [
                [
                    {
                        'step_id': 'step-1',
                        'state_name': 'start',
                        'network_name': 'main',
                        'timestamp': 1234567890.0,
                        'data_mode': 'copy',
                        'status': 'INVALID_STATUS'
                    }
                ]
            ]
        }
        
        with pytest.raises(ValueError):
            ExecutionHistory.from_dict(history_data)
    
    def test_complete_round_trip(self):
        """Test complete round-trip serialization/deserialization."""
        # Create original history
        original = ExecutionHistory(
            fsm_name='round_trip_fsm',
            execution_id=str(uuid.uuid4()),
            data_mode=DataHandlingMode.REFERENCE,
            max_depth=10
        )
        
        # Add some steps
        step1 = original.add_step('start', 'main')
        original.update_step(step1.step_id, status=ExecutionStatus.COMPLETED)
        
        step2 = original.add_step('process', 'main')
        original.update_step(step2.step_id, 
                           status=ExecutionStatus.COMPLETED,
                           metrics={'rows': 100})
        
        step3 = original.add_step('end', 'main')
        original.update_step(step3.step_id, status=ExecutionStatus.COMPLETED)
        
        original.finalize()
        
        # Serialize and deserialize
        serialized = original.to_dict()
        deserialized = ExecutionHistory.from_dict(serialized)
        
        # Verify properties match
        assert deserialized.fsm_name == original.fsm_name
        assert deserialized.execution_id == original.execution_id
        assert deserialized.data_mode == original.data_mode
        assert deserialized.total_steps == original.total_steps
        assert deserialized.failed_steps == original.failed_steps
        assert deserialized.skipped_steps == original.skipped_steps
        
        # Verify tree structure matches
        assert len(deserialized.tree_roots) == len(original.tree_roots)
        
        # Verify step data by comparing paths
        original_paths = original.get_all_paths()
        deserialized_paths = deserialized.get_all_paths()
        
        assert len(deserialized_paths) == len(original_paths)
        
        # Compare first path (should only be one in this test)
        if original_paths and deserialized_paths:
            orig_path = original_paths[0]
            deser_path = deserialized_paths[0]
            
            assert len(deser_path) == len(orig_path)
            
            for orig, deser in zip(orig_path, deser_path):
                assert deser.step_id == orig.step_id
                assert deser.state_name == orig.state_name
                assert deser.network_name == orig.network_name
                assert deser.status == orig.status
                assert deser.metrics == orig.metrics