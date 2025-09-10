"""Tests for ResultFormatter module."""

import pytest
from dataknobs_fsm.core.result_formatter import ResultFormatter
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.core.modes import ProcessingMode


class TestResultFormatter:
    """Test suite for ResultFormatter."""
    
    @pytest.fixture
    def basic_context(self):
        """Create a basic execution context for testing."""
        context = ExecutionContext(
            data_mode=ProcessingMode.SINGLE
        )
        context.data = {"test": "data", "value": 42}
        context.current_state = "processing"
        context.state_history = ["start", "validate"]
        context.metadata = {"execution_id": "test123"}
        return context
    
    @pytest.fixture
    def batch_context(self):
        """Create a batch execution context for testing."""
        context = ExecutionContext(
            data_mode=ProcessingMode.BATCH
        )
        context.batch_data = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"}
        ]
        context.batch_errors = [(1, Exception("Error on item 2"))]
        context.current_state = "completed"
        return context
    
    @pytest.fixture
    def stream_context(self):
        """Create a stream execution context for testing."""
        context = ExecutionContext(
            data_mode=ProcessingMode.STREAM
        )
        context.processed_chunks = 5
        context.current_state = "streaming"
        context.metadata = {"stream_id": "stream_001"}
        return context
    
    def test_format_single_result_success(self, basic_context):
        """Test formatting a successful single execution result."""
        result = ResultFormatter.format_single_result(
            context=basic_context,
            success=True,
            result="Success"
        )
        
        assert result['success'] is True
        assert result['final_state'] == "processing"
        assert result['data'] == {"test": "data", "value": 42}
        assert result['path'] == ["start", "validate", "processing"]
        assert result['error'] is None
        assert result['metadata'] == {"execution_id": "test123"}
    
    def test_format_single_result_failure(self, basic_context):
        """Test formatting a failed single execution result."""
        result = ResultFormatter.format_single_result(
            context=basic_context,
            success=False,
            result="Validation failed"
        )
        
        assert result['success'] is False
        assert result['error'] == "Validation failed"
        assert result['final_state'] == "processing"
        assert result['path'] == ["start", "validate", "processing"]
    
    def test_format_single_result_with_exception(self, basic_context):
        """Test formatting a result with an exception."""
        error = ValueError("Invalid input data")
        
        result = ResultFormatter.format_single_result(
            context=basic_context,
            success=False,
            error=error
        )
        
        assert result['success'] is False
        assert result['error'] == "Invalid input data"
        assert result['final_state'] == "processing"
    
    def test_format_batch_result(self, batch_context):
        """Test formatting batch execution results."""
        batch_results = [
            {"final_state": "done", "data": {"id": 1, "result": "processed"}},
            {"final_state": "error", "data": {"id": 2}},
            {"final_state": "done", "data": {"id": 3, "result": "processed"}}
        ]
        
        formatted = ResultFormatter.format_batch_result(
            context=batch_context,
            batch_results=batch_results
        )
        
        assert len(formatted) == 3
        
        # Check first item (success)
        assert formatted[0]['index'] == 0
        assert formatted[0]['success'] is True
        assert formatted[0]['final_state'] == "done"
        assert formatted[0]['error'] is None
        
        # Check second item (has error)
        assert formatted[1]['index'] == 1
        assert formatted[1]['success'] is False
        assert formatted[1]['final_state'] == "error"
        assert "Error on item 2" in formatted[1]['error']
        
        # Check third item (success)
        assert formatted[2]['index'] == 2
        assert formatted[2]['success'] is True
        assert formatted[2]['final_state'] == "done"
    
    def test_format_batch_result_empty(self):
        """Test formatting empty batch results."""
        context = ExecutionContext(data_mode=ProcessingMode.BATCH)
        
        formatted = ResultFormatter.format_batch_result(
            context=context,
            batch_results=[]
        )
        
        assert formatted == []
    
    def test_format_stream_result(self, stream_context):
        """Test formatting stream chunk results."""
        chunk_data = {"chunk_id": 100, "records": 50}
        
        result = ResultFormatter.format_stream_result(
            context=stream_context,
            chunk_result=chunk_data,
            chunk_index=10
        )
        
        assert result['chunk_index'] == 10
        assert result['chunks_processed'] == 5
        assert result['current_state'] == "streaming"
        assert result['data'] == chunk_data
        assert result['metadata'] == {"stream_id": "stream_001"}
    
    def test_format_async_result(self, basic_context):
        """Test formatting async execution results."""
        # Should be identical to format_single_result
        result = ResultFormatter.format_async_result(
            context=basic_context,
            success=True,
            result="Async success"
        )
        
        assert result['success'] is True
        assert result['final_state'] == "processing"
        assert result['error'] is None
    
    def test_format_step_result(self, basic_context):
        """Test formatting step-by-step execution results."""
        basic_context.previous_state = "validate"
        
        result = ResultFormatter.format_step_result(
            context=basic_context,
            new_state="completed",
            transition_taken=True
        )
        
        assert result['previous_state'] == "validate"
        assert result['current_state'] == "processing"
        assert result['new_state'] == "completed"
        assert result['transition_taken'] is True
        assert result['path'] == ["start", "validate", "processing"]
        assert result['data'] == {"test": "data", "value": 42}
    
    def test_format_step_result_no_transition(self, basic_context):
        """Test formatting step result when no transition occurred."""
        result = ResultFormatter.format_step_result(
            context=basic_context,
            new_state=None,
            transition_taken=False
        )
        
        assert result['new_state'] is None
        assert result['transition_taken'] is False
        assert result['current_state'] == "processing"
    
    def test_format_error_result(self, basic_context):
        """Test formatting error results with context."""
        error = RuntimeError("State execution failed")
        
        result = ResultFormatter.format_error_result(
            context=basic_context,
            error=error,
            error_state="processing"
        )
        
        assert result['success'] is False
        assert result['error'] == "State execution failed"
        assert result['error_type'] == "RuntimeError"
        assert result['error_state'] == "processing"
        assert result['final_state'] == "processing"
        assert result['path'] == ["start", "validate", "processing"]
    
    def test_format_error_result_without_error_state(self, basic_context):
        """Test formatting error result without explicit error state."""
        error = Exception("General error")
        
        result = ResultFormatter.format_error_result(
            context=basic_context,
            error=error
        )
        
        assert result['error_state'] == "processing"  # Uses current state
        assert result['error'] == "General error"
    
    def test_get_complete_path_empty_history(self):
        """Test getting complete path with empty history."""
        context = ExecutionContext()
        context.current_state = "only_state"
        
        path = ResultFormatter._get_complete_path(context)
        
        assert path == ["only_state"]
    
    def test_get_complete_path_with_history(self):
        """Test getting complete path with state history."""
        context = ExecutionContext()
        context.state_history = ["start", "middle"]
        context.current_state = "end"
        
        path = ResultFormatter._get_complete_path(context)
        
        assert path == ["start", "middle", "end"]
    
    def test_get_complete_path_current_in_history(self):
        """Test getting path when current state is already in history."""
        context = ExecutionContext()
        context.state_history = ["start", "middle", "end"]
        context.current_state = "end"
        
        path = ResultFormatter._get_complete_path(context)
        
        # Should not duplicate the current state
        assert path == ["start", "middle", "end"]
    
    def test_get_complete_path_no_current_state(self):
        """Test getting path when there's no current state."""
        context = ExecutionContext()
        context.state_history = ["start", "middle"]
        context.current_state = None
        
        path = ResultFormatter._get_complete_path(context)
        
        assert path == ["start", "middle"]
    
    def test_format_performance_result(self, basic_context):
        """Test formatting result with performance metrics."""
        base_result = {
            'success': True,
            'final_state': 'completed',
            'data': {'processed': True}
        }
        
        result = ResultFormatter.format_performance_result(
            context=basic_context,
            result=base_result
        )
        
        assert 'performance' in result
        assert 'resource_usage' in result
        assert result['success'] is True
        assert result['final_state'] == 'completed'
        
        # Check performance stats structure
        perf = result['performance']
        assert 'elapsed_time' in perf
        assert 'states_visited' in perf
        assert 'current_state' in perf
        
        # Check resource usage structure
        resources = result['resource_usage']
        assert 'total_resources' in resources
        assert 'allocated' in resources
    
    def test_format_with_empty_metadata(self):
        """Test formatting when context has no metadata."""
        context = ExecutionContext()
        context.data = {"test": "data"}
        context.current_state = "state1"
        
        result = ResultFormatter.format_single_result(
            context=context,
            success=True
        )
        
        assert result['metadata'] == {}
    
    def test_format_batch_with_no_errors(self):
        """Test batch formatting when there are no errors."""
        context = ExecutionContext(data_mode=ProcessingMode.BATCH)
        context.batch_errors = []  # No errors
        
        batch_results = [
            {"final_state": "done", "data": {"id": i}}
            for i in range(3)
        ]
        
        formatted = ResultFormatter.format_batch_result(
            context=context,
            batch_results=batch_results
        )
        
        # All should be successful
        for result in formatted:
            assert result['success'] is True
            assert result['error'] is None