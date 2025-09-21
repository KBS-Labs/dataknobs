"""Tests for FSM-specific exception types (core/exceptions.py).

This test suite covers:
- CircuitBreakerError with wait_time
- ETLError for ETL operations
- BulkheadTimeoutError for queue timeouts
- Exception propagation in patterns
- Exception serialization and context preservation
"""

import pytest
import json
import pickle
from typing import Dict, Any

from dataknobs_fsm.core.exceptions import (
    FSMError,
    InvalidConfigurationError,
    StateExecutionError,
    TransitionError,
    ResourceError,
    ValidationError,
    TimeoutError,
    ConcurrencyError,
    CircuitBreakerError,
    ETLError,
    BulkheadTimeoutError
)


class TestFSMExceptionTypes:
    """Tests for FSM exception types."""
    
    def test_base_fsm_error(self):
        """Test base FSMError with message and details."""
        error = FSMError("Base error", {"key": "value", "code": 500})
        
        assert str(error) == "Base error"
        assert error.details == {"key": "value", "code": 500}
        
        # Test without details
        error2 = FSMError("Simple error")
        assert error2.details == {}
    
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError(
            "Invalid network configuration",
            {"network": "main", "reason": "no start state"}
        )
        
        assert isinstance(error, FSMError)
        assert "Invalid network configuration" in str(error)
        assert error.details["network"] == "main"
        assert error.details["reason"] == "no start state"
    
    def test_state_execution_error(self):
        """Test StateExecutionError with state name."""
        error = StateExecutionError(
            state_name="process",
            message="Transform function failed",
            details={"function": "validate_data", "error": "type mismatch"}
        )
        
        assert "State 'process' execution failed: Transform function failed" in str(error)
        assert error.state_name == "process"
        assert error.details["function"] == "validate_data"
    
    def test_transition_error(self):
        """Test TransitionError with from/to states."""
        error = TransitionError(
            from_state="start",
            to_state="process",
            message="Condition not met",
            details={"condition": "data.valid == true", "actual": False}
        )
        
        assert "Transition from 'start' to 'process' failed: Condition not met" in str(error)
        assert error.from_state == "start"
        assert error.to_state == "process"
        assert error.details["actual"] is False
    
    def test_resource_error(self):
        """Test ResourceError with resource ID."""
        error = ResourceError(
            resource_id="database_main",
            message="Connection failed",
            details={"host": "localhost", "port": 5432, "retries": 3}
        )
        
        assert "Resource 'database_main' error: Connection failed" in str(error)
        assert error.resource_id == "database_main"
        assert error.details["retries"] == 3
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Schema validation failed",
            {"field": "email", "constraint": "format", "value": "invalid"}
        )
        
        assert isinstance(error, FSMError)
        assert "Schema validation failed" in str(error)
        assert error.details["field"] == "email"
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError(
            "Operation timed out after 30 seconds",
            {"operation": "database_query", "timeout": 30}
        )
        
        assert isinstance(error, FSMError)
        assert "Operation timed out after 30 seconds" in str(error)
        assert error.details["timeout"] == 30
    
    def test_concurrency_error(self):
        """Test ConcurrencyError."""
        error = ConcurrencyError(
            "Maximum concurrent executions reached",
            {"max": 10, "current": 10, "queue_size": 5}
        )
        
        assert isinstance(error, FSMError)
        assert "Maximum concurrent executions reached" in str(error)
        assert error.details["max"] == 10
    
    def test_circuit_breaker_error_with_wait_time(self):
        """Test CircuitBreakerError with wait_time."""
        # Test with wait_time
        error = CircuitBreakerError(
            wait_time=5.5,
            details={"failures": 10, "threshold": 5}
        )
        
        assert "Circuit breaker is open (wait 5.5s)" in str(error)
        assert error.wait_time == 5.5
        assert error.details["failures"] == 10
        
        # Test without wait_time
        error2 = CircuitBreakerError(details={"state": "open"})
        assert str(error2) == "Circuit breaker is open"
        assert error2.wait_time is None
    
    def test_etl_error(self):
        """Test ETLError for ETL operations."""
        error = ETLError(
            "Transform stage failed",
            {"stage": "transform", "record": 1234, "pipeline": "customer_etl"}
        )
        
        assert isinstance(error, FSMError)
        assert "Transform stage failed" in str(error)
        assert error.details["stage"] == "transform"
        assert error.details["record"] == 1234
    
    def test_bulkhead_timeout_error(self):
        """Test BulkheadTimeoutError for queue timeouts."""
        error = BulkheadTimeoutError(
            "Queue timeout after 10 seconds",
            {"queue_size": 100, "position": 50, "timeout": 10}
        )
        
        assert isinstance(error, FSMError)
        assert "Queue timeout after 10 seconds" in str(error)
        assert error.details["queue_size"] == 100
        assert error.details["position"] == 50
    
    def test_exception_inheritance_chain(self):
        """Test that all exceptions inherit from FSMError."""
        exceptions = [
            InvalidConfigurationError("test"),
            StateExecutionError("state", "test"),
            TransitionError("from", "to", "test"),
            ResourceError("resource", "test"),
            ValidationError("test"),
            TimeoutError("test"),
            ConcurrencyError("test"),
            CircuitBreakerError(),
            ETLError("test"),
            BulkheadTimeoutError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, FSMError)
            assert isinstance(exc, Exception)
    
    def test_exception_serialization(self):
        """Test that exceptions can be serialized and deserialized."""
        # Test JSON serialization of exception details
        error = StateExecutionError(
            state_name="validate",
            message="Validation failed",
            details={
                "field": "email",
                "value": "test@example.com",
                "rules": ["required", "email"],
                "nested": {"key": "value"}
            }
        )
        
        # Serialize details to JSON
        json_details = json.dumps(error.details)
        restored_details = json.loads(json_details)
        
        assert restored_details == error.details
        assert restored_details["nested"]["key"] == "value"
    
    def test_exception_context_preservation(self):
        """Test that exception context is preserved through re-raising."""
        original_details = {"original": True, "step": 1}
        
        try:
            raise StateExecutionError("state1", "First error", original_details)
        except StateExecutionError as e:
            # Add more context
            e.details["step"] = 2
            e.details["additional"] = "info"
            
            # Re-raise with preserved context
            try:
                raise e
            except StateExecutionError as e2:
                assert e2.state_name == "state1"
                assert e2.details["original"] is True
                assert e2.details["step"] == 2
                assert e2.details["additional"] == "info"
    
    def test_exception_propagation_pattern(self):
        """Test exception propagation through function calls."""
        def inner_function():
            raise ResourceError("db", "Connection lost", {"retry": 3})
        
        def middle_function():
            try:
                inner_function()
            except ResourceError as e:
                # Add context and re-raise as different error
                raise StateExecutionError(
                    "data_load",
                    f"Failed due to resource error: {e}",
                    {"original_error": str(e), "resource": e.resource_id}
                )
        
        def outer_function():
            try:
                middle_function()
            except StateExecutionError as e:
                # Add more context
                e.details["handler"] = "outer_function"
                raise
        
        with pytest.raises(StateExecutionError) as exc_info:
            outer_function()
        
        error = exc_info.value
        assert error.state_name == "data_load"
        assert "resource error" in str(error).lower()
        assert error.details["resource"] == "db"
        assert error.details["handler"] == "outer_function"
    
    def test_exception_with_none_details(self):
        """Test exceptions handle None details gracefully."""
        error = FSMError("Test error", None)
        assert error.details == {}
        
        error2 = StateExecutionError("state", "message", None)
        assert error2.details == {}
    
    def test_circuit_breaker_error_formatting(self):
        """Test CircuitBreakerError message formatting."""
        # Test various wait times
        test_cases = [
            (0.1, "Circuit breaker is open (wait 0.1s)"),
            (1.0, "Circuit breaker is open (wait 1.0s)"),
            (10.5, "Circuit breaker is open (wait 10.5s)"),
            (None, "Circuit breaker is open"),
            (0, "Circuit breaker is open")
        ]
        
        for wait_time, expected_msg in test_cases:
            if wait_time:
                error = CircuitBreakerError(wait_time=wait_time)
            else:
                error = CircuitBreakerError()
            
            assert str(error) == expected_msg
    
    def test_exception_comparison(self):
        """Test that exceptions can be compared."""
        error1 = ValidationError("Field error", {"field": "name"})
        error2 = ValidationError("Field error", {"field": "name"})
        error3 = ValidationError("Different error", {"field": "email"})
        
        # Same message and type
        assert str(error1) == str(error2)
        assert type(error1) == type(error2)
        
        # Different message
        assert str(error1) != str(error3)
    
    def test_exception_details_modification(self):
        """Test that exception details can be modified after creation."""
        error = ETLError("Initial error", {"stage": "extract"})
        
        # Modify details
        error.details["records_processed"] = 100
        error.details["stage"] = "transform"
        
        assert error.details["records_processed"] == 100
        assert error.details["stage"] == "transform"
    
    def test_exception_chaining(self):
        """Test exception chaining with __cause__."""
        original = ValueError("Original error")
        
        try:
            raise original
        except ValueError as e:
            fsm_error = FSMError("FSM wrapper error", {"wrapped": True})
            fsm_error.__cause__ = e
            
            try:
                raise fsm_error
            except FSMError as final:
                assert final.__cause__ is original
                assert str(final.__cause__) == "Original error"
                assert final.details["wrapped"] is True
    
    def test_exception_str_representation(self):
        """Test string representation of exceptions."""
        test_cases = [
            (FSMError("Simple message"), "Simple message"),
            (InvalidConfigurationError("Bad config"), "Bad config"),
            (StateExecutionError("test_state", "Failed"), "State 'test_state' execution failed: Failed"),
            (TransitionError("A", "B", "No path"), "Transition from 'A' to 'B' failed: No path"),
            (ResourceError("res_1", "Timeout"), "Resource 'res_1' error: Timeout"),
            (CircuitBreakerError(wait_time=2.5), "Circuit breaker is open (wait 2.5s)"),
        ]
        
        for error, expected in test_cases:
            assert str(error) == expected