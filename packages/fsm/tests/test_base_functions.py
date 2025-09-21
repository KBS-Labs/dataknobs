"""Tests for base function interfaces and classes."""

import pytest
from typing import Any, Dict, Optional, Tuple

from dataknobs_fsm.functions.base import (
    FunctionType,
    ExecutionResult,
    IValidationFunction,
    ITransformFunction,
    IStateTestFunction,
    IEndStateTestFunction,
    ResourceStatus,
    ResourceConfig,
    ValidationError,
    TransformError,
    StateTransitionError,
    ResourceError,
    ConfigurationError,
    BaseFunction,
    CompositeFunction,
)


class TestExecutionResult:
    """Tests for ExecutionResult."""
    
    def test_success_result(self):
        """Test creating a successful result."""
        result = ExecutionResult.success_result(
            data={"key": "value"},
            metadata={"time": 100}
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.metadata == {"time": 100}
    
    def test_failure_result(self):
        """Test creating a failure result."""
        result = ExecutionResult.failure_result(
            error="Something went wrong",
            metadata={"code": 500}
        )
        
        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.metadata == {"code": 500}
    
    def test_direct_construction(self):
        """Test direct construction of ExecutionResult."""
        result = ExecutionResult(
            success=True,
            data="test",
            error=None,
            metadata={"test": True}
        )
        
        assert result.success is True
        assert result.data == "test"
        assert result.metadata == {"test": True}


class MockValidationFunction(IValidationFunction):
    """Mock implementation of validation function."""
    
    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        if self.should_pass:
            return ExecutionResult.success_result(data)
        else:
            return ExecutionResult.failure_result("Validation failed")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        return {"mock": True, "should_pass": self.should_pass}


class MockTransformFunction(ITransformFunction):
    """Mock implementation of transform function."""
    
    def __init__(self, transform_fn=None):
        self.transform_fn = transform_fn or (lambda x: x)
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        try:
            result = self.transform_fn(data)
            return ExecutionResult.success_result(result)
        except Exception as e:
            return ExecutionResult.failure_result(str(e))
    
    def get_transform_description(self) -> str:
        return "Mock transform function"


class MockStateTestFunction(IStateTestFunction):
    """Mock implementation of state test function."""
    
    def __init__(self, test_result: bool = True):
        self.test_result = test_result
    
    def test(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        if self.test_result:
            return True, "Test passed"
        else:
            return False, "Test failed"
    
    def get_test_description(self) -> str:
        return "Mock state test"


class MockEndStateTestFunction(IEndStateTestFunction):
    """Mock implementation of end state test function."""
    
    def __init__(self, should_end: bool = False):
        self.should_end_value = should_end
    
    def should_end(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        if self.should_end_value:
            return True, "End condition met"
        else:
            return False, "Continue processing"
    
    def get_end_condition(self) -> str:
        return "Mock end condition"


class TestFunctionImplementations:
    """Test the mock function implementations."""
    
    def test_validation_function_pass(self):
        """Test validation function that passes."""
        func = MockValidationFunction(should_pass=True)
        result = func.validate({"key": "value"})
        
        assert result.success is True
        assert result.data == {"key": "value"}
        
        rules = func.get_validation_rules()
        assert rules["should_pass"] is True
    
    def test_validation_function_fail(self):
        """Test validation function that fails."""
        func = MockValidationFunction(should_pass=False)
        result = func.validate({"key": "value"})
        
        assert result.success is False
        assert result.error == "Validation failed"
    
    def test_transform_function_success(self):
        """Test successful transformation."""
        func = MockTransformFunction(lambda x: x * 2)
        result = func.transform(5)
        
        assert result.success is True
        assert result.data == 10
    
    def test_transform_function_error(self):
        """Test transformation that raises error."""
        func = MockTransformFunction(lambda x: x / 0)
        result = func.transform(5)
        
        assert result.success is False
        assert "division by zero" in result.error.lower()
    
    def test_state_test_function_pass(self):
        """Test state test that passes."""
        func = MockStateTestFunction(test_result=True)
        passed, reason = func.test({"data": "test"})
        
        assert passed is True
        assert reason == "Test passed"
    
    def test_state_test_function_fail(self):
        """Test state test that fails."""
        func = MockStateTestFunction(test_result=False)
        passed, reason = func.test({"data": "test"})
        
        assert passed is False
        assert reason == "Test failed"
    
    def test_end_state_test_function_continue(self):
        """Test end state test that continues."""
        func = MockEndStateTestFunction(should_end=False)
        should_end, reason = func.should_end({"data": "test"})
        
        assert should_end is False
        assert reason == "Continue processing"
    
    def test_end_state_test_function_end(self):
        """Test end state test that ends."""
        func = MockEndStateTestFunction(should_end=True)
        should_end, reason = func.should_end({"data": "test"})
        
        assert should_end is True
        assert reason == "End condition met"


class TestExceptions:
    """Test custom exception classes."""
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Validation failed",
            validation_errors=["Field required", "Invalid format"]
        )
        
        assert str(error) == "Validation failed"
        assert error.validation_errors == ["Field required", "Invalid format"]
    
    def test_transform_error(self):
        """Test TransformError."""
        error = TransformError("Transform failed")
        assert str(error) == "Transform failed"
    
    def test_state_transition_error(self):
        """Test StateTransitionError."""
        error = StateTransitionError(
            "Cannot transition",
            from_state="state_a",
            to_state="state_b"
        )
        
        assert str(error) == "Cannot transition"
        assert error.from_state == "state_a"
        assert error.to_state == "state_b"
    
    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError(
            "Connection failed",
            resource_name="database",
            operation="connect"
        )
        
        assert str(error) == "Connection failed"
        assert error.resource_name == "database"
        assert error.operation == "connect"
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"


class TestBaseFunction:
    """Test BaseFunction class."""
    
    def test_initialization(self):
        """Test BaseFunction initialization."""
        func = BaseFunction("test_func", "Test description")
        
        assert func.name == "test_func"
        assert func.description == "Test description"
        assert func.execution_count == 0
        assert func.error_count == 0
    
    def test_record_execution(self):
        """Test recording execution statistics."""
        func = BaseFunction("test_func")
        
        func._record_execution(True)
        func._record_execution(True)
        func._record_execution(False)
        
        assert func.execution_count == 3
        assert func.error_count == 1
    
    def test_get_stats(self):
        """Test getting execution statistics."""
        func = BaseFunction("test_func")
        
        func._record_execution(True)
        func._record_execution(True)
        func._record_execution(False)
        func._record_execution(True)
        
        stats = func.get_stats()
        
        assert stats["executions"] == 4
        assert stats["errors"] == 1
        assert stats["success_rate"] == 0.75
    
    def test_get_stats_no_executions(self):
        """Test stats with no executions."""
        func = BaseFunction("test_func")
        stats = func.get_stats()
        
        assert stats["executions"] == 0
        assert stats["errors"] == 0
        assert stats["success_rate"] == 0


class TestCompositeFunction:
    """Test CompositeFunction class."""
    
    def test_initialization(self):
        """Test CompositeFunction initialization."""
        sub1 = BaseFunction("sub1")
        sub2 = BaseFunction("sub2")
        
        composite = CompositeFunction(
            "composite",
            [sub1, sub2],
            "Composite function"
        )
        
        assert composite.name == "composite"
        assert composite.description == "Composite function"
        assert len(composite.functions) == 2
    
    def test_add_function(self):
        """Test adding a function."""
        composite = CompositeFunction("composite", [])
        func = BaseFunction("new_func")
        
        composite.add_function(func)
        
        assert len(composite.functions) == 1
        assert composite.functions[0] == func
    
    def test_remove_function(self):
        """Test removing a function."""
        sub1 = BaseFunction("sub1")
        sub2 = BaseFunction("sub2")
        composite = CompositeFunction("composite", [sub1, sub2])
        
        removed = composite.remove_function("sub1")
        
        assert removed is True
        assert len(composite.functions) == 1
        assert composite.functions[0] == sub2
    
    def test_remove_nonexistent_function(self):
        """Test removing a function that doesn't exist."""
        composite = CompositeFunction("composite", [])
        
        removed = composite.remove_function("nonexistent")
        
        assert removed is False


class TestResourceConfig:
    """Test ResourceConfig dataclass."""
    
    def test_resource_config(self):
        """Test creating ResourceConfig."""
        config = ResourceConfig(
            name="test_db",
            type="postgresql",
            connection_params={"host": "localhost", "port": 5432},
            pool_size=10,
            timeout=30.0,
            retry_policy={"max_retries": 3},
            health_check_interval=60.0
        )
        
        assert config.name == "test_db"
        assert config.type == "postgresql"
        assert config.connection_params["host"] == "localhost"
        assert config.pool_size == 10
        assert config.timeout == 30.0
        assert config.retry_policy["max_retries"] == 3
        assert config.health_check_interval == 60.0
    
    def test_resource_config_minimal(self):
        """Test creating ResourceConfig with minimal params."""
        config = ResourceConfig(
            name="simple",
            type="memory",
            connection_params={}
        )
        
        assert config.name == "simple"
        assert config.type == "memory"
        assert config.pool_size is None
        assert config.timeout is None