"""Tests for state definitions and instances."""

import pytest
from datetime import datetime
from typing import Any, Dict, Optional
from time import sleep

from dataknobs_data.fields import Field, FieldType
from dataknobs_fsm.core.state import (
    StateType,
    StateStatus,
    StateSchema,
    StateDefinition,
    StateInstance,
)
from dataknobs_fsm.core.data_modes import DataHandlingMode, DataModeManager
from dataknobs_fsm.functions.base import (
    IValidationFunction,
    ITransformFunction,
    ExecutionResult,
)


class MockValidationFunction(IValidationFunction):
    """Mock validation function for testing."""
    
    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        if self.should_pass:
            return ExecutionResult.success_result(data)
        return ExecutionResult.failure_result("Validation failed")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        return {"mock": True}


class MockTransformFunction(ITransformFunction):
    """Mock transform function for testing."""
    
    def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        if isinstance(data, dict):
            data["transformed"] = True
        return ExecutionResult.success_result(data)
    
    def get_transform_description(self) -> str:
        return "Mock transform"


class TestStateSchema:
    """Tests for StateSchema."""
    
    def test_schema_creation(self):
        """Test creating a state schema."""
        schema = StateSchema(
            fields=[
                Field("name", "test", FieldType.STRING),
                Field("age", 25, FieldType.INTEGER),
            ],
            required_fields={"name"},
            allow_extra_fields=True
        )
        
        assert len(schema.fields) == 2
        assert "name" in schema.required_fields
        assert schema.allow_extra_fields is True
    
    def test_validate_valid_data(self):
        """Test validating valid data."""
        schema = StateSchema(
            fields=[
                Field("name", "", FieldType.STRING),
                Field("age", 0, FieldType.INTEGER),
            ],
            required_fields={"name"}
        )
        
        data = {"name": "Alice", "age": 30}
        is_valid, errors = schema.validate(data)
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_missing_required(self):
        """Test validation with missing required fields."""
        schema = StateSchema(
            fields=[Field("name", "", FieldType.STRING)],
            required_fields={"name", "email"}
        )
        
        data = {"name": "Alice"}
        is_valid, errors = schema.validate(data)
        
        assert is_valid is False
        assert "Required field 'email' is missing" in errors
    
    def test_validate_wrong_type(self):
        """Test validation with wrong field type."""
        schema = StateSchema(
            fields=[Field("age", 0, FieldType.INTEGER)]
        )
        
        data = {"age": "not a number"}
        is_valid, errors = schema.validate(data)
        
        assert is_valid is False
        assert any("invalid type" in error for error in errors)
    
    def test_validate_extra_fields_allowed(self):
        """Test validation with extra fields allowed."""
        schema = StateSchema(
            fields=[Field("name", "", FieldType.STRING)],
            allow_extra_fields=True
        )
        
        data = {"name": "Alice", "extra": "value"}
        is_valid, errors = schema.validate(data)
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_extra_fields_disallowed(self):
        """Test validation with extra fields disallowed."""
        schema = StateSchema(
            fields=[Field("name", "", FieldType.STRING)],
            allow_extra_fields=False
        )
        
        data = {"name": "Alice", "extra": "value"}
        is_valid, errors = schema.validate(data)
        
        assert is_valid is False
        assert "Unexpected field 'extra'" in errors


class TestStateDefinition:
    """Tests for StateDefinition."""
    
    def test_basic_creation(self):
        """Test creating a basic state definition."""
        state = StateDefinition(
            name="test_state",
            type=StateType.NORMAL,
            description="Test state"
        )
        
        assert state.name == "test_state"
        assert state.type == StateType.NORMAL
        assert state.description == "Test state"
    
    def test_start_state(self):
        """Test identifying start states."""
        state = StateDefinition(name="start", type=StateType.START)
        assert state.is_start_state() is True
        assert state.is_end_state() is False
    
    def test_end_state(self):
        """Test identifying end states."""
        state = StateDefinition(name="end", type=StateType.END)
        assert state.is_start_state() is False
        assert state.is_end_state() is True
    
    def test_add_validation_function(self):
        """Test adding validation functions."""
        state = StateDefinition(name="test")
        func = MockValidationFunction()
        
        state.add_validation_function(func)
        
        assert len(state.validation_functions) == 1
        assert state.validation_functions[0] == func
    
    def test_add_transform_function(self):
        """Test adding transform functions."""
        state = StateDefinition(name="test")
        func = MockTransformFunction()
        
        state.add_transform_function(func)
        
        assert len(state.transform_functions) == 1
        assert state.transform_functions[0] == func
    
    def test_validate_data_with_schema(self):
        """Test data validation with schema."""
        schema = StateSchema(
            fields=[Field("value", 0, FieldType.INTEGER)],
            required_fields={"value"}
        )
        state = StateDefinition(name="test", schema=schema)
        
        # Valid data
        is_valid, errors = state.validate_data({"value": 42})
        assert is_valid is True
        
        # Invalid data
        is_valid, errors = state.validate_data({"wrong": "field"})
        assert is_valid is False
    
    def test_validate_data_without_schema(self):
        """Test data validation without schema."""
        state = StateDefinition(name="test")
        
        is_valid, errors = state.validate_data({"any": "data"})
        
        assert is_valid is True
        assert errors == []
    
    def test_resource_requirements(self):
        """Test resource requirements."""
        from dataknobs_fsm.functions.base import ResourceConfig
        
        state = StateDefinition(
            name="test",
            resource_requirements=[
                ResourceConfig(
                    name="db",
                    type="database",
                    connection_params={"host": "localhost"}
                )
            ]
        )
        
        assert len(state.resource_requirements) == 1
        assert state.resource_requirements[0].name == "db"


class TestStateInstance:
    """Tests for StateInstance."""
    
    def test_basic_creation(self):
        """Test creating a state instance."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        assert instance.id is not None
        assert instance.definition == definition
        assert instance.status == StateStatus.PENDING
        assert instance.data == {}
    
    def test_enter_state(self):
        """Test entering a state."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        input_data = {"key": "value"}
        instance.enter(input_data)
        
        assert instance.status == StateStatus.ACTIVE
        assert instance.entry_time is not None
        assert instance.execution_count == 1
        # Default mode is COPY, so data should be copied
        assert instance.data == input_data
        assert instance.data is not input_data
    
    def test_exit_state(self):
        """Test exiting a state."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.enter({"key": "value"})
        result = instance.exit(commit=True)
        
        assert instance.status == StateStatus.COMPLETED
        assert instance.exit_time is not None
        assert result == {"key": "value"}
    
    def test_fail_state(self):
        """Test failing a state."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.enter({})
        instance.fail("Something went wrong")
        
        assert instance.status == StateStatus.FAILED
        assert instance.error_count == 1
        assert instance.last_error == "Something went wrong"
        assert instance.exit_time is not None
    
    def test_pause_resume(self):
        """Test pausing and resuming."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.enter({})
        instance.pause()
        assert instance.status == StateStatus.PAUSED
        
        instance.resume()
        assert instance.status == StateStatus.ACTIVE
    
    def test_skip(self):
        """Test skipping a state."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.skip()
        
        assert instance.status == StateStatus.SKIPPED
        assert instance.exit_time is not None
    
    def test_modify_data(self):
        """Test modifying state data."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.enter({"key1": "value1"})
        instance.modify_data({"key2": "value2", "key1": "modified"})
        
        assert instance.data["key1"] == "modified"
        assert instance.data["key2"] == "value2"
    
    def test_resource_management(self):
        """Test resource management."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        resource = {"connection": "db_conn"}
        instance.add_resource("database", resource)
        
        assert instance.get_resource("database") == resource
        assert instance.get_resource("nonexistent") is None
        
        instance.release_resources()
        assert instance.get_resource("database") is None
    
    def test_arc_execution_tracking(self):
        """Test arc execution tracking."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.record_arc_execution("arc1")
        instance.record_arc_execution("arc2")
        
        assert instance.executed_arcs == ["arc1", "arc2"]
    
    def test_get_duration(self):
        """Test getting execution duration."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        # No duration before entering
        assert instance.get_duration() is None
        
        # Duration while active
        instance.enter({})
        sleep(0.1)
        duration = instance.get_duration()
        assert duration is not None
        assert duration >= 0.1
        
        # Duration after exit
        instance.exit()
        final_duration = instance.get_duration()
        assert final_duration is not None
        assert final_duration >= 0.1
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        definition = StateDefinition(name="test")
        instance = StateInstance(definition=definition)
        
        instance.enter({"key": "value"})
        instance.next_state = "next"
        
        data = instance.to_dict()
        
        assert data["id"] == instance.id
        assert data["name"] == "test"
        assert data["status"] == "active"
        assert data["data"] == {"key": "value"}
        assert data["next_state"] == "next"
        assert data["entry_time"] is not None
    
    def test_data_mode_direct(self):
        """Test DIRECT data mode."""
        definition = StateDefinition(name="test", data_mode=DataHandlingMode.DIRECT)
        instance = StateInstance(definition=definition)
        
        input_data = {"key": "value"}
        instance.enter(input_data)
        
        # In DIRECT mode, data should be the same object
        assert instance.data is input_data
        
        instance.modify_data({"key": "modified"})
        assert input_data["key"] == "modified"
    
    def test_data_mode_reference(self):
        """Test REFERENCE data mode."""
        definition = StateDefinition(name="test", data_mode=DataHandlingMode.REFERENCE)
        instance = StateInstance(definition=definition)
        
        input_data = {"key": "value"}
        instance.enter(input_data)
        
        # In REFERENCE mode, data should be the same object
        assert instance.data is input_data
        
        instance.modify_data({"key": "modified"})
        assert input_data["key"] == "modified"


@pytest.mark.integration
class TestStateIntegration:
    """Integration tests for states."""
    
    def test_state_with_validation(self):
        """Test state with validation functions."""
        schema = StateSchema(
            fields=[Field("value", 0, FieldType.INTEGER)],
            required_fields={"value"}
        )
        
        definition = StateDefinition(
            name="validated_state",
            schema=schema,
            validation_functions=[MockValidationFunction(should_pass=True)]
        )
        
        instance = StateInstance(definition=definition)
        
        # Valid data
        is_valid, errors = definition.validate_data({"value": 42})
        assert is_valid is True
        
        instance.enter({"value": 42})
        assert instance.status == StateStatus.ACTIVE
    
    def test_state_with_transforms(self):
        """Test state with transform functions."""
        definition = StateDefinition(
            name="transform_state",
            transform_functions=[MockTransformFunction()]
        )
        
        instance = StateInstance(definition=definition)
        instance.enter({"original": "data"})
        
        # Apply transform (would be done by execution engine)
        transform = definition.transform_functions[0]
        result = transform.transform(instance.data)
        if result.success:
            instance.data = result.data
        
        assert "transformed" in instance.data
        assert instance.data["transformed"] is True
