"""Tests for custom functions registration and usage in FSM."""

import pytest
from typing import Dict, Any

from dataknobs_fsm.api.simple import SimpleFSM, create_fsm
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import FunctionReference


class TestCustomFunctionRegistration:
    """Test custom function registration in FSMBuilder."""
    
    def test_register_function_in_builder(self):
        """Test registering a custom function in FSMBuilder."""
        def custom_func(state):
            return {"result": "custom"}
        
        builder = FSMBuilder()
        builder.register_function("my_custom_func", custom_func)
        
        assert builder._function_manager.has_function("my_custom_func")
        wrapper = builder._function_manager.get_function("my_custom_func")
        assert wrapper is not None
        assert wrapper.func == custom_func
    
    def test_resolve_registered_function(self):
        """Test resolving a registered function."""
        def custom_func(state):
            return {"result": "custom"}
        
        builder = FSMBuilder()
        builder.register_function("my_custom_func", custom_func)
        
        func_ref = FunctionReference(
            type="registered",
            name="my_custom_func"
        )
        
        resolved = builder._resolve_function(func_ref)
        # The new function manager returns a wrapper
        assert resolved is not None
        # Check if it's a wrapper with the original function
        if hasattr(resolved, 'func'):
            assert resolved.func == custom_func
        elif hasattr(resolved, 'wrapper') and hasattr(resolved.wrapper, 'func'):
            # InterfaceWrapper case
            assert resolved.wrapper.func == custom_func
        else:
            # Direct function (shouldn't happen with new system but handle it)
            assert resolved == custom_func
    
    def test_resolve_unregistered_function_fails(self):
        """Test that resolving an unregistered function raises error."""
        builder = FSMBuilder()
        
        func_ref = FunctionReference(
            type="registered",
            name="nonexistent_func"
        )
        
        with pytest.raises(ValueError, match="Registered function not found"):
            builder._resolve_function(func_ref)


class TestSimpleFSMCustomFunctions:
    """Test custom functions in SimpleFSM."""
    
    @pytest.fixture
    def simple_config(self):
        """Simple FSM config that uses custom functions."""
        return {
            "name": "test_custom",
            "main_network": "main",
            "data_mode": {"default": "direct"},
            "networks": [{
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True
                    },
                    {
                        "name": "transform",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "custom_transform"
                            }
                        }
                    },
                    {
                        "name": "end",
                        "is_end": True
                    }
                ],
                "arcs": [
                    {"from": "start", "to": "transform"},
                    {"from": "transform", "to": "end"}
                ]
            }]
        }
    
    def test_simple_fsm_with_custom_functions(self, simple_config):
        """Test SimpleFSM initialization with custom functions."""
        def custom_transform(state):
            data = state.data.copy()
            data["transformed"] = True
            data["value"] = data.get("value", 0) * 2
            return data
        
        fsm = SimpleFSM(
            simple_config,
            custom_functions={
                "custom_transform": custom_transform
            }
        )
        
        result = fsm.process({"value": 5})
        
        assert result["success"] is True
        assert result["final_state"] == "end"
        assert result["data"]["transformed"] is True
        assert result["data"]["value"] == 10
    
    def test_create_fsm_with_custom_functions(self, simple_config):
        """Test create_fsm factory with custom functions."""
        def custom_transform(state):
            data = state.data.copy()
            data["doubled"] = data.get("input", 0) * 2
            return data
        
        fsm = create_fsm(
            simple_config,
            custom_functions={
                "custom_transform": custom_transform
            }
        )
        
        result = fsm.process({"input": 7})
        
        assert result["success"] is True
        assert result["data"]["doubled"] == 14
    
    def test_multiple_custom_functions(self):
        """Test FSM with multiple custom functions."""
        config = {
            "name": "multi_custom",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "first",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "first_func"
                            }
                        }
                    },
                    {
                        "name": "second",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "second_func"
                            }
                        }
                    },
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "first"},
                    {"from": "first", "to": "second"},
                    {"from": "second", "to": "end"}
                ]
            }]
        }
        
        def first_func(state):
            data = state.data.copy()
            data["first"] = "processed"
            return data
        
        def second_func(state):
            data = state.data.copy()
            data["second"] = "processed"
            return data
        
        fsm = SimpleFSM(
            config,
            custom_functions={
                "first_func": first_func,
                "second_func": second_func
            }
        )
        
        result = fsm.process({})
        
        assert result["success"] is True
        assert result["data"]["first"] == "processed"
        assert result["data"]["second"] == "processed"


class TestValidationPipeline:
    """Test validation pipeline pattern with custom functions."""
    
    def test_validation_with_routing(self):
        """Test validation that routes to different end states."""
        config = {
            "name": "validator",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "validate",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "validate_data"
                            }
                        }
                    },
                    {"name": "valid_end", "is_end": True},
                    {"name": "invalid_end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "validate"},
                    {
                        "from": "validate",
                        "to": "valid_end",
                        "condition": {
                            "type": "inline",
                            "code": "data.get('is_valid', False)"
                        }
                    },
                    {
                        "from": "validate",
                        "to": "invalid_end",
                        "condition": {
                            "type": "inline",
                            "code": "not data.get('is_valid', False)"
                        }
                    }
                ]
            }]
        }
        
        def validate_data(state):
            data = state.data.copy()
            value = data.get("value", 0)
            data["is_valid"] = value > 0
            data["validation_message"] = "Valid" if data["is_valid"] else "Invalid: value must be positive"
            return data
        
        fsm = SimpleFSM(
            config,
            custom_functions={"validate_data": validate_data}
        )
        
        # Test valid data
        result = fsm.process({"value": 10})
        assert result["success"] is True
        assert result["final_state"] == "valid_end"
        assert result["data"]["is_valid"] is True
        
        # Test invalid data
        result = fsm.process({"value": -5})
        assert result["success"] is True
        assert result["final_state"] == "invalid_end"
        assert result["data"]["is_valid"] is False
        assert "Invalid" in result["data"]["validation_message"]
    
    def test_multi_field_validation(self):
        """Test validating multiple fields with result aggregation."""
        config = {
            "name": "multi_validator",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "validate_all",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "validate_fields"
                            }
                        }
                    },
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "validate_all"},
                    {"from": "validate_all", "to": "end"}
                ]
            }]
        }
        
        def validate_fields(state):
            data = state.data.copy()
            
            # Initialize validation results
            data["validations"] = {}
            
            # Validate email
            email = data.get("email", "")
            data["validations"]["email"] = {
                "valid": "@" in email and "." in email,
                "error": None if "@" in email and "." in email else "Invalid email format"
            }
            
            # Validate age
            age = data.get("age", 0)
            data["validations"]["age"] = {
                "valid": 18 <= age <= 100,
                "error": None if 18 <= age <= 100 else "Age must be between 18 and 100"
            }
            
            # Calculate overall validity
            data["all_valid"] = all(
                v["valid"] for v in data["validations"].values()
            )
            
            return data
        
        fsm = SimpleFSM(
            config,
            custom_functions={"validate_fields": validate_fields}
        )
        
        # Test with all valid data
        result = fsm.process({"email": "user@example.com", "age": 25})
        assert result["success"] is True
        assert result["data"]["all_valid"] is True
        assert result["data"]["validations"]["email"]["valid"] is True
        assert result["data"]["validations"]["age"]["valid"] is True
        
        # Test with invalid email
        result = fsm.process({"email": "invalid", "age": 25})
        assert result["success"] is True
        assert result["data"]["all_valid"] is False
        assert result["data"]["validations"]["email"]["valid"] is False
        assert result["data"]["validations"]["email"]["error"] == "Invalid email format"
        
        # Test with invalid age
        result = fsm.process({"email": "user@example.com", "age": 15})
        assert result["success"] is True
        assert result["data"]["all_valid"] is False
        assert result["data"]["validations"]["age"]["valid"] is False
        assert result["data"]["validations"]["age"]["error"] == "Age must be between 18 and 100"


class TestCustomFunctionErrors:
    """Test error handling for custom functions."""
    
    def test_missing_registered_function(self):
        """Test that missing registered function raises appropriate error."""
        config = {
            "name": "test",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "transform",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "missing_func"
                            }
                        }
                    },
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "transform"},
                    {"from": "transform", "to": "end"}
                ]
            }]
        }
        
        # Should raise error when building FSM without the required function
        with pytest.raises(ValueError, match="Registered function not found: missing_func"):
            fsm = SimpleFSM(config)
    
    def test_state_transform_exception_handling(self):
        """Test handling of exceptions in state transform functions.
        
        According to the design, state transform failures mark the state as failed
        but don't stop FSM execution - the FSM continues processing.
        """
        config = {
            "name": "test",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {
                        "name": "transform_state",
                        "functions": {
                            "transform": {
                                "type": "registered",
                                "name": "failing_state_transform"
                            }
                        }
                    },
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {"from": "start", "to": "transform_state"},
                    {"from": "transform_state", "to": "end"}
                ]
            }]
        }
        
        def failing_state_transform(state):
            raise ValueError("Intentional state transform error")
        
        fsm = SimpleFSM(
            config,
            custom_functions={"failing_state_transform": failing_state_transform}
        )
        
        result = fsm.process({})
        
        # State transforms failing should not stop execution
        # The FSM should continue and reach the end state
        assert result["success"] is True
        assert result["final_state"] == "end"
        # The transform_state should be in the path even though its transform failed
        assert "transform_state" in result["path"]
    
    def test_arc_transform_exception_handling(self):
        """Test handling of exceptions in arc transform functions.
        
        According to the design, arc transform failures prevent the transition
        from occurring - the FSM stays in the source state.
        """
        config = {
            "name": "test",
            "main_network": "main",
            "networks": [{
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "middle"},
                    {"name": "end", "is_end": True}
                ],
                "arcs": [
                    {
                        "from": "start",
                        "to": "middle",
                        "transform": {
                            "type": "registered",
                            "name": "failing_arc_transform"
                        }
                    },
                    {"from": "middle", "to": "end"},
                    {"from": "start", "to": "end"}  # Alternative path
                ]
            }]
        }
        
        def failing_arc_transform(data, context):
            raise ValueError("Intentional arc transform error")
        
        fsm = SimpleFSM(
            config,
            custom_functions={"failing_arc_transform": failing_arc_transform}
        )
        
        result = fsm.process({})
        
        # Arc transform failure should prevent transition to middle
        # FSM should try alternative path or stop at start
        assert "middle" not in result["path"]
        # Should either reach end via alternative path or stop at start
        assert result["final_state"] in ["start", "end"]