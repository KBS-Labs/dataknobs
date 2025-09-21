"""Tests for network and arc components."""

from typing import Any, Dict

import pytest

from dataknobs_fsm.core import (
    Arc,
    ArcDefinition,
    ArcExecution,
    DataIsolationMode,
    NetworkResourceRequirements,
    PushArc,
    State,
    StateMode,
    StateNetwork,
)
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.functions.base import FunctionContext


class TestStateNetwork:
    """Test StateNetwork functionality."""
    
    def test_network_creation(self):
        """Test creating a state network."""
        network = StateNetwork("test_network", "Test network description")
        
        assert network.name == "test_network"
        assert network.description == "Test network description"
        assert len(network._states) == 0
        assert len(network._arcs) == 0
        assert network._initial_state is None
        assert len(network._final_states) == 0
    
    def test_add_state(self):
        """Test adding states to network."""
        network = StateNetwork("test")
        
        # Add normal state
        state1 = State(name="state1")
        network.add_state(state1)
        
        assert "state1" in network._states
        assert network.get_state("state1") == state1
        
        # Add initial state
        state2 = State(name="initial")
        network.add_state(state2, initial=True)
        
        assert network._initial_state == "initial"
        
        # Add final state
        state3 = State(name="final")
        network.add_state(state3, final=True)
        
        assert "final" in network._final_states
        
        # Try adding duplicate
        with pytest.raises(ValueError, match="already exists"):
            network.add_state(State(name="state1"))
        
        # Try adding second initial state
        with pytest.raises(ValueError, match="Initial state already set"):
            network.add_state(State(name="another"), initial=True)
    
    def test_remove_state(self):
        """Test removing states from network."""
        network = StateNetwork("test")
        
        # Add states
        network.add_state(State(name="state1"))
        network.add_state(State(name="state2"), initial=True)
        network.add_state(State(name="state3"), final=True)
        
        # Add arc
        network.add_arc("state1", "state2")
        
        # Remove state
        network.remove_state("state1")
        
        assert "state1" not in network._states
        assert len(network._arcs) == 0  # Arc should be removed
        
        # Remove initial state
        network.remove_state("state2")
        assert network._initial_state is None
        
        # Remove final state
        network.remove_state("state3")
        assert "state3" not in network._final_states
        
        # Try removing non-existent state
        with pytest.raises(KeyError):
            network.remove_state("nonexistent")
    
    def test_add_arc(self):
        """Test adding arcs between states."""
        network = StateNetwork("test")
        
        # Add states
        network.add_state(State(name="s1"))
        network.add_state(State(name="s2"))
        network.add_state(State(name="s3"))
        
        # Add arc
        arc1 = network.add_arc("s1", "s2", pre_test="check_condition")
        
        assert arc1.source_state == "s1"
        assert arc1.target_state == "s2"
        assert arc1.pre_test == "check_condition"
        assert arc1 in network._arcs
        assert arc1 in network.get_arcs_from_state("s1")
        
        # Add arc with transform
        arc2 = network.add_arc(
            "s2", "s3",
            transform="process_data",
            metadata={"priority": 1}
        )
        
        assert arc2.transform == "process_data"
        assert arc2.metadata["priority"] == 1
        
        # Try adding arc with non-existent states
        with pytest.raises(ValueError, match="Source state .* not found"):
            network.add_arc("nonexistent", "s2")
        
        with pytest.raises(ValueError, match="Target state .* not found"):
            network.add_arc("s1", "nonexistent")
    
    def test_remove_arc(self):
        """Test removing arcs from network."""
        network = StateNetwork("test")
        
        # Setup states and arcs
        network.add_state(State(name="s1"))
        network.add_state(State(name="s2"))
        arc = network.add_arc("s1", "s2")
        
        # Remove arc
        network.remove_arc(arc)
        
        assert arc not in network._arcs
        assert len(network.get_arcs_from_state("s1")) == 0
        
        # Try removing non-existent arc
        fake_arc = Arc("s1", "s2")
        with pytest.raises(ValueError, match="Arc not found"):
            network.remove_arc(fake_arc)
    
    def test_get_arcs(self):
        """Test getting arcs from/to states."""
        network = StateNetwork("test")
        
        # Setup network
        network.add_state(State(name="s1"))
        network.add_state(State(name="s2"))
        network.add_state(State(name="s3"))
        
        arc1 = network.add_arc("s1", "s2")
        arc2 = network.add_arc("s1", "s3")
        arc3 = network.add_arc("s2", "s3")
        
        # Get arcs from state
        arcs_from_s1 = network.get_arcs_from_state("s1")
        assert len(arcs_from_s1) == 2
        assert arc1 in arcs_from_s1
        assert arc2 in arcs_from_s1
        
        # Get arcs to state
        arcs_to_s3 = network.get_arcs_to_state("s3")
        assert len(arcs_to_s3) == 2
        assert arc2 in arcs_to_s3
        assert arc3 in arcs_to_s3
        
        # Non-existent state
        assert network.get_arcs_from_state("nonexistent") == []
        assert network.get_arcs_to_state("nonexistent") == []
    
    def test_network_validation(self):
        """Test network validation."""
        network = StateNetwork("test")
        
        # Empty network
        is_valid, errors = network.validate()
        assert not is_valid
        assert "No initial state" in str(errors)
        assert "No final states" in str(errors)
        
        # Add initial and final states
        network.add_state(State(name="initial"), initial=True)
        network.add_state(State(name="final"), final=True)
        
        # Add connection
        network.add_arc("initial", "final")
        
        is_valid, errors = network.validate()
        assert is_valid
        assert len(errors) == 0
        
        # Add unreachable state
        network.add_state(State(name="unreachable"))
        
        is_valid, errors = network.validate()
        assert not is_valid
        assert any("unreachable" in error for error in errors)
        
        # Add state with no outgoing arcs
        network.add_state(State(name="dead_end"))
        network.add_arc("initial", "dead_end")
        
        is_valid, errors = network.validate()
        assert not is_valid
        assert any("no outgoing arcs" in error for error in errors)
    
    def test_reachability_analysis(self):
        """Test finding reachable states."""
        network = StateNetwork("test")
        
        # Create branching network
        network.add_state(State(name="start"), initial=True)
        network.add_state(State(name="a"))
        network.add_state(State(name="b"))
        network.add_state(State(name="c"))
        network.add_state(State(name="end"), final=True)
        network.add_state(State(name="isolated"))
        
        network.add_arc("start", "a")
        network.add_arc("start", "b")
        network.add_arc("a", "c")
        network.add_arc("b", "c")
        network.add_arc("c", "end")
        
        reachable = network._find_reachable_states("start")
        
        assert "start" in reachable
        assert "a" in reachable
        assert "b" in reachable
        assert "c" in reachable
        assert "end" in reachable
        assert "isolated" not in reachable
    
    def test_cycle_detection(self):
        """Test detecting cycles in network."""
        network = StateNetwork("test")
        
        # Create network with cycle
        network.add_state(State(name="s1"))
        network.add_state(State(name="s2"))
        network.add_state(State(name="s3"))
        network.add_state(State(name="s4"), final=True)
        
        network.add_arc("s1", "s2")
        network.add_arc("s2", "s3")
        network.add_arc("s3", "s1")  # Creates cycle
        network.add_arc("s3", "s4")
        
        cycles = network._find_cycles()
        
        assert len(cycles) > 0
        # Should find the s1 -> s2 -> s3 -> s1 cycle
        cycle = cycles[0]
        assert "s1" in cycle
        assert "s2" in cycle
        assert "s3" in cycle
    
    def test_resource_requirements(self):
        """Test resource requirement tracking."""
        network = StateNetwork("test")
        
        # Add states with resource requirements
        # (In real usage, states would have actual resource requirements)
        network.add_state(State(name="s1"))
        network.add_state(State(name="s2"))
        
        # Get requirements (should be empty for basic states)
        reqs = network.get_resource_requirements()
        assert reqs.is_empty()
        
        # Test merging requirements
        reqs1 = NetworkResourceRequirements()
        reqs1.databases.add("db1")
        reqs1.filesystems.add("fs1")
        
        reqs2 = NetworkResourceRequirements()
        reqs2.databases.add("db2")
        reqs2.llms.add("gpt4")
        reqs2.streaming_enabled = True
        
        reqs1.merge(reqs2)
        
        assert "db1" in reqs1.databases
        assert "db2" in reqs1.databases
        assert "fs1" in reqs1.filesystems
        assert "gpt4" in reqs1.llms
        assert reqs1.streaming_enabled is True
    
    def test_network_serialization(self):
        """Test converting network to/from dictionary."""
        network = StateNetwork("test", "Test network")
        
        # Build network
        network.add_state(State(name="start"), initial=True)
        network.add_state(State(name="middle"))
        network.add_state(State(name="end"), final=True)
        
        network.add_arc("start", "middle", pre_test="check")
        network.add_arc("middle", "end", transform="process")
        
        # Convert to dict
        data = network.to_dict()
        
        assert data["name"] == "test"
        assert data["description"] == "Test network"
        assert data["initial_state"] == "start"
        assert "end" in data["final_states"]
        assert len(data["states"]) == 3
        assert len(data["arcs"]) == 2
        
        # Convert back from dict
        network2 = StateNetwork.from_dict(data)
        
        assert network2.name == network.name
        assert network2.description == network.description
        assert network2._initial_state == network._initial_state
        assert network2._final_states == network._final_states
        assert len(network2._states) == len(network._states)
        assert len(network2._arcs) == len(network._arcs)


class TestArcDefinition:
    """Test ArcDefinition and related classes."""
    
    def test_arc_definition_creation(self):
        """Test creating arc definitions."""
        arc_def = ArcDefinition(
            target_state="next_state",
            pre_test="check_condition",
            transform="process_data",
            priority=5,
            metadata={"key": "value"},
            required_resources={"database": "main_db"}
        )
        
        assert arc_def.target_state == "next_state"
        assert arc_def.pre_test == "check_condition"
        assert arc_def.transform == "process_data"
        assert arc_def.priority == 5
        assert arc_def.metadata["key"] == "value"
        assert arc_def.required_resources["database"] == "main_db"
    
    def test_push_arc_creation(self):
        """Test creating push arc definitions."""
        push_arc = PushArc(
            target_state="sub_start",
            target_network="sub_network",
            return_state="continue",
            isolation_mode=DataIsolationMode.COPY,
            pass_context=True,
            data_mapping={"field1": "sub_field1"},
            result_mapping={"sub_result": "field2"}
        )
        
        assert push_arc.target_state == "sub_start"
        assert push_arc.target_network == "sub_network"
        assert push_arc.return_state == "continue"
        assert push_arc.isolation_mode == DataIsolationMode.COPY
        assert push_arc.pass_context is True
        assert push_arc.data_mapping["field1"] == "sub_field1"
        assert push_arc.result_mapping["sub_result"] == "field2"
    
    def test_arc_definition_hash(self):
        """Test arc definition hashing."""
        arc1 = ArcDefinition("state1", "test", "transform", 1)
        arc2 = ArcDefinition("state1", "test", "transform", 1)
        arc3 = ArcDefinition("state2", "test", "transform", 1)
        
        assert hash(arc1) == hash(arc2)
        assert hash(arc1) != hash(arc3)
        
        # Can be used in sets
        arc_set = {arc1, arc2, arc3}
        assert len(arc_set) == 2  # arc1 and arc2 are same


class TestArcExecution:
    """Test ArcExecution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test functions
        self.test_data = {"value": 10}
        
        def always_true(data, context):
            return True
        
        def always_false(data, context):
            return False
        
        def double_value(data, context):
            if isinstance(data, dict):
                data["value"] = data.get("value", 0) * 2
            return data
        
        def fail_function(data, context):
            raise ValueError("Test error")
        
        self.function_registry = {
            "always_true": always_true,
            "always_false": always_false,
            "double_value": double_value,
            "fail_function": fail_function
        }
    
    def test_can_execute_no_pretest(self):
        """Test can_execute with no pre-test."""
        arc_def = ArcDefinition(target_state="next")
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        # Mock execution context
        class MockContext:
            pass
        
        # Should always return True without pre-test
        assert arc_exec.can_execute(MockContext(), self.test_data) is True
    
    def test_can_execute_with_pretest(self):
        """Test can_execute with pre-test function."""
        # Arc with pre-test that returns True
        arc_def1 = ArcDefinition(
            target_state="next",
            pre_test="always_true"
        )
        arc_exec1 = ArcExecution(arc_def1, "current", self.function_registry)
        
        class MockContext:
            pass
        
        assert arc_exec1.can_execute(MockContext(), self.test_data) is True
        
        # Arc with pre-test that returns False
        arc_def2 = ArcDefinition(
            target_state="next",
            pre_test="always_false"
        )
        arc_exec2 = ArcExecution(arc_def2, "current", self.function_registry)
        
        assert arc_exec2.can_execute(MockContext(), self.test_data) is False
    
    def test_can_execute_missing_function(self):
        """Test can_execute with missing pre-test function."""
        arc_def = ArcDefinition(
            target_state="next",
            pre_test="nonexistent"
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        class MockContext:
            pass
        
        with pytest.raises(FunctionError, match="not found"):
            arc_exec.can_execute(MockContext(), self.test_data)
    
    def test_execute_no_transform(self):
        """Test execute with no transform function."""
        arc_def = ArcDefinition(target_state="next")
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        class MockContext:
            pass
        
        # Should pass data through unchanged
        result = arc_exec.execute(MockContext(), self.test_data)
        assert result == self.test_data
        
        # Check statistics
        assert arc_exec.execution_count == 1
        assert arc_exec.success_count == 1
        assert arc_exec.failure_count == 0
    
    def test_execute_with_transform(self):
        """Test execute with transform function."""
        arc_def = ArcDefinition(
            target_state="next",
            transform="double_value"
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        class MockContext:
            pass
        
        data = {"value": 10}
        result = arc_exec.execute(MockContext(), data)
        
        assert result["value"] == 20
        assert arc_exec.execution_count == 1
        assert arc_exec.success_count == 1
    
    def test_execute_with_failure(self):
        """Test execute with failing transform."""
        arc_def = ArcDefinition(
            target_state="next",
            transform="fail_function"
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        class MockContext:
            pass
        
        with pytest.raises(FunctionError, match="Arc execution failed"):
            arc_exec.execute(MockContext(), self.test_data)
        
        assert arc_exec.execution_count == 1
        assert arc_exec.success_count == 0
        assert arc_exec.failure_count == 1
    
    def test_execute_push_arc(self):
        """Test executing push arc."""
        push_arc = PushArc(
            target_state="sub_start",
            target_network="sub_network",
            return_state="continue",
            isolation_mode=DataIsolationMode.COPY,
            data_mapping={"value": "input_value"},
            result_mapping={"output_value": "result"}
        )
        
        arc_exec = ArcExecution(push_arc, "current", self.function_registry)
        
        class MockContext:
            def push_network(self, network, return_state):
                self.pushed_network = network
                self.return_state = return_state
        
        context = MockContext()
        data = {"value": 100}
        
        # Execute push (simplified version)
        result = arc_exec.execute_push(push_arc, context, data)
        
        assert context.pushed_network == "sub_network"
        assert context.return_state == "continue"
        
        # In real implementation, this would handle sub-network execution
        assert result is not None
    
    def test_statistics_tracking(self):
        """Test execution statistics tracking."""
        arc_def = ArcDefinition(
            target_state="next",
            transform="double_value"
        )
        arc_exec = ArcExecution(arc_def, "current", self.function_registry)
        
        class MockContext:
            pass
        
        # Execute multiple times
        for i in range(5):
            arc_exec.execute(MockContext(), {"value": i})
        
        stats = arc_exec.get_statistics()
        
        assert stats["source_state"] == "current"
        assert stats["target_state"] == "next"
        assert stats["execution_count"] == 5
        assert stats["success_count"] == 5
        assert stats["failure_count"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["total_execution_time"] > 0
        assert stats["average_execution_time"] > 0