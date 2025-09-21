"""Tests for ContextFactory module."""

import pytest
from dataknobs_data import Record

from dataknobs_fsm.core.context_factory import ContextFactory
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.state import StateDefinition, StateType, StateInstance
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.resources.manager import ResourceManager


class TestContextFactory:
    """Test suite for ContextFactory."""
    
    @pytest.fixture
    def simple_fsm(self):
        """Create a simple FSM for testing."""
        fsm = FSM(
            name="test_fsm",
            data_mode=ProcessingMode.SINGLE,
            transaction_mode=TransactionMode.NONE
        )
        
        # Create network with states
        network = StateNetwork("main")
        
        # Add start state
        start_state = StateDefinition(
            name="start",
            type=StateType.START
        )
        network.add_state(start_state)
        
        # Add processing state
        process_state = StateDefinition(
            name="process",
            type=StateType.NORMAL
        )
        network.add_state(process_state)
        
        # Add end state
        end_state = StateDefinition(
            name="end",
            type=StateType.END
        )
        network.add_state(end_state)
        
        fsm.add_network(network, is_main=True)
        
        return fsm
    
    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager for testing."""
        return ResourceManager()
    
    def test_create_context_with_dict_data(self, simple_fsm, resource_manager):
        """Test creating context with dictionary data."""
        data = {"key": "value", "number": 42}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=data,
            resource_manager=resource_manager
        )
        
        assert context.data == data
        assert context.current_state == "start"
        assert context.data_mode == ProcessingMode.SINGLE
        assert context.current_state_instance is not None
        assert context.current_state_instance.definition.name == "start"
    
    def test_create_context_with_record_data(self, simple_fsm, resource_manager):
        """Test creating context with Record data."""
        data = {"key": "value", "number": 42}
        record = Record(data)
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=record,
            resource_manager=resource_manager
        )
        
        assert context.data == data
        assert context.current_state == "start"
    
    def test_create_context_with_initial_state(self, simple_fsm, resource_manager):
        """Test creating context with specified initial state."""
        data = {"test": "data"}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=data,
            initial_state="process",
            resource_manager=resource_manager
        )
        
        assert context.current_state == "process"
        assert context.current_state_instance.definition.name == "process"
    
    def test_create_context_with_unknown_initial_state(self, simple_fsm, resource_manager):
        """Test creating context with unknown initial state."""
        data = {"test": "data"}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=data,
            initial_state="unknown_state",
            resource_manager=resource_manager
        )
        
        # Should create a minimal state definition
        assert context.current_state == "unknown_state"
        assert context.current_state_instance is not None
        assert context.current_state_instance.definition.type == StateType.NORMAL
    
    def test_create_context_with_transaction_mode(self, simple_fsm, resource_manager):
        """Test creating context with transaction mode."""
        data = {"test": "data"}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=data,
            data_mode=ProcessingMode.SINGLE,
            transaction_mode=TransactionMode.PER_RECORD,
            resource_manager=resource_manager
        )
        
        assert context.transaction_mode == TransactionMode.PER_RECORD
    
    def test_create_batch_context(self, simple_fsm):
        """Test creating a batch processing context."""
        batch_data = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"}
        ]
        
        context = ContextFactory.create_batch_context(
            fsm=simple_fsm,
            batch_data=batch_data,
            data_mode=ProcessingMode.BATCH
        )
        
        assert context.data_mode == ProcessingMode.BATCH
        assert context.batch_data == batch_data
        assert context.current_state == "start"
        assert context.current_state_instance is not None
    
    def test_create_batch_context_with_resources(self, simple_fsm):
        """Test creating batch context with resources."""
        batch_data = [{"id": i} for i in range(5)]
        resources = {"db": {"connection": "test"}}
        
        context = ContextFactory.create_batch_context(
            fsm=simple_fsm,
            batch_data=batch_data,
            resources=resources
        )
        
        assert context.resource_limits == resources
        assert len(context.batch_data) == 5
    
    def test_create_stream_context(self, simple_fsm):
        """Test creating a stream processing context."""
        from dataknobs_fsm.streaming.core import StreamContext, StreamConfig
        
        config = StreamConfig(
            chunk_size=100,
            buffer_size=1000
        )
        stream_ctx = StreamContext(config=config)
        
        context = ContextFactory.create_stream_context(
            fsm=simple_fsm,
            stream_context=stream_ctx,
            resources={"stream": {"buffer_size": 1024}}
        )
        
        assert context.data_mode == ProcessingMode.STREAM
        assert context.stream_context == stream_ctx
        assert context.current_state == "start"
        assert context.resource_limits == {"stream": {"buffer_size": 1024}}
    
    def test_resolve_initial_state_with_no_states(self, resource_manager):
        """Test resolving initial state when FSM has no states."""
        # Create FSM with no networks/states
        empty_fsm = FSM(
            name="empty_fsm",
            data_mode=ProcessingMode.SINGLE
        )
        
        context = ContextFactory.create_context(
            fsm=empty_fsm,
            data={"test": "data"},
            resource_manager=resource_manager
        )
        
        # Should default to 'start' with minimal definition
        assert context.current_state == "start"
        assert context.current_state_instance.definition.type == StateType.START
    
    def test_resolve_initial_state_with_multiple_networks(self, resource_manager):
        """Test resolving initial state with multiple networks."""
        fsm = FSM(name="multi_network_fsm")
        
        # Create first network without start state
        network1 = StateNetwork("network1")
        network1.add_state(StateDefinition(name="state1", type=StateType.NORMAL))
        fsm.add_network(network1)
        
        # Create second network with start state
        network2 = StateNetwork("network2")
        start_state = StateDefinition(name="begin", type=StateType.START)
        network2.add_state(start_state)
        fsm.add_network(network2)
        
        context = ContextFactory.create_context(
            fsm=fsm,
            data={"test": "data"},
            resource_manager=resource_manager
        )
        
        # Should find the start state in network2
        assert context.current_state == "begin"
    
    def test_create_context_with_all_parameters(self, simple_fsm, resource_manager):
        """Test creating context with all parameters specified."""
        data = {"complex": "data", "nested": {"key": "value"}}
        resources = {"cache": {"size": 100}, "db": {"pool": 10}}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=data,
            initial_state="process",
            data_mode=ProcessingMode.SINGLE,
            transaction_mode=TransactionMode.PER_SESSION,
            resources=resources,
            resource_manager=resource_manager
        )
        
        assert context.data == data
        assert context.current_state == "process"
        assert context.data_mode == ProcessingMode.SINGLE
        assert context.transaction_mode == TransactionMode.PER_SESSION
        assert context.resource_limits == resources
        assert context.resource_manager == resource_manager
    
    def test_state_instance_data_copy(self, simple_fsm, resource_manager):
        """Test that state instance gets a copy of the data."""
        original_data = {"mutable": ["list"], "dict": {"key": "value"}}
        
        context = ContextFactory.create_context(
            fsm=simple_fsm,
            data=original_data,
            resource_manager=resource_manager
        )
        
        # Verify state instance has a copy
        assert context.current_state_instance.data == original_data
        assert context.current_state_instance.data is not original_data
        
        # Modify context data
        context.data["new_key"] = "new_value"
        
        # State instance data should not be affected
        assert "new_key" not in context.current_state_instance.data